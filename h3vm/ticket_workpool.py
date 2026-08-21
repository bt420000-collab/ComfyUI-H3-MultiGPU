from __future__ import annotations

import logging
from collections import defaultdict

from .persistent_workpool import PersistentWorkpoolMLPFabric
from .critical_path import CriticalPathMLPFabric

LOG = logging.getLogger("H3VM")


class LocalTicketWorkpoolMLPFabric(PersistentWorkpoolMLPFabric):
    """Dev16.1: one transport packet, many GPU1-local compute tickets.

    The inter-GPU contract stays exact and unchanged: stage the helper token slice
    once, run all MLP work locally on GPU1, and return the final hidden slice once.
    Only the *local* helper compute grain changes. This directly probes the observed
    30ms/52ms helper-kernel cliff without adding PCIe traffic.

    Four Turbo steps form a one-shot matrix:
      1 = MONO   (one helper call)
      2 = DUAL   (two balanced 128-row-aligned tickets)
      3 = TRIPLE (three balanced tickets)
      4 = AUTO   (per block, replay the fastest safe ticket count from steps 1..3)
    """

    ticket_workpool_mode = True

    def __init__(self, *args, fixed_primary_fraction=0.68, **kwargs):
        kwargs["min_primary_fraction"] = float(fixed_primary_fraction)
        kwargs["max_primary_fraction"] = float(fixed_primary_fraction)
        kwargs["one_shot_matrix"] = False
        super().__init__(*args, **kwargs)
        self.fixed_primary_fraction = float(fixed_primary_fraction)
        self._ticket_phase = "MONO"
        self._ticket_plan = {}
        self._ticket_history = defaultdict(list)
        self._ticket_step_counts = defaultdict(int)
        self._ticket_step_rows = defaultdict(int)
        self._ticket_step_helper_ms = defaultdict(float)

    @staticmethod
    def _balanced_bounds(rows: int, count: int, align: int = 128):
        rows = int(rows); count = max(1, min(int(count), rows))
        if count <= 1 or rows < align * 2:
            return [(0, rows)]
        cuts = [0]
        for i in range(1, count):
            raw = rows * i / count
            cut = int(round(raw / align)) * align
            lo = cuts[-1] + align
            hi = rows - align * (count - i)
            cut = max(lo, min(hi, cut))
            cuts.append(cut)
        cuts.append(rows)
        return [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1) if cuts[i+1] > cuts[i]]

    def _best_ticket_count(self, index: int):
        hist = self._ticket_history.get(int(index), [])
        safe = [r for r in hist if r["stall_ms"] <= self.primary_stall_budget_ms]
        if not safe:
            safe = hist
        if not safe:
            return 1
        # Shadow path includes the exact same stage/return traffic, so minimizing
        # shadow_ms picks the locally fastest compute grain without hiding PCIe tax.
        best = min(safe, key=lambda r: (r["shadow_ms"], r["ticket_count"]))
        return int(best["ticket_count"])

    def begin_step(self, step: int):
        # Parent also fixes the cross-prompt summary pollution before it builds its
        # plan. Then override fractions: Dev16.1 changes only GPU1 local grain.
        super().begin_step(step)
        self._ticket_step_counts = defaultdict(int)
        self._ticket_step_rows = defaultdict(int)
        self._ticket_step_helper_ms = defaultdict(float)
        self._global_credit = 0
        for idx in self.active_blocks:
            self.block_primary_fraction[int(idx)] = self.fixed_primary_fraction

        step = int(step)
        if step <= 1:
            self._ticket_phase = "MONO"; default = 1
        elif step == 2:
            self._ticket_phase = "DUAL"; default = 2
        elif step == 3:
            self._ticket_phase = "TRIPLE"; default = 3
        else:
            self._ticket_phase = "AUTO"; default = None
        self._phase = self._ticket_phase
        self._ticket_plan = {
            int(i): (self._best_ticket_count(i) if default is None else default)
            for i in self.active_blocks
        }
        counts = defaultdict(int)
        for n in self._ticket_plan.values():
            counts[int(n)] += 1
        LOG.info(
            "H3VM DEV16.1 TICKET PLAN step=%d phase=%s | root=%.2f helper=%.1f%% | tickets=%s | transport=ONE-IN/ONE-OUT",
            step, self._ticket_phase, self.fixed_primary_fraction,
            (1.0-self.fixed_primary_fraction)*100.0, dict(sorted(counts.items())),
        )

    def _split(self, index: int, seq: int):
        frac = self.fixed_primary_fraction
        self._dispatch_fraction[int(index)] = frac
        return self._rounded_cut(seq, frac), frac

    def _dispatch_helper_packet(self, index, helper_mlp, remote_in):
        import torch
        count = int(self._ticket_plan.get(int(index), 1))
        bounds = self._balanced_bounds(int(remote_in.shape[0]), count)
        # MONO is the exact Dev16.0 helper path.
        if len(bounds) <= 1:
            return helper_mlp(remote_in), {
                "ticket_count": 1, "ticket_rows": [int(remote_in.shape[0])], "ticket_events": []
            }

        events = []
        rows_meta = []
        # Reuse the persistent helper input as the final output arena. Each ticket
        # overwrites only its own rows *after* helper_mlp has consumed that slice on
        # the same CUDA stream. No new GPU output slab and no extra PCIe transfer.
        for a, b in bounds:
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            y = helper_mlp(remote_in[a:b])
            remote_in[a:b].copy_(y)
            e1.record()
            events.append((e0, e1))
            rows_meta.append(int(b-a))
        return remote_in, {
            "ticket_count": len(bounds), "ticket_rows": rows_meta, "ticket_events": events
        }

    def on_block_complete(self, index: int):
        index = int(index)
        pending = self._pending.get(index)
        meta = dict((pending or {}).get("helper_meta") or {})
        super().on_block_complete(index)
        hist = self._history.get(index, [])
        if not hist:
            return
        last = hist[-1]
        count = int(meta.get("ticket_count", self._ticket_plan.get(index, 1)))
        rows = list(meta.get("ticket_rows") or [int(last.get("helper_rows", 0))])
        ticket_ms = []
        for pair in meta.get("ticket_events") or []:
            try:
                ticket_ms.append(float(pair[0].elapsed_time(pair[1])))
            except Exception:
                pass
        rec = dict(last)
        rec.update({"ticket_count": count, "ticket_rows": rows, "ticket_ms": ticket_ms})
        self._ticket_history[index].append(rec)
        self._ticket_step_counts[count] += 1
        self._ticket_step_rows[count] += sum(rows)
        self._ticket_step_helper_ms[count] += float(last.get("helper_ms", 0.0))

        if self.telemetry and (self.calls <= 4 or count > 1 or last.get("stall_ms", 0.0) > self.primary_stall_budget_ms):
            LOG.info(
                "H3VM DEV16.1 TICKET #%d | phase=%s block=%d count=%d rows=%s ticket_ms=%s | helper=%.1fms shadow=%.1fms root=%.1fms slack=%+.1fms",
                self.calls, self._ticket_phase, index, count, rows,
                [round(v,1) for v in ticket_ms], float(last["helper_ms"]),
                float(last["shadow_ms"]), float(last["owner_ms"]), float(last["slack_ms"]),
            )

    def step_summary(self):
        base = super().step_summary()
        base["phase"] = self._ticket_phase
        base["ticket_counts"] = dict(sorted(self._ticket_step_counts.items()))
        base["ticket_rows"] = dict(sorted(self._ticket_step_rows.items()))
        base["ticket_helper_ms"] = {int(k): round(float(v), 2) for k,v in sorted(self._ticket_step_helper_ms.items())}
        return base

    def record_step_runtime(self, **rec):
        super().record_step_runtime(**rec)
        if self._step_runtime:
            item = self._step_runtime[-1]
            item["phase"] = self._ticket_phase
            item["ticket_counts"] = dict(sorted(self._ticket_step_counts.items()))
            item["ticket_helper_ms"] = {int(k): round(float(v),2) for k,v in sorted(self._ticket_step_helper_ms.items())}

    def final_summary(self):
        # Deliberately do not call Dev16.0 final_summary: its fraction optimizer is
        # irrelevant here. One prompt in, one prompt out.
        total = self._ring_hits_total + self._ring_misses_total
        best = {int(i): self._best_ticket_count(i) for i in sorted(self.active_blocks)}
        valid = list(self._step_runtime)
        fastest = min(valid, key=lambda r: r.get("wall_ms", 1e30)) if valid else None
        return {
            "mode": "DEV16.1_LOCAL_TICKET_QUEUE_ONE_SHOT",
            "contract": "ONE_STAGE_IN__LOCAL_TICKETS__ONE_RETURN_OUT",
            "fixed_root_fraction": self.fixed_primary_fraction,
            "ring_slots": self.ring_slots,
            "ring_hit_rate": round(self._ring_hits_total/total, 4) if total else 0.0,
            "ring_init_ms_total": round(self._ring_init_ms_total, 2),
            "steps": self._step_runtime,
            "best_ticket_count_by_block": best,
            "fastest_step": fastest,
        }


class LoadShiftWorkpoolMLPFabric(PersistentWorkpoolMLPFabric):
    """Dev16.2: spend proven GPU1 slack by moving *more rows* off GPU0.

    Dev16.1 proved that DUAL local tickets are numerically stable through the
    second denoise step and reduce helper compute without touching PCIe count.
    Dev16.2 therefore freezes ticket_count=2 and changes only the row split.

    One-shot matrix:
      step1 = BASE32  : 68/32 reference
      step2 = SHIFT34 : 66/34 for every active helper block
      step3 = FEED    : per-block 60/40, 62/38, 64/36, 66/34 or 68/32,
                        selected from step2 slack. This is the CPU "watch the
                        face and feed more" experiment.
      step4 = LOCK    : replay the fastest *safe* fraction observed per block.

    Cross-GPU contract remains exactly one stage-in and one return-out.  DUAL
    tickets execute only inside GPU1.  Unlike Dev16.1 TRIPLE, output is written
    to a separate local slab so helper input is never overwritten in-place.
    """

    ticket_workpool_mode = True
    load_shift_mode = True

    def __init__(self, *args, **kwargs):
        kwargs['min_primary_fraction'] = 0.60
        kwargs['max_primary_fraction'] = 0.68
        kwargs['one_shot_matrix'] = False
        super().__init__(*args, **kwargs)
        self._ticket_phase = 'BASE32'
        self._ticket_step_counts = defaultdict(int)
        self._ticket_step_rows = defaultdict(int)
        self._ticket_step_helper_ms = defaultdict(float)
        self._chosen_fraction = {}

    def _best_safe_fraction_for_speed(self, index: int):
        hist = self._history.get(int(index), [])
        safe = [r for r in hist if r['stall_ms'] <= self.primary_stall_budget_ms and r['slack_ms'] >= 2.0]
        if not safe:
            safe = [r for r in hist if r['stall_ms'] <= self.primary_stall_budget_ms and r['slack_ms'] >= 0.0]
        if not safe:
            return 0.68
        # Per-block critical time is max(owner, shadow). Pick the observation that
        # actually shortened that merge window, not simply the deepest split.
        best = min(safe, key=lambda r: (max(float(r['owner_ms']), float(r['shadow_ms'])), float(r['fraction'])))
        return float(best['fraction'])

    def _step3_fraction(self, index: int):
        last = self._last(index)
        if last is None:
            return 0.66, 'no_profile'
        if last['stall_ms'] > self.primary_stall_budget_ms or last['slack_ms'] < 0.0:
            return 0.68, 'retreat68'
        slack = float(last['slack_ms'])
        # Spend the measured slack aggressively enough that one run explores the
        # useful range. The primary still owns at least 60% and a miss lasts only
        # for this probe step; LOCK will not replay an unsafe fraction.
        if slack >= 15.0:
            return 0.60, 'feed40'
        if slack >= 10.0:
            return 0.62, 'feed38'
        if slack >= 6.0:
            return 0.64, 'feed36'
        if slack >= 2.0:
            return 0.66, 'hold34'
        return 0.68, 'guard32'

    def begin_step(self, step: int):
        # Bypass Dev16.0 fraction policy but keep prompt-reset + base telemetry.
        if int(step) <= 1 and self._step_runtime:
            self._reset_sample_metrics()
        CriticalPathMLPFabric.begin_step(self, step)
        self._step_ring_hits = 0
        self._step_ring_misses = 0
        self._step_ring_init_ms = 0.0
        self._step_policy_counts = defaultdict(int)
        self._dispatch_fraction = {}
        self._global_credit = 0
        self._recent_slack.clear()
        self._ticket_step_counts = defaultdict(int)
        self._ticket_step_rows = defaultdict(int)
        self._ticket_step_helper_ms = defaultdict(float)

        step = int(step)
        if step <= 1:
            self._ticket_phase = self._phase = 'BASE32'
            plan = {int(i):(0.68,'base32') for i in self.active_blocks}
        elif step == 2:
            self._ticket_phase = self._phase = 'SHIFT34'
            plan = {int(i):(0.66,'shift34') for i in self.active_blocks}
        elif step == 3:
            self._ticket_phase = self._phase = 'FEED'
            plan = {int(i):self._step3_fraction(i) for i in self.active_blocks}
        else:
            self._ticket_phase = self._phase = 'LOCK'
            plan = {int(i):(self._best_safe_fraction_for_speed(i),'lock') for i in self.active_blocks}

        self._chosen_fraction = {}
        for i,(frac,action) in plan.items():
            frac = round(max(0.60, min(0.68, float(frac))), 2)
            self.block_primary_fraction[i] = frac
            self._chosen_fraction[i] = frac
            self._step_policy_counts[action] += 1
        helper_avg = 1.0 - sum(self._chosen_fraction.values())/max(1,len(self._chosen_fraction))
        LOG.info(
            'H3VM DEV16.2 LOAD-SHIFT PLAN step=%d phase=%s | helper_avg=%.1f%% | fractions=%s | policy=%s | local=DUAL SAFE-OUT',
            step, self._ticket_phase, helper_avg*100.0,
            {f:sum(1 for v in self._chosen_fraction.values() if abs(v-f)<1e-9) for f in sorted(set(self._chosen_fraction.values()))},
            dict(self._step_policy_counts),
        )

    def _split(self, index: int, seq: int):
        frac = float(self.block_primary_fraction.get(int(index), 0.68))
        self._dispatch_fraction[int(index)] = frac
        return self._rounded_cut(seq, frac), frac

    def _dispatch_helper_packet(self, index, helper_mlp, remote_in):
        import torch
        rows = int(remote_in.shape[0])
        bounds = LocalTicketWorkpoolMLPFabric._balanced_bounds(rows, 2)
        if len(bounds) <= 1:
            return helper_mlp(remote_in), {'ticket_count':1,'ticket_rows':[rows],'ticket_events':[]}
        # Separate output slab: avoids the Dev16.1 TRIPLE in-place corruption path.
        out = torch.empty_like(remote_in)
        events=[]; rows_meta=[]
        for a,b in bounds:
            e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
            e0.record()
            y=helper_mlp(remote_in[a:b])
            out[a:b].copy_(y)
            e1.record()
            events.append((e0,e1)); rows_meta.append(int(b-a))
            del y
        return out, {'ticket_count':len(bounds),'ticket_rows':rows_meta,'ticket_events':events}

    def on_block_complete(self, index: int):
        index=int(index)
        pending=self._pending.get(index)
        meta=dict((pending or {}).get('helper_meta') or {})
        super().on_block_complete(index)
        hist=self._history.get(index,[])
        if not hist:
            return
        last=hist[-1]
        count=int(meta.get('ticket_count',2))
        rows=list(meta.get('ticket_rows') or [int(last.get('helper_rows',0))])
        ticket_ms=[]
        for pair in meta.get('ticket_events') or []:
            try: ticket_ms.append(float(pair[0].elapsed_time(pair[1])))
            except Exception: pass
        self._ticket_step_counts[count]+=1
        self._ticket_step_rows[count]+=sum(rows)
        self._ticket_step_helper_ms[count]+=float(last.get('helper_ms',0.0))
        if self.telemetry and (self.calls <= 4 or last.get('stall_ms',0.0)>self.primary_stall_budget_ms or last.get('slack_ms',0.0)>=8.0):
            LOG.info(
                'H3VM DEV16.2 SHIFT #%d | phase=%s block=%d frac=%.2f rows(root/helper)=%d/%d tickets=%s ticket_ms=%s | root=%.1f shadow=%.1f slack=%+.1f',
                self.calls,self._ticket_phase,index,float(last['fraction']),
                int(last.get('helper_rows',0)+self._pending.get(index,{}).get('cut',0)) if False else int((pending or {}).get('cut',0)),
                int(last.get('helper_rows',0)),rows,[round(v,1) for v in ticket_ms],
                float(last['owner_ms']),float(last['shadow_ms']),float(last['slack_ms'])
            )

    def step_summary(self):
        base=super().step_summary()
        base['phase']=self._ticket_phase
        base['fraction_counts']={str(f):sum(1 for v in self._chosen_fraction.values() if abs(v-f)<1e-9) for f in sorted(set(self._chosen_fraction.values()))}
        base['ticket_counts']=dict(sorted(self._ticket_step_counts.items()))
        base['ticket_rows']=dict(sorted(self._ticket_step_rows.items()))
        base['ticket_helper_ms']={int(k):round(float(v),2) for k,v in sorted(self._ticket_step_helper_ms.items())}
        return base

    def record_step_runtime(self, **rec):
        super().record_step_runtime(**rec)
        if self._step_runtime:
            item=self._step_runtime[-1]
            item['phase']=self._ticket_phase
            item['fraction_counts']={str(f):sum(1 for v in self._chosen_fraction.values() if abs(v-f)<1e-9) for f in sorted(set(self._chosen_fraction.values()))}
            item['ticket_helper_ms']={int(k):round(float(v),2) for k,v in sorted(self._ticket_step_helper_ms.items())}

    def final_summary(self):
        total=self._ring_hits_total+self._ring_misses_total
        valid=list(self._step_runtime)
        fastest=min(valid,key=lambda r:r.get('wall_ms',1e30)) if valid else None
        return {
            'mode':'DEV16.2_LOAD_SHIFT_ONE_SHOT',
            'contract':'ONE_STAGE_IN__DUAL_LOCAL__ONE_RETURN_OUT',
            'ring_slots':self.ring_slots,
            'ring_hit_rate':round(self._ring_hits_total/total,4) if total else 0.0,
            'ring_init_ms_total':round(self._ring_init_ms_total,2),
            'steps':self._step_runtime,
            'locked_fraction_by_block':{int(i):round(float(self.block_primary_fraction.get(i,0.68)),2) for i in sorted(self.active_blocks)},
            'fastest_step':fastest,
        }
