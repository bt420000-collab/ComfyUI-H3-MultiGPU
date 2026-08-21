from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict, deque

from .critical_path import CriticalPathMLPFabric

LOG = logging.getLogger("H3VM")


class PersistentWorkpoolMLPFabric(CriticalPathMLPFabric):
    """Dev16.0 one-shot persistent workpool experiment.

    This keeps Dev15.2's exact token-row MLP split, but changes the scheduler:
    * a fixed relay ring is preallocated on first real sequence shape;
    * every denoise step pre-assigns a complete helper work plan;
    * recent slack becomes a bounded credit that can feed a little more work to GPU1;
    * the four Turbo steps act as PROFILE -> EXPAND -> HARVEST -> LOCK.

    It is intentionally conservative: no background thread touches DynamicVRAM,
    no whole-block ownership leaves GPU0, and a ring miss falls back to the old
    ephemeral relay path rather than waiting on GPU0.
    """

    workpool_mode = True

    def __init__(self, *args, ring_slots=2, deadline_margin_ms=4.0,
                 min_primary_fraction=0.64, max_primary_fraction=0.72,
                 one_shot_matrix=True, **kwargs):
        # We own the adaptive policy here.  The parent scheduler's binary
        # disable/return-to-root policy is bypassed in on_block_complete().
        kwargs["adaptive_slack"] = False
        kwargs["min_primary_fraction"] = float(min_primary_fraction)
        kwargs["max_primary_fraction"] = float(max_primary_fraction)
        super().__init__(*args, **kwargs)

        self.min_primary_fraction = max(0.60, min(float(min_primary_fraction), self.primary_fraction))
        self.max_primary_fraction = max(self.primary_fraction, min(0.78, float(max_primary_fraction)))
        self.ring_slots = max(1, min(3, int(ring_slots)))
        self.deadline_margin_ms = max(0.0, float(deadline_margin_ms))
        self.one_shot_matrix = bool(one_shot_matrix)

        self._candidate_fracs = []
        f = self.max_primary_fraction
        while f >= self.min_primary_fraction - 1e-9:
            self._candidate_fracs.append(round(f, 2))
            f -= 0.02
        self._candidate_fracs = sorted(set(self._candidate_fracs), reverse=True)

        self._history = defaultdict(list)
        self._global_credit = 0
        self._recent_slack = deque(maxlen=4)
        self._phase = "PROFILE"
        self._rings = {}
        self._ring_cursor = 0
        self._ring_init_ms_total = 0.0
        self._ring_hits_total = 0
        self._ring_misses_total = 0
        self._step_ring_hits = 0
        self._step_ring_misses = 0
        self._step_ring_init_ms = 0.0
        self._step_policy_counts = defaultdict(int)
        self._step_runtime = []
        self._sample_started = time.perf_counter()
        self._dispatch_fraction = {}

    @staticmethod
    def _rounded_cut(seq: int, frac: float) -> int:
        cut = int(round(seq * float(frac) / 128.0)) * 128
        return max(128, min(seq - 128, cut)) if seq >= 256 else max(1, seq // 2)

    def _last(self, index):
        hist = self._history.get(int(index), [])
        return hist[-1] if hist else None

    def _best_safe_fraction(self, index):
        hist = self._history.get(int(index), [])
        safe = [r for r in hist if r["stall_ms"] <= self.primary_stall_budget_ms and r["slack_ms"] >= 0.0]
        if not safe:
            return min(self.max_primary_fraction, max(self.primary_fraction, 0.70))
        # Lowest root share observed to finish by the merge deadline wins.
        return max(self.min_primary_fraction, min(float(r["fraction"]) for r in safe))

    def _plan_fraction(self, index: int, step: int):
        index = int(index)
        last = self._last(index)
        if step <= 1 or last is None:
            return self.primary_fraction, "profile"

        if step == 2:
            if last["stall_ms"] > self.primary_stall_budget_ms:
                return min(self.max_primary_fraction, float(last["fraction"]) + 0.02), "retreat"
            if last["slack_ms"] >= self.deadline_margin_ms + 8.0:
                return max(self.min_primary_fraction, float(last["fraction"]) - 0.02), "expand"
            return float(last["fraction"]), "hold"

        if step == 3:
            if last["stall_ms"] > self.primary_stall_budget_ms:
                return min(self.max_primary_fraction, float(last["fraction"]) + 0.02), "retreat"
            if last["slack_ms"] >= self.deadline_margin_ms + 4.0:
                return max(self.min_primary_fraction, float(last["fraction"]) - 0.02), "harvest"
            if last["slack_ms"] < self.deadline_margin_ms:
                return min(self.max_primary_fraction, float(last["fraction"]) + 0.02), "guard"
            return float(last["fraction"]), "hold"

        return self._best_safe_fraction(index), "lock"

    def _reset_sample_metrics(self):
        """Reset per-prompt telemetry without touching model residency.

        Dev16.0 originally accumulated step JSON across consecutive prompts.
        Reset here so every one-shot report describes exactly one Queue run.
        """
        self._history.clear()
        self._step_runtime.clear()
        self._global_credit = 0
        self._recent_slack.clear()
        self._ring_hits_total = 0
        self._ring_misses_total = 0
        self._ring_init_ms_total = 0.0
        self._sample_started = time.perf_counter()

    def begin_step(self, step: int):
        if int(step) <= 1 and self._step_runtime:
            self._reset_sample_metrics()
        super().begin_step(step)
        self._step_ring_hits = 0
        self._step_ring_misses = 0
        self._step_ring_init_ms = 0.0
        self._step_policy_counts = defaultdict(int)
        self._dispatch_fraction = {}
        self._global_credit = max(-2, min(2, self._global_credit))
        self._recent_slack.clear()

        if step <= 1:
            self._phase = "PROFILE"
        elif step == 2:
            self._phase = "EXPAND"
        elif step == 3:
            self._phase = "HARVEST"
        else:
            self._phase = "LOCK"

        for idx in sorted(self.active_blocks):
            frac, action = self._plan_fraction(idx, int(step))
            frac = round(max(self.min_primary_fraction, min(self.max_primary_fraction, frac)), 2)
            self.block_primary_fraction[int(idx)] = frac
            self._step_policy_counts[action] += 1

        helper_avg = 1.0 - (
            sum(self.block_primary_fraction.values()) / max(1, len(self.block_primary_fraction))
        )
        LOG.info(
            "H3VM DEV16 WORKPOOL PLAN step=%d phase=%s | preassigned=%d helper_avg=%.1f%% | "
            "policy=%s | credit=%+d",
            int(step), self._phase, len(self.active_blocks), helper_avg * 100.0,
            dict(self._step_policy_counts), int(self._global_credit),
        )

    def _split(self, index: int, seq: int):
        # The step plan is the hard reservation.  Credit only adds/removes one
        # 2-point helper ticket and only when this exact block already has a safe
        # observation. This is the "CPU watches the face and feeds the next job"
        # layer without allowing queue starvation to stall GPU0.
        index = int(index)
        planned = float(self.block_primary_fraction.get(index, self.primary_fraction))
        actual = planned
        last = self._last(index)
        if last is not None and last["stall_ms"] <= self.primary_stall_budget_ms:
            if self._global_credit >= 2 and last["slack_ms"] >= self.deadline_margin_ms + 6.0:
                actual = max(self.min_primary_fraction, planned - 0.02)
            elif self._global_credit <= -2:
                actual = min(self.max_primary_fraction, planned + 0.02)
        actual = round(actual, 2)
        self._dispatch_fraction[index] = actual
        return self._rounded_cut(seq, actual), actual

    def _ring_key(self, seq, hidden, dtype, owner, helper):
        return (int(seq), int(hidden), str(dtype), str(owner), str(helper))

    def _make_ring(self, seq, hidden, dtype, owner, helper):
        import torch
        t0 = time.perf_counter()
        max_helper_rows = int(seq) - self._rounded_cut(int(seq), self.min_primary_fraction)
        shape = (self.ring_slots, max_helper_rows, int(hidden))
        pinned = True
        try:
            stage_host = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            return_host = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        except Exception:
            pinned = False
            stage_host = torch.empty(shape, dtype=dtype, device="cpu")
            return_host = torch.empty(shape, dtype=dtype, device="cpu")

        # These two buffers are the persistent "exam desks".  Intermediate FFN
        # workspace is still owned by the existing quantized MLP kernels.
        with torch.cuda.device(helper):
            helper_in = torch.empty(shape, dtype=dtype, device=helper)
        with torch.cuda.device(owner):
            owner_return = torch.empty(shape, dtype=dtype, device=owner)

        ring = {
            "max_rows": max_helper_rows,
            "stage_host": stage_host,
            "return_host": return_host,
            "helper_in": helper_in,
            "owner_return": owner_return,
            "free_events": [None] * self.ring_slots,
            "pinned": pinned,
        }
        dt = (time.perf_counter() - t0) * 1000.0
        self._ring_init_ms_total += dt
        self._step_ring_init_ms += dt
        gib = (
            stage_host.numel() * stage_host.element_size()
            + return_host.numel() * return_host.element_size()
            + helper_in.numel() * helper_in.element_size()
            + owner_return.numel() * owner_return.element_size()
        ) / (1024 ** 3)
        LOG.info(
            "H3VM DEV16 PERSISTENT RING READY | slots=%d max_helper_rows=%d hidden=%d pinned=%s total_buffers=%.2fGiB init=%.1fms",
            self.ring_slots, max_helper_rows, int(hidden), pinned, gib, dt,
        )
        return ring

    def _acquire_ring_slot(self, ring):
        for j in range(self.ring_slots):
            i = (self._ring_cursor + j) % self.ring_slots
            ev = ring["free_events"][i]
            if ev is None:
                self._ring_cursor = (i + 1) % self.ring_slots
                return i
            try:
                if ev.query():
                    self._ring_cursor = (i + 1) % self.ring_slots
                    return i
            except Exception:
                pass
        return None

    def _stage_persistent(self, remote_x, owner, helper, caller_stream, ring, slot):
        import torch
        rows = int(remote_x.shape[0])
        host = ring["stage_host"][slot, :rows]
        remote = ring["helper_in"][slot, :rows]
        d2h = self.d2h_streams[owner]
        h2d = self.h2d_streams[helper]

        d0 = torch.cuda.Event(enable_timing=True); d1 = torch.cuda.Event(enable_timing=True)
        h0 = torch.cuda.Event(enable_timing=True); h1 = torch.cuda.Event(enable_timing=True)
        ready = torch.cuda.Event()
        d2h.wait_stream(caller_stream)
        with torch.cuda.device(owner), torch.cuda.stream(d2h):
            d0.record(d2h)
            host.copy_(remote_x, non_blocking=bool(ring["pinned"]))
            d1.record(d2h)
        # Host relay still has one mandatory ownership hand-off. This wait is on
        # the side lane only; GPU0's MLP kernel was already launched.
        d1.synchronize()
        with torch.cuda.device(helper), torch.cuda.stream(h2d):
            h0.record(h2d)
            remote.copy_(host, non_blocking=bool(ring["pinned"]))
            h1.record(h2d)
            ready.record(h2d)
        return remote, ready, (d0, d1, h0, h1)

    def _return_persistent(self, remote_out, owner, helper, helper_done, ring, slot):
        import torch
        rows = int(remote_out.shape[0])
        host = ring["return_host"][slot, :rows]
        returned = ring["owner_return"][slot, :rows]
        d2h = self.d2h_streams[helper]
        h2d = self.h2d_streams[owner]

        d0 = torch.cuda.Event(enable_timing=True); d1 = torch.cuda.Event(enable_timing=True)
        h0 = torch.cuda.Event(enable_timing=True); h1 = torch.cuda.Event(enable_timing=True)
        ready = torch.cuda.Event()
        d2h.wait_event(helper_done)
        with torch.cuda.device(helper), torch.cuda.stream(d2h):
            d0.record(d2h)
            host.copy_(remote_out, non_blocking=bool(ring["pinned"]))
            d1.record(d2h)
        d1.synchronize()
        with torch.cuda.device(owner), torch.cuda.stream(h2d):
            h0.record(h2d)
            returned.copy_(host, non_blocking=bool(ring["pinned"]))
            h1.record(h2d)
            ready.record(h2d)
        return returned, ready, (d0, d1, h0, h1)

    def _dispatch_helper_packet(self, index, helper_mlp, remote_in):
        """Run one helper packet. Subclasses may tile locally on GPU1.

        Cross-GPU transport is intentionally outside this hook: one stage in,
        one final return out. The default preserves Dev16.0 monolithic behavior.
        """
        return helper_mlp(remote_in), {"ticket_count": 1, "ticket_rows": [int(remote_in.shape[0])], "ticket_events": []}

    def execute(self, index: int, owner_mlp, x):
        import torch
        import comfy.model_management as mm

        index = int(index)
        seq = int(x.shape[0])
        owner = self.owner_device(index)
        helper = self.helper_device(index)
        if (
            index not in self.active_blocks
            or seq < self.min_sequence_length
            or not self._sidecar_ready
            or not self._helper_has_runway(helper)
        ):
            if index in self.active_blocks:
                self._step_skipped += 1
                if not self._sidecar_ready:
                    self._step_startup_skipped += 1
            return self._run_mlp(owner_mlp, x)
        if x.device != owner:
            raise RuntimeError(f"H3VM Dev16 MLP owner mismatch block={index}: x={x.device} owner={owner}")

        helper_mlp = self.helper_mlp_by_block[index]
        cut, primary_fraction = self._split(index, seq)
        x_primary = x[:cut]
        x_secondary = x[cut:]
        if owner == self.primary:
            local_x, remote_x = x_primary, x_secondary
            remote_first = False
        else:
            remote_x, local_x = x_primary, x_secondary
            remote_first = True

        caller_stream = torch.cuda.current_stream(owner)
        owner_stream = self.compute_streams[owner]
        helper_stream = self.compute_streams[helper]
        owner_stream.wait_stream(caller_stream)

        # Always launch the red line first.
        o0 = torch.cuda.Event(enable_timing=True); o1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(owner), torch.cuda.stream(owner_stream), mm.cuda_device_context(owner):
            o0.record(owner_stream)
            local_out = self._run_mlp(owner_mlp, local_x)
            o1.record(owner_stream)

        key = self._ring_key(seq, int(x.shape[-1]), x.dtype, owner, helper)
        ring = self._rings.get(key)
        if ring is None:
            try:
                ring = self._make_ring(seq, int(x.shape[-1]), x.dtype, owner, helper)
                self._rings[key] = ring
            except Exception as exc:
                LOG.warning("H3VM DEV16 ring init failed; using legacy relay: %r", exc)
                ring = None

        slot = self._acquire_ring_slot(ring) if ring is not None else None
        t_shadow = time.perf_counter()
        if ring is not None and slot is not None and int(x_secondary.shape[0]) <= int(ring["max_rows"]):
            self._ring_hits_total += 1; self._step_ring_hits += 1
            remote_in, remote_ready, stage_events = self._stage_persistent(
                remote_x.contiguous(), owner, helper, caller_stream, ring, slot
            )
        else:
            self._ring_misses_total += 1; self._step_ring_misses += 1
            slot = None
            remote_in, remote_ready, stage_events = self._stage_to_helper(
                remote_x.contiguous(), owner, helper, caller_stream
            )

        h0 = torch.cuda.Event(enable_timing=True); h1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(helper), torch.cuda.stream(helper_stream), mm.cuda_device_context(helper):
            helper_stream.wait_event(remote_ready)
            h0.record(helper_stream)
            remote_out, helper_meta = self._dispatch_helper_packet(index, helper_mlp, remote_in)
            h1.record(helper_stream)

        if slot is not None:
            returned, returned_ready, return_events = self._return_persistent(
                remote_out, owner, helper, h1, ring, slot
            )
        else:
            returned, returned_ready, return_events = self._return_to_owner(
                remote_out, owner, helper, h1
            )
        shadow_enqueue_ms = (time.perf_counter() - t_shadow) * 1000.0

        caller_stream.wait_event(o1)
        caller_stream.wait_event(returned_ready)
        if remote_first:
            out = torch.cat((returned, local_out), dim=0)
        else:
            out = torch.cat((local_out, returned), dim=0)
        local_out.record_stream(caller_stream)
        returned.record_stream(caller_stream)
        out.record_stream(caller_stream)
        if slot is not None:
            free_event = torch.cuda.Event()
            free_event.record(caller_stream)
            ring["free_events"][slot] = free_event

        self._pending[index] = {
            "owner": owner, "helper": helper, "cut": cut, "seq": seq,
            "owner_events": (o0, o1), "helper_events": (h0, h1),
            "stage_events": stage_events, "return_events": return_events,
            "shadow_enqueue_ms": shadow_enqueue_ms,
            "primary_fraction": primary_fraction,
            "ring_slot": slot,
            "helper_meta": helper_meta,
        }
        return out

    def on_block_complete(self, index: int):
        index = int(index)
        rec = self._pending.pop(index, None)
        if rec is None:
            return
        owner_ms = self._elapsed(rec["owner_events"])
        helper_ms = self._elapsed(rec["helper_events"])
        stage_d2h = self._elapsed(rec["stage_events"][:2])
        stage_h2d = self._elapsed(rec["stage_events"][2:])
        ret_d2h = self._elapsed(rec["return_events"][:2])
        ret_h2d = self._elapsed(rec["return_events"][2:])
        shadow_ms = stage_d2h + stage_h2d + helper_ms + ret_d2h + ret_h2d
        slack_ms = owner_ms - shadow_ms
        stall_est = max(0.0, -slack_ms)

        self.calls += 1
        self._step_calls += 1
        self._step_owner_ms += owner_ms
        self._step_shadow_ms += shadow_ms
        self._step_stall_est_ms += stall_est
        self._step_slack_ms.append(slack_ms)

        frac = float(rec.get("primary_fraction", self.primary_fraction))
        hist_rec = {
            "step": int(self._step), "fraction": frac,
            "owner_ms": owner_ms, "shadow_ms": shadow_ms,
            "helper_ms": helper_ms, "slack_ms": slack_ms,
            "stall_ms": stall_est, "helper_rows": int(rec["seq"] - rec["cut"]),
        }
        self._history[index].append(hist_rec)

        self._recent_slack.append(slack_ms)
        if stall_est > self.primary_stall_budget_ms:
            self._global_credit = max(-4, self._global_credit - 2)
            self._step_adjust_up += 1
            action = "credit_brake"
        elif slack_ms >= self.deadline_margin_ms + 6.0:
            self._global_credit = min(4, self._global_credit + 1)
            self._step_adjust_down += 1
            action = "credit_feed"
        else:
            if self._global_credit > 0:
                self._global_credit -= 1
            elif self._global_credit < 0:
                self._global_credit += 1
            action = "credit_hold"

        if self.telemetry and (self.calls <= 4 or stall_est > self.primary_stall_budget_ms or action == "credit_feed"):
            LOG.info(
                "H3VM DEV16 MLP #%d | phase=%s block=%d rows(root/helper)=%d/%d frac=%.2f | "
                "root=%.1fms shadow=%.1fms [stage %.1f+%.1f compute %.1f return %.1f+%.1f] | "
                "slack=%+.1fms credit=%+d action=%s ring=%s",
                self.calls, self._phase, index, rec["cut"], rec["seq"] - rec["cut"], frac,
                owner_ms, shadow_ms, stage_d2h, stage_h2d, helper_ms, ret_d2h, ret_h2d,
                slack_ms, self._global_credit, action, "hit" if rec.get("ring_slot") is not None else "fallback",
            )

    def step_summary(self):
        base = super().step_summary()
        base.update({
            "phase": self._phase,
            "ring_hits": self._step_ring_hits,
            "ring_misses": self._step_ring_misses,
            "ring_init_ms": self._step_ring_init_ms,
            "credit": self._global_credit,
            "policy": dict(self._step_policy_counts),
        })
        return base

    def record_step_runtime(self, **rec):
        item = {"step": int(self._step), "phase": self._phase}
        item.update({k: float(v) for k, v in rec.items()})
        item["ring_hits"] = int(self._step_ring_hits)
        item["ring_misses"] = int(self._step_ring_misses)
        item["ring_init_ms"] = float(self._step_ring_init_ms)
        item["stall_est_ms"] = float(self._step_stall_est_ms)
        item["slack_avg_ms"] = float(sum(self._step_slack_ms) / len(self._step_slack_ms)) if self._step_slack_ms else 0.0
        fracs = [float(self.block_primary_fraction.get(i, self.primary_fraction)) for i in self.active_blocks]
        item["root_fraction_avg"] = float(sum(fracs) / len(fracs)) if fracs else self.primary_fraction
        self._step_runtime.append(item)

    def final_summary(self):
        best = {int(i): round(float(self._best_safe_fraction(i)), 2) for i in sorted(self.active_blocks)}
        total = self._ring_hits_total + self._ring_misses_total
        hit_rate = self._ring_hits_total / total if total else 0.0
        valid_steps = [r for r in self._step_runtime if r.get("step", 0) >= 2]
        fastest = min(valid_steps, key=lambda r: r.get("wall_ms", 1e30)) if valid_steps else None
        return {
            "mode": "DEV16.0_PERSISTENT_WORKPOOL_ONE_SHOT",
            "ring_slots": self.ring_slots,
            "ring_hit_rate": round(hit_rate, 4),
            "ring_init_ms_total": round(self._ring_init_ms_total, 2),
            "steps": self._step_runtime,
            "best_root_fraction_by_block": best,
            "fastest_post_profile_step": fastest,
        }

    def close(self):
        self._rings.clear()
        self._history.clear()
        super().close()
