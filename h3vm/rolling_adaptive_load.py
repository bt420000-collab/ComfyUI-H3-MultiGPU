from __future__ import annotations

import json
import logging

from .critical_path import CriticalPathMLPFabric
from .rolling_helper_matrix import RollingCoverageMLPFabric

LOG = logging.getLogger("H3VM")


class RollingAdaptiveLoadMLPFabric(RollingCoverageMLPFabric):
    """Dev17.1: all-50 rolling helper coverage + cross-step preassigned load shift.

    The helper universe and rolling weight queue are inherited from Dev17.0. The
    important difference is that coverage is fixed at all 50 blocks and token
    fractions are planned *before* each denoise step from measurements collected
    on previous steps. There is no same-step CPU steering of future blocks.

    Matrix:
      step1 CALIBRATE   : 68/32 on all 50 blocks
      step2 SHIFT34     : 66/34 on all 50 blocks
      step3 FEED        : per-block 68/32 .. 60/40 from step2 slack
      step4 LOCK        : fastest safe fraction observed on steps 1..3

    A fraction is considered safe only when helper stall estimate <= the primary
    stall budget and measured slack is non-negative. The lock phase additionally
    prefers at least 2ms slack where possible to avoid balancing on a knife edge.
    """

    rolling_adaptive_load_mode = True
    rolling_helper_matrix_mode = True

    def __init__(self, *args, **kwargs):
        kwargs["coverage_schedule"] = (50, 50, 50, 50)
        kwargs["primary_fraction"] = 0.68
        super().__init__(*args, **kwargs)
        self.min_primary_fraction = 0.60
        self.max_primary_fraction = 0.72
        self.target_slack_ms = 9999.0  # base fabric acts only as a miss guard
        self.harvest_confirmations = 99
        self._observations = {}  # step -> block -> metrics
        self._planned_fractions = {}
        self._phase = "CALIBRATE"

    def _reset_sample(self):
        super()._reset_sample()
        self._observations.clear()
        self._planned_fractions.clear()
        self._phase = "CALIBRATE"

    @staticmethod
    def _clamp_frac(x):
        return min(0.72, max(0.60, round(float(x) + 1e-9, 2)))

    def _feed_fraction(self, index: int):
        """Choose step3 split from the completed step2 measurement."""
        m = self._observations.get(2, {}).get(int(index))
        if not m:
            return 0.68
        slack = float(m.get("slack_ms", -1e9))
        stall = float(m.get("stall_ms", 1e9))
        if stall > self.primary_stall_budget_ms or slack < 0.0:
            return 0.68
        # Conservative pre-issued credits. The 60/40 cut requires a very large
        # proven runway; intermediate bands move only one or two 2%-credits.
        if slack >= 24.0:
            return 0.60
        if slack >= 18.0:
            return 0.62
        if slack >= 12.0:
            return 0.64
        if slack >= 6.0:
            return 0.66
        return 0.68

    def _lock_fraction(self, index: int):
        """Pick the fastest safe tested split, preferring >=2ms measured runway."""
        candidates = []
        soft = []
        for step in (1, 2, 3):
            m = self._observations.get(step, {}).get(int(index))
            if not m:
                continue
            if float(m.get("stall_ms", 1e9)) > self.primary_stall_budget_ms:
                continue
            if float(m.get("slack_ms", -1e9)) < 0.0:
                continue
            row = (float(m.get("window_ms", 1e30)), float(m.get("fraction", 0.68)), -float(m.get("slack_ms", 0.0)))
            soft.append(row)
            if float(m.get("slack_ms", 0.0)) >= 2.0:
                candidates.append(row)
        pool = candidates or soft
        if not pool:
            return 0.68
        # Window first, then deeper helper share, then larger safety slack.
        best = min(pool, key=lambda t: (t[0], t[1], t[2]))
        return self._clamp_frac(best[1])

    def _plan_for_step(self, step: int):
        if step <= 1:
            self._phase = "CALIBRATE"
            return {i: 0.68 for i in self._all_indices}
        if step == 2:
            self._phase = "SHIFT34"
            return {i: 0.66 for i in self._all_indices}
        if step == 3:
            self._phase = "FEED"
            return {i: self._feed_fraction(i) for i in self._all_indices}
        self._phase = "LOCK"
        return {i: self._lock_fraction(i) for i in self._all_indices}

    def begin_step(self, step: int):
        self._flush_helper_prefetch()
        if int(step) <= 1 and self._step_runtime:
            self._reset_sample()

        # Bypass RollingCoverageMLPFabric.begin_step because Dev17.0 resets every
        # block to 68/32. We want the completed previous-step plan instead.
        CriticalPathMLPFabric.begin_step(self, step)
        self._coverage_target = len(self._all_indices)
        self.active_blocks = set(self._all_indices)
        self._disabled_blocks.clear()
        self._miss_counts = {int(i): 0 for i in self._all_indices}
        self._positive_slack_streak = {int(i): 0 for i in self._all_indices}

        plan = self._plan_for_step(int(step))
        self._planned_fractions = dict(plan)
        for idx, frac in plan.items():
            self.block_primary_fraction[int(idx)] = self._clamp_frac(frac)

        self._helper_prefetch_host_ms = 0.0
        self._helper_prefetch_prime_ms = 0.0
        self._helper_prefetch_calls = 0
        self._helper_prefetch_errors = 0
        self._helper_prefetch_broken = False

        hist = {}
        for frac in plan.values():
            key = f"{self._clamp_frac(frac):.2f}"
            hist[key] = hist.get(key, 0) + 1
        LOG.info(
            "H3VM DEV17.1 LOAD PLAN step=%d phase=%s | coverage=50/50 | root_hist=%s helper_avg=%.1f%% | preassigned=YES",
            int(step), self._phase, hist,
            (1.0 - (sum(plan.values()) / max(1, len(plan)))) * 100.0,
        )
        self._build_and_prime_prefetch({"prefetch_dynamic_vbars": True})

    def on_block_complete(self, index: int):
        # Preserve the exact CUDA event packet before the base guard consumes it.
        index = int(index)
        rec = self._pending.get(index)
        super().on_block_complete(index)
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
        stall_ms = max(0.0, -slack_ms)
        self._observations.setdefault(int(self._step), {})[index] = {
            "fraction": float(rec.get("primary_fraction", self.primary_fraction)),
            "owner_ms": owner_ms,
            "helper_compute_ms": helper_ms,
            "shadow_ms": shadow_ms,
            "window_ms": max(owner_ms, shadow_ms),
            "slack_ms": slack_ms,
            "stall_ms": stall_ms,
        }

    def step_summary(self):
        base = super().step_summary()
        hist = {}
        for idx in self.active_blocks:
            frac = float(self._planned_fractions.get(int(idx), self.primary_fraction))
            key = f"{frac:.2f}"
            hist[key] = hist.get(key, 0) + 1
        base.update({
            "phase": self._phase,
            "planned_fraction_hist": hist,
            "planned_fraction_avg": (
                sum(float(v) for v in self._planned_fractions.values()) / len(self._planned_fractions)
                if self._planned_fractions else self.primary_fraction
            ),
        })
        return base

    def record_step_runtime(self, **rec):
        obs = self._observations.get(int(self._step), {})
        fractions = [float(m.get("fraction", 0.68)) for m in obs.values()]
        safe = [m for m in obs.values() if float(m.get("stall_ms", 0.0)) <= self.primary_stall_budget_ms and float(m.get("slack_ms", -1.0)) >= 0.0]
        hist = {}
        for frac in fractions:
            key = f"{frac:.2f}"
            hist[key] = hist.get(key, 0) + 1
        item = {
            "step": int(self._step),
            "phase": self._phase,
            "coverage": 50,
            "actual_sidecar_calls": int(self._step_calls),
            "helper_prefetch_host_ms": round(float(self._helper_prefetch_host_ms), 3),
            "helper_prefetch_prime_ms": round(float(self._helper_prefetch_prime_ms), 3),
            "helper_prefetch_calls": int(self._helper_prefetch_calls),
            "helper_prefetch_errors": int(self._helper_prefetch_errors),
            "primary_wait_helper_est_ms": round(float(self._step_stall_est_ms), 3),
            "slack_avg_ms": round(float(sum(self._step_slack_ms) / len(self._step_slack_ms)), 3) if self._step_slack_ms else 0.0,
            "root_fraction_hist": hist,
            "root_fraction_avg_actual": round(sum(fractions) / len(fractions), 4) if fractions else 0.68,
            "safe_blocks": len(safe),
        }
        item.update({k: round(float(v), 3) for k, v in rec.items()})
        self._step_runtime.append(item)

        if int(self._step) >= 4:
            valid = [x for x in self._step_runtime if x.get("actual_sidecar_calls", 0) > 0]
            safe_steps = [x for x in valid if x.get("primary_wait_helper_est_ms", 1e30) <= 1.0]
            fastest = min(safe_steps or valid, key=lambda x: x.get("wall_ms", 1e30)) if valid else None
            lowest_primary = min(safe_steps or valid, key=lambda x: x.get("primary_compute_ms", 1e30)) if valid else None
            lock_plan = {str(i): round(float(self._planned_fractions.get(i, 0.68)), 2) for i in self._all_indices}
            lock_hist = {}
            for frac in lock_plan.values():
                key = f"{float(frac):.2f}"
                lock_hist[key] = lock_hist.get(key, 0) + 1
            verdict = {
                "mode": "DEV17.1_ROLLING_50_ADAPTIVE_LOAD",
                "contract": "ALL50_ROLLING__CROSS_STEP_PREASSIGNED__NO_SAME_STEP_STEERING",
                "helper_universe": len(self._all_indices),
                "steps": self._step_runtime,
                "lock_fraction_by_block": lock_plan,
                "lock_hist": lock_hist,
                "fastest_safe_wall": fastest,
                "lowest_safe_primary_compute": lowest_primary,
            }
            LOG.info("H3VM DEV17.1 FINAL RESULT %s", json.dumps(verdict, separators=(",", ":")))

    def close(self):
        self._observations.clear()
        self._planned_fractions.clear()
        super().close()
