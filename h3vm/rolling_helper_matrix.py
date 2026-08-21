from __future__ import annotations

import json
import logging
import time

from .critical_path import CriticalPathMLPFabric, select_sidecar_blocks

LOG = logging.getLogger("H3VM")


class RollingCoverageMLPFabric(CriticalPathMLPFabric):
    """Dev17 one-shot 20/30/40/50-block rolling helper stress test.

    All 50 helper MLP mirrors exist as CPU-backed DynamicVRAM modules, but GPU1
    keeps only a bounded hot working set. A ComfyUI prefetch queue walks the
    active helper modules in block order. The current helper is consumed just
    before MLP execution; that consume also starts the next helper-weight load,
    so the next block's attention/root work becomes prefetch runway.

    This deliberately changes *coverage*, not token fraction. GPU0 remains the
    single root and exact token-row reconstruction is unchanged.
    """

    rolling_helper_matrix_mode = True

    def __init__(self, *args, coverage_schedule=(20, 30, 40, 50), **kwargs):
        kwargs["adaptive_slack"] = True
        # Guard only. We reset every step to the same baseline fraction so the
        # matrix measures coverage rather than accumulating fraction tuning.
        kwargs["min_primary_fraction"] = float(kwargs.get("primary_fraction", 0.68))
        kwargs["max_primary_fraction"] = max(0.72, float(kwargs.get("primary_fraction", 0.68)))
        kwargs["fraction_step"] = 0.02
        kwargs["target_slack_ms"] = 9999.0  # never harvest deeper in this experiment
        kwargs["resident_sidecar_packet"] = False
        super().__init__(*args, **kwargs)

        self._all_indices = sorted(int(i) for i in self.helper_mlp_by_block)
        self._coverage_schedule = tuple(max(0, min(len(self._all_indices), int(x))) for x in coverage_schedule)
        self._nested_order = self._make_nested_order()
        self._coverage_target = 0

        self._helper_prefetch_q = None
        self._helper_prefetch_indices = []
        self._helper_prefetch_host_ms = 0.0
        self._helper_prefetch_prime_ms = 0.0
        self._helper_prefetch_calls = 0
        self._helper_prefetch_errors = 0
        self._helper_prefetch_broken = False
        self._step_runtime = []
        self._sample_serial = 0

    def _make_nested_order(self):
        """Preserve the validated Dev15.2 twenty-block set, then fill gaps."""
        total = len(self._all_indices)
        if total <= 0:
            return []
        # For H3's 50 blocks this reproduces the historical Dev15.2 work packet.
        base_count = min(20, total)
        base = [i for i in select_sidecar_blocks(total, base_count, 5 if total > 5 else 0) if i in self.helper_mlp_by_block]
        selected = list(dict.fromkeys(base))
        remaining = [i for i in self._all_indices if i not in selected]

        # Greedy farthest-point fill keeps every expansion spatially distributed
        # instead of adding ten adjacent blocks at a time.
        while remaining:
            if not selected:
                pick = remaining[len(remaining) // 2]
            else:
                def score(idx):
                    return (min(abs(idx - s) for s in selected), -idx)
                pick = max(remaining, key=score)
            selected.append(pick)
            remaining.remove(pick)
        return selected

    def _reset_sample(self):
        self._step_runtime.clear()
        self._sample_serial += 1

    def _flush_helper_prefetch(self):
        q = self._helper_prefetch_q
        self._helper_prefetch_q = None
        self._helper_prefetch_indices = []
        if q is None:
            return
        try:
            import comfy.model_prefetch as mp
            if len(q) >= 2:
                mp.prefetch_queue_pop(q, self.secondary, None)
        except Exception:
            pass

    def _build_and_prime_prefetch(self, transformer_options=None, start_index=None):
        if self._helper_prefetch_q is not None or self._helper_prefetch_broken:
            return
        if not self.sidecar_ready() or not self.active_blocks:
            return
        try:
            import comfy.model_prefetch as mp
            opts = dict(transformer_options or {})
            opts["prefetch_dynamic_vbars"] = True
            indices = sorted(int(i) for i in self.active_blocks if start_index is None or int(i) >= int(start_index))
            if not indices:
                return
            modules = [self.helper_mlp_by_block[i] for i in indices]
            t0 = time.perf_counter()
            q = mp.make_prefetch_queue(modules, self.secondary, opts)
            build_ms = (time.perf_counter() - t0) * 1000.0
            if q is None:
                self._helper_prefetch_broken = True
                self._helper_prefetch_errors += 1
                LOG.warning("H3VM DEV17 helper prefetch queue unavailable; active helpers fail-open to direct DynamicVRAM")
                return
            self._helper_prefetch_q = q
            self._helper_prefetch_indices = indices
            # Same topology as the proven root queue: prime the first item once,
            # then each real block consumes its ready entry and launches the next.
            t1 = time.perf_counter()
            mp.prefetch_queue_pop(q, self.secondary, self.helper_mlp_by_block[indices[0]])
            prime_ms = (time.perf_counter() - t1) * 1000.0
            self._helper_prefetch_prime_ms += build_ms + prime_ms
            LOG.info(
                "H3VM DEV17 ROLLING HELPER QUEUE READY | coverage=%d first=%d last=%d build+prime=%.1fms",
                len(indices), indices[0], indices[-1], build_ms + prime_ms,
            )
        except Exception as exc:
            self._helper_prefetch_q = None
            self._helper_prefetch_broken = True
            self._helper_prefetch_errors += 1
            LOG.warning("H3VM DEV17 helper prefetch queue failed; direct DynamicVRAM fallback: %r", exc)

    def begin_step(self, step: int):
        self._flush_helper_prefetch()
        if int(step) <= 1 and self._step_runtime:
            self._reset_sample()
        super().begin_step(step)

        pos = min(max(1, int(step)), len(self._coverage_schedule)) - 1
        self._coverage_target = self._coverage_schedule[pos]
        self.active_blocks = set(self._nested_order[: self._coverage_target])
        # No cross-step fraction learning in this matrix. Each coverage point
        # starts from exactly the same 68/32 (or configured) split.
        self._disabled_blocks.clear()
        for idx in self._all_indices:
            self.block_primary_fraction[int(idx)] = self.primary_fraction
            self._miss_counts[int(idx)] = 0
            self._positive_slack_streak[int(idx)] = 0

        self._helper_prefetch_host_ms = 0.0
        self._helper_prefetch_prime_ms = 0.0
        self._helper_prefetch_calls = 0
        self._helper_prefetch_errors = 0
        self._helper_prefetch_broken = False

        LOG.info(
            "H3VM DEV17 COVERAGE PLAN step=%d | active=%d/%d helper_share=%.1f%% | blocks=%s",
            int(step), len(self.active_blocks), len(self._all_indices),
            (1.0 - self.primary_fraction) * 100.0, sorted(self.active_blocks),
        )
        # Step 2+ normally enters after the async sidecar prepare has completed.
        # Priming here gives block 0 runway when 40/50-block coverage includes it.
        self._build_and_prime_prefetch({"prefetch_dynamic_vbars": True})

    def prefetch_helper(self, index: int, transformer_options=None):
        # Called before the whole block, therefore before attention. If startup
        # preparation finished mid-step, this is the first safe chance to create
        # and prime the helper queue without a second CPU/GPU residency thread.
        if self._helper_prefetch_broken or int(index) not in self.active_blocks:
            return None
        if not self.sidecar_ready():
            return None
        self._build_and_prime_prefetch(transformer_options, start_index=int(index))
        return None

    def _consume_helper_prefetch(self, index: int):
        if self._helper_prefetch_q is None or self._helper_prefetch_broken:
            return
        try:
            import comfy.model_prefetch as mp
            t0 = time.perf_counter()
            mp.prefetch_queue_pop(self._helper_prefetch_q, self.secondary, self.helper_mlp_by_block[int(index)])
            dt = (time.perf_counter() - t0) * 1000.0
            self._helper_prefetch_host_ms += dt
            self._helper_prefetch_calls += 1
        except Exception as exc:
            self._helper_prefetch_errors += 1
            self._helper_prefetch_broken = True
            self._helper_prefetch_q = None
            LOG.warning("H3VM DEV17 helper prefetch consume failed block=%d; direct fallback: %r", int(index), exc)

    def execute(self, index: int, owner_mlp, x):
        index = int(index)
        if index in self.active_blocks and self.sidecar_ready():
            # This wait is the quantity that decides the whole experiment. If
            # previous blocks/attention hid the weight DMA, it should be ~0.
            self._consume_helper_prefetch(index)
        return super().execute(index, owner_mlp, x)

    def step_summary(self):
        base = super().step_summary()
        base.update({
            "coverage_target": int(self._coverage_target),
            "coverage_active": int(len(self.active_blocks)),
            "helper_prefetch_host_ms": float(self._helper_prefetch_host_ms),
            "helper_prefetch_prime_ms": float(self._helper_prefetch_prime_ms),
            "helper_prefetch_calls": int(self._helper_prefetch_calls),
            "helper_prefetch_errors": int(self._helper_prefetch_errors),
            "helper_prefetch_broken": bool(self._helper_prefetch_broken),
        })
        self._flush_helper_prefetch()
        return base

    def record_step_runtime(self, **rec):
        item = {
            "step": int(self._step),
            "coverage": int(self._coverage_target),
            "actual_sidecar_calls": int(self._step_calls),
            "helper_prefetch_host_ms": round(float(self._helper_prefetch_host_ms), 3),
            "helper_prefetch_prime_ms": round(float(self._helper_prefetch_prime_ms), 3),
            "helper_prefetch_calls": int(self._helper_prefetch_calls),
            "helper_prefetch_errors": int(self._helper_prefetch_errors),
            "primary_wait_helper_est_ms": round(float(self._step_stall_est_ms), 3),
            "slack_avg_ms": round(float(sum(self._step_slack_ms) / len(self._step_slack_ms)), 3) if self._step_slack_ms else 0.0,
        }
        item.update({k: round(float(v), 3) for k, v in rec.items()})
        self._step_runtime.append(item)
        if int(self._step) >= len(self._coverage_schedule):
            valid = [x for x in self._step_runtime if x.get("actual_sidecar_calls", 0) > 0]
            fastest = min(valid, key=lambda x: x.get("wall_ms", 1e30)) if valid else None
            lowest_primary = min(valid, key=lambda x: x.get("primary_compute_ms", 1e30)) if valid else None
            verdict = {
                "mode": "DEV17_ROLLING_MLP_20_30_40_50",
                "helper_universe": len(self._all_indices),
                "fixed_root_fraction": self.primary_fraction,
                "steps": self._step_runtime,
                "fastest_wall": fastest,
                "lowest_primary_compute": lowest_primary,
            }
            LOG.info("H3VM DEV17 FINAL RESULT %s", json.dumps(verdict, separators=(",", ":")))

    def close(self):
        self._flush_helper_prefetch()
        self._step_runtime.clear()
        super().close()
