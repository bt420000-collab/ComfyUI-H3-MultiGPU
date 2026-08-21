from __future__ import annotations

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor

LOG = logging.getLogger("H3VM")


def _tensor_nbytes(value):
    import torch
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_nbytes(x) for x in value)
    if isinstance(value, dict):
        return sum(_tensor_nbytes(x) for x in value.values())
    return 0


def _used_gib(device):
    import torch
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / (1024 ** 3)


class BoundedPinnedMailbox:
    """Small explicit pinned host cache for activation/snapshot traffic only.

    This intentionally ignores ComfyUI's global pinned-memory policy. It never
    pins model weights and never grows beyond ``cap_mb``. The safe launch profile
    can therefore keep ``--disable-pinned-memory`` while H3VM owns a small DMA
    runway for its own host mailbox.
    """

    def __init__(self, cap_mb=1024, stats_recorder=None):
        self.cap_bytes = max(0, int(cap_mb)) * 1024 * 1024
        self._buffers = {}
        self._bytes = 0
        self._warned = set()
        self._stats_recorder = stats_recorder
        self._lock = threading.RLock()

    @staticmethod
    def _nbytes(t):
        return int(t.numel() * t.element_size())

    def _record(self, nbytes, dt):
        fn = self._stats_recorder
        if fn is not None:
            try:
                fn("bounded_pinned", int(nbytes), float(dt))
            except Exception:
                pass

    def _alloc(self, key, template):
        import torch
        with self._lock:
            spec = (tuple(template.shape), template.dtype)
            entry = self._buffers.get(key)
            if entry is not None and entry["spec"] == spec:
                return entry["tensor"]
            if entry is not None:
                self._bytes -= entry["bytes"]
                self._buffers.pop(key, None)

            need = self._nbytes(template)
            if self._bytes + need > self.cap_bytes:
                if key not in self._warned:
                    LOG.warning(
                        "H3VM Dev9.3.1 pinned mailbox fallback | key=%s need=%.1fMiB used/cap=%.1f/%.1fMiB",
                        key, need/(1024**2), self._bytes/(1024**2), self.cap_bytes/(1024**2),
                    )
                    self._warned.add(key)
                return None
            try:
                out = torch.empty(tuple(template.shape), dtype=template.dtype, device="cpu", pin_memory=True)
            except Exception as e:
                if key not in self._warned:
                    LOG.warning("H3VM Dev9.3.1 pinned mailbox allocation failed key=%s: %r", key, e)
                    self._warned.add(key)
                return None
            self._buffers[key] = {"tensor": out, "spec": spec, "bytes": need}
            self._bytes += need
            return out

    def to_host(self, src, key):
        import torch
        if src.device.type == "cpu":
            return src
        buf = self._alloc(key, src)
        t0 = time.perf_counter()
        if buf is None:
            out = src.to("cpu")
            torch.cuda.synchronize(src.device)
            self._record(self._nbytes(src), time.perf_counter() - t0)
            return out
        buf.copy_(src, non_blocking=True)
        torch.cuda.synchronize(src.device)
        self._record(self._nbytes(src), time.perf_counter() - t0)
        return buf

    def to_device(self, src, dst_device):
        import torch
        dst = torch.device(dst_device)
        if src.device == dst:
            return src
        t0 = time.perf_counter()
        if src.device.type == "cpu" and bool(src.is_pinned()):
            out = torch.empty_like(src, device=dst)
            out.copy_(src, non_blocking=True)
        else:
            out = src.to(dst, non_blocking=False)
        torch.cuda.synchronize(dst)
        self._record(self._nbytes(src), time.perf_counter() - t0)
        return out

    def predict(self, current, previous, beta, key="predict"):
        """First-order predictor with a pinned output when budget allows."""
        import torch
        if current.device.type != "cpu" or previous.device.type != "cpu":
            raise RuntimeError("Pinned mailbox predictor expects CPU snapshots")
        out = self._alloc(key, current)
        t0 = time.perf_counter()
        if out is None:
            out = torch.add(current, current, alpha=float(beta))
            out.add_(previous, alpha=-float(beta))
        else:
            torch.mul(current, 1.0 + float(beta), out=out)
            out.add_(previous, alpha=-float(beta))
        return out, (time.perf_counter() - t0) * 1000.0

    @property
    def allocated_mib(self):
        return self._bytes / (1024 ** 2)

    def clear(self):
        with self._lock:
            self._buffers.clear()
            self._bytes = 0
            self._warned.clear()


def should_use_exact_step(step: int, expected_steps: int, refresh_interval: int, exact_last_step: bool, exact_steps=None) -> bool:
    """Return True when the tail must consume the current prefix snapshot.

    Step 1 is always exact to seed the mailbox. Optional periodic refreshes limit
    stale-feature drift. The last expected call can also be exact so the final
    denoise prediction is not based on a one-call-old boundary activation.
    """
    step = int(step)
    expected_steps = max(0, int(expected_steps))
    refresh_interval = max(0, int(refresh_interval))
    if exact_steps is not None and step in set(int(x) for x in exact_steps):
        return True
    if step <= 1:
        return True
    if exact_last_step and expected_steps > 0 and step == expected_steps:
        return True
    if refresh_interval > 0 and (step - 1) % refresh_interval == 0:
        return True
    return False


class SnapshotIslandRuntime:
    """Approximate two-island H3 pipeline with a host-owned stale snapshot mailbox.

    There is intentionally no GPU0->GPU1 or GPU1->GPU0 tensor handoff in this
    runtime. Each accelerator talks only to ordinary pageable system RAM:

      primary fixed/front-end -> host -> secondary prefix island
      secondary prefix output -> host snapshot mailbox
      previous host snapshot -> primary tail island

    After the warm-up call, the primary tail consumes boundary h from the
    previous denoise/model call while the secondary computes the current prefix.
    This removes the exact inter-island barrier at the cost of one-call-stale
    boundary features. It is an experiment, not an exact H3 execution mode.
    """

    def __init__(self, primary_device, secondary_device, prefix_blocks, tail_blocks, split_count, *,
                 space, expected_steps=20, refresh_interval=0, exact_last_step=True,
                 prefix_prefetch=True, tail_prefetch=True, telemetry=True,
                 predictor_mode="stale", predictor_beta=0.75,
                 launch_tail_before_stage=False, host_feeder="pageable",
                 pinned_mailbox_mb=1024, exact_steps=None, quality_profile="FAST | E-S-P-E"):
        self.primary_device = primary_device
        self.secondary_device = secondary_device
        self.prefix_blocks = dict(prefix_blocks)
        self.tail_blocks = dict(tail_blocks)
        self.split_count = int(split_count)
        self.first_prefix = 0
        self.last_prefix = self.split_count - 1
        self.first_tail = self.split_count
        self.last_tail = max(self.tail_blocks) if self.tail_blocks else self.split_count - 1
        self.space = space
        self.expected_steps = max(0, int(expected_steps))
        self.refresh_interval = max(0, int(refresh_interval))
        self.exact_last_step = bool(exact_last_step)
        self.prefix_prefetch = bool(prefix_prefetch)
        self.tail_prefetch = bool(tail_prefetch)
        self.telemetry = bool(telemetry)
        self.predictor_mode = str(predictor_mode)
        if self.predictor_mode not in ("stale", "linear"):
            raise ValueError(f"Unsupported snapshot predictor mode: {self.predictor_mode}")
        self.predictor_beta = float(predictor_beta)
        self.launch_tail_before_stage = bool(launch_tail_before_stage)
        self.host_feeder = str(host_feeder)
        if self.host_feeder not in ("pageable", "bounded_pinned"):
            raise ValueError(f"Unsupported host feeder: {self.host_feeder}")
        self.pinned_mailbox_mb = max(0, int(pinned_mailbox_mb))
        self.exact_steps = frozenset(int(x) for x in (exact_steps or ()))
        self.quality_profile = str(quality_profile)
        self._pinned = (
            BoundedPinnedMailbox(cap_mb=self.pinned_mailbox_mb, stats_recorder=getattr(self.space.transport, "_record", None))
            if self.host_feeder == "bounded_pinned" else None
        )

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="H3VM-PrimaryIsland")
        self._step = 0
        self._snapshot_cpu = None
        self._snapshot_prev_cpu = None
        self._pending_snapshot_cpu = None
        self._future = None
        self._tail_output = None
        self._prefix_t_emb = None
        self._prefix_segments = None
        self._prefix_rope = None
        self._prefix_queue = None
        self._prefix_started = None
        self._prefix_compute_started = None
        self._step_started = None
        self._step_stats0 = None
        self._exact_this_step = True
        self._snapshot_write_ms = 0.0
        self._stage_prefix_ms = 0.0
        self._predictor_used = False
        self._peak_primary = 0.0
        self._peak_secondary = 0.0
        self._prefix_logged = False
        self._tail_logged = False

    @staticmethod
    def _stats_snapshot(space):
        s = space.transport.stats
        by = s.get("by_mode", {})
        direct = by.get("direct_d2d", {"moves": 0, "bytes": 0, "seconds": 0.0})
        return (
            int(s.get("moves", 0)), int(s.get("bytes", 0)), float(s.get("seconds", 0.0)),
            int(direct.get("moves", 0)), int(direct.get("bytes", 0)),
        )

    def _sample_memory(self):
        if not self.telemetry:
            return
        try:
            self._peak_primary = max(self._peak_primary, _used_gib(self.primary_device))
            self._peak_secondary = max(self._peak_secondary, _used_gib(self.secondary_device))
        except Exception:
            pass

    def _make_prefetch(self, blocks, device, transformer_options, enabled, which):
        if not enabled:
            return None
        try:
            import comfy.model_prefetch
            opts = dict(transformer_options or {})
            opts["prefetch_dynamic_vbars"] = True
            queue = [blocks[i] for i in sorted(blocks)]
            out = comfy.model_prefetch.make_prefetch_queue(queue, device, opts)
            flag = "_prefix_logged" if which == "prefix" else "_tail_logged"
            if not getattr(self, flag):
                LOG.info(
                    "H3VM Dev9 %s island prefetch | active=%s blocks=%d device=%s",
                    which, out is not None, len(queue), device,
                )
                setattr(self, flag, True)
            return out
        except Exception as e:
            LOG.warning("H3VM Dev9 %s island prefetch disabled: %r", which, e)
            return None

    def _finish_prefetch(self, queue, device, which):
        """Safely retire the last prefetched block without draining past the sentinel.

        ComfyUI's queue is shaped as [None] + blocks + [None]. One pop per block
        plus one final flush is valid. Dev9.3.1 experimentally added an extra early
        pop to 'prime' the queue, which shifted the queue by one and made the
        final flush consume the last sentinel, after which model_prefetch tried
        to read queue[0] from an empty list. Do not pre-consume the queue.
        """
        if queue is None:
            return
        try:
            import comfy.model_prefetch
            # Normal unprimed path reaches the flush with two entries left.
            # If a future ComfyUI change or partial failure leaves only the final
            # sentinel, do not pop it because prefetch_queue_pop indexes queue[0]
            # after the pop.
            if len(queue) >= 2:
                comfy.model_prefetch.prefetch_queue_pop(queue, device, None)
            elif len(queue) == 1:
                LOG.debug("H3VM Dev9.3.1 %s prefetch already drained to sentinel; skip final pop", which)
        except Exception as e:
            LOG.warning("H3VM Dev9.3.1 %s prefetch finalization failed: %r", which, e)

    def _host_only_tree(self, value, device, *, feeder_key=None):
        """Acquire a device replica strictly through host memory.

        Large activations can use the bounded pinned mailbox. Tiny metadata stays
        pageable so the explicit pinned budget is spent only on the hot path.
        """
        import torch
        if torch.is_tensor(value):
            if value.device == device:
                return value
            if self._pinned is not None and feeder_key is not None:
                host = self._pinned.to_host(value, feeder_key)
                return self._pinned.to_device(host, device)
            host = self.space.transport.move_tensor(value, "cpu", mode="neutral_pageable")
            return self.space.transport.move_tensor(host, device, mode="neutral_pageable")
        if isinstance(value, tuple):
            return tuple(self._host_only_tree(x, device) for x in value)
        if isinstance(value, list):
            return [self._host_only_tree(x, device) for x in value]
        if isinstance(value, dict):
            return {k: self._host_only_tree(v, device) for k, v in value.items()}
        return value

    def _run_tail(self, snapshot_cpu, t_emb, mod_segments, rope_freqs, transformer_options, step,
                  predictor_prev_cpu=None, predictor_mode="stale", predictor_beta=0.75):
        import torch
        import comfy.model_management
        import comfy.model_prefetch

        if snapshot_cpu is None:
            raise RuntimeError("H3VM Dev9 tail island received no host snapshot")
        worker_started = time.perf_counter()
        predictor_ms = 0.0
        predictor_used = False
        tail_snapshot = snapshot_cpu
        if predictor_mode == "linear" and predictor_prev_cpu is not None:
            if tuple(predictor_prev_cpu.shape) == tuple(snapshot_cpu.shape):
                if self._pinned is not None:
                    tail_snapshot, predictor_ms = self._pinned.predict(
                        snapshot_cpu, predictor_prev_cpu, float(predictor_beta), key="predict"
                    )
                else:
                    pred0 = time.perf_counter()
                    tail_snapshot = torch.add(snapshot_cpu, snapshot_cpu, alpha=float(predictor_beta))
                    tail_snapshot.add_(predictor_prev_cpu, alpha=-float(predictor_beta))
                    predictor_ms = (time.perf_counter() - pred0) * 1000.0
                predictor_used = True

        queue = self._make_prefetch(
            self.tail_blocks, self.primary_device, transformer_options,
            self.tail_prefetch, "tail",
        )
        with comfy.model_management.cuda_device_context(self.primary_device):
            read0 = time.perf_counter()
            if self._pinned is not None:
                h = self._pinned.to_device(tail_snapshot, self.primary_device)
            else:
                h = self.space.transport.move_tensor(
                    tail_snapshot, self.primary_device, mode="neutral_pageable"
                )
            snapshot_read_ms = (time.perf_counter() - read0) * 1000.0
            compute0 = time.perf_counter()
            for i in range(self.first_tail, self.last_tail + 1):
                block = self.tail_blocks[i]
                if queue is not None:
                    comfy.model_prefetch.prefetch_queue_pop(queue, self.primary_device, block)
                h = block(
                    h, t_emb, mod_segments, rope_freqs,
                    transformer_options=transformer_options or {},
                )
            self._finish_prefetch(queue, self.primary_device, "tail")
            torch.cuda.synchronize(self.primary_device)
            tail_compute_ms = (time.perf_counter() - compute0) * 1000.0
        return {
            "h": h,
            "snapshot_read_ms": snapshot_read_ms,
            "predictor_ms": predictor_ms,
            "predictor_used": predictor_used,
            "tail_compute_ms": tail_compute_ms,
            "tail_wall_ms": (time.perf_counter() - worker_started) * 1000.0,
            "step": int(step),
        }

    def _reset_call_state(self):
        self._pending_snapshot_cpu = None
        self._future = None
        self._tail_output = None
        self._prefix_t_emb = None
        self._prefix_segments = None
        self._prefix_rope = None
        self._prefix_queue = None
        self._prefix_started = None
        self._prefix_compute_started = None
        self._step_started = None
        self._step_stats0 = None
        self._snapshot_write_ms = 0.0
        self._stage_prefix_ms = 0.0
        self._predictor_used = False

    def _reset_sampling_run(self):
        self._snapshot_cpu = None
        self._snapshot_prev_cpu = None
        self._step = 0
        self._reset_call_state()

    def _begin_prefix(self, h, t_emb, mod_segments, rope_freqs, transformer_options):
        import torch
        self._step += 1
        step = self._step
        self._step_started = time.perf_counter()
        self._step_stats0 = self._stats_snapshot(self.space)
        self._peak_primary = self._peak_secondary = 0.0
        self._exact_this_step = should_use_exact_step(
            step, self.expected_steps, self.refresh_interval, self.exact_last_step, self.exact_steps
        )

        # Shape changes imply a new logical sampling run even if expected_steps was
        # misconfigured. Never feed a stale mailbox across incompatible shapes.
        if self._snapshot_cpu is not None and tuple(self._snapshot_cpu.shape) != tuple(h.shape):
            LOG.warning(
                "H3VM Dev9 snapshot shape changed %s -> %s; resetting stale mailbox",
                tuple(self._snapshot_cpu.shape), tuple(h.shape),
            )
            self._snapshot_cpu = None
            self._snapshot_prev_cpu = None
            self._step = 1
            step = 1
            self._exact_this_step = True

        can_pipeline = (not self._exact_this_step and self._snapshot_cpu is not None)

        # Full-throttle mode launches the primary tail *before* staging the
        # current prefix to the secondary island. The tail only consumes the
        # previous host snapshot, so it has no dependency on this staging work.
        # This overlaps GPU0 tail startup with GPU0->RAM->GPU1 input staging.
        if can_pipeline and self.launch_tail_before_stage:
            self._future = self._executor.submit(
                self._run_tail,
                self._snapshot_cpu,
                t_emb,
                mod_segments,
                rope_freqs,
                dict(transformer_options or {}),
                step,
                self._snapshot_prev_cpu,
                self.predictor_mode,
                self.predictor_beta,
            )
        else:
            self._future = None

        # Keep ComfyUI's DynamicVRAM prefetch queue unmodified until the first
        # real block call. Dev9.3's extra early queue pop was unsafe because it
        # consumed one queue entry too many and could poison CUDA before flush.
        self._prefix_queue = self._make_prefetch(
            self.prefix_blocks, self.secondary_device, transformer_options,
            self.prefix_prefetch, "prefix",
        )

        # Current prefix input and its tiny read-only metadata enter the
        # secondary island through host memory. There is still no GPU-to-GPU
        # tensor handoff.
        stage0 = time.perf_counter()
        h2 = self._host_only_tree(h, self.secondary_device, feeder_key="stage_h")
        self._prefix_t_emb = self._host_only_tree(t_emb, self.secondary_device)
        self._prefix_segments = self._host_only_tree(mod_segments, self.secondary_device)
        self._prefix_rope = self._host_only_tree(rope_freqs, self.secondary_device)
        self._stage_prefix_ms = (time.perf_counter() - stage0) * 1000.0

        if can_pipeline and self._future is None:
            self._future = self._executor.submit(
                self._run_tail,
                self._snapshot_cpu,
                t_emb,
                mod_segments,
                rope_freqs,
                dict(transformer_options or {}),
                step,
                self._snapshot_prev_cpu,
                self.predictor_mode,
                self.predictor_beta,
            )
        self._prefix_started = time.perf_counter()
        self._prefix_compute_started = self._prefix_started
        self._sample_memory()
        return h2

    def prefix_execute(self, index, h, t_emb, mod_segments, rope_freqs, transformer_options=None):
        import torch
        import comfy.model_management
        import comfy.model_prefetch

        if index == self.first_prefix:
            h = self._begin_prefix(h, t_emb, mod_segments, rope_freqs, transformer_options)
        elif self._prefix_t_emb is None:
            h = self._begin_prefix(h, t_emb, mod_segments, rope_freqs, transformer_options)
        elif getattr(h, "device", None) != self.secondary_device:
            h = self._host_only_tree(h, self.secondary_device)

        block = self.prefix_blocks[index]
        with comfy.model_management.cuda_device_context(self.secondary_device):
            if self._prefix_queue is not None:
                comfy.model_prefetch.prefetch_queue_pop(self._prefix_queue, self.secondary_device, block)
            h = block(
                h, self._prefix_t_emb, self._prefix_segments, self._prefix_rope,
                transformer_options=transformer_options or {},
            )

        if index == self.last_prefix:
            self._finish_prefetch(self._prefix_queue, self.secondary_device, "prefix")
            torch.cuda.synchronize(self.secondary_device)
            self._prefix_compute_ms = (time.perf_counter() - self._prefix_compute_started) * 1000.0
            snap0 = time.perf_counter()
            if self._pinned is not None:
                snap_key = f"snapshot_{self._step % 3}"
                self._pending_snapshot_cpu = self._pinned.to_host(h, snap_key)
            else:
                self._pending_snapshot_cpu = self.space.transport.move_tensor(
                    h, "cpu", mode="neutral_pageable"
                ).contiguous()
            self._snapshot_write_ms = (time.perf_counter() - snap0) * 1000.0
            self._sample_memory()
        return h

    def _consume_tail(self, t_emb, mod_segments, rope_freqs, transformer_options):
        import torch
        step = self._step
        wait0 = time.perf_counter()
        if self._future is not None:
            result = self._future.result()
        else:
            # Exact warm-up/refresh/flush: consume the current prefix snapshot.
            result = self._run_tail(
                self._pending_snapshot_cpu,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options or {},
                step,
                None,
                "stale",
                self.predictor_beta,
            )
        wait_ms = (time.perf_counter() - wait0) * 1000.0
        self._tail_output = result["h"]
        self._sample_memory()

        s1 = self._stats_snapshot(self.space)
        s0 = self._step_stats0 or (0, 0, 0.0, 0, 0)
        moves = s1[0] - s0[0]
        moved = s1[1] - s0[1]
        transport_ms = max(0.0, s1[2] - s0[2]) * 1000.0
        direct_moves = s1[3] - s0[3]
        direct_bytes = s1[4] - s0[4]
        step_wall_ms = (time.perf_counter() - self._step_started) * 1000.0 if self._step_started else 0.0
        theoretical_serial = float(getattr(self, "_prefix_compute_ms", 0.0)) + float(result["tail_compute_ms"])
        overlap_factor = theoretical_serial / max(1e-9, step_wall_ms - self._stage_prefix_ms)

        predictor_used = bool(result.get("predictor_used", False))
        mode_name = "exact" if self._exact_this_step else ("predicted-pipeline" if predictor_used else "stale-pipeline")
        if self.telemetry and (step <= 3 or step % 10 == 0 or (self.expected_steps and step == self.expected_steps)):
            LOG.info(
                "H3VM Dev10.1 QUALITY step #%d | profile=%s mode=%s stale_age=%d | prefix=%.1fms tail=%.1fms "
                "bubble=%.1fms tail_wait=%.1fms overlap=%.2fx | host stage=%.1fms predictor=%.1fms beta=%.2f "
                "snapshot write/read=%.1f/%.1fms | host moves=%d payload=%.1fMiB transport=%.1fms | "
                "feeder=%s pinned=%.1fMiB | D2D moves=%d bytes=%.1fMiB | peak %s=%.2fGiB %s=%.2fGiB",
                step,
                self.quality_profile,
                mode_name,
                0 if self._exact_this_step else 1,
                float(getattr(self, "_prefix_compute_ms", 0.0)),
                float(result["tail_compute_ms"]),
                max(0.0, float(result["tail_compute_ms"]) - float(getattr(self, "_prefix_compute_ms", 0.0))),
                wait_ms,
                overlap_factor,
                self._stage_prefix_ms,
                float(result.get("predictor_ms", 0.0)),
                self.predictor_beta,
                self._snapshot_write_ms,
                float(result["snapshot_read_ms"]),
                moves, moved/(1024**2), transport_ms,
                self.host_feeder,
                float(self._pinned.allocated_mib if self._pinned is not None else 0.0),
                direct_moves, direct_bytes/(1024**2),
                self.primary_device, self._peak_primary,
                self.secondary_device, self._peak_secondary,
            )
        return self._tail_output

    def tail_execute(self, index, h, t_emb, mod_segments, rope_freqs, transformer_options=None):
        if index == self.first_tail:
            h = self._consume_tail(t_emb, mod_segments, rope_freqs, transformer_options)
        else:
            h = self._tail_output if self._tail_output is not None else h

        if index == self.last_tail:
            # Publish the current secondary prefix snapshot only after the current
            # tail has consumed the prior version. Next call sees exactly age=1.
            self._snapshot_prev_cpu = self._snapshot_cpu
            self._snapshot_cpu = self._pending_snapshot_cpu
            finished = self.expected_steps > 0 and self._step >= self.expected_steps
            self._reset_call_state()
            if finished:
                # Prevent stale data from leaking into the next queued prompt.
                self._snapshot_cpu = None
                self._snapshot_prev_cpu = None
                self._step = 0
        return h

    def close(self):
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._snapshot_cpu = None
        self._snapshot_prev_cpu = None
        self._pending_snapshot_cpu = None
        self._future = None
        self._tail_output = None
        self._prefix_queue = None
        if self._pinned is not None:
            self._pinned.clear()

    def __del__(self):
        self.close()


class PrefixProxyFactory:
    @staticmethod
    def make(runtime, index):
        import torch
        import weakref

        class H3SnapshotPrefixProxy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._runtime_ref = weakref.ref(runtime)
                self.index = int(index)

            def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
                runtime_obj = self._runtime_ref()
                if runtime_obj is None:
                    raise RuntimeError("H3VM snapshot runtime has been released")
                return runtime_obj.prefix_execute(
                    self.index, x, t_emb, mod_segments, rope_freqs,
                    transformer_options=transformer_options,
                )

        return H3SnapshotPrefixProxy()


class TailProxyFactory:
    @staticmethod
    def make(runtime, index):
        import torch
        import weakref

        class H3SnapshotTailProxy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._runtime_ref = weakref.ref(runtime)
                self.index = int(index)

            def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
                runtime_obj = self._runtime_ref()
                if runtime_obj is None:
                    raise RuntimeError("H3VM snapshot runtime has been released")
                return runtime_obj.tail_execute(
                    self.index, x, t_emb, mod_segments, rope_freqs,
                    transformer_options=transformer_options,
                )

        return H3SnapshotTailProxy()


def make_island_root(source_base_model, source_dm, block_map, total_blocks, label):
    import torch

    class EmptySlot(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError(f"H3VM Dev9 empty {label} island slot executed")

    class IslandDiffusionView(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([
                block_map[i] if i in block_map else EmptySlot()
                for i in range(total_blocks)
            ])
            for name in ("hidden_size", "sigma_shift_video", "sigma_shift_audio", "use_adaln_curves", "dtype"):
                if hasattr(source_dm, name):
                    setattr(self, name, getattr(source_dm, name))

    class H3IslandRoot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = IslandDiffusionView()
            self.island_label = str(label)
            for name in ("manual_cast_dtype", "model_dtype", "dtype"):
                if hasattr(source_base_model, name):
                    setattr(self, name, getattr(source_base_model, name))

        def memory_required(self, input_shape=None):
            return 0

    return H3IslandRoot()
