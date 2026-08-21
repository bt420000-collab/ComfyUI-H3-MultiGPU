from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

LOG = logging.getLogger("H3VM")


def _tensor_nbytes(value):
    try:
        import torch
        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return sum(_tensor_nbytes(x) for x in value)
    if isinstance(value, dict):
        return sum(_tensor_nbytes(x) for x in value.values())
    return 0


def _mem_snapshot(device):
    import torch
    free, total = torch.cuda.mem_get_info(device)
    return {
        "free": int(free),
        "total": int(total),
        "used": int(total - free),
        "alloc": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
    }


def _gib(v):
    return float(v) / (1024 ** 3)


class PrimaryFirstExactRuntime:
    """Dev11 exact primary-first H3 executor.

    Semantics are deliberately boring and exact: blocks execute in their original
    order, with a host-neutral boundary transfer between the two islands. There is
    no stale snapshot and no predictor.

    The optimization under test is *load order*, not math:
      1) the 16G primary prefix is the only island registered as an additional
         model at sampler startup;
      2) on the first real prefix call, a worker starts loading the 8G tail;
      3) primary blocks compute while the secondary tail is being prepared;
      4) the exact current boundary activation is staged through CPU and the tail
         begins only after both the boundary and the secondary patcher are ready.
    """

    def __init__(self, primary_device, secondary_device, prefix_blocks, tail_blocks, split_count, *,
                 space, expected_steps=4, prefix_prefetch=True, tail_prefetch=True,
                 telemetry=True, lazy_secondary=True, hard_cleanup_after_sample=False):
        self.primary_device = primary_device
        self.secondary_device = secondary_device
        self.prefix_blocks = dict(prefix_blocks)
        self.tail_blocks = dict(tail_blocks)
        self.split_count = int(split_count)
        self.first_prefix = min(self.prefix_blocks)
        self.last_prefix = max(self.prefix_blocks)
        self.first_tail = min(self.tail_blocks)
        self.last_tail = max(self.tail_blocks)
        self.space = space
        self.expected_steps = max(1, int(expected_steps))
        self.prefix_prefetch = bool(prefix_prefetch)
        self.tail_prefetch = False  # Dev11.0.5 safety: never prefetch the 8G tail
        self.telemetry = bool(telemetry)
        self.lazy_secondary = bool(lazy_secondary)
        self.hard_cleanup_after_sample = bool(hard_cleanup_after_sample)

        self.secondary_patcher = None
        self._load_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="H3VM-SecondaryFeed")
        self._secondary_future = None
        self._secondary_ready = False
        self._secondary_load_ms = 0.0
        self._secondary_lock = threading.RLock()

        self._step = 0
        self._boundary_cpu = None
        self._tail_output = None
        self._prefix_queue = None
        self._tail_queue = None
        self._t_emb = None
        self._segments = None
        self._rope = None
        self._step_started = None
        self._prefix_started = None
        self._prefix_ms = 0.0
        self._boundary_write_ms = 0.0
        self._boundary_read_ms = 0.0
        self._return_primary_ms = 0.0
        self._peak_primary = 0.0
        self._peak_secondary = 0.0
        self._run_active = False
        self._sampling_generation = 0

    def bind_secondary_patcher(self, patcher):
        self.secondary_patcher = patcher

    def _sample_memory(self):
        try:
            p = _mem_snapshot(self.primary_device)
            s = _mem_snapshot(self.secondary_device)
            self._peak_primary = max(self._peak_primary, _gib(p["used"]))
            self._peak_secondary = max(self._peak_secondary, _gib(s["used"]))
        except Exception:
            pass

    def _make_prefetch(self, blocks, device, transformer_options, enabled, label):
        if not enabled:
            return None
        try:
            import comfy.model_prefetch
            opts = dict(transformer_options or {})
            opts["prefetch_dynamic_vbars"] = True
            queue = [blocks[i] for i in sorted(blocks)]
            out = comfy.model_prefetch.make_prefetch_queue(queue, device, opts)
            LOG.debug("H3VM Dev11 %s prefetch active=%s blocks=%d device=%s", label, out is not None, len(queue), device)
            return out
        except Exception as e:
            LOG.warning("H3VM Dev11 %s prefetch disabled: %r", label, e)
            return None

    def _finish_prefetch(self, queue, device, label):
        if queue is None:
            return
        try:
            import comfy.model_prefetch
            if len(queue) >= 2:
                comfy.model_prefetch.prefetch_queue_pop(queue, device, None)
        except Exception as e:
            LOG.warning("H3VM Dev11 %s prefetch finalization warning: %r", label, e)

    def _load_secondary_worker(self):
        import comfy.model_management as mm
        if self.secondary_patcher is None:
            raise RuntimeError("H3VM Dev11 secondary patcher is not bound")
        t0 = time.perf_counter()
        # Use ComfyUI's normal DynamicVRAM-aware loader. This is intentionally
        # launched only after the primary model has already started the call.
        mm.load_models_gpu([self.secondary_patcher])
        dt = (time.perf_counter() - t0) * 1000.0
        with self._secondary_lock:
            self._secondary_ready = True
            self._secondary_load_ms = dt
        return dt

    def _ensure_secondary_feed_started(self):
        if self.secondary_patcher is None:
            raise RuntimeError("H3VM Dev11 secondary patcher is not bound")
        with self._secondary_lock:
            if self._secondary_ready:
                return
            if self._secondary_future is None:
                self._secondary_future = self._load_executor.submit(self._load_secondary_worker)
                LOG.info("H3VM Dev11 LAZY FEED | secondary load started behind primary compute | device=%s", self.secondary_device)

    def _wait_secondary_ready(self):
        if not self.lazy_secondary:
            return 0.0
        self._ensure_secondary_feed_started()
        t0 = time.perf_counter()
        fut = self._secondary_future
        if fut is not None:
            fut.result()
        return (time.perf_counter() - t0) * 1000.0

    def sampling_begin(self):
        self._sampling_generation += 1
        self._run_active = True
        self._step = 0
        self._boundary_cpu = None
        self._tail_output = None
        self._secondary_future = None
        self._secondary_ready = False
        self._secondary_load_ms = 0.0
        try:
            p = _mem_snapshot(self.primary_device)
            s = _mem_snapshot(self.secondary_device)
            LOG.info(
                "H3VM Dev11 PRE-FLIGHT gen=%d | %s free=%.2fGiB used=%.2fGiB torch=%.2f/%.2fGiB | "
                "%s free=%.2fGiB used=%.2fGiB torch=%.2f/%.2fGiB",
                self._sampling_generation,
                self.primary_device, _gib(p['free']), _gib(p['used']), _gib(p['alloc']), _gib(p['reserved']),
                self.secondary_device, _gib(s['free']), _gib(s['used']), _gib(s['alloc']), _gib(s['reserved']),
            )
        except Exception as e:
            LOG.warning("H3VM Dev11 pre-flight telemetry unavailable: %r", e)

    def _begin_prefix(self, h, t_emb, mod_segments, rope_freqs, transformer_options):
        self._step += 1
        self._step_started = time.perf_counter()
        self._prefix_started = self._step_started
        self._peak_primary = self._peak_secondary = 0.0
        self._boundary_write_ms = self._boundary_read_ms = self._return_primary_ms = 0.0
        self._tail_output = None
        self._boundary_cpu = None
        self._t_emb = t_emb
        self._segments = mod_segments
        self._rope = rope_freqs

        # This is the central Dev11 experiment. Tail loading starts only after the
        # first exact model call has entered the primary prefix.
        if self.lazy_secondary:
            self._ensure_secondary_feed_started()

        self._prefix_queue = self._make_prefetch(
            self.prefix_blocks, self.primary_device, transformer_options,
            self.prefix_prefetch, "primary-prefix",
        )
        self._sample_memory()
        return h

    def prefix_execute(self, index, h, t_emb, mod_segments, rope_freqs, transformer_options=None):
        import torch
        import comfy.model_management as mm
        import comfy.model_prefetch

        if index == self.first_prefix:
            h = self._begin_prefix(h, t_emb, mod_segments, rope_freqs, transformer_options)
        if getattr(h, "device", None) != self.primary_device:
            # Fixed/front-end normally already leaves h on primary, but preserve
            # exact semantics if Comfy changes that placement.
            host = self.space.transport.move_tensor(h, "cpu", mode="neutral_pageable")
            h = self.space.transport.move_tensor(host, self.primary_device, mode="neutral_pageable")

        block = self.prefix_blocks[index]
        with mm.cuda_device_context(self.primary_device):
            if self._prefix_queue is not None:
                comfy.model_prefetch.prefetch_queue_pop(self._prefix_queue, self.primary_device, block)
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options or {})

        if index == self.last_prefix:
            self._finish_prefetch(self._prefix_queue, self.primary_device, "primary-prefix")
            torch.cuda.synchronize(self.primary_device)
            self._prefix_ms = (time.perf_counter() - self._prefix_started) * 1000.0
            t0 = time.perf_counter()
            self._boundary_cpu = self.space.transport.move_tensor(h, "cpu", mode="neutral_pageable").contiguous()
            self._boundary_write_ms = (time.perf_counter() - t0) * 1000.0
            self._sample_memory()
        return h

    def _run_tail_exact(self, transformer_options):
        import torch
        import comfy.model_management as mm
        import comfy.model_prefetch

        if self._boundary_cpu is None:
            raise RuntimeError("H3VM Dev11 missing current exact boundary")

        secondary_wait_ms = self._wait_secondary_ready()
        self._tail_queue = self._make_prefetch(
            self.tail_blocks, self.secondary_device, transformer_options,
            self.tail_prefetch, "secondary-tail",
        )
        with mm.cuda_device_context(self.secondary_device):
            t0 = time.perf_counter()
            h = self.space.transport.move_tensor(
                self._boundary_cpu, self.secondary_device, mode="neutral_pageable"
            )
            self._boundary_read_ms = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            t_emb = self._t_emb
            seg = self._segments
            rope = self._rope
            # Tiny metadata follows the same host-neutral contract only when needed.
            if getattr(t_emb, "device", None) != self.secondary_device:
                t_emb = self.space.transport.move_tensor(
                    self.space.transport.move_tensor(t_emb, "cpu", mode="neutral_pageable"),
                    self.secondary_device, mode="neutral_pageable")
            def move_tree(v):
                import torch as _torch
                if _torch.is_tensor(v):
                    if v.device == self.secondary_device:
                        return v
                    return self.space.transport.move_tensor(
                        self.space.transport.move_tensor(v, "cpu", mode="neutral_pageable"),
                        self.secondary_device, mode="neutral_pageable")
                if isinstance(v, tuple): return tuple(move_tree(x) for x in v)
                if isinstance(v, list): return [move_tree(x) for x in v]
                if isinstance(v, dict): return {k: move_tree(x) for k, x in v.items()}
                return v
            seg = move_tree(seg)
            rope = move_tree(rope)

            for i in range(self.first_tail, self.last_tail + 1):
                block = self.tail_blocks[i]
                if self._tail_queue is not None:
                    comfy.model_prefetch.prefetch_queue_pop(self._tail_queue, self.secondary_device, block)
                h = block(h, t_emb, seg, rope, transformer_options=transformer_options or {})
            self._finish_prefetch(self._tail_queue, self.secondary_device, "secondary-tail")
            torch.cuda.synchronize(self.secondary_device)
            tail_ms = (time.perf_counter() - t1) * 1000.0

        # Main H3 fixed/final layers live on the primary card. Return the exact
        # tail output through host memory before control reaches final_layer.
        r0 = time.perf_counter()
        host_out = self.space.transport.move_tensor(h, "cpu", mode="neutral_pageable")
        h = self.space.transport.move_tensor(host_out, self.primary_device, mode="neutral_pageable")
        self._return_primary_ms = (time.perf_counter() - r0) * 1000.0
        self._sample_memory()
        return h, tail_ms, secondary_wait_ms

    def tail_execute(self, index, h, t_emb, mod_segments, rope_freqs, transformer_options=None):
        if index == self.first_tail:
            wall0 = time.perf_counter()
            self._tail_output, tail_ms, wait_ms = self._run_tail_exact(transformer_options or {})
            step_wall = (time.perf_counter() - self._step_started) * 1000.0
            if self.telemetry:
                LOG.info(
                    "H3VM Dev11 EXACT step #%d | primary_prefix=%.1fms secondary_tail=%.1fms | "
                    "secondary_feed=%.1fms wait_at_boundary=%.1fms | boundary write/read=%.1f/%.1fms return_primary=%.1fms | "
                    "step_wall=%.1fms payload=%.1fMiB | peak %s=%.2fGiB %s=%.2fGiB",
                    self._step, self._prefix_ms, tail_ms,
                    self._secondary_load_ms, wait_ms,
                    self._boundary_write_ms, self._boundary_read_ms, self._return_primary_ms,
                    step_wall, _tensor_nbytes(self._boundary_cpu)/(1024**2),
                    self.primary_device, self._peak_primary,
                    self.secondary_device, self._peak_secondary,
                )
            h = self._tail_output
        else:
            h = self._tail_output if self._tail_output is not None else h

        if index == self.last_tail:
            self._boundary_cpu = None
            self._tail_output = None
            self._prefix_queue = None
            self._tail_queue = None
            self._t_emb = self._segments = self._rope = None
            if self.expected_steps and self._step >= self.expected_steps:
                self._step = 0
        return h

    def sampling_end(self):
        try:
            if self._secondary_future is not None:
                self._secondary_future.result()
        except Exception as e:
            LOG.warning("H3VM Dev11 secondary feed completion warning: %r", e)
        try:
            import comfy.model_prefetch as mp
            mp.cleanup_prefetch_queues()
        except Exception:
            pass
        self._boundary_cpu = None
        self._tail_output = None
        self._prefix_queue = None
        self._tail_queue = None
        self._t_emb = self._segments = self._rope = None
        self._step = 0
        self._run_active = False

        if self.hard_cleanup_after_sample:
            try:
                import comfy.model_management as mm
                mm.unload_all_models()
                mm.soft_empty_cache()
            except Exception as e:
                LOG.warning("H3VM Dev11 hard cleanup warning: %r", e)

    def close(self):
        try:
            self.sampling_end()
        except Exception:
            pass
        try:
            self._load_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.secondary_patcher = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class PrimaryPrefixProxyFactory:
    @staticmethod
    def make(runtime, index):
        import torch
        import weakref

        class H3PrimaryPrefixProxy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._runtime_ref = weakref.ref(runtime)
                self.index = int(index)

            def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
                rt = self._runtime_ref()
                if rt is None:
                    raise RuntimeError("H3VM Dev11 runtime released")
                return rt.prefix_execute(
                    self.index, x, t_emb, mod_segments, rope_freqs,
                    transformer_options=transformer_options,
                )

        return H3PrimaryPrefixProxy()


class SecondaryTailProxyFactory:
    @staticmethod
    def make(runtime, index):
        import torch
        import weakref

        class H3SecondaryTailProxy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._runtime_ref = weakref.ref(runtime)
                self.index = int(index)

            def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
                rt = self._runtime_ref()
                if rt is None:
                    raise RuntimeError("H3VM Dev11 runtime released")
                return rt.tail_execute(
                    self.index, x, t_emb, mod_segments, rope_freqs,
                    transformer_options=transformer_options,
                )

        return H3SecondaryTailProxy()
