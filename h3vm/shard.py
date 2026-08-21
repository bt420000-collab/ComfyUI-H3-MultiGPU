from __future__ import annotations
import logging
import time
from typing import Any

LOG = logging.getLogger("H3VM")


def _move_tree(value, device, *, non_blocking=False):
    import torch
    if torch.is_tensor(value):
        if value.device == device:
            return value
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, tuple):
        return tuple(_move_tree(x, device, non_blocking=non_blocking) for x in value)
    if isinstance(value, list):
        return [_move_tree(x, device, non_blocking=non_blocking) for x in value]
    if isinstance(value, dict):
        return {k: _move_tree(v, device, non_blocking=non_blocking) for k, v in value.items()}
    return value


def _used_gib(device):
    import torch
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / (1024 ** 3)


def _tensor_nbytes(value):
    import torch
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_nbytes(x) for x in value)
    if isinstance(value, dict):
        return sum(_tensor_nbytes(x) for x in value.values())
    return 0


class H3RemoteSpanRouter:
    """Routes a contiguous prefix of H3 DiT blocks to the secondary GPU.

    Dev5 additions are deliberately local to the private shard:
    - optional secondary DynamicVRAM prefetch queue;
    - span/transfer/peak-memory telemetry;
    - no global state mutation and no additional model copies.
    """
    def __init__(self, primary_device, secondary_device, block_map: dict[int, Any], first: int, last: int,
                 *, secondary_prefetch=False, telemetry=True):
        self.primary_device = primary_device
        self.secondary_device = secondary_device
        self.block_map = block_map
        self.first = int(first)
        self.last = int(last)
        self.secondary_prefetch = bool(secondary_prefetch)
        self.telemetry = bool(telemetry)
        self._t_emb = None
        self._segments = None
        self._rope = None
        self._prefetch_queue = None
        self._span_started = None
        self._in_ms = 0.0
        self._peak_primary = 0.0
        self._peak_secondary = 0.0
        self._span_count = 0
        self._prefetch_logged = False
        self._last_span_end = None
        self._primary_gap_ms = None
        self._h_in_bytes = 0
        self._h_out_bytes = 0

    def _clear(self):
        self._t_emb = None
        self._segments = None
        self._rope = None
        self._prefetch_queue = None
        self._span_started = None

    def _make_prefetch(self, transformer_options):
        if not self.secondary_prefetch:
            return None
        try:
            import comfy.model_prefetch
            opts = dict(transformer_options or {})
            opts["prefetch_dynamic_vbars"] = True
            queue = [self.block_map[i] for i in range(self.first, self.last + 1)]
            out = comfy.model_prefetch.make_prefetch_queue(queue, self.secondary_device, opts)
            if not self._prefetch_logged:
                LOG.info("H3VM Dev6 secondary private prefetch | requested=on active=%s blocks=%d",
                         out is not None, len(queue))
                self._prefetch_logged = True
            return out
        except Exception as e:
            LOG.warning("H3VM Dev6 secondary prefetch disabled after setup failure: %r", e)
            self.secondary_prefetch = False
            return None

    def _begin(self, h, t_emb, mod_segments, rope_freqs, transformer_options):
        now = time.perf_counter()
        self._primary_gap_ms = ((now - self._last_span_end) * 1000.0) if self._last_span_end is not None else None
        self._span_started = now
        self._h_in_bytes = _tensor_nbytes(h)
        t0 = time.perf_counter()
        h2 = _move_tree(h, self.secondary_device, non_blocking=False)
        self._t_emb = _move_tree(t_emb, self.secondary_device, non_blocking=False)
        self._segments = _move_tree(mod_segments, self.secondary_device, non_blocking=False)
        self._rope = _move_tree(rope_freqs, self.secondary_device, non_blocking=False)
        self._in_ms = (time.perf_counter() - t0) * 1000.0
        self._prefetch_queue = self._make_prefetch(transformer_options)
        if self.telemetry:
            try:
                self._peak_primary = _used_gib(self.primary_device)
                self._peak_secondary = _used_gib(self.secondary_device)
            except Exception:
                pass
        return h2

    def _sample_memory(self):
        if not self.telemetry:
            return
        try:
            self._peak_primary = max(self._peak_primary, _used_gib(self.primary_device))
            self._peak_secondary = max(self._peak_secondary, _used_gib(self.secondary_device))
        except Exception:
            pass

    def execute(self, index: int, h, t_emb, mod_segments, rope_freqs, transformer_options=None):
        import comfy.model_management
        import comfy.model_prefetch

        if index == self.first:
            h = self._begin(h, t_emb, mod_segments, rope_freqs, transformer_options)
        elif self._t_emb is None:
            h = self._begin(h, t_emb, mod_segments, rope_freqs, transformer_options)
        elif getattr(h, "device", None) != self.secondary_device:
            h = _move_tree(h, self.secondary_device, non_blocking=False)

        block = self.block_map[index]
        try:
            with comfy.model_management.cuda_device_context(self.secondary_device):
                if self._prefetch_queue is not None:
                    comfy.model_prefetch.prefetch_queue_pop(
                        self._prefetch_queue, self.secondary_device, block
                    )
                h = block(h, self._t_emb, self._segments, self._rope,
                          transformer_options=transformer_options or {})
            if index == self.first or index == self.last or ((index - self.first) % 4 == 0):
                self._sample_memory()

            if index == self.last:
                if self._prefetch_queue is not None:
                    comfy.model_prefetch.prefetch_queue_pop(
                        self._prefetch_queue, self.secondary_device, None
                    )
                self._h_out_bytes = _tensor_nbytes(h)
                t0 = time.perf_counter()
                h = _move_tree(h, self.primary_device, non_blocking=False)
                out_ms = (time.perf_counter() - t0) * 1000.0
                span_end = time.perf_counter()
                span_ms = (span_end - self._span_started) * 1000.0 if self._span_started else 0.0
                self._last_span_end = span_end
                self._sample_memory()
                self._span_count += 1
                if self.telemetry and (self._span_count <= 3 or self._span_count % 10 == 0):
                    in_gbps = (self._h_in_bytes / 1e9) / max(self._in_ms / 1000.0, 1e-9)
                    out_gbps = (self._h_out_bytes / 1e9) / max(out_ms / 1000.0, 1e-9)
                    duty = None
                    if self._primary_gap_ms is not None:
                        duty = span_ms / max(1e-9, span_ms + self._primary_gap_ms) * 100.0
                    LOG.info(
                        "H3VM Dev6 span #%d | secondary blocks=%d-%d | span=%.1fms primary_gap=%s duty=%s | "
                        "handoff in=%.1fms %.2fGB/s out=%.1fms %.2fGB/s bytes=%.1f/%.1fMiB | "
                        "peak %s=%.2fGiB %s=%.2fGiB | prefetch=%s",
                        self._span_count, self.first, self.last, span_ms,
                        "%.1fms" % self._primary_gap_ms if self._primary_gap_ms is not None else "n/a",
                        "%.1f%%" % duty if duty is not None else "n/a",
                        self._in_ms, in_gbps, out_ms, out_gbps,
                        self._h_in_bytes/(1024**2), self._h_out_bytes/(1024**2),
                        self.primary_device, self._peak_primary, self.secondary_device, self._peak_secondary,
                        self._prefetch_queue is not None,
                    )
                self._clear()
            return h
        except Exception:
            self._clear()
            raise


class H3RemoteBlockProxyFactory:
    @staticmethod
    def make(router: H3RemoteSpanRouter, index: int):
        import torch

        class H3RemoteBlockProxy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.router = router
                self.index = int(index)

            def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
                return self.router.execute(self.index, x, t_emb, mod_segments, rope_freqs,
                                           transformer_options=transformer_options)

        return H3RemoteBlockProxy()


def make_shard_root(source_base_model, source_dm, block_map: dict[int, Any], total_blocks: int):
    import torch

    class EmptySlot(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError("H3VM empty shard slot was executed unexpectedly.")

    class ShardDiffusionView(torch.nn.Module):
        def __init__(self):
            super().__init__()
            slots = []
            for i in range(total_blocks):
                slots.append(block_map[i] if i in block_map else EmptySlot())
            self.blocks = torch.nn.ModuleList(slots)
            for name in ("hidden_size", "sigma_shift_video", "sigma_shift_audio", "use_adaln_curves", "dtype"):
                if hasattr(source_dm, name):
                    setattr(self, name, getattr(source_dm, name))

    class ShardRoot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = ShardDiffusionView()
            for name in ("manual_cast_dtype", "model_dtype", "dtype"):
                if hasattr(source_base_model, name):
                    setattr(self, name, getattr(source_base_model, name))

        def memory_required(self, input_shape=None):
            return 0

    return ShardRoot()
