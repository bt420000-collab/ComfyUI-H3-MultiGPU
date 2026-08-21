from __future__ import annotations

import logging
import time

LOG = logging.getLogger("H3VM")


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


class H3GlobalSpanRouter:
    """H3 block-span router backed by GlobalMemorySpace."""

    def __init__(self, primary_device, secondary_device, block_map, first, last, *, space, secondary_prefetch=True, telemetry=True):
        self.primary_device = primary_device
        self.secondary_device = secondary_device
        self.block_map = block_map
        self.first = int(first)
        self.last = int(last)
        self.space = space
        self.secondary_prefetch = bool(secondary_prefetch)
        self.telemetry = bool(telemetry)
        self._h = None
        self._t_emb = None
        self._segments = None
        self._rope = None
        self._prefetch_queue = None
        self._span_started = None
        self._span_count = 0
        self._last_span_end = None
        self._primary_gap_ms = None
        self._peak_primary = 0.0
        self._peak_secondary = 0.0
        self._prefetch_logged = False
        self._in_stats0 = None

    def _clear(self):
        if self._h is not None:
            self.space.drop(self._h)
        self._h = None
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
                LOG.info("H3VM Dev7 secondary private prefetch | requested=on active=%s blocks=%d", out is not None, len(queue))
                self._prefetch_logged = True
            return out
        except Exception as e:
            LOG.warning("H3VM Dev7 secondary prefetch disabled after setup failure: %r", e)
            self.secondary_prefetch = False
            return None

    def _sample_memory(self):
        if not self.telemetry:
            return
        try:
            self._peak_primary = max(self._peak_primary, _used_gib(self.primary_device))
            self._peak_secondary = max(self._peak_secondary, _used_gib(self.secondary_device))
        except Exception:
            pass

    @staticmethod
    def _stats_snapshot(stats):
        return (int(stats["moves"]), int(stats["bytes"]), float(stats["seconds"]))

    def _begin(self, h, t_emb, mod_segments, rope_freqs, transformer_options):
        now = time.perf_counter()
        self._primary_gap_ms = ((now - self._last_span_end) * 1000.0) if self._last_span_end is not None else None
        self._span_started = now
        self._in_stats0 = self._stats_snapshot(self.space.transport.stats)
        self._h = self.space.adopt(h, name=f"h3.activation.step{self._span_count+1}", ram_backing=self.space.transport.selected_mode == "neutral_pageable")
        h2 = self._h.acquire(self.secondary_device, prefer_neutral=self.space.transport.selected_mode == "neutral_pageable")
        self._t_emb = self.space.readonly_tree(t_emb, self.secondary_device)
        self._segments = self.space.readonly_tree(mod_segments, self.secondary_device)
        self._rope = self.space.readonly_tree(rope_freqs, self.secondary_device)
        self._prefetch_queue = self._make_prefetch(transformer_options)
        self._sample_memory()
        return h2

    def execute(self, index, h, t_emb, mod_segments, rope_freqs, transformer_options=None):
        import comfy.model_management
        import comfy.model_prefetch
        if index == self.first or self._h is None:
            h = self._begin(h, t_emb, mod_segments, rope_freqs, transformer_options)
        elif getattr(h, "device", None) != self.secondary_device:
            self._h.publish(h)
            h = self._h.acquire(self.secondary_device)
        block = self.block_map[index]
        try:
            with comfy.model_management.cuda_device_context(self.secondary_device):
                if self._prefetch_queue is not None:
                    comfy.model_prefetch.prefetch_queue_pop(self._prefetch_queue, self.secondary_device, block)
                h = block(h, self._t_emb, self._segments, self._rope, transformer_options=transformer_options or {})
            if index == self.first or index == self.last or ((index - self.first) % 4 == 0):
                self._sample_memory()
            if index == self.last:
                if self._prefetch_queue is not None:
                    comfy.model_prefetch.prefetch_queue_pop(self._prefetch_queue, self.secondary_device, None)
                self._h.publish(h, device=self.secondary_device)
                h = self._h.acquire(self.primary_device, prefer_neutral=self.space.transport.selected_mode == "neutral_pageable")
                span_end = time.perf_counter()
                span_ms = (span_end - self._span_started) * 1000.0 if self._span_started else 0.0
                self._last_span_end = span_end
                self._sample_memory()
                self._span_count += 1
                s1 = self._stats_snapshot(self.space.transport.stats)
                s0 = self._in_stats0 or (0, 0, 0.0)
                moves = s1[0] - s0[0]
                moved = s1[1] - s0[1]
                transfer_s = max(0.0, s1[2] - s0[2])
                payload_gbps = (moved / 1e9) / max(transfer_s, 1e-9) if moved else 0.0
                duty = None
                if self._primary_gap_ms is not None:
                    duty = span_ms / max(1e-9, span_ms + self._primary_gap_ms) * 100.0
                if self.telemetry and (self._span_count <= 3 or self._span_count % 10 == 0):
                    LOG.info("H3VM Dev7 GLOBAL span #%d | gid=%s mode=%s | blocks=%d-%d | span=%.1fms primary_gap=%s duty=%s | global_moves=%d payload=%.1fMiB transport=%.1fms effective=%.2fGB/s | peak %s=%.2fGiB %s=%.2fGiB | prefetch=%s", self._span_count, self._h.id, self.space.transport.selected_mode, self.first, self.last, span_ms, "%.1fms" % self._primary_gap_ms if self._primary_gap_ms is not None else "n/a", "%.1f%%" % duty if duty is not None else "n/a", moves, moved/(1024**2), transfer_s*1000.0, payload_gbps, self.primary_device, self._peak_primary, self.secondary_device, self._peak_secondary, self._prefetch_queue is not None)
                self._clear()
            return h
        except Exception:
            self._clear()
            raise


class H3GlobalBlockProxyFactory:
    @staticmethod
    def make(router, index):
        import torch
        class H3GlobalBlockProxy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.router = router
                self.index = int(index)
            def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
                return self.router.execute(self.index, x, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        return H3GlobalBlockProxy()
