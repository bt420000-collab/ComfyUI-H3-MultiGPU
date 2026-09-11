from __future__ import annotations

"""Bounded double-buffer host relay for H3 VRAM Master.

This overlay upgrades TransportEngine's explicit ``neutral_pinned`` GPU->GPU
copy path.  It reuses the existing PinnedRing buffers and pipelines chunk i H2D
with chunk i+1 D2H.  Cross-device CUDA event waits are intentionally avoided:
the CPU synchronizes only the source D2H event before handing a host slot to the
destination stream.

The path is opt-out and fail-soft.  Any runtime problem falls back to the legacy
PinnedRing bounce for the current transfer.
"""

import logging
import os
import time

LOG = logging.getLogger("H3VM")
_PATCH_MARKER = "_h3vm_segmented_relay_v1"
_WARNED = set()
_LOGGED = set()


def _threshold_bytes() -> int:
    raw = os.environ.get("H3VM_SEGMENTED_RELAY_MIN_MIB", "8")
    try:
        return max(0, int(float(raw) * 1024 * 1024))
    except Exception:
        return 8 * 1024 * 1024


def pipeline_chunk_plan(total_bytes: int, chunk_bytes: int) -> tuple[int, ...]:
    total = max(0, int(total_bytes))
    chunk = max(1, int(chunk_bytes))
    if total == 0:
        return ()
    out = []
    offset = 0
    while offset < total:
        n = min(chunk, total - offset)
        out.append(int(n))
        offset += n
    return tuple(out)


class SegmentedPinnedRelay:
    """Two-stream, N-slot relay using buffers already owned by PinnedRing."""

    def __init__(self, ring):
        import torch

        if ring is None or not getattr(ring, "_buffers", None):
            raise ValueError("SegmentedPinnedRelay requires an initialized PinnedRing")
        if len(ring._buffers) < 2:
            raise ValueError("SegmentedPinnedRelay requires at least two pinned slots")
        self.ring = ring
        self.torch = torch
        self._streams = {}

    def _stream_pair(self, src, dst):
        torch = self.torch
        key = (str(src), str(dst))
        pair = self._streams.get(key)
        if pair is None:
            pair = (torch.cuda.Stream(device=src), torch.cuda.Stream(device=dst))
            self._streams[key] = pair
        return pair

    def bounce(self, src, dst_device):
        torch = self.torch
        if not torch.is_tensor(src):
            raise TypeError("SegmentedPinnedRelay.bounce expects a tensor")
        if src.device.type != "cuda":
            raise ValueError("SegmentedPinnedRelay source must be CUDA")
        dst_device = torch.device(dst_device)
        if dst_device.type != "cuda":
            raise ValueError("SegmentedPinnedRelay destination must be CUDA")
        if src.device == dst_device:
            return src

        src = src if src.is_contiguous() else src.contiguous()
        dst = torch.empty_like(src, device=dst_device)
        sf = src.view(torch.uint8).reshape(-1)
        df = dst.view(torch.uint8).reshape(-1)
        if sf.numel() == 0:
            return dst

        d2h_stream, h2d_stream = self._stream_pair(src.device, dst_device)
        producer = torch.cuda.current_stream(src.device)
        d2h_stream.wait_stream(producer)

        buffers = self.ring._buffers
        chunk_bytes = int(self.ring.ring_bytes)
        slot_done = [None] * len(buffers)
        offset = 0
        chunk_index = 0

        while offset < sf.numel():
            slot = chunk_index % len(buffers)
            previous = slot_done[slot]
            if previous is not None:
                previous.synchronize()

            n = min(chunk_bytes, int(sf.numel()) - offset)
            host = buffers[slot][:n]

            d2h_done = torch.cuda.Event()
            with torch.cuda.device(src.device), torch.cuda.stream(d2h_stream):
                host.copy_(sf[offset:offset + n], non_blocking=True)
                d2h_done.record(d2h_stream)

            # No cross-device stream wait here.  A short CPU event wait makes the
            # host slot ownership explicit and works on native Windows/WDDM.
            d2h_done.synchronize()

            h2d_done = torch.cuda.Event()
            with torch.cuda.device(dst_device), torch.cuda.stream(h2d_stream):
                df[offset:offset + n].copy_(host, non_blocking=True)
                h2d_done.record(h2d_stream)
            slot_done[slot] = h2d_done

            offset += n
            chunk_index += 1

        for done in slot_done:
            if done is not None:
                done.synchronize()
        return dst


def install_segmented_relay_patch() -> bool:
    if os.environ.get("H3VM_DISABLE_SEGMENTED_RELAY", "0") == "1":
        return False

    from .global_memory import TransportEngine
    current = TransportEngine.move_tensor
    if getattr(current, _PATCH_MARKER, False):
        return True
    original = current

    def move_tensor(self, tensor, dst_device, *, mode=None):
        import torch

        selected = mode or self.selected_mode
        dst = torch.device(dst_device)
        eligible = (
            selected == "neutral_pinned"
            and torch.is_tensor(tensor)
            and tensor.device.type == "cuda"
            and dst.type == "cuda"
            and tensor.device != dst
            and getattr(self, "ring", None) is not None
            and len(getattr(self.ring, "_buffers", ())) >= 2
            and int(tensor.numel() * tensor.element_size()) >= _threshold_bytes()
        )
        if not eligible:
            return original(self, tensor, dst_device, mode=mode)

        try:
            relay = getattr(self, "_h3vm_segmented_relay", None)
            if relay is None:
                relay = SegmentedPinnedRelay(self.ring)
                self._h3vm_segmented_relay = relay
            nbytes = int(tensor.numel() * tensor.element_size())
            t0 = time.perf_counter()
            out = relay.bounce(tensor, dst)
            dt = time.perf_counter() - t0
            self._record("neutral_pinned", nbytes, dt)
            self.stats["segmented_moves"] = int(self.stats.get("segmented_moves", 0)) + 1
            self.stats["segmented_bytes"] = int(self.stats.get("segmented_bytes", 0)) + nbytes
            key = id(self)
            if key not in _LOGGED:
                _LOGGED.add(key)
                LOG.info(
                    "H3VM VRAM MASTER SEGMENTED RELAY ACTIVE | chunk=%dMiB slots=%d threshold=%dMiB",
                    int(self.ring.ring_bytes // (1024 ** 2)), len(self.ring._buffers),
                    int(_threshold_bytes() // (1024 ** 2)),
                )
            return out
        except Exception as exc:
            key = id(self)
            if key not in _WARNED:
                _WARNED.add(key)
                LOG.warning("H3VM segmented relay fallback to legacy pinned bounce | %r", exc)
            return original(self, tensor, dst_device, mode=mode)

    setattr(move_tensor, _PATCH_MARKER, True)
    TransportEngine._h3vm_original_move_tensor = original
    TransportEngine.move_tensor = move_tensor
    LOG.info("H3VM VRAM Master segmented relay overlay installed | bounded double buffer")
    return True
