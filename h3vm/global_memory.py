from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger("H3VM")
_GID = itertools.count(1)


def _device_key(device) -> str:
    import torch
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return str(d)


def _tensor_bytes(t) -> int:
    return int(t.numel() * t.element_size())


def _sync(device):
    import torch
    d = torch.device(device)
    if d.type == "cuda":
        torch.cuda.synchronize(d)


def _free_gib(device) -> float:
    import torch
    d = torch.device(device)
    if d.type == "cpu":
        try:
            import psutil
            return float(psutil.virtual_memory().available) / (1024 ** 3)
        except Exception:
            return 0.0
    if d.type != "cuda":
        return 0.0
    free, _ = torch.cuda.mem_get_info(d)
    return float(free) / (1024 ** 3)


@dataclass(frozen=True)
class ComputeBlock:
    """A compute endpoint. Data belongs to GlobalMemorySpace, not this block."""

    name: str
    device: str
    kind: str

    @classmethod
    def cpu(cls):
        return cls("cpu", "cpu", "cpu")

    @classmethod
    def cuda(cls, index: int, name: str | None = None):
        import torch
        d = torch.device("cuda", int(index))
        return cls(name or torch.cuda.get_device_name(d), str(d), "cuda")

    def free_gib(self) -> float:
        return _free_gib(self.device)

    def execute(self, fn, *args, **kwargs):
        import torch
        if self.kind == "cuda":
            with torch.cuda.device(torch.device(self.device)):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)


class PinnedRing:
    """Small explicit byte-ring transfer runway, never a model warehouse.

    One pair of uint8 host buffers is shared by every dtype, so the pinned-memory
    upper bound is exact: `ring_mb * slots`, not multiplied by model dtypes.
    """

    def __init__(self, ring_mb: int = 64, slots: int = 2):
        import torch
        self.ring_bytes = int(ring_mb) * 1024 * 1024
        self.slots = max(1, int(slots))
        self._torch = torch
        self._buffers = [
            torch.empty(self.ring_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
            for _ in range(self.slots)
        ]
        self._cursor = 0

    @property
    def total_pinned_bytes_upper_bound(self) -> int:
        return self.ring_bytes * self.slots

    def _next(self):
        out = self._buffers[self._cursor]
        self._cursor = (self._cursor + 1) % len(self._buffers)
        return out

    def bounce(self, src, dst_device):
        import torch
        if not torch.is_tensor(src):
            raise TypeError("PinnedRing.bounce expects a tensor")
        if not src.is_contiguous():
            src = src.contiguous()
        dst = torch.empty_like(src, device=dst_device)
        # Reinterpret contiguous tensors as raw bytes. This keeps the runway
        # dtype-agnostic and prevents one pinned allocation per model dtype.
        sf = src.view(torch.uint8).view(-1)
        df = dst.view(torch.uint8).view(-1)
        offset = 0
        while offset < sf.numel():
            buf = self._next()
            n = min(self.ring_bytes, sf.numel() - offset)
            view = buf[:n]
            view.copy_(sf[offset:offset+n], non_blocking=False)
            df[offset:offset+n].copy_(view, non_blocking=False)
            offset += n
        _sync(dst_device)
        return dst


class TransportEngine:
    """Physical transport beneath GlobalMemorySpace.

    Logical objects never expose transport policy to compute code. Backends can
    be swapped between direct driver D2D and host-neutral routes.
    """

    MODES = ("auto", "direct_d2d", "neutral_pageable", "neutral_pinned")

    def __init__(self, primary, secondary, *, mode="auto", benchmark_mb=64,
                 benchmark_repeats=3, allow_explicit_pinned=False,
                 pinned_ring_mb=64, pinned_ring_slots=2, host_only=False):
        import torch
        self.primary = torch.device(primary)
        self.secondary = torch.device(secondary)
        self.requested_mode = str(mode)
        if self.requested_mode not in self.MODES:
            raise ValueError(f"Unknown GlobalMemory transport mode: {mode}")
        self.allow_explicit_pinned = bool(allow_explicit_pinned)
        self.pinned_ring_mb = int(pinned_ring_mb)
        self.host_only = bool(host_only)
        if self.host_only and self.requested_mode not in ("neutral_pageable", "neutral_pinned"):
            raise RuntimeError("GlobalMemory host_only mode requires a host-neutral transport")
        self.ring = None
        self.benchmark = self.probe(
            size_mb=int(benchmark_mb), repeats=int(benchmark_repeats),
            test_pinned=self.allow_explicit_pinned,
        )
        self.selected_mode = self._select_mode()
        if self.selected_mode == "neutral_pinned":
            if not self.allow_explicit_pinned:
                raise RuntimeError("neutral_pinned requires allow_explicit_pinned=True")
            self.ring = PinnedRing(ring_mb=self.pinned_ring_mb, slots=pinned_ring_slots)
        self.stats = {
            "moves": 0,
            "bytes": 0,
            "seconds": 0.0,
            "by_mode": {},
        }

    def _select_mode(self):
        if self.requested_mode != "auto":
            if self.requested_mode == "neutral_pinned" and not self.allow_explicit_pinned:
                raise RuntimeError("neutral_pinned selected but explicit pinned staging is disabled")
            return self.requested_mode
        candidates = []
        for mode in ("direct_d2d", "neutral_pageable", "neutral_pinned"):
            if mode == "neutral_pinned" and not self.allow_explicit_pinned:
                continue
            vals = []
            for direction in ("ab", "ba"):
                v = self.benchmark.get(mode, {}).get(direction)
                if v is not None:
                    vals.append(float(v))
            if len(vals) == 2:
                candidates.append((min(vals), mode))
        if not candidates:
            return "direct_d2d"
        candidates.sort(reverse=True)
        return candidates[0][1]

    @staticmethod
    def _alloc_pair(src_device, dst_device, nbytes):
        import torch
        with torch.cuda.device(src_device):
            src = torch.empty(int(nbytes), dtype=torch.uint8, device=src_device)
            src.fill_(0x3C)
        with torch.cuda.device(dst_device):
            dst = torch.empty(int(nbytes), dtype=torch.uint8, device=dst_device)
        return src, dst

    @staticmethod
    def _time_payload(fn, src_device, dst_device, payload_bytes, repeats):
        _sync(src_device)
        _sync(dst_device)
        fn()  # warm-up
        _sync(src_device)
        _sync(dst_device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        _sync(src_device)
        _sync(dst_device)
        dt = max(1e-9, time.perf_counter() - t0)
        return (payload_bytes * repeats) / dt / 1e9

    def _probe_one(self, src_device, dst_device, nbytes, repeats, test_pinned):
        import torch
        src, dst = self._alloc_pair(src_device, dst_device, nbytes)
        out = {"direct_d2d": None, "neutral_pageable": None, "neutral_pinned": None,
               "errors": {}}
        if not self.host_only:
            try:
                out["direct_d2d"] = self._time_payload(
                    lambda: dst.copy_(src, non_blocking=False), src_device, dst_device, nbytes, repeats
                )
            except Exception as e:
                out["errors"]["direct_d2d"] = repr(e)

        try:
            host = torch.empty(nbytes, dtype=torch.uint8, device="cpu")
            def pageable():
                host.copy_(src, non_blocking=False)
                dst.copy_(host, non_blocking=False)
            out["neutral_pageable"] = self._time_payload(
                pageable, src_device, dst_device, nbytes, repeats
            )
            del host
        except Exception as e:
            out["errors"]["neutral_pageable"] = repr(e)

        if test_pinned:
            try:
                hostp = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
                def pinned():
                    hostp.copy_(src, non_blocking=False)
                    dst.copy_(hostp, non_blocking=False)
                out["neutral_pinned"] = self._time_payload(
                    pinned, src_device, dst_device, nbytes, repeats
                )
                del hostp
            except Exception as e:
                out["errors"]["neutral_pinned"] = repr(e)

        del src, dst
        for d in (src_device, dst_device):
            try:
                _sync(d)
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return out

    def probe(self, size_mb=64, repeats=3, test_pinned=False):
        import torch
        nbytes = max(1, int(size_mb)) * 1024 * 1024
        a, b = self.primary, self.secondary
        fn = getattr(torch.cuda, "can_device_access_peer", None)
        peer_ab = peer_ba = False
        if fn is not None:
            try:
                peer_ab = bool(fn(a.index, b.index))
                peer_ba = bool(fn(b.index, a.index))
            except Exception:
                pass
        ab = self._probe_one(a, b, nbytes, max(1, repeats), test_pinned)
        ba = self._probe_one(b, a, nbytes, max(1, repeats), test_pinned)
        result = {
            "size_mb": int(size_mb),
            "repeats": int(repeats),
            "peer_ab": peer_ab,
            "peer_ba": peer_ba,
            "direct_d2d": {"ab": ab["direct_d2d"], "ba": ba["direct_d2d"]},
            "neutral_pageable": {"ab": ab["neutral_pageable"], "ba": ba["neutral_pageable"]},
            "neutral_pinned": {"ab": ab["neutral_pinned"], "ba": ba["neutral_pinned"]},
            "errors": {"ab": ab["errors"], "ba": ba["errors"]},
        }
        return result

    def _record(self, mode, nbytes, dt):
        self.stats["moves"] += 1
        self.stats["bytes"] += int(nbytes)
        self.stats["seconds"] += float(dt)
        d = self.stats["by_mode"].setdefault(mode, {"moves": 0, "bytes": 0, "seconds": 0.0})
        d["moves"] += 1
        d["bytes"] += int(nbytes)
        d["seconds"] += float(dt)

    def move_tensor(self, tensor, dst_device, *, mode=None):
        import torch
        if not torch.is_tensor(tensor):
            return tensor
        dst = torch.device(dst_device)
        if tensor.device == dst:
            return tensor
        mode = mode or self.selected_mode
        nbytes = _tensor_bytes(tensor)
        t0 = time.perf_counter()
        if dst.type == "cpu":
            out = tensor.to("cpu")
        elif tensor.device.type == "cpu":
            out = tensor.to(dst, non_blocking=False)
            _sync(dst)
        elif mode == "direct_d2d":
            out = tensor.to(dst, non_blocking=False)
            _sync(dst)
        elif mode == "neutral_pageable":
            host = tensor.to("cpu")
            out = host.to(dst, non_blocking=False)
            _sync(dst)
        elif mode == "neutral_pinned":
            if self.ring is None:
                raise RuntimeError("neutral_pinned requested without an initialized pinned ring")
            out = self.ring.bounce(tensor, dst)
        else:
            raise RuntimeError(f"Unsupported GlobalMemory transport mode: {mode}")
        dt = time.perf_counter() - t0
        self._record(mode, nbytes, dt)
        return out

    def report(self):
        return {
            "requested_mode": self.requested_mode,
            "selected_mode": self.selected_mode,
            "allow_explicit_pinned": self.allow_explicit_pinned,
            "pinned_ring_mb": self.pinned_ring_mb,
            "host_only": self.host_only,
            "benchmark": self.benchmark,
            "runtime_stats": self.stats,
        }


class GlobalTensor:
    """Device-agnostic logical tensor with versioned physical replicas.

    This is the core inversion: compute devices are caches/accelerators. The
    logical object owns identity and freshness. A RAM backing is created only
    when the selected transport policy or explicit checkpointing needs it.
    """

    def __init__(self, space: "GlobalMemorySpace", tensor, *, name=None, ram_backing=False):
        import torch
        if not torch.is_tensor(tensor):
            raise TypeError("GlobalTensor requires a torch.Tensor")
        self.space = space
        self.id = next(_GID)
        self.name = name or f"tensor-{self.id}"
        self.version = 1
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.replicas: dict[str, tuple[int, Any]] = {_device_key(tensor.device): (self.version, tensor)}
        self.ram_backing = None
        self.ram_version = 0
        if ram_backing:
            self.checkpoint_ram()

    def publish(self, tensor, *, device=None):
        key = _device_key(device if device is not None else tensor.device)
        self.version += 1
        self.replicas = {key: (self.version, tensor)}
        self.ram_backing = None
        self.ram_version = 0
        return self

    def _fresh_replica(self):
        for key, (ver, tensor) in self.replicas.items():
            if ver == self.version:
                return key, tensor
        return None, None

    def checkpoint_ram(self):
        import torch
        if self.ram_version == self.version and self.ram_backing is not None:
            return self.ram_backing
        _, src = self._fresh_replica()
        if src is None:
            raise RuntimeError(f"GlobalTensor {self.name} has no fresh physical replica")
        self.ram_backing = self.space.transport.move_tensor(src, "cpu") if src.device.type != "cpu" else src
        self.ram_version = self.version
        self.replicas["cpu"] = (self.version, self.ram_backing)
        return self.ram_backing

    def acquire(self, device, *, prefer_neutral=False):
        import torch
        dst = torch.device(device)
        key = _device_key(dst)
        got = self.replicas.get(key)
        if got is not None and got[0] == self.version:
            return got[1]

        if prefer_neutral or self.space.transport.selected_mode.startswith("neutral_"):
            src = self.checkpoint_ram()
        else:
            _, src = self._fresh_replica()
            if src is None:
                src = self.checkpoint_ram()
        out = self.space.transport.move_tensor(src, dst)
        self.replicas[key] = (self.version, out)
        return out

    def release_replica(self, device):
        key = _device_key(device)
        self.replicas.pop(key, None)

    def describe(self):
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "shape": self.shape,
            "dtype": str(self.dtype),
            "replicas": {k: v[0] for k, v in self.replicas.items()},
            "ram_backed": self.ram_backing is not None and self.ram_version == self.version,
        }


class GlobalMemorySpace:
    """Global logical memory with CPU/GPU compute blocks attached as accelerators."""

    def __init__(self, primary, secondary, *, transport_mode="auto", benchmark_mb=64,
                 benchmark_repeats=3, allow_explicit_pinned=False,
                 pinned_ring_mb=64, pinned_ring_slots=2, host_only=False):
        import torch
        self.primary = torch.device(primary)
        self.secondary = torch.device(secondary)
        self.transport = TransportEngine(
            self.primary, self.secondary,
            mode=transport_mode,
            benchmark_mb=benchmark_mb,
            benchmark_repeats=benchmark_repeats,
            allow_explicit_pinned=allow_explicit_pinned,
            pinned_ring_mb=pinned_ring_mb,
            pinned_ring_slots=pinned_ring_slots,
            host_only=host_only,
        )
        self.compute_blocks = [
            ComputeBlock.cpu(),
            ComputeBlock.cuda(self.primary.index, "primary"),
            ComputeBlock.cuda(self.secondary.index, "secondary"),
        ]
        self.live: dict[int, GlobalTensor] = {}

    def adopt(self, tensor, *, name=None, ram_backing=False):
        gt = GlobalTensor(self, tensor, name=name, ram_backing=ram_backing)
        self.live[gt.id] = gt
        return gt

    def drop(self, gt: GlobalTensor):
        self.live.pop(gt.id, None)

    def readonly_tree(self, value, device):
        import torch
        if torch.is_tensor(value):
            gt = self.adopt(value, name="readonly", ram_backing=False)
            try:
                return gt.acquire(device)
            finally:
                self.drop(gt)
        if isinstance(value, tuple):
            return tuple(self.readonly_tree(x, device) for x in value)
        if isinstance(value, list):
            return [self.readonly_tree(x, device) for x in value]
        if isinstance(value, dict):
            return {k: self.readonly_tree(v, device) for k, v in value.items()}
        return value

    def report(self):
        return {
            "architecture": "GLOBAL_MEMORY",
            "compute_blocks": [
                {"name": b.name, "device": b.device, "kind": b.kind, "free_gib": b.free_gib()}
                for b in self.compute_blocks
            ],
            "live_global_tensors": len(self.live),
            "transport": self.transport.report(),
        }

    def report_json(self):
        return json.dumps(self.report(), ensure_ascii=False, indent=2, default=str)
