from __future__ import annotations

import json
import logging

LOG = logging.getLogger("H3VM")


def resolve_device(option):
    import torch
    import comfy.model_management
    try:
        d = comfy.model_management.resolve_gpu_device_option(option)
    except Exception:
        d = None
    if d is None and isinstance(option, str) and option.startswith("gpu:"):
        d = torch.device("cuda:" + option.split(":", 1)[1])
    if d is None:
        raise RuntimeError(f"H3VM Global Memory cannot resolve device {option!r}")
    return torch.device(d)


def run_global_memory_probe(primary_device, secondary_device, *, benchmark_mb=64,
                            repeats=3, allow_explicit_pinned=False, pinned_ring_mb=64):
    import torch
    from .global_memory import GlobalMemorySpace

    a = resolve_device(primary_device)
    b = resolve_device(secondary_device)
    if a == b or a.type != "cuda" or b.type != "cuda":
        raise RuntimeError(f"Global Memory Probe needs two different CUDA GPUs, got {a} and {b}")

    space = GlobalMemorySpace(
        a, b,
        transport_mode="auto",
        benchmark_mb=int(benchmark_mb),
        benchmark_repeats=int(repeats),
        allow_explicit_pinned=bool(allow_explicit_pinned),
        pinned_ring_mb=int(pinned_ring_mb),
    )

    # Prove the logical-object API itself with a tiny round trip. GPU identity is
    # not part of the GlobalTensor contract: a compute block acquires a replica.
    with torch.cuda.device(a):
        x = torch.arange(1024, dtype=torch.float32, device=a)
    gt = space.adopt(x, name="probe.logical.tensor", ram_backing=True)
    y = gt.acquire(b, prefer_neutral=True)
    z = gt.acquire(a)
    torch.cuda.synchronize(a)
    torch.cuda.synchronize(b)
    checksum = float(y[:16].float().sum().cpu())
    logical = gt.describe()
    space.drop(gt)
    del x, y, z

    report = space.report()
    report["logical_tensor_probe"] = {
        "ok": abs(checksum - 120.0) < 1e-5,
        "checksum_first16": checksum,
        "object": logical,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    LOG.info("H3VM Dev7 GLOBAL MEMORY PROBE\n%s", text)
    print("[H3VM DEV7 GLOBAL MEMORY PROBE]\n" + text, flush=True)
    return text
