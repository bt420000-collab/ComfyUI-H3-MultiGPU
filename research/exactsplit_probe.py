from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path


def _sync(torch, *devices):
    for device in devices:
        torch.cuda.synchronize(device)


def _device_info(torch, device):
    p = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "device": str(device),
        "name": p.name,
        "sm_count": int(p.multi_processor_count),
        "total_gib": total / (1024 ** 3),
        "free_gib": free / (1024 ** 3),
    }


def _bench_dual_gemm(torch, d0, d1, *, size=4096, repeats=5, dtype_name="float16"):
    dtype = getattr(torch, dtype_name)

    def alloc(device):
        with torch.cuda.device(device):
            a = torch.randn((size, size), device=device, dtype=dtype)
            b = torch.randn((size, size), device=device, dtype=dtype)
        return a, b

    a0, b0 = alloc(d0)
    a1, b1 = alloc(d1)

    # Warm both devices independently.
    for device, a, b in ((d0, a0, b0), (d1, a1, b1)):
        with torch.cuda.device(device):
            _ = a @ b
        torch.cuda.synchronize(device)

    serial = []
    concurrent = []
    keep = None
    for _ in range(max(1, repeats)):
        _sync(torch, d0, d1)
        t0 = time.perf_counter()
        with torch.cuda.device(d0):
            y0 = a0 @ b0
        torch.cuda.synchronize(d0)
        with torch.cuda.device(d1):
            y1 = a1 @ b1
        torch.cuda.synchronize(d1)
        serial.append((time.perf_counter() - t0) * 1000.0)

        _sync(torch, d0, d1)
        t0 = time.perf_counter()
        # CUDA launches are asynchronous. Launch one workload on each device,
        # then synchronize both. If the driver permits real dual-device overlap,
        # wall time approaches max(single0, single1), not their sum.
        with torch.cuda.device(d0):
            z0 = a0 @ b0
        with torch.cuda.device(d1):
            z1 = a1 @ b1
        _sync(torch, d0, d1)
        concurrent.append((time.perf_counter() - t0) * 1000.0)
        keep = (y0, y1, z0, z1)

    del keep, a0, b0, a1, b1
    for d in (d0, d1):
        with torch.cuda.device(d):
            torch.cuda.empty_cache()

    s = sum(serial) / len(serial)
    c = sum(concurrent) / len(concurrent)
    return {
        "matrix": int(size),
        "dtype": dtype_name,
        "repeats": int(repeats),
        "serial_ms": s,
        "concurrent_ms": c,
        "overlap_speedup": s / max(c, 1e-9),
        "concurrency_efficiency_pct": 100.0 * (s / max(c, 1e-9)) / 2.0,
    }


def _payload_projection(report, *, tokens, hidden, dtype_bytes):
    payload_bytes = int(tokens) * int(hidden) * int(dtype_bytes)
    payload_gb = payload_bytes / 1e9
    modes = {}
    for size_key, result in report["transport"].items():
        for mode in ("direct_d2d", "neutral_pageable", "neutral_pinned"):
            row = result.get(mode, {}) or {}
            vals = [row.get("ab"), row.get("ba")]
            vals = [float(v) for v in vals if v is not None and float(v) > 0]
            if len(vals) != 2:
                continue
            # Conservative bidirectional floor. ExactSplit must survive the slow
            # direction because ownership boundaries can alternate.
            floor_gbps = min(vals)
            estimated_one_way_ms = 1000.0 * payload_gb / floor_gbps
            old = modes.get(mode)
            candidate = {
                "floor_gbps": floor_gbps,
                "estimated_one_way_ms": estimated_one_way_ms,
                "source_probe_size_mb": int(size_key),
            }
            if old is None or int(size_key) > old["source_probe_size_mb"]:
                modes[mode] = candidate
    return {
        "tokens": int(tokens),
        "hidden": int(hidden),
        "dtype_bytes": int(dtype_bytes),
        "payload_mib": payload_bytes / (1024 ** 2),
        "modes": modes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="H3VM ExactSplit R0: Windows/no-P2P transport + dual-compute probe"
    )
    parser.add_argument("--primary", type=int, default=0)
    parser.add_argument("--secondary", type=int, default=1)
    parser.add_argument("--sizes-mb", default="8,16,32,64,128,256")
    parser.add_argument("--transfer-repeats", type=int, default=3)
    parser.add_argument("--gemm", type=int, default=4096)
    parser.add_argument("--gemm-repeats", type=int, default=5)
    parser.add_argument("--gemm-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--tokens", type=int, default=16384,
                        help="Hypothetical activation sequence length used only for payload projection")
    parser.add_argument("--hidden", type=int, default=3072,
                        help="Hypothetical hidden width used only for payload projection")
    parser.add_argument("--dtype-bytes", type=int, default=2,
                        help="Activation bytes/element used only for payload projection")
    parser.add_argument("--output", default="exactsplit_probe_report.json")
    args = parser.parse_args()

    import torch
    from h3vm.global_memory import TransportEngine

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("ExactSplit probe requires at least two CUDA GPUs")

    d0 = torch.device("cuda", args.primary)
    d1 = torch.device("cuda", args.secondary)
    sizes = sorted({max(1, int(x.strip())) for x in args.sizes_mb.split(",") if x.strip()})

    peer_fn = getattr(torch.cuda, "can_device_access_peer", None)
    try:
        peer_ab = bool(peer_fn(d0.index, d1.index)) if peer_fn else False
        peer_ba = bool(peer_fn(d1.index, d0.index)) if peer_fn else False
    except Exception:
        peer_ab = peer_ba = False

    report = {
        "schema": "h3vm-exactsplit-r0-probe-v1",
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "peer_ab": peer_ab,
            "peer_ba": peer_ba,
        },
        "devices": [_device_info(torch, d0), _device_info(torch, d1)],
        "transport": {},
    }

    # Reuse the production GlobalMemory probe so R0 measures the exact transport
    # primitives H3VM already knows how to execute on Windows/WDDM.
    engine = TransportEngine(
        d0,
        d1,
        mode="auto",
        benchmark_mb=min(32, max(sizes)),
        benchmark_repeats=max(1, args.transfer_repeats),
        allow_explicit_pinned=True,
        pinned_ring_mb=64,
        pinned_ring_slots=2,
        host_only=False,
    )
    report["transport_selected_mode"] = engine.selected_mode
    report["transport_boot_probe"] = engine.benchmark

    for size_mb in sizes:
        print(f"[ExactSplit R0] probing {size_mb} MiB transport...")
        report["transport"][str(size_mb)] = engine.probe(
            size_mb=size_mb,
            repeats=max(1, args.transfer_repeats),
            test_pinned=True,
        )

    print(f"[ExactSplit R0] probing dual GEMM concurrency {args.gemm}x{args.gemm}...")
    report["dual_compute"] = _bench_dual_gemm(
        torch,
        d0,
        d1,
        size=max(512, args.gemm),
        repeats=max(1, args.gemm_repeats),
        dtype_name=args.gemm_dtype,
    )

    report["activation_payload_projection"] = _payload_projection(
        report,
        tokens=args.tokens,
        hidden=args.hidden,
        dtype_bytes=args.dtype_bytes,
    )

    # First hard gate. This is deliberately conservative: if independent compute
    # cannot overlap by at least ~1.5x, no clever ExactSplit layout can rescue the
    # design on this runtime.
    speedup = report["dual_compute"]["overlap_speedup"]
    report["gate_dual_compute"] = {
        "pass": bool(speedup >= 1.50),
        "threshold": 1.50,
        "measured": speedup,
    }

    best_floor = 0.0
    best_mode = None
    for mode, row in report["activation_payload_projection"]["modes"].items():
        if row["floor_gbps"] > best_floor:
            best_floor = row["floor_gbps"]
            best_mode = mode
    report["best_transport"] = {
        "mode": best_mode,
        "bidirectional_floor_gbps": best_floor,
    }

    out = Path(args.output)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== H3VM ExactSplit R0 summary ===")
    print(json.dumps({
        "devices": [d["name"] for d in report["devices"]],
        "peer": [peer_ab, peer_ba],
        "selected_transport": report["transport_selected_mode"],
        "best_transport": report["best_transport"],
        "dual_compute": report["dual_compute"],
        "payload_projection": report["activation_payload_projection"],
        "gate_dual_compute": report["gate_dual_compute"],
        "report": str(out.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
