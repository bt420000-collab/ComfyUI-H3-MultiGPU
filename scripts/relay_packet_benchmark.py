from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bytes(values):
    return sum(int(x.numel() * x.element_size()) for x in values)


def main():
    ap = argparse.ArgumentParser(description="Compare six host moves vs one H3VM relay packet move")
    ap.add_argument("--primary", default="cuda:0")
    ap.add_argument("--secondary", default="cuda:1")
    ap.add_argument("--sequence", type=int, default=8192)
    ap.add_argument("--heads", type=int, default=28)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--transport", choices=("neutral_pageable", "neutral_pinned"), default="neutral_pageable")
    args = ap.parse_args()

    import torch
    from h3vm.global_memory import TransportEngine
    from h3vm.relay_packet import move_tensor_group

    p = torch.device(args.primary)
    s = torch.device(args.secondary)
    seq = int(args.sequence)
    heads = int(args.heads)
    hd = int(args.head_dim)

    # Representative prequantized attention shard payload.  Exact comfy-kitchen
    # layouts can vary by build; the benchmark is about host transaction/setup
    # overhead and packet transport, not attention kernel arithmetic.
    with torch.cuda.device(p):
        values = (
            torch.randint(-127, 127, (1, heads, seq, hd), dtype=torch.int8, device=p),
            torch.randint(-127, 127, (1, heads, seq, hd), dtype=torch.int8, device=p),
            torch.randint(-127, 127, (heads * hd, seq), dtype=torch.int8, device=p),
            torch.ones((1, heads, seq), dtype=torch.float32, device=p),
            torch.ones((1, heads, seq), dtype=torch.float32, device=p),
            torch.ones((heads * hd,), dtype=torch.float32, device=p),
        )

    allow_pinned = args.transport == "neutral_pinned"
    engine = TransportEngine(
        p, s, mode=args.transport, benchmark_mb=8, benchmark_repeats=1,
        allow_explicit_pinned=allow_pinned, pinned_ring_mb=16, pinned_ring_slots=2,
        host_only=True,
    )

    def legacy_once():
        return tuple(engine.move_tensor(v, s, mode=args.transport) for v in values)

    def packet_once():
        moved, info = move_tensor_group(engine, values, s, mode=args.transport)
        return moved, info

    legacy_once(); packet_once()
    torch.cuda.synchronize(p); torch.cuda.synchronize(s)

    t0 = time.perf_counter()
    for _ in range(max(1, args.repeats)):
        legacy = legacy_once()
    torch.cuda.synchronize(s)
    legacy_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(max(1, args.repeats)):
        packet, info = packet_once()
    torch.cuda.synchronize(s)
    packet_s = time.perf_counter() - t0

    # One exact byte-level verification is enough for the transport codec.
    max_bad = 0
    for src, dst in zip(values, packet):
        bad = int((src.cpu() != dst.cpu()).sum().item())
        max_bad = max(max_bad, bad)

    payload_mib = _bytes(values) / (1024 ** 2)
    print("=== H3 VRAM Master Relay Packet Benchmark ===")
    print(f"transport: {args.transport}")
    print(f"payload:   {payload_mib:.2f} MiB in {len(values)} tensors")
    print(f"legacy:    {legacy_s / max(1,args.repeats) * 1000:.2f} ms/iter (6 moves)")
    print(f"packet:    {packet_s / max(1,args.repeats) * 1000:.2f} ms/iter (1 move)")
    print(f"speedup:   {legacy_s / max(packet_s,1e-9):.3f}x")
    print(f"packet bytes: {info.packet_bytes} padding={info.padding_bytes}")
    print(f"byte mismatches: {max_bad}")
    if max_bad:
        raise SystemExit("RELAY PACKET PARITY: FAIL")
    print("RELAY PACKET PARITY: PASS")


if __name__ == "__main__":
    main()
