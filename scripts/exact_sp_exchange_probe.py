from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="H3 VRAM Master two-GPU Exact-SP exchange parity probe")
    ap.add_argument("--primary", default="cuda:0")
    ap.add_argument("--secondary", default="cuda:1")
    ap.add_argument("--sequence", type=int, default=101)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--primary-heads", type=int, default=28)
    ap.add_argument("--secondary-heads", type=int, default=28)
    ap.add_argument("--head-dim", type=int, default=16)
    ap.add_argument("--transport", choices=("neutral_pageable", "neutral_pinned", "direct_d2d"), default="neutral_pageable")
    args = ap.parse_args()

    import torch
    from h3vm.sequence_partition import SequencePartition
    from h3vm.global_memory import TransportEngine
    from h3vm.exact_sp_exchange import ExactSPExchangeFabric

    p = torch.device(args.primary)
    s = torch.device(args.secondary)
    part = SequencePartition.balanced(int(args.sequence), parts=2)
    pc, sc = map(int, part.counts)

    allow_pinned = args.transport == "neutral_pinned"
    engine = TransportEngine(
        p, s, mode=args.transport, benchmark_mb=8, benchmark_repeats=1,
        allow_explicit_pinned=allow_pinned, pinned_ring_mb=16, pinned_ring_slots=2,
        host_only=args.transport.startswith("neutral_"),
    )
    fabric = ExactSPExchangeFabric(p, s, transport_engine=engine)

    torch.manual_seed(1234)
    with torch.cuda.device(p):
        hp = torch.randn((pc, int(args.hidden)), device=p, dtype=torch.float32)
        ph = torch.randn((1, int(args.primary_heads), int(args.sequence), int(args.head_dim)), device=p, dtype=torch.float32)
    with torch.cuda.device(s):
        hs = torch.randn((sc, int(args.hidden)), device=s, dtype=torch.float32)
        sh = torch.randn((1, int(args.secondary_heads), int(args.sequence), int(args.head_dim)), device=s, dtype=torch.float32)

    hp_cpu = hp.cpu(); hs_cpu = hs.cpu(); ph_cpu = ph.cpu(); sh_cpu = sh.cpu()

    full_p, full_s = fabric.gather_hidden(hp, hs, part, seq_axis=0)
    expected_hidden = torch.cat((hp_cpu, hs_cpu), dim=0)
    hidden_err_p = float((full_p.cpu() - expected_hidden).abs().max().item())
    hidden_err_s = float((full_s.cpu() - expected_hidden).abs().max().item())

    out_p, out_s = fabric.redistribute_heads(ph, sh, part, seq_axis=2, head_axis=1)
    expected_heads = torch.cat((ph_cpu, sh_cpu), dim=1)
    expected_p = expected_heads[:, :, :pc, :]
    expected_s = expected_heads[:, :, pc:pc + sc, :]
    head_err_p = float((out_p.cpu() - expected_p).abs().max().item())
    head_err_s = float((out_s.cpu() - expected_s).abs().max().item())

    worst = max(hidden_err_p, hidden_err_s, head_err_p, head_err_s)
    print("=== H3 VRAM Master Exact-SP Exchange Probe ===")
    print(f"partition: {part.counts}")
    print(f"transport: {args.transport}")
    print(f"hidden max_abs primary/secondary: {hidden_err_p:.9g} / {hidden_err_s:.9g}")
    print(f"heads  max_abs primary/secondary: {head_err_p:.9g} / {head_err_s:.9g}")
    print(f"transport stats: {fabric.stats.as_dict()}")
    if worst != 0.0:
        raise SystemExit("EXACT-SP EXCHANGE PARITY: FAIL")
    print("EXACT-SP EXCHANGE PARITY: PASS")


if __name__ == "__main__":
    main()
