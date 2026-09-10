from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-root", required=True)
    ap.add_argument("--primary", default="cuda:0")
    ap.add_argument("--secondary", default="cuda:1")
    ap.add_argument("--sequence", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=5376)
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--element-size", type=int, default=2)
    ap.add_argument("--host-gbps", type=float, default=None)
    ap.add_argument("--ratio", default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    comfy_root = Path(args.comfy_root).resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(comfy_root))

    from h3vm.compute_planner import probe_cuda_pair, build_runtime_plan
    from h3vm.exact_sp_lab import build_exact_sp_plan

    pair = probe_cuda_pair(args.primary, args.secondary)
    mode4 = build_runtime_plan(pair, backend="MODE4", manual_ratio=args.ratio)
    exact = build_exact_sp_plan(
        pair,
        seq_len=args.sequence,
        hidden_dim=args.hidden,
        heads=args.heads,
        head_dim=args.head_dim,
        element_size=args.element_size,
        manual_ratio=args.ratio,
        host_relay_gbps=args.host_gbps,
    )

    data = {
        "product": "ComfyUI-H3-VRAM-Master",
        "primary": pair.primary.__dict__,
        "secondary": pair.secondary.__dict__,
        "near_symmetric": pair.near_symmetric,
        "p2p": [pair.p2p_ab, pair.p2p_ba],
        "mode4": {
            "summary": mode4.summary(),
            "reason": mode4.reason,
            "primary_blocks": mode4.mode4_primary_blocks,
            "secondary_blocks": mode4.mode4_secondary_blocks,
        },
        "exact_sp": {
            "summary": exact.runtime.summary(),
            "sequence_counts": exact.sequence.counts,
            "head_counts": exact.head_counts,
            "hidden_exchange_mib_per_block": exact.hidden_exchange_bytes_per_block / (1024 ** 2),
            "head_exchange_mib_per_block": exact.head_exchange_bytes_per_block / (1024 ** 2),
            "total_exchange_mib_per_block": exact.total_exchange_bytes_per_block / (1024 ** 2),
            "recommended": exact.recommended,
            "reason": exact.recommendation_reason,
        },
    }
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
