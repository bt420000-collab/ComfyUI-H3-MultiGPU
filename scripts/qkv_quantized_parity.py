from __future__ import annotations

"""Manual real-model parity check for H3VM quantized QKV head sharding.

Run inside the same Python environment as ComfyUI. This script does not modify a
workflow or start multi-GPU execution. It validates the arithmetic prerequisite
for future Exact-SP by comparing one stock H3 QKV projection against two
quantized head-row shards reconstructed in original [Q|K|V] order.
"""

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _install_paths(comfy_root: str):
    root = Path(comfy_root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(_repo_root()))
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--model", required=True, help="Full path to the H3 diffusion model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--split-head", type=int, default=None,
                        help="Head index where the two shards split; default is half")
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    args = parser.parse_args()

    _install_paths(args.comfy_root)

    import torch
    import comfy.model_management as mm
    import comfy.sd
    from h3vm.quant_shard import shard_qkv_heads

    device = torch.device(args.device)
    patcher = comfy.sd.load_diffusion_model(args.model)
    patcher.model.load_device = device
    patcher.load_device = device
    mm.load_models_gpu([patcher], force_full_load=True)

    dit = patcher.model.diffusion_model
    attention = dit.blocks[0].attn
    heads = int(attention.heads)
    head_dim = int(attention.head_dim)
    split = int(args.split_head if args.split_head is not None else heads // 2)
    if not 0 < split < heads:
        raise ValueError(f"split-head must be within 1..{heads - 1}")

    dtype = getattr(dit, "dtype", torch.float16)
    hidden = torch.randn(
        max(1, int(args.sequence)),
        int(attention.qkv_proj.in_features),
        dtype=dtype,
        device=device,
    )

    with torch.inference_mode():
        expected = attention.qkv_proj(hidden)
        left = shard_qkv_heads(attention.qkv_proj, heads, head_dim, 0, split)
        right = shard_qkv_heads(attention.qkv_proj, heads, head_dim, split, heads)
        left_out = left(hidden)
        right_out = right(hidden)

        left_inner = split * head_dim
        right_inner = (heads - split) * head_dim
        lq, lk, lv = left_out.split(left_inner, dim=-1)
        rq, rk, rv = right_out.split(right_inner, dim=-1)
        actual = torch.cat((lq, rq, lk, rk, lv, rv), dim=-1)

    absolute = (actual.float() - expected.float()).abs()
    relative = absolute / expected.float().abs().clamp_min(1e-6)
    print(
        "H3VM QKV QUANT SHARD PARITY | "
        f"heads={heads} split={split}/{heads - split} seq={hidden.shape[0]} "
        f"dtype={dtype} device={device}"
    )
    print(
        "max_abs={:.6e} mean_abs={:.6e} max_rel={:.6e}".format(
            absolute.max().item(),
            absolute.mean().item(),
            relative.max().item(),
        )
    )
    torch.testing.assert_close(actual, expected, rtol=float(args.rtol), atol=float(args.atol))
    print("PARITY: PASS")


if __name__ == "__main__":
    main()