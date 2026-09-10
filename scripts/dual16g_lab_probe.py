from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_lab_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "h3vm" / "dual16g_lab.py"
    spec = importlib.util.spec_from_file_location("h3vm_dual16g_lab_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Inspect two CUDA GPUs for the H3VM symmetric dual-16G lab path."
    )
    parser.add_argument("--primary", default="cuda:0")
    parser.add_argument("--secondary", default="cuda:1")
    parser.add_argument(
        "--primary-fraction",
        type=float,
        default=None,
        help="Dry-run future manual head split, e.g. 0.55. Does not change runtime settings.",
    )
    args = parser.parse_args()

    mod = load_lab_module()
    report = mod.inspect_visible_pair(args.primary, args.secondary)
    if args.primary_fraction is not None:
        report["manual_h3_heads_preview"] = mod.manual_dual_head_counts(
            56, args.primary_fraction
        )
        report["manual_primary_fraction"] = args.primary_fraction
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
