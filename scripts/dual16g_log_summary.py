from __future__ import annotations

"""Summarize H3VM dual-16G telemetry from a captured ComfyUI console log."""

import argparse
import json
import re
import statistics
from pathlib import Path


RE_POLICY = re.compile(
    r"\[H3VM DUAL16 LAB\].*?mode=(\S+)\s+MLP=(\d+)/(\d+)\s+ATTN=(\d+)/(\d+)"
)
RE_ATTN = re.compile(
    r"H3VM Dev14 CPM attn.*?root=([\d.]+)ms\s+stage=([\d.]+)ms\s+helper=([\d.]+)ms\s+return=([\d.]+)ms"
)
RE_MLP = re.compile(
    r"H3VM Dev14\.1 CPM MLP.*?root=([\d.]+)ms\s+shadow=([\d.]+)ms\s+"
    r"\[stage\s+([\d.]+)\+([\d.]+)\s+compute\s+([\d.]+)\s+return\s+([\d.]+)\+([\d.]+)\].*?"
    r"slack=([+-]?[\d.]+)ms\s+stall=([\d.]+)ms.*?root_fraction\s+([\d.]+)->([\d.]+)\s+action=(\w+)"
)
RE_CAP_ATTN = re.compile(
    r"H3VM CAPACITY ATTN.*?stage_acc=([\d.]+)ms\s+return_acc=([\d.]+)ms\s+"
    r"root_acc=([\d.]+)ms\s+helper_acc=([\d.]+)ms"
)
RE_ERRORS = re.compile(r"\b(OOM|out of memory|fallback|disabled|failed|exception|traceback)\b", re.I)


def _stats(values):
    values = [float(x) for x in values]
    if not values:
        return None
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def summarize(text: str):
    policies = []
    attn = []
    mlp = []
    cap_attn = []
    suspicious = []

    for line_no, line in enumerate(text.splitlines(), 1):
        if m := RE_POLICY.search(line):
            policies.append({
                "mode": m.group(1),
                "mlp": [int(m.group(2)), int(m.group(3))],
                "attention": [int(m.group(4)), int(m.group(5))],
            })
        if m := RE_ATTN.search(line):
            attn.append(tuple(map(float, m.groups())))
        if m := RE_MLP.search(line):
            vals = list(m.groups())
            mlp.append({
                "root": float(vals[0]),
                "shadow": float(vals[1]),
                "stage_d2h": float(vals[2]),
                "stage_h2d": float(vals[3]),
                "helper": float(vals[4]),
                "return_d2h": float(vals[5]),
                "return_h2d": float(vals[6]),
                "slack": float(vals[7]),
                "stall": float(vals[8]),
                "fraction_before": float(vals[9]),
                "fraction_after": float(vals[10]),
                "action": vals[11],
            })
        if m := RE_CAP_ATTN.search(line):
            cap_attn.append(tuple(map(float, m.groups())))
        if RE_ERRORS.search(line):
            suspicious.append({"line": line_no, "text": line.strip()[:500]})

    out = {
        "policy_events": policies,
        "attention": {
            "samples": len(attn),
            "root_ms": _stats([x[0] for x in attn]),
            "stage_ms": _stats([x[1] for x in attn]),
            "helper_ms": _stats([x[2] for x in attn]),
            "return_ms": _stats([x[3] for x in attn]),
        },
        "mlp": {
            "samples": len(mlp),
            "root_ms": _stats([x["root"] for x in mlp]),
            "shadow_ms": _stats([x["shadow"] for x in mlp]),
            "slack_ms": _stats([x["slack"] for x in mlp]),
            "stall_ms": _stats([x["stall"] for x in mlp]),
            "helper_ms": _stats([x["helper"] for x in mlp]),
            "actions": {a: sum(1 for x in mlp if x["action"] == a) for a in sorted({x["action"] for x in mlp})},
            "fraction_after": _stats([x["fraction_after"] for x in mlp]),
        },
        "capacity_attention_accumulated": {
            "samples": len(cap_attn),
            "latest": None if not cap_attn else {
                "stage_ms": cap_attn[-1][0],
                "return_ms": cap_attn[-1][1],
                "root_ms": cap_attn[-1][2],
                "helper_ms": cap_attn[-1][3],
            },
        },
        "suspicious_lines": suspicious[:100],
    }

    if mlp:
        median_slack = statistics.median(x["slack"] for x in mlp)
        median_stall = statistics.median(x["stall"] for x in mlp)
        if median_stall > 0.5 or median_slack < -0.5:
            out["hint"] = "helper path is commonly late; bias more work toward GPU0"
        elif median_slack > 8.0:
            out["hint"] = "helper has measurable slack; test a lower primary fraction"
        else:
            out["hint"] = "MLP split is near the measured relay/compute balance region"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="Captured ComfyUI console log (UTF-8 or Windows text)")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    path = Path(args.log)
    text = path.read_text(encoding="utf-8", errors="replace")
    result = summarize(text)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
