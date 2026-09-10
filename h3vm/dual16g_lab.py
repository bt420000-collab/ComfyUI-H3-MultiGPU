from __future__ import annotations

"""Lab-only policy for near-symmetric dual-GPU H3VM systems.

The public rc6 defaults were tuned around an asymmetric 16G+8G pair. Two
near-identical 16 GB GPUs should not inherit the same 68/32 MLP split and
16-head helper ceiling. This module keeps equal-card experimentation isolated
from the stable public policy and from PM/task scheduling.

Enable the lab overlay with:
    H3VM_DUAL16_SYMMETRIC_LAB=1

Optional local tuning knob (not a public UI contract):
    H3VM_DUAL16_PRIMARY_FRACTION=0.56

The current lab knob changes the MLP row split only. Attention remains a
balanced 28/28 target for H3's 56 heads, with the existing runtime free-memory
safety cap still allowed to reduce helper work when necessary.
"""

import os


_TRUE = {"1", "true", "yes", "on"}


def lab_enabled() -> bool:
    return os.environ.get("H3VM_DUAL16_SYMMETRIC_LAB", "0").strip().lower() in _TRUE


def near_symmetric_values(a: float, b: float, tolerance: float = 0.08) -> bool:
    a = float(a)
    b = float(b)
    tolerance = max(0.0, float(tolerance))
    if a <= 0.0 or b <= 0.0:
        return False
    return abs(a - b) / max(a, b) <= tolerance


def _resolve_cuda_device(value):
    import torch

    text = str(value)
    if text.startswith("gpu:"):
        return torch.device("cuda", int(text.split(":", 1)[1]))
    return torch.device(text)


def pair_is_near_symmetric(primary, secondary, *, memory_tolerance=0.08, sm_tolerance=0.08):
    import torch

    a = _resolve_cuda_device(primary)
    b = _resolve_cuda_device(secondary)
    pa = torch.cuda.get_device_properties(a)
    pb = torch.cuda.get_device_properties(b)
    return (
        near_symmetric_values(pa.total_memory, pb.total_memory, memory_tolerance)
        and near_symmetric_values(pa.multi_processor_count, pb.multi_processor_count, sm_tolerance)
    )


def symmetric_head_counts(heads: int) -> list[int]:
    heads = int(heads)
    if heads < 2:
        raise ValueError("dual-GPU head split requires at least two heads")
    base, extra = divmod(heads, 2)
    return [base + extra, base]


def manual_dual_head_counts(heads: int, primary_fraction: float) -> list[int]:
    """Future public UI hook: integer head split from a primary-side fraction."""
    heads = int(heads)
    fraction = float(primary_fraction)
    if heads < 2:
        raise ValueError("dual-GPU head split requires at least two heads")
    if not 0.0 < fraction < 1.0:
        raise ValueError("primary_fraction must be between 0 and 1")
    primary = min(heads - 1, max(1, round(heads * fraction)))
    return [primary, heads - primary]


def requested_primary_fraction(default: float = 0.56) -> float:
    raw = os.environ.get("H3VM_DUAL16_PRIMARY_FRACTION")
    if raw is None or not str(raw).strip():
        return float(default)
    value = float(raw)
    # Lab guardrail. Wider public/manual ranges can be considered after hardware
    # telemetry proves them useful; this first sweep targets 50/50..60/40.
    if not 0.50 <= value <= 0.60:
        raise ValueError(
            "H3VM_DUAL16_PRIMARY_FRACTION must be between 0.50 and 0.60 in this lab"
        )
    return value


def symmetric_policy_overrides(mode: str, *, primary_fraction: float | None = None) -> dict:
    """Return the conservative first-pass equal-card execution overrides."""
    mode = str(mode)
    fraction = requested_primary_fraction() if primary_fraction is None else float(primary_fraction)
    if not 0.50 <= fraction <= 0.60:
        raise ValueError("symmetric lab primary fraction must be within 0.50..0.60")

    if mode == "DUAL_QUIET":
        return {
            "attention_head_balance": "balanced",
            "attention_helper_head_cap": 28,
            "mlp_primary_fraction": fraction,
            # Keep adaptive slack alive around the selected center. Host relay
            # overhead can make exact 50/50 compute slower even on equal GPUs.
            "critical_path_min_primary_fraction": max(0.50, fraction - 0.04),
            "critical_path_max_primary_fraction": min(0.64, fraction + 0.06),
            "critical_path_fraction_step": 0.02,
        }
    if mode == "DUAL_CAPACITY":
        return {
            "attention_head_balance": "balanced",
            "attention_helper_head_cap": 28,
            "capacity_helper_heads": 28,
            "mlp_primary_fraction": fraction,
            # Capacity is deterministic placement/chunking, so keep the chosen
            # ratio fixed instead of letting speed-path adaptation move it.
            "critical_path_min_primary_fraction": fraction,
            "critical_path_max_primary_fraction": fraction,
        }
    return {}


def apply_execution_overrides(kwargs: dict, *, mode: str, primary, secondary):
    """Apply the opt-in equal-card policy and return (new_kwargs, report_or_none)."""
    if not lab_enabled() or str(mode) not in {"DUAL_QUIET", "DUAL_CAPACITY"}:
        return dict(kwargs), None
    if not pair_is_near_symmetric(primary, secondary):
        return dict(kwargs), {
            "active": False,
            "reason": "selected GPUs are not near-symmetric",
            "mode": str(mode),
        }

    fraction = requested_primary_fraction()
    overrides = symmetric_policy_overrides(str(mode), primary_fraction=fraction)
    out = dict(kwargs)
    out.update(overrides)
    return out, {
        "active": True,
        "mode": str(mode),
        "primary_fraction": fraction,
        "attention_heads": symmetric_head_counts(56),
        "overrides": overrides,
    }


def inspect_visible_pair(primary="cuda:0", secondary="cuda:1"):
    import torch

    a = _resolve_cuda_device(primary)
    b = _resolve_cuda_device(secondary)
    pa = torch.cuda.get_device_properties(a)
    pb = torch.cuda.get_device_properties(b)
    peer_fn = getattr(torch.cuda, "can_device_access_peer", None)
    peer_ab = peer_ba = False
    if peer_fn is not None:
        try:
            peer_ab = bool(peer_fn(a.index, b.index))
            peer_ba = bool(peer_fn(b.index, a.index))
        except Exception:
            pass
    symmetric = pair_is_near_symmetric(a, b)
    report = {
        "primary": {
            "device": str(a),
            "name": torch.cuda.get_device_name(a),
            "vram_gib": round(pa.total_memory / (1024 ** 3), 3),
            "sm": int(pa.multi_processor_count),
        },
        "secondary": {
            "device": str(b),
            "name": torch.cuda.get_device_name(b),
            "vram_gib": round(pb.total_memory / (1024 ** 3), 3),
            "sm": int(pb.multi_processor_count),
        },
        "near_symmetric": bool(symmetric),
        "recommended_h3_heads": symmetric_head_counts(56) if symmetric else None,
        "recommended_mlp_primary_fraction": requested_primary_fraction() if symmetric else None,
        "p2p": {"ab": peer_ab, "ba": peer_ba},
    }
    return report
