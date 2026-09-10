from __future__ import annotations

"""Lab-only symmetric dual-GPU planner.

The public runtime historically assumes an asymmetric primary/helper pair in a
few relay heuristics. On two near-identical 16 GB GPUs that unnecessarily caps
GPU1 below an even 28/28 H3 attention split. This module provides an opt-in
runtime patch while the behavior is validated on real hardware.

Enable with:
    H3VM_DUAL16_SYMMETRIC_LAB=1

No PM/task scheduling concepts live here. The patch only changes attention-head
count planning for H3VMRelayAttentionParallel when the selected pair is judged
near-symmetric by physical VRAM and SM count.
"""

import logging
import os

LOG = logging.getLogger("H3VM")
_PATCHED = False
_LOGGED = set()


def near_symmetric_values(a: float, b: float, tolerance: float = 0.08) -> bool:
    a = float(a)
    b = float(b)
    tolerance = max(0.0, float(tolerance))
    if a <= 0.0 or b <= 0.0:
        return False
    return abs(a - b) / max(a, b) <= tolerance


def pair_is_near_symmetric(primary, secondary, *, memory_tolerance=0.08, sm_tolerance=0.08):
    import torch

    pa = torch.cuda.get_device_properties(primary)
    pb = torch.cuda.get_device_properties(secondary)
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
    """Future UI hook: exact integer head split from a primary-side fraction."""
    heads = int(heads)
    fraction = float(primary_fraction)
    if heads < 2:
        raise ValueError("dual-GPU head split requires at least two heads")
    if not 0.0 < fraction < 1.0:
        raise ValueError("primary_fraction must be between 0 and 1")
    primary = min(heads - 1, max(1, round(heads * fraction)))
    return [primary, heads - primary]


def inspect_visible_pair(primary="cuda:0", secondary="cuda:1"):
    import torch

    a = torch.device(primary)
    b = torch.device(secondary)
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
    return {
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
        "p2p": {"ab": peer_ab, "ba": peer_ba},
    }


def install_dual16g_symmetric_relay_patch() -> bool:
    """Opt-in patch: do not apply the historical small-GPU head cap to equal cards."""
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("H3VM_DUAL16_SYMMETRIC_LAB", "0").strip().lower() not in {
        "1", "true", "yes", "on"
    }:
        return False

    from .attention_parallel import H3VMRelayAttentionParallel

    if getattr(H3VMRelayAttentionParallel._relay_counts, "_h3vm_dual16_lab", False):
        _PATCHED = True
        return True

    original = H3VMRelayAttentionParallel._relay_counts

    def patched(self, heads, devices, seq_len, dim_head):
        configured_a, configured_b = self.pair
        try:
            symmetric = pair_is_near_symmetric(configured_a, configured_b)
        except Exception as exc:
            LOG.warning("H3VM dual16 lab symmetry probe failed; using stock planner | %s", exc)
            symmetric = False

        if not symmetric:
            return original(self, heads, devices, seq_len, dim_head)

        counts = symmetric_head_counts(heads)
        key = (str(configured_a), str(configured_b), int(heads))
        if key not in _LOGGED:
            LOG.info(
                "H3VM DUAL16 SYMMETRIC LAB | pair=%s/%s | heads=%d -> %d/%d | "
                "historical helper cap bypassed for near-identical GPUs",
                configured_a, configured_b, int(heads), counts[0], counts[1],
            )
            _LOGGED.add(key)
        return counts

    patched._h3vm_dual16_lab = True
    patched._h3vm_original = original
    H3VMRelayAttentionParallel._relay_counts = patched
    _PATCHED = True
    LOG.warning(
        "H3VM DUAL16 SYMMETRIC LAB enabled | experimental equal-card relay planner active"
    )
    return True
