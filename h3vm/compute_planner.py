from __future__ import annotations

"""H3 VRAM Master compute planning primitives.

This module is deliberately device-light: policy arithmetic is pure Python and
CUDA probing is isolated in ``probe_cuda_pair``. Public execution backends can
share one description of the selected GPU pair without importing private PM,
worker, job, lease, or project orchestration concepts.
"""

from dataclasses import dataclass
import os
import sys


_RATIO_PRESETS = {
    "50/50": 0.50,
    "52/48": 0.52,
    "54/46": 0.54,
    "56/44": 0.56,
    "58/42": 0.58,
    "60/40": 0.60,
}


def parse_primary_fraction(value, *, default=None):
    """Parse a hidden/lab compute ratio into the primary-device fraction.

    Accepted forms: ``AUTO``, ``0.54``, ``54``, ``54/46`` and the validated
    preset spellings above. The public UI is intentionally not wired yet.
    """
    if value is None:
        return default
    text = str(value).strip().upper()
    if not text or text == "AUTO":
        return default
    if text in _RATIO_PRESETS:
        return _RATIO_PRESETS[text]
    if "/" in text:
        left, right = text.split("/", 1)
        a, b = float(left), float(right)
        if a <= 0 or b <= 0:
            raise ValueError("compute ratio parts must be positive")
        fraction = a / (a + b)
    else:
        number = float(text)
        fraction = number / 100.0 if number > 1.0 else number
    if not 0.20 <= fraction <= 0.80:
        raise ValueError(f"unsafe H3VM primary fraction: {fraction:.4f}")
    return float(fraction)


@dataclass(frozen=True)
class DeviceProfile:
    label: str
    name: str
    vram_gib: float
    sms: int


@dataclass(frozen=True)
class PairProfile:
    primary: DeviceProfile
    secondary: DeviceProfile
    p2p_ab: bool = False
    p2p_ba: bool = False
    platform: str = sys.platform

    @property
    def bidirectional_p2p(self) -> bool:
        return bool(self.p2p_ab and self.p2p_ba)

    @property
    def vram_similarity(self) -> float:
        a = max(1e-6, float(self.primary.vram_gib))
        b = max(1e-6, float(self.secondary.vram_gib))
        return min(a, b) / max(a, b)

    @property
    def sm_similarity(self) -> float:
        a = max(1, int(self.primary.sms))
        b = max(1, int(self.secondary.sms))
        return min(a, b) / max(a, b)

    @property
    def near_symmetric(self) -> bool:
        return self.vram_similarity >= 0.92 and self.sm_similarity >= 0.92

    @property
    def host_relay_only(self) -> bool:
        return not self.bidirectional_p2p


@dataclass(frozen=True)
class RuntimePlan:
    backend: str
    primary_fraction: float
    secondary_fraction: float
    mode4_primary_blocks: int
    mode4_secondary_blocks: int
    attention_primary_heads: int
    attention_secondary_heads: int
    transport: str
    near_symmetric: bool
    manual_ratio: bool
    exact_sp_tier: str
    reason: str

    def summary(self) -> str:
        return (
            f"backend={self.backend} compute={self.primary_fraction*100:.0f}/"
            f"{self.secondary_fraction*100:.0f} blocks={self.mode4_primary_blocks}/"
            f"{self.mode4_secondary_blocks} heads={self.attention_primary_heads}/"
            f"{self.attention_secondary_heads} transport={self.transport} "
            f"exact_sp={self.exact_sp_tier}"
        )


def _integer_split(total: int, primary_fraction: float, *, alignment: int = 1):
    total = int(total)
    alignment = max(1, int(alignment))
    raw = int(round(total * float(primary_fraction)))
    if alignment > 1:
        raw = int(round(raw / alignment) * alignment)
    primary = min(total - 1, max(1, raw))
    return primary, total - primary


def _auto_primary_fraction(pair: PairProfile, backend: str) -> tuple[float, str]:
    backend = str(backend).upper()
    if pair.near_symmetric:
        return 0.50, "near-symmetric GPU pair"

    # Mode4's historical public heterogeneous profile is 28/22 blocks, i.e.
    # 56/44 primary/secondary. Preserve that proven baseline instead of trying
    # to infer a new ratio from product names.
    if backend == "MODE4":
        return 0.56, "validated heterogeneous Mode4 baseline"

    # Exact/SP arithmetic is compute-oriented. Use SM share as the first-order
    # target, then cap it by a conservative VRAM share so a small helper is not
    # assigned a compute slice it cannot keep resident.
    sm_total = max(2, int(pair.primary.sms) + int(pair.secondary.sms))
    sm_fraction = float(pair.primary.sms) / sm_total
    vram_total = max(1e-6, float(pair.primary.vram_gib) + float(pair.secondary.vram_gib))
    vram_fraction = float(pair.primary.vram_gib) / vram_total
    fraction = max(0.35, min(0.72, max(sm_fraction, vram_fraction - 0.04)))
    return fraction, "SM/VRAM weighted asymmetric plan"


def build_runtime_plan(pair: PairProfile, *, backend: str, manual_ratio=None,
                       total_blocks: int = 50, total_heads: int = 56) -> RuntimePlan:
    backend_key = str(backend).upper()
    env_ratio = os.environ.get("H3VM_COMPUTE_RATIO")
    explicit = manual_ratio if manual_ratio is not None else env_ratio
    auto_fraction, reason = _auto_primary_fraction(pair, backend_key)
    fraction = parse_primary_fraction(explicit, default=auto_fraction)
    is_manual = parse_primary_fraction(explicit, default=None) is not None

    pb, sb = _integer_split(total_blocks, fraction)
    ph, sh = _integer_split(total_heads, fraction)

    if pair.bidirectional_p2p:
        transport = "p2p"
        exact_tier = "READY"
    elif str(pair.platform).lower().startswith("win"):
        transport = "host_pageable"
        exact_tier = "LAB_HOST_RELAY"
    else:
        transport = "host_relay"
        exact_tier = "LAB_HOST_RELAY"

    if backend_key == "MODE4" and pair.near_symmetric and not is_manual:
        # Real dual-16G production reference validated 25/25. Keep this exact
        # result rather than relying on rounding side effects.
        pb, sb = 25, 25
        fraction = 0.50
        reason = "validated symmetric dual-16G Mode4 25/25"

    return RuntimePlan(
        backend=backend_key,
        primary_fraction=float(fraction),
        secondary_fraction=1.0 - float(fraction),
        mode4_primary_blocks=int(pb),
        mode4_secondary_blocks=int(sb),
        attention_primary_heads=int(ph),
        attention_secondary_heads=int(sh),
        transport=transport,
        near_symmetric=pair.near_symmetric,
        manual_ratio=bool(is_manual),
        exact_sp_tier=exact_tier,
        reason=reason,
    )


def probe_cuda_pair(primary="cuda:0", secondary="cuda:1") -> PairProfile:
    """Probe only physical properties needed by public runtime policy."""
    import torch

    p = torch.device(primary)
    s = torch.device(secondary)
    pp = torch.cuda.get_device_properties(p)
    sp = torch.cuda.get_device_properties(s)

    fn = getattr(torch.cuda, "can_device_access_peer", None)
    p2p_ab = p2p_ba = False
    if fn is not None:
        try:
            p2p_ab = bool(fn(p.index, s.index))
            p2p_ba = bool(fn(s.index, p.index))
        except Exception:
            pass

    gib = float(1024 ** 3)
    return PairProfile(
        primary=DeviceProfile(str(p), str(pp.name), float(pp.total_memory) / gib,
                              int(pp.multi_processor_count)),
        secondary=DeviceProfile(str(s), str(sp.name), float(sp.total_memory) / gib,
                                int(sp.multi_processor_count)),
        p2p_ab=p2p_ab,
        p2p_ba=p2p_ba,
        platform=sys.platform,
    )
