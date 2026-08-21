"""H3VM Master Loader policy helpers.

Pure/lightweight policy module.  No torch/comfy imports at module import time.
The node entrypoint lives in the package __init__.py so old workflows remain
backward compatible while new workflows can collapse H3 tuning into one node.
"""
from __future__ import annotations

import math
import os

MODES = (
    "SINGLE_GPU",
    "DUAL_QUIET",
    "DUAL_CAPACITY",
    "DUAL_SYNC_ACCEL",
)

MASTER_GPU_MODES = (
    "SINGLE_GPU｜单卡·标准模式",
    "DUAL_QUIET｜双卡协同·日常推荐",
    "DUAL_CAPACITY｜双卡扩显存·高清模式",
    "DUAL_SYNC_ACCEL｜双卡同步加速·需外部后端",
)


def normalize_mode(value: str) -> str:
    text = str(value).strip()
    for mode in MODES:
        if text == mode or text.startswith(mode + "｜"):
            return mode
    raise ValueError(f"Unknown H3VM mode: {value}")

STEP_PRESETS = (
    "4步｜极速",
    "6步｜平衡",
    "8步｜质量",
)

ASPECTS = ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9")

CAPACITY_VRAM_PROFILES = (
    "SAFE｜保守·最稳",
    "RESIDENT4｜驻留+·多用显存",
    "RESIDENT8｜高驻留·减少搬运",
    "MLP36｜副卡36%·轻度加活",
    "MLP40｜副卡40%·高负载",
    "HEAD20｜注意力+·副卡多分担",
    "BALANCE｜均衡·推荐实验",
    "MAX_TEST｜极限·可能爆显存",
)

# Capacity Lab profiles intentionally expose the three variables we actually
# want to measure on the user's asymmetric 16G+8G pair:
#   1) DynamicVRAM residency/trim cadence;
#   2) MLP token-row compute split;
#   3) QKV attention head split.
#
# The option label itself is the user-facing explanation.  Technical split,
# head count and trim cadence stay in telemetry/logs instead of the dropdown.
# SAFE reproduces Capacity V1.  The
# higher profiles are experiments, not promises: MAX_TEST is allowed to OOM
# and exists specifically to locate the 8GB helper card's real capacity wall.
_CAPACITY_VRAM_PROFILE_RATIOS = {
    "SAFE｜保守·最稳": {
        "primary_reserve_frac": 0.52, "secondary_reserve_frac": 0.55,
        "primary_cache_frac": 0.047, "secondary_cache_frac": 0.094,
        "trim_interval": 1, "mlp_primary_fraction": 0.68, "helper_heads": 16,
    },
    "RESIDENT4｜驻留+·多用显存": {
        "primary_reserve_frac": 0.43, "secondary_reserve_frac": 0.38,
        "primary_cache_frac": 0.16, "secondary_cache_frac": 0.24,
        "trim_interval": 4, "mlp_primary_fraction": 0.68, "helper_heads": 16,
    },
    "RESIDENT8｜高驻留·减少搬运": {
        "primary_reserve_frac": 0.36, "secondary_reserve_frac": 0.29,
        "primary_cache_frac": 0.24, "secondary_cache_frac": 0.34,
        "trim_interval": 8, "mlp_primary_fraction": 0.68, "helper_heads": 16,
    },
    "MLP36｜副卡36%·轻度加活": {
        "primary_reserve_frac": 0.43, "secondary_reserve_frac": 0.38,
        "primary_cache_frac": 0.16, "secondary_cache_frac": 0.24,
        "trim_interval": 4, "mlp_primary_fraction": 0.64, "helper_heads": 16,
    },
    "MLP40｜副卡40%·高负载": {
        "primary_reserve_frac": 0.41, "secondary_reserve_frac": 0.35,
        "primary_cache_frac": 0.18, "secondary_cache_frac": 0.27,
        "trim_interval": 4, "mlp_primary_fraction": 0.60, "helper_heads": 16,
    },
    "HEAD20｜注意力+·副卡多分担": {
        # More helper QKV/attention activation needs a little more workspace
        # headroom than MLP-only profiles.
        "primary_reserve_frac": 0.43, "secondary_reserve_frac": 0.42,
        "primary_cache_frac": 0.16, "secondary_cache_frac": 0.20,
        "trim_interval": 4, "mlp_primary_fraction": 0.68, "helper_heads": 20,
    },
    "BALANCE｜均衡·推荐实验": {
        "primary_reserve_frac": 0.37, "secondary_reserve_frac": 0.32,
        "primary_cache_frac": 0.22, "secondary_cache_frac": 0.30,
        "trim_interval": 4, "mlp_primary_fraction": 0.60, "helper_heads": 20,
    },
    "MAX_TEST｜极限·可能爆显存": {
        "primary_reserve_frac": 0.28, "secondary_reserve_frac": 0.22,
        "primary_cache_frac": 0.30, "secondary_cache_frac": 0.42,
        "trim_interval": 8, "mlp_primary_fraction": 0.56, "helper_heads": 20,
    },
}

# Backward aliases let old v0.18 workflows open instead of failing on the old
# four labels.  They map to the closest v0.19 experiment rather than silently
# changing H3 math in arbitrary ways.
_CAPACITY_PROFILE_ALIASES = {
    # v0.18 labels
    "SAFE｜保守（当前）": "SAFE｜保守·最稳",
    "BALANCED｜适度上调": "RESIDENT4｜驻留+·多用显存",
    "HIGH｜高占用实验": "BALANCE｜均衡·推荐实验",
    "MAX_TEST｜极限实验": "MAX_TEST｜极限·可能爆显存",
    # v0.19 Capacity Lab technical labels
    "SAFE｜68/32 MLP｜40/16头｜每1块回收": "SAFE｜保守·最稳",
    "RESIDENT4｜68/32 MLP｜40/16头｜每4块回收": "RESIDENT4｜驻留+·多用显存",
    "RESIDENT8｜68/32 MLP｜40/16头｜每8块回收": "RESIDENT8｜高驻留·减少搬运",
    "MLP36｜64/36 MLP｜40/16头｜每4块回收": "MLP36｜副卡36%·轻度加活",
    "MLP40｜60/40 MLP｜40/16头｜每4块回收": "MLP40｜副卡40%·高负载",
    "HEAD20｜68/32 MLP｜36/20头｜每4块回收": "HEAD20｜注意力+·副卡多分担",
    "BALANCE｜60/40 MLP｜36/20头｜每4块回收": "BALANCE｜均衡·推荐实验",
    "MAX_TEST｜56/44 MLP｜36/20头｜每8块回收": "MAX_TEST｜极限·可能爆显存",
}


def resolve_capacity_vram_profile(profile: str, primary_total_gb: float, secondary_total_gb: float) -> dict:
    """Translate a Capacity Lab option into legal memory + compute policy.

    The resolver always keeps at least 0.5 GiB as a legal reserve and clamps hot
    cache to VRAM left after reserve.  Higher profiles can still OOM at runtime
    because activations are dynamic; that is intentional for MAX_TEST.
    """
    raw = str(profile)
    key = _CAPACITY_PROFILE_ALIASES.get(raw, raw)
    if key not in _CAPACITY_VRAM_PROFILE_RATIOS:
        raise ValueError(f"Unknown Capacity profile: {profile}")
    ptotal = max(1.0, float(primary_total_gb))
    stotal = max(1.0, float(secondary_total_gb))
    r = _CAPACITY_VRAM_PROFILE_RATIOS[key]

    def _legal(total, reserve_frac, cache_frac):
        reserve = max(0.5, total * float(reserve_frac))
        reserve = min(reserve, max(0.5, total - 0.5))
        usable = max(0.0, total - reserve)
        cache = max(0.0, total * float(cache_frac))
        cache = min(cache, usable)
        return round(reserve, 3), round(cache, 3)

    p_reserve, p_cache = _legal(ptotal, r["primary_reserve_frac"], r["primary_cache_frac"])
    s_reserve, s_cache = _legal(stotal, r["secondary_reserve_frac"], r["secondary_cache_frac"])
    return {
        "profile": key,
        "requested_profile": raw,
        "primary_runtime_reserve_gb": p_reserve,
        "secondary_runtime_reserve_gb": s_reserve,
        "primary_hot_cache_gb": p_cache,
        "secondary_hot_cache_gb": s_cache,
        "single_root_trim_interval": int(r["trim_interval"]),
        "mlp_primary_fraction": float(r["mlp_primary_fraction"]),
        "capacity_helper_heads": int(r["helper_heads"]),
    }

RESOLUTION_PRESETS = (
    "0.2MP｜预览",
    "0.5MP｜快速",
    "0.7MP｜中高",
    "768P｜原生高清",
    "1.5MP｜超清实验",
    "2.0MP｜1080级实验",
    "CUSTOM｜自定义",
)

# Official ModelTC/ComfyUI 16:9 reference table where available.  Keeping the
# exact published dimensions avoids tiny aspect/rounding drift in comparisons.
_REFERENCE_16_9 = {
    "0.2MP｜预览": (608, 352),
    "0.5MP｜快速": (960, 544),
    "0.7MP｜中高": (1152, 640),
    "768P｜原生高清": (1344, 768),
    "1.5MP｜超清实验": (1664, 928),
    "2.0MP｜1080级实验": (1920, 1088),
}
_TARGET_MP = {
    "0.2MP｜预览": 0.20,
    "0.5MP｜快速": 0.50,
    "0.7MP｜中高": 0.70,
    "768P｜原生高清": None,
    "1.5MP｜超清实验": 1.50,
    "2.0MP｜1080级实验": 2.00,
}
_RATIO = {
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "21:9": 21 / 9,
}
_NATIVE_SHORT = 768


def _round32(x: float) -> int:
    return max(32, int(round(float(x) / 32.0)) * 32)


def _ceil32(x: float) -> int:
    return max(32, int(math.ceil(float(x) / 32.0)) * 32)


def resolve_resolution(aspect: str, preset: str, custom_width: int, custom_height: int) -> tuple[int, int]:
    aspect = str(aspect)
    preset = str(preset)
    if aspect not in _RATIO:
        raise ValueError(f"Unsupported aspect ratio: {aspect}")

    if preset.startswith("CUSTOM"):
        # H3 requires multiples of 32.  Round to nearest instead of silently
        # changing only one axis and distorting the requested ratio.
        return _round32(custom_width), _round32(custom_height)

    if aspect == "16:9" and preset in _REFERENCE_16_9:
        return _REFERENCE_16_9[preset]
    if aspect == "9:16" and preset in _REFERENCE_16_9:
        w, h = _REFERENCE_16_9[preset]
        return h, w

    ratio = _RATIO[aspect]
    if preset == "768P｜原生高清":
        if ratio >= 1.0:
            h = _NATIVE_SHORT
            w = _round32(h * ratio)
        else:
            w = _NATIVE_SHORT
            h = _round32(w / ratio)
        return int(w), int(h)

    mp = _TARGET_MP.get(preset)
    if mp is None:
        raise ValueError(f"Unsupported resolution preset: {preset}")
    pixels = float(mp) * 1_000_000.0
    w = math.sqrt(pixels * ratio)
    h = w / ratio
    # Conservative capacity preset: ceil each dimension to the next valid H3
    # multiple so the label never promises more pixels than are actually run.
    return _ceil32(w), _ceil32(h)


def duration_to_length(seconds: float) -> int:
    """24fps, snapped upward to the H3 17*n+5 frame grid."""
    frames = max(5, int(round(max(float(seconds), 0.05) * 24.0)))
    return int(frames + (5 - (frames % 17)) % 17)


def step_value(step_preset: str) -> int:
    text = str(step_preset).strip()
    if text.startswith("4"):
        return 4
    if text.startswith("6"):
        return 6
    if text.startswith("8"):
        return 8
    raise ValueError(f"Unknown H3 Turbo step preset: {step_preset}")


def lora_family(lora_name: str) -> str:
    name = os.path.basename(str(lora_name)).lower()
    if "fl2v_turbo_8step_v1.0" in name:
        return "LIGHTX2V_V1_8STEP"
    if "fl2v_turbo_4step_v1.0_768p" in name:
        return "LIGHTX2V_V1_4STEP_768P"
    if "turbo_v4_step600" in name:
        return "LARRY_V4_600"
    if "turbo_4step" in name or "ckpt850" in name or "ckpt500" in name:
        return "LARRY_4STEP_LINE"
    return "CUSTOM_TURBO"


def resolve_steps(lora_name: str, step_preset: str) -> tuple[int, str | None]:
    """Return runtime steps and an optional compatibility note.

    We intentionally do not offer arbitrary integers in the Master Loader.
    LightX2V 8-step v1.0 officially documents 8 NFE and also 4-step inference;
    6 is left available as a clearly-labelled interpolation/compatibility test.
    The dedicated LightX2V 4-step 768p checkpoint is hard-locked to 4.
    """
    requested = step_value(step_preset)
    family = lora_family(lora_name)
    if family == "LIGHTX2V_V1_4STEP_768P":
        if requested != 4:
            return 4, "LightX2V 4-step 768p LoRA is locked to 4 steps; Master Loader forced 4."
        return 4, None
    if family == "LIGHTX2V_V1_8STEP" and requested == 6:
        return 6, "LightX2V v1.0 officially documents 8 steps (and 4-step use); 6-step is a compatibility experiment."
    if family == "LARRY_V4_600" and requested == 4:
        return 4, "Larry v4-600 supports 4 steps, but 6-8 is the current quality recommendation."
    return requested, None


def mode_summary(mode: str) -> str:
    return {
        "SINGLE_GPU": "GPU0 single-card baseline",
        "DUAL_QUIET": "Dev18 dual cooperative / quiet mode",
        "DUAL_CAPACITY": "pooled-VRAM capacity mode",
        "DUAL_SYNC_ACCEL": "reserved external synchronous multi-GPU backend",
    }.get(str(mode), str(mode))


def apply_sigma_shift(model, shift_video: float = 12.0, shift_audio: float = 3.0):
    """In-place equivalent of current ComfyUI MiniMaxH3SigmaShift.

    The H3VM model patcher is already a private loader result, so applying the
    object patch in place avoids another expensive/ambiguous clone while keeping
    the exact ModelSamplingAV schedule contract used by current ComfyUI.
    """
    import comfy.model_sampling

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    original = model.get_model_object("model_sampling")
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=float(shift_video), audio_shift=float(shift_audio))
    if hasattr(original, "noise_scale"):
        model_sampling.set_noise_scale(original.noise_scale)
    model.add_object_patch("model_sampling", model_sampling)
    to = model.model_options["transformer_options"] = model.model_options.get("transformer_options", {}).copy()
    to["minimax_h3_sigma_shift_video"] = float(shift_video)
    to["minimax_h3_sigma_shift_audio"] = float(shift_audio)
    return model
