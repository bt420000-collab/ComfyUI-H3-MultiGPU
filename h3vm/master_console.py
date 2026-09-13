"""H3VM Core policy helpers.

Pure/lightweight policy module.  No torch/comfy imports at module import time.
The public node entrypoint lives in the package __init__.py. This module stays
lightweight and contains policy/value translation only.
"""
from __future__ import annotations

import math
import os

MODES = (
    "SINGLE_GPU",
    "DUAL_QUIET",
    "DUAL_CAPACITY",
    "DUAL_SYNC_ACCEL",
    "DUAL_EXACT_SP",
)

MASTER_GPU_MODES = (
    "SINGLE_GPU｜单卡·标准模式",
    "DUAL_QUIET｜双卡协同·日常推荐",
    "DUAL_CAPACITY｜双卡扩显存·高清模式",
    "DUAL_SYNC_ACCEL｜双卡满血·近似执行",
    "DUAL_EXACT_SP｜双卡精确并行·预览",
)

# Public product surface is intentionally small. Internal/legacy backend names
# remain accepted by normalize_mode() so old workflows and lab nodes keep working.
PRODUCT_MODES = ("双卡极速", "双卡扩容", "双卡后台")
LEGACY_PRODUCT_MODE_MAP = {
    "单卡模式": "SINGLE_GPU",
    "双卡协同｜日常推荐": "DUAL_QUIET",
    "双卡扩容": "DUAL_CAPACITY",
    "双卡极速｜近似执行": "DUAL_SYNC_ACCEL",
    "双卡原生｜预览": "DUAL_EXACT_SP",
}
PUBLIC_PRODUCT_MODE_MAP = {
    "双卡极速": "DUAL_SYNC_ACCEL",
    "双卡扩容": "DUAL_CAPACITY",
    "双卡后台": "DUAL_QUIET",
}
PARTICIPATION_PRESETS = ("全｜100%", "高｜75%", "中｜50%", "低｜25%", "自定义")
_PARTICIPATION_VALUES = {
    "全｜100%": 100.0,
    "高｜75%": 75.0,
    "中｜50%": 50.0,
    "低｜25%": 25.0,
}
_CAPACITY_PARTICIPATION_ANCHORS = (
    (25.0, "SAFE｜保守·最稳"),
    (50.0, "RESIDENT4｜驻留+·多用显存"),
    (75.0, "RESIDENT8｜高驻留·减少搬运"),
    (100.0, "BALANCE｜均衡·推荐实验"),
)
STOCK_LORA = "无｜原版模型"
STOCK_STEPS = "20步｜原版模型"
CUSTOM_STEPS = "自定义步数"


def validate_sampling_contract(mode, turbo_lora, steps, style_lora_stack=None, custom_steps=8):
    normalize_mode(mode)
    del style_lora_stack
    if str(steps).strip() == CUSTOM_STEPS:
        requested = max(1, int(custom_steps))
    else:
        requested = step_value(steps)
    _, note = resolve_steps(turbo_lora, requested)
    return requested, note


def normalize_mode(value: str) -> str:
    text = str(value).strip()
    if text in PUBLIC_PRODUCT_MODE_MAP:
        return PUBLIC_PRODUCT_MODE_MAP[text]
    if text in LEGACY_PRODUCT_MODE_MAP:
        return LEGACY_PRODUCT_MODE_MAP[text]
    for mode in MODES:
        if text == mode or text.startswith(mode + "｜"):
            return mode
    raise ValueError(f"Unknown H3VM mode: {value}")


def resolve_public_mode(multi_gpu_enabled, gpu_mode) -> str:
    if not bool(multi_gpu_enabled):
        return "SINGLE_GPU"
    return normalize_mode(gpu_mode)


def resolve_participation_percent(preset, custom=100) -> float:
    key = str(preset).strip()
    if key in _PARTICIPATION_VALUES:
        return float(_PARTICIPATION_VALUES[key])
    if key == "自定义":
        return float(max(1.0, min(100.0, float(custom))))
    try:
        return float(max(1.0, min(100.0, float(key))))
    except Exception as exc:
        raise ValueError(f"Unknown H3VM participation preset: {preset}") from exc


def capacity_profile_for_participation(percent) -> str:
    value = max(1.0, min(100.0, float(percent)))
    return min(_CAPACITY_PARTICIPATION_ANCHORS, key=lambda item: (abs(item[0] - value), item[0]))[1]


def scale_secondary_work(base_count: int, percent, *, minimum: int = 1, maximum: int | None = None) -> int:
    base = max(0, int(base_count))
    if base == 0:
        return 0
    value = max(1.0, min(100.0, float(percent)))
    scaled = int(base * value / 100.0 + 0.5)
    if maximum is None:
        maximum = base
    return max(int(minimum), min(int(maximum), scaled))

STEP_PRESETS = ("4步｜极速", "6步｜平衡", "8步｜质量", CUSTOM_STEPS)
ASPECTS = ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9")
CAPACITY_VRAM_PROFILES = (
    "SAFE｜保守·最稳", "RESIDENT4｜驻留+·多用显存", "RESIDENT8｜高驻留·减少搬运",
    "MLP36｜副卡36%·轻度加活", "MLP40｜副卡40%·高负载", "HEAD20｜注意力+·副卡多分担",
    "BALANCE｜均衡·推荐实验", "MAX_TEST｜极限·可能爆显存",
)
_CAPACITY_VRAM_PROFILE_RATIOS = {
    "SAFE｜保守·最稳": {"primary_reserve_frac": 0.52, "secondary_reserve_frac": 0.55, "primary_cache_frac": 0.047, "secondary_cache_frac": 0.094, "trim_interval": 1, "mlp_primary_fraction": 0.68, "helper_heads": 16},
    "RESIDENT4｜驻留+·多用显存": {"primary_reserve_frac": 0.43, "secondary_reserve_frac": 0.38, "primary_cache_frac": 0.16, "secondary_cache_frac": 0.24, "trim_interval": 4, "mlp_primary_fraction": 0.68, "helper_heads": 16},
    "RESIDENT8｜高驻留·减少搬运": {"primary_reserve_frac": 0.36, "secondary_reserve_frac": 0.29, "primary_cache_frac": 0.24, "secondary_cache_frac": 0.34, "trim_interval": 8, "mlp_primary_fraction": 0.68, "helper_heads": 16},
    "MLP36｜副卡36%·轻度加活": {"primary_reserve_frac": 0.43, "secondary_reserve_frac": 0.38, "primary_cache_frac": 0.16, "secondary_cache_frac": 0.24, "trim_interval": 4, "mlp_primary_fraction": 0.64, "helper_heads": 16},
    "MLP40｜副卡40%·高负载": {"primary_reserve_frac": 0.41, "secondary_reserve_frac": 0.35, "primary_cache_frac": 0.18, "secondary_cache_frac": 0.27, "trim_interval": 4, "mlp_primary_fraction": 0.60, "helper_heads": 16},
    "HEAD20｜注意力+·副卡多分担": {"primary_reserve_frac": 0.43, "secondary_reserve_frac": 0.42, "primary_cache_frac": 0.16, "secondary_cache_frac": 0.20, "trim_interval": 4, "mlp_primary_fraction": 0.68, "helper_heads": 20},
    "BALANCE｜均衡·推荐实验": {"primary_reserve_frac": 0.37, "secondary_reserve_frac": 0.32, "primary_cache_frac": 0.22, "secondary_cache_frac": 0.30, "trim_interval": 4, "mlp_primary_fraction": 0.60, "helper_heads": 20},
    "MAX_TEST｜极限·可能爆显存": {"primary_reserve_frac": 0.28, "secondary_reserve_frac": 0.22, "primary_cache_frac": 0.30, "secondary_cache_frac": 0.42, "trim_interval": 8, "mlp_primary_fraction": 0.56, "helper_heads": 20},
}
_CAPACITY_PROFILE_ALIASES = {
    "SAFE｜保守（当前）": "SAFE｜保守·最稳", "BALANCED｜适度上调": "RESIDENT4｜驻留+·多用显存", "HIGH｜高占用实验": "BALANCE｜均衡·推荐实验", "MAX_TEST｜极限实验": "MAX_TEST｜极限·可能爆显存",
    "SAFE｜68/32 MLP｜40/16头｜每1块回收": "SAFE｜保守·最稳", "RESIDENT4｜68/32 MLP｜40/16头｜每4块回收": "RESIDENT4｜驻留+·多用显存", "RESIDENT8｜68/32 MLP｜40/16头｜每8块回收": "RESIDENT8｜高驻留·减少搬运", "MLP36｜64/36 MLP｜40/16头｜每4块回收": "MLP36｜副卡36%·轻度加活", "MLP40｜60/40 MLP｜40/16头｜每4块回收": "MLP40｜副卡40%·高负载", "HEAD20｜68/32 MLP｜36/20头｜每4块回收": "HEAD20｜注意力+·副卡多分担", "BALANCE｜60/40 MLP｜36/20头｜每4块回收": "BALANCE｜均衡·推荐实验", "MAX_TEST｜56/44 MLP｜36/20头｜每8块回收": "MAX_TEST｜极限·可能爆显存",
}


def resolve_capacity_vram_profile(profile: str, primary_total_gb: float, secondary_total_gb: float) -> dict:
    raw = str(profile); key = _CAPACITY_PROFILE_ALIASES.get(raw, raw)
    if key not in _CAPACITY_VRAM_PROFILE_RATIOS: raise ValueError(f"Unknown Capacity profile: {profile}")
    ptotal=max(1.0,float(primary_total_gb)); stotal=max(1.0,float(secondary_total_gb)); r=_CAPACITY_VRAM_PROFILE_RATIOS[key]
    def _legal(total,reserve_frac,cache_frac):
        reserve=max(0.5,total*float(reserve_frac)); reserve=min(reserve,max(0.5,total-0.5)); usable=max(0.0,total-reserve); cache=max(0.0,total*float(cache_frac)); cache=min(cache,usable); return round(reserve,3),round(cache,3)
    p_reserve,p_cache=_legal(ptotal,r["primary_reserve_frac"],r["primary_cache_frac"]); s_reserve,s_cache=_legal(stotal,r["secondary_reserve_frac"],r["secondary_cache_frac"])
    return {"profile":key,"requested_profile":raw,"primary_runtime_reserve_gb":p_reserve,"secondary_runtime_reserve_gb":s_reserve,"primary_hot_cache_gb":p_cache,"secondary_hot_cache_gb":s_cache,"single_root_trim_interval":int(r["trim_interval"]),"mlp_primary_fraction":float(r["mlp_primary_fraction"]),"capacity_helper_heads":int(r["helper_heads"])}

RESOLUTION_PRESETS=("0.2MP｜预览","0.5MP｜快速","0.7MP｜中高","768P｜原生高清","1.5MP｜超清实验","2.0MP｜1080级实验","CUSTOM｜自定义")
_REFERENCE_16_9={"0.2MP｜预览":(608,352),"0.5MP｜快速":(960,544),"0.7MP｜中高":(1152,640),"768P｜原生高清":(1344,768),"1.5MP｜超清实验":(1664,928),"2.0MP｜1080级实验":(1920,1088)}
_TARGET_MP={"0.2MP｜预览":0.20,"0.5MP｜快速":0.50,"0.7MP｜中高":0.70,"768P｜原生高清":None,"1.5MP｜超清实验":1.50,"2.0MP｜1080级实验":2.00}
_RATIO={"16:9":16/9,"9:16":9/16,"1:1":1.0,"4:3":4/3,"3:4":3/4,"21:9":21/9}; _NATIVE_SHORT=768

def _round32(x: float)->int: return max(32,int(round(float(x)/32.0))*32)
def _ceil32(x: float)->int: return max(32,int(math.ceil(float(x)/32.0))*32)
def resolve_resolution(aspect,preset,custom_width,custom_height):
    aspect=str(aspect); preset=str(preset)
    if aspect not in _RATIO: raise ValueError(f"Unsupported aspect ratio: {aspect}")
    if preset.startswith("CUSTOM"): return _round32(custom_width),_round32(custom_height)
    if aspect=="16:9" and preset in _REFERENCE_16_9: return _REFERENCE_16_9[preset]
    if aspect=="9:16" and preset in _REFERENCE_16_9:
        w,h=_REFERENCE_16_9[preset]; return h,w
    ratio=_RATIO[aspect]
    if preset=="768P｜原生高清":
        if ratio>=1.0: h=_NATIVE_SHORT; w=_round32(h*ratio)
        else: w=_NATIVE_SHORT; h=_round32(w/ratio)
        return int(w),int(h)
    mp=_TARGET_MP.get(preset)
    if mp is None: raise ValueError(f"Unsupported resolution preset: {preset}")
    pixels=float(mp)*1_000_000.0; w=math.sqrt(pixels*ratio); h=w/ratio; return _ceil32(w),_ceil32(h)

def duration_to_length(seconds: float)->int:
    frames=max(5,int(round(max(float(seconds),0.05)*24.0))); return int(frames+(5-(frames%17))%17)

def step_value(step_preset:str)->int:
    if isinstance(step_preset,(int,float)): return max(1,int(step_preset))
    text=str(step_preset).strip()
    if text==STOCK_STEPS or text.startswith("20"): return 20
    if text.startswith("4"): return 4
    if text.startswith("6"): return 6
    if text.startswith("8"): return 8
    raise ValueError(f"Unknown H3 Turbo step preset: {step_preset}")

def lora_family(lora_name:str)->str:
    if str(lora_name).strip()==STOCK_LORA: return "STOCK_H3"
    name=os.path.basename(str(lora_name)).lower()
    if "fl2v_turbo_8step_v1.0" in name: return "LIGHTX2V_V1_8STEP"
    if "fl2v_turbo_4step_v1.0_768p" in name: return "LIGHTX2V_V1_4STEP_768P"
    if "turbo_v4_step600" in name: return "LARRY_V4_600"
    if "turbo_4step" in name or "ckpt850" in name or "ckpt500" in name: return "LARRY_4STEP_LINE"
    return "CUSTOM_TURBO"

def resolve_steps(lora_name:str,step_preset:str):
    requested=step_value(step_preset); family=lora_family(lora_name)
    if family=="LIGHTX2V_V1_4STEP_768P": return (requested,"当前 LoRA 名称标注为 4-step；已按你的设置运行，不会强制改步数。") if requested!=4 else (4,None)
    if family=="LIGHTX2V_V1_8STEP" and requested==6: return 6,"LightX2V v1.0 officially documents 8 steps (and 4-step use); 6-step is a compatibility experiment."
    if family=="LARRY_V4_600" and requested==4: return 4,"Larry v4-600 supports 4 steps, but 6-8 is the current quality recommendation."
    return requested,None

def mode_summary(mode:str)->str:
    return {"SINGLE_GPU":"GPU0 single-card baseline","DUAL_QUIET":"dual-GPU background / reserved-compute mode","DUAL_CAPACITY":"dual-GPU capacity mode","DUAL_SYNC_ACCEL":"dual-GPU speed mode","DUAL_EXACT_SP":"internal exact 2-rank reference backend"}.get(str(mode),str(mode))

def apply_sigma_shift(model,shift_video:float=12.0,shift_audio:float=3.0):
    import comfy.model_sampling
    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST): pass
    original=model.get_model_object("model_sampling"); model_sampling=ModelSamplingAdvanced(model.model.model_config); model_sampling.set_parameters(shift=float(shift_video),audio_shift=float(shift_audio))
    if hasattr(original,"noise_scale"): model_sampling.set_noise_scale(original.noise_scale)
    model.add_object_patch("model_sampling",model_sampling); to=model.model_options["transformer_options"]=model.model_options.get("transformer_options",{}).copy(); to["minimax_h3_sigma_shift_video"]=float(shift_video); to["minimax_h3_sigma_shift_audio"]=float(shift_audio); return model
