"""H3VM v0.20.0-rc6 public product wrapper.

The frozen rc1 runtime lives in base_runtime.py. rc5 added the public MODEL->MODEL
Core engine, mode-independent ordinary LoRA stack, and built-in Stock-H3 Mode4.
rc6 adds Windows multi-GPU compatibility guards without rewriting the proven
Quiet/Capacity/Mode4 execution algorithms.
"""
from .base_runtime import *  # noqa: F401,F403
from . import base_runtime as _base

WEB_DIRECTORY = _base.WEB_DIRECTORY
H3VM_SHOW_LAB_NODES = _base.H3VM_SHOW_LAB_NODES

MASTER_GPU_MODES_RC5 = (
    "SINGLE_GPU｜单卡·标准模式",
    "DUAL_QUIET｜双卡协同·日常推荐",
    "DUAL_CAPACITY｜双卡扩显存·高清模式",
    "DUAL_SYNC_ACCEL｜双卡满血·原版20步",
)
CORE_GPU_MODES_RC5 = (
    "SINGLE_GPU｜单卡·执行引擎",
    "DUAL_QUIET｜双卡协同·执行引擎",
    "DUAL_CAPACITY｜双卡扩显存·执行引擎",
    "DUAL_SYNC_ACCEL｜双卡满血·Snapshot引擎",
)


class H3VMStyleLoRAStack:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        from .h3vm.style_lora import NONE_LORA
        choices = [NONE_LORA] + list(folder_paths.get_filename_list("loras"))
        req = {}
        for i in range(1, 5):
            req[f"lora_{i}"] = (choices, {"default": NONE_LORA})
            req[f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05})
        return {"required": req, "optional": {"previous_stack": ("H3VM_LORA_STACK",)}}

    RETURN_TYPES = ("H3VM_LORA_STACK", "STRING")
    RETURN_NAMES = ("style_lora_stack", "summary")
    FUNCTION = "build"
    CATEGORY = "MiniMaxH3/H3VM"
    DESCRIPTION = "普通风格/角色 LoRA 叠加。每节点 4 槽，可串接 previous_stack；与 Turbo 加速 LoRA 分离。"

    def build(self, lora_1, strength_1, lora_2, strength_2, lora_3, strength_3,
              lora_4, strength_4, previous_stack=None):
        from .h3vm.style_lora import append_style_loras, stack_summary
        stack = append_style_loras(previous_stack, [
            (lora_1, strength_1), (lora_2, strength_2), (lora_3, strength_3), (lora_4, strength_4)
        ])
        return stack, stack_summary(stack)


class H3VMCoreEngine:
    @classmethod
    def INPUT_TYPES(cls):
        from .h3vm.master_console import CAPACITY_VRAM_PROFILES
        return {"required": {
            "model": ("MODEL",),
            "gpu_mode": (CORE_GPU_MODES_RC5, {"default": "DUAL_QUIET｜双卡协同·执行引擎"}),
            "capacity_vram_profile": (CAPACITY_VRAM_PROFILES, {"default": "SAFE｜保守·最稳"}),
            "expected_steps_hint": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1,
                "tooltip": "只给 H3VM 做预取/最后一步调度提示；不修改主工作流实际 Steps。"}),
            "telemetry": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL", "H3VM_MODE")
    RETURN_NAMES = ("model", "mode")
    FUNCTION = "adapt"
    CATEGORY = "MiniMaxH3/H3VM"
    DESCRIPTION = "纯执行引擎：MODEL -> H3VM -> MODEL。不修改 Prompt/Seed/分辨率/Sampler/Sigmas/实际 Steps。"

    def adapt(self, model, gpu_mode="DUAL_QUIET｜双卡协同·执行引擎",
              capacity_vram_profile="SAFE｜保守·最稳", expected_steps_hint=4, telemetry=True):
        from .h3vm.master_console import normalize_mode
        from .h3vm.core_adapter import H3VMCoreConfig, adapt_model
        mode = normalize_mode(gpu_mode)
        out = adapt_model(model, config=H3VMCoreConfig(
            mode=mode, primary_device="gpu:0", secondary_device="gpu:1",
            capacity_vram_profile=str(capacity_vram_profile), expected_steps=max(1, int(expected_steps_hint)),
            telemetry=bool(telemetry),
        ))
        return out, mode


class H3VMMasterLoader(_base.H3VMMasterLoader):
    """Standalone cockpit stays integrated; external workflows should use H3VMCoreEngine."""

    @classmethod
    def INPUT_TYPES(cls):
        spec = _base.H3VMMasterLoader.INPUT_TYPES()
        spec = {k: dict(v) for k, v in spec.items()}
        req = dict(spec["required"])
        req["gpu_mode"] = (MASTER_GPU_MODES_RC5, {"default": "DUAL_QUIET｜双卡协同·日常推荐"})
        spec["required"] = req
        opt = dict(spec.get("optional", {}))
        opt["style_lora_stack"] = ("H3VM_LORA_STACK",)
        spec["optional"] = opt
        return spec

    def load(self, unet_name, turbo_lora, lora_strength=1.0, gpu_mode="DUAL_QUIET",
             capacity_vram_profile="SAFE｜保守·最稳", steps="4步｜极速", duration_seconds=15.0,
             aspect_ratio="16:9", resolution="768P｜原生高清", custom_width=1344, custom_height=768,
             seed=0, telemetry=True, prompt="", style_lora_stack=None):
        from .h3vm.master_console import (
            resolve_resolution, duration_to_length, resolve_steps, lora_family,
            mode_summary, apply_sigma_shift, normalize_mode,
        )
        from .h3vm.style_lora import stack_summary
        mode = normalize_mode(gpu_mode)
        width, height = resolve_resolution(aspect_ratio, resolution, custom_width, custom_height)
        length = duration_to_length(float(duration_seconds))
        if mode == "DUAL_SYNC_ACCEL":
            step_count = 20
            note = "Mode4 standalone uses Stock H3 FullThrottle; Turbo and 4/6/8 preset ignored; ordinary Style LoRA remains enabled."
            family = "STOCK_H3"
        else:
            step_count, note = resolve_steps(str(turbo_lora), str(steps))
            family = lora_family(str(turbo_lora))
        if note:
            print(f"[H3VM MASTER] Turbo step note: {note}", flush=True)
        if mode == "DUAL_CAPACITY" and int(step_count) < 8:
            print("[H3VM MASTER] Capacity audio note: 8-step recommended; lower steps remain experimental.", flush=True)
        print(
            f"[H3VM MASTER] mode={mode_summary(mode)} | lora={turbo_lora} family={family} | steps={step_count} | "
            f"duration={float(duration_seconds):.2f}s -> length={length} | canvas={width}x{height} | "
            f"capacity_vram={capacity_vram_profile} | style_lora={stack_summary(style_lora_stack)}",
            flush=True,
        )
        from .h3vm import core_adapter as _rc5_core  # noqa: F401
        model = H3VMCapacityModeTurboLoader().load(
            mode=mode, unet_name=str(unet_name), lora_name=str(turbo_lora), strength=float(lora_strength),
            low_vram=False, primary_device="gpu:0", secondary_device="gpu:1",
            capacity_vram_profile=str(capacity_vram_profile), capacity_mlp_chunk_rows=4096,
            capacity_outproj_chunk_rows=4096, capacity_helper_heads=16,
            capacity_attention_kernel="INT8_CURRENT", telemetry=bool(telemetry),
            expected_steps=int(step_count), style_lora_stack=style_lora_stack,
        )[0]
        import comfy.samplers
        if mode == "DUAL_SYNC_ACCEL":
            sampler = comfy.samplers.sampler_object("res_multistep")
            sampler_kind = "stock_res_multistep+20step(no_sigma_shift)"
        else:
            model = apply_sigma_shift(model, 12.0, 3.0)
            if family.startswith("LIGHTX2V_"):
                sampler = comfy.samplers.sampler_object("euler")
                sampler_kind = "euler+ModelSamplingAV(12/3)"
            else:
                from .h3vm.turbo_compat import _load_larry_module
                larry = _load_larry_module()
                if not hasattr(larry, "_turbo_sampler"):
                    raise RuntimeError("Installed ComfyUI-MiniMax-H3-Turbo lacks _turbo_sampler; please update it.")
                sampler = comfy.samplers.KSAMPLER(larry._turbo_sampler)
                sampler_kind = "larry_dual_clock(12/3)"
        print(f"[H3VM MASTER] sampler={sampler_kind}", flush=True)
        return model, mode, str(prompt), int(width), int(height), int(length), int(step_count), int(seed), sampler


PUBLIC_NODE_CLASS_MAPPINGS = dict(_base.PUBLIC_NODE_CLASS_MAPPINGS)
PUBLIC_NODE_CLASS_MAPPINGS.update({
    "H3VMMasterLoader": H3VMMasterLoader,
    "H3VMCoreEngine": H3VMCoreEngine,
    "H3VMStyleLoRAStack": H3VMStyleLoRAStack,
})
PUBLIC_NODE_DISPLAY_NAME_MAPPINGS = dict(_base.PUBLIC_NODE_DISPLAY_NAME_MAPPINGS)
PUBLIC_NODE_DISPLAY_NAME_MAPPINGS.update({
    "H3VMMasterLoader": "H3VM Multi-GPU Loader｜H3多卡加载器",
    "H3VMCoreEngine": "H3VM Core｜多卡执行引擎",
    "H3VMStyleLoRAStack": "H3VM Style LoRA Stack｜普通LoRA叠加",
})
LAB_NODE_CLASS_MAPPINGS = _base.LAB_NODE_CLASS_MAPPINGS
LAB_NODE_DISPLAY_NAME_MAPPINGS = _base.LAB_NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = dict(PUBLIC_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(PUBLIC_NODE_DISPLAY_NAME_MAPPINGS)
if H3VM_SHOW_LAB_NODES:
    NODE_CLASS_MAPPINGS.update(LAB_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(LAB_NODE_DISPLAY_NAME_MAPPINGS)

print("[H3VM v0.20.0-rc6] Master Loader + public MODEL->MODEL Core + Windows multi-GPU compatibility guard ready", flush=True)
