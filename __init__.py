"""ComfyUI-H3-VRAM-Master v0.21.1 focused public release.

Public surface:
    MODEL -> H3VM Core -> MODEL + H3VM_MODE + H3VM_VAE_POLICY
    LATENT + VAE + H3VM_MODE + H3VM_VAE_POLICY -> H3VM Video VAE Decode -> IMAGE

Installing the plugin is side-effect free. Runtime patches are activated lazily only
when an H3VM execution path actually runs. Standard ComfyUI CLIP/text encoder,
VAE loader, sampler, prompt, seed, duration, and resolution controls are untouched.
"""
from __future__ import annotations

from pathlib import Path

try:
    H3VM_VERSION = Path(__file__).resolve().with_name("VERSION").read_text(encoding="utf-8").strip()
except Exception:
    H3VM_VERSION = "unknown"
__version__ = H3VM_VERSION

from .h3vm.master_console import PRODUCT_MODES, PARTICIPATION_PRESETS

MODE4_PREDICTORS = (
    "SPECTRAL｜频谱极速",
    "LINEAR｜标准极速",
)
UI_LANGUAGES = ["中文", "English"]
WEB_DIRECTORY = "./web"


class H3VMCoreEngine:
    """Pure MODEL -> MODEL H3 multi-GPU execution adapter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ui_language": (UI_LANGUAGES, {"default": "中文"}),
                "model": ("MODEL",),
                "multi_gpu_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "开启",
                        "label_off": "关闭",
                        "tooltip": "关闭后保持单卡执行；开启后使用所选 H3VM 双卡策略。",
                    },
                ),
                "dual_vae_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "开启",
                        "label_off": "关闭",
                        "tooltip": "统一控制 H3VM Video VAE 双卡解码。双卡性能差距很大时可关闭。",
                    },
                ),
                "gpu_mode": (
                    PRODUCT_MODES,
                    {
                        "default": "双卡极速",
                        "tooltip": "双卡极速：速度优先；双卡扩容：显存优先；双卡后台：保守协同。",
                    },
                ),
                "mode4_predictor": (
                    MODE4_PREDICTORS,
                    {
                        "default": "SPECTRAL｜频谱极速",
                        "tooltip": "仅双卡极速生效。频谱极速为默认；标准极速使用 LINEAR Predictor。",
                    },
                ),
                "gpu_participation": (
                    PARTICIPATION_PRESETS,
                    {
                        "default": "全｜100%",
                        "tooltip": "副卡参与强度。不同模式会映射到各自已验证的安全参数。",
                    },
                ),
                "custom_participation": (
                    "INT",
                    {
                        "default": 100,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "仅当副卡参与度选择“自定义”时生效。",
                    },
                ),
                "expected_steps_hint": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "只用于 H3VM 预取/末步质量调度提示；真实采样边界由 Sampler 自动识别。填错不会跨任务复用状态，也不会修改原工作流实际采样步数。",
                    },
                ),
                "telemetry": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "H3VM_MODE", "H3VM_VAE_POLICY")
    RETURN_NAMES = ("model", "mode", "vae_policy")
    FUNCTION = "adapt"
    CATEGORY = "MiniMaxH3/H3VM"
    DESCRIPTION = (
        "H3 VRAM Master Core：只接管 MODEL 执行与显存/双卡调度。"
        "不修改 Prompt、Seed、分辨率、时长、Sampler、Sigmas、实际 Steps、CLIP 或 VAE Loader。"
    )

    def adapt(
        self,
        model,
        ui_language="中文",
        multi_gpu_enabled=True,
        dual_vae_enabled=True,
        gpu_mode="双卡极速",
        mode4_predictor="SPECTRAL｜频谱极速",
        gpu_participation="全｜100%",
        custom_participation=100,
        expected_steps_hint=20,
        telemetry=True,
    ):
        del ui_language
        from .h3vm import activate_runtime

        activate_runtime()
        from .h3vm.master_console import resolve_public_mode, resolve_participation_percent
        from .h3vm.core_adapter import H3VMCoreConfig, adapt_model

        mode = resolve_public_mode(multi_gpu_enabled, gpu_mode)
        participation = resolve_participation_percent(gpu_participation, custom_participation)
        out = adapt_model(
            model,
            config=H3VMCoreConfig(
                mode=mode,
                primary_device="gpu:0",
                secondary_device="gpu:1",
                capacity_vram_profile="SAFE｜保守·最稳",
                expected_steps=max(1, int(expected_steps_hint)),
                telemetry=bool(telemetry),
                mode4_predictor=str(mode4_predictor),
                secondary_participation=float(participation),
                public_controls=True,
            ),
        )
        vae_policy = {
            "dual_vae_enabled": bool(dual_vae_enabled) and mode != "SINGLE_GPU",
        }
        return out, mode, vae_policy


class H3VMModeAwareVideoVAEDecode:
    """MiniMax H3 Video VAE decoder controlled by the H3VM Core VAE policy."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "ui_language": (UI_LANGUAGES, {"default": "中文"}),
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "mode": ("H3VM_MODE",),
                "vae_policy": ("H3VM_VAE_POLICY",),
                "vae_name": (
                    folder_paths.get_filename_list("vae"),
                    {
                        "tooltip": "副卡加载的 Video VAE 文件。必须与连接到 vae 输入的 MiniMax H3 Video VAE 使用同一权重；H3VM 会在可验证时检查并拒绝明显不一致。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "MiniMaxH3/H3VM"
    DESCRIPTION = (
        "MiniMax H3 Video VAE 可选双卡加速。双卡开关由 H3VM Core 统一控制；"
        "策略关闭或 Core 为单卡模式时直接使用原生单卡 VAE 解码。"
        "双卡模式下 vae_name 必须与连接的主 VAE 使用同一权重。"
    )

    @staticmethod
    def _native_decode(vae, samples):
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        images = vae.decode(latent)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

    def decode(
        self,
        samples,
        vae,
        mode,
        vae_policy,
        vae_name,
        ui_language="中文",
    ):
        del ui_language
        mode_text = str(mode).strip()
        policy = vae_policy if isinstance(vae_policy, dict) else {}
        use_dual = bool(policy.get("dual_vae_enabled", False)) and not mode_text.startswith("SINGLE_GPU")
        if not use_dual:
            from .h3vm import runtime_active
            if runtime_active():
                from .h3vm.dual_video_vae import _cleanup_h3_runtime
                _cleanup_h3_runtime()
            return (self._native_decode(vae, samples),)

        from .h3vm import activate_runtime
        activate_runtime()
        from .h3vm.dual_video_vae import dual_decode_h3_video

        return (
            dual_decode_h3_video(
                vae=vae,
                samples=samples,
                vae_name=str(vae_name),
                primary_device="gpu:0",
                secondary_device="gpu:1",
                cleanup_h3=True,
                telemetry=True,
                post_cleanup=True,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "H3VMCoreEngine": H3VMCoreEngine,
    "H3VMModeAwareVideoVAEDecode": H3VMModeAwareVideoVAEDecode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3VMCoreEngine": "H3VM Core｜模型执行引擎",
    "H3VMModeAwareVideoVAEDecode": "H3VM Video VAE｜双卡加速解码",
}

print(
    f"[ComfyUI-H3-VRAM-Master {H3VM_VERSION}] focused plugin ready | runtime lazy | Core + optional Dual Video VAE",
    flush=True,
)
