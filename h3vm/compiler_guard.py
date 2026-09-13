from __future__ import annotations

"""Local compatibility guard for ComfyUI's allocation compiler.

H3VM intentionally crosses CUDA devices inside one logical H3 block.  Comfy's
allocation graph is recorded for the root device and cannot legally capture
those secondary-device allocations/copies.  Pause only around H3VM proxy work;
the compiler remains enabled for every other node and model.
"""

from contextlib import contextmanager, nullcontext
from functools import wraps
from threading import RLock


_PATCHER_ISLAND_LOCK = RLock()


@contextmanager
def pause_comfy_allocation_graph():
    try:
        import comfy.model_prefetch as model_prefetch
        guard = model_prefetch.pause_malloc_graph(sync=True)
    except Exception:
        guard = nullcontext()
    with guard:
        yield


@contextmanager
def legacy_model_patcher_island():
    """Create one legacy, locally no-pin model while AIMDO stays global."""
    import comfy.model_patcher as model_patcher

    with _PATCHER_ISLAND_LOCK:
        original_core = model_patcher.CoreModelPatcher
        original_legacy = model_patcher.ModelPatcher
        local = getattr(model_patcher, "_H3VMNoPinModelPatcher", None)
        if local is None:
            class H3VMNoPinModelPatcher(original_legacy):
                def pin_weight_to_device(self, key):
                    del key
                    return False

                def unpin_weight(self, key):
                    del key
                    return None

                def unpin_all_weights(self):
                    return None

            H3VMNoPinModelPatcher.__name__ = "H3VMNoPinModelPatcher"
            local = H3VMNoPinModelPatcher
            model_patcher._H3VMNoPinModelPatcher = local
        model_patcher.ModelPatcher = local
        model_patcher.CoreModelPatcher = local
        try:
            yield
        finally:
            model_patcher.ModelPatcher = original_legacy
            model_patcher.CoreModelPatcher = original_core


def install_model_compiler_guard(patcher):
    """Pause Comfy allocation graphs for the complete H3 diffusion call."""
    key = "h3vm_cross_device_allocation_guard"
    attachments = getattr(patcher, "attachments", {}) or {}
    if attachments.get(key):
        return patcher

    import comfy.patcher_extension

    def diffusion_wrapper(executor, *args, **kwargs):
        with pause_comfy_allocation_graph():
            return executor(*args, **kwargs)

    patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        key,
        diffusion_wrapper,
    )
    try:
        patcher.set_attachments(key, True)
    except Exception:
        pass
    return patcher


def install_minimax_text_encoder_compat():
    try:
        import folder_paths
        import nodes
        import torch
        import comfy.sd
    except Exception:
        return False
    loader = nodes.CLIPLoader
    original = loader.load_clip
    if getattr(original, "_h3vm_minimax_compat", False):
        return True
    @wraps(original)
    def load_clip(self, clip_name, type="stable_diffusion", device="default"):
        if str(type).strip().lower() != "minimax":
            return original(self, clip_name, type=type, device=device)
        clip_type = getattr(comfy.sd.CLIPType, "MINIMAX", comfy.sd.CLIPType.STABLE_DIFFUSION)
        model_options = {}
        if device == "cpu":
            cpu = torch.device("cpu"); model_options["load_device"] = cpu; model_options["offload_device"] = cpu
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        kwargs = dict(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
        try:
            with legacy_model_patcher_island():
                clip = comfy.sd.load_clip(disable_dynamic=True, **kwargs)
            print("[H3VM] MiniMax H3 text encoder | per-model DynamicVRAM bypass=enabled", flush=True)
            return (clip,)
        except TypeError:
            return original(self, clip_name, type=type, device=device)
    load_clip._h3vm_minimax_compat = True
    load_clip._h3vm_original = original
    loader.load_clip = load_clip
    return True


def install_minimax_video_vae_compat():
    try:
        import nodes
    except Exception:
        return False
    loader = nodes.VAELoader
    original = loader.load_vae
    if getattr(original, "_h3vm_minimax_vae_compat", False):
        return True
    @wraps(original)
    def load_vae(self, vae_name):
        name = str(vae_name).strip().lower()
        if "minimax_h3_video_vae" not in name:
            return original(self, vae_name)
        with legacy_model_patcher_island():
            result = original(self, vae_name)
        print("[H3VM] MiniMax H3 video VAE | per-model DynamicVRAM bypass=enabled", flush=True)
        return result
    load_vae._h3vm_minimax_vae_compat = True
    load_vae._h3vm_original = original
    loader.load_vae = load_vae
    return True
