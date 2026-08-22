"""Mode-independent standard LoRA support for H3VM.

This layer is intentionally separate from Turbo/accelerator LoRAs.  It mirrors
ComfyUI's LoraLoaderModelOnly contract (ordinary ModelPatcher weight patches),
then replays the resulting patch state into every H3VM execution patcher that
owns real H3 weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import os


NONE_LORA = "NONE｜不使用"


@dataclass(frozen=True)
class StyleLoRASpec:
    name: str
    strength: float = 1.0


def is_acceleration_lora(name: str) -> bool:
    """Known H3 accelerator adapters must stay on the accelerator path."""
    n = os.path.basename(str(name)).lower()
    known = (
        "minimax_h3_turbo_v4_step600",
        "minimax_h3_fl2v_turbo_8step",
        "minimax_h3_fl2v_turbo_4step",
        "turbo_v4_step600",
    )
    return any(x in n for x in known)


def normalize_style_lora_stack(stack) -> tuple[StyleLoRASpec, ...]:
    if stack is None:
        return ()
    out = []
    for item in stack:
        if isinstance(item, StyleLoRASpec):
            spec = item
        elif isinstance(item, dict):
            spec = StyleLoRASpec(str(item.get("name", "")), float(item.get("strength", 1.0)))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            spec = StyleLoRASpec(str(item[0]), float(item[1]))
        else:
            raise RuntimeError(f"Invalid H3VM style LoRA stack entry: {item!r}")
        name = str(spec.name).strip()
        strength = float(spec.strength)
        if not name or name == NONE_LORA or abs(strength) < 1e-12:
            continue
        if is_acceleration_lora(name):
            raise RuntimeError(
                f"H3VM Style LoRA Stack only accepts ordinary ModelPatcher LoRAs; "
                f"{name!r} is a known H3 accelerator LoRA. Put it in the Turbo/Accelerator slot instead."
            )
        out.append(StyleLoRASpec(name, strength))
    return tuple(out)


def append_style_loras(previous, entries) -> tuple[StyleLoRASpec, ...]:
    base = list(normalize_style_lora_stack(previous))
    base.extend(normalize_style_lora_stack(entries))
    return tuple(base)


def _copy_patch_dict(patches):
    return {key: list(values) for key, values in (patches or {}).items()}


def snapshot_weight_patch_state(patcher) -> dict:
    return _copy_patch_dict(getattr(patcher, "patches", {}) or {})


def merge_weight_patch_state(target, patch_state: dict) -> int:
    """Append captured weight patches that physically exist in ``target``.

    H3VM subset/helper patchers intentionally contain only part of the H3 tree.
    Filtering by the target state_dict prevents fixed-layer patches from being
    copied into block-only islands and prevents block patches from being copied
    into helpers that do not own those weights.
    """
    if target is None or not patch_state:
        return 0
    try:
        target_keys = set(target.model.state_dict().keys())
    except Exception:
        target_keys = None
    current = _copy_patch_dict(getattr(target, "patches", {}) or {})
    added = 0
    for key, values in patch_state.items():
        if target_keys is not None and key not in target_keys:
            continue
        bucket = current.setdefault(key, [])
        bucket.extend(list(values))
        added += len(values)
    target.patches = current
    return int(added)




def filter_weight_patch_state_inplace(target) -> int:
    """Drop patch keys no longer physically present after H3VM proxy partitioning."""
    if target is None:
        return 0
    current = _copy_patch_dict(getattr(target, "patches", {}) or {})
    if not current:
        return 0
    keys = set(target.model.state_dict().keys())
    target.patches = {k: v for k, v in current.items() if k in keys}
    return sum(len(v) for v in target.patches.values())

def apply_style_stack_to_private_patcher(patcher, stack):
    """Apply ordinary LoRAs before H3VM partitions the model.

    Uses the same public ComfyUI path as LoraLoaderModelOnly.  The returned
    patcher is still a normal ModelPatcher whose ``patches`` dict can be safely
    replayed to H3VM subset/helper patchers.
    """
    specs = normalize_style_lora_stack(stack)
    if not specs:
        return patcher, {}, ()

    import folder_paths
    import comfy.sd
    import comfy.utils

    model = patcher
    for spec in specs:
        try:
            path = folder_paths.get_full_path_or_raise("loras", spec.name)
        except AttributeError:
            path = folder_paths.get_full_path("loras", spec.name)
            if path is None:
                raise RuntimeError(f"H3VM cannot find style LoRA: {spec.name}")
        try:
            raw, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        except TypeError:
            raw = comfy.utils.load_torch_file(path, safe_load=True)
            metadata = None
        try:
            model, _ = comfy.sd.load_lora_for_models(
                model, None, raw, float(spec.strength), 0.0, lora_metadata=metadata
            )
        except TypeError:
            model, _ = comfy.sd.load_lora_for_models(model, None, raw, float(spec.strength), 0.0)

    # Standard LoraLoaderModelOnly should only add ordinary weight patch state.
    # Fail closed if a custom LoRA loader unexpectedly introduced runtime hooks.
    dirty = []
    for attr, label in (
        ("injections", "runtime injections"),
        ("object_patches", "object patches"),
        ("weight_wrapper_patches", "weight-wrapper patches"),
        ("hook_patches", "hook patches"),
    ):
        if getattr(model, attr, None):
            dirty.append(label)
    if dirty:
        raise RuntimeError(
            "H3VM ordinary Style LoRA path produced unsupported runtime state: " + ", ".join(dirty)
        )

    state = snapshot_weight_patch_state(model)
    return model, state, specs


def stack_summary(stack) -> str:
    specs = normalize_style_lora_stack(stack)
    if not specs:
        return "OFF"
    return " + ".join(f"{os.path.basename(x.name)}@{x.strength:g}" for x in specs)
