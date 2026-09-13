"""H3VM v0.21.1 compatibility wrapper around the proven runtime."""
from __future__ import annotations

from .loader_base import *  # noqa: F401,F403
from . import loader_base as _base

_resolve_device = _base._resolve_device
_validate_h3 = _base._validate_h3
_load_private_h3 = _base._load_private_h3
_base_stream = _base.build_h3_streaming_exact_turbo
_base_snapshot = _base.build_h3_snapshot_islands_full_throttle


def _attachment(model, key):
    return (getattr(model, "attachments", {}) or {}).get(key)


def _streaming_private_loader_bridge():
    from contextlib import contextmanager
    @contextmanager
    def bridge():
        original = _base._load_private_h3
        replacement = globals().get("_load_private_h3", original)
        if replacement is original:
            yield
            return
        _base._load_private_h3 = replacement
        try:
            yield
        finally:
            _base._load_private_h3 = original
    return bridge()


def _style_loader_bridge(style_lora_stack):
    from contextlib import contextmanager
    from .style_lora import apply_style_stack_to_private_patcher
    @contextmanager
    def bridge():
        original = _base._load_private_h3
        captured = {"state": {}, "specs": ()}
        def load_private(unet_name, primary, safe_profile, *args, **kwargs):
            patcher, dm = original(unet_name, primary, safe_profile, *args, **kwargs)
            patcher, state, specs = apply_style_stack_to_private_patcher(patcher, style_lora_stack)
            captured["state"] = state
            captured["specs"] = specs
            return patcher, _base._validate_h3(patcher)
        _base._load_private_h3 = load_private
        try:
            yield captured
        finally:
            _base._load_private_h3 = original
    return bridge()


def _replay_streaming_style(model, patch_state):
    if not patch_state:
        return
    from .style_lora import merge_weight_patch_state, filter_weight_patch_state_inplace
    targets = [_attachment(model, "h3vm_dev12_primary_stream"), _attachment(model, "h3vm_dev12_secondary_stream")]
    helpers = _attachment(model, "h3vm_dev12_3_mlp_helpers") or {}
    targets += [helpers.get("primary"), helpers.get("secondary")]
    count = sum(merge_weight_patch_state(t, patch_state) for t in targets if t is not None)
    main_count = filter_weight_patch_state_inplace(model)
    print(f"[H3VM Style LoRA replay] streaming helper_entries={count} main_entries={main_count}", flush=True)


def _replay_snapshot_style(model, patch_state):
    if not patch_state:
        return
    from .style_lora import merge_weight_patch_state, filter_weight_patch_state_inplace
    count = 0
    for key in ("h3vm_snapshot_prefix", "h3vm_snapshot_tail"):
        target = _attachment(model, key)
        if target is not None:
            count += merge_weight_patch_state(target, patch_state)
    main_count = filter_weight_patch_state_inplace(model)
    print(f"[H3VM Style LoRA replay] snapshot island_entries={count} main_entries={main_count}", flush=True)


def build_h3_streaming_exact_turbo(*args, style_lora_stack=None, **kwargs):
    with _streaming_private_loader_bridge():
        if not style_lora_stack:
            return _base_stream(*args, **kwargs)
        from .style_lora import stack_summary
        with _style_loader_bridge(style_lora_stack) as captured:
            model = _base_stream(*args, **kwargs)
    _replay_streaming_style(model, captured["state"])
    print(f"[H3VM Style LoRA] mode=StreamingExact stack={stack_summary(captured['specs'])}", flush=True)
    return model


def build_h3_snapshot_islands_full_throttle(*args, style_lora_stack=None, **kwargs):
    """Accept v0.21.1 predictor controls while reusing the proven Mode4 builder."""
    from .v0211_mode4 import install_snapshot_runtime_patch, configure_mode4_patcher
    install_snapshot_runtime_patch()
    requested = str(kwargs.get("predictor_mode", "linear_raw"))
    ridge = float(kwargs.pop("spectral_ridge", 0.02))
    for key in (
        "spectral_degree", "spectral_history", "spectral_mix", "spectral_max_delta_ratio",
        "spectral_adapt", "spectral_coordinate", "spectral_confidence", "spectral_debug",
    ):
        kwargs.pop(key, None)
    kwargs["predictor_mode"] = "linear" if requested.lower() not in ("stale",) else "stale"

    if not style_lora_stack:
        model = _base_snapshot(*args, **kwargs)
    else:
        from .style_lora import stack_summary
        with _style_loader_bridge(style_lora_stack) as captured:
            model = _base_snapshot(*args, **kwargs)
        _replay_snapshot_style(model, captured["state"])
        print(f"[H3VM Style LoRA] mode=SnapshotFullThrottle stack={stack_summary(captured['specs'])}", flush=True)
    return configure_mode4_patcher(model, requested_predictor=requested, spectral_ridge=ridge)


def build_h3_exact_sp_lab(*args, **kwargs):
    try:
        from .exact_sp_loader import build_h3_exact_sp_lab as impl
    except Exception as exc:
        raise RuntimeError("Exact-SP is not part of the v0.21.1 public surface") from exc
    return impl(*args, **kwargs)

build_h3_exact_sp_preview = build_h3_exact_sp_lab
