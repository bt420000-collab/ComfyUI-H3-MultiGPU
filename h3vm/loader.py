"""rc5 compatibility wrapper around the frozen rc1 H3VM runtime.

The heavy proven runtime stays byte-for-byte in loader_base.py.  This module only
adds ordinary Style/character LoRA replay and keeps all scheduling algorithms in
the base runtime untouched.
"""
from __future__ import annotations

from .loader_base import *  # noqa: F401,F403
from . import loader_base as _base

# ``from ... import *`` intentionally skips underscore-prefixed names.  The
# public MODEL->MODEL Core adapter imports these helpers from this compatibility
# wrapper, so re-export only the small private surface it actually needs.  This
# does not alter loader_base or any scheduling/runtime algorithm.
_resolve_device = _base._resolve_device
_validate_h3 = _base._validate_h3
_load_private_h3 = _base._load_private_h3

_base_stream = _base.build_h3_streaming_exact_turbo
_base_snapshot = _base.build_h3_snapshot_islands_full_throttle


def _attachment(model, key):
    return (getattr(model, "attachments", {}) or {}).get(key)


def _streaming_private_loader_bridge():
    """Forward a temporary wrapper-level private loader into loader_base.

    The public Core prebuilt-MODEL adapter temporarily replaces this module's
    ``_load_private_h3``.  The frozen streaming builder itself lives in
    ``loader_base.py`` and therefore resolves its own module global.  Only when
    the wrapper helper has actually been replaced do we mirror it into
    loader_base for the duration of this one build, then restore it immediately.
    Ordinary Master Loader calls take the no-op path.
    """
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
    """Patch base _load_private_h3 so Style LoRA exists before island creation."""
    from contextlib import contextmanager
    from .style_lora import apply_style_stack_to_private_patcher

    @contextmanager
    def bridge():
        original = _base._load_private_h3
        captured = {"state": {}, "specs": ()}

        def load_private(unet_name, primary, safe_profile):
            patcher, dm = original(unet_name, primary, safe_profile)
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
    targets = [
        _attachment(model, "h3vm_dev12_primary_stream"),
        _attachment(model, "h3vm_dev12_secondary_stream"),
    ]
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
    if not style_lora_stack:
        return _base_snapshot(*args, **kwargs)
    from .style_lora import stack_summary
    with _style_loader_bridge(style_lora_stack) as captured:
        model = _base_snapshot(*args, **kwargs)
    _replay_snapshot_style(model, captured["state"])
    print(f"[H3VM Style LoRA] mode=SnapshotFullThrottle stack={stack_summary(captured['specs'])}", flush=True)
    return model
