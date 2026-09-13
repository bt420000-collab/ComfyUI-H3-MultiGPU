"""H3VM rc5 public Core engine.

Keeps the rc1 execution algorithms frozen in core_adapter_base.py / loader_base.py
and adds the product boundary agreed for rc5: standalone Master Loader remains a
cockpit; external workflows receive a pure MODEL -> MODEL execution engine.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import os
import sys
import threading

from . import core_adapter_base as _base

_CORE_BUILD_LOCK = threading.RLock()
_capacity_vram_plan = _base._capacity_vram_plan


def _runtime_ready(model):
    from .compiler_guard import install_model_compiler_guard
    return install_model_compiler_guard(model)


@dataclass(frozen=True)
class H3VMCoreConfig:
    mode: str = "DUAL_QUIET"
    primary_device: str = "gpu:0"
    secondary_device: str = "gpu:1"
    capacity_vram_profile: str = "SAFE｜保守·最稳"
    expected_steps: int = 4
    telemetry: bool = True
    capacity_mlp_chunk_rows: int = 4096
    capacity_outproj_chunk_rows: int = 4096
    capacity_attention_kernel: str = "INT8_CURRENT"
    mode4_predictor: str = "LINEAR｜标准稳定"
    secondary_participation: float = 100.0
    public_controls: bool = False


def _normalize_mode(mode):
    from .master_console import normalize_mode
    return normalize_mode(mode)


def _base_config(config):
    return _base.H3VMCoreConfig(
        mode=_normalize_mode(config.mode),
        primary_device=config.primary_device,
        secondary_device=config.secondary_device,
        capacity_vram_profile=config.capacity_vram_profile,
        expected_steps=max(1, int(config.expected_steps)),
        telemetry=bool(config.telemetry),
        capacity_mlp_chunk_rows=int(config.capacity_mlp_chunk_rows),
        capacity_outproj_chunk_rows=int(config.capacity_outproj_chunk_rows),
        capacity_attention_kernel=str(config.capacity_attention_kernel),
        secondary_participation=float(getattr(config, "secondary_participation", 100.0)),
        public_controls=bool(getattr(config, "public_controls", False)),
    )


def execution_kwargs(config):
    mode = _normalize_mode(config.mode)
    if mode == "DUAL_SYNC_ACCEL":
        raise RuntimeError("Mode4 uses the dedicated Snapshot FullThrottle engine")
    if mode == "DUAL_EXACT_SP":
        raise RuntimeError("Exact-SP uses the dedicated paired-block runtime")
    original_plan = _base._capacity_vram_plan
    _base._capacity_vram_plan = _capacity_vram_plan
    try:
        return _base.execution_kwargs(_base_config(config))
    finally:
        _base._capacity_vram_plan = original_plan


def _streaming_host_policy():
    """Respect ComfyUI's global pinned-memory policy for exact streaming modes.

    The public runtime may use a small explicit pinned runway only when ComfyUI
    itself has disabled global pinned memory. Otherwise H3VM leaves host pinning
    ownership to ComfyUI and keeps its own helper path pageable.
    """
    pinned_disabled = True
    try:
        from comfy.cli_args import args
        pinned_disabled = bool(getattr(args, "disable_pinned_memory", False))
    except Exception:
        pinned_disabled = True
    return {
        "safe_profile": bool(pinned_disabled),
        "explicit_pinned_allowed": bool(pinned_disabled),
        "policy": "h3vm_bounded_pinned" if pinned_disabled else "comfy_global_pinned+pageable_h3vm",
    }


def _common_builder_kwargs(config):
    host = _streaming_host_policy()
    print(
        f"[H3VM CORE] streaming host policy | {host['policy']} | safe_profile={host['safe_profile']}",
        flush=True,
    )
    return dict(
        primary_device=str(config.primary_device), secondary_device=str(config.secondary_device),
        stripe_size=50, expected_steps=max(1, int(config.expected_steps)), safe_profile=bool(host["safe_profile"]),
        hard_cleanup_after_sample=False, telemetry=bool(config.telemetry),
    )


def _mode4_fullthrottle_host_policy():
    """Choose a host feeder that is safe on native Windows multi-GPU.

    The locally validated dual-16G path keeps Mode4 pageable on Windows rather
    than silently re-enabling an explicit pinned mailbox after ComfyUI disabled
    global pinned memory. Non-Windows retains the historical bounded-pinned path
    when global pinning is disabled.
    """
    pinned_disabled = True
    try:
        from comfy.cli_args import args
        pinned_disabled = bool(getattr(args, "disable_pinned_memory", False))
    except Exception:
        pinned_disabled = True

    if sys.platform == "win32":
        return dict(
            safe_profile=bool(pinned_disabled), host_feeder="pageable", pinned_mailbox_mb=0,
            policy="windows_multigpu_pageable_guard",
        )
    if pinned_disabled:
        return dict(
            safe_profile=True, host_feeder="bounded_pinned", pinned_mailbox_mb=1024,
            policy="h3vm_bounded_pinned",
        )
    return dict(
        safe_profile=False, host_feeder="pageable", pinned_mailbox_mb=0,
        policy="comfy_global_pinned+pageable_h3vm",
    )


def _choice_key(value):
    return str(value).strip().split("｜", 1)[0].strip().lower()


def _mode4_predictor_config(config):
    """Resolve the public LAB7 pair to locked, validated runtime profiles."""
    mode = {
        "linear": "linear_raw",
        "spectral": "spectral_raw",
        # Preserve explicit values from the sealed LAB branch for workflow compatibility.
        "linear_raw": "linear_raw",
        "spectral_raw": "spectral_raw",
    }.get(_choice_key(getattr(config, "mode4_predictor", "LINEAR")), "linear_raw")
    return {
        "predictor_mode": mode,
        "predictor_beta": 0.75,
        "spectral_degree": 2,
        "spectral_history": 4,
        "spectral_ridge": 0.020,
        "spectral_mix": 0.50,
        "spectral_max_delta_ratio": 1.25,
        "spectral_adapt": True,
        "spectral_coordinate": "timestep",
        "spectral_confidence": "off",
        "spectral_debug": "summary",
    }


def build_asset_model(*, unet_name, turbo_lora_name, turbo_strength=1.0,
                      turbo_low_vram=False, config, style_lora_stack=None):
    from .loader import (
        build_h3_exact_sp_lab,
        build_h3_streaming_exact_turbo,
        build_h3_snapshot_islands_full_throttle,
    )
    mode = _normalize_mode(config.mode)
    from .master_console import STOCK_LORA
    accelerator_lora = None if str(turbo_lora_name) == STOCK_LORA else str(turbo_lora_name)
    legacy_exact = os.environ.get("H3VM_ENABLE_EXACT_SP_LAB", "0") == "1"
    if legacy_exact and mode != "DUAL_EXACT_SP":
        raise RuntimeError(
            "H3VM_ENABLE_EXACT_SP_LAB=1 no longer hijacks another backend. "
            "Select DUAL_EXACT_SP explicitly in H3VM VRAM Master Loader. "
            "This fail-closed change prevents a Capacity/Turbo workflow from silently "
            "running Exact-SP with the wrong sampler contract."
        )

    print(f"[H3VM CORE] asset model path | mode={mode}", flush=True)
    if mode == "DUAL_EXACT_SP":
        # Native head-sharded Exact-SP is used for an unpatched Stock model.
        # Accelerator/Style/PDD-compatible patched models use the proven exact
        # streaming fabric: it preserves every block and the workflow's sampling
        # math, but does not require sharding adapter output rows.
        if accelerator_lora or style_lora_stack:
            compat = replace(config, mode="DUAL_QUIET")
            print(
                "[H3VM CORE] Mode5 exact compatibility path | adapter patches preserved | "
                "no predictor/stale reuse",
                flush=True,
            )
            return _runtime_ready(build_h3_streaming_exact_turbo(
                unet_name=str(unet_name), turbo_lora_name=accelerator_lora,
                turbo_strength=float(turbo_strength), turbo_low_vram=bool(turbo_low_vram),
                style_lora_stack=style_lora_stack,
                **_common_builder_kwargs(compat), **execution_kwargs(compat),
            ))
        host = _streaming_host_policy()
        print(
            "[H3VM CORE] Mode5 native Exact-SP | Stock weights | workflow steps preserved",
            flush=True,
        )
        return _runtime_ready(build_h3_exact_sp_lab(
            unet_name=str(unet_name),
            primary_device=str(config.primary_device),
            secondary_device=str(config.secondary_device),
            expected_steps=max(1, int(config.expected_steps)),
            transport_mode="auto",
            safe_profile=bool(host["safe_profile"]),
            telemetry=bool(config.telemetry),
        ))
    if mode == "DUAL_SYNC_ACCEL":
        hp = _mode4_fullthrottle_host_policy()
        predictor = _mode4_predictor_config(config)
        print(
            f"[H3VM CORE] Mode4 Predictor-V7 LAB7 | profile={predictor['predictor_mode']} "
            f"coordinate=timestep degree=2 history=4 ridge=0.020 confidence=off",
            flush=True,
        )
        print(f"[H3VM CORE] Mode4 host policy | {hp['policy']} | feeder={hp['host_feeder']} pinned_cap={hp['pinned_mailbox_mb']}MiB", flush=True)
        return _runtime_ready(build_h3_snapshot_islands_full_throttle(
            unet_name=str(unet_name), primary_device=str(config.primary_device), secondary_device=str(config.secondary_device),
            secondary_blocks_target=25, primary_reserve_gb=2.5, secondary_reserve_gb=1.5,
            secondary_overcommit_mb=1024, safe_profile=bool(hp["safe_profile"]), expected_steps=max(1, int(config.expected_steps)),
            refresh_interval=0, exact_last_step=True, **predictor,
            prefix_prefetch=True, tail_prefetch=True, launch_tail_before_stage=True,
            host_feeder=str(hp["host_feeder"]), pinned_mailbox_mb=int(hp["pinned_mailbox_mb"]),
            secondary_ram_backing_gb=2.0, telemetry=bool(config.telemetry), turbo_lora_name=accelerator_lora,
            turbo_strength=float(turbo_strength), turbo_low_vram=bool(turbo_low_vram),
            style_lora_stack=style_lora_stack,
        ))
    return _runtime_ready(build_h3_streaming_exact_turbo(
        unet_name=str(unet_name), turbo_lora_name=accelerator_lora,
        turbo_strength=float(turbo_strength), turbo_low_vram=bool(turbo_low_vram),
        style_lora_stack=style_lora_stack,
        **_common_builder_kwargs(config), **execution_kwargs(config),
    ))


def _filtered_copy(source, target):
    from .style_lora import merge_weight_patch_state
    target.patches = {}
    count = merge_weight_patch_state(target, getattr(source, "patches", {}) or {})
    if hasattr(source, "patches_uuid"):
        target.patches_uuid = source.patches_uuid
    if hasattr(source, "force_cast_weights"):
        target.force_cast_weights = source.force_cast_weights
    return int(count)


@contextmanager
def _filtered_prebuilt_replay():
    original = _base._copy_weight_patch_state
    _base._copy_weight_patch_state = _filtered_copy
    try:
        yield
    finally:
        _base._copy_weight_patch_state = original


@contextmanager
def _prebuilt_snapshot_bridge(private):
    from . import loader_base
    original = loader_base._load_private_h3
    def load_private(_name, _primary, _safe, local_no_pin=False):
        # The private patcher already exists in the prebuilt MODEL path.
        # Accept the loader_base interface flag so Mode4 pageable/no-pin
        # builds can cross this bridge without signature drift.
        del local_no_pin
        return private, loader_base._validate_h3(private)
    loader_base._load_private_h3 = load_private
    try:
        yield
    finally:
        loader_base._load_private_h3 = original


def adapt_model(model, *, config):
    mode = _normalize_mode(config.mode)
    if mode == "DUAL_EXACT_SP":
        # PDD and other experimental accelerators arrive through MODEL->MODEL as
        # ordinary prebuilt patch state.  Mode5 keeps them on an exact 50-block
        # execution path; only Mode4 enables predictor/stale-boundary reuse.
        compat = replace(config, mode="DUAL_QUIET")
        print(
            "[H3VM CORE] Mode5 prebuilt compatibility path | exact execution | "
            "PDD/LoRA patch state is not policy-blocked",
            flush=True,
        )
        out = _base.adapt_model(model, config=_base_config(compat))
        try:
            out.set_attachments("h3vm_mode5_compat", {
                "requested_mode": "DUAL_EXACT_SP",
                "runtime": "RAM_FIRST_EXACT_NO_PREDICTOR",
            })
        except Exception:
            pass
        return _runtime_ready(out)
    if mode != "DUAL_SYNC_ACCEL":
        with _filtered_prebuilt_replay():
            return _runtime_ready(_base.adapt_model(model, config=_base_config(config)))

    import torch
    from .loader import _resolve_device, _validate_h3, build_h3_snapshot_islands_full_throttle, _replay_snapshot_style
    unsupported = _base._unsupported_prebuilt_state(model)
    if unsupported:
        raise RuntimeError(
            "H3VM Core prebuilt MODEL currently supports ordinary weight-patch LoRA state only. Unsupported state: "
            + ", ".join(unsupported)
        )
    if not hasattr(model, "deepclone_multigpu") or getattr(model, "cached_patcher_init", None) is None:
        raise RuntimeError("This MODEL cannot be safely deep-cloned for H3VM Core")
    primary = _resolve_device(config.primary_device)
    private = model.deepclone_multigpu(new_load_device=primary)
    private.offload_device = torch.device("cpu")
    if hasattr(private, "remove_additional_models"):
        private.remove_additional_models("multigpu")
    _validate_h3(private)
    patch_state = {k: list(v) for k, v in (getattr(private, "patches", {}) or {}).items()}
    hp = _mode4_fullthrottle_host_policy()
    predictor = _mode4_predictor_config(config)
    print(
        f"[H3VM CORE] prebuilt Mode4 Predictor-V7 LAB7 | steps_hint={max(1,int(config.expected_steps))} | "
        f"profile={predictor['predictor_mode']} coordinate=timestep | "
        f"host={hp['policy']} | workflow sampler/sigmas unchanged", flush=True,
    )
    with _CORE_BUILD_LOCK, _prebuilt_snapshot_bridge(private):
        out = build_h3_snapshot_islands_full_throttle(
            unet_name="<prebuilt-model>", primary_device=str(config.primary_device), secondary_device=str(config.secondary_device),
            secondary_blocks_target=25, primary_reserve_gb=2.5, secondary_reserve_gb=1.5,
            secondary_overcommit_mb=1024, safe_profile=bool(hp["safe_profile"]),
            expected_steps=max(1, int(config.expected_steps)), refresh_interval=0, exact_last_step=True,
            **predictor, prefix_prefetch=True, tail_prefetch=True,
            launch_tail_before_stage=True, host_feeder=str(hp["host_feeder"]),
            pinned_mailbox_mb=int(hp["pinned_mailbox_mb"]), secondary_ram_backing_gb=2.0,
            telemetry=bool(config.telemetry), turbo_lora_name=None, style_lora_stack=None,
        )
    _replay_snapshot_style(out, patch_state)
    try:
        out.set_attachments("h3vm_core_adapter", {
            "mode": mode, "source": "prebuilt_model", "weight_patch_entries": sum(len(v) for v in patch_state.values()),
            "capacity_vram_profile": str(config.capacity_vram_profile),
        })
    except Exception:
        pass
    return _runtime_ready(out)


# Compatibility exports for the lightweight contract test / downstream dev tools.
_copy_weight_patch_state = _filtered_copy
