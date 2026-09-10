"""H3VM rc5 public Core engine.

Keeps the rc1 execution algorithms frozen in core_adapter_base.py / loader_base.py
and adds the product boundary agreed for rc5: standalone Master Loader remains a
cockpit; external workflows receive a pure MODEL -> MODEL execution engine.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sys
import threading

from . import core_adapter_base as _base

_CORE_BUILD_LOCK = threading.RLock()


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


def _normalize_mode(mode):
    from .master_console import normalize_mode
    return normalize_mode(mode)


def _capacity_vram_plan(profile: str, primary_device: str, secondary_device: str) -> dict:
    """Public Capacity plan plus opt-in equal-card lab overlay.

    Keeping this wrapper in core_adapter preserves the rc6 base runtime while
    letting the lab branch tune 16G+16G without touching PM/private scheduling.
    """
    plan = dict(_base._capacity_vram_plan(profile, primary_device, secondary_device))
    try:
        from .dual16g_lab import lab_enabled, pair_is_near_symmetric, requested_primary_fraction
        if lab_enabled() and pair_is_near_symmetric(primary_device, secondary_device):
            plan["mlp_primary_fraction"] = float(requested_primary_fraction())
            plan["capacity_helper_heads"] = 28
    except Exception as exc:
        # A requested lab configuration should fail loudly rather than silently
        # claiming symmetric tuning. Ordinary rc6 behavior remains unaffected
        # when the lab flag is not enabled.
        from .dual16g_lab import lab_enabled
        if lab_enabled():
            raise RuntimeError(f"H3VM dual16 lab Capacity policy failed: {exc}") from exc
    return plan


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
    )


def execution_kwargs(config):
    mode = _normalize_mode(config.mode)
    if mode == "DUAL_SYNC_ACCEL":
        raise RuntimeError("Mode4 uses the dedicated Snapshot FullThrottle engine")
    original_plan = _base._capacity_vram_plan
    _base._capacity_vram_plan = _capacity_vram_plan
    try:
        kwargs = _base.execution_kwargs(_base_config(config))
    finally:
        _base._capacity_vram_plan = original_plan

    from .dual16g_lab import apply_execution_overrides
    tuned, report = apply_execution_overrides(
        kwargs,
        mode=mode,
        primary=config.primary_device,
        secondary=config.secondary_device,
    )
    if report is not None:
        if report.get("active"):
            print(
                "[H3VM DUAL16 LAB] symmetric policy ACTIVE | "
                f"mode={mode} MLP={report['primary_fraction']*100:.0f}/{(1.0-report['primary_fraction'])*100:.0f} "
                f"ATTN={report['attention_heads'][0]}/{report['attention_heads'][1]} | "
                "manual UI remains hidden",
                flush=True,
            )
        else:
            print(
                f"[H3VM DUAL16 LAB] inactive | mode={mode} reason={report.get('reason')}",
                flush=True,
            )
    return tuned


def _common_builder_kwargs(config):
    return dict(
        primary_device=str(config.primary_device), secondary_device=str(config.secondary_device),
        stripe_size=50, expected_steps=max(1, int(config.expected_steps)), safe_profile=True,
        hard_cleanup_after_sample=False, telemetry=bool(config.telemetry),
    )


def _mode4_fullthrottle_host_policy():
    pinned_disabled = True
    try:
        from comfy.cli_args import args
        pinned_disabled = bool(getattr(args, "disable_pinned_memory", False))
    except Exception:
        pass
    if pinned_disabled:
        return dict(safe_profile=True, host_feeder="bounded_pinned", pinned_mailbox_mb=1024,
                    policy="h3vm_bounded_pinned")
    return dict(safe_profile=False, host_feeder="pageable", pinned_mailbox_mb=0,
                policy="comfy_global_pinned+pageable_h3vm")


def build_asset_model(*, unet_name, turbo_lora_name, turbo_strength=1.0,
                      turbo_low_vram=False, config, style_lora_stack=None):
    from .loader import build_h3_streaming_exact_turbo, build_h3_snapshot_islands_full_throttle
    mode = _normalize_mode(config.mode)
    print(f"[H3VM CORE] asset model path | mode={mode}", flush=True)
    if mode == "DUAL_SYNC_ACCEL":
        hp = _mode4_fullthrottle_host_policy()
        print("[H3VM CORE] Mode4 Stock H3 FullThrottle | Dev9.4 22/28 RAM2G PREDICT075 | Turbo ignored", flush=True)
        print(f"[H3VM CORE] Mode4 host policy | {hp['policy']} | feeder={hp['host_feeder']} pinned_cap={hp['pinned_mailbox_mb']}MiB", flush=True)
        return build_h3_snapshot_islands_full_throttle(
            unet_name=str(unet_name), primary_device=str(config.primary_device), secondary_device=str(config.secondary_device),
            secondary_blocks_target=22, primary_reserve_gb=2.5, secondary_reserve_gb=1.5,
            secondary_overcommit_mb=1024, safe_profile=bool(hp["safe_profile"]), expected_steps=20,
            refresh_interval=0, exact_last_step=True, predictor_mode="linear", predictor_beta=0.75,
            prefix_prefetch=True, tail_prefetch=True, launch_tail_before_stage=True,
            host_feeder=str(hp["host_feeder"]), pinned_mailbox_mb=int(hp["pinned_mailbox_mb"]),
            secondary_ram_backing_gb=2.0, telemetry=bool(config.telemetry), turbo_lora_name=None,
            style_lora_stack=style_lora_stack,
        )
    return build_h3_streaming_exact_turbo(
        unet_name=str(unet_name), turbo_lora_name=str(turbo_lora_name),
        turbo_strength=float(turbo_strength), turbo_low_vram=bool(turbo_low_vram),
        style_lora_stack=style_lora_stack,
        **_common_builder_kwargs(config), **execution_kwargs(config),
    )


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
    def load_private(_name, _primary, _safe):
        return private, loader_base._validate_h3(private)
    loader_base._load_private_h3 = load_private
    try:
        yield
    finally:
        loader_base._load_private_h3 = original


def adapt_model(model, *, config):
    mode = _normalize_mode(config.mode)
    if mode != "DUAL_SYNC_ACCEL":
        with _filtered_prebuilt_replay():
            # core_adapter_base calls its own execution_kwargs, so temporarily
            # redirect its Capacity planner and then apply the symmetric runtime
            # kwargs by using the public adapter path below where possible.
            # Prebuilt model adaptation remains rc6-compatible; the dual16 lab is
            # primarily validated through the integrated/Master path first.
            return _base.adapt_model(model, config=_base_config(config))

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
    print(
        f"[H3VM CORE] prebuilt Mode4 Snapshot engine | steps_hint={max(1,int(config.expected_steps))} | "
        f"host={hp['policy']} | workflow sampler/sigmas unchanged", flush=True,
    )
    with _CORE_BUILD_LOCK, _prebuilt_snapshot_bridge(private):
        out = build_h3_snapshot_islands_full_throttle(
            unet_name="<prebuilt-model>", primary_device=str(config.primary_device), secondary_device=str(config.secondary_device),
            secondary_blocks_target=22, primary_reserve_gb=2.5, secondary_reserve_gb=1.5,
            secondary_overcommit_mb=1024, safe_profile=bool(hp["safe_profile"]),
            expected_steps=max(1, int(config.expected_steps)), refresh_interval=0, exact_last_step=True,
            predictor_mode="linear", predictor_beta=0.75, prefix_prefetch=True, tail_prefetch=True,
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
    return out


def install_runtime_bridge():
    package_name = __package__ or ""
    parent_name = package_name.rsplit(".h3vm", 1)[0] if ".h3vm" in package_name else package_name.rsplit(".", 1)[0]
    parent = sys.modules.get(parent_name)
    cls = getattr(parent, "H3VMCapacityModeTurboLoader", None) if parent is not None else None
    if cls is None:
        return False

    def core_load(self, mode, unet_name, lora_name, strength=1.0, low_vram=False,
                  primary_device="gpu:0", secondary_device="gpu:1", capacity_vram_profile="SAFE｜保守·最稳",
                  capacity_mlp_chunk_rows=4096, capacity_outproj_chunk_rows=4096, capacity_helper_heads=16,
                  capacity_attention_kernel="INT8_CURRENT", telemetry=True, expected_steps=4, style_lora_stack=None):
        del self, capacity_helper_heads
        cfg = H3VMCoreConfig(mode=str(mode), primary_device=str(primary_device), secondary_device=str(secondary_device),
                             capacity_vram_profile=str(capacity_vram_profile), expected_steps=int(expected_steps),
                             telemetry=bool(telemetry), capacity_mlp_chunk_rows=int(capacity_mlp_chunk_rows),
                             capacity_outproj_chunk_rows=int(capacity_outproj_chunk_rows),
                             capacity_attention_kernel=str(capacity_attention_kernel))
        return (build_asset_model(unet_name=str(unet_name), turbo_lora_name=str(lora_name),
                                  turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
                                  config=cfg, style_lora_stack=style_lora_stack),)

    original_load = cls.load
    cls._h3vm_original_load = original_load
    cls.load = core_load
    cls._h3vm_core_bridged = True
    print("[H3VM CORE] rc5 runtime bridge installed | Master Loader -> shared Core", flush=True)
    return True

# Compatibility exports for the lightweight contract test / downstream dev tools.
_copy_weight_patch_state = _filtered_copy
