"""Shared H3VM execution core.

This module separates *how a MODEL is obtained* from *how H3VM executes it*.
The product Master Loader keeps its convenient model/Turbo/prompt controls,
while advanced workflows can hand an already-built ComfyUI MODEL to the same
H3VM placement policy.

The first generic prebuilt-MODEL contract intentionally supports standard
ModelPatcher weight patches (the common LoraLoaderModelOnly path) and fails
closed for runtime injections/object/hook/weight-wrapper patches. Those richer
mechanisms can hold direct module references and need explicit per-device helper
remapping before H3VM can claim exact arithmetic.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sys
import threading


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


def _normalize_mode(mode: str) -> str:
    from .master_console import normalize_mode
    return normalize_mode(mode)


def _capacity_vram_plan(profile: str, primary_device: str, secondary_device: str) -> dict:
    import torch
    from .loader import _resolve_device
    from .master_console import resolve_capacity_vram_profile

    primary = _resolve_device(primary_device)
    secondary = _resolve_device(secondary_device)
    pgib = float(torch.cuda.get_device_properties(primary).total_memory) / float(1024 ** 3)
    sgib = float(torch.cuda.get_device_properties(secondary).total_memory) / float(1024 ** 3)
    return resolve_capacity_vram_profile(str(profile), pgib, sgib)


def execution_kwargs(config: H3VMCoreConfig) -> dict:
    """Single source of truth for public H3VM execution-mode policy."""
    mode = _normalize_mode(config.mode)
    if mode == "DUAL_SYNC_ACCEL":
        raise RuntimeError(
            "H3VM DUAL_SYNC_ACCEL is a reserved backend slot. Install/wire a compatible "
            "same-model synchronous multi-GPU backend before selecting this mode."
        )

    if mode == "SINGLE_GPU":
        return dict(
            primary_runtime_reserve_gb=5.25, secondary_runtime_reserve_gb=2.5,
            primary_hot_cache_gb=5.0, secondary_hot_cache_gb=0.0,
            one_ahead_prefetch=True, trim_on_stripe_boundary=True,
            attention_mode="off", attention_head_balance="sm_weighted",
            attention_min_sequence_length=8192, attention_relay_min_gbps=0.0,
            attention_helper_head_cap=16, attention_helper_safety_mb=1024,
            attention_host_ring_mb=64,
            mlp_token_parallel=False, mlp_primary_fraction=1.0,
            mlp_min_sequence_length=8192, mlp_helper_safety_mb=768,
            single_root=True, single_root_trim_interval=20,
            critical_path_mode=False,
        )

    if mode == "DUAL_QUIET":
        return dict(
            primary_runtime_reserve_gb=5.25, secondary_runtime_reserve_gb=2.5,
            primary_hot_cache_gb=5.0, secondary_hot_cache_gb=3.5,
            one_ahead_prefetch=True, trim_on_stripe_boundary=True,
            attention_mode="force_host", attention_head_balance="sm_weighted",
            attention_min_sequence_length=8192, attention_relay_min_gbps=0.0,
            attention_helper_head_cap=16, attention_helper_safety_mb=1024,
            attention_host_ring_mb=64,
            mlp_token_parallel=True, mlp_primary_fraction=0.68,
            mlp_min_sequence_length=8192, mlp_helper_safety_mb=768,
            single_root=True, single_root_trim_interval=20,
            critical_path_mode=True, sidecar_mlp_blocks=50, sidecar_start_block=0,
            primary_stall_budget_ms=1.0,
            critical_path_adaptive_slack=True,
            critical_path_min_primary_fraction=0.68,
            critical_path_max_primary_fraction=0.72,
            critical_path_fraction_step=0.02,
            critical_path_target_slack_ms=9999.0,
            critical_path_resident_sidecar=False,
            critical_path_pipeline_window=3,
            critical_path_harvest_confirmations=99,
            critical_path_rolling_retire=True,
            attention_primary_first=True,
            critical_path_post_attention_island=True,
        )

    if mode != "DUAL_CAPACITY":
        raise ValueError(f"Unknown H3VM mode: {mode}")

    plan = _capacity_vram_plan(
        config.capacity_vram_profile, config.primary_device, config.secondary_device
    )
    mlp_fraction = float(plan["mlp_primary_fraction"])
    helper_heads = int(plan["capacity_helper_heads"])
    trim_interval = int(plan["single_root_trim_interval"])
    print(
        f"[H3VM CORE] DUAL_CAPACITY | profile={plan['profile']} | "
        f"reserve={plan['primary_runtime_reserve_gb']:.2f}/{plan['secondary_runtime_reserve_gb']:.2f}GiB | "
        f"hot={plan['primary_hot_cache_gb']:.2f}/{plan['secondary_hot_cache_gb']:.2f}GiB | "
        f"MLP={mlp_fraction*100:.0f}/{(1.0-mlp_fraction)*100:.0f} | "
        f"QKV heads={56-helper_heads}/{helper_heads} | trim_every={trim_interval}",
        flush=True,
    )
    return dict(
        primary_runtime_reserve_gb=float(plan["primary_runtime_reserve_gb"]),
        secondary_runtime_reserve_gb=float(plan["secondary_runtime_reserve_gb"]),
        primary_hot_cache_gb=float(plan["primary_hot_cache_gb"]),
        secondary_hot_cache_gb=float(plan["secondary_hot_cache_gb"]),
        one_ahead_prefetch=False, trim_on_stripe_boundary=True,
        attention_mode="off", attention_head_balance="sm_weighted",
        attention_min_sequence_length=1, attention_relay_min_gbps=0.0,
        attention_helper_head_cap=helper_heads, attention_helper_safety_mb=128,
        attention_host_ring_mb=64,
        mlp_token_parallel=True, mlp_primary_fraction=mlp_fraction,
        mlp_min_sequence_length=1, mlp_helper_safety_mb=128,
        single_root=True, single_root_trim_interval=trim_interval,
        critical_path_mode=True, sidecar_mlp_blocks=50, sidecar_start_block=0,
        primary_stall_budget_ms=1.0e9,
        critical_path_adaptive_slack=False,
        critical_path_min_primary_fraction=mlp_fraction,
        critical_path_max_primary_fraction=mlp_fraction,
        critical_path_fraction_step=0.01,
        critical_path_target_slack_ms=999999.0,
        critical_path_resident_sidecar=False,
        critical_path_pipeline_window=1,
        critical_path_harvest_confirmations=999,
        critical_path_rolling_retire=False,
        attention_primary_first=False,
        critical_path_post_attention_island=False,
        capacity_mode=True,
        capacity_mlp_chunk_rows=int(config.capacity_mlp_chunk_rows),
        capacity_outproj_chunk_rows=int(config.capacity_outproj_chunk_rows),
        capacity_helper_heads=helper_heads,
        capacity_attention_kernel=str(config.capacity_attention_kernel),
    )


def _common_builder_kwargs(config: H3VMCoreConfig) -> dict:
    return dict(
        primary_device=str(config.primary_device),
        secondary_device=str(config.secondary_device),
        stripe_size=50,
        expected_steps=max(1, int(config.expected_steps)),
        safe_profile=True,
        hard_cleanup_after_sample=False,
        telemetry=bool(config.telemetry),
    )


def build_asset_model(*, unet_name: str, turbo_lora_name: str, turbo_strength: float = 1.0,
                      turbo_low_vram: bool = False, config: H3VMCoreConfig):
    """Integrated path used by the product Master Loader.

    The already-validated Larry/LightX Turbo compatibility layer is preserved;
    only mode policy is centralized here.
    """
    from .loader import build_h3_streaming_exact_turbo

    mode = _normalize_mode(config.mode)
    print(f"[H3VM CORE] asset model path | mode={mode}", flush=True)
    # The lock also protects the temporary prebuilt bridge used by adapt_model().
    with _CORE_BUILD_LOCK:
        return build_h3_streaming_exact_turbo(
            unet_name=str(unet_name),
            turbo_lora_name=str(turbo_lora_name),
            turbo_strength=float(turbo_strength),
            turbo_low_vram=bool(turbo_low_vram),
            **_common_builder_kwargs(config),
            **execution_kwargs(config),
        )


def _unsupported_prebuilt_state(model) -> list[str]:
    unsupported = []
    checks = (
        ("injections", "runtime injections"),
        ("object_patches", "object patches"),
        ("weight_wrapper_patches", "weight-wrapper patches"),
        ("hook_patches", "hook patches"),
    )
    for attr, label in checks:
        if getattr(model, attr, None):
            unsupported.append(label)
    return unsupported


def _copy_weight_patch_state(source, target) -> int:
    """Copy ordinary ModelPatcher weight patches onto an H3VM subset patcher."""
    patches = getattr(source, "patches", {}) or {}
    target.patches = {key: list(values) for key, values in patches.items()}
    if hasattr(source, "patches_uuid"):
        target.patches_uuid = source.patches_uuid
    if hasattr(source, "force_cast_weights"):
        target.force_cast_weights = source.force_cast_weights
    return sum(len(v) for v in target.patches.values())


class _PrebuiltWeightPatchPlan:
    """Sentinel consumed by the temporary Turbo replay bridge."""


@contextmanager
def _prebuilt_streaming_bridge(private_patcher):
    """Feed a prebuilt patcher through the proven streaming builder unchanged.

    The existing builder is intentionally left untouched. During model
    construction only, its private checkpoint load and Turbo replay hooks are
    replaced with a bridge that:
      1. returns the caller's safe ComfyUI deepclone;
      2. mirrors standard weight patches onto each island/helper patcher.

    All functions are restored before this context exits. Both generic and asset
    builds share _CORE_BUILD_LOCK, so another loader cannot observe the bridge.
    """
    from . import loader as loader_mod
    from . import turbo_compat as turbo_mod

    original_load = loader_mod._load_private_h3
    original_prepare = turbo_mod.prepare_turbo_plan
    original_streaming = turbo_mod.apply_turbo_plan_to_streaming_islands
    original_mlp = turbo_mod.apply_turbo_plan_to_mlp_helpers
    original_attention = turbo_mod.apply_turbo_plan_to_attention_helpers

    def load_private(_unet_name, _primary, _safe_profile):
        return private_patcher, loader_mod._validate_h3(private_patcher)

    def prepare_plan(_patcher, _dm, _lora_name, _strength, _low_vram=False):
        return _PrebuiltWeightPatchPlan()

    def replay_streaming(*, plan, main_patcher, dm, primary_island_patcher,
                         secondary_island_patcher, owner_map, primary_device,
                         secondary_device):
        del plan, dm, owner_map, primary_device, secondary_device
        count = _copy_weight_patch_state(main_patcher, primary_island_patcher)
        if secondary_island_patcher is not None:
            count += _copy_weight_patch_state(main_patcher, secondary_island_patcher)
        return {"source": "prebuilt_weight_patches", "patch_entries": int(count)}

    def replay_helpers(*, plan, primary_helper_patcher, secondary_helper_patcher,
                       owner_map, primary_device, secondary_device, helper_indices=None):
        del plan, owner_map, primary_device, secondary_device, helper_indices
        count = 0
        if primary_helper_patcher is not None:
            count += _copy_weight_patch_state(private_patcher, primary_helper_patcher)
        if secondary_helper_patcher is not None:
            count += _copy_weight_patch_state(private_patcher, secondary_helper_patcher)
        return {"source": "prebuilt_weight_patches", "patch_entries": int(count)}

    def replay_attention(**kwargs):
        # Capacity attention helpers live in the same helper patcher trees as MLP.
        # replay_helpers already copied the complete patch dict, so no second copy
        # is required. Keep the callable because the builder invokes this hook.
        return {"source": "prebuilt_weight_patches", "patch_entries": 0}

    loader_mod._load_private_h3 = load_private
    turbo_mod.prepare_turbo_plan = prepare_plan
    turbo_mod.apply_turbo_plan_to_streaming_islands = replay_streaming
    turbo_mod.apply_turbo_plan_to_mlp_helpers = replay_helpers
    turbo_mod.apply_turbo_plan_to_attention_helpers = replay_attention
    try:
        yield
    finally:
        loader_mod._load_private_h3 = original_load
        turbo_mod.prepare_turbo_plan = original_prepare
        turbo_mod.apply_turbo_plan_to_streaming_islands = original_streaming
        turbo_mod.apply_turbo_plan_to_mlp_helpers = original_mlp
        turbo_mod.apply_turbo_plan_to_attention_helpers = original_attention


def adapt_model(model, *, config: H3VMCoreConfig):
    """Adapt an already-built ComfyUI MODEL into the H3VM execution fabric.

    Supported now:
      * clean H3 ModelPatcher;
      * standard ModelPatcher weight patches / common LoraLoaderModelOnly output.

    Fail-closed now:
      * runtime injection adapters;
      * object/hook/weight-wrapper patch mechanisms.
    """
    import torch
    from .loader import _resolve_device, _validate_h3, build_h3_streaming_exact_turbo

    mode = _normalize_mode(config.mode)
    unsupported = _unsupported_prebuilt_state(model)
    if unsupported:
        raise RuntimeError(
            "H3VM Core prebuilt MODEL currently supports weight-patch LoRA state only. "
            "Unsupported state: " + ", ".join(unsupported) + ". "
            "Refusing silent fallback because helper-GPU arithmetic would differ."
        )
    if not hasattr(model, "deepclone_multigpu"):
        raise RuntimeError("Current ComfyUI ModelPatcher lacks deepclone_multigpu(). Update ComfyUI.")
    if getattr(model, "cached_patcher_init", None) is None:
        raise RuntimeError(
            "This MODEL cannot be safely deep-cloned for H3VM because its loader did not register "
            "cached_patcher_init. Use a core ComfyUI UNET/Checkpoint loader or compatible custom loader."
        )

    primary = _resolve_device(config.primary_device)
    # ComfyUI's official multigpu deepclone reloads pristine weights and carries
    # ModelPatcher patch state without mutating the workflow's input MODEL.
    private = model.deepclone_multigpu(new_load_device=primary)
    private.offload_device = torch.device("cpu")
    if hasattr(private, "remove_additional_models"):
        private.remove_additional_models("multigpu")
    _validate_h3(private)

    patch_count = sum(len(v) for v in getattr(private, "patches", {}).values())
    print(
        f"[H3VM CORE] prebuilt MODEL path | mode={mode} | weight_patch_entries={patch_count} | "
        "private deepclone=yes",
        flush=True,
    )

    with _CORE_BUILD_LOCK:
        with _prebuilt_streaming_bridge(private):
            out = build_h3_streaming_exact_turbo(
                unet_name="<prebuilt-model>",
                turbo_lora_name="<prebuilt-weight-patches>",
                turbo_strength=1.0,
                turbo_low_vram=False,
                **_common_builder_kwargs(config),
                **execution_kwargs(config),
            )
    try:
        out.set_attachments("h3vm_core_adapter", {
            "mode": mode,
            "source": "prebuilt_model",
            "weight_patch_entries": int(patch_count),
            "capacity_vram_profile": str(config.capacity_vram_profile),
        })
    except Exception:
        pass
    return out


def install_runtime_bridge() -> bool:
    """Make the existing integrated loader consume the shared Core policy.

    No public node is added. The current H3VMMasterLoader still calls
    H3VMCapacityModeTurboLoader, but that class is rebound here to build through
    build_asset_model(). Thus the familiar integrated UI is preserved while the
    generic MODEL API and Master Loader share one mode policy implementation.
    """
    package_name = __package__ or ""
    parent_name = package_name.rsplit(".h3vm", 1)[0] if ".h3vm" in package_name else package_name.rsplit(".", 1)[0]
    parent = sys.modules.get(parent_name)
    if parent is None:
        return False
    cls = getattr(parent, "H3VMCapacityModeTurboLoader", None)
    if cls is None:
        return False
    if getattr(cls, "_h3vm_core_bridged", False):
        return True

    original_load = cls.load

    def core_load(self, mode, unet_name, lora_name, strength=1.0, low_vram=False,
                  primary_device="gpu:0", secondary_device="gpu:1",
                  capacity_vram_profile="SAFE｜保守·最稳",
                  capacity_mlp_chunk_rows=4096, capacity_outproj_chunk_rows=4096,
                  capacity_helper_heads=16, capacity_attention_kernel="INT8_CURRENT",
                  telemetry=True, expected_steps=4):
        del self, capacity_helper_heads
        config = H3VMCoreConfig(
            mode=str(mode),
            primary_device=str(primary_device),
            secondary_device=str(secondary_device),
            capacity_vram_profile=str(capacity_vram_profile),
            expected_steps=int(expected_steps),
            telemetry=bool(telemetry),
            capacity_mlp_chunk_rows=int(capacity_mlp_chunk_rows),
            capacity_outproj_chunk_rows=int(capacity_outproj_chunk_rows),
            capacity_attention_kernel=str(capacity_attention_kernel),
        )
        model = build_asset_model(
            unet_name=str(unet_name),
            turbo_lora_name=str(lora_name),
            turbo_strength=float(strength),
            turbo_low_vram=bool(low_vram),
            config=config,
        )
        return (model,)

    cls._h3vm_original_load = original_load
    cls.load = core_load
    cls._h3vm_core_bridged = True
    print("[H3VM CORE] runtime bridge installed | Master Loader -> shared Core", flush=True)
    return True
