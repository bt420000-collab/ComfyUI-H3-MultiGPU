"""Shared H3VM execution core.

This module separates *how a MODEL is obtained* from *how H3VM executes it*.
The product Master Loader can keep its convenient model/Turbo/prompt controls,
while advanced workflows can hand an already-built ComfyUI MODEL to the same
H3VM placement policy.

The prebuilt-MODEL path is intentionally fail-closed for runtime injections and
object patches in this first public revision. Standard ModelPatcher weight
patches (the common LoraLoaderModelOnly path) are supported and mirrored onto
H3VM block/helper patchers by the streaming builder. Injection-style adapters
need device-aware helper hook remapping and must not be silently approximated.
"""
from __future__ import annotations

from dataclasses import dataclass


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
    """Return the single source of truth for H3VM execution-mode policy."""
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

    This preserves H3VM's already-validated Larry/LightX Turbo compatibility
    machinery while sharing exactly the same mode policy as adapt_model().
    """
    from .loader import build_h3_streaming_exact_turbo

    mode = _normalize_mode(config.mode)
    print(f"[H3VM CORE] asset model path | mode={mode}", flush=True)
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
        value = getattr(model, attr, None)
        if value:
            unsupported.append(label)
    return unsupported


def adapt_model(model, *, config: H3VMCoreConfig):
    """Adapt an already-built ComfyUI MODEL into the H3VM execution fabric.

    Supported now:
      * clean H3 ModelPatcher;
      * standard ModelPatcher weight patches / common LoraLoaderModelOnly output.

    Fail-closed now:
      * runtime injection adapters;
      * object/hook/weight-wrapper patch mechanisms.

    The fail-closed rule is deliberate. Those mechanisms can hold direct module
    references and must be remapped to H3VM helper modules per device before we
    can claim exact arithmetic.
    """
    import torch
    from .loader import _resolve_device, _validate_h3, build_h3_streaming_exact_turbo

    mode = _normalize_mode(config.mode)
    unsupported = _unsupported_prebuilt_state(model)
    if unsupported:
        raise RuntimeError(
            "H3VM Core prebuilt MODEL currently supports weight-patch LoRA state only. "
            "Unsupported state: " + ", ".join(unsupported) + ". "
            "Do not silently continue: place injection-style accelerators outside this Core path "
            "until device-aware helper remapping lands."
        )
    if not hasattr(model, "deepclone_multigpu"):
        raise RuntimeError("Current ComfyUI ModelPatcher lacks deepclone_multigpu(). Update ComfyUI.")
    if getattr(model, "cached_patcher_init", None) is None:
        raise RuntimeError(
            "This MODEL cannot be safely deep-cloned for H3VM because its loader did not register "
            "cached_patcher_init. Use a core ComfyUI UNET/Checkpoint loader or a compatible custom loader."
        )

    primary = _resolve_device(config.primary_device)
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
    out = build_h3_streaming_exact_turbo(
        unet_name="<prebuilt-model>",
        turbo_lora_name="<prebuilt-weight-patches>",
        turbo_strength=1.0,
        turbo_low_vram=False,
        _prebuilt_patcher=private,
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
