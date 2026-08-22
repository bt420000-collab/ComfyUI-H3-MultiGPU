from __future__ import annotations
import logging

LOG = logging.getLogger("H3VM")


def _resolve_device(option):
    import torch
    import comfy.model_management
    try:
        d = comfy.model_management.resolve_gpu_device_option(option)
    except Exception:
        d = None
    if d is None and isinstance(option, str) and option.startswith("gpu:"):
        d = torch.device("cuda:" + option.split(":", 1)[1])
    if d is None:
        raise RuntimeError(f"H3VM cannot resolve device {option!r}")
    return torch.device(d)


def _validate_h3(patcher):
    try:
        dm = patcher.model.diffusion_model
    except Exception as e:
        raise RuntimeError("H3VM loaded object is not a diffusion ModelPatcher") from e
    required = ("blocks", "video_patch_proj", "audio_patch_proj", "final_layer", "hidden_size")
    missing = [x for x in required if not hasattr(dm, x)]
    if missing or len(dm.blocks) != 50:
        raise RuntimeError(f"H3VM supports current 50-block MiniMax H3 only; missing={missing}, blocks={len(dm.blocks)}")
    return dm


def _ensure_clean_patcher(patcher):
    dirty = []
    if getattr(patcher, "patches", None): dirty.append("weight patches/LoRA")
    if getattr(patcher, "object_patches", None): dirty.append("object patches")
    if getattr(patcher, "injections", None): dirty.append("injections")
    if dirty:
        raise RuntimeError(
            "H3VM physical/hybrid shard currently requires an unpatched H3. "
            "Do not apply Larry/LoRA before this loader. Unsupported state: " + ", ".join(dirty)
        )


def _load_private_h3(unet_name, primary, safe_profile):
    import torch
    import folder_paths
    import comfy.sd
    from comfy.cli_args import args

    if safe_profile and not bool(getattr(args, "disable_pinned_memory", False)):
        raise RuntimeError(
            "H3VM safe profile requires ComfyUI to start with --disable-pinned-memory. "
            "This check runs before the H3 checkpoint is loaded."
        )
    path = folder_paths.get_full_path("diffusion_models", unet_name)
    if path is None:
        raise RuntimeError(f"H3VM cannot find diffusion model: {unet_name}")
    patcher = comfy.sd.load_diffusion_model(
        path,
        model_options={"load_device": primary, "offload_device": torch.device("cpu")},
    )
    dm = _validate_h3(patcher)
    _ensure_clean_patcher(patcher)
    return patcher, dm


def _choose_placement(*, dm, patcher, primary, secondary, split_mode, secondary_blocks,
                      primary_reserve_gb, secondary_reserve_gb):
    import torch
    import comfy.model_management
    from .planner import BlockStat, choose_contiguous_prefix, Placement

    block_bytes = [int(comfy.model_management.module_size(b)) for b in dm.blocks]
    total_bytes = int(comfy.model_management.module_size(patcher.model))
    blocks_total = sum(block_bytes)
    fixed_bytes = max(0, total_bytes - blocks_total)
    stats = [BlockStat(i, b) for i, b in enumerate(block_bytes)]
    ptotal = int(torch.cuda.get_device_properties(primary).total_memory)
    stotal = int(torch.cuda.get_device_properties(secondary).total_memory)

    if split_mode in ("memory_aware", "balanced", "speed_first"):
        policy = "balanced" if split_mode in ("memory_aware", "balanced") else "speed_first"
        placement = choose_contiguous_prefix(
            stats, fixed_bytes, ptotal, stotal,
            primary_reserve_gib=float(primary_reserve_gb),
            secondary_reserve_gib=float(secondary_reserve_gb),
            policy=policy,
        )
        count = int(placement.secondary_count)
    else:
        count = max(1, min(49, int(secondary_blocks)))
        sbytes = sum(block_bytes[:count])
        pbytes = blocks_total - sbytes
        pcap = int(ptotal - primary_reserve_gb * (1024**3))
        scap = int(stotal - secondary_reserve_gb * (1024**3))
        if sbytes > scap or fixed_bytes + pbytes > pcap:
            raise RuntimeError("H3VM manual split violates the selected reserve budget; refusing avoidable OOM risk.")
        placement = Placement(count, sbytes, pbytes, fixed_bytes, pcap, scap)

    return placement, count, block_bytes, total_bytes, blocks_total, fixed_bytes


def _partition_private_h3(*, patcher, dm, primary, secondary, count, block_bytes, total_bytes,
                          secondary_prefetch=False, telemetry=True):
    import torch
    import comfy.model_management
    from .shard import H3RemoteSpanRouter, H3RemoteBlockProxyFactory, make_shard_root

    original_blocks = list(dm.blocks)
    owned = {i: original_blocks[i] for i in range(count)}
    router = H3RemoteSpanRouter(
        primary, secondary, owned, 0, count - 1,
        secondary_prefetch=secondary_prefetch,
        telemetry=telemetry,
    )
    for i in range(count):
        dm.blocks[i] = H3RemoteBlockProxyFactory.make(router, i)

    shard_root = make_shard_root(patcher.model, dm, owned, len(original_blocks))
    ShardPatcher = patcher.__class__
    shard_patcher = ShardPatcher(
        shard_root,
        load_device=secondary,
        offload_device=torch.device("cpu"),
        size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )

    patcher.size = 0
    shard_patcher.size = 0
    primary_tree_bytes = int(comfy.model_management.module_size(patcher.model))
    secondary_tree_bytes = int(comfy.model_management.module_size(shard_root))

    patcher.cached_patcher_init = None
    shard_patcher.cached_patcher_init = None
    patcher.set_additional_models("h3vm_physical_shard", [shard_patcher])
    patcher.set_attachments("h3vm_router", router)
    patcher.set_attachments("h3vm_secondary", shard_patcher)

    expected_secondary = sum(block_bytes[:count])
    if secondary_tree_bytes < int(expected_secondary * 0.98):
        raise RuntimeError(
            f"H3VM shard tree unexpectedly small: {secondary_tree_bytes/(1024**3):.2f}GiB; "
            f"expected about {expected_secondary/(1024**3):.2f}GiB"
        )
    if primary_tree_bytes + secondary_tree_bytes > int(total_bytes * 1.03):
        raise RuntimeError("H3VM detected apparent weight duplication after partitioning; refusing to continue.")

    return router, shard_patcher, primary_tree_bytes, secondary_tree_bytes


def _common_preflight(primary_device, secondary_device):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("H3VM requires at least two CUDA GPUs.")
    primary = _resolve_device(primary_device)
    secondary = _resolve_device(secondary_device)
    if primary.type != "cuda" or secondary.type != "cuda" or primary == secondary:
        raise RuntimeError(f"H3VM needs two different CUDA devices, got {primary} and {secondary}")
    return primary, secondary


def build_h3_physical_shard(*, unet_name: str, primary_device: str, secondary_device: str,
                            split_mode: str, secondary_blocks: int,
                            primary_reserve_gb: float, secondary_reserve_gb: float,
                            safe_profile: bool):
    """Known-good Dev4-compatible path."""
    from comfy.cli_args import args
    primary, secondary = _common_preflight(primary_device, secondary_device)
    LOG.info(
        "H3VM Dev4 fallback private load | model=%s | primary=%s | secondary=%s | pinned_disabled=%s",
        unet_name, primary, secondary, bool(getattr(args, "disable_pinned_memory", False)),
    )
    patcher, dm = _load_private_h3(unet_name, primary, safe_profile)
    placement, count, block_bytes, total_bytes, blocks_total, fixed_bytes = _choose_placement(
        dm=dm, patcher=patcher, primary=primary, secondary=secondary,
        split_mode=split_mode, secondary_blocks=secondary_blocks,
        primary_reserve_gb=primary_reserve_gb, secondary_reserve_gb=secondary_reserve_gb,
    )
    _, _, primary_tree_bytes, secondary_tree_bytes = _partition_private_h3(
        patcher=patcher, dm=dm, primary=primary, secondary=secondary, count=count,
        block_bytes=block_bytes, total_bytes=total_bytes, secondary_prefetch=False, telemetry=False,
    )
    LOG.info(
        "H3VM Dev4 fallback registered | total=%.2fGiB fixed=%.2fGiB | primary=%.2fGiB secondary=%.2fGiB | secondary blocks=0-%d",
        total_bytes/(1024**3), fixed_bytes/(1024**3), primary_tree_bytes/(1024**3), secondary_tree_bytes/(1024**3), count-1,
    )
    print(
        f"[H3VM DEV4 FALLBACK READY] {primary}: blocks {count}-49 + fixed | "
        f"{secondary}: blocks 0-{count-1} | registered {primary_tree_bytes/(1024**3):.2f}+{secondary_tree_bytes/(1024**3):.2f}GiB",
        flush=True,
    )
    return patcher


def build_h3_hybrid_shard(*, unet_name: str, primary_device: str, secondary_device: str,
                          split_mode: str, secondary_blocks: int,
                          primary_reserve_gb: float, secondary_reserve_gb: float,
                          safe_profile: bool, attention_mode: str, head_balance: str,
                          min_sequence_length: int, secondary_prefetch: bool, probe_link: bool):
    import torch
    from comfy.cli_args import args
    from .peer_link import probe_peer_copy

    primary, secondary = _common_preflight(primary_device, secondary_device)
    if safe_profile and not bool(getattr(args, "disable_pinned_memory", False)):
        raise RuntimeError(
            "H3VM Dev5 safe profile requires --disable-pinned-memory before any H3VM CUDA probe/load work."
        )
    LOG.info(
        "H3VM Dev5 private load | model=%s | primary=%s (%s) | secondary=%s (%s) | pinned_disabled=%s",
        unet_name, primary, torch.cuda.get_device_name(primary), secondary, torch.cuda.get_device_name(secondary),
        bool(getattr(args, "disable_pinned_memory", False)),
    )

    link = None
    if probe_link or attention_mode != "off":
        link = probe_peer_copy(primary, secondary, size_mb=32, repeats=4)
        LOG.info(
            "H3VM Dev5 link probe | %s->%s peer=%s bw=%s | %s->%s peer=%s bw=%s | error=%s",
            primary, secondary, link["peer_ab"],
            "%.2fGB/s" % link["gbps_ab"] if link["gbps_ab"] is not None else "n/a",
            secondary, primary, link["peer_ba"],
            "%.2fGB/s" % link["gbps_ba"] if link["gbps_ba"] is not None else "n/a",
            link["error"],
        )

    patcher, dm = _load_private_h3(unet_name, primary, safe_profile)
    placement, count, block_bytes, total_bytes, blocks_total, fixed_bytes = _choose_placement(
        dm=dm, patcher=patcher, primary=primary, secondary=secondary,
        split_mode=split_mode, secondary_blocks=secondary_blocks,
        primary_reserve_gb=primary_reserve_gb, secondary_reserve_gb=secondary_reserve_gb,
    )
    router, shard_patcher, primary_tree_bytes, secondary_tree_bytes = _partition_private_h3(
        patcher=patcher, dm=dm, primary=primary, secondary=secondary, count=count,
        block_bytes=block_bytes, total_bytes=total_bytes,
        secondary_prefetch=secondary_prefetch, telemetry=True,
    )

    attention = None
    if attention_mode != "off":
        from .attention_parallel import H3VMTwoGPUAttentionParallel
        # If the explicit peer-copy probe itself failed, auto mode must not gamble.
        effective_mode = attention_mode
        if attention_mode == "auto" and link is not None and link.get("error"):
            effective_mode = "off"
            LOG.warning("H3VM Dev5 attention auto-disabled because the peer-copy probe failed: %s", link["error"])
        attention = H3VMTwoGPUAttentionParallel(
            primary, secondary,
            mode=effective_mode,
            head_balance=head_balance,
            min_sequence_length=min_sequence_length,
        )
        if attention.enabled:
            if not hasattr(patcher, "set_model_optimized_attention"):
                raise RuntimeError("Current ComfyUI ModelPatcher lacks set_model_optimized_attention; cannot enable Dev5 hybrid attention.")
            patcher.set_model_optimized_attention(attention)
            patcher.set_attachments("h3vm_dev5_attention", attention)
        else:
            LOG.info("H3VM Dev5 hybrid attention inactive | reason=%s", attention.disable_reason)

    patcher.set_attachments("h3vm_dev5_link_probe", link)
    patcher.set_attachments("h3vm_dev5_config", {
        "split_mode": split_mode,
        "secondary_prefetch": secondary_prefetch,
        "attention_mode": attention_mode,
        "head_balance": head_balance,
        "min_sequence_length": min_sequence_length,
    })

    LOG.info(
        "H3VM Dev5 partition | total=%.2fGiB blocks=%.2fGiB fixed=%.2fGiB | secondary blocks=0-%d",
        total_bytes/(1024**3), blocks_total/(1024**3), fixed_bytes/(1024**3), count-1,
    )
    LOG.info(
        "H3VM Dev5 registered trees | primary=%s %.2fGiB | secondary=%s %.2fGiB | sum=%.2fGiB | duplication=%s",
        primary, primary_tree_bytes/(1024**3), secondary, secondary_tree_bytes/(1024**3),
        (primary_tree_bytes+secondary_tree_bytes)/(1024**3),
        (primary_tree_bytes + secondary_tree_bytes > int(total_bytes * 1.03)),
    )
    LOG.info(
        "H3VM Dev5 budget | primary cap=%.2fGiB planned_weights=%.2fGiB reserve=%.2fGiB | secondary cap=%.2fGiB planned_weights=%.2fGiB reserve=%.2fGiB",
        placement.primary_capacity/(1024**3), (placement.primary_fixed_bytes+placement.primary_block_bytes)/(1024**3), primary_reserve_gb,
        placement.secondary_capacity/(1024**3), placement.secondary_bytes/(1024**3), secondary_reserve_gb,
    )
    print(
        f"[H3VM DEV5 READY] one private H3 | {primary}: blocks {count}-49 + fixed | "
        f"{secondary}: blocks 0-{count-1} | prefetch={'on' if secondary_prefetch else 'off'} | "
        f"hybrid_attention={'on' if (attention is not None and attention.enabled) else 'off'} | "
        f"registered {primary_tree_bytes/(1024**3):.2f}+{secondary_tree_bytes/(1024**3):.2f}GiB",
        flush=True,
    )
    return patcher


def build_h3_relay_shard(*, unet_name: str, primary_device: str, secondary_device: str,
                         split_mode: str, secondary_blocks: int,
                         primary_reserve_gb: float, secondary_reserve_gb: float,
                         safe_profile: bool, attention_mode: str, head_balance: str,
                         min_sequence_length: int, secondary_prefetch: bool,
                         probe_link: bool, relay_min_gbps: float,
                         helper_head_cap: int, helper_safety_mb: int):
    """Dev6 transport-aware H3 shard.

    Keeps Dev5's proven one-copy physical block partition. If direct CUDA P2P is
    unavailable, benchmarks the same fallback cross-device copy route already
    used by the span router and can run a conservative relay attention path.
    """
    import torch
    from comfy.cli_args import args
    from .peer_link import probe_copy_paths

    primary, secondary = _common_preflight(primary_device, secondary_device)
    if safe_profile and not bool(getattr(args, "disable_pinned_memory", False)):
        raise RuntimeError(
            "H3VM Dev6 safe profile requires --disable-pinned-memory before any H3VM CUDA probe/load work."
        )

    LOG.info(
        "H3VM Dev6 private load | model=%s | primary=%s (%s) | secondary=%s (%s) | pinned_disabled=%s",
        unet_name, primary, torch.cuda.get_device_name(primary),
        secondary, torch.cuda.get_device_name(secondary),
        bool(getattr(args, "disable_pinned_memory", False)),
    )

    link = None
    if probe_link or attention_mode != "off":
        link = probe_copy_paths(primary, secondary, size_mb=64, repeats=4)
        LOG.info(
            "H3VM Dev6 transport probe | %s->%s peer=%s copy=%s | %s->%s peer=%s copy=%s | errors=%s / %s",
            primary, secondary, link["peer_ab"],
            "%.2fGB/s" % link["copy_gbps_ab"] if link["copy_gbps_ab"] is not None else "n/a",
            secondary, primary, link["peer_ba"],
            "%.2fGB/s" % link["copy_gbps_ba"] if link["copy_gbps_ba"] is not None else "n/a",
            link["error_ab"], link["error_ba"],
        )

    patcher, dm = _load_private_h3(unet_name, primary, safe_profile)
    placement, count, block_bytes, total_bytes, blocks_total, fixed_bytes = _choose_placement(
        dm=dm, patcher=patcher, primary=primary, secondary=secondary,
        split_mode=split_mode, secondary_blocks=secondary_blocks,
        primary_reserve_gb=primary_reserve_gb, secondary_reserve_gb=secondary_reserve_gb,
    )
    router, shard_patcher, primary_tree_bytes, secondary_tree_bytes = _partition_private_h3(
        patcher=patcher, dm=dm, primary=primary, secondary=secondary, count=count,
        block_bytes=block_bytes, total_bytes=total_bytes,
        secondary_prefetch=secondary_prefetch, telemetry=True,
    )

    attention = None
    if attention_mode != "off":
        from .attention_parallel import H3VMRelayAttentionParallel
        attention = H3VMRelayAttentionParallel(
            primary, secondary,
            mode=attention_mode,
            head_balance=head_balance,
            min_sequence_length=min_sequence_length,
            link_info=link or {},
            relay_min_gbps=relay_min_gbps,
            helper_head_cap=helper_head_cap,
            helper_safety_mb=helper_safety_mb,
        )
        if attention.enabled:
            if not hasattr(patcher, "set_model_optimized_attention"):
                raise RuntimeError(
                    "Current ComfyUI ModelPatcher lacks set_model_optimized_attention; "
                    "cannot enable Dev6 transport-aware attention."
                )
            patcher.set_model_optimized_attention(attention)
            patcher.set_attachments("h3vm_dev6_attention", attention)
        else:
            LOG.info("H3VM Dev6 attention inactive | reason=%s", attention.disable_reason)

    patcher.set_attachments("h3vm_dev6_link_probe", link)
    patcher.set_attachments("h3vm_dev6_config", {
        "split_mode": split_mode,
        "secondary_prefetch": secondary_prefetch,
        "attention_mode": attention_mode,
        "head_balance": head_balance,
        "min_sequence_length": min_sequence_length,
        "relay_min_gbps": relay_min_gbps,
        "helper_head_cap": helper_head_cap,
        "helper_safety_mb": helper_safety_mb,
    })

    LOG.info(
        "H3VM Dev6 partition | total=%.2fGiB blocks=%.2fGiB fixed=%.2fGiB | secondary blocks=0-%d",
        total_bytes/(1024**3), blocks_total/(1024**3), fixed_bytes/(1024**3), count-1,
    )
    LOG.info(
        "H3VM Dev6 registered trees | primary=%s %.2fGiB | secondary=%s %.2fGiB | sum=%.2fGiB | duplication=%s",
        primary, primary_tree_bytes/(1024**3), secondary, secondary_tree_bytes/(1024**3),
        (primary_tree_bytes+secondary_tree_bytes)/(1024**3),
        (primary_tree_bytes + secondary_tree_bytes > int(total_bytes * 1.03)),
    )
    LOG.info(
        "H3VM Dev6 budget | primary cap=%.2fGiB planned_weights=%.2fGiB reserve=%.2fGiB | "
        "secondary cap=%.2fGiB planned_weights=%.2fGiB reserve=%.2fGiB",
        placement.primary_capacity/(1024**3),
        (placement.primary_fixed_bytes+placement.primary_block_bytes)/(1024**3),
        primary_reserve_gb,
        placement.secondary_capacity/(1024**3),
        placement.secondary_bytes/(1024**3),
        secondary_reserve_gb,
    )
    print(
        f"[H3VM DEV6 READY] one private H3 | {primary}: blocks {count}-49 + fixed | "
        f"{secondary}: blocks 0-{count-1} | prefetch={'on' if secondary_prefetch else 'off'} | "
        f"attention={'%s' % (attention.transport if (attention is not None and attention.enabled) else 'off')} | "
        f"registered {primary_tree_bytes/(1024**3):.2f}+{secondary_tree_bytes/(1024**3):.2f}GiB",
        flush=True,
    )
    return patcher


def _partition_global_h3(*, patcher, dm, primary, secondary, count, block_bytes, total_bytes,
                         space, secondary_prefetch=True, telemetry=True):
    import torch
    import comfy.model_management
    from .global_router import H3GlobalSpanRouter, H3GlobalBlockProxyFactory
    from .shard import make_shard_root

    original_blocks = list(dm.blocks)
    owned = {i: original_blocks[i] for i in range(count)}
    router = H3GlobalSpanRouter(
        primary, secondary, owned, 0, count - 1,
        space=space,
        secondary_prefetch=secondary_prefetch,
        telemetry=telemetry,
    )
    for i in range(count):
        dm.blocks[i] = H3GlobalBlockProxyFactory.make(router, i)

    shard_root = make_shard_root(patcher.model, dm, owned, len(original_blocks))
    ShardPatcher = patcher.__class__
    shard_patcher = ShardPatcher(
        shard_root,
        load_device=secondary,
        offload_device=torch.device("cpu"),
        size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )
    patcher.size = 0
    shard_patcher.size = 0
    primary_tree_bytes = int(comfy.model_management.module_size(patcher.model))
    secondary_tree_bytes = int(comfy.model_management.module_size(shard_root))

    patcher.cached_patcher_init = None
    shard_patcher.cached_patcher_init = None
    patcher.set_additional_models("h3vm_global_physical_shard", [shard_patcher])
    patcher.set_attachments("h3vm_global_router", router)
    patcher.set_attachments("h3vm_global_space", space)
    patcher.set_attachments("h3vm_global_secondary", shard_patcher)

    expected_secondary = sum(block_bytes[:count])
    if secondary_tree_bytes < int(expected_secondary * 0.98):
        raise RuntimeError(
            f"H3VM Global shard tree unexpectedly small: {secondary_tree_bytes/(1024**3):.2f}GiB; "
            f"expected about {expected_secondary/(1024**3):.2f}GiB"
        )
    if primary_tree_bytes + secondary_tree_bytes > int(total_bytes * 1.03):
        raise RuntimeError("H3VM Global detected apparent weight duplication; refusing to continue.")
    return router, shard_patcher, primary_tree_bytes, secondary_tree_bytes


def build_h3_global_memory_shard(*, unet_name: str, primary_device: str, secondary_device: str,
                                 split_mode: str, secondary_blocks: int,
                                 primary_reserve_gb: float, secondary_reserve_gb: float,
                                 safe_profile: bool, transport_mode: str,
                                 benchmark_transport: bool, benchmark_mb: int,
                                 benchmark_repeats: int, allow_explicit_pinned: bool,
                                 pinned_ring_mb: int, secondary_prefetch: bool):
    """Dev7: model runs on GLOBAL MEMORY; GPUs are attached compute blocks.

    Milestone scope:
    - keep the proven one-copy physical H3 weight partition;
    - replace H3 activation handoff ownership with GlobalTensor acquire/publish;
    - transport is a backend policy (direct or host-neutral), hidden from H3;
    - CPU is registered as a first-class ComputeBlock for future VAE work;
    - no broad pinned RAM and no allocator monkeypatching.
    """
    import torch
    from comfy.cli_args import args
    from .global_memory import GlobalMemorySpace

    primary, secondary = _common_preflight(primary_device, secondary_device)
    if safe_profile and not bool(getattr(args, "disable_pinned_memory", False)):
        raise RuntimeError(
            "H3VM Dev7 safe profile requires --disable-pinned-memory before Global Memory initialization."
        )
    if pinned_ring_mb > 128 and allow_explicit_pinned:
        raise RuntimeError(
            "H3VM Dev7 intentionally caps the first explicit pinned runway at 128MB per slot. "
            "Use <=128MB until Global Memory telemetry is characterized."
        )

    LOG.info(
        "H3VM Dev7 GLOBAL private load | model=%s | primary=%s (%s) | secondary=%s (%s) | "
        "pinned_disabled=%s | transport=%s",
        unet_name, primary, torch.cuda.get_device_name(primary),
        secondary, torch.cuda.get_device_name(secondary),
        bool(getattr(args, "disable_pinned_memory", False)), transport_mode,
    )

    # The transport engine always performs a small characterization pass so that
    # the report is self-contained. benchmark_transport=False simply uses a
    # smaller 8MB/1-repeat probe to reduce startup cost.
    space = GlobalMemorySpace(
        primary, secondary,
        transport_mode=transport_mode,
        benchmark_mb=int(benchmark_mb if benchmark_transport else 8),
        benchmark_repeats=int(benchmark_repeats if benchmark_transport else 1),
        allow_explicit_pinned=bool(allow_explicit_pinned),
        pinned_ring_mb=int(pinned_ring_mb),
        pinned_ring_slots=2,
    )
    r = space.transport.report()
    b = r["benchmark"]
    LOG.info(
        "H3VM Dev7 GLOBAL transport | selected=%s requested=%s | P2P=%s/%s | "
        "direct=%.2f/%.2fGB/s neutral_pageable=%.2f/%.2fGB/s neutral_pinned=%s/%s",
        r["selected_mode"], r["requested_mode"], b["peer_ab"], b["peer_ba"],
        float(b["direct_d2d"]["ab"] or 0.0), float(b["direct_d2d"]["ba"] or 0.0),
        float(b["neutral_pageable"]["ab"] or 0.0), float(b["neutral_pageable"]["ba"] or 0.0),
        "%.2f" % b["neutral_pinned"]["ab"] if b["neutral_pinned"]["ab"] is not None else "off",
        "%.2f" % b["neutral_pinned"]["ba"] if b["neutral_pinned"]["ba"] is not None else "off",
    )

    patcher, dm = _load_private_h3(unet_name, primary, safe_profile)
    placement, count, block_bytes, total_bytes, blocks_total, fixed_bytes = _choose_placement(
        dm=dm, patcher=patcher, primary=primary, secondary=secondary,
        split_mode=split_mode, secondary_blocks=secondary_blocks,
        primary_reserve_gb=primary_reserve_gb, secondary_reserve_gb=secondary_reserve_gb,
    )
    router, shard_patcher, primary_tree_bytes, secondary_tree_bytes = _partition_global_h3(
        patcher=patcher, dm=dm, primary=primary, secondary=secondary, count=count,
        block_bytes=block_bytes, total_bytes=total_bytes, space=space,
        secondary_prefetch=secondary_prefetch, telemetry=True,
    )
    patcher.set_attachments("h3vm_dev7_config", {
        "architecture": "GLOBAL_MEMORY",
        "transport_mode": transport_mode,
        "selected_transport": space.transport.selected_mode,
        "split_mode": split_mode,
        "secondary_prefetch": secondary_prefetch,
        "allow_explicit_pinned": allow_explicit_pinned,
        "pinned_ring_mb": pinned_ring_mb,
    })

    LOG.info(
        "H3VM Dev7 GLOBAL partition | total=%.2fGiB blocks=%.2fGiB fixed=%.2fGiB | "
        "primary=%.2fGiB secondary=%.2fGiB | secondary blocks=0-%d",
        total_bytes/(1024**3), blocks_total/(1024**3), fixed_bytes/(1024**3),
        primary_tree_bytes/(1024**3), secondary_tree_bytes/(1024**3), count-1,
    )
    LOG.info(
        "H3VM Dev7 GLOBAL compute fabric | cpu=enabled | %s free=%.2fGiB | %s free=%.2fGiB | "
        "logical_owner=GlobalMemorySpace",
        primary, space.compute_blocks[1].free_gib(), secondary, space.compute_blocks[2].free_gib(),
    )
    print(
        f"[H3VM DEV7 GLOBAL READY] GlobalMemorySpace owns tensors | CPU + {primary} + {secondary} are compute blocks | "
        f"transport={space.transport.selected_mode} | weights={primary_tree_bytes/(1024**3):.2f}+{secondary_tree_bytes/(1024**3):.2f}GiB | "
        f"secondary blocks=0-{count-1}",
        flush=True,
    )
    return patcher


def _partition_global_h3_tp(*, patcher, dm, primary, secondary, count, block_bytes, total_bytes,
                            space, placement, tp_scope, primary_ffn_groups,
                            tp_prefetch=True, secondary_prefetch=True, telemetry=True):
    """Dev8 physical topology.

    Whole-block attention/norm/adaln ownership remains the proven Dev7 prefix
    split. On selected blocks Dev8 row-shards only FC1 in packed ConvRot W4A4
    form. Both GPUs see the same full FC1 input and compute disjoint output rows.
    The original full FC2 deliberately stays on the whole-block owner so Dev8
    does not change ConvRot's full-K activation-scale arithmetic. Remote FC1
    helpers live under per-device ModelPatchers and are registered only once.
    """
    import torch
    import comfy.model_management
    from .global_router import H3GlobalSpanRouter, H3GlobalBlockProxyFactory
    from .shard import make_shard_root
    from .tp_mlp import install_tp_mlp_fabric, make_helper_root, plan_tp_primary_groups

    original_blocks = list(dm.blocks)

    base_primary_bytes = int(placement.primary_fixed_bytes + placement.primary_block_bytes)
    base_secondary_bytes = int(placement.secondary_bytes)
    tp_plan = plan_tp_primary_groups(
        blocks=original_blocks,
        split_count=count,
        tp_scope=tp_scope,
        requested_groups=int(primary_ffn_groups),
        base_primary_bytes=base_primary_bytes,
        base_secondary_bytes=base_secondary_bytes,
        primary_capacity=int(placement.primary_capacity),
        secondary_capacity=int(placement.secondary_capacity),
        safety_margin_bytes=4 * 1024 * 1024,
    )
    effective_ffn_groups = int(tp_plan["effective_groups"])
    if effective_ffn_groups != int(primary_ffn_groups):
        LOG.warning(
            "H3VM Dev8.2 TP auto-fit | requested primary_ffn_groups=%d -> effective=%d | "
            "projected weights=%.3f+%.3fGiB | safe capacities=%.3f+%.3fGiB | margin=%.1fMiB",
            int(primary_ffn_groups), effective_ffn_groups,
            tp_plan["primary_bytes"]/(1024**3), tp_plan["secondary_bytes"]/(1024**3),
            placement.primary_capacity/(1024**3), placement.secondary_capacity/(1024**3),
            tp_plan["margin_bytes"]/(1024**2),
        )
    else:
        LOG.info(
            "H3VM Dev8.2 TP auto-fit | requested/effective primary_ffn_groups=%d | "
            "projected weights=%.3f+%.3fGiB",
            effective_ffn_groups, tp_plan["primary_bytes"]/(1024**3), tp_plan["secondary_bytes"]/(1024**3),
        )

    fabric, primary_helpers, secondary_helpers, geometry, tp_blocks = install_tp_mlp_fabric(
        blocks=original_blocks,
        split_count=count,
        primary=primary,
        secondary=secondary,
        space=space,
        tp_scope=tp_scope,
        primary_ffn_groups=effective_ffn_groups,
        tp_prefetch=bool(tp_prefetch),
        telemetry=bool(telemetry),
    )

    # The block objects in dm.blocks are the same objects as original_blocks, so
    # the MLP replacements above are already visible to the main model.
    owned = {i: original_blocks[i] for i in range(count)}
    router = H3GlobalSpanRouter(
        primary, secondary, owned, 0, count - 1,
        space=space,
        secondary_prefetch=secondary_prefetch,
        telemetry=telemetry,
    )
    for i in range(count):
        dm.blocks[i] = H3GlobalBlockProxyFactory.make(router, i)

    shard_root = make_shard_root(patcher.model, dm, owned, len(original_blocks))
    ShardPatcher = patcher.__class__
    shard_patcher = ShardPatcher(
        shard_root,
        load_device=secondary,
        offload_device=torch.device("cpu"),
        size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )

    primary_helper_root = make_helper_root(primary_helpers, "primary") if primary_helpers else None
    secondary_helper_root = make_helper_root(secondary_helpers, "secondary") if secondary_helpers else None
    helper_patchers = []
    primary_helper_patcher = None
    secondary_helper_patcher = None
    if primary_helper_root is not None:
        primary_helper_patcher = ShardPatcher(
            primary_helper_root,
            load_device=primary,
            offload_device=torch.device("cpu"),
            size=0,
            weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
        )
        helper_patchers.append(primary_helper_patcher)
    if secondary_helper_root is not None:
        secondary_helper_patcher = ShardPatcher(
            secondary_helper_root,
            load_device=secondary,
            offload_device=torch.device("cpu"),
            size=0,
            weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
        )
        helper_patchers.append(secondary_helper_patcher)

    for p in [patcher, shard_patcher, primary_helper_patcher, secondary_helper_patcher]:
        if p is None:
            continue
        p.size = 0
        p.cached_patcher_init = None

    primary_main_bytes = int(comfy.model_management.module_size(patcher.model))
    secondary_block_bytes = int(comfy.model_management.module_size(shard_root))
    primary_helper_bytes = int(comfy.model_management.module_size(primary_helper_root)) if primary_helper_root is not None else 0
    secondary_helper_bytes = int(comfy.model_management.module_size(secondary_helper_root)) if secondary_helper_root is not None else 0
    primary_tree_bytes = primary_main_bytes + primary_helper_bytes
    secondary_tree_bytes = secondary_block_bytes + secondary_helper_bytes
    combined = primary_tree_bytes + secondary_tree_bytes

    # FC1 rows are partitioned, not duplicated, and FC2 stays whole on its owner.
    # Allow a small accounting margin for wrapper/module bookkeeping; anything
    # beyond 6% strongly suggests accidental whole-weight duplication or a
    # dequantized copy and is refused before inference.
    if combined > int(total_bytes * 1.06):
        raise RuntimeError(
            "H3VM Dev8.2 detected apparent tensor-shard duplication/dequantization: "
            f"registered={combined/(1024**3):.2f}GiB original={total_bytes/(1024**3):.2f}GiB"
        )
    if combined < int(total_bytes * 0.92):
        raise RuntimeError(
            "H3VM Dev8.2 tensor-shard tree is unexpectedly small; refusing a potentially incomplete model: "
            f"registered={combined/(1024**3):.2f}GiB original={total_bytes/(1024**3):.2f}GiB"
        )

    if primary_tree_bytes > int(placement.primary_capacity):
        raise RuntimeError(
            f"H3VM Dev8.2 primary registered weights {primary_tree_bytes/(1024**3):.2f}GiB exceed "
            f"safe capacity {placement.primary_capacity/(1024**3):.2f}GiB. Reduce primary_ffn_groups or TP scope."
        )
    if secondary_tree_bytes > int(placement.secondary_capacity):
        raise RuntimeError(
            f"H3VM Dev8.2 secondary registered weights {secondary_tree_bytes/(1024**3):.2f}GiB exceed "
            f"safe capacity {placement.secondary_capacity/(1024**3):.2f}GiB. Increase primary_ffn_groups or reduce TP scope."
        )

    patcher.set_additional_models("h3vm_global_tp_physical_shard", [shard_patcher])
    if helper_patchers:
        patcher.set_additional_models("h3vm_global_tp_helpers", helper_patchers)
    patcher.set_attachments("h3vm_global_router", router)
    patcher.set_attachments("h3vm_global_space", space)
    patcher.set_attachments("h3vm_global_secondary", shard_patcher)
    patcher.set_attachments("h3vm_tp_fabric", fabric)
    patcher.set_attachments("h3vm_tp_blocks", tp_blocks)
    patcher.set_attachments("h3vm_tp_geometry", geometry)
    patcher.set_attachments("h3vm_tp_helpers", {
        "primary": primary_helper_patcher,
        "secondary": secondary_helper_patcher,
    })

    return {
        "router": router,
        "shard_patcher": shard_patcher,
        "fabric": fabric,
        "geometry": geometry,
        "tp_blocks": tp_blocks,
        "primary_main_bytes": primary_main_bytes,
        "secondary_block_bytes": secondary_block_bytes,
        "primary_helper_bytes": primary_helper_bytes,
        "secondary_helper_bytes": secondary_helper_bytes,
        "primary_tree_bytes": primary_tree_bytes,
        "secondary_tree_bytes": secondary_tree_bytes,
        "combined_bytes": combined,
        "tp_plan": tp_plan,
        "requested_primary_ffn_groups": int(primary_ffn_groups),
        "effective_primary_ffn_groups": effective_ffn_groups,
    }


def build_h3_global_tensor_fabric(*, unet_name: str, primary_device: str, secondary_device: str,
                                  split_mode: str, secondary_blocks: int,
                                  primary_reserve_gb: float, secondary_reserve_gb: float,
                                  safe_profile: bool, transport_mode: str,
                                  benchmark_transport: bool, benchmark_mb: int,
                                  benchmark_repeats: int, allow_explicit_pinned: bool,
                                  pinned_ring_mb: int, secondary_prefetch: bool,
                                  tp_scope: str, primary_ffn_groups: int,
                                  tp_prefetch: bool, tp_telemetry: bool):
    """Dev8 Global Tensor Fabric.

    Dev7 proved device-agnostic ownership and a 16GB+8GB one-copy H3. Dev8 takes
    the next step: selected H3 FC1 layers are physically row-sharded in packed
    ConvRot W4A4 form and both CUDA accelerators execute those FC1 shards
    concurrently. No P2P is required; GlobalMemorySpace supplies the helper input
    replica and returns the helper FC1 rows. The untouched full FC2 stays on the
    block owner to preserve the original full-K activation-scale semantics.

    The default boundary_2 scope is a surgical correctness/performance probe.
    all_blocks is intentionally available only after that probe is healthy.
    """
    import torch
    from comfy.cli_args import args
    from .global_memory import GlobalMemorySpace

    primary, secondary = _common_preflight(primary_device, secondary_device)
    if safe_profile and not bool(getattr(args, "disable_pinned_memory", False)):
        raise RuntimeError(
            "H3VM Dev8.2 safe profile requires --disable-pinned-memory before Global Tensor Fabric initialization."
        )
    if pinned_ring_mb > 128 and allow_explicit_pinned:
        raise RuntimeError("H3VM Dev8.2 caps the explicit pinned runway at 128MB per slot.")

    LOG.info(
        "H3VM Dev8.2 GLOBAL TENSOR private load | model=%s | primary=%s (%s) | secondary=%s (%s) | "
        "pinned_disabled=%s transport=%s tp_scope=%s primary_ffn_groups=%d",
        unet_name, primary, torch.cuda.get_device_name(primary),
        secondary, torch.cuda.get_device_name(secondary),
        bool(getattr(args, "disable_pinned_memory", False)), transport_mode,
        tp_scope, int(primary_ffn_groups),
    )

    space = GlobalMemorySpace(
        primary, secondary,
        transport_mode=transport_mode,
        benchmark_mb=int(benchmark_mb if benchmark_transport else 8),
        benchmark_repeats=int(benchmark_repeats if benchmark_transport else 1),
        allow_explicit_pinned=bool(allow_explicit_pinned),
        pinned_ring_mb=int(pinned_ring_mb),
        pinned_ring_slots=2,
    )
    r = space.transport.report()
    b = r["benchmark"]
    LOG.info(
        "H3VM Dev8.2 transport | selected=%s requested=%s | P2P=%s/%s | "
        "direct=%.2f/%.2fGB/s neutral_pageable=%.2f/%.2fGB/s neutral_pinned=%s/%s",
        r["selected_mode"], r["requested_mode"], b["peer_ab"], b["peer_ba"],
        float(b["direct_d2d"]["ab"] or 0.0), float(b["direct_d2d"]["ba"] or 0.0),
        float(b["neutral_pageable"]["ab"] or 0.0), float(b["neutral_pageable"]["ba"] or 0.0),
        "%.2f" % b["neutral_pinned"]["ab"] if b["neutral_pinned"]["ab"] is not None else "off",
        "%.2f" % b["neutral_pinned"]["ba"] if b["neutral_pinned"]["ba"] is not None else "off",
    )

    patcher, dm = _load_private_h3(unet_name, primary, safe_profile)
    placement, count, block_bytes, total_bytes, blocks_total, fixed_bytes = _choose_placement(
        dm=dm, patcher=patcher, primary=primary, secondary=secondary,
        split_mode=split_mode, secondary_blocks=secondary_blocks,
        primary_reserve_gb=primary_reserve_gb, secondary_reserve_gb=secondary_reserve_gb,
    )

    out = _partition_global_h3_tp(
        patcher=patcher, dm=dm, primary=primary, secondary=secondary,
        count=count, block_bytes=block_bytes, total_bytes=total_bytes,
        space=space, placement=placement,
        tp_scope=tp_scope, primary_ffn_groups=int(primary_ffn_groups),
        tp_prefetch=bool(tp_prefetch), secondary_prefetch=bool(secondary_prefetch),
        telemetry=bool(tp_telemetry),
    )
    g = out["geometry"]
    tp_blocks = out["tp_blocks"]
    if out["effective_primary_ffn_groups"] != out["requested_primary_ffn_groups"]:
        LOG.info(
            "H3VM Dev8.2 TP placement adjusted safely | requested=%d effective=%d",
            out["requested_primary_ffn_groups"], out["effective_primary_ffn_groups"],
        )

    patcher.set_attachments("h3vm_dev8_config", {
        "architecture": "GLOBAL_TENSOR_FABRIC",
        "transport_mode": transport_mode,
        "selected_transport": space.transport.selected_mode,
        "split_mode": split_mode,
        "secondary_prefetch": secondary_prefetch,
        "tp_scope": tp_scope,
        "tp_blocks": tp_blocks,
        "primary_ffn_groups_requested": int(primary_ffn_groups),
        "primary_ffn_groups_effective": int(out["effective_primary_ffn_groups"]),
        "tp_prefetch": bool(tp_prefetch),
    })

    LOG.info(
        "H3VM Dev8.2 FC1 TP geometry | layout=%s scale=%s convrot=%s | hidden=%d ffn=%d group=%d groups=%d | "
        "primary=%d groups/%d channels (%.1f%%) secondary=%d channels | blocks=%s",
        g.get("fc1_layout", "off"), g.get("fc1_scale_mode", "n/a"), g.get("fc1_convrot", False),
        g["hidden"], g["ffn"], g["group"], g["groups"],
        g["primary_groups"], g["primary_ffn"],
        (100.0 * g["primary_ffn"] / max(1, g["ffn"])), g["secondary_ffn"], tp_blocks,
    )
    LOG.info(
        "H3VM Dev8.2 registered fabric | original=%.2fGiB | primary main=%.2f helper=%.2f total=%.2fGiB | "
        "secondary blocks=%.2f helper=%.2f total=%.2fGiB | combined=%.2fGiB",
        total_bytes/(1024**3),
        out["primary_main_bytes"]/(1024**3), out["primary_helper_bytes"]/(1024**3), out["primary_tree_bytes"]/(1024**3),
        out["secondary_block_bytes"]/(1024**3), out["secondary_helper_bytes"]/(1024**3), out["secondary_tree_bytes"]/(1024**3),
        out["combined_bytes"]/(1024**3),
    )
    LOG.info(
        "H3VM Dev8.2 safe capacities | primary=%.2fGiB secondary=%.2fGiB | CPU free=%.1fGiB | logical_owner=GlobalMemorySpace",
        placement.primary_capacity/(1024**3), placement.secondary_capacity/(1024**3),
        space.compute_blocks[0].free_gib(),
    )
    print(
        f"[H3VM DEV8.2 GLOBAL TENSOR READY] transport={space.transport.selected_mode} | "
        f"whole-block split secondary=0-{count-1} | TP blocks={tp_blocks} | "
        f"FC1 primary/secondary={g['primary_ffn']}/{g['secondary_ffn']} channels | "
        f"weights={out['primary_tree_bytes']/(1024**3):.2f}+{out['secondary_tree_bytes']/(1024**3):.2f}GiB",
        flush=True,
    )
    return patcher


def build_h3_snapshot_islands(*, unet_name: str, primary_device: str, secondary_device: str,
                              split_mode: str, secondary_blocks: int,
                              primary_reserve_gb: float, secondary_reserve_gb: float,
                              safe_profile: bool, expected_steps: int,
                              refresh_interval: int, exact_last_step: bool,
                              prefix_prefetch: bool, tail_prefetch: bool,
                              telemetry: bool):
    """Dev9 experimental stale-snapshot compute islands.

    This is deliberately *not* exact tensor parallelism. There is no CUDA
    device-to-device activation handoff. The two GPUs are independent compute
    islands attached to a pageable-RAM snapshot mailbox. After an exact warm-up,
    the primary tail consumes the previous model-call boundary snapshot while
    the secondary computes the current prefix, allowing both accelerators to
    execute concurrently on a single H3 denoise stream.
    """
    import torch
    import comfy.model_management
    from comfy.cli_args import args
    from .global_memory import GlobalMemorySpace
    from .snapshot_islands import (
        SnapshotIslandRuntime,
        PrefixProxyFactory,
        TailProxyFactory,
        make_island_root,
    )

    primary, secondary = _common_preflight(primary_device, secondary_device)
    if safe_profile and not bool(getattr(args, "disable_pinned_memory", False)):
        raise RuntimeError(
            "H3VM Dev9 safe profile requires --disable-pinned-memory before Snapshot Islands initialization."
        )
    expected_steps = max(1, int(expected_steps))
    refresh_interval = max(0, int(refresh_interval))

    LOG.warning(
        "H3VM Dev9 SNAPSHOT ISLANDS is an APPROXIMATE lab mode: one-call-stale boundary features are used "
        "between exact refreshes. Compare output quality against Dev7/Dev8 before production use."
    )
    LOG.info(
        "H3VM Dev9 private load | model=%s | primary=%s (%s) | secondary=%s (%s) | "
        "host_only=True transport=neutral_pageable expected_steps=%d refresh_interval=%d exact_last=%s",
        unet_name, primary, torch.cuda.get_device_name(primary),
        secondary, torch.cuda.get_device_name(secondary),
        expected_steps, refresh_interval, bool(exact_last_step),
    )

    # The defining invariant of this experiment: transport may touch CPU RAM and
    # either GPU independently, but it never probes or executes a GPU-to-GPU copy.
    space = GlobalMemorySpace(
        primary, secondary,
        transport_mode="neutral_pageable",
        benchmark_mb=16,
        benchmark_repeats=1,
        allow_explicit_pinned=False,
        pinned_ring_mb=32,
        pinned_ring_slots=1,
        host_only=True,
    )
    report = space.transport.report()
    b = report["benchmark"]
    LOG.info(
        "H3VM Dev9 HOST-ONLY transport | selected=%s host_only=%s | P2P=%s/%s | "
        "direct=SKIPPED/SKIPPED neutral_pageable=%.2f/%.2fGB/s",
        report["selected_mode"], report.get("host_only", False),
        b["peer_ab"], b["peer_ba"],
        float(b["neutral_pageable"]["ab"] or 0.0),
        float(b["neutral_pageable"]["ba"] or 0.0),
    )

    patcher, dm = _load_private_h3(unet_name, primary, safe_profile)
    placement, count, block_bytes, total_bytes, blocks_total, fixed_bytes = _choose_placement(
        dm=dm, patcher=patcher, primary=primary, secondary=secondary,
        split_mode=split_mode, secondary_blocks=secondary_blocks,
        primary_reserve_gb=primary_reserve_gb, secondary_reserve_gb=secondary_reserve_gb,
    )
    if count <= 0 or count >= len(dm.blocks):
        raise RuntimeError(f"H3VM Dev9 requires a non-empty prefix and tail; split_count={count}")

    original_blocks = list(dm.blocks)
    prefix_blocks = {i: original_blocks[i] for i in range(count)}
    tail_blocks = {i: original_blocks[i] for i in range(count, len(original_blocks))}

    runtime = SnapshotIslandRuntime(
        primary, secondary,
        prefix_blocks, tail_blocks, count,
        space=space,
        expected_steps=expected_steps,
        refresh_interval=refresh_interval,
        exact_last_step=bool(exact_last_step),
        prefix_prefetch=bool(prefix_prefetch),
        tail_prefetch=bool(tail_prefetch),
        telemetry=bool(telemetry),
    )

    # Replace the entire DiT loop with lightweight island proxies. Actual block
    # weights live only in their island roots, preserving one physical H3 copy.
    for i in range(count):
        dm.blocks[i] = PrefixProxyFactory.make(runtime, i)
    for i in range(count, len(original_blocks)):
        dm.blocks[i] = TailProxyFactory.make(runtime, i)

    prefix_root = make_island_root(patcher.model, dm, prefix_blocks, len(original_blocks), "secondary-prefix")
    tail_root = make_island_root(patcher.model, dm, tail_blocks, len(original_blocks), "primary-tail")
    Patcher = patcher.__class__
    prefix_patcher = Patcher(
        prefix_root,
        load_device=secondary,
        offload_device=torch.device("cpu"),
        size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )
    tail_patcher = Patcher(
        tail_root,
        load_device=primary,
        offload_device=torch.device("cpu"),
        size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )

    for p in (patcher, prefix_patcher, tail_patcher):
        p.size = 0
        p.cached_patcher_init = None

    turbo_counts = None
    if turbo_plan is not None:
        from .turbo_compat import apply_turbo_plan_to_islands
        turbo_counts = apply_turbo_plan_to_islands(
            plan=turbo_plan, main_patcher=patcher, dm=dm,
            prefix_patcher=prefix_patcher, tail_patcher=tail_patcher, split_count=count,
            primary_device=primary, secondary_device=secondary,
        )

    main_fixed_bytes = int(comfy.model_management.module_size(patcher.model))
    prefix_tree_bytes = int(comfy.model_management.module_size(prefix_root))
    tail_tree_bytes = int(comfy.model_management.module_size(tail_root))
    primary_registered = main_fixed_bytes + tail_tree_bytes
    secondary_registered = prefix_tree_bytes
    combined = primary_registered + secondary_registered

    if combined > int(total_bytes * 1.03) or combined < int(total_bytes * 0.97):
        raise RuntimeError(
            "H3VM Dev9 island partition failed one-copy accounting: "
            f"registered={combined/(1024**3):.2f}GiB original={total_bytes/(1024**3):.2f}GiB"
        )
    if primary_registered > int(placement.primary_capacity):
        raise RuntimeError(
            f"H3VM Dev9 primary island weights {primary_registered/(1024**3):.2f}GiB exceed "
            f"safe capacity {placement.primary_capacity/(1024**3):.2f}GiB"
        )
    if secondary_registered > int(placement.secondary_capacity):
        raise RuntimeError(
            f"H3VM Dev9 secondary island weights {secondary_registered/(1024**3):.2f}GiB exceed "
            f"safe capacity {placement.secondary_capacity/(1024**3):.2f}GiB"
        )

    patcher.set_additional_models("h3vm_snapshot_islands", [prefix_patcher, tail_patcher])
    patcher.set_attachments("h3vm_snapshot_runtime", runtime)
    patcher.set_attachments("h3vm_global_space", space)
    patcher.set_attachments("h3vm_snapshot_prefix", prefix_patcher)
    patcher.set_attachments("h3vm_snapshot_tail", tail_patcher)
    patcher.set_attachments("h3vm_dev9_config", {
        "architecture": "SNAPSHOT_COMPUTE_ISLANDS",
        "exact": False,
        "host_only": True,
        "transport": "neutral_pageable",
        "split_count": count,
        "expected_steps": expected_steps,
        "refresh_interval": refresh_interval,
        "exact_last_step": bool(exact_last_step),
    })

    LOG.info(
        "H3VM Dev9 registered islands | original=%.2fGiB fixed=%.2fGiB | primary tail+fixed=%.2fGiB "
        "secondary prefix=%.2fGiB | split=%d/%d | CPU free=%.1fGiB",
        total_bytes/(1024**3), fixed_bytes/(1024**3),
        primary_registered/(1024**3), secondary_registered/(1024**3),
        count, len(original_blocks)-count,
        next((x.free_gib() for x in space.compute_blocks if x.kind == "cpu"), 0.0),
    )
    print(
        f"[H3VM DEV9 SNAPSHOT ISLANDS READY] NO-D2D runtime | "
        f"{secondary}: blocks 0-{count-1} current prefix | "
        f"{primary}: blocks {count}-49 stale/exact tail | "
        f"weights {primary_registered/(1024**3):.2f}+{secondary_registered/(1024**3):.2f}GiB | "
        f"expected_steps={expected_steps} refresh={refresh_interval}",
        flush=True,
    )
    return patcher


def _choose_full_throttle_placement(*, dm, patcher, primary, secondary,
                                    primary_reserve_gb: float, secondary_reserve_gb: float,
                                    secondary_overcommit_mb: int, requested_secondary_blocks: int = 0, secondary_ram_backing_gb: float = 0.0):
    """Choose a compute-balanced contiguous prefix for Snapshot Islands.

    Dev9's ordinary planner was deliberately conservative and required every
    registered prefix weight byte to fit inside the secondary fast-pool reserve.
    Snapshot Islands + DynamicVRAM can safely experiment with a small elastic
    prefix overcommit because weights remain CPU-backed and the native VBAR
    prefetch/eviction path owns physical residency.

    This helper never overcommits the primary card. On the secondary card the
    user supplied overcommit only reduces the nominal reserve, and a hard 0.50GiB
    workspace reserve is always preserved in Dev9.2 redline mode.
    """
    import torch
    import comfy.model_management
    from .planner import Placement

    GIB = 1024 ** 3
    MIB = 1024 ** 2
    block_bytes = [int(comfy.model_management.module_size(b)) for b in dm.blocks]
    total_bytes = int(comfy.model_management.module_size(patcher.model))
    blocks_total = sum(block_bytes)
    fixed_bytes = max(0, total_bytes - blocks_total)

    pp = torch.cuda.get_device_properties(primary)
    sp = torch.cuda.get_device_properties(secondary)
    ptotal = int(pp.total_memory)
    stotal = int(sp.total_memory)
    pcap = max(0, int(ptotal - float(primary_reserve_gb) * GIB))
    nominal_scap = max(0, int(stotal - float(secondary_reserve_gb) * GIB))

    overcommit = max(0, int(secondary_overcommit_mb)) * MIB
    hard_reserve = max(int(0.50 * GIB), int(float(secondary_reserve_gb) * GIB) - overcommit)
    elastic_scap = max(0, stotal - hard_reserve)
    ram_backing = max(0, int(float(secondary_ram_backing_gb) * GIB))
    registered_scap = elastic_scap + ram_backing

    def score(props):
        sm = max(1, int(getattr(props, "multi_processor_count", 1)))
        clock = max(1, int(getattr(props, "clock_rate", 1)))
        return float(sm * clock)

    pscore = score(pp)
    sscore = score(sp)
    ideal = int(round(len(block_bytes) * sscore / max(1e-9, pscore + sscore)))
    ideal = max(1, min(len(block_bytes) - 1, ideal))
    requested = max(0, int(requested_secondary_blocks))
    target = requested if requested > 0 else ideal

    candidates = []
    running = 0
    for n in range(1, len(block_bytes)):
        running += block_bytes[n - 1]
        sbytes = running
        pbytes = blocks_total - sbytes
        pused = fixed_bytes + pbytes
        if pused <= pcap and sbytes <= registered_scap:
            # Closest compute target wins. If equally close, prefer the larger
            # secondary prefix because the measured 16G+8G H3 workload is
            # typically primary-tail limited.
            candidates.append((abs(n - target), -n, n, sbytes, pbytes))

    if not candidates:
        raise RuntimeError(
            "H3VM Dev9.2 Redline found no legal split even with the selected "
            "secondary elastic overcommit. Reduce reserves or overcommit less aggressively."
        )

    _, _, count, sbytes, pbytes = min(candidates)
    placement = Placement(
        count, sbytes, pbytes, fixed_bytes,
        pcap, registered_scap,
    )
    meta = {
        "ideal_blocks": ideal,
        "target_blocks": target,
        "primary_score": pscore,
        "secondary_score": sscore,
        "nominal_secondary_capacity": nominal_scap,
        "elastic_secondary_capacity": elastic_scap,
        "registered_secondary_capacity": registered_scap,
        "secondary_ram_backing_bytes": ram_backing,
        "effective_secondary_reserve_gb": hard_reserve / GIB,
        "secondary_overcommit_mb": int(secondary_overcommit_mb),
    }
    return placement, count, block_bytes, total_bytes, blocks_total, fixed_bytes, meta


def build_h3_snapshot_islands_full_throttle(*, unet_name: str, primary_device: str, secondary_device: str,
                                             primary_reserve_gb: float, secondary_reserve_gb: float,
                                             secondary_overcommit_mb: int, secondary_blocks_target: int,
                                             safe_profile: bool, expected_steps: int,
                                             refresh_interval: int, exact_last_step: bool,
                                             predictor_mode: str, predictor_beta: float,
                                             prefix_prefetch: bool, tail_prefetch: bool,
                                             launch_tail_before_stage: bool, host_feeder: str,
                                             pinned_mailbox_mb: int, secondary_ram_backing_gb: float, telemetry: bool,
                                             turbo_lora_name: str | None = None, turbo_strength: float = 1.0,
                                             turbo_low_vram: bool = False, exact_steps=None, quality_profile: str = "FAST | E-S-P-E"):
    """Dev9.1 Full Throttle Snapshot Compute Islands.

    Goals:
      * keep NO-D2D semantics;
      * move the stale tail launch ahead of current-prefix host staging;
      * use a compute-balanced, slightly elastic secondary prefix so the weaker
        GPU spends less time idle while the primary tail finishes;
      * optionally extrapolate the boundary snapshot from the previous two
        denoise calls, reducing stale drift without forcing an exact barrier.
    """
    import torch
    import comfy.model_management
    from comfy.cli_args import args
    from .global_memory import GlobalMemorySpace
    from .snapshot_islands import (
        SnapshotIslandRuntime,
        PrefixProxyFactory,
        TailProxyFactory,
        make_island_root,
    )

    primary, secondary = _common_preflight(primary_device, secondary_device)
    if safe_profile and not bool(getattr(args, "disable_pinned_memory", False)):
        raise RuntimeError(
            "H3VM Dev9.1 safe profile requires --disable-pinned-memory before Full Throttle initialization."
        )
    expected_steps = max(1, int(expected_steps))
    refresh_interval = max(0, int(refresh_interval))
    predictor_mode = str(predictor_mode)
    if predictor_mode not in ("stale", "linear"):
        raise RuntimeError(f"H3VM Dev9.1 unsupported predictor_mode={predictor_mode!r}")
    predictor_beta = float(predictor_beta)

    host_feeder = str(host_feeder)
    if host_feeder not in ("pageable", "bounded_pinned"):
        raise RuntimeError(f"H3VM Dev9.3.1 unsupported host_feeder={host_feeder!r}")
    pinned_mailbox_mb = max(0, min(2048, int(pinned_mailbox_mb)))

    LOG.warning(
        "H3VM Dev9.4 RAM-BACKED REDLINE remains APPROXIMATE: stale/predicted boundary features are used "
        "between exact refreshes. Validate quality against REFRESH5/Dev7."
    )
    LOG.info(
        "H3VM Dev9.4 private load | model=%s | primary=%s (%s) | secondary=%s (%s) | "
        "NO-D2D host_only=True | refresh=%d predictor=%s beta=%.2f | overcommit=%dMiB | feeder=%s pinned_cap=%dMiB",
        unet_name, primary, torch.cuda.get_device_name(primary),
        secondary, torch.cuda.get_device_name(secondary),
        refresh_interval, predictor_mode, predictor_beta, int(secondary_overcommit_mb),
        host_feeder, pinned_mailbox_mb,
    )

    space = GlobalMemorySpace(
        primary, secondary,
        transport_mode="neutral_pageable",
        benchmark_mb=16,
        benchmark_repeats=1,
        allow_explicit_pinned=False,
        pinned_ring_mb=32,
        pinned_ring_slots=1,
        host_only=True,
    )
    report = space.transport.report()
    b = report["benchmark"]
    LOG.info(
        "H3VM Dev9.2 HOST-ONLY transport | selected=%s | P2P=%s/%s | direct=SKIPPED/SKIPPED "
        "neutral_pageable=%.2f/%.2fGB/s",
        report["selected_mode"], b["peer_ab"], b["peer_ba"],
        float(b["neutral_pageable"]["ab"] or 0.0), float(b["neutral_pageable"]["ba"] or 0.0),
    )

    patcher, dm = _load_private_h3(unet_name, primary, safe_profile)
    turbo_plan = None
    if turbo_lora_name:
        from .turbo_compat import prepare_turbo_plan
        turbo_plan = prepare_turbo_plan(
            patcher, dm, str(turbo_lora_name), float(turbo_strength), bool(turbo_low_vram)
        )
    placement, count, block_bytes, total_bytes, blocks_total, fixed_bytes, balance = _choose_full_throttle_placement(
        dm=dm, patcher=patcher, primary=primary, secondary=secondary,
        primary_reserve_gb=primary_reserve_gb,
        secondary_reserve_gb=secondary_reserve_gb,
        secondary_overcommit_mb=int(secondary_overcommit_mb),
        requested_secondary_blocks=int(secondary_blocks_target),
        secondary_ram_backing_gb=float(secondary_ram_backing_gb),
    )
    if count <= 0 or count >= len(dm.blocks):
        raise RuntimeError(f"H3VM Dev9.1 requires non-empty islands; split_count={count}")

    LOG.info(
        "H3VM Dev9.2 compute balance | theoretical target=%d requested=%s -> effective split=%d/%d | "
        "score primary/secondary=%.0f/%.0f | secondary nominal/elastic cap=%.2f/%.2fGiB "
        "effective reserve=%.2fGiB | RAM backing=%.2fGiB registered cap=%.2fGiB",
        balance["ideal_blocks"],
        str(balance["target_blocks"]) if int(secondary_blocks_target) > 0 else "auto",
        count, len(dm.blocks) - count,
        balance["primary_score"], balance["secondary_score"],
        balance["nominal_secondary_capacity"]/(1024**3),
        balance["elastic_secondary_capacity"]/(1024**3),
        balance["effective_secondary_reserve_gb"],
        balance["secondary_ram_backing_bytes"]/(1024**3),
        balance["registered_secondary_capacity"]/(1024**3),
    )

    original_blocks = list(dm.blocks)
    prefix_blocks = {i: original_blocks[i] for i in range(count)}
    tail_blocks = {i: original_blocks[i] for i in range(count, len(original_blocks))}

    runtime = SnapshotIslandRuntime(
        primary, secondary,
        prefix_blocks, tail_blocks, count,
        space=space,
        expected_steps=expected_steps,
        refresh_interval=refresh_interval,
        exact_last_step=bool(exact_last_step),
        prefix_prefetch=bool(prefix_prefetch),
        tail_prefetch=bool(tail_prefetch),
        telemetry=bool(telemetry),
        predictor_mode=predictor_mode,
        predictor_beta=predictor_beta,
        launch_tail_before_stage=bool(launch_tail_before_stage),
        host_feeder=host_feeder,
        pinned_mailbox_mb=pinned_mailbox_mb,
        exact_steps=exact_steps,
        quality_profile=quality_profile,
    )

    for i in range(count):
        dm.blocks[i] = PrefixProxyFactory.make(runtime, i)
    for i in range(count, len(original_blocks)):
        dm.blocks[i] = TailProxyFactory.make(runtime, i)

    prefix_root = make_island_root(patcher.model, dm, prefix_blocks, len(original_blocks), "secondary-prefix-full-throttle")
    tail_root = make_island_root(patcher.model, dm, tail_blocks, len(original_blocks), "primary-tail-full-throttle")
    Patcher = patcher.__class__
    prefix_patcher = Patcher(
        prefix_root, load_device=secondary, offload_device=torch.device("cpu"), size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )
    tail_patcher = Patcher(
        tail_root, load_device=primary, offload_device=torch.device("cpu"), size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )
    for p in (patcher, prefix_patcher, tail_patcher):
        p.size = 0
        p.cached_patcher_init = None

    turbo_counts = None
    if turbo_plan is not None:
        from .turbo_compat import apply_turbo_plan_to_islands
        turbo_counts = apply_turbo_plan_to_islands(
            plan=turbo_plan, main_patcher=patcher, dm=dm,
            prefix_patcher=prefix_patcher, tail_patcher=tail_patcher, split_count=count,
            primary_device=primary, secondary_device=secondary,
        )

    main_fixed_bytes = int(comfy.model_management.module_size(patcher.model))
    prefix_tree_bytes = int(comfy.model_management.module_size(prefix_root))
    tail_tree_bytes = int(comfy.model_management.module_size(tail_root))
    primary_registered = main_fixed_bytes + tail_tree_bytes
    secondary_registered = prefix_tree_bytes
    combined = primary_registered + secondary_registered

    if combined > int(total_bytes * 1.03) or combined < int(total_bytes * 0.97):
        raise RuntimeError(
            "H3VM Dev9.2 partition failed one-copy accounting: "
            f"registered={combined/(1024**3):.2f}GiB original={total_bytes/(1024**3):.2f}GiB"
        )
    if primary_registered > int(placement.primary_capacity):
        raise RuntimeError(
            f"H3VM Dev9.2 primary island weights {primary_registered/(1024**3):.2f}GiB exceed "
            f"capacity {placement.primary_capacity/(1024**3):.2f}GiB"
        )
    if secondary_registered > int(placement.secondary_capacity):
        raise RuntimeError(
            f"H3VM Dev9.2 secondary island weights {secondary_registered/(1024**3):.2f}GiB exceed "
            f"elastic capacity {placement.secondary_capacity/(1024**3):.2f}GiB"
        )

    patcher.set_additional_models("h3vm_snapshot_islands_full_throttle", [prefix_patcher, tail_patcher])
    patcher.set_attachments("h3vm_snapshot_runtime", runtime)
    patcher.set_attachments("h3vm_global_space", space)
    patcher.set_attachments("h3vm_snapshot_prefix", prefix_patcher)
    patcher.set_attachments("h3vm_snapshot_tail", tail_patcher)
    patcher.set_attachments("h3vm_dev9_1_config", {
        "architecture": "SNAPSHOT_COMPUTE_ISLANDS_FULL_THROTTLE",
        "host_only": True,
        "transport": "neutral_pageable",
        "split_count": count,
        "refresh_interval": refresh_interval,
        "predictor_mode": predictor_mode,
        "predictor_beta": predictor_beta,
        "quality_profile": quality_profile,
        "exact_steps": sorted(int(x) for x in (exact_steps or ())),
        "launch_tail_before_stage": bool(launch_tail_before_stage),
        "secondary_overcommit_mb": int(secondary_overcommit_mb),
        "effective_secondary_reserve_gb": balance["effective_secondary_reserve_gb"],
        "host_feeder": host_feeder,
        "pinned_mailbox_mb": pinned_mailbox_mb,
        "secondary_ram_backing_gb": float(secondary_ram_backing_gb),
        "turbo_lora_name": str(turbo_lora_name) if turbo_lora_name else None,
        "turbo_strength": float(turbo_strength),
        "turbo_low_vram": bool(turbo_low_vram),
        "turbo_counts": turbo_counts,
    })

    LOG.info(
        "H3VM Dev9.2 registered islands | original=%.2fGiB fixed=%.2fGiB | primary=%.2fGiB secondary=%.2fGiB | "
        "split=%d/%d | CPU free=%.1fGiB",
        total_bytes/(1024**3), fixed_bytes/(1024**3),
        primary_registered/(1024**3), secondary_registered/(1024**3),
        count, len(original_blocks)-count,
        next((x.free_gib() for x in space.compute_blocks if x.kind == "cpu"), 0.0),
    )
    print(
        f"[H3VM DEV9.4 RAM BACKED READY] NO-D2D | "
        f"{secondary}: blocks 0-{count-1} | {primary}: blocks {count}-49 | "
        f"refresh={refresh_interval} predictor={predictor_mode}:{predictor_beta:.2f} | "
        f"feeder={host_feeder} pinned_cap={pinned_mailbox_mb}MiB RAM_backing={float(secondary_ram_backing_gb):.2f}GiB | "
        f"weights {primary_registered/(1024**3):.2f}+{secondary_registered/(1024**3):.2f}GiB"
        + (f" | TURBO={turbo_lora_name} strength={float(turbo_strength):.2f} mode={'merge' if turbo_low_vram else 'bypass'}" if turbo_lora_name else ""),
        flush=True,
    )
    return patcher


def build_h3_primary_first_exact_turbo(*, unet_name: str, turbo_lora_name: str, turbo_strength: float,
                                        turbo_low_vram: bool, primary_device: str, secondary_device: str,
                                        primary_runtime_target_pct: float = 90.0,
                                        primary_workspace_reserve_gb: float = 1.25,
                                        secondary_runtime_target_pct: float = 88.0,
                                        secondary_workspace_reserve_gb: float = 0.75,
                                        cut_override: int = 0, expected_steps: int = 4,
                                        safe_profile: bool = True, prefix_prefetch: bool = True,
                                        tail_prefetch: bool = False, lazy_secondary: bool = True,
                                        hard_cleanup_after_sample: bool = False, telemetry: bool = True):
    """Dev11.0 exact primary-first / lazy-secondary H3 Turbo runtime.

    This intentionally removes stale/predictor semantics. It keeps the original
    block order exact and changes only placement/load timing:
      * primary (normally 16G) owns the contiguous prefix and main fixed layers;
      * secondary (normally 8G) owns the remaining tail;
      * only the primary island is an additional model at sampler startup;
      * the secondary tail is loaded lazily in a worker after primary compute begins.
    """
    import torch
    import comfy.model_management as mm
    import comfy.patcher_extension
    from comfy.cli_args import args
    from .global_memory import GlobalMemorySpace
    from .snapshot_islands import make_island_root
    from .primary_first_exact import (
        PrimaryFirstExactRuntime,
        PrimaryPrefixProxyFactory,
        SecondaryTailProxyFactory,
    )

    primary, secondary = _common_preflight(primary_device, secondary_device)
    if safe_profile and not bool(getattr(args, "disable_pinned_memory", False)):
        raise RuntimeError("H3VM Dev11 safe profile requires --disable-pinned-memory")

    # Keep old global/pinned experiments out of the new exact branch.
    try:
        import comfy.model_prefetch as mp
        mp.cleanup_prefetch_queues()
    except Exception:
        pass

    patcher, dm = _load_private_h3(unet_name, primary, safe_profile)

    from .turbo_compat import prepare_turbo_plan
    turbo_plan = prepare_turbo_plan(
        patcher, dm, str(turbo_lora_name), float(turbo_strength), bool(turbo_low_vram)
    )

    block_bytes = [int(mm.module_size(b)) for b in dm.blocks]
    total_bytes = int(mm.module_size(patcher.model))
    blocks_total = sum(block_bytes)
    fixed_bytes = max(0, total_bytes - blocks_total)
    GIB = 1024 ** 3
    ptotal = int(torch.cuda.get_device_properties(primary).total_memory)
    stotal = int(torch.cuda.get_device_properties(secondary).total_memory)

    p_target = max(50.0, min(96.0, float(primary_runtime_target_pct))) / 100.0
    s_target = max(50.0, min(94.0, float(secondary_runtime_target_pct))) / 100.0
    p_weight_cap = max(0, int(ptotal * p_target - float(primary_workspace_reserve_gb) * GIB))
    s_weight_cap = max(0, int(stotal * s_target - float(secondary_workspace_reserve_gb) * GIB))

    # Dev11.0.1 planner policy:
    #   * keep the fragile 8G card on its requested budget;
    #   * if no cut exists by only a small accounting margin, relax ONLY the
    #     16G primary target in 0.5%% steps (up to +3%% / 93%% hard ceiling);
    #   * still preserve the explicit workspace reserve.
    # This matches the primary-first design: feed the large card a little more
    # rather than pushing the small card onto the VRAM cliff.
    requested_p_target = p_target
    effective_p_target = p_target
    planner_relaxed = False

    def _candidates_for_caps(pcap, scap):
        out = []
        prefix_sum = 0
        for n in range(1, len(block_bytes)):
            prefix_sum += block_bytes[n - 1]
            pb = fixed_bytes + prefix_sum
            sb = blocks_total - prefix_sum
            if pb <= pcap and sb <= scap:
                out.append((n, pb, sb))
        return out

    if int(cut_override) > 0:
        cut = max(1, min(49, int(cut_override)))
        pbytes = fixed_bytes + sum(block_bytes[:cut])
        sbytes = sum(block_bytes[cut:])
        if pbytes > p_weight_cap or sbytes > s_weight_cap:
            raise RuntimeError(
                "H3VM Dev11 cut_override violates runtime target budget: "
                f"cut={cut} primary={pbytes/GIB:.2f}/{p_weight_cap/GIB:.2f}GiB "
                f"secondary={sbytes/GIB:.2f}/{s_weight_cap/GIB:.2f}GiB. "
                "Raise the primary runtime target or lower its workspace reserve; "
                "do not lower the target percentage."
            )
    else:
        candidates = _candidates_for_caps(p_weight_cap, s_weight_cap)

        if not candidates:
            # Auto-relax the large card only. 0.5%% granularity is intentional so
            # a tiny module_size accounting mismatch does not abort before the
            # runtime can even measure real occupancy.
            hard_primary_target = min(0.93, requested_p_target + 0.03)
            trial = requested_p_target + 0.005
            while trial <= hard_primary_target + 1e-9:
                trial_cap = max(0, int(ptotal * trial - float(primary_workspace_reserve_gb) * GIB))
                trial_candidates = _candidates_for_caps(trial_cap, s_weight_cap)
                if trial_candidates:
                    candidates = trial_candidates
                    effective_p_target = trial
                    p_weight_cap = trial_cap
                    planner_relaxed = True
                    break
                trial += 0.005

        if not candidates:
            # Give a useful nearest-cut diagnostic instead of the old misleading
            # advice (lowering a target makes the cap smaller and cannot help).
            scored = []
            prefix_sum = 0
            for n in range(1, len(block_bytes)):
                prefix_sum += block_bytes[n - 1]
                pb = fixed_bytes + prefix_sum
                sb = blocks_total - prefix_sum
                p_over = max(0, pb - p_weight_cap)
                s_over = max(0, sb - s_weight_cap)
                scored.append((p_over + s_over, n, pb, sb, p_over, s_over))
            _, near_n, near_pb, near_sb, near_po, near_so = min(scored, key=lambda x: x[0])
            raise RuntimeError(
                "H3VM Dev11 found no legal primary-first cut even after primary-only auto-relax. "
                f"Nearest cut={near_n}/{len(block_bytes)-near_n}: "
                f"primary={near_pb/GIB:.2f}GiB overflow={near_po/(1024**2):.0f}MiB, "
                f"secondary={near_sb/GIB:.2f}GiB overflow={near_so/(1024**2):.0f}MiB. "
                "Raise the primary runtime target or lower its workspace reserve first; "
                "avoid relaxing the 8G card unless telemetry proves headroom."
            )
        # Feed the large card first: the largest legal prefix wins.
        cut, pbytes, sbytes = max(candidates, key=lambda x: x[0])

    LOG.info(
        "H3VM Dev11.0.1 PRIMARY-FIRST planner | primary requested/effective=%.1f/%.1f%% reserve=%.2fGiB cap=%.2fGiB | "
        "secondary target=%.1f%% reserve=%.2fGiB cap=%.2fGiB | cut=%d/%d weights=%.2f+%.2fGiB | auto_relax=%s",
        requested_p_target * 100.0, effective_p_target * 100.0,
        float(primary_workspace_reserve_gb), p_weight_cap/GIB,
        s_target * 100.0, float(secondary_workspace_reserve_gb), s_weight_cap/GIB,
        cut, len(dm.blocks)-cut, pbytes/GIB, sbytes/GIB, planner_relaxed,
    )

    space = GlobalMemorySpace(
        primary, secondary,
        transport_mode="neutral_pageable",
        benchmark_mb=16,
        benchmark_repeats=1,
        allow_explicit_pinned=False,
        pinned_ring_mb=32,
        pinned_ring_slots=1,
        host_only=True,
    )

    original_blocks = list(dm.blocks)
    prefix_blocks = {i: original_blocks[i] for i in range(cut)}
    tail_blocks = {i: original_blocks[i] for i in range(cut, len(original_blocks))}

    runtime = PrimaryFirstExactRuntime(
        primary, secondary,
        prefix_blocks, tail_blocks, cut,
        space=space,
        expected_steps=int(expected_steps),
        prefix_prefetch=bool(prefix_prefetch),
        tail_prefetch=bool(tail_prefetch),
        telemetry=bool(telemetry),
        lazy_secondary=bool(lazy_secondary),
        hard_cleanup_after_sample=bool(hard_cleanup_after_sample),
    )

    for i in range(cut):
        dm.blocks[i] = PrimaryPrefixProxyFactory.make(runtime, i)
    for i in range(cut, len(original_blocks)):
        dm.blocks[i] = SecondaryTailProxyFactory.make(runtime, i)

    primary_root = make_island_root(patcher.model, dm, prefix_blocks, len(original_blocks), "dev11-primary-prefix")
    secondary_root = make_island_root(patcher.model, dm, tail_blocks, len(original_blocks), "dev11-secondary-tail")
    Patcher = patcher.__class__
    primary_patcher = Patcher(
        primary_root, load_device=primary, offload_device=torch.device("cpu"), size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )
    secondary_patcher = Patcher(
        secondary_root, load_device=secondary, offload_device=torch.device("cpu"), size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )
    for p in (patcher, primary_patcher, secondary_patcher):
        p.size = 0
        p.cached_patcher_init = None

    from .turbo_compat import apply_turbo_plan_to_islands
    turbo_counts = apply_turbo_plan_to_islands(
        plan=turbo_plan,
        main_patcher=patcher,
        dm=dm,
        prefix_patcher=primary_patcher,
        tail_patcher=secondary_patcher,
        split_count=cut,
        primary_device=primary,
        secondary_device=secondary,
        prefix_device=primary,
        tail_device=secondary,
    )

    runtime.bind_secondary_patcher(secondary_patcher)

    main_fixed_bytes = int(mm.module_size(patcher.model))
    primary_tree_bytes = int(mm.module_size(primary_root))
    secondary_tree_bytes = int(mm.module_size(secondary_root)) if secondary_root is not None else 0
    registered = main_fixed_bytes + primary_tree_bytes + secondary_tree_bytes
    if not (int(total_bytes * 0.97) <= registered <= int(total_bytes * 1.03)):
        raise RuntimeError(
            f"H3VM Dev11 one-copy accounting failed registered={registered/GIB:.2f}GiB original={total_bytes/GIB:.2f}GiB"
        )

    # Primary-first is real here: only main + primary prefix participate in the
    # normal sampler model load. Secondary tail stays attached to the runtime and
    # is loaded lazily from the first prefix call.
    patcher.set_additional_models(
        "h3vm_dev11_primary_first",
        [primary_patcher] if bool(lazy_secondary) else [primary_patcher, secondary_patcher],
    )
    patcher.set_attachments("h3vm_dev11_runtime", runtime)
    patcher.set_attachments("h3vm_dev11_primary", primary_patcher)
    patcher.set_attachments("h3vm_dev11_secondary_lazy", secondary_patcher)
    patcher.set_attachments("h3vm_global_space", space)
    patcher.set_attachments("h3vm_dev11_config", {
        "architecture": "PRIMARY_FIRST_EXACT_LAZY_SECONDARY",
        "exact": True,
        "split_count": cut,
        "primary_runtime_target_pct": effective_p_target * 100.0,
        "primary_runtime_target_requested_pct": requested_p_target * 100.0,
        "secondary_runtime_target_pct": s_target * 100.0,
        "primary_workspace_reserve_gb": float(primary_workspace_reserve_gb),
        "secondary_workspace_reserve_gb": float(secondary_workspace_reserve_gb),
        "lazy_secondary": bool(lazy_secondary),
        "expected_steps": int(expected_steps),
        "turbo_lora_name": str(turbo_lora_name),
        "turbo_strength": float(turbo_strength),
        "turbo_counts": turbo_counts,
    })

    def sample_wrapper(executor, *args, **kwargs):
        runtime.sampling_begin()
        try:
            return executor(*args, **kwargs)
        finally:
            runtime.sampling_end()

    patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        "h3vm_dev11_lifecycle",
        sample_wrapper,
    )

    LOG.info(
        "H3VM Dev11 registered | exact EEEE | %s primary blocks 0-%d + fixed = %.2fGiB | "
        "%s lazy tail blocks %d-49 = %.2fGiB | secondary excluded from startup additional_models",
        primary, cut-1, (main_fixed_bytes + primary_tree_bytes)/GIB,
        secondary, cut, secondary_tree_bytes/GIB,
    )
    print(
        f"[H3VM DEV11 PRIMARY-FIRST READY] EXACT | {primary}: blocks 0-{cut-1} + fixed | "
        f"{secondary}: LAZY blocks {cut}-49 | target={effective_p_target*100:.1f}%/{s_target*100:.1f}% | "
        f"weights {(main_fixed_bytes+primary_tree_bytes)/GIB:.2f}+{secondary_tree_bytes/GIB:.2f}GiB | "
        f"TURBO={turbo_lora_name} strength={float(turbo_strength):.2f}",
        flush=True,
    )
    return patcher


def build_h3_streaming_exact_turbo(*, unet_name: str, turbo_lora_name: str, turbo_strength: float,
                                   turbo_low_vram: bool, primary_device: str, secondary_device: str,
                                   stripe_size: int = 2,
                                   primary_runtime_reserve_gb: float = 5.0,
                                   secondary_runtime_reserve_gb: float = 4.0,
                                   primary_hot_cache_gb: float = 3.5,
                                   secondary_hot_cache_gb: float = 2.0,
                                   expected_steps: int = 4,
                                   safe_profile: bool = True,
                                   one_ahead_prefetch: bool = True,
                                   trim_on_stripe_boundary: bool = True,
                                   hard_cleanup_after_sample: bool = False,
                                   telemetry: bool = True,
                                   attention_mode: str = "off",
                                   attention_head_balance: str = "sm_weighted",
                                   attention_min_sequence_length: int = 16384,
                                   attention_relay_min_gbps: float = 4.0,
                                   attention_helper_head_cap: int = 16,
                                   attention_helper_safety_mb: int = 1024,
                                   attention_host_ring_mb: int = 64,
                                   mlp_token_parallel: bool = False,
                                   mlp_primary_fraction: float = 0.60,
                                   mlp_min_sequence_length: int = 8192,
                                   mlp_helper_safety_mb: int = 1024,
                                   single_root: bool = False,
                                   single_root_trim_interval: int = 2,
                                   critical_path_mode: bool = False,
                                   sidecar_mlp_blocks: int = 14,
                                   sidecar_start_block: int = 0,
                                   primary_stall_budget_ms: float = 1.0,
                                   critical_path_adaptive_slack: bool = False,
                                   critical_path_min_primary_fraction: float = 0.68,
                                   critical_path_max_primary_fraction: float = 0.84,
                                   critical_path_fraction_step: float = 0.02,
                                   critical_path_target_slack_ms: float = 8.0,
                                   critical_path_resident_sidecar: bool = False,
                                   critical_path_pipeline_window: int = 1,
                                   critical_path_harvest_confirmations: int = 1,
                                   critical_path_rolling_retire: bool = False,
                                   attention_primary_first: bool = False,
                                   critical_path_persistent_workpool: bool = False,
                                   workpool_ring_slots: int = 2,
                                   workpool_deadline_margin_ms: float = 4.0,
                                   workpool_one_shot_matrix: bool = True,
                                   workpool_local_ticket_matrix: bool = False,
                                   workpool_load_shift_matrix: bool = False,
                                   critical_path_rolling_helper_matrix: bool = False,
                                   critical_path_rolling_adaptive_load: bool = False,
                                   critical_path_post_attention_island: bool = False,
                                   capacity_mode: bool = False,
                                   capacity_mlp_chunk_rows: int = 4096,
                                   capacity_outproj_chunk_rows: int = 4096,
                                   capacity_helper_heads: int = 16,
                                   capacity_attention_kernel: str = "INT8_CURRENT"):
    """Dev12.1 RAM-first striped streaming exact H3 Turbo runtime.

    Unlike Dev11, this branch does not ask either GPU to own a resident contiguous
    prefix/tail that fits its VRAM. All 50 original H3 blocks remain CPU-backed and
    are divided into small alternating *execution islands*. DynamicVRAM faults only
    the hot block working set into each GPU while explicit runtime reserves protect
    attention/MLP/LoRA workspace.

    Block order remains 0..49 on every denoise call. Device changes use the existing
    host-neutral CPU transport, so there is no stale state, prediction, cache skip,
    or D2D dependency.
    """
    import torch
    import comfy.model_management as mm
    import comfy.patcher_extension
    from comfy.cli_args import args
    from .global_memory import GlobalMemorySpace
    from .snapshot_islands import make_island_root
    from .streaming_exact import StreamingExactRuntime, StreamingBlockProxyFactory, striped_owner_map

    primary, secondary = _common_preflight(primary_device, secondary_device)
    if safe_profile and not bool(getattr(args, "disable_pinned_memory", False)):
        raise RuntimeError("H3VM Dev12.1 safe profile requires --disable-pinned-memory")

    attention_mode = str(attention_mode)
    attention_link = None
    if attention_mode not in ("off", "force_host"):
        from .peer_link import probe_copy_paths
        # Probe before model residency grows. The relay path is exact head
        # partitioning; it only changes where disjoint attention heads execute.
        attention_link = probe_copy_paths(primary, secondary, size_mb=32, repeats=3)
        LOG.info(
            "H3VM Dev12.2 attention link | peer=%s/%s copy=%.2f/%.2fGB/s errors=%s/%s",
            attention_link.get("peer_ab"), attention_link.get("peer_ba"),
            float(attention_link.get("copy_gbps_ab") or 0.0),
            float(attention_link.get("copy_gbps_ba") or 0.0),
            attention_link.get("error_ab"), attention_link.get("error_ba"),
        )

    stripe_size = max(1, min(10, int(stripe_size)))
    expected_steps = max(1, int(expected_steps))

    # Clear abandoned queues from older H3VM generations before creating a new
    # DynamicVRAM streaming runtime.
    try:
        import comfy.model_prefetch as mp
        mp.cleanup_prefetch_queues()
    except Exception:
        pass

    patcher, dm = _load_private_h3(unet_name, primary, safe_profile)

    from .turbo_compat import prepare_turbo_plan
    turbo_plan = prepare_turbo_plan(
        patcher, dm, str(turbo_lora_name), float(turbo_strength), bool(turbo_low_vram)
    )

    block_bytes = [int(mm.module_size(b)) for b in dm.blocks]
    total_bytes = int(mm.module_size(patcher.model))
    blocks_total = sum(block_bytes)
    fixed_bytes = max(0, total_bytes - blocks_total)
    GIB = 1024 ** 3

    ptotal = int(torch.cuda.get_device_properties(primary).total_memory)
    stotal = int(torch.cuda.get_device_properties(secondary).total_memory)
    p_reserve = float(primary_runtime_reserve_gb)
    s_reserve = float(secondary_runtime_reserve_gb)
    p_cache = float(primary_hot_cache_gb)
    s_cache = float(secondary_hot_cache_gb)

    if p_reserve * GIB >= ptotal - int(0.5 * GIB):
        raise RuntimeError("H3VM Dev12 primary runtime reserve leaves less than 0.5GiB usable VRAM")
    if s_reserve * GIB >= stotal - int(0.5 * GIB):
        raise RuntimeError("H3VM Dev12 secondary runtime reserve leaves less than 0.5GiB usable VRAM")
    if p_cache * GIB > max(0, ptotal - int(p_reserve * GIB)):
        raise RuntimeError("H3VM Dev12 primary hot cache exceeds VRAM left after runtime reserve")
    if s_cache * GIB > max(0, stotal - int(s_reserve * GIB)):
        raise RuntimeError("H3VM Dev12 secondary hot cache exceeds VRAM left after runtime reserve")

    # Dev13 collapses whole-block ownership onto the physical primary. GPU1 is
    # no longer a second block island; it is a pure Attention/MLP coprocessor.
    # This removes every full-hidden-state boundary handoff while preserving the
    # exact 0..49 block order.
    if bool(single_root):
        owner_map = {i: 0 for i in range(len(dm.blocks))}
    else:
        owner_map = striped_owner_map(len(dm.blocks), stripe_size)
    original_blocks = list(dm.blocks)
    primary_blocks = {i: original_blocks[i] for i in range(len(original_blocks)) if owner_map[i] == 0}
    secondary_blocks = {i: original_blocks[i] for i in range(len(original_blocks)) if owner_map[i] == 1}

    primary_mlp_helpers = {}
    secondary_mlp_helpers = {}
    helper_mlp_by_block = {}
    helper_attn_by_block = {}
    helper_packet_by_block = {}
    helper_indices = None
    if bool(mlp_token_parallel):
        if bool(critical_path_mode):
            from .critical_path import select_sidecar_blocks, build_selected_helper_maps
            helper_indices = select_sidecar_blocks(len(original_blocks), int(sidecar_mlp_blocks), int(sidecar_start_block))
            if bool(capacity_mode):
                from .capacity_mode import build_capacity_helper_maps
                (primary_mlp_helpers, secondary_mlp_helpers, helper_mlp_by_block,
                 helper_attn_by_block, helper_packet_by_block) = build_capacity_helper_maps(
                    original_blocks, owner_map, helper_indices
                )
            elif bool(critical_path_post_attention_island):
                from .post_attention_island import build_post_attention_helper_maps
                primary_mlp_helpers, secondary_mlp_helpers, helper_mlp_by_block = build_post_attention_helper_maps(
                    original_blocks, owner_map, helper_indices
                )
            else:
                primary_mlp_helpers, secondary_mlp_helpers, helper_mlp_by_block = build_selected_helper_maps(
                    original_blocks, owner_map, helper_indices
                )
        else:
            from .mlp_token_parallel import build_helper_maps
            primary_mlp_helpers, secondary_mlp_helpers, helper_mlp_by_block = build_helper_maps(
                original_blocks, owner_map
            )

    space = GlobalMemorySpace(
        primary, secondary,
        transport_mode="neutral_pageable",
        benchmark_mb=16,
        benchmark_repeats=1,
        allow_explicit_pinned=False,
        pinned_ring_mb=32,
        pinned_ring_slots=1,
        host_only=True,
    )

    runtime = StreamingExactRuntime(
        primary, secondary,
        {i: original_blocks[i] for i in range(len(original_blocks))},
        owner_map,
        space=space,
        expected_steps=expected_steps,
        stripe_size=stripe_size,
        primary_runtime_reserve_gb=p_reserve,
        secondary_runtime_reserve_gb=s_reserve,
        primary_hot_cache_gb=p_cache,
        secondary_hot_cache_gb=s_cache,
        one_ahead_prefetch=bool(one_ahead_prefetch),
        trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
        telemetry=bool(telemetry),
        hard_cleanup_after_sample=bool(hard_cleanup_after_sample),
        single_root=bool(single_root),
        single_root_trim_interval=max(1, int(single_root_trim_interval)),
        critical_path_pipeline_window=max(1, int(critical_path_pipeline_window)),
        critical_path_rolling_retire=bool(critical_path_rolling_retire),
    )

    # Replace all original block paths in the main H3 with zero-weight proxies.
    # The actual block objects are registered exactly once across the two CPU-
    # backed streaming island roots below.
    for i in range(len(original_blocks)):
        dm.blocks[i] = StreamingBlockProxyFactory.make(runtime, i)

    primary_root = make_island_root(
        patcher.model, dm, primary_blocks, len(original_blocks),
        "dev13-primary-single-root" if bool(single_root) else "dev12-primary-striped-stream"
    )
    # Dev13H: Single-Root means exactly that.  GPU1 is a compute coprocessor,
    # not an empty whole-block ModelPatcher.  Creating a zero-parameter island
    # still reaches AIMDO ModelVBAR on Windows and can fail while reserving its
    # virtual address window.
    secondary_root = None if bool(single_root) else make_island_root(
        patcher.model, dm, secondary_blocks, len(original_blocks),
        "dev12-secondary-striped-stream"
    )
    Patcher = patcher.__class__
    primary_island_patcher = Patcher(
        primary_root,
        load_device=primary,
        offload_device=torch.device("cpu"),
        size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )
    secondary_island_patcher = None if secondary_root is None else Patcher(
        secondary_root,
        load_device=secondary,
        offload_device=torch.device("cpu"),
        size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )

    primary_mlp_helper_patcher = None
    secondary_mlp_helper_patcher = None
    primary_mlp_helper_root = None
    secondary_mlp_helper_root = None
    if bool(mlp_token_parallel):
        # Never create empty helper patchers either.  In Dev13 all whole blocks
        # belong to GPU0, so only GPU1 needs the mirrored helper MLP tree.
        if primary_mlp_helpers:
            primary_mlp_helper_root = make_island_root(
                patcher.model, dm, primary_mlp_helpers, len(original_blocks), "dev12.3-primary-mlp-helper"
            )
            primary_mlp_helper_patcher = Patcher(
                primary_mlp_helper_root, load_device=primary, offload_device=torch.device("cpu"), size=0,
                weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
            )
        if secondary_mlp_helpers:
            secondary_mlp_helper_root = make_island_root(
                patcher.model, dm, secondary_mlp_helpers, len(original_blocks), "dev12.3-secondary-mlp-helper"
            )
            secondary_mlp_helper_patcher = Patcher(
                secondary_mlp_helper_root, load_device=secondary, offload_device=torch.device("cpu"), size=0,
                weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
            )

    for p in (patcher, primary_island_patcher, secondary_island_patcher,
              primary_mlp_helper_patcher, secondary_mlp_helper_patcher):
        if p is None:
            continue
        p.size = 0
        p.cached_patcher_init = None

    from .turbo_compat import apply_turbo_plan_to_streaming_islands
    turbo_counts = apply_turbo_plan_to_streaming_islands(
        plan=turbo_plan,
        main_patcher=patcher,
        dm=dm,
        primary_island_patcher=primary_island_patcher,
        secondary_island_patcher=secondary_island_patcher,
        owner_map=owner_map,
        primary_device=primary,
        secondary_device=secondary,
    )

    mlp_helper_turbo_counts = None
    if bool(mlp_token_parallel):
        from .turbo_compat import apply_turbo_plan_to_mlp_helpers
        mlp_helper_turbo_counts = apply_turbo_plan_to_mlp_helpers(
            plan=turbo_plan,
            primary_helper_patcher=primary_mlp_helper_patcher,
            secondary_helper_patcher=secondary_mlp_helper_patcher,
            owner_map=owner_map,
            primary_device=primary,
            secondary_device=secondary,
            helper_indices=helper_indices,
        )
        if bool(capacity_mode):
            from .turbo_compat import apply_turbo_plan_to_attention_helpers
            apply_turbo_plan_to_attention_helpers(
                plan=turbo_plan,
                primary_helper_patcher=primary_mlp_helper_patcher,
                secondary_helper_patcher=secondary_mlp_helper_patcher,
                owner_map=owner_map,
                primary_device=primary,
                secondary_device=secondary,
                helper_indices=helper_indices,
            )

    runtime.bind_patchers(
        primary_island_patcher, secondary_island_patcher,
        primary_mlp_helper_patcher, secondary_mlp_helper_patcher,
    )

    attention = None
    if attention_mode != "off":
        from .attention_parallel import H3VMRelayAttentionParallel
        attention = H3VMRelayAttentionParallel(
            primary, secondary,
            mode=attention_mode,
            head_balance=str(attention_head_balance),
            min_sequence_length=max(1, int(attention_min_sequence_length)),
            link_info=attention_link or {},
            relay_min_gbps=float(attention_relay_min_gbps),
            helper_head_cap=max(1, int(attention_helper_head_cap)),
            helper_safety_mb=max(128, int(attention_helper_safety_mb)),
            host_ring_mb=max(16, int(attention_host_ring_mb)),
            primary_first=bool(attention_primary_first),
        )
        if attention.enabled:
            if not hasattr(patcher, "set_model_optimized_attention"):
                raise RuntimeError(
                    "Current ComfyUI ModelPatcher lacks set_model_optimized_attention; "
                    "cannot enable Dev12.1 exact head parallel attention."
                )
            patcher.set_model_optimized_attention(attention)
            patcher.set_attachments("h3vm_dev12_2_attention", attention)
            LOG.info(
                "H3VM Dev12.2 ATTENTION PARALLEL armed | mode=%s transport=%s balance=%s min_seq=%d helper_cap=%d safety=%dMB host_ring=%dMB",
                attention_mode, getattr(attention, "transport", None),
                attention_head_balance, int(attention_min_sequence_length),
                int(attention_helper_head_cap), int(attention_helper_safety_mb),
                int(attention_host_ring_mb),
            )
        else:
            LOG.warning(
                "H3VM Dev12.2 attention inactive | mode=%s reason=%s",
                attention_mode, getattr(attention, "disable_reason", "unknown"),
            )

    # Dev12.3H: install token-parallel MLP into the *actual* RAM-first blocks.
    # The previous Dev12.3 release created helper mirrors but accidentally placed
    # this fabric-install section in the legacy Dev11 builder, so MLP=token2gpu
    # could be printed without a single MLP call being intercepted. Fail closed
    # here: when requested, all 50 original H3 MLP.forward methods must be patched.
    mlp_fabric = None
    if bool(mlp_token_parallel):
        from .global_memory import TransportEngine
        from .mlp_token_parallel import install_token_parallel_mlp
        host_engine = getattr(attention, "host_engine", None) if attention is not None else None
        if host_engine is None:
            try:
                host_engine = TransportEngine(
                    primary, secondary, mode="neutral_pinned", benchmark_mb=32, benchmark_repeats=2,
                    allow_explicit_pinned=True, pinned_ring_mb=max(64, int(attention_host_ring_mb)),
                    pinned_ring_slots=2, host_only=True,
                )
            except Exception:
                host_engine = TransportEngine(
                    primary, secondary, mode="neutral_pageable", benchmark_mb=16, benchmark_repeats=1,
                    allow_explicit_pinned=False, pinned_ring_mb=0, pinned_ring_slots=1, host_only=True,
                )
        if bool(critical_path_mode):
            if bool(capacity_mode):
                from .capacity_mode import CapacityFabric
                mlp_fabric = CapacityFabric(
                    primary=primary, secondary=secondary, owner_map=owner_map,
                    helper_mlp_by_block=helper_mlp_by_block,
                    helper_attn_by_block=helper_attn_by_block,
                    helper_packet_by_block=helper_packet_by_block,
                    primary_fraction=float(mlp_primary_fraction),
                    min_sequence_length=int(mlp_min_sequence_length),
                    helper_safety_mb=int(mlp_helper_safety_mb),
                    primary_stall_budget_ms=1.0e9,
                    min_primary_fraction=float(mlp_primary_fraction),
                    max_primary_fraction=float(mlp_primary_fraction),
                    fraction_step=0.01, target_slack_ms=999999.0,
                    resident_sidecar_packet=False, harvest_confirmations=999,
                    mlp_chunk_rows=int(capacity_mlp_chunk_rows),
                    outproj_chunk_rows=int(capacity_outproj_chunk_rows),
                    helper_heads=int(capacity_helper_heads),
                    attention_kernel=str(capacity_attention_kernel),
                    telemetry=bool(telemetry),
                )
            elif bool(critical_path_post_attention_island):
                from .post_attention_island import PostAttentionRowIslandFabric
                mlp_fabric = PostAttentionRowIslandFabric(
                    primary=primary, secondary=secondary, owner_map=owner_map,
                    helper_mlp_by_block=helper_mlp_by_block,
                    primary_fraction=0.68,
                    min_sequence_length=int(mlp_min_sequence_length),
                    helper_safety_mb=int(mlp_helper_safety_mb),
                    primary_stall_budget_ms=float(primary_stall_budget_ms),
                    min_primary_fraction=0.68, max_primary_fraction=0.72,
                    fraction_step=0.02, target_slack_ms=9999.0,
                    resident_sidecar_packet=False, harvest_confirmations=99,
                    telemetry=bool(telemetry),
                )
            elif bool(critical_path_rolling_adaptive_load):
                from .rolling_adaptive_load import RollingAdaptiveLoadMLPFabric
                mlp_fabric = RollingAdaptiveLoadMLPFabric(
                    primary=primary, secondary=secondary, owner_map=owner_map,
                    helper_mlp_by_block=helper_mlp_by_block,
                    primary_fraction=0.68,
                    min_sequence_length=int(mlp_min_sequence_length),
                    helper_safety_mb=int(mlp_helper_safety_mb),
                    primary_stall_budget_ms=float(primary_stall_budget_ms),
                    min_primary_fraction=0.60,
                    max_primary_fraction=0.72, fraction_step=0.02,
                    target_slack_ms=9999.0, resident_sidecar_packet=False,
                    harvest_confirmations=99, telemetry=bool(telemetry),
                )
            elif bool(critical_path_rolling_helper_matrix):
                from .rolling_helper_matrix import RollingCoverageMLPFabric
                mlp_fabric = RollingCoverageMLPFabric(
                    primary=primary, secondary=secondary, owner_map=owner_map,
                    helper_mlp_by_block=helper_mlp_by_block,
                    primary_fraction=float(mlp_primary_fraction),
                    min_sequence_length=int(mlp_min_sequence_length),
                    helper_safety_mb=int(mlp_helper_safety_mb),
                    primary_stall_budget_ms=float(primary_stall_budget_ms),
                    min_primary_fraction=float(mlp_primary_fraction),
                    max_primary_fraction=0.72, fraction_step=0.02,
                    target_slack_ms=9999.0, resident_sidecar_packet=False,
                    harvest_confirmations=99, telemetry=bool(telemetry),
                    coverage_schedule=(20, 30, 40, 50),
                )
            elif bool(critical_path_persistent_workpool):
                if bool(workpool_load_shift_matrix):
                    from .ticket_workpool import LoadShiftWorkpoolMLPFabric
                    fabric_cls = LoadShiftWorkpoolMLPFabric
                elif bool(workpool_local_ticket_matrix):
                    from .ticket_workpool import LocalTicketWorkpoolMLPFabric
                    fabric_cls = LocalTicketWorkpoolMLPFabric
                else:
                    from .persistent_workpool import PersistentWorkpoolMLPFabric
                    fabric_cls = PersistentWorkpoolMLPFabric
                # Dev16.0 compatibility marker: mlp_fabric = PersistentWorkpoolMLPFabric
                mlp_fabric = fabric_cls(
                    primary=primary, secondary=secondary, owner_map=owner_map,
                    helper_mlp_by_block=helper_mlp_by_block,
                    primary_fraction=float(mlp_primary_fraction),
                    min_sequence_length=int(mlp_min_sequence_length),
                    helper_safety_mb=int(mlp_helper_safety_mb),
                    primary_stall_budget_ms=float(primary_stall_budget_ms),
                    min_primary_fraction=float(critical_path_min_primary_fraction),
                    max_primary_fraction=float(critical_path_max_primary_fraction),
                    fraction_step=float(critical_path_fraction_step),
                    target_slack_ms=float(critical_path_target_slack_ms),
                    resident_sidecar_packet=bool(critical_path_resident_sidecar),
                    harvest_confirmations=max(1, int(critical_path_harvest_confirmations)),
                    ring_slots=max(1, int(workpool_ring_slots)),
                    deadline_margin_ms=float(workpool_deadline_margin_ms),
                    one_shot_matrix=bool(workpool_one_shot_matrix),
                    telemetry=bool(telemetry),
                )
            else:
                from .critical_path import CriticalPathMLPFabric
                mlp_fabric = CriticalPathMLPFabric(
                    primary=primary, secondary=secondary, owner_map=owner_map,
                    helper_mlp_by_block=helper_mlp_by_block,
                    primary_fraction=float(mlp_primary_fraction),
                    min_sequence_length=int(mlp_min_sequence_length),
                    helper_safety_mb=int(mlp_helper_safety_mb),
                    primary_stall_budget_ms=float(primary_stall_budget_ms),
                    adaptive_slack=bool(critical_path_adaptive_slack),
                    min_primary_fraction=float(critical_path_min_primary_fraction),
                    max_primary_fraction=float(critical_path_max_primary_fraction),
                    fraction_step=float(critical_path_fraction_step),
                    target_slack_ms=float(critical_path_target_slack_ms),
                    resident_sidecar_packet=bool(critical_path_resident_sidecar),
                    harvest_confirmations=max(1, int(critical_path_harvest_confirmations)),
                    telemetry=bool(telemetry),
                )
        else:
            from .mlp_token_parallel import TokenParallelMLPFabric
            mlp_fabric = TokenParallelMLPFabric(
                primary=primary, secondary=secondary, owner_map=owner_map,
                helper_mlp_by_block=helper_mlp_by_block, host_engine=host_engine,
                primary_fraction=float(mlp_primary_fraction),
                min_sequence_length=int(mlp_min_sequence_length),
                helper_safety_mb=int(mlp_helper_safety_mb), telemetry=bool(telemetry),
            )
        if bool(capacity_mode):
            from .capacity_mode import install_capacity_mode
            installed = install_capacity_mode(original_blocks, mlp_fabric)
            if installed != len(original_blocks):
                raise RuntimeError(
                    f"H3VM Capacity fail-closed: patched {installed}/{len(original_blocks)} blocks"
                )
        elif bool(critical_path_post_attention_island):
            from .post_attention_island import install_post_attention_island
            installed = install_post_attention_island(original_blocks, mlp_fabric)
            if installed != len(original_blocks):
                raise RuntimeError(
                    f"H3VM Dev18 post-attention island fail-closed: patched {installed}/{len(original_blocks)} blocks"
                )
        else:
            installed = install_token_parallel_mlp(original_blocks, mlp_fabric)
            if installed != len(original_blocks):
                raise RuntimeError(
                    f"H3VM Dev12.3H token MLP fail-closed: patched {installed}/{len(original_blocks)} blocks"
                )
        runtime.bind_mlp_fabric(mlp_fabric)
        patcher.set_attachments("h3vm_dev12_3_mlp_fabric", mlp_fabric)
        patcher.set_attachments("h3vm_dev12_3_mlp_helpers", {
            "primary": primary_mlp_helper_patcher,
            "secondary": secondary_mlp_helper_patcher,
        })
        if bool(critical_path_mode):
            if bool(capacity_mode):
                LOG.info(
                    "H3VM CAPACITY armed | exact QKV head shard + token-chunk MLP | blocks=50 | root/helper=%.0f/%.0f | mlp_chunk=%d outproj_chunk=%d helper_heads=%d attn=%s",
                    float(mlp_primary_fraction)*100.0, (1.0-float(mlp_primary_fraction))*100.0,
                    int(capacity_mlp_chunk_rows), int(capacity_outproj_chunk_rows), int(capacity_helper_heads), str(capacity_attention_kernel),
                )
            elif bool(critical_path_post_attention_island):
                LOG.info(
                    "H3VM DEV18 POST-ATTENTION ROW ISLAND armed | coverage=50/50 split=68/32 | matrix=WARMUP_BASELINE>MLP_ONLY_BASELINE>ISLAND>ISLAND_REPEAT"
                )
            if bool(critical_path_persistent_workpool):
                LOG.info(
                    "H3VM DEV16.%s PERSISTENT WORKPOOL armed | ring_slots=%d margin=%.1fms one_shot=%s min_root=%.2f max_root=%.2f",
                    "1-TICKET" if bool(workpool_local_ticket_matrix) else "0", int(workpool_ring_slots), float(workpool_deadline_margin_ms), bool(workpool_one_shot_matrix),
                    float(critical_path_min_primary_fraction), float(critical_path_max_primary_fraction),
                )
            LOG.info(
                "H3VM Dev14.1 CRITICAL PATH MLP armed | patched=%d sidecar_blocks=%d start_block=%d indices=%s root_fraction=%.3f "
                "adaptive=%s range=%.2f..%.2f step=%.2f target_slack=%.1fms | min_seq=%d safety=%dMB stall_budget=%.1fms | "
                "pipeline=%d confirm=%d rolling=%s per-block-prefetch=OFF",
                installed, len(helper_indices or []), int(sidecar_start_block), helper_indices, float(mlp_primary_fraction),
                bool(critical_path_adaptive_slack), float(critical_path_min_primary_fraction), float(critical_path_max_primary_fraction),
                float(critical_path_fraction_step), float(critical_path_target_slack_ms),
                int(mlp_min_sequence_length), int(mlp_helper_safety_mb), float(primary_stall_budget_ms),
                int(critical_path_pipeline_window), int(critical_path_harvest_confirmations), bool(critical_path_rolling_retire),
            )
        else:
            LOG.info(
                "H3VM Dev12.3H TOKEN MLP armed | blocks=%d primary_fraction=%.3f min_seq=%d safety=%dMB transport=%s",
                installed, float(mlp_primary_fraction), int(mlp_min_sequence_length), int(mlp_helper_safety_mb),
                getattr(host_engine, "selected_mode", "unknown"),
            )

    main_fixed_bytes = int(mm.module_size(patcher.model))
    primary_tree_bytes = int(mm.module_size(primary_root))
    secondary_tree_bytes = int(mm.module_size(secondary_root)) if secondary_root is not None else 0
    registered = main_fixed_bytes + primary_tree_bytes + secondary_tree_bytes
    if not (int(total_bytes * 0.97) <= registered <= int(total_bytes * 1.03)):
        raise RuntimeError(
            f"H3VM Dev12 one-copy accounting failed registered={registered/GIB:.2f}GiB original={total_bytes/GIB:.2f}GiB"
        )

    # IMPORTANT: streaming islands are intentionally NOT startup additional_models.
    # The runtime cold-starts them inside OUTER_SAMPLE with explicit reserve budgets.
    # This prevents Comfy's default model loading phase from opportunistically filling
    # the 8G card before the first real block executes.
    patcher.set_attachments("h3vm_dev12_runtime", runtime)
    patcher.set_attachments("h3vm_dev12_primary_stream", primary_island_patcher)
    patcher.set_attachments("h3vm_dev12_secondary_stream", secondary_island_patcher)
    patcher.set_attachments("h3vm_global_space", space)
    patcher.set_attachments("h3vm_dev12_config", {
        "architecture": ("CRITICAL_PATH_SINGLE_ROOT_SIDECAR" if bool(critical_path_mode) else "RAM_FIRST_SINGLE_ROOT_COPROCESSOR") if bool(single_root) else "RAM_FIRST_STRIPED_STREAMING_EXACT",
        "exact": True,
        "stripe_size": stripe_size,
        "owner_map": dict(owner_map),
        "primary_runtime_reserve_gb": p_reserve,
        "secondary_runtime_reserve_gb": s_reserve,
        "primary_hot_cache_gb": p_cache,
        "secondary_hot_cache_gb": s_cache,
        "one_ahead_prefetch": bool(one_ahead_prefetch),
        "trim_on_stripe_boundary": bool(trim_on_stripe_boundary),
        "expected_steps": expected_steps,
        "turbo_lora_name": str(turbo_lora_name),
        "turbo_strength": float(turbo_strength),
        "turbo_counts": turbo_counts,
        "attention_mode": attention_mode,
        "attention_enabled": bool(attention is not None and attention.enabled),
        "attention_transport": getattr(attention, "transport", None) if attention is not None else None,
        "attention_head_balance": str(attention_head_balance),
        "attention_min_sequence_length": int(attention_min_sequence_length),
        "attention_relay_min_gbps": float(attention_relay_min_gbps),
        "attention_helper_head_cap": int(attention_helper_head_cap),
        "attention_helper_safety_mb": int(attention_helper_safety_mb),
        "attention_host_ring_mb": int(attention_host_ring_mb),
        "mlp_token_parallel": bool(mlp_token_parallel),
        "mlp_primary_fraction": float(mlp_primary_fraction),
        "mlp_min_sequence_length": int(mlp_min_sequence_length),
        "mlp_helper_safety_mb": int(mlp_helper_safety_mb),
        "mlp_helper_turbo_counts": mlp_helper_turbo_counts,
        "single_root": bool(single_root),
        "single_root_trim_interval": max(1, int(single_root_trim_interval)),
        "critical_path_mode": bool(critical_path_mode),
        "sidecar_mlp_blocks": int(sidecar_mlp_blocks),
        "sidecar_start_block": int(sidecar_start_block),
        "sidecar_helper_indices": list(helper_indices or []),
        "primary_stall_budget_ms": float(primary_stall_budget_ms),
        "critical_path_adaptive_slack": bool(critical_path_adaptive_slack),
        "critical_path_min_primary_fraction": float(critical_path_min_primary_fraction),
        "critical_path_max_primary_fraction": float(critical_path_max_primary_fraction),
        "critical_path_fraction_step": float(critical_path_fraction_step),
        "critical_path_target_slack_ms": float(critical_path_target_slack_ms),
        "critical_path_resident_sidecar": bool(critical_path_resident_sidecar),
        "attention_primary_first": bool(attention_primary_first),
        "critical_path_persistent_workpool": bool(critical_path_persistent_workpool),
        "workpool_ring_slots": int(workpool_ring_slots),
        "workpool_local_ticket_matrix": bool(workpool_local_ticket_matrix),
        "workpool_deadline_margin_ms": float(workpool_deadline_margin_ms),
        "workpool_one_shot_matrix": bool(workpool_one_shot_matrix),
        "capacity_mode": bool(capacity_mode),
        "capacity_mlp_chunk_rows": int(capacity_mlp_chunk_rows),
        "capacity_outproj_chunk_rows": int(capacity_outproj_chunk_rows),
        "capacity_helper_heads": int(capacity_helper_heads),
        "capacity_attention_kernel": str(capacity_attention_kernel),
    })

    def sample_wrapper(executor, *args, **kwargs):
        runtime.sampling_begin()
        try:
            return executor(*args, **kwargs)
        finally:
            runtime.sampling_end()

    patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        "h3vm_dev12_streaming_lifecycle",
        sample_wrapper,
    )

    p_indices = sorted(primary_blocks)
    s_indices = sorted(secondary_blocks)
    if bool(single_root):
        LOG.info(
            "H3VM Dev13 SINGLE-ROOT planner | fixed=%.2fGiB blocks=%.2fGiB | "
            "%s root blocks=%d backing=%.2fGiB reserve=%.2fGiB hot_cache=%.2fGiB | "
            "%s coprocessor block_island=%d helper_cache=%.2fGiB reserve=%.2fGiB | trim_interval=%d",
            fixed_bytes/GIB, blocks_total/GIB, primary, len(p_indices), primary_tree_bytes/GIB,
            p_reserve, p_cache, secondary, len(s_indices), s_cache, s_reserve,
            max(1, int(single_root_trim_interval)),
        )
        if bool(critical_path_mode):
            print(
                f"[H3VM DEV14.1 SLACK-HARVEST READY] EXACT | root={primary} blocks={len(p_indices)} NONSTOP | "
                f"sidecar={secondary} whole-block-owners=0 | boundaries=0 | sidecar_MLP={len(helper_indices or [])} blocks start>={int(sidecar_start_block)} | "
                f"RAM backing={primary_tree_bytes/GIB:.2f}GiB blocks | root reserve/cache={p_reserve:.2f}/{p_cache:.2f}GiB | "
                f"helper reserve/cache={s_reserve:.2f}/{s_cache:.2f}GiB | ATTN=root-first | "
                f"MLP=root-first initial {float(mlp_primary_fraction)*100:.0f}/{(1.0-float(mlp_primary_fraction))*100:.0f} adaptive={bool(critical_path_adaptive_slack)} | "
                f"stall_budget={float(primary_stall_budget_ms):.1f}ms | sidecar_prepare=ASYNC | TURBO={turbo_lora_name} strength={float(turbo_strength):.2f}", flush=True,
            )
        else:
            print(
                f"[H3VM DEV13 SINGLE-ROOT READY] EXACT | root={primary} blocks={len(p_indices)} | "
                f"coprocessor={secondary} whole-block-owners=0 | boundaries=0 target | "
                f"RAM backing={primary_tree_bytes/GIB:.2f}GiB blocks | "
                f"root reserve/cache={p_reserve:.2f}/{p_cache:.2f}GiB | helper reserve/cache={s_reserve:.2f}/{s_cache:.2f}GiB | "
                f"trim_every={max(1, int(single_root_trim_interval))} blocks | "
                f"ATTN={'on:'+str(getattr(attention, 'transport', None)) if (attention is not None and attention.enabled) else 'off'} | "
                f"MLP={'token2gpu' if mlp_token_parallel else 'single'} | "
                f"TURBO={turbo_lora_name} strength={float(turbo_strength):.2f}", flush=True,
            )
    else:
        LOG.info(
            "H3VM Dev12 RAM-FIRST planner | stripe=%d | fixed=%.2fGiB blocks=%.2fGiB | "
            "%s stream blocks=%d weight_backing=%.2fGiB reserve=%.2fGiB hot_cache=%.2fGiB | "
            "%s stream blocks=%d weight_backing=%.2fGiB reserve=%.2fGiB hot_cache=%.2fGiB",
            stripe_size, fixed_bytes/GIB, blocks_total/GIB,
            primary, len(p_indices), primary_tree_bytes/GIB, p_reserve, p_cache,
            secondary, len(s_indices), secondary_tree_bytes/GIB, s_reserve, s_cache,
        )
        print(
            f"[H3VM DEV12 RAM-FIRST READY] EXACT | stripe={stripe_size} | "
            f"RAM backing={primary_tree_bytes/GIB:.2f}+{secondary_tree_bytes/GIB:.2f}GiB blocks | "
            f"{primary} reserve/cache={p_reserve:.2f}/{p_cache:.2f}GiB | "
            f"{secondary} reserve/cache={s_reserve:.2f}/{s_cache:.2f}GiB | "
            f"prefetch={'1-ahead' if one_ahead_prefetch else 'off'} | "
            f"ATTN={'on:'+str(getattr(attention, 'transport', None)) if (attention is not None and attention.enabled) else 'off'} | "
            f"MLP={'token2gpu' if mlp_token_parallel else 'single'} | "
            f"TURBO={turbo_lora_name} strength={float(turbo_strength):.2f}",
            flush=True,
        )
    return patcher
