"""H3VM Multi-GPU Loader for MiniMax H3 / ComfyUI.

Public entrypoint exposes a compact multi-GPU loader and mode-aware Video VAE
decode node. Historical engineering loaders remain in the codebase as optional
labs and can be exposed with H3VM_SHOW_LAB_NODES=1. Heavy torch/comfy imports
remain delayed until execution where practical.
"""

import os

WEB_DIRECTORY = "./web"
H3VM_SHOW_LAB_NODES = os.environ.get("H3VM_SHOW_LAB_NODES", "0").strip().lower() in {"1", "true", "yes", "on"}

if H3VM_SHOW_LAB_NODES:
    print('[H3VM DEV13.0H LOADED] Single-Root RAM Fabric + GPU Coprocessor + NO EMPTY VBAR', flush=True)



class H3VMPhysicalShardLoader:
    """Known-good Dev4 fallback, intentionally preserved."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "split_mode": (["memory_aware", "manual"], {"default": "memory_aware"}),
                "secondary_blocks": ("INT", {"default": 17, "min": 1, "max": 49, "step": 1}),
                "primary_reserve_gb": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 8.0, "step": 0.25}),
                "secondary_reserve_gb": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 4.0, "step": 0.25}),
                "safe_profile": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM"
    DESCRIPTION = "Known-good Dev4 physical block shard fallback. Base H3 only."

    def load(self, unet_name, primary_device, secondary_device, split_mode, secondary_blocks,
             primary_reserve_gb=2.5, secondary_reserve_gb=1.5, safe_profile=True):
        from .h3vm.loader import build_h3_physical_shard
        return (build_h3_physical_shard(
            unet_name=unet_name,
            primary_device=primary_device,
            secondary_device=secondary_device,
            split_mode=split_mode,
            secondary_blocks=int(secondary_blocks),
            primary_reserve_gb=float(primary_reserve_gb),
            secondary_reserve_gb=float(secondary_reserve_gb),
            safe_profile=bool(safe_profile),
        ),)


class H3VMHybridShardLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "split_mode": (["speed_first", "balanced", "manual"], {"default": "speed_first"}),
                "secondary_blocks": ("INT", {"default": 17, "min": 1, "max": 49, "step": 1}),
                "primary_reserve_gb": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 8.0, "step": 0.25}),
                "secondary_reserve_gb": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 4.0, "step": 0.25}),
                "safe_profile": ("BOOLEAN", {"default": True}),
                "attention_mode": (["auto", "off", "force_2gpu"], {"default": "auto"}),
                "head_balance": (["sm_weighted", "balanced"], {"default": "sm_weighted"}),
                "min_sequence_length": ("INT", {
                    "default": 16384, "min": 1, "max": 1000000, "step": 1024,
                    "tooltip": "In auto mode, use two-GPU attention only at or above this packed token count."
                }),
                "secondary_prefetch": ("BOOLEAN", {"default": True}),
                "probe_link": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM"
    DESCRIPTION = (
        "Dev5 hybrid H3 runtime. Keeps one physically sharded H3, adds a private prefetch queue "
        "for the secondary-owned block span, and can split exact Comfy-Kitchen INT8 attention "
        "heads across the two selected GPUs when bidirectional CUDA peer access is available."
    )

    def load(self, unet_name, primary_device, secondary_device, split_mode, secondary_blocks,
             primary_reserve_gb=2.5, secondary_reserve_gb=1.5, safe_profile=True,
             attention_mode="auto", head_balance="sm_weighted", min_sequence_length=16384,
             secondary_prefetch=True, probe_link=True):
        from .h3vm.loader import build_h3_hybrid_shard
        return (build_h3_hybrid_shard(
            unet_name=unet_name,
            primary_device=primary_device,
            secondary_device=secondary_device,
            split_mode=split_mode,
            secondary_blocks=int(secondary_blocks),
            primary_reserve_gb=float(primary_reserve_gb),
            secondary_reserve_gb=float(secondary_reserve_gb),
            safe_profile=bool(safe_profile),
            attention_mode=str(attention_mode),
            head_balance=str(head_balance),
            min_sequence_length=int(min_sequence_length),
            secondary_prefetch=bool(secondary_prefetch),
            probe_link=bool(probe_link),
        ),)


class H3VMRelayShardLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "split_mode": (["speed_first", "balanced", "manual"], {"default": "speed_first"}),
                "secondary_blocks": ("INT", {"default": 17, "min": 1, "max": 49, "step": 1}),
                "primary_reserve_gb": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 8.0, "step": 0.25}),
                "secondary_reserve_gb": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 4.0, "step": 0.25}),
                "safe_profile": ("BOOLEAN", {"default": True}),
                "attention_mode": (["relay_auto", "off", "force_relay", "p2p_only"], {"default": "relay_auto"}),
                "head_balance": (["sm_weighted", "balanced"], {"default": "sm_weighted"}),
                "min_sequence_length": ("INT", {
                    "default": 1, "min": 1, "max": 1000000, "step": 1024,
                    "tooltip": "Testing default=1. Raise to 16384/32768 after relay performance is characterized."
                }),
                "secondary_prefetch": ("BOOLEAN", {"default": True}),
                "probe_link": ("BOOLEAN", {"default": True}),
                "relay_min_gbps": ("FLOAT", {
                    "default": 4.0, "min": 0.5, "max": 64.0, "step": 0.5,
                    "tooltip": "relay_auto is disabled if either fallback D2D direction benchmarks below this."
                }),
                "helper_head_cap": ("INT", {
                    "default": 12, "min": 1, "max": 28, "step": 1,
                    "tooltip": "Safety cap for helper attention heads. 12 is conservative for an 8GB helper holding H3 weights."
                }),
                "helper_safety_mb": ("INT", {
                    "default": 512, "min": 128, "max": 2048, "step": 128,
                    "tooltip": "Physical free-VRAM floor kept before assigning helper attention heads."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM"
    DESCRIPTION = (
        "Dev6 transport-aware H3 runtime. Preserves the Dev5 one-copy physical block shard and "
        "secondary prefetch. If direct CUDA P2P is unavailable, it benchmarks the ordinary "
        "cross-device copy path and can use a conservative relay attention mode where only "
        "quantized Q/K/V head slices and attention outputs cross GPUs."
    )

    def load(self, unet_name, primary_device, secondary_device, split_mode, secondary_blocks,
             primary_reserve_gb=2.5, secondary_reserve_gb=1.5, safe_profile=True,
             attention_mode="relay_auto", head_balance="sm_weighted", min_sequence_length=1,
             secondary_prefetch=True, probe_link=True, relay_min_gbps=4.0,
             helper_head_cap=12, helper_safety_mb=512):
        from .h3vm.loader import build_h3_relay_shard
        return (build_h3_relay_shard(
            unet_name=unet_name,
            primary_device=primary_device,
            secondary_device=secondary_device,
            split_mode=split_mode,
            secondary_blocks=int(secondary_blocks),
            primary_reserve_gb=float(primary_reserve_gb),
            secondary_reserve_gb=float(secondary_reserve_gb),
            safe_profile=bool(safe_profile),
            attention_mode=str(attention_mode),
            head_balance=str(head_balance),
            min_sequence_length=int(min_sequence_length),
            secondary_prefetch=bool(secondary_prefetch),
            probe_link=bool(probe_link),
            relay_min_gbps=float(relay_min_gbps),
            helper_head_cap=int(helper_head_cap),
            helper_safety_mb=int(helper_safety_mb),
        ),)



class H3VMGlobalMemoryProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "benchmark_mb": ("INT", {"default": 64, "min": 8, "max": 256, "step": 8}),
                "repeats": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "allow_explicit_pinned": ("BOOLEAN", {"default": False}),
                "pinned_ring_mb": ("INT", {
                    "default": 64, "min": 8, "max": 128, "step": 8,
                    "tooltip": "Only used when explicit pinned staging is enabled. Small runway, never model storage."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_json",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "MiniMaxH3/H3VM/GlobalMemory"
    DESCRIPTION = (
        "Dev7 Global Memory transport lab. Benchmarks direct fallback D2D and host-neutral routes, "
        "then proves a device-agnostic GlobalTensor can be acquired by both GPUs. No H3 model load."
    )

    def run(self, primary_device, secondary_device, benchmark_mb=64, repeats=3,
            allow_explicit_pinned=False, pinned_ring_mb=64):
        from .h3vm.global_probe import run_global_memory_probe
        return (run_global_memory_probe(
            primary_device, secondary_device,
            benchmark_mb=int(benchmark_mb), repeats=int(repeats),
            allow_explicit_pinned=bool(allow_explicit_pinned),
            pinned_ring_mb=int(pinned_ring_mb),
        ),)


class H3VMGlobalMemoryLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "split_mode": (["speed_first", "balanced", "manual"], {"default": "speed_first"}),
                "secondary_blocks": ("INT", {"default": 17, "min": 1, "max": 49, "step": 1}),
                "primary_reserve_gb": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 8.0, "step": 0.25}),
                "secondary_reserve_gb": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 4.0, "step": 0.25}),
                "safe_profile": ("BOOLEAN", {"default": True}),
                "transport_mode": (["auto", "neutral_pageable", "direct_d2d", "neutral_pinned"], {"default": "auto"}),
                "benchmark_transport": ("BOOLEAN", {"default": True}),
                "benchmark_mb": ("INT", {"default": 64, "min": 8, "max": 256, "step": 8}),
                "benchmark_repeats": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "allow_explicit_pinned": ("BOOLEAN", {"default": False}),
                "pinned_ring_mb": ("INT", {"default": 64, "min": 8, "max": 128, "step": 8}),
                "secondary_prefetch": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM/GlobalMemory"
    DESCRIPTION = (
        "Dev7 Global Memory Fabric. The logical tensor belongs to GlobalMemorySpace; CPU/GPU0/GPU1 are "
        "compute blocks. Keeps the proven one-copy H3 weight shard while routing activation handoffs "
        "through a replaceable transport backend. neutral_pageable is the pure no-P2P pool proof."
    )

    def load(self, unet_name, primary_device, secondary_device, split_mode, secondary_blocks,
             primary_reserve_gb=2.5, secondary_reserve_gb=1.5, safe_profile=True,
             transport_mode="auto", benchmark_transport=True, benchmark_mb=64,
             benchmark_repeats=3, allow_explicit_pinned=False, pinned_ring_mb=64,
             secondary_prefetch=True):
        from .h3vm.loader import build_h3_global_memory_shard
        return (build_h3_global_memory_shard(
            unet_name=unet_name,
            primary_device=primary_device,
            secondary_device=secondary_device,
            split_mode=split_mode,
            secondary_blocks=int(secondary_blocks),
            primary_reserve_gb=float(primary_reserve_gb),
            secondary_reserve_gb=float(secondary_reserve_gb),
            safe_profile=bool(safe_profile),
            transport_mode=str(transport_mode),
            benchmark_transport=bool(benchmark_transport),
            benchmark_mb=int(benchmark_mb),
            benchmark_repeats=int(benchmark_repeats),
            allow_explicit_pinned=bool(allow_explicit_pinned),
            pinned_ring_mb=int(pinned_ring_mb),
            secondary_prefetch=bool(secondary_prefetch),
        ),)


class H3VMGlobalTensorFabricLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "split_mode": (["speed_first", "balanced", "manual"], {"default": "speed_first"}),
                "secondary_blocks": ("INT", {"default": 17, "min": 1, "max": 49, "step": 1}),
                "primary_reserve_gb": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 8.0, "step": 0.25}),
                "secondary_reserve_gb": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 4.0, "step": 0.25}),
                "safe_profile": ("BOOLEAN", {"default": True}),
                "transport_mode": (["auto", "neutral_pageable", "direct_d2d", "neutral_pinned"], {"default": "auto"}),
                "benchmark_transport": ("BOOLEAN", {"default": True}),
                "benchmark_mb": ("INT", {"default": 64, "min": 8, "max": 256, "step": 8}),
                "benchmark_repeats": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "allow_explicit_pinned": ("BOOLEAN", {"default": False}),
                "pinned_ring_mb": ("INT", {"default": 64, "min": 8, "max": 128, "step": 8}),
                "secondary_prefetch": ("BOOLEAN", {"default": True}),
                "tp_scope": (["boundary_2", "boundary_6", "all_blocks", "off"], {"default": "boundary_2"}),
                "primary_ffn_groups": ("INT", {
                    "default": 36, "min": 1, "max": 55, "step": 1,
                    "tooltip": "H3 ffn=14336 = 56 scheduling groups of 256. 36/20 gives 64.3%/35.7%, close to the measured 5060 Ti/5060 compute ratio."
                }),
                "tp_prefetch": ("BOOLEAN", {"default": True}),
                "tp_telemetry": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM/GlobalMemory"
    DESCRIPTION = (
        "Dev8.2 Global Tensor Fabric. Keeps Dev7 GLOBAL MEMORY ownership, then row-shards selected H3 FC1 "
        "layers in their native quantized layout (TensorWise INT8 or ConvRot W4A4). Both accelerators execute "
        "disjoint FC1 rows concurrently; GlobalMemorySpace stages the helper input and returns its FC1 rows. "
        "The original full FC2 stays on the block owner to preserve full-K quantization semantics. "
        "No CUDA P2P required. Start with boundary_2."
    )

    def load(self, unet_name, primary_device, secondary_device, split_mode, secondary_blocks,
             primary_reserve_gb=2.5, secondary_reserve_gb=1.5, safe_profile=True,
             transport_mode="auto", benchmark_transport=True, benchmark_mb=64,
             benchmark_repeats=3, allow_explicit_pinned=False, pinned_ring_mb=64,
             secondary_prefetch=True, tp_scope="boundary_2", primary_ffn_groups=36,
             tp_prefetch=True, tp_telemetry=True):
        from .h3vm.loader import build_h3_global_tensor_fabric
        return (build_h3_global_tensor_fabric(
            unet_name=unet_name,
            primary_device=primary_device,
            secondary_device=secondary_device,
            split_mode=split_mode,
            secondary_blocks=int(secondary_blocks),
            primary_reserve_gb=float(primary_reserve_gb),
            secondary_reserve_gb=float(secondary_reserve_gb),
            safe_profile=bool(safe_profile),
            transport_mode=str(transport_mode),
            benchmark_transport=bool(benchmark_transport),
            benchmark_mb=int(benchmark_mb),
            benchmark_repeats=int(benchmark_repeats),
            allow_explicit_pinned=bool(allow_explicit_pinned),
            pinned_ring_mb=int(pinned_ring_mb),
            secondary_prefetch=bool(secondary_prefetch),
            tp_scope=str(tp_scope),
            primary_ffn_groups=int(primary_ffn_groups),
            tp_prefetch=bool(tp_prefetch),
            tp_telemetry=bool(tp_telemetry),
        ),)


class H3VMSnapshotIslandsLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "split_mode": (["speed_first", "balanced", "manual"], {"default": "speed_first"}),
                "secondary_blocks": ("INT", {"default": 17, "min": 1, "max": 49, "step": 1}),
                "primary_reserve_gb": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 8.0, "step": 0.25}),
                "secondary_reserve_gb": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 4.0, "step": 0.25}),
                "safe_profile": ("BOOLEAN", {"default": True}),
                "expected_steps": ("INT", {
                    "default": 20, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Expected H3 model calls for one sampling run. Set this to the sampler step count for BasicGuider single-call sampling."
                }),
                "refresh_interval": ("INT", {
                    "default": 0, "min": 0, "max": 20, "step": 1,
                    "tooltip": "0 = only warm-up/last are exact. N = force an exact boundary refresh every N stale calls."
                }),
                "exact_last_step": ("BOOLEAN", {"default": True}),
                "prefix_prefetch": ("BOOLEAN", {"default": True}),
                "tail_prefetch": ("BOOLEAN", {"default": False}),
                "telemetry": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM/GlobalMemory"
    DESCRIPTION = (
        "Dev9 experimental Snapshot Compute Islands. No GPU-to-GPU activation copy is executed. "
        "Both accelerators communicate only with ordinary pageable system RAM. After an exact warm-up, "
        "the primary tail consumes the previous model-call boundary snapshot while the secondary computes "
        "the current prefix. This is approximate stale-feature inference, intended for A/B speed/quality testing."
    )

    def load(self, unet_name, primary_device, secondary_device, split_mode, secondary_blocks,
             primary_reserve_gb=2.5, secondary_reserve_gb=1.5, safe_profile=True,
             expected_steps=20, refresh_interval=0, exact_last_step=True,
             prefix_prefetch=True, tail_prefetch=True, telemetry=True):
        from .h3vm.loader import build_h3_snapshot_islands
        return (build_h3_snapshot_islands(
            unet_name=unet_name,
            primary_device=primary_device,
            secondary_device=secondary_device,
            split_mode=split_mode,
            secondary_blocks=int(secondary_blocks),
            primary_reserve_gb=float(primary_reserve_gb),
            secondary_reserve_gb=float(secondary_reserve_gb),
            safe_profile=bool(safe_profile),
            expected_steps=int(expected_steps),
            refresh_interval=int(refresh_interval),
            exact_last_step=bool(exact_last_step),
            prefix_prefetch=bool(prefix_prefetch),
            tail_prefetch=bool(tail_prefetch),
            telemetry=bool(telemetry),
        ),)


class H3VMSnapshotIslandsFullThrottleLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "secondary_blocks_target": ("INT", {
                    "default": 0, "min": 0, "max": 49, "step": 1,
                    "tooltip": "0 = compute-balance automatically from GPU SM/clock and memory budget. Nonzero forces a target, still clamped by elastic capacity."
                }),
                "primary_reserve_gb": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 8.0, "step": 0.25}),
                "secondary_reserve_gb": ("FLOAT", {"default": 1.5, "min": 0.75, "max": 4.0, "step": 0.25}),
                "secondary_overcommit_mb": ("INT", {
                    "default": 512, "min": 0, "max": 1024, "step": 64,
                    "tooltip": "Elastic DynamicVRAM allowance beyond the nominal secondary reserve. Dev9.2 redline keeps a hard 0.50GiB floor; use only with DynamicVRAM and --disable-pinned-memory."
                }),
                "safe_profile": ("BOOLEAN", {"default": True}),
                "expected_steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "refresh_interval": ("INT", {
                    "default": 5, "min": 0, "max": 20, "step": 1,
                    "tooltip": "5 is the current quality-safe baseline. 0 means only warm-up/last exact."
                }),
                "exact_last_step": ("BOOLEAN", {"default": True}),
                "predictor_mode": (["stale", "linear"], {"default": "stale"}),
                "predictor_beta": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.25, "step": 0.05,
                    "tooltip": "For linear mode: S_hat=(1+beta)S[t-1]-beta*S[t-2]."
                }),
                "prefix_prefetch": ("BOOLEAN", {"default": True}),
                "tail_prefetch": ("BOOLEAN", {"default": True}),
                "launch_tail_before_stage": ("BOOLEAN", {"default": True}),
                "host_feeder": (["bounded_pinned", "pageable"], {
                    "default": "bounded_pinned",
                    "tooltip": "Dev9.3: bounded_pinned pins only H3VM activation/snapshot mailbox buffers. It does not enable ComfyUI global weight pinning."
                }),
                "pinned_mailbox_mb": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 128,
                    "tooltip": "Hard upper bound for H3VM-owned pinned host buffers. Keep --disable-pinned-memory in the safe startup profile."
                }),
                "secondary_ram_backing_gb": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 4.0, "step": 0.25,
                    "tooltip": "Dev9.4: allow the secondary registered island to exceed its physical fast-pool capacity by this much, relying on CPU-backed DynamicVRAM/VBAR. Pageable RAM backing only, never broad pinned weights."
                }),
                "telemetry": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM/GlobalMemory"
    DESCRIPTION = (
        "Dev9.4 RAM-Backed Redline Snapshot Islands. Keeps NO-D2D semantics, starts the stale tail before "
        "current-prefix host staging, compute-balances the block split with a bounded DynamicVRAM elastic "
        "allowance, and optionally uses a host-side linear snapshot predictor to reduce stale drift without "
        "adding a current-step inter-GPU barrier. Experimental approximate inference."
    )

    def load(self, unet_name, primary_device, secondary_device, secondary_blocks_target=0,
             primary_reserve_gb=2.5, secondary_reserve_gb=1.5, secondary_overcommit_mb=512,
             safe_profile=True, expected_steps=20, refresh_interval=5, exact_last_step=True,
             predictor_mode="stale", predictor_beta=0.75,
             prefix_prefetch=True, tail_prefetch=True, launch_tail_before_stage=True,
             host_feeder="bounded_pinned", pinned_mailbox_mb=1024, secondary_ram_backing_gb=0.0, telemetry=True):
        from .h3vm.loader import build_h3_snapshot_islands_full_throttle
        return (build_h3_snapshot_islands_full_throttle(
            unet_name=unet_name,
            primary_device=primary_device,
            secondary_device=secondary_device,
            secondary_blocks_target=int(secondary_blocks_target),
            primary_reserve_gb=float(primary_reserve_gb),
            secondary_reserve_gb=float(secondary_reserve_gb),
            secondary_overcommit_mb=int(secondary_overcommit_mb),
            safe_profile=bool(safe_profile),
            expected_steps=int(expected_steps),
            refresh_interval=int(refresh_interval),
            exact_last_step=bool(exact_last_step),
            predictor_mode=str(predictor_mode),
            predictor_beta=float(predictor_beta),
            prefix_prefetch=bool(prefix_prefetch),
            tail_prefetch=bool(tail_prefetch),
            launch_tail_before_stage=bool(launch_tail_before_stage),
            host_feeder=str(host_feeder),
            pinned_mailbox_mb=int(pinned_mailbox_mb),
            secondary_ram_backing_gb=float(secondary_ram_backing_gb),
            telemetry=bool(telemetry),
        ),)


class H3VMSnapshotIslandsTurboLoader:
    """Dev9.5 Larry Turbo v4 + 22/28 Snapshot Islands integration.

    Applies Larry's Turbo LoRA to the private H3 before/while ownership is split,
    then runs the same NO-D2D PREDICT075 island runtime. The bypass adapters are
    device-aware so prefix LoRA math lives on cuda:1 and tail LoRA math on cuda:0.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "low_vram": ("BOOLEAN", {"default": False,
                    "tooltip": "OFF=bypass/sharp like Larry default. ON=merge/lower VRAM but softer on quantized base."}),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "secondary_blocks_target": ("INT", {"default": 22, "min": 1, "max": 49, "step": 1}),
                "primary_reserve_gb": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 8.0, "step": 0.25}),
                "secondary_reserve_gb": ("FLOAT", {"default": 1.5, "min": 0.75, "max": 4.0, "step": 0.25}),
                "secondary_overcommit_mb": ("INT", {"default": 1024, "min": 0, "max": 1024, "step": 64}),
                "secondary_ram_backing_gb": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.25}),
                "safe_profile": ("BOOLEAN", {"default": True}),
                "expected_steps": ("INT", {"default": 4, "min": 1, "max": 50, "step": 1,
                    "tooltip": "Must match BasicScheduler steps so exact_last_step lands on the real final denoise call."}),
                "predictor_beta": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.25, "step": 0.05}),
                "host_feeder": (["bounded_pinned", "pageable"], {"default": "bounded_pinned"}),
                "pinned_mailbox_mb": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 128}),
                "telemetry": ("BOOLEAN", {"default": True}),
                "quality_profile": ([
                    "FAST | E-S-P-E",
                    "SAFE | E-S-E-E",
                    "NO-PREDICT | E-S-S-E",
                    "EXACT4 | E-E-E-E",
                ], {"default": "FAST | E-S-P-E", "tooltip": "4-step quality matrix: E=exact, S=stale pipeline, P=linear predicted pipeline."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM/GlobalMemory"
    DESCRIPTION = (
        "Dev10.1 4-step quality matrix on the 22/28 NO-D2D Snapshot Islands runtime. "
        "Use FAST/SAFE/NO-PREDICT/EXACT4 to isolate snapshot approximation from Turbo quality."
    )

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1", secondary_blocks_target=22,
             primary_reserve_gb=2.5, secondary_reserve_gb=1.5, secondary_overcommit_mb=1024,
             secondary_ram_backing_gb=2.0, safe_profile=True, expected_steps=4,
             predictor_beta=0.75, host_feeder="bounded_pinned", pinned_mailbox_mb=1024,
             telemetry=True, quality_profile="FAST | E-S-P-E"):
        from .h3vm.loader import build_h3_snapshot_islands_full_throttle
        profile = str(quality_profile)
        profiles = {
            "FAST | E-S-P-E": ({1, 4}, "linear"),
            "SAFE | E-S-E-E": ({1, 3, 4}, "linear"),
            "NO-PREDICT | E-S-S-E": ({1, 4}, "stale"),
            "EXACT4 | E-E-E-E": ({1, 2, 3, 4}, "stale"),
        }
        if profile not in profiles:
            raise ValueError(f"Unknown H3VM 4-step quality profile: {profile}")
        exact_steps, predictor_mode = profiles[profile]
        if int(expected_steps) != 4:
            raise ValueError("Dev10.1 quality matrix profiles require expected_steps=4")
        return (build_h3_snapshot_islands_full_throttle(
            unet_name=unet_name,
            primary_device=primary_device,
            secondary_device=secondary_device,
            secondary_blocks_target=int(secondary_blocks_target),
            primary_reserve_gb=float(primary_reserve_gb),
            secondary_reserve_gb=float(secondary_reserve_gb),
            secondary_overcommit_mb=int(secondary_overcommit_mb),
            safe_profile=bool(safe_profile),
            expected_steps=int(expected_steps),
            refresh_interval=0,
            exact_last_step=True,
            predictor_mode=predictor_mode,
            predictor_beta=float(predictor_beta),
            prefix_prefetch=True,
            tail_prefetch=True,
            launch_tail_before_stage=True,
            host_feeder=str(host_feeder),
            pinned_mailbox_mb=int(pinned_mailbox_mb),
            secondary_ram_backing_gb=float(secondary_ram_backing_gb),
            telemetry=bool(telemetry),
            turbo_lora_name=str(lora_name),
            turbo_strength=float(strength),
            turbo_low_vram=bool(low_vram),
            exact_steps=exact_steps,
            quality_profile=profile,
        ),)


class H3VMPrimaryFirstExactTurboLoader:
    """Dev11 primary-first exact 4-step Turbo loader.

    The 16G primary owns the front of H3 and starts first. The 8G tail is not
    registered as a startup additional model; it is loaded lazily after primary
    compute begins. No stale snapshot or predictor is used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "low_vram": ("BOOLEAN", {"default": False}),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "primary_runtime_target_pct": ("FLOAT", {
                    "default": 90.0, "min": 70.0, "max": 96.0, "step": 1.0,
                    "tooltip": "Projected runtime occupancy target for the large primary card. Weight planning subtracts workspace reserve before choosing the cut."
                }),
                "primary_workspace_reserve_gb": ("FLOAT", {
                    "default": 1.25, "min": 0.5, "max": 4.0, "step": 0.25,
                }),
                "secondary_runtime_target_pct": ("FLOAT", {
                    "default": 88.0, "min": 70.0, "max": 94.0, "step": 1.0,
                }),
                "secondary_workspace_reserve_gb": ("FLOAT", {
                    "default": 0.75, "min": 0.5, "max": 2.0, "step": 0.25,
                }),
                "cut_override": ("INT", {
                    "default": 0, "min": 0, "max": 49, "step": 1,
                    "tooltip": "0=auto. Nonzero forces number of front blocks on the primary, still checked against both runtime budgets."
                }),
                "expected_steps": ("INT", {"default": 4, "min": 1, "max": 50, "step": 1}),
                "lazy_secondary": ("BOOLEAN", {"default": True}),
                "prefix_prefetch": ("BOOLEAN", {"default": True}),
                "tail_prefetch": ("BOOLEAN", {"default": True}),
                "hard_cleanup_after_sample": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Experimental. Normal Dev11 workflow relies on Dual Video VAE pre/post cleanup. Enable only for H3-only lifecycle debugging."
                }),
                "telemetry": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM/Dev11"
    DESCRIPTION = (
        "Dev11 exact primary-first H3 Turbo. Main/primary prefix starts on the large GPU; "
        "the small-GPU tail is loaded lazily behind primary compute. Exact current-step boundary only."
    )

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_target_pct=90.0, primary_workspace_reserve_gb=1.25,
             secondary_runtime_target_pct=88.0, secondary_workspace_reserve_gb=0.75,
             cut_override=0, expected_steps=4, lazy_secondary=True,
             prefix_prefetch=True, tail_prefetch=True,
             hard_cleanup_after_sample=False, telemetry=True):
        from .h3vm.loader import build_h3_primary_first_exact_turbo
        return (build_h3_primary_first_exact_turbo(
            unet_name=str(unet_name),
            turbo_lora_name=str(lora_name),
            turbo_strength=float(strength),
            turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device),
            secondary_device=str(secondary_device),
            primary_runtime_target_pct=float(primary_runtime_target_pct),
            primary_workspace_reserve_gb=float(primary_workspace_reserve_gb),
            secondary_runtime_target_pct=float(secondary_runtime_target_pct),
            secondary_workspace_reserve_gb=float(secondary_workspace_reserve_gb),
            cut_override=int(cut_override),
            expected_steps=int(expected_steps),
            safe_profile=True,
            prefix_prefetch=bool(prefix_prefetch),
            tail_prefetch=bool(tail_prefetch),
            lazy_secondary=bool(lazy_secondary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample),
            telemetry=bool(telemetry),
        ),)


class H3VMStreamingExactTurboLoader:
    """Dev12 RAM-first striped streaming exact 4-step Turbo loader."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "low_vram": ("BOOLEAN", {"default": False}),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "stripe_size": ("INT", {
                    "default": 2, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Number of consecutive H3 blocks per execution stripe before switching GPU. 2 is the Dev12 proof-of-concept default."
                }),
                "primary_runtime_reserve_gb": ("FLOAT", {
                    "default": 5.0, "min": 2.0, "max": 10.0, "step": 0.25,
                    "tooltip": "VRAM runway reserved for fixed layers, activations, attention/MLP/LoRA workspace and transport on the primary GPU."
                }),
                "secondary_runtime_reserve_gb": ("FLOAT", {
                    "default": 4.0, "min": 2.0, "max": 6.5, "step": 0.25,
                    "tooltip": "VRAM runway protected on the 8G helper. Dev12 intentionally keeps this large."
                }),
                "primary_hot_cache_gb": ("FLOAT", {
                    "default": 3.5, "min": 0.5, "max": 8.0, "step": 0.25,
                    "tooltip": "Maximum desired resident DynamicVRAM block cache on primary. Cold weights return to RAM at stripe boundaries."
                }),
                "secondary_hot_cache_gb": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 4.5, "step": 0.25,
                    "tooltip": "Maximum desired resident DynamicVRAM block cache on secondary. Keep enough room for the ~1.8GiB MLP/LoRA workspace spike seen in Dev11."
                }),
                "expected_steps": ("INT", {"default": 4, "min": 1, "max": 50, "step": 1}),
                "one_ahead_prefetch": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use ComfyUI DynamicVBAR one-ahead prefetch only. No whole-island warmup."
                }),
                "trim_on_stripe_boundary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Evict cold block residency back to system RAM whenever execution hands off to the other GPU."
                }),
                "hard_cleanup_after_sample": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Extra allocator cleanup for debugging. Block islands are already cold-trimmed after every sample."
                }),
                "telemetry": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM/Dev12"
    DESCRIPTION = (
        "Dev12 RAM-first exact H3 Turbo. System RAM is the block-weight backing store; "
        "both GPUs keep only a bounded DynamicVRAM hot working set and explicit workspace reserves. "
        "Blocks execute in original order through striped GPU ownership with host-neutral exact handoffs."
    )

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1", stripe_size=2,
             primary_runtime_reserve_gb=5.0, secondary_runtime_reserve_gb=4.0,
             primary_hot_cache_gb=3.5, secondary_hot_cache_gb=2.0,
             expected_steps=4, one_ahead_prefetch=True,
             trim_on_stripe_boundary=True, hard_cleanup_after_sample=False,
             telemetry=True):
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name),
            turbo_lora_name=str(lora_name),
            turbo_strength=float(strength),
            turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device),
            secondary_device=str(secondary_device),
            stripe_size=int(stripe_size),
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb),
            secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps),
            safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch),
            trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample),
            telemetry=bool(telemetry),
        ),)


class H3VMStreamingExactAttentionTurboLoader:
    """Dev12.1 RAM-first streaming + exact two-GPU attention-head parallel Turbo loader."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "low_vram": ("BOOLEAN", {"default": False}),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "stripe_size": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "primary_runtime_reserve_gb": ("FLOAT", {
                    "default": 5.5, "min": 2.0, "max": 10.0, "step": 0.25,
                    "tooltip": "Extra runway for helper attention plus fixed/runtime workspace on the 16G card."
                }),
                "secondary_runtime_reserve_gb": ("FLOAT", {
                    "default": 4.75, "min": 2.0, "max": 6.5, "step": 0.25,
                    "tooltip": "Protect the 8G card for attention helper workspace and MLP/LoRA spikes."
                }),
                "primary_hot_cache_gb": ("FLOAT", {
                    "default": 3.0, "min": 0.5, "max": 8.0, "step": 0.25,
                }),
                "secondary_hot_cache_gb": ("FLOAT", {
                    "default": 1.25, "min": 0.5, "max": 4.5, "step": 0.25,
                }),
                "expected_steps": ("INT", {"default": 4, "min": 1, "max": 50, "step": 1}),
                "one_ahead_prefetch": ("BOOLEAN", {"default": False, "tooltip": "Dev12.1 default isolates attention parallel from block prefetch contention. Re-enable only after the first 5s validation."}),
                "trim_on_stripe_boundary": ("BOOLEAN", {"default": True}),
                "attention_mode": (["relay_auto", "force_relay", "p2p_only", "off"], {"default": "relay_auto"}),
                "attention_head_balance": (["sm_weighted", "balanced"], {"default": "sm_weighted"}),
                "attention_min_sequence_length": ("INT", {
                    "default": 8192, "min": 1, "max": 262144, "step": 1024,
                    "tooltip": "Only split attention across two GPUs above this sequence length."
                }),
                "attention_relay_min_gbps": ("FLOAT", {
                    "default": 3.0, "min": 0.5, "max": 64.0, "step": 0.5,
                    "tooltip": "relay_auto disables helper attention if measured fallback copy bandwidth is below this threshold."
                }),
                "attention_helper_head_cap": ("INT", {
                    "default": 24, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Head cap for the physical 8G secondary GPU, independent of which GPU owns the current stripe. Runtime free-VRAM may reduce it further."
                }),
                "attention_helper_safety_mb": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 128,
                    "tooltip": "Physical free-VRAM guard left untouched when sizing helper head count."
                }),
                "hard_cleanup_after_sample": ("BOOLEAN", {"default": False}),
                "telemetry": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM/Dev12.1"
    DESCRIPTION = (
        "Dev12.1 combines RAM-first striped DynamicVRAM streaming with exact disjoint attention-head parallelism. "
        "The current block remains mathematically ordered, while its attention heads are computed concurrently on both GPUs. "
        "When native P2P is unavailable, relay_auto uses the measured CUDA fallback copy route only if it is fast enough."
    )

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1", stripe_size=2,
             primary_runtime_reserve_gb=5.5, secondary_runtime_reserve_gb=4.75,
             primary_hot_cache_gb=3.0, secondary_hot_cache_gb=1.25,
             expected_steps=4, one_ahead_prefetch=False, trim_on_stripe_boundary=True,
             attention_mode="relay_auto", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_relay_min_gbps=3.0,
             attention_helper_head_cap=24, attention_helper_safety_mb=1024,
             hard_cleanup_after_sample=False, telemetry=True):
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device),
            stripe_size=int(stripe_size),
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb),
            secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch),
            trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample),
            telemetry=bool(telemetry),
            attention_mode=str(attention_mode),
            attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length),
            attention_relay_min_gbps=float(attention_relay_min_gbps),
            attention_helper_head_cap=int(attention_helper_head_cap),
            attention_helper_safety_mb=int(attention_helper_safety_mb),
        ),)


class H3VMStreamingExactHostAttentionTurboLoader:
    """Dev12.2 RAM-first streaming + bounded host-relay exact attention parallel."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "low_vram": ("BOOLEAN", {"default": False}),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "stripe_size": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "primary_runtime_reserve_gb": ("FLOAT", {
                    "default": 5.5, "min": 2.0, "max": 10.0, "step": 0.25,
                    "tooltip": "Keep the Dev12 low-pressure runway on the 16G card."
                }),
                "secondary_runtime_reserve_gb": ("FLOAT", {
                    "default": 4.75, "min": 2.0, "max": 6.5, "step": 0.25,
                    "tooltip": "Protect the 8G card from helper-attention and MLP/LoRA spikes."
                }),
                "primary_hot_cache_gb": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 8.0, "step": 0.25}),
                "secondary_hot_cache_gb": ("FLOAT", {"default": 1.25, "min": 0.5, "max": 4.5, "step": 0.25}),
                "expected_steps": ("INT", {"default": 4, "min": 1, "max": 50, "step": 1}),
                "one_ahead_prefetch": ("BOOLEAN", {"default": False,
                    "tooltip": "Keep block prefetch off while validating host-relay attention overlap."}),
                "trim_on_stripe_boundary": ("BOOLEAN", {"default": True}),
                "attention_mode": (["host_auto", "force_host", "p2p_only", "off"], {"default": "host_auto"}),
                "attention_head_balance": (["sm_weighted", "balanced"], {"default": "sm_weighted"}),
                "attention_min_sequence_length": ("INT", {
                    "default": 8192, "min": 1, "max": 262144, "step": 1024,
                    "tooltip": "Only split attention across both GPUs above this sequence length."
                }),
                "attention_helper_head_cap": ("INT", {
                    "default": 24, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Hard head ceiling for the physical 8G GPU."
                }),
                "attention_helper_safety_mb": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 128,
                    "tooltip": "Free-VRAM floor preserved before assigning helper heads."
                }),
                "attention_host_ring_mb": ("INT", {
                    "default": 64, "min": 16, "max": 256, "step": 16,
                    "tooltip": "Per-slot H3VM-owned pinned host DMA runway. Two slots are allocated; never stores model weights."
                }),
                "hard_cleanup_after_sample": ("BOOLEAN", {"default": False}),
                "telemetry": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM/Dev12.2"
    DESCRIPTION = (
        "Dev12.2 keeps the RAM-first/low-VRAM Dev12 execution model, but bypasses slow WDDM fallback D2D "
        "for helper attention. Quantized helper heads cross through a tiny bounded host RAM DMA runway, then "
        "root/helper INT8 attention kernels launch concurrently. Exact head partitioning; no stale/predict/cache skip."
    )

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1", stripe_size=2,
             primary_runtime_reserve_gb=5.5, secondary_runtime_reserve_gb=4.75,
             primary_hot_cache_gb=3.0, secondary_hot_cache_gb=1.25,
             expected_steps=4, one_ahead_prefetch=False, trim_on_stripe_boundary=True,
             attention_mode="host_auto", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=24,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             hard_cleanup_after_sample=False, telemetry=True):
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device),
            stripe_size=int(stripe_size),
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb),
            secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch),
            trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample),
            telemetry=bool(telemetry),
            attention_mode=str(attention_mode),
            attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length),
            attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap),
            attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
        ),)


class H3VMDualVideoVAEDecode:
    """Dev10 dual-GPU temporal chunk decoder for MiniMax H3 Video VAE."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "cleanup_h3": ("BOOLEAN", {"default": True,
                    "tooltip": "Unload/demote H3 residency before bringing both Video VAE decoders online."}),
                "telemetry": ("BOOLEAN", {"default": True}),
                "post_cleanup": ("BOOLEAN", {"default": True,
                    "tooltip": "Dev11: after dual decode, unload both VAE residencies, drop secondary VAE cache, and soft-empty both CUDA caches so the next prompt starts clean."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "MiniMaxH3/H3VM/VAE"
    DESCRIPTION = (
        "Dev10 experimental dual-GPU MiniMax H3 Video VAE decoder. It keeps the official temporal "
        "chunk/overlap plan, schedules independent chunks across two isolated VAE instances, then "
        "merges on CPU. No GPU-to-GPU tensor handoff is required."
    )

    def decode(self, samples, vae, vae_name, primary_device="gpu:0", secondary_device="gpu:1",
               cleanup_h3=True, telemetry=True, post_cleanup=True):
        from .h3vm.dual_video_vae import dual_decode_h3_video
        return (dual_decode_h3_video(
            vae=vae, samples=samples, vae_name=vae_name,
            primary_device=str(primary_device), secondary_device=str(secondary_device),
            cleanup_h3=bool(cleanup_h3), telemetry=bool(telemetry),
            post_cleanup=bool(post_cleanup),
        ),)




class H3VMDev122HForceHostAttentionTurboLoader(H3VMStreamingExactHostAttentionTurboLoader):
    """Dev12.2H fail-closed validation node. Always forces bounded host relay."""
    CATEGORY = "MiniMaxH3/H3VM/Dev12.2H"
    DESCRIPTION = (
        "Fail-closed Dev12.2 Host-Relay validation node. Uses a unique node id so an older plugin cannot "
        "silently fall back to Dev12.1. Attention transport is forced through H3VM-owned bounded host RAM."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        # Keep the widget visible but make the only legal value force_host.
        req["attention_mode"] = (["force_host"], {"default": "force_host"})
        spec = dict(spec)
        spec["required"] = req
        return spec

    def load(self, *args, attention_mode="force_host", **kwargs):
        print('[H3VM DEV12.2H FORCE-HOST NODE] attention_mode=force_host', flush=True)
        return super().load(*args, attention_mode="force_host", **kwargs)



class H3VMDev123TokenMLPTurboLoader(H3VMDev122HForceHostAttentionTurboLoader):
    """Dev12.3 fail-closed dual-CUDA Attention + token-parallel MLP."""
    CATEGORY = "MiniMaxH3/H3VM/Dev12.3"
    DESCRIPTION = (
        "Dev12.3 keeps the Dev12 RAM-first low-residency runtime and force-host exact attention, then "
        "splits the H3 MLP by token rows. Each GPU runs complete FC1->SwiGLU->FC2 on disjoint token ranges; "
        "only MLP input/output hidden rows cross the bounded host relay. This preserves per-token INT8 quantization."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        req["primary_runtime_reserve_gb"] = ("FLOAT", {"default": 6.0, "min": 2.0, "max": 10.0, "step": 0.25})
        req["secondary_runtime_reserve_gb"] = ("FLOAT", {"default": 5.25, "min": 2.0, "max": 6.75, "step": 0.25})
        req["primary_hot_cache_gb"] = ("FLOAT", {"default": 2.5, "min": 0.5, "max": 8.0, "step": 0.25})
        req["secondary_hot_cache_gb"] = ("FLOAT", {"default": 0.75, "min": 0.25, "max": 4.5, "step": 0.25})
        req["mlp_primary_fraction"] = ("FLOAT", {
            "default": 0.60, "min": 0.30, "max": 0.80, "step": 0.01,
            "tooltip": "Fraction of token rows assigned to the physical primary GPU, independent of stripe owner."
        })
        req["mlp_min_sequence_length"] = ("INT", {
            "default": 8192, "min": 256, "max": 262144, "step": 256,
            "tooltip": "Below this packed sequence length, keep the MLP on the block owner."
        })
        req["mlp_helper_safety_mb"] = ("INT", {
            "default": 768, "min": 256, "max": 4096, "step": 128,
            "tooltip": "Physical free-VRAM floor before helper MLP participation."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1", stripe_size=2,
             primary_runtime_reserve_gb=6.0, secondary_runtime_reserve_gb=5.25,
             primary_hot_cache_gb=2.5, secondary_hot_cache_gb=0.75,
             expected_steps=4, one_ahead_prefetch=False, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=24,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             mlp_primary_fraction=0.60, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, hard_cleanup_after_sample=False, telemetry=True):
        print('[H3VM DEV12.3 TOKEN-MLP NODE] ATTN=force_host MLP=token2gpu', flush=True)
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device),
            stripe_size=int(stripe_size),
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb),
            secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch),
            trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length), attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap),
            attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
            mlp_token_parallel=True, mlp_primary_fraction=float(mlp_primary_fraction),
            mlp_min_sequence_length=int(mlp_min_sequence_length),
            mlp_helper_safety_mb=int(mlp_helper_safety_mb),
        ),)



class H3VMDev123HTokenMLPConnectedTurboLoader(H3VMDev123TokenMLPTurboLoader):
    """Dev12.3H: fail-closed wiring hotfix for token-parallel MLP."""
    CATEGORY = "MiniMaxH3/H3VM/Dev12.3H"
    DESCRIPTION = (
        "Dev12.3H wiring hotfix. Same exact RAM-first + host-relay attention + token-parallel MLP design, "
        "but the MLP fabric is installed in the actual Dev12 streaming builder and must patch all 50 H3 blocks."
    )

    def load(self, *args, **kwargs):
        print('[H3VM DEV12.3H CONNECTED TOKEN-MLP NODE] fail_closed=50/50', flush=True)
        return super().load(*args, **kwargs)


class H3VMDev13SingleRootCoprocessorTurboLoader(H3VMDev123TokenMLPTurboLoader):
    """Dev13: all 50 H3 blocks stay rooted on GPU0; GPU1 is compute-only helper."""
    CATEGORY = "MiniMaxH3/H3VM/Dev13"
    DESCRIPTION = (
        "Dev13 removes striped whole-block ownership. The physical primary executes all 50 H3 blocks "
        "from RAM-backed DynamicVRAM, while the secondary participates only inside exact Attention and "
        "token-parallel MLP. This targets zero full-hidden-state block boundaries."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        # Stripe ownership is gone; keep the UI honest by removing the obsolete control.
        req.pop("stripe_size", None)
        req["mlp_primary_fraction"] = ("FLOAT", {
            "default": 0.50, "min": 0.30, "max": 0.75, "step": 0.01,
            "tooltip": "Dev13 initial token split. 0.50 is intentionally neutral for the first single-root benchmark."
        })
        req["enable_token_mlp"] = ("BOOLEAN", {
            "default": False,
            "tooltip": "LAB toggle. Keep OFF for the first Dev13 boundary-removal benchmark; turn ON only for the second A/B run."
        })
        req["single_root_trim_interval"] = ("INT", {
            "default": 2, "min": 1, "max": 10, "step": 1,
            "tooltip": "Trim cold root/helper DynamicVRAM weights every N blocks; no activation boundary transfer is performed."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_reserve_gb=6.0, secondary_runtime_reserve_gb=5.25,
             primary_hot_cache_gb=2.5, secondary_hot_cache_gb=0.75,
             expected_steps=4, one_ahead_prefetch=False, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=24,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             mlp_primary_fraction=0.50, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, enable_token_mlp=False, single_root_trim_interval=2,
             hard_cleanup_after_sample=False, telemetry=True):
        print(f'[H3VM DEV13 SINGLE-ROOT NODE] root=gpu0 helper=gpu1 boundaries=0 target MLP={"on" if enable_token_mlp else "off"}', flush=True)
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device),
            stripe_size=50,
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb),
            secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch),
            trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length), attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap),
            attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
            mlp_token_parallel=bool(enable_token_mlp), mlp_primary_fraction=float(mlp_primary_fraction),
            mlp_min_sequence_length=int(mlp_min_sequence_length),
            mlp_helper_safety_mb=int(mlp_helper_safety_mb),
            single_root=True, single_root_trim_interval=int(single_root_trim_interval),
        ),)



class H3VMDev130HSingleRootNoEmptyVBARTurboLoader(H3VMDev13SingleRootCoprocessorTurboLoader):
    """Dev13.0H: fail-closed Single-Root without empty GPU1 VBAR islands."""
    CATEGORY = "MiniMaxH3/H3VM/Dev13.0H"
    DESCRIPTION = (
        "Dev13.0H hotfix. Keeps all 50 H3 blocks rooted on GPU0 and completely omits the empty GPU1 "
        "whole-block ModelPatcher/VBAR. GPU1 exists only as the exact Attention/MLP coprocessor."
    )

    def load(self, *args, **kwargs):
        print('[H3VM DEV13.0H NO-EMPTY-VBAR NODE] root=gpu0 helper=gpu1 whole_block_secondary=NONE', flush=True)
        return super().load(*args, **kwargs)


class H3VMDev131FedCoprocessorTurboLoader(H3VMDev130HSingleRootNoEmptyVBARTurboLoader):
    """Dev13.1: feed the 8G coprocessor with a bounded resident helper cache."""
    CATEGORY = "MiniMaxH3/H3VM/Dev13.1"
    DESCRIPTION = (
        "Dev13.1 keeps Single-Root exact semantics but stops starving GPU1: a bounded 3.5GiB "
        "MLP-helper working set may remain resident on the 8G card, trims are less frequent, and "
        "one-ahead root prefetch is enabled. RAM remains the backing store."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        req["secondary_runtime_reserve_gb"] = ("FLOAT", {
            "default": 3.0, "min": 2.0, "max": 5.0, "step": 0.25,
            "tooltip": "Runtime/activation reserve on the 8G coprocessor. The rest may become bounded helper hot cache."
        })
        req["secondary_hot_cache_gb"] = ("FLOAT", {
            "default": 3.5, "min": 1.0, "max": 4.5, "step": 0.25,
            "tooltip": "Resident MLP-helper weight window on GPU1. This is a cache, not a second full model."
        })
        req["mlp_primary_fraction"] = ("FLOAT", {
            "default": 0.60, "min": 0.40, "max": 0.75, "step": 0.01,
            "tooltip": "Keep 60/40 initially: GPU0 compute overlaps the unavoidable host staging to GPU1."
        })
        req["enable_token_mlp"] = ("BOOLEAN", {
            "default": True,
            "tooltip": "Dev13.1 is specifically the fed-coprocessor MLP experiment."
        })
        req["single_root_trim_interval"] = ("INT", {
            "default": 8, "min": 2, "max": 25, "step": 1,
            "tooltip": "Trim every N blocks. Helper weights are retained up to secondary_hot_cache_gb instead of evicted to zero."
        })
        req["one_ahead_prefetch"] = ("BOOLEAN", {
            "default": True,
            "tooltip": "Overlap next root-block weight DMA with current block compute."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, *args, **kwargs):
        kwargs["enable_token_mlp"] = True
        print("[H3VM DEV13.1 FED-COPROCESSOR NODE] helper_cache=bounded-resident MLP=on root_prefetch=on", flush=True)
        return super().load(*args, **kwargs)


class H3VMDev14CriticalPathTurboLoader(H3VMDev130HSingleRootNoEmptyVBARTurboLoader):
    """Dev14: engineering-network critical-path scheduler."""
    CATEGORY = "MiniMaxH3/H3VM/Dev14"
    DESCRIPTION = (
        "Dev14 treats GPU0 as the CPM critical path. GPU1 is a fixed resident sidecar: "
        "primary Attention launches first, and only a small evenly-spread MLP package is delegated. "
        "No per-block helper prefetch is allowed; helper deadline misses disable that block on later steps."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        req["primary_runtime_reserve_gb"] = ("FLOAT", {
            "default": 5.5, "min": 3.0, "max": 8.0, "step": 0.25,
            "tooltip": "GPU0 critical-path workspace reserve. Do not starve the root."
        })
        req["primary_hot_cache_gb"] = ("FLOAT", {
            "default": 3.0, "min": 1.0, "max": 5.0, "step": 0.25,
            "tooltip": "GPU0 block hot cache; root one-ahead prefetch is allowed."
        })
        req["secondary_runtime_reserve_gb"] = ("FLOAT", {
            "default": 3.25, "min": 2.5, "max": 5.0, "step": 0.25,
            "tooltip": "8G sidecar runtime reserve. Helper work is deliberately small."
        })
        req["secondary_hot_cache_gb"] = ("FLOAT", {
            "default": 3.5, "min": 2.0, "max": 4.0, "step": 0.25,
            "tooltip": "Bounded resident sidecar MLP package; not a whole-model cache."
        })
        req["attention_helper_head_cap"] = ("INT", {
            "default": 16, "min": 4, "max": 24, "step": 1,
            "tooltip": "Small-GPU attention cap. Primary launches first and must remain the critical path."
        })
        req["mlp_primary_fraction"] = ("FLOAT", {
            "default": 0.75, "min": 0.60, "max": 0.90, "step": 0.01,
            "tooltip": "Root token share on selected MLP blocks. Default gives GPU1 only 25% so it should finish early."
        })
        req["sidecar_mlp_blocks"] = ("INT", {
            "default": 14, "min": 4, "max": 18, "step": 1,
            "tooltip": "Fixed evenly-spread MLP work package resident on the 8G sidecar. No 50-block helper prefetch."
        })
        req["primary_stall_budget_ms"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.5,
            "tooltip": "If helper estimated deadline miss exceeds this, that block is disabled on later denoise steps."
        })
        req["enable_token_mlp"] = ("BOOLEAN", {
            "default": True, "tooltip": "Dev14 fixed sidecar package. Keep ON for the critical-path experiment."
        })
        req["single_root_trim_interval"] = ("INT", {
            "default": 8, "min": 2, "max": 25, "step": 1,
            "tooltip": "Root/helper cache maintenance interval. Sidecar package is retained up to its hot-cache budget."
        })
        req["one_ahead_prefetch"] = ("BOOLEAN", {
            "default": True, "tooltip": "Root block one-ahead only. Sidecar per-block helper prefetch is forbidden."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_reserve_gb=5.5, secondary_runtime_reserve_gb=3.25,
             primary_hot_cache_gb=3.0, secondary_hot_cache_gb=3.5,
             expected_steps=4, one_ahead_prefetch=True, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=16,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             mlp_primary_fraction=0.75, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, enable_token_mlp=True, single_root_trim_interval=8,
             sidecar_mlp_blocks=14, primary_stall_budget_ms=1.0,
             hard_cleanup_after_sample=False, telemetry=True):
        print(
            f"[H3VM DEV14 CRITICAL-PATH NODE] root=gpu0 nonstop | sidecar_blocks={int(sidecar_mlp_blocks)} "
            f"helper_tokens={(1.0-float(mlp_primary_fraction))*100:.0f}% | per_block_prefetch=OFF | guard={float(primary_stall_budget_ms):.1f}ms",
            flush=True,
        )
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device),
            stripe_size=50,
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb),
            secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch),
            trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length), attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap),
            attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
            mlp_token_parallel=bool(enable_token_mlp), mlp_primary_fraction=float(mlp_primary_fraction),
            mlp_min_sequence_length=int(mlp_min_sequence_length),
            mlp_helper_safety_mb=int(mlp_helper_safety_mb),
            single_root=True, single_root_trim_interval=int(single_root_trim_interval),
            critical_path_mode=True, sidecar_mlp_blocks=int(sidecar_mlp_blocks),
            primary_stall_budget_ms=float(primary_stall_budget_ms), attention_primary_first=True,
        ),)

class H3VMDev141SlackHarvestTurboLoader(H3VMDev14CriticalPathTurboLoader):
    """Dev14.1: harvest positive CPM slack without extending startup/critical path."""
    CATEGORY = "MiniMaxH3/H3VM/Dev14"
    DESCRIPTION = (
        "Dev14.1 keeps GPU0 on the critical path, launches the GPU1 resident sidecar asynchronously behind the first "
        "few H3 blocks, and adaptively converts positive helper slack into less GPU0 MLP work on later denoise steps."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        req["secondary_runtime_reserve_gb"] = ("FLOAT", {
            "default": 2.75, "min": 2.25, "max": 4.0, "step": 0.25,
            "tooltip": "GPU1 runtime reserve. Dev14.1 uses a ~4GiB fixed MLP package plus Attention workspace."
        })
        req["secondary_hot_cache_gb"] = ("FLOAT", {
            "default": 4.25, "min": 3.0, "max": 4.75, "step": 0.25,
            "tooltip": "Resident helper package budget for 18 selected MLP blocks on the 8GiB sidecar."
        })
        req["mlp_primary_fraction"] = ("FLOAT", {
            "default": 0.72, "min": 0.68, "max": 0.82, "step": 0.01,
            "tooltip": "Initial root token share. Per-block CPM slack then moves this toward 0.68 or back toward the root."
        })
        req["sidecar_mlp_blocks"] = ("INT", {
            "default": 18, "min": 10, "max": 20, "step": 1,
            "tooltip": "Fixed resident helper-weight package. Dev14.1 starts using it only after a GPU0 launch runway."
        })
        req["single_root_trim_interval"] = ("INT", {
            "default": 10, "min": 4, "max": 25, "step": 1,
            "tooltip": "Cache maintenance interval; helper package stays resident within its bounded cache."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_reserve_gb=5.5, secondary_runtime_reserve_gb=2.75,
             primary_hot_cache_gb=3.0, secondary_hot_cache_gb=4.25,
             expected_steps=4, one_ahead_prefetch=True, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=16,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             mlp_primary_fraction=0.72, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, enable_token_mlp=True, single_root_trim_interval=10,
             sidecar_mlp_blocks=18, primary_stall_budget_ms=1.0,
             hard_cleanup_after_sample=False, telemetry=True):
        print(
            f"[H3VM DEV14.1 SLACK-HARVEST NODE] GPU0=critical nonstop | sidecar_blocks={int(sidecar_mlp_blocks)} "
            f"initial_helper_tokens={(1.0-float(mlp_primary_fraction))*100:.0f}% | launch_runway=5 blocks | sidecar_prepare=ASYNC",
            flush=True,
        )
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device),
            stripe_size=50,
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb),
            secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch),
            trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length), attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap),
            attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
            mlp_token_parallel=bool(enable_token_mlp), mlp_primary_fraction=float(mlp_primary_fraction),
            mlp_min_sequence_length=int(mlp_min_sequence_length),
            mlp_helper_safety_mb=int(mlp_helper_safety_mb),
            single_root=True, single_root_trim_interval=int(single_root_trim_interval),
            critical_path_mode=True, sidecar_mlp_blocks=int(sidecar_mlp_blocks), sidecar_start_block=5,
            primary_stall_budget_ms=float(primary_stall_budget_ms),
            critical_path_adaptive_slack=True, critical_path_min_primary_fraction=0.68,
            critical_path_max_primary_fraction=0.84, critical_path_fraction_step=0.02,
            critical_path_target_slack_ms=8.0, attention_primary_first=True,
        ),)


class H3VMDev142CriticalPathCompressorTurboLoader(H3VMDev141SlackHarvestTurboLoader):
    """Dev14.2: compress the GPU0 critical path with a fully resident 8G sidecar packet."""
    CATEGORY = "MiniMaxH3/H3VM/Dev14"
    DESCRIPTION = (
        "Dev14.2 keeps GPU0 as the only critical-path owner, force-loads one bounded fixed MLP helper packet on GPU1 "
        "in the background, warms that packet before it can join the graph, and harvests helper slack toward a ~60/40 "
        "token split without ever making sidecar preparation a prerequisite for GPU0 progress."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        req["primary_runtime_reserve_gb"] = ("FLOAT", {
            "default": 5.25, "min": 4.5, "max": 7.0, "step": 0.25,
            "tooltip": "GPU0 workspace reserve. Dev14.2 gives the root a slightly larger hot window while protecting runtime space."
        })
        req["primary_hot_cache_gb"] = ("FLOAT", {
            "default": 3.5, "min": 2.5, "max": 5.0, "step": 0.25,
            "tooltip": "Root DynamicVRAM hot cache. Larger than Dev14.1 to shorten weight-fault segments on the critical line."
        })
        req["secondary_runtime_reserve_gb"] = ("FLOAT", {
            "default": 2.5, "min": 2.25, "max": 3.5, "step": 0.25,
            "tooltip": "GPU1 runtime/Attention reserve. The selected helper packet itself is force-resident in the remaining 8GiB budget."
        })
        req["secondary_hot_cache_gb"] = ("FLOAT", {
            "default": 4.75, "min": 4.0, "max": 5.25, "step": 0.25,
            "tooltip": "Retention ceiling for the ~20-block resident sidecar packet. It is not a per-block prefetch cache."
        })
        req["mlp_primary_fraction"] = ("FLOAT", {
            "default": 0.68, "min": 0.60, "max": 0.78, "step": 0.01,
            "tooltip": "Initial root token share. Positive CPM slack is harvested toward 0.60; deadline pressure returns work to GPU0."
        })
        req["sidecar_mlp_blocks"] = ("INT", {
            "default": 20, "min": 16, "max": 20, "step": 1,
            "tooltip": "Fixed resident GPU1 work packet. 20 selected H3 MLPs stays within the tested 8GiB envelope with runtime reserve."
        })
        req["single_root_trim_interval"] = ("INT", {
            "default": 20, "min": 8, "max": 25, "step": 1,
            "tooltip": "Coarse root-cache maintenance only. The resident sidecar packet is retained; no per-block helper trimming."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_reserve_gb=5.25, secondary_runtime_reserve_gb=2.5,
             primary_hot_cache_gb=3.5, secondary_hot_cache_gb=4.75,
             expected_steps=4, one_ahead_prefetch=True, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=16,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             mlp_primary_fraction=0.68, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, enable_token_mlp=True, single_root_trim_interval=20,
             sidecar_mlp_blocks=20, primary_stall_budget_ms=1.0,
             hard_cleanup_after_sample=False, telemetry=True):
        print(
            f"[H3VM DEV14.2 CRITICAL-PATH COMPRESSOR NODE] GPU0=critical nonstop | resident_sidecar={int(sidecar_mlp_blocks)} blocks | "
            f"initial={float(mlp_primary_fraction)*100:.0f}/{(1.0-float(mlp_primary_fraction))*100:.0f} | "
            f"adaptive_floor=60/40 | helper_packet=FULL-RESIDENT-ASYNC | per_block_prefetch=OFF",
            flush=True,
        )
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device),
            stripe_size=50,
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb),
            secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch),
            trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length), attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap),
            attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
            mlp_token_parallel=bool(enable_token_mlp), mlp_primary_fraction=float(mlp_primary_fraction),
            mlp_min_sequence_length=int(mlp_min_sequence_length),
            mlp_helper_safety_mb=int(mlp_helper_safety_mb),
            single_root=True, single_root_trim_interval=int(single_root_trim_interval),
            critical_path_mode=True, sidecar_mlp_blocks=int(sidecar_mlp_blocks), sidecar_start_block=5,
            primary_stall_budget_ms=float(primary_stall_budget_ms),
            critical_path_adaptive_slack=True, critical_path_min_primary_fraction=0.60,
            critical_path_max_primary_fraction=0.80, critical_path_fraction_step=0.02,
            critical_path_target_slack_ms=4.0, critical_path_resident_sidecar=True,
            attention_primary_first=True,
        ),)


class H3VMDev150RedlinePipelineTurboLoader(H3VMDev142CriticalPathCompressorTurboLoader):
    """Dev15.0: critical-line pipelining. GPU0 is never synchronized block-by-block."""
    CATEGORY = "MiniMaxH3/H3VM/Dev15"
    DESCRIPTION = (
        "Dev15 treats GPU0 as the CPM red line: 4 H3 blocks are queued as a CUDA window before one retirement barrier, "
        "so CPU weight logistics can overlap GPU0 execution. GPU1 keeps the resident 20-block sidecar packet, while deeper "
        "MLP cuts require repeated positive slack instead of reacting to one noisy timing sample."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        req["mlp_primary_fraction"] = ("FLOAT", {
            "default": 0.68, "min": 0.64, "max": 0.76, "step": 0.01,
            "tooltip": "Initial root share. Dev15 allows a validated first cut to 66%; deeper cuts require two positive-slack confirmations."
        })
        req["single_root_trim_interval"] = ("INT", {
            "default": 20, "min": 12, "max": 50, "step": 1,
            "tooltip": "Root cache maintenance interval. CUDA retirement is separately pipelined in 4-block windows."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_reserve_gb=5.25, secondary_runtime_reserve_gb=2.5,
             primary_hot_cache_gb=3.5, secondary_hot_cache_gb=4.75,
             expected_steps=4, one_ahead_prefetch=True, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=16,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             mlp_primary_fraction=0.68, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, enable_token_mlp=True, single_root_trim_interval=20,
             sidecar_mlp_blocks=20, primary_stall_budget_ms=1.0,
             hard_cleanup_after_sample=False, telemetry=True):
        print(
            f"[H3VM DEV15 REDLINE NODE] GPU0=critical nonstop | pipeline=4 blocks | resident_sidecar={int(sidecar_mlp_blocks)} | "
            f"MLP initial={float(mlp_primary_fraction)*100:.0f}/{(1.0-float(mlp_primary_fraction))*100:.0f} | "
            f"deep_harvest=2x-confirm | per-block GPU0 synchronize=OFF",
            flush=True,
        )
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device),
            stripe_size=50,
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb),
            secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch),
            trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length), attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap),
            attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
            mlp_token_parallel=bool(enable_token_mlp), mlp_primary_fraction=float(mlp_primary_fraction),
            mlp_min_sequence_length=int(mlp_min_sequence_length),
            mlp_helper_safety_mb=int(mlp_helper_safety_mb),
            single_root=True, single_root_trim_interval=int(single_root_trim_interval),
            critical_path_mode=True, sidecar_mlp_blocks=int(sidecar_mlp_blocks), sidecar_start_block=5,
            primary_stall_budget_ms=float(primary_stall_budget_ms),
            critical_path_adaptive_slack=True, critical_path_min_primary_fraction=0.64,
            critical_path_max_primary_fraction=0.78, critical_path_fraction_step=0.02,
            critical_path_target_slack_ms=6.0, critical_path_resident_sidecar=True,
            critical_path_pipeline_window=4, critical_path_harvest_confirmations=2,
            attention_primary_first=True,
        ),)


class H3VMDev151ResourceConstrainedRedlineTurboLoader(H3VMDev150RedlinePipelineTurboLoader):
    """Dev15.1: protect GPU0 kernel throughput from side-line DMA contention."""
    CATEGORY = "MiniMaxH3/H3VM/Dev15"
    DESCRIPTION = (
        "Dev15.1 is resource-constrained CPM: GPU0 keeps a larger DynamicVRAM hot set, only two H3 blocks are allowed "
        "ahead of the CPU, normal retirements wait on the critical-stream event rather than synchronizing the whole GPU, "
        "and the resident GPU1 MLP sidecar is held at the validated ~68/32 split instead of chasing utilization."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        req["primary_runtime_reserve_gb"] = ("FLOAT", {
            "default": 5.25, "min": 4.75, "max": 7.0, "step": 0.25,
            "tooltip": "Keep the proven long-run GPU0 workspace reserve; Dev15.1 spends only genuinely spare VRAM on hot weights."
        })
        req["primary_hot_cache_gb"] = ("FLOAT", {
            "default": 5.0, "min": 3.5, "max": 5.75, "step": 0.25,
            "tooltip": "Larger root hot set. DynamicVRAM still respects the runtime reserve, so this is a retention ceiling rather than forced occupancy."
        })
        req["mlp_primary_fraction"] = ("FLOAT", {
            "default": 0.68, "min": 0.68, "max": 0.72, "step": 0.01,
            "tooltip": "Validated safe sidecar band. This release studies the root critical path, not aggressive helper harvesting."
        })
        req["single_root_trim_interval"] = ("INT", {
            "default": 20, "min": 16, "max": 40, "step": 1,
            "tooltip": "Rare cache-maintenance points are the only places Dev15.1 permits a device-wide GPU0 barrier."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_reserve_gb=5.25, secondary_runtime_reserve_gb=2.5,
             primary_hot_cache_gb=5.0, secondary_hot_cache_gb=4.75,
             expected_steps=4, one_ahead_prefetch=True, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=16,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             mlp_primary_fraction=0.68, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, enable_token_mlp=True, single_root_trim_interval=20,
             sidecar_mlp_blocks=20, primary_stall_budget_ms=1.0,
             hard_cleanup_after_sample=False, telemetry=True):
        print(
            f"[H3VM DEV15.1 RESOURCE-CONSTRAINED REDLINE NODE] GPU0=critical | pipeline=2 | root_hot={float(primary_hot_cache_gb):.2f}GiB | "
            f"resident_sidecar={int(sidecar_mlp_blocks)} @ {float(mlp_primary_fraction)*100:.0f}/{(1.0-float(mlp_primary_fraction))*100:.0f} | "
            f"retire=EVENT-ONLY | device_barrier=TRIM-ONLY",
            flush=True,
        )
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device),
            stripe_size=50,
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb),
            secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch),
            trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length), attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap),
            attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
            mlp_token_parallel=bool(enable_token_mlp), mlp_primary_fraction=float(mlp_primary_fraction),
            mlp_min_sequence_length=int(mlp_min_sequence_length),
            mlp_helper_safety_mb=int(mlp_helper_safety_mb),
            single_root=True, single_root_trim_interval=int(single_root_trim_interval),
            critical_path_mode=True, sidecar_mlp_blocks=int(sidecar_mlp_blocks), sidecar_start_block=5,
            primary_stall_budget_ms=float(primary_stall_budget_ms),
            critical_path_adaptive_slack=True, critical_path_min_primary_fraction=0.68,
            critical_path_max_primary_fraction=0.72, critical_path_fraction_step=0.02,
            critical_path_target_slack_ms=6.0, critical_path_resident_sidecar=True,
            critical_path_pipeline_window=2, critical_path_harvest_confirmations=3,
            attention_primary_first=True,
        ),)


if H3VM_SHOW_LAB_NODES:
    print("[H3VM DEV15.1 LOADED] Resource-Constrained REDLINE | event-only retirement | trim-only device barriers", flush=True)


class H3VMDev152RollingRedlineTurboLoader(H3VMDev151ResourceConstrainedRedlineTurboLoader):
    """Dev15.2: true rolling critical-path retirement instead of batch drains."""
    CATEGORY = "MiniMaxH3/H3VM/Dev15"
    DESCRIPTION = (
        "Dev15.2 keeps a three-block rolling CPM window. When the CPU gets too far ahead it retires only the oldest "
        "critical-stream event, never the newest block in the batch. GPU0 keeps the validated 5GiB root hot layer and "
        "the GPU1 resident sidecar stays at the safe 68/32 split."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        req["primary_hot_cache_gb"] = ("FLOAT", {
            "default": 5.0, "min": 4.5, "max": 5.5, "step": 0.25,
            "tooltip": "Keep Dev15.1's proven root hot layer; Dev15.2 studies retirement topology, not cache size."
        })
        req["mlp_primary_fraction"] = ("FLOAT", {
            "default": 0.68, "min": 0.68, "max": 0.72, "step": 0.01,
            "tooltip": "Conservative resident-sidecar split. No deeper harvesting in this experiment."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_reserve_gb=5.25, secondary_runtime_reserve_gb=2.5,
             primary_hot_cache_gb=5.0, secondary_hot_cache_gb=4.75,
             expected_steps=4, one_ahead_prefetch=True, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=16,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             mlp_primary_fraction=0.68, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, enable_token_mlp=True, single_root_trim_interval=20,
             sidecar_mlp_blocks=20, primary_stall_budget_ms=1.0,
             hard_cleanup_after_sample=False, telemetry=True):
        print(
            f"[H3VM DEV15.2 ROLLING-REDLINE NODE] GPU0=critical | rolling_window=3 | root_hot={float(primary_hot_cache_gb):.2f}GiB | "
            f"resident_sidecar={int(sidecar_mlp_blocks)} @ {float(mlp_primary_fraction)*100:.0f}/{(1.0-float(mlp_primary_fraction))*100:.0f} | "
            f"retire=OLDEST-EVENT | batch_drain=OFF | device_barrier=TRIM-ONLY",
            flush=True,
        )
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device),
            stripe_size=50,
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb),
            secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch),
            trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length), attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap),
            attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
            mlp_token_parallel=bool(enable_token_mlp), mlp_primary_fraction=float(mlp_primary_fraction),
            mlp_min_sequence_length=int(mlp_min_sequence_length),
            mlp_helper_safety_mb=int(mlp_helper_safety_mb),
            single_root=True, single_root_trim_interval=int(single_root_trim_interval),
            critical_path_mode=True, sidecar_mlp_blocks=int(sidecar_mlp_blocks), sidecar_start_block=5,
            primary_stall_budget_ms=float(primary_stall_budget_ms),
            critical_path_adaptive_slack=True, critical_path_min_primary_fraction=0.68,
            critical_path_max_primary_fraction=0.72, critical_path_fraction_step=0.02,
            critical_path_target_slack_ms=6.0, critical_path_resident_sidecar=True,
            critical_path_pipeline_window=3, critical_path_harvest_confirmations=3,
            critical_path_rolling_retire=True, attention_primary_first=True,
        ),)


class H3VMDev160PersistentWorkpoolLabTurboLoader(H3VMDev152RollingRedlineTurboLoader):
    """Dev16.0: one Queue run profiles, expands, harvests and locks GPU1 work."""
    CATEGORY = "MiniMaxH3/H3VM/Dev16"
    DESCRIPTION = (
        "Dev16.0 one-shot persistent workpool laboratory. It starts from the proven Dev15.2 rolling red line, "
        "preallocates a bounded two-GPU relay ring on the first real H3 sequence, preassigns all sidecar MLP work "
        "for each denoise step, and uses bounded runtime credit to feed GPU1 only while measured slack remains positive. "
        "A normal 4-step Turbo generation automatically runs PROFILE -> EXPAND -> HARVEST -> LOCK and prints one JSON final report."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        req["workpool_ring_slots"] = ("INT", {
            "default": 2, "min": 1, "max": 3, "step": 1,
            "tooltip": "Persistent relay slots. 2 is the intended 16G+8G test point; ring misses fall back instead of waiting."
        })
        req["workpool_min_root_fraction"] = ("FLOAT", {
            "default": 0.64, "min": 0.62, "max": 0.68, "step": 0.02,
            "tooltip": "Maximum harvest limit. The one-shot matrix only approaches it after positive measured slack."
        })
        req["workpool_deadline_margin_ms"] = ("FLOAT", {
            "default": 4.0, "min": 0.0, "max": 12.0, "step": 1.0,
            "tooltip": "GPU1 must preserve this much predicted merge slack before CPU gives it another 2% token ticket."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_reserve_gb=5.25, secondary_runtime_reserve_gb=2.5,
             primary_hot_cache_gb=5.0, secondary_hot_cache_gb=4.75,
             expected_steps=4, one_ahead_prefetch=True, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=16,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             hard_cleanup_after_sample=False, telemetry=True,
             mlp_primary_fraction=0.68, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, enable_token_mlp=True, single_root_trim_interval=20,
             sidecar_mlp_blocks=20, primary_stall_budget_ms=1.0,
             workpool_ring_slots=2, workpool_min_root_fraction=0.64,
             workpool_deadline_margin_ms=4.0):
        print(
            f"[H3VM DEV16.0 ONE-SHOT WORKPOOL NODE] GPU0=critical NONSTOP | GPU1=resident workpool | "
            f"matrix=PROFILE>EXPAND>HARVEST>LOCK | ring={int(workpool_ring_slots)} | "
            f"root_range={float(workpool_min_root_fraction):.2f}..0.72 | margin={float(workpool_deadline_margin_ms):.1f}ms",
            flush=True,
        )
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device),
            stripe_size=50,
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb),
            secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch),
            trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length), attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap),
            attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
            mlp_token_parallel=bool(enable_token_mlp), mlp_primary_fraction=float(mlp_primary_fraction),
            mlp_min_sequence_length=int(mlp_min_sequence_length),
            mlp_helper_safety_mb=int(mlp_helper_safety_mb),
            single_root=True, single_root_trim_interval=int(single_root_trim_interval),
            critical_path_mode=True, sidecar_mlp_blocks=int(sidecar_mlp_blocks), sidecar_start_block=5,
            primary_stall_budget_ms=float(primary_stall_budget_ms),
            critical_path_adaptive_slack=False,
            critical_path_min_primary_fraction=float(workpool_min_root_fraction),
            critical_path_max_primary_fraction=0.72, critical_path_fraction_step=0.02,
            critical_path_target_slack_ms=6.0, critical_path_resident_sidecar=True,
            critical_path_pipeline_window=3, critical_path_harvest_confirmations=1,
            critical_path_rolling_retire=True, attention_primary_first=True,
            critical_path_persistent_workpool=True,
            workpool_ring_slots=int(workpool_ring_slots),
            workpool_deadline_margin_ms=float(workpool_deadline_margin_ms),
            workpool_one_shot_matrix=True,
        ),)


class H3VMDev161LocalTicketQueueTurboLoader(H3VMDev160PersistentWorkpoolLabTurboLoader):
    """Dev16.1: one-in/one-out helper packet, 1/2/3/AUTO GPU1-local ticket matrix."""
    CATEGORY = "MiniMaxH3/H3VM/Dev16"
    DESCRIPTION = (
        "Dev16.1 one-shot local ticket queue. Cross-GPU traffic is fixed: one helper token slice goes to GPU1 and one final hidden slice returns. "
        "Inside GPU1 only, the same exact MLP is tested as 1, 2, 3 balanced local tickets, then AUTO replays the best per block. "
        "This tests the observed helper 30ms/52ms kernel cliff without adding inter-GPU communication."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        # Dev16.1 isolates local compute grain; no fraction-harvest knob is exposed.
        req.pop("workpool_min_root_fraction", None)
        req.pop("workpool_deadline_margin_ms", None)
        req["workpool_ring_slots"] = ("INT", {
            "default": 1, "min": 1, "max": 2, "step": 1,
            "tooltip": "One persistent slot is enough for the serial block dependency; a miss falls back and never waits GPU0."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_reserve_gb=5.25, secondary_runtime_reserve_gb=2.5,
             primary_hot_cache_gb=5.0, secondary_hot_cache_gb=4.75,
             expected_steps=4, one_ahead_prefetch=True, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=16,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             hard_cleanup_after_sample=False, telemetry=True,
             mlp_primary_fraction=0.68, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, enable_token_mlp=True, single_root_trim_interval=20,
             sidecar_mlp_blocks=20, primary_stall_budget_ms=1.0, workpool_ring_slots=1):
        print(
            f"[H3VM DEV16.1 LOCAL-TICKET NODE] GPU0=critical NONSTOP | GPU1=ONE-IN/LOCAL-TICKETS/ONE-OUT | "
            f"matrix=MONO>DUAL>TRIPLE>AUTO | root=0.68 helper=0.32 | ring={int(workpool_ring_slots)}",
            flush=True,
        )
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device), stripe_size=50,
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb), secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch), trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length), attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap), attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
            mlp_token_parallel=bool(enable_token_mlp), mlp_primary_fraction=0.68,
            mlp_min_sequence_length=int(mlp_min_sequence_length), mlp_helper_safety_mb=int(mlp_helper_safety_mb),
            single_root=True, single_root_trim_interval=int(single_root_trim_interval),
            critical_path_mode=True, sidecar_mlp_blocks=int(sidecar_mlp_blocks), sidecar_start_block=5,
            primary_stall_budget_ms=float(primary_stall_budget_ms), critical_path_adaptive_slack=False,
            critical_path_min_primary_fraction=0.68, critical_path_max_primary_fraction=0.68,
            critical_path_fraction_step=0.02, critical_path_target_slack_ms=6.0,
            critical_path_resident_sidecar=True, critical_path_pipeline_window=3,
            critical_path_harvest_confirmations=1, critical_path_rolling_retire=True, attention_primary_first=True,
            critical_path_persistent_workpool=True, workpool_ring_slots=int(workpool_ring_slots),
            workpool_deadline_margin_ms=4.0, workpool_one_shot_matrix=False, workpool_local_ticket_matrix=True,
        ),)


class H3VMDev162LoadShiftMatrixTurboLoader(H3VMDev161LocalTicketQueueTurboLoader):
    """Dev16.2: DUAL-safe local compute plus aggressive per-block load shift."""
    CATEGORY = "MiniMaxH3/H3VM/Dev16"
    DESCRIPTION = (
        "Dev16.2 one-shot load-shift matrix. Keeps one stage-in / one return-out and only the numerically-safe DUAL local ticket path. "
        "The four Turbo steps test 68/32, 66/34, slack-driven 60..68 root shares, then lock the fastest safe split per block."
    )

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_reserve_gb=5.25, secondary_runtime_reserve_gb=2.5,
             primary_hot_cache_gb=5.0, secondary_hot_cache_gb=4.75,
             expected_steps=4, one_ahead_prefetch=True, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=16,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             hard_cleanup_after_sample=False, telemetry=True,
             mlp_primary_fraction=0.68, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, enable_token_mlp=True, single_root_trim_interval=20,
             sidecar_mlp_blocks=20, primary_stall_budget_ms=1.0, workpool_ring_slots=1):
        print(
            f"[H3VM DEV16.2 LOAD-SHIFT NODE] GPU0=critical NONSTOP | GPU1=DUAL local safe | "
            f"matrix=68/32>66/34>SLACK-FEED(60..68)>LOCK | ring={int(workpool_ring_slots)}",
            flush=True,
        )
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name), turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device), stripe_size=50,
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb), secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb), secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True, one_ahead_prefetch=bool(one_ahead_prefetch),
            trim_on_stripe_boundary=bool(trim_on_stripe_boundary), hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance), attention_min_sequence_length=int(attention_min_sequence_length),
            attention_relay_min_gbps=0.0, attention_helper_head_cap=int(attention_helper_head_cap), attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb), mlp_token_parallel=bool(enable_token_mlp), mlp_primary_fraction=0.68,
            mlp_min_sequence_length=int(mlp_min_sequence_length), mlp_helper_safety_mb=int(mlp_helper_safety_mb), single_root=True,
            single_root_trim_interval=int(single_root_trim_interval), critical_path_mode=True, sidecar_mlp_blocks=int(sidecar_mlp_blocks), sidecar_start_block=5,
            primary_stall_budget_ms=float(primary_stall_budget_ms), critical_path_adaptive_slack=False, critical_path_min_primary_fraction=0.60,
            critical_path_max_primary_fraction=0.68, critical_path_fraction_step=0.02, critical_path_target_slack_ms=6.0,
            critical_path_resident_sidecar=True, critical_path_pipeline_window=3, critical_path_harvest_confirmations=1,
            critical_path_rolling_retire=True, attention_primary_first=True, critical_path_persistent_workpool=True,
            workpool_ring_slots=int(workpool_ring_slots), workpool_deadline_margin_ms=4.0, workpool_one_shot_matrix=False,
            workpool_local_ticket_matrix=False, workpool_load_shift_matrix=True,
        ),)


class H3VMDev17RollingMLPStressTurboLoader(H3VMDev152RollingRedlineTurboLoader):
    """Dev17: all-50 helper universe with a 20/30/40/50 rolling coverage matrix."""
    CATEGORY = "MiniMaxH3/H3VM/Dev17"
    DESCRIPTION = (
        "Dev17 one-shot rolling MLP stress test. GPU0 remains the single root. GPU1 owns no whole block and keeps only a bounded "
        "DynamicVRAM helper hot set. Four Turbo steps test 20, 30, 40 and 50 helper-covered blocks at a fixed 68/32 token split. "
        "A helper prefetch queue streams future FC1/FC2 weights from RAM to GPU1 and reports exposed helper-prefetch host wait."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        req["secondary_hot_cache_gb"] = ("FLOAT", {
            "default": 3.5, "min": 2.5, "max": 4.5, "step": 0.25,
            "tooltip": "Rolling GPU1 helper-weight hot set. 3.5GiB is intentionally below the old 4.31GiB resident packet so new helper blocks must stream."
        })
        req["mlp_primary_fraction"] = ("FLOAT", {
            "default": 0.68, "min": 0.68, "max": 0.68, "step": 0.01,
            "tooltip": "Fixed split for the stress matrix. Dev17 measures coverage only: 20 > 30 > 40 > 50 blocks."
        })
        req["sidecar_mlp_blocks"] = ("INT", {
            "default": 50, "min": 50, "max": 50, "step": 1,
            "tooltip": "CPU-backed helper universe. GPU1 does NOT resident all 50; DynamicVRAM keeps a rolling bounded hot set."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_reserve_gb=5.25, secondary_runtime_reserve_gb=2.5,
             primary_hot_cache_gb=5.0, secondary_hot_cache_gb=3.5,
             expected_steps=4, one_ahead_prefetch=True, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=16,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             mlp_primary_fraction=0.68, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, enable_token_mlp=True, single_root_trim_interval=20,
             sidecar_mlp_blocks=50, primary_stall_budget_ms=1.0,
             hard_cleanup_after_sample=False, telemetry=True):
        print(
            f"[H3VM DEV17 ROLLING-MLP STRESS NODE] GPU0=critical NONSTOP | helper_universe=50 | "
            f"coverage=20>30>40>50 | split=68/32 | helper_hot={float(secondary_hot_cache_gb):.2f}GiB | resident_packet=OFF",
            flush=True,
        )
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device), stripe_size=50,
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb), secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch), trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length), attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap), attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
            mlp_token_parallel=bool(enable_token_mlp), mlp_primary_fraction=0.68,
            mlp_min_sequence_length=int(mlp_min_sequence_length), mlp_helper_safety_mb=int(mlp_helper_safety_mb),
            single_root=True, single_root_trim_interval=int(single_root_trim_interval),
            critical_path_mode=True, sidecar_mlp_blocks=50, sidecar_start_block=0,
            primary_stall_budget_ms=float(primary_stall_budget_ms),
            critical_path_adaptive_slack=True, critical_path_min_primary_fraction=0.68,
            critical_path_max_primary_fraction=0.72, critical_path_fraction_step=0.02,
            critical_path_target_slack_ms=9999.0, critical_path_resident_sidecar=False,
            critical_path_pipeline_window=3, critical_path_harvest_confirmations=99,
            critical_path_rolling_retire=True, attention_primary_first=True,
            critical_path_rolling_helper_matrix=True,
        ),)


class H3VMDev171RollingAdaptiveLoadTurboLoader(H3VMDev17RollingMLPStressTurboLoader):
    """Dev17.1: all-50 rolling helper + cross-step preassigned adaptive load shift."""
    CATEGORY = "MiniMaxH3/H3VM/Dev17"
    DESCRIPTION = (
        "Dev17.1 one-shot load-shift matrix on the proven all-50 rolling MLP helper. Coverage stays 50/50 for all four Turbo steps. "
        "Step1 calibrates at 68/32, Step2 preassigns 66/34, Step3 preassigns per-block 60..68% root fractions from completed Step2 slack, "
        "and Step4 locks the fastest safe tested fraction for every block. No same-step CPU steering and no resident 50-block packet."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        req["mlp_primary_fraction"] = ("FLOAT", {
            "default": 0.68, "min": 0.68, "max": 0.68, "step": 0.01,
            "tooltip": "Step1 calibration root fraction. Later steps are preassigned automatically from prior-step telemetry."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_reserve_gb=5.25, secondary_runtime_reserve_gb=2.5,
             primary_hot_cache_gb=5.0, secondary_hot_cache_gb=3.5,
             expected_steps=4, one_ahead_prefetch=True, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=16,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             mlp_primary_fraction=0.68, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, enable_token_mlp=True, single_root_trim_interval=20,
             sidecar_mlp_blocks=50, primary_stall_budget_ms=1.0,
             hard_cleanup_after_sample=False, telemetry=True):
        print(
            f"[H3VM DEV17.1 ROLLING-ADAPTIVE NODE] GPU0=critical NONSTOP | coverage=50/50 ALL STEPS | "
            f"matrix=68/32>66/34>PREASSIGNED-FEED>LOCK | helper_hot={float(secondary_hot_cache_gb):.2f}GiB | no-same-step-steering",
            flush=True,
        )
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device), stripe_size=50,
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb), secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch), trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length), attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap), attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
            mlp_token_parallel=bool(enable_token_mlp), mlp_primary_fraction=0.68,
            mlp_min_sequence_length=int(mlp_min_sequence_length), mlp_helper_safety_mb=int(mlp_helper_safety_mb),
            single_root=True, single_root_trim_interval=int(single_root_trim_interval),
            critical_path_mode=True, sidecar_mlp_blocks=50, sidecar_start_block=0,
            primary_stall_budget_ms=float(primary_stall_budget_ms),
            critical_path_adaptive_slack=True, critical_path_min_primary_fraction=0.60,
            critical_path_max_primary_fraction=0.72, critical_path_fraction_step=0.02,
            critical_path_target_slack_ms=9999.0, critical_path_resident_sidecar=False,
            critical_path_pipeline_window=3, critical_path_harvest_confirmations=99,
            critical_path_rolling_retire=True, attention_primary_first=True,
            critical_path_rolling_adaptive_load=True,
        ),)



class H3VMModeSelector:
    """One switch shared by DiT placement and Video-VAE placement."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["SINGLE_GPU", "DUAL_QUIET", "DUAL_CAPACITY", "DUAL_SYNC_ACCEL"], {"default": "DUAL_QUIET"}),
            }
        }

    RETURN_TYPES = ("H3VM_MODE",)
    RETURN_NAMES = ("mode",)
    FUNCTION = "select"
    CATEGORY = "MiniMaxH3/H3VM/Capacity"
    DESCRIPTION = (
        "SINGLE_GPU = GPU0-only DiT/VAE baseline; DUAL_QUIET = frozen Dev18 dual-compute + Dual VAE; "
        "DUAL_CAPACITY = activation-first pooled VRAM mode for resolutions that OOM on a 16GB card."
    )

    def select(self, mode="DUAL_QUIET"):
        return (str(mode),)


class H3VMCapacityModeTurboLoader:
    """Unified H3 loader with Single / Dual Quiet / Dual Capacity placement."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        from .h3vm.master_console import CAPACITY_VRAM_PROFILES
        return {
            "required": {
                "mode": ("H3VM_MODE",),
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "low_vram": ("BOOLEAN", {"default": False}),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "capacity_vram_profile": (CAPACITY_VRAM_PROFILES, {
                    "default": "SAFE｜保守·最稳",
                    "tooltip": "仅双卡扩显存模式生效。技术参数会打印到控制台日志。"
                }),
                "capacity_mlp_chunk_rows": ("INT", {
                    "default": 4096, "min": 512, "max": 16384, "step": 512,
                    "tooltip": "Only used by DUAL_CAPACITY. Bounds the large SwiGLU/FFN intermediate."
                }),
                "capacity_outproj_chunk_rows": ("INT", {
                    "default": 4096, "min": 512, "max": 16384, "step": 512,
                    "tooltip": "Only used by DUAL_CAPACITY. Bounds attention all-head concat/OutProj workspace."
                }),
                "capacity_helper_heads": ("INT", {
                    "default": 16, "min": 4, "max": 24, "step": 1,
                    "tooltip": "Only used by DUAL_CAPACITY. H3 has 56 heads; default gives GPU0/GPU1 = 40/16."
                }),
                "capacity_attention_kernel": (["INT8_CURRENT", "OFFICIAL_OPTIMIZED"], {
                    "default": "INT8_CURRENT",
                    "tooltip": "Audio A/B lab switch. OFFICIAL_OPTIMIZED uses ComfyUI optimized_attention (Sage when --use-sage-attention)."
                }),
                "telemetry": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM/Capacity"
    DESCRIPTION = (
        "Unified exact H3 placement switch. DUAL_CAPACITY sacrifices speed for peak-memory reduction: "
        "QKV projection is head-sharded before materialization, post-attention MLP is token-sharded across GPUs, "
        "and each MLP/OutProj side is micro-chunked. FixedSlot and the rejected Dev18.2 relay pipeline are not used."
    )

    def load(self, mode, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1", capacity_vram_profile="SAFE｜保守·最稳",
             capacity_mlp_chunk_rows=4096, capacity_outproj_chunk_rows=4096,
             capacity_helper_heads=16, capacity_attention_kernel="INT8_CURRENT", telemetry=True, expected_steps=4):
        from .h3vm.loader import build_h3_streaming_exact_turbo

        mode = str(mode)
        common = dict(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device),
            stripe_size=50, expected_steps=int(expected_steps), safe_profile=True,
            hard_cleanup_after_sample=False, telemetry=bool(telemetry),
        )

        if mode == "SINGLE_GPU":
            print(
                "[H3VM MODE] SINGLE_GPU | GPU0-only exact DiT | GPU1 compute disabled | VAE mode follows shared switch",
                flush=True,
            )
            return (build_h3_streaming_exact_turbo(
                **common,
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
            ),)

        if mode == "DUAL_QUIET":
            print(
                "[H3VM MODE] DUAL_QUIET | frozen Dev18 cooperative compute | 50-block 68/32 + host attention + Dual VAE",
                flush=True,
            )
            return (build_h3_streaming_exact_turbo(
                **common,
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
            ),)

        if mode != "DUAL_CAPACITY":
            raise ValueError(f"Unknown H3VM mode: {mode}")

        # Product-facing VRAM tiers scale to the detected CUDA card sizes and are
        # clamped to the same legality rules enforced by the streaming loader.
        import torch
        from .h3vm.master_console import resolve_capacity_vram_profile
        pdev = torch.device("cuda:0" if str(primary_device) == "gpu:0" else "cuda:1")
        sdev = torch.device("cuda:0" if str(secondary_device) == "gpu:0" else "cuda:1")
        pgib = float(torch.cuda.get_device_properties(pdev).total_memory) / float(1024 ** 3)
        sgib = float(torch.cuda.get_device_properties(sdev).total_memory) / float(1024 ** 3)
        vram_plan = resolve_capacity_vram_profile(str(capacity_vram_profile), pgib, sgib)

        profile_mlp_fraction = float(vram_plan["mlp_primary_fraction"])
        profile_helper_heads = int(vram_plan["capacity_helper_heads"])
        profile_trim_interval = int(vram_plan["single_root_trim_interval"])

        print(
            f"[H3VM MODE] DUAL_CAPACITY | profile={vram_plan['profile']} | "
            f"reserve={vram_plan['primary_runtime_reserve_gb']:.2f}/{vram_plan['secondary_runtime_reserve_gb']:.2f}GiB | "
            f"hot={vram_plan['primary_hot_cache_gb']:.2f}/{vram_plan['secondary_hot_cache_gb']:.2f}GiB | "
            f"MLP={profile_mlp_fraction*100:.0f}/{(1.0-profile_mlp_fraction)*100:.0f} | "
            f"QKV heads={56-profile_helper_heads}/{profile_helper_heads} | trim_every={profile_trim_interval} blocks | "
            f"MLP chunk={int(capacity_mlp_chunk_rows)} | OutProj chunk={int(capacity_outproj_chunk_rows)} | "
            f"attn={str(capacity_attention_kernel)}",
            flush=True,
        )
        return (build_h3_streaming_exact_turbo(
            **common,
            # Workspace-first: keep only a tiny rolling weight cache and reserve
            # most VRAM for sequence-sized activations.
            primary_runtime_reserve_gb=float(vram_plan["primary_runtime_reserve_gb"]),
            secondary_runtime_reserve_gb=float(vram_plan["secondary_runtime_reserve_gb"]),
            primary_hot_cache_gb=float(vram_plan["primary_hot_cache_gb"]),
            secondary_hot_cache_gb=float(vram_plan["secondary_hot_cache_gb"]),
            one_ahead_prefetch=False, trim_on_stripe_boundary=True,
            # Capacity fabric owns attention before full-QKV materialization.
            attention_mode="off", attention_head_balance="sm_weighted",
            attention_min_sequence_length=1, attention_relay_min_gbps=0.0,
            attention_helper_head_cap=profile_helper_heads, attention_helper_safety_mb=128,
            attention_host_ring_mb=64,
            mlp_token_parallel=True, mlp_primary_fraction=profile_mlp_fraction,
            mlp_min_sequence_length=1, mlp_helper_safety_mb=128,
            single_root=True, single_root_trim_interval=profile_trim_interval,
            critical_path_mode=True, sidecar_mlp_blocks=50, sidecar_start_block=0,
            primary_stall_budget_ms=1.0e9,
            critical_path_adaptive_slack=False,
            critical_path_min_primary_fraction=profile_mlp_fraction,
            critical_path_max_primary_fraction=profile_mlp_fraction,
            critical_path_fraction_step=0.01,
            critical_path_target_slack_ms=999999.0,
            critical_path_resident_sidecar=False,
            critical_path_pipeline_window=1,
            critical_path_harvest_confirmations=999,
            critical_path_rolling_retire=False,
            attention_primary_first=False,
            critical_path_post_attention_island=False,
            capacity_mode=True,
            capacity_mlp_chunk_rows=int(capacity_mlp_chunk_rows),
            capacity_outproj_chunk_rows=int(capacity_outproj_chunk_rows),
            capacity_helper_heads=profile_helper_heads,
            capacity_attention_kernel=str(capacity_attention_kernel),
        ),)


class H3VMMasterLoader:
    """One H3 cockpit: model + Turbo + placement + common generation controls."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        from .h3vm.master_console import CAPACITY_VRAM_PROFILES, MASTER_GPU_MODES
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "turbo_lora": (folder_paths.get_filename_list("loras"),),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "gpu_mode": (MASTER_GPU_MODES, {"default": "DUAL_QUIET｜双卡协同·日常推荐"}),
                "capacity_vram_profile": (CAPACITY_VRAM_PROFILES, {
                    "default": "SAFE｜保守·最稳",
                    "tooltip": "仅双卡扩显存模式生效。技术参数会打印到控制台日志。"
                }),
                "steps": (["4步｜极速", "6步｜平衡", "8步｜质量"], {"default": "4步｜极速"}),
                "duration_seconds": ("FLOAT", {"default": 15.0, "min": 0.1, "max": 600.0, "step": 0.1}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], {"default": "16:9"}),
                "resolution": (["0.2MP｜预览", "0.5MP｜快速", "0.7MP｜中高", "768P｜原生高清", "1.5MP｜超清实验", "2.0MP｜1080级实验", "CUSTOM｜自定义"], {"default": "768P｜原生高清"}),
                "custom_width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "custom_height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "telemetry": ("BOOLEAN", {"default": True}),
                # Keep Prompt last. On current ComfyUI frontend advanced widgets are
                # collapsed by default; the bundled frontend extension only enlarges
                # the textarea when the user expands it.
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "advanced": True,
                    "default": "",
                    "placeholder": "Prompt｜折叠时不占空间；展开后可写长 H3 提示词",
                    "tooltip": "Prompt is intentionally the last advanced widget. Expand it only when editing long H3 prompts."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "H3VM_MODE", "STRING", "INT", "INT", "INT", "INT", "INT", "SAMPLER")
    RETURN_NAMES = ("model", "mode", "prompt", "width", "height", "length", "steps", "seed", "sampler")
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/H3VM"
    DESCRIPTION = ""

    def load(self, unet_name, turbo_lora, lora_strength=1.0, gpu_mode="DUAL_QUIET",
             capacity_vram_profile="SAFE｜保守·最稳", steps="4步｜极速", duration_seconds=15.0,
             aspect_ratio="16:9", resolution="768P｜原生高清", custom_width=1344, custom_height=768,
             seed=0, telemetry=True, prompt=""):
        from .h3vm.master_console import (
            resolve_resolution, duration_to_length, resolve_steps, lora_family, mode_summary, apply_sigma_shift, normalize_mode
        )

        mode = normalize_mode(gpu_mode)
        if mode == "DUAL_SYNC_ACCEL":
            raise RuntimeError(
                "H3VM DUAL_SYNC_ACCEL is a reserved backend slot. Install/wire a compatible external "
                "same-model synchronous multi-GPU backend before selecting this mode. "
                "SINGLE_GPU / DUAL_QUIET / DUAL_CAPACITY are available now."
            )

        width, height = resolve_resolution(aspect_ratio, resolution, custom_width, custom_height)
        length = duration_to_length(float(duration_seconds))
        step_count, note = resolve_steps(str(turbo_lora), str(steps))
        family = lora_family(str(turbo_lora))
        if note:
            print(f"[H3VM MASTER] Turbo step note: {note}", flush=True)
        if mode == "DUAL_CAPACITY" and int(step_count) < 8:
            print(
                "[H3VM MASTER] Capacity audio note: 8-step is recommended. "
                "4-step produced noticeably worse audio in current 16G+8G testing; 6-step remains experimental.",
                flush=True,
            )

        print(
            f"[H3VM MASTER] mode={mode_summary(mode)} | lora={turbo_lora} family={family} | "
            f"steps={step_count} | duration={float(duration_seconds):.2f}s -> length={length} | "
            f"canvas={width}x{height} ({aspect_ratio}, {resolution}) | capacity_vram={capacity_vram_profile}",
            flush=True,
        )

        # Audio A/B established that INT8-current vs official optimized/Sage was
        # not the audible difference. Keep Capacity's proven V1 attention path in
        # the product node; the lab loader remains available for diagnostics.
        model = H3VMCapacityModeTurboLoader().load(
            mode=mode, unet_name=str(unet_name), lora_name=str(turbo_lora),
            strength=float(lora_strength), low_vram=False,
            primary_device="gpu:0", secondary_device="gpu:1",
            capacity_vram_profile=str(capacity_vram_profile),
            capacity_mlp_chunk_rows=4096, capacity_outproj_chunk_rows=4096,
            capacity_helper_heads=16, capacity_attention_kernel="INT8_CURRENT",
            telemetry=bool(telemetry), expected_steps=int(step_count),
        )[0]

        # Both supported Turbo families use 12/3 flow shifts, but their published
        # sampler contracts differ. LightX2V uses stock Euler; Larry keeps its
        # custom dual-clock sampler.
        model = apply_sigma_shift(model, 12.0, 3.0)
        import comfy.samplers
        if family.startswith("LIGHTX2V_"):
            sampler = comfy.samplers.sampler_object("euler")
            sampler_kind = "euler+ModelSamplingAV(12/3)"
        else:
            from .h3vm.turbo_compat import _load_larry_module
            larry = _load_larry_module()
            if not hasattr(larry, "_turbo_sampler"):
                raise RuntimeError("Installed ComfyUI-MiniMax-H3-Turbo lacks _turbo_sampler; please update it.")
            sampler = comfy.samplers.KSAMPLER(larry._turbo_sampler)
            sampler_kind = "larry_dual_clock(12/3)"
        print(f"[H3VM MASTER] sampler={sampler_kind}", flush=True)

        return (
            model, mode, str(prompt), int(width), int(height), int(length),
            int(step_count), int(seed), sampler,
        )


class H3VMModeAwareVideoVAEDecode:
    """Use the same H3VM mode switch for the Video-VAE phase."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "mode": ("H3VM_MODE",),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "primary_device": (["gpu:0", "gpu:1"], {"default": "gpu:0"}),
                "secondary_device": (["gpu:1", "gpu:0"], {"default": "gpu:1"}),
                "cleanup_h3": ("BOOLEAN", {"default": True}),
                "telemetry": ("BOOLEAN", {"default": True}),
                "post_cleanup": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "MiniMaxH3/H3VM/Capacity"
    DESCRIPTION = "SINGLE_GPU decodes on GPU0 only; both dual modes retain the proven Dev10 temporal-chunk Dual VAE."

    def decode(self, samples, vae, mode, vae_name, primary_device="gpu:0", secondary_device="gpu:1",
               cleanup_h3=True, telemetry=True, post_cleanup=True):
        mode = str(mode)
        if mode != "SINGLE_GPU":
            from .h3vm.dual_video_vae import dual_decode_h3_video
            return (dual_decode_h3_video(
                vae=vae, samples=samples, vae_name=str(vae_name),
                primary_device=str(primary_device), secondary_device=str(secondary_device),
                cleanup_h3=bool(cleanup_h3), telemetry=bool(telemetry),
                post_cleanup=bool(post_cleanup),
            ),)

        from .h3vm.dual_video_vae import _cleanup_h3_runtime, _post_cleanup_dual_vae
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        if bool(cleanup_h3):
            _cleanup_h3_runtime()
        images = vae.decode(latent)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        if bool(post_cleanup):
            _post_cleanup_dual_vae(drop_secondary_cache=True)
        print("[H3VM VAE MODE] SINGLE_GPU | Video VAE decoded on primary only", flush=True)
        return (images,)


print("[H3VM v0.20.0-rc1] Multi-GPU Loader ready | Single / Dual Quiet / Dual Capacity", flush=True)

class H3VMDev18PostAttentionIslandTurboLoader(H3VMDev17RollingMLPStressTurboLoader):
    """Dev18: all-50 rolling helper + larger post-attention row compute island."""
    CATEGORY = "MiniMaxH3/H3VM/Dev18"
    DESCRIPTION = (
        "Dev18 one-shot A/B test on the proven 50-block rolling MLP base. Step1 warms the 68/32 MLP-only path, Step2 measures a steady "
        "MLP-only baseline, and Steps3-4 move Norm2 + AdaLN MLP modulation + FC1/SwiGLU/FC2 + gate/residual into the same GPU1 row island. "
        "GPU0 keeps the complete attention spine. Cross-GPU row-state traffic remains one sequence-sized input and one output per helper slice."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        req = dict(spec.get("required", {}))
        req["mlp_primary_fraction"] = ("FLOAT", {
            "default": 0.68, "min": 0.68, "max": 0.68, "step": 0.01,
            "tooltip": "Fixed Dev17-proven 68/32 split. Dev18 changes work depth, not token share."
        })
        req["sidecar_mlp_blocks"] = ("INT", {
            "default": 50, "min": 50, "max": 50, "step": 1,
            "tooltip": "All 50 blocks use the rolling GPU1 helper universe."
        })
        spec = dict(spec); spec["required"] = req
        return spec

    def load(self, unet_name, lora_name, strength=1.0, low_vram=False,
             primary_device="gpu:0", secondary_device="gpu:1",
             primary_runtime_reserve_gb=5.25, secondary_runtime_reserve_gb=2.5,
             primary_hot_cache_gb=5.0, secondary_hot_cache_gb=3.5,
             expected_steps=4, one_ahead_prefetch=True, trim_on_stripe_boundary=True,
             attention_mode="force_host", attention_head_balance="sm_weighted",
             attention_min_sequence_length=8192, attention_helper_head_cap=16,
             attention_helper_safety_mb=1024, attention_host_ring_mb=64,
             mlp_primary_fraction=0.68, mlp_min_sequence_length=8192,
             mlp_helper_safety_mb=768, enable_token_mlp=True, single_root_trim_interval=20,
             sidecar_mlp_blocks=50, primary_stall_budget_ms=1.0,
             hard_cleanup_after_sample=False, telemetry=True):
        print(
            f"[H3VM DEV18 POST-ATTN ISLAND NODE] GPU0=attention critical spine | GPU1=rolling row island | "
            f"coverage=50/50 split=68/32 | matrix=WARMUP-MLP>BASELINE-MLP>POST-ISLAND>POST-ISLAND | "
            f"helper_hot={float(secondary_hot_cache_gb):.2f}GiB",
            flush=True,
        )
        from .h3vm.loader import build_h3_streaming_exact_turbo
        return (build_h3_streaming_exact_turbo(
            unet_name=str(unet_name), turbo_lora_name=str(lora_name),
            turbo_strength=float(strength), turbo_low_vram=bool(low_vram),
            primary_device=str(primary_device), secondary_device=str(secondary_device), stripe_size=50,
            primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
            secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
            primary_hot_cache_gb=float(primary_hot_cache_gb), secondary_hot_cache_gb=float(secondary_hot_cache_gb),
            expected_steps=int(expected_steps), safe_profile=True,
            one_ahead_prefetch=bool(one_ahead_prefetch), trim_on_stripe_boundary=bool(trim_on_stripe_boundary),
            hard_cleanup_after_sample=bool(hard_cleanup_after_sample), telemetry=bool(telemetry),
            attention_mode="force_host", attention_head_balance=str(attention_head_balance),
            attention_min_sequence_length=int(attention_min_sequence_length), attention_relay_min_gbps=0.0,
            attention_helper_head_cap=int(attention_helper_head_cap), attention_helper_safety_mb=int(attention_helper_safety_mb),
            attention_host_ring_mb=int(attention_host_ring_mb),
            mlp_token_parallel=bool(enable_token_mlp), mlp_primary_fraction=0.68,
            mlp_min_sequence_length=int(mlp_min_sequence_length), mlp_helper_safety_mb=int(mlp_helper_safety_mb),
            single_root=True, single_root_trim_interval=int(single_root_trim_interval),
            critical_path_mode=True, sidecar_mlp_blocks=50, sidecar_start_block=0,
            primary_stall_budget_ms=float(primary_stall_budget_ms),
            critical_path_adaptive_slack=True, critical_path_min_primary_fraction=0.68,
            critical_path_max_primary_fraction=0.72, critical_path_fraction_step=0.02,
            critical_path_target_slack_ms=9999.0, critical_path_resident_sidecar=False,
            critical_path_pipeline_window=3, critical_path_harvest_confirmations=99,
            critical_path_rolling_retire=True, attention_primary_first=True,
            critical_path_post_attention_island=True,
        ),)


if H3VM_SHOW_LAB_NODES:
    print("[H3VM DEV18 LOADED] Post-Attention Row Island | MLP baseline > island A/B", flush=True)


if H3VM_SHOW_LAB_NODES:
    print("[H3VM DEV17.1 LOADED] Rolling 50-Block Adaptive Load | 32>34>FEED>LOCK", flush=True)


if H3VM_SHOW_LAB_NODES:
    print("[H3VM DEV17 LOADED] Rolling MLP Stress | 20>30>40>50 helper coverage", flush=True)


if H3VM_SHOW_LAB_NODES:
    print("[H3VM DEV16.2 LOADED] Load Shift Matrix | 68/32>66/34>FEED>LOCK", flush=True)

if H3VM_SHOW_LAB_NODES:
    print("[H3VM DEV16.1 LOADED] Local Ticket Queue | MONO>DUAL>TRIPLE>AUTO", flush=True)


if H3VM_SHOW_LAB_NODES:
    print("[H3VM DEV16.0 LOADED] Persistent Workpool | one-shot 4-step matrix", flush=True)


if H3VM_SHOW_LAB_NODES:
    print("[H3VM DEV15.2 LOADED] Rolling REDLINE | oldest-event retirement | 3-block CPM window", flush=True)


PUBLIC_NODE_CLASS_MAPPINGS = {
    "H3VMMasterLoader": H3VMMasterLoader,
    "H3VMModeAwareVideoVAEDecode": H3VMModeAwareVideoVAEDecode,
}
PUBLIC_NODE_DISPLAY_NAME_MAPPINGS = {
    "H3VMMasterLoader": "H3VM Multi-GPU Loader｜H3多卡加载器",
    "H3VMModeAwareVideoVAEDecode": "H3VM Video VAE Decode｜跟随显卡模式",
}

# Development/lab nodes remain in the codebase for research and backwards
# compatibility, but are hidden from normal users.  Developers can expose them
# before starting ComfyUI with: H3VM_SHOW_LAB_NODES=1
LAB_NODE_CLASS_MAPPINGS = {
    "H3VMModeSelector": H3VMModeSelector,
    "H3VMCapacityModeTurboLoader": H3VMCapacityModeTurboLoader,
    "H3VMDev18PostAttentionIslandTurboLoader": H3VMDev18PostAttentionIslandTurboLoader,
    "H3VMDev171RollingAdaptiveLoadTurboLoader": H3VMDev171RollingAdaptiveLoadTurboLoader,
    "H3VMDev17RollingMLPStressTurboLoader": H3VMDev17RollingMLPStressTurboLoader,
    "H3VMDev162LoadShiftMatrixTurboLoader": H3VMDev162LoadShiftMatrixTurboLoader,
    "H3VMDev161LocalTicketQueueTurboLoader": H3VMDev161LocalTicketQueueTurboLoader,
    "H3VMDev160PersistentWorkpoolLabTurboLoader": H3VMDev160PersistentWorkpoolLabTurboLoader,
    "H3VMDev152RollingRedlineTurboLoader": H3VMDev152RollingRedlineTurboLoader,
    "H3VMDev151ResourceConstrainedRedlineTurboLoader": H3VMDev151ResourceConstrainedRedlineTurboLoader,
    "H3VMDev150RedlinePipelineTurboLoader": H3VMDev150RedlinePipelineTurboLoader,
    "H3VMDev142CriticalPathCompressorTurboLoader": H3VMDev142CriticalPathCompressorTurboLoader,
    "H3VMDev141SlackHarvestTurboLoader": H3VMDev141SlackHarvestTurboLoader,
    "H3VMDev14CriticalPathTurboLoader": H3VMDev14CriticalPathTurboLoader,
    "H3VMDev131FedCoprocessorTurboLoader": H3VMDev131FedCoprocessorTurboLoader,
    "H3VMDev130HSingleRootNoEmptyVBARTurboLoader": H3VMDev130HSingleRootNoEmptyVBARTurboLoader,
    "H3VMDev13SingleRootCoprocessorTurboLoader": H3VMDev13SingleRootCoprocessorTurboLoader,
    "H3VMDev123HTokenMLPConnectedTurboLoader": H3VMDev123HTokenMLPConnectedTurboLoader,
    "H3VMDev123TokenMLPTurboLoader": H3VMDev123TokenMLPTurboLoader,
    "H3VMDev122HForceHostAttentionTurboLoader": H3VMDev122HForceHostAttentionTurboLoader,
    "H3VMPhysicalShardLoader": H3VMPhysicalShardLoader,
    "H3VMHybridShardLoader": H3VMHybridShardLoader,
    "H3VMRelayShardLoader": H3VMRelayShardLoader,
    "H3VMGlobalMemoryProbe": H3VMGlobalMemoryProbe,
    "H3VMGlobalMemoryLoader": H3VMGlobalMemoryLoader,
    "H3VMGlobalTensorFabricLoader": H3VMGlobalTensorFabricLoader,
    "H3VMSnapshotIslandsLoader": H3VMSnapshotIslandsLoader,
    "H3VMSnapshotIslandsFullThrottleLoader": H3VMSnapshotIslandsFullThrottleLoader,
    "H3VMSnapshotIslandsTurboLoader": H3VMSnapshotIslandsTurboLoader,
    "H3VMPrimaryFirstExactTurboLoader": H3VMPrimaryFirstExactTurboLoader,
    "H3VMStreamingExactTurboLoader": H3VMStreamingExactTurboLoader,
    "H3VMStreamingExactAttentionTurboLoader": H3VMStreamingExactAttentionTurboLoader,
    "H3VMStreamingExactHostAttentionTurboLoader": H3VMStreamingExactHostAttentionTurboLoader,
    "H3VMDualVideoVAEDecode": H3VMDualVideoVAEDecode,
}
LAB_NODE_DISPLAY_NAME_MAPPINGS = {'H3VMModeSelector': 'H3VM Mode [Single / Dual Quiet / Dual Capacity]',
 'H3VMCapacityModeTurboLoader': 'H3VM Capacity Loader [3 Modes]',
 'H3VMDev18PostAttentionIslandTurboLoader': 'H3 Dev18 Post-Attention Row Island [MLP A/B]',
 'H3VMDev171RollingAdaptiveLoadTurboLoader': 'H3 Dev17.1 Rolling 50 Adaptive Load [32>34>FEED>LOCK]',
 'H3VMDev17RollingMLPStressTurboLoader': 'H3 Dev17 Rolling MLP Stress [20>30>40>50]',
 'H3VMDev162LoadShiftMatrixTurboLoader': 'H3 Dev16.2 Load Shift [32>34>FEED>LOCK]',
 'H3VMDev161LocalTicketQueueTurboLoader': 'H3 Dev16.1 Local Ticket Queue [ONE-IN / N LOCAL / ONE-OUT]',
 'H3VMDev160PersistentWorkpoolLabTurboLoader': 'H3 Dev16.0 Persistent Workpool [ONE-SHOT LAB]',
 'H3VMDev152RollingRedlineTurboLoader': 'H3 Dev15.2 Rolling REDLINE [3-Block Oldest-Event]',
 'H3VMDev151ResourceConstrainedRedlineTurboLoader': 'H3 Dev15.1 Critical Path [RESOURCE-CONSTRAINED REDLINE]',
 'H3VMDev150RedlinePipelineTurboLoader': 'H3 Dev15 Critical Path [REDLINE 4-BLOCK PIPELINE]',
 'H3VMDev142CriticalPathCompressorTurboLoader': 'H3 Dev14.2 Critical Path [RESIDENT SIDECAR + COMPRESS]',
 'H3VMDev141SlackHarvestTurboLoader': 'H3 Dev14.1 Critical Path [SLACK HARVEST + FAST START]',
 'H3VMDev14CriticalPathTurboLoader': 'H3 Dev14 Critical Path Scheduler [PRIMARY NONSTOP]',
 'H3VMDev131FedCoprocessorTurboLoader': 'H3 Dev13.1 Single-Root [Fed 8G Coprocessor]',
 'H3VMDev130HSingleRootNoEmptyVBARTurboLoader': 'H3 Dev13.0H Single-Root + GPU Coprocessor [NO EMPTY VBAR]',
 'H3VMDev13SingleRootCoprocessorTurboLoader': 'H3 Dev13 Single-Root RAM Fabric + GPU Coprocessor [ZERO BOUNDARY]',
 'H3VMDev123HTokenMLPConnectedTurboLoader': 'H3 Dev12.3H RAM-First + Attention + Token MLP [CONNECTED]',
 'H3VMDev123TokenMLPTurboLoader': 'H3 Dev12.3 RAM-First + Attention + Token MLP [DUAL CUDA]',
 'H3VMDev122HForceHostAttentionTurboLoader': 'H3 Dev12.2H RAM-First + FORCE Host Attention [FAIL-CLOSED]',
 'H3VMPhysicalShardLoader': 'H3 Virtual Loader [Dev4 Fallback]',
 'H3VMHybridShardLoader': 'H3 Virtual Loader [Dev5 Hybrid Scheduler]',
 'H3VMRelayShardLoader': 'H3 Virtual Loader [Dev6 Relay Scheduler]',
 'H3VMGlobalMemoryProbe': 'H3VM Global Memory Probe [Dev7]',
 'H3VMGlobalMemoryLoader': 'H3 Global Memory Fabric [Dev7]',
 'H3VMGlobalTensorFabricLoader': 'H3 Global Tensor Fabric [Dev8.2]',
 'H3VMSnapshotIslandsLoader': 'H3 Snapshot Compute Islands [Dev9 LAB]',
 'H3VMSnapshotIslandsFullThrottleLoader': 'H3 Snapshot Islands [Dev9.3.1 SAFE FEEDER]',
 'H3VMSnapshotIslandsTurboLoader': 'H3 Turbo 4-Step Quality Matrix [Dev10.1 | 22/28]',
 'H3VMPrimaryFirstExactTurboLoader': 'H3 Dev11.0.5 Primary-First Exact Turbo [Lazy 8G Feed]',
 'H3VMStreamingExactTurboLoader': 'H3 Dev12 RAM-First Streaming Exact Turbo [Striped Hot Cache]',
 'H3VMStreamingExactAttentionTurboLoader': 'H3 Dev12.1 RAM-First + Exact Attention Parallel [Dual CUDA]',
 'H3VMStreamingExactHostAttentionTurboLoader': 'H3 Dev12.2 RAM-First + Host-Relay Attention [Dual CUDA]',
 'H3VMDualVideoVAEDecode': 'H3 Dual Video VAE Decode [Dev11 Cleanup]'}

NODE_CLASS_MAPPINGS = dict(PUBLIC_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(PUBLIC_NODE_DISPLAY_NAME_MAPPINGS)
if H3VM_SHOW_LAB_NODES:
    NODE_CLASS_MAPPINGS.update(LAB_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(LAB_NODE_DISPLAY_NAME_MAPPINGS)
    print(f"[H3VM] Lab nodes enabled: {len(LAB_NODE_CLASS_MAPPINGS)}", flush=True)
