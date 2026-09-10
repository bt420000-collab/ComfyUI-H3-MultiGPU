# H3VM Multi-GPU Loader for ComfyUI

[中文说明](README.md) | **English**

H3VM is a multi-GPU loading, execution, and VRAM scheduling layer for **MiniMax H3 in ComfyUI**.

The project is not tied to one fixed workflow. Its goal is to make H3 easier to run on heterogeneous multi-GPU systems by providing both an integrated standalone controller and a reusable `MODEL -> MODEL` execution engine that can be inserted into larger H3 pipelines.

> Current version: **v0.20.0-rc6**

## Two ways to use H3VM

### 1. H3VM Multi-GPU Loader

An integrated standalone controller for direct generation and quick A/B testing.

It keeps common generation controls in one node, including:

- H3 diffusion model
- Turbo LoRA and strength
- 4 / 6 / 8-step presets
- duration in seconds
- aspect ratio and common resolution presets
- seed
- GPU execution mode
- dual-GPU capacity profile
- prompt
- ordinary Style / Character LoRA stack

### 2. H3VM Core

A reusable execution engine for complex production workflows.

The public interface is intentionally narrow:

```text
MODEL
  |
  v
H3VM Core
  |
  v
MODEL
```

Your workflow remains responsible for model selection and generation logic, for example:

```text
Ref2VA / FL2VA
      |
      v
base-model routing
      |
      v
Turbo / SigmaShift
      |
      v
Style / Character LoRA(s)
      |
      v
H3VM Core
      |
      v
Guider / Scheduler / Sampler
```

H3VM Core is responsible only for GPU placement, multi-GPU execution, VRAM scheduling, runtime transport, and telemetry. It does **not** modify your prompt, seed, resolution, sampler, sigmas, or actual sampling step count.

The current Generic Core supports clean H3 `ModelPatcher` objects and ordinary weight patches such as common `LoraLoaderModelOnly` LoRAs. Runtime injections, unsupported object patches, hook patches, and weight-wrapper patches remain fail-closed so GPU0 and GPU1 cannot silently execute different model states.

## Author

- GitHub: `bt420000-collab`
- Bilibili: https://space.bilibili.com/16826253

## GPU modes

### SINGLE_GPU | Standard single-GPU mode

Uses the primary GPU only.

Recommended as the quality, performance, and VRAM baseline before testing multi-GPU modes.

### DUAL_QUIET | Cooperative dual-GPU mode

The current daily-use recommendation.

This mode keeps the mature cooperative H3 execution path together with Dual Video VAE support. Its main purpose is to share work between two GPUs, reduce sustained load on the primary card, and lower fan / thermal pressure. On the current development system it also provides a small wall-time improvement in some workloads.

This mode is based on the frozen, validated Dev18 cooperative execution path rather than the abandoned FixedSlot or relay-pipeline experiments.

### DUAL_CAPACITY | Dual-GPU VRAM capacity mode

Designed primarily for workloads that do not fit on one GPU.

The success criterion is not maximum GPU utilization. The goal is:

> a workload that OOMs on a single card can complete by using both GPUs plus host RAM as one execution and memory system.

The current implementation reduces per-GPU peak memory through techniques including:

- RAM backing
- QKV head sharding
- MLP token sharding
- micro-chunking
- dynamic model residency
- reduced hot-cache pressure

At the moment, **8 steps are recommended** for validation. On the current 16GB + 8GB test machine, 4-step Capacity runs have shown a significant audio-quality drop. INT8 Attention and the official optimized / SageAttention path produced similar audio quality in A/B testing, so the issue is not currently attributed to the Attention kernel itself.

### DUAL_SYNC_ACCEL | Stock H3 FullThrottle 20-step mode

Mode 4 exposes the historical Dev9.4 Stock H3 FullThrottle path.

In the standalone Master Loader this mode intentionally uses the stock H3 sampling contract:

- Stock H3 model behavior
- `res_multistep`
- 20 steps
- no Turbo SigmaShift
- Turbo acceleration LoRA is ignored
- ordinary Style / Character LoRAs remain supported
- 22 / 28 Snapshot Islands split
- 2GB RAM backing
- linear predictor 0.75

Important: “Stock H3” here refers to the **stock H3 weights and sampling contract**. The Dev9.4 FullThrottle executor itself uses stale / predicted boundary snapshots, so it is an experimental approximate dual-GPU execution path rather than strictly synchronous layer-by-layer exact execution.

When Mode 4 is used through the public H3VM Core engine, H3VM does not force the workflow to 20 steps. The workflow owns its actual sampler and step count; `expected_steps_hint` is only an execution hint for prefetching and end-of-run scheduling.

## Capacity VRAM profiles

The Capacity profile selector only affects `DUAL_CAPACITY`.

Available profiles:

- `SAFE` | conservative and most stable
- `RESIDENT4` | slightly more model residency
- `RESIDENT8` | higher residency, less movement
- `MLP36` | more MLP work on the secondary GPU
- `MLP40` | higher secondary-GPU MLP load
- `HEAD20` | more attention heads on the secondary GPU
- `BALANCE` | balanced experimental profile
- `MAX_TEST` | red-line profile, OOM is possible

The UI intentionally keeps these names simple. Detailed MLP ratios, attention-head splits, trim intervals, VRAM reserves, and hot-cache values are printed to the console for reproducible test reports.

## Ordinary Style / Character LoRAs

Use the public node:

```text
H3VM Style LoRA Stack
```

Ordinary Style / Character LoRAs are separate from Turbo acceleration LoRAs.

They use the same general `ModelPatcher` weight-patch semantics as ComfyUI's standard `LoraLoaderModelOnly`, and H3VM propagates the relevant patches to the root / island / helper patchers that actually own the corresponding weights.

Current support:

- `SINGLE_GPU`: supported
- `DUAL_QUIET`: supported
- `DUAL_CAPACITY`: supported
- `DUAL_SYNC_ACCEL` / Stock FullThrottle: ordinary Style LoRAs supported; Turbo acceleration LoRA remains disabled in standalone Mode 4

Each Style LoRA Stack node provides four slots. Multiple stack nodes can be chained through `previous_stack`, so the architecture is not limited to four LoRAs.

Known Larry / LightX H3 Turbo weights belong in the Turbo path and must not be disguised as ordinary Style LoRAs.

## Turbo LoRA support

The standalone Master Loader currently recognizes common H3 Turbo families including:

- Larry: `minimax_h3_turbo_v4_step600_ema.safetensors`
- ModelTC / LightX2V: `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`

The Master Loader selects the corresponding sampler semantics and handles the 12 / 3 video-audio sigma shift used by the supported Turbo paths.

The public H3VM Core is deliberately different: it does not choose or rewrite your Turbo path. A larger workflow should build the final model state first, then pass that model into H3VM Core.

## Installation

1. Put the `ComfyUI-H3-MultiGPU` folder inside:

```text
ComfyUI/custom_nodes/
```

2. Install dependencies from `requirements.txt` if your environment does not already contain them.

3. Restart ComfyUI.

4. Search for:

```text
H3VM Multi-GPU Loader
H3VM Core
H3VM Style LoRA Stack
```

When upgrading, deleting the old `ComfyUI-H3-MultiGPU` directory before extracting the new version is recommended. This avoids stale files from older experimental builds.

By default, only public user-facing nodes are shown. Historical research / lab nodes remain in the codebase for reproducibility and backward compatibility but are hidden from the normal node menu.

Developers can expose the lab nodes by setting this environment variable before starting ComfyUI:

```text
H3VM_SHOW_LAB_NODES=1
```

## Dual-GPU detection and identical cards

H3VM does **not** require two different GPU model names. Two RTX 3080 cards, two RTX 5060 Ti cards, and other identical-model pairs are valid. What matters is that the current ComfyUI/Python process can see two different CUDA **logical devices**.

Dual-GPU modes normally select `gpu:0` + `gpu:1`. These are PyTorch-visible logical indices. If the startup environment sets `CUDA_VISIBLE_DEVICES`, physical GPUs can be filtered and renumbered. For example, `CUDA_VISIBLE_DEVICES=0` exposes only one logical `cuda:0` even if the machine physically contains two GPUs. Expose both target cards, for example with `CUDA_VISIBLE_DEVICES=0,1`, and restart ComfyUI before using a dual-GPU mode.

Recent Windows ComfyUI builds may intentionally expose only GPU0 by default as a workaround for NVIDIA/CUDA multi-GPU issues. Start ComfyUI with `--cuda-device all` when needed and confirm that both `cuda:0` and `cuda:1` appear in the startup log.

The improved preflight reports:

- the number of PyTorch-visible CUDA devices
- `CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES`
- visible GPU names and VRAM
- a best-effort physical `nvidia-smi` inventory when preflight fails

On success, the console prints `[H3VM GPU PREFLIGHT]` with the resolved primary and secondary devices. This makes it clear whether the machine lacks a second GPU or the current process is simply hiding it.

### rc6 Windows comfy-kitchen DLPack guard

On Windows multi-GPU systems, a quantized tensor can already live on `cuda:1` while the Python thread current CUDA device is still `cuda:0`. comfy-kitchen's CUDA backend exports tensors through DLPack, and PyTorch rejects that export when the current device index does not match the tensor device.

rc6 installs a narrow compatibility guard only when Windows, at least two visible CUDA devices, and the comfy-kitchen CUDA backend are all present. Before DLPack export, H3VM switches the current CUDA device to the tensor-owning device. The guard patches only comfy-kitchen's private DLPack helper and does not rewrite Quiet, Capacity, or Mode4 scheduling. Set `H3VM_DISABLE_CK_MULTIGPU_GUARD=1` to opt out for troubleshooting.

Typical fixed error:

```text
BufferError: Can't export tensors on a different CUDA device index.
Expected: 1. Current device: 0.
```

See `RC6_WINDOWS_MULTIGPU_COMPAT.md` for details. This guard does not claim to fix unrelated Windows/NVIDIA host-memory, DynamicVRAM, or driver-level multi-GPU failures.

## Recommended first test

For the first validation run, keep the same prompt, seed, model, LoRA, and resolution. Change only the GPU mode.

Recommended order:

1. `SINGLE_GPU`
2. `DUAL_QUIET`
3. If a higher-resolution run OOMs on one GPU, try `DUAL_CAPACITY`
4. Test Stock H3 FullThrottle separately with `DUAL_SYNC_ACCEL`

For Capacity profile testing, a useful sequence is:

```text
SAFE -> RESIDENT8 -> MLP40 -> BALANCE
```

`MAX_TEST` is intended for finding the VRAM wall. OOM is an acceptable experimental result for that profile.

## Current validation platform

Primary development and validation system:

- Windows 11
- NVIDIA GeForce RTX 5060 Ti 16GB + RTX 5060 8GB
- PCIe 5.0 x8 + x8
- no CUDA P2P between the two GPUs
- cross-GPU relay through host RAM when required
- ComfyUI 0.33.0
- PyTorch 2.13.0 + CUDA 13.0
- DynamicVRAM / comfy-aimdo
- SageAttention

Other GPU combinations are very welcome. Heterogeneous dual-GPU systems are one of the main targets of this project.

## Project status

- `DUAL_QUIET`: current daily recommendation
- `DUAL_CAPACITY`: usable experimental capacity mode, especially valuable for high-resolution tests and mixed-VRAM GPU pairs
- `DUAL_SYNC_ACCEL`: built-in Stock H3 20-step FullThrottle path based on Dev9.4 Snapshot Islands; experimental approximate execution
- Public `H3VM Core`: intended as the reusable execution layer for larger H3 workflows

H3VM does not treat “GPU1 is busy” as proof of acceleration.

For performance modes, **wall time and the primary-GPU critical path** are the real success criteria.

For Capacity mode, the main success criterion is different:

> single GPU OOMs, multi-GPU H3VM completes the workload.

## Testing and contributing

Different GPU combinations are extremely valuable to this project.

If you test H3VM, please use `TEST_REPORT_TEMPLATE.md` and include as much of the following as possible:

- GPU0 model and VRAM
- GPU1 model and VRAM
- H3VM mode
- Capacity profile if applicable
- resolution
- duration
- steps
- whether the run completed
- total wall time
- peak dedicated VRAM on both GPUs
- audio quality observations
- console logs for H3VM placement / scheduling

Bug reports, compatibility findings, performance measurements, and pull requests are welcome.

## Design philosophy

H3VM is gradually moving away from being a model-specific loader and toward a reusable **H3 multi-GPU execution / memory layer**.

The intended boundary is:

> The workflow defines *what model state should be executed*. H3VM defines *how that model state is executed across one or more GPUs*.

That means the standalone Master Loader can remain convenient and opinionated, while the public Core engine stays narrow and composable for production pipelines.

## License

See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
