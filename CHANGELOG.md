# Changelog

## Unreleased

## v0.20.0-rc6

- Promoted the dual-GPU visibility/preflight work from `Unreleased`: explicit `gpu:N` / `cuda:N` selections now resolve as distinct PyTorch-logical CUDA devices before falling back to ComfyUI's resolver.
- Identical GPU model names are explicitly supported; for example, two RTX 3080 cards or two RTX 5060 Ti cards are valid when both are visible to the current ComfyUI/Python process.
- When fewer than two CUDA devices are visible, H3VM reports `torch.cuda.device_count()`, `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES`, visible GPU names/VRAM, and a best-effort `nvidia-smi` physical inventory instead of only the generic “requires at least two CUDA GPUs” error.
- Added a narrow Windows multi-GPU comfy-kitchen DLPack device-context guard. When a quantized tensor lives on `cuda:1` while the thread current device is still `cuda:0`, H3VM switches the current CUDA device before comfy-kitchen exports the tensor through DLPack.
- This fixes the observed `BufferError: Can't export tensors on a different CUDA device index. Expected: 1. Current device: 0.` path while loading secondary-GPU ConvRot INT8 weights.
- The comfy-kitchen guard is Windows-only, requires at least two PyTorch-visible CUDA devices, patches only `comfy_kitchen.backends.cuda._wrap_for_dlpack`, is idempotent/best-effort, and can be disabled with `H3VM_DISABLE_CK_MULTIGPU_GUARD=1`.
- Added a lightweight contract test for the cuda:0 -> cuda:1 -> cuda:0 context-switch behavior and wired it into CI.
- No Quiet, Capacity, or Mode4 scheduling algorithm was rewritten for rc6; the compatibility layer is installed before the runtime bridge.
- On recent Windows ComfyUI builds that intentionally expose only GPU0 by default, start ComfyUI with `--cuda-device all` so H3VM can see both logical CUDA devices. This is separate from the comfy-kitchen DLPack guard.

## v0.20.0-rc5

- Added public `H3VM Core｜多卡执行引擎`: pure `MODEL -> MODEL` integration node for external workflows.
- Master Loader remains the standalone cockpit and keeps its existing model/Turbo/steps/prompt/resolution controls.
- Core does not choose or modify Turbo LoRA, ordinary LoRA, Prompt, Seed, Resolution, Sampler, Sigmas, or actual sampling steps.
- `expected_steps_hint` is execution metadata only; Mode4 Snapshot uses it for runtime scheduling and no longer hardcodes 20 steps on the prebuilt-MODEL path.
- Public Core supports clean H3 plus ordinary ModelPatcher weight patches (including normal `LoraLoaderModelOnly` output) in all four modes.
- Generic subset/helper LoRA replay now filters patch keys to weights physically owned by each execution patcher.
- Mode4 prebuilt MODEL path now reuses the Dev9.4 Snapshot FullThrottle engine while preserving the workflow's sampler/sigma/step contract.
- Runtime injections, object patches, hook patches, and weight-wrapper patches remain fail-closed until explicitly remapped for helper GPUs.

## v0.20.0-rc4

- Mode 4 Stock FullThrottle now auto-detects ComfyUI global pinned-memory policy.
- With `--disable-pinned-memory`, historical Dev9.4 behavior is preserved: H3VM owns a bounded 1024MiB pinned activation mailbox.
- With normal ComfyUI pinned memory enabled, Mode 4 no longer aborts; H3VM activation/snapshot traffic automatically uses pageable RAM and does not create a second explicit pinned pool.
- Ordinary Style/character LoRA remains enabled in Mode 4; Turbo LoRA remains ignored there.
- SINGLE_GPU, DUAL_QUIET, and DUAL_CAPACITY scheduling are unchanged.

## v0.20.0-rc3

- Added public `H3VM Style LoRA Stack｜普通LoRA叠加` node for ordinary style/character LoRAs.
- Style LoRA is mode-independent and is replayed to the actual H3 weight owners in SINGLE_GPU, DUAL_QUIET, DUAL_CAPACITY, and Mode 4 Stock FullThrottle.
- Each Stack node holds four LoRAs and can chain `previous_stack`, so the stack is not architecturally limited to four entries.
- Ordinary Style LoRA is separated from Turbo/accelerator LoRA. Known Larry/LightX accelerator LoRAs are rejected from the Style Stack and must stay in the Turbo slot.
- Mode 4 still ignores Turbo LoRA and forces stock `res_multistep` + 20 steps, but ordinary Style LoRA remains enabled.
- LoRA patch replay is filtered by each helper/island patcher's real `state_dict` ownership before copying, preventing unrelated patches from being attached to partial helper trees.
- DUAL_QUIET and DUAL_CAPACITY scheduling algorithms remain unchanged.

## v0.20.0-rc2

- Reused public Mode 4 (`DUAL_SYNC_ACCEL`) as built-in Stock H3 FullThrottle mode.
- Restored the validated Dev9.4 22/28 Snapshot-Islands profile with 2GB RAM backing and linear predictor beta 0.75.
- Mode 4 now forces stock `res_multistep` + 20 steps, ignores Turbo LoRA, and skips Turbo SigmaShift.
- DUAL_QUIET and DUAL_CAPACITY algorithms are unchanged.
- Kept the rc1 Core runtime bridge and generic prebuilt-MODEL work intact.

## v0.20.0-rc1

- 产品入口更名为 `H3VM Multi-GPU Loader｜H3多卡加载器`。
- GPU 模式改为短中文说明，技术细节移到日志。
- Capacity 实验档位改为简洁中文说明，同时保留 v0.18 / v0.19 工作流别名兼容。
- Capacity 档位仅在 `DUAL_CAPACITY` 下显示并生效。
- Custom Width / Height 仅在 `CUSTOM｜自定义` 分辨率下显示。
- Prompt 保持最下方折叠，展开后提供 420px 大输入区。
- 删除节点内额外说明板与彩色帮助文字。
- README 改为面向公开发布的安装与测试说明。
- 明确记录：当前测试机上 Capacity 8-step 推荐；4-step 存在音质风险。
