# H3VM v0.20.0-rc6 Windows Multi-GPU Compatibility Notes

## 中文

rc6 是一次窄范围兼容更新，不重写 H3VM 已验证的 Quiet / Capacity / Mode4 调度算法。

### 修复了什么

在 Windows 双 NVIDIA GPU 环境中，H3VM 会把真实量化权重分配到不同 CUDA 逻辑设备。例如某个 ConvRot INT8 shard 已经位于 `cuda:1`，但当前 Python 线程的 CUDA current device 仍可能停在 `cuda:0`。

comfy-kitchen CUDA backend 当前会通过：

```python
Tensor.__dlpack__(stream=-1)
```

把张量交给 native CUDA 扩展。PyTorch 要求 DLPack 导出时 current CUDA device 与 tensor device 一致，否则会出现：

```text
BufferError: Can't export tensors on a different CUDA device index.
Expected: 1. Current device: 0.
```

rc6 新增 `h3vm/comfy_kitchen_multigpu.py`。在 Windows、当前进程可见至少两张 CUDA GPU、并且 comfy-kitchen CUDA backend 可用时，它会在 DLPack 导出前把 current CUDA device 切换到 tensor 所在设备。

该 Guard：

- 仅 Windows 生效
- 仅多 CUDA GPU 生效
- 只补丁 `comfy_kitchen.backends.cuda._wrap_for_dlpack`
- 可重复安装，不叠加 wrapper
- comfy-kitchen CUDA backend 不存在时自动跳过
- 可设置 `H3VM_DISABLE_CK_MULTIGPU_GUARD=1` 手动关闭

### 双卡可见性

近期 Windows ComfyUI 版本可能为了规避 NVIDIA/CUDA 多 GPU 问题，默认只向进程暴露 GPU0。若 H3VM 报当前进程只看到一张 CUDA GPU，请使用：

```text
--cuda-device all
```

重新启动 ComfyUI，并确认启动日志同时列出 `cuda:0` 和 `cuda:1`。

这和 rc6 的 comfy-kitchen DLPack Guard 是两个独立问题：前者解决“第二张卡是否可见”，后者解决“cuda:1 tensor 是否在正确 CUDA device context 中交给 comfy-kitchen”。

### 范围说明

rc6 不声称解决所有 Windows / NVIDIA 多 GPU 的 host-memory、DynamicVRAM 或驱动级问题。它只针对已经复现并验证的 comfy-kitchen DLPack device-index mismatch 路径。

本次兼容修复已在 Windows 双 RTX 5060 Ti 16GB、PyTorch 2.13.0 + CUDA 13.0、ComfyUI 0.35.0 环境的 H3VM production 路径中复现并验证通过，然后以窄作用域 Guard 形式移植到公版。

## English

rc6 is a narrow compatibility update. It does not rewrite the validated H3VM Quiet, Capacity, or Mode4 scheduling algorithms.

On Windows multi-NVIDIA-GPU systems, a real quantized H3VM shard can live on `cuda:1` while the Python thread current CUDA device remains `cuda:0`. comfy-kitchen exports CUDA tensors to its native extension through `Tensor.__dlpack__(stream=-1)`, and PyTorch rejects that export when the current device index does not match the tensor device.

rc6 adds `h3vm/comfy_kitchen_multigpu.py`, which switches the current CUDA device to the tensor-owning device before comfy-kitchen performs the DLPack export. The shim is Windows-only, multi-GPU-only, idempotent, best-effort, and can be disabled with `H3VM_DISABLE_CK_MULTIGPU_GUARD=1`.

Recent Windows ComfyUI builds may expose only GPU0 by default. Use `--cuda-device all` when needed and verify that both `cuda:0` and `cuda:1` appear in the startup log.

The CUDA visibility fix and the comfy-kitchen DLPack guard solve two different layers of the problem. rc6 does not claim to fix unrelated Windows/NVIDIA host-memory, DynamicVRAM, or driver-level failures.
