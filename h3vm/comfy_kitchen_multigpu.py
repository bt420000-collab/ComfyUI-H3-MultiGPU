"""Windows multi-GPU compatibility guard for comfy-kitchen CUDA DLPack exports.

H3VM can place real quantized model shards on more than one CUDA logical
device. comfy-kitchen's CUDA backend exports tensors to its native extension
through ``Tensor.__dlpack__(stream=-1)``. PyTorch requires the thread's current
CUDA device to match the tensor's CUDA device at export time. On Windows
multi-GPU ComfyUI processes the current device can remain ``cuda:0`` while a
quantized H3VM island is being loaded on ``cuda:1``, which can raise::

    BufferError: Can't export tensors on a different CUDA device index.
    Expected: 1. Current device: 0.

This shim switches the current CUDA device to the tensor-owning device before
delegating to comfy-kitchen's private DLPack helper. The switch is deliberately
sticky for the current thread because the native comfy-kitchen CUDA launch
follows immediately after the DLPack exports and must inherit the same device
context. A later export for another CUDA device switches the context again.

Scope is intentionally narrow:
* Windows only
* two or more PyTorch-visible CUDA devices only
* comfy-kitchen CUDA backend only
* patches only ``comfy_kitchen.backends.cuda._wrap_for_dlpack``
* idempotent and best-effort

Set ``H3VM_DISABLE_CK_MULTIGPU_GUARD=1`` to opt out. Remove this compatibility
shim once upstream comfy-kitchen provides an equivalent device-context guard.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

LOG = logging.getLogger(__name__)

_PATCH_MARKER = "_h3vm_multigpu_dlpack_guard_v1"
_ORIGINAL_ATTR = "_h3vm_original_wrap_for_dlpack"
_seen_switches: set[tuple[int, int]] = set()


def install_comfy_kitchen_multigpu_dlpack_guard() -> bool:
    """Install H3VM's narrow Windows multi-GPU DLPack device-context guard.

    Returns ``True`` when the guard is installed or already installed. Returns
    ``False`` when the current platform/runtime does not need it, the user has
    opted out, or the target comfy-kitchen helper is unavailable.
    """

    if os.environ.get("H3VM_DISABLE_CK_MULTIGPU_GUARD", "0") == "1":
        return False
    if sys.platform != "win32":
        return False

    try:
        import torch
    except Exception as exc:  # pragma: no cover - import safety
        LOG.debug("H3VM CK multi-GPU guard skipped: torch import failed: %s", exc)
        return False

    try:
        if not torch.cuda.is_available() or int(torch.cuda.device_count()) < 2:
            return False
    except Exception as exc:  # pragma: no cover - defensive
        LOG.debug("H3VM CK multi-GPU guard skipped: CUDA probe failed: %s", exc)
        return False

    try:
        import comfy_kitchen.backends.cuda as ck_cuda
    except Exception as exc:
        LOG.debug(
            "H3VM CK multi-GPU guard skipped: comfy-kitchen CUDA backend unavailable: %s",
            exc,
        )
        return False

    current = getattr(ck_cuda, "_wrap_for_dlpack", None)
    if current is None or not callable(current):
        LOG.warning("H3VM CK multi-GPU guard not installed: _wrap_for_dlpack unavailable")
        return False

    if getattr(current, _PATCH_MARKER, False):
        return True

    # If H3VM is reloaded, wrap the preserved upstream helper instead of
    # stacking another compatibility wrapper on top of our own wrapper.
    original = getattr(ck_cuda, _ORIGINAL_ATTR, None)
    if original is None or not callable(original):
        original = current
        setattr(ck_cuda, _ORIGINAL_ATTR, original)

    def _guarded_wrap_for_dlpack(tensor: Any):
        try:
            if bool(getattr(tensor, "is_cuda", False)):
                device = getattr(tensor, "device", None)
                target = getattr(device, "index", None)
                if target is None and hasattr(tensor, "get_device"):
                    target = int(tensor.get_device())
                if target is not None:
                    target = int(target)
                    active = int(torch.cuda.current_device())
                    if active != target:
                        # Deliberately sticky. The native comfy-kitchen CUDA
                        # call follows the DLPack export and should inherit the
                        # tensor-owning device context.
                        torch.cuda.set_device(target)
                        key = (active, target)
                        if key not in _seen_switches:
                            _seen_switches.add(key)
                            LOG.info(
                                "H3VM CK multi-GPU DLPack guard | current cuda:%d -> cuda:%d",
                                active,
                                target,
                            )
        except Exception as exc:
            # Keep upstream failure reporting canonical if context switching
            # itself fails for an unexpected runtime.
            LOG.warning("H3VM CK multi-GPU DLPack guard context switch failed: %s", exc)

        return original(tensor)

    setattr(_guarded_wrap_for_dlpack, _PATCH_MARKER, True)
    _guarded_wrap_for_dlpack.__name__ = getattr(original, "__name__", "_wrap_for_dlpack")
    _guarded_wrap_for_dlpack.__doc__ = getattr(original, "__doc__", None)
    ck_cuda._wrap_for_dlpack = _guarded_wrap_for_dlpack

    LOG.info(
        "H3VM CK multi-GPU DLPack guard installed | visible=%d | policy=sticky_tensor_device",
        int(torch.cuda.device_count()),
    )
    return True
