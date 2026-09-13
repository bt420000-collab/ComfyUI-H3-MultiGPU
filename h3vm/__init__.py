"""H3 VRAM Master runtime package.

Importing this package is side-effect free. Runtime compatibility hooks are
installed only when an H3VM execution path is actually used.
"""
from __future__ import annotations

from threading import RLock

_RUNTIME_LOCK = RLock()
_RUNTIME_ACTIVE = False
_RUNTIME_ACTIVATING = False


def runtime_active() -> bool:
    return bool(_RUNTIME_ACTIVE)


def activate_runtime() -> bool:
    global _RUNTIME_ACTIVE, _RUNTIME_ACTIVATING
    if _RUNTIME_ACTIVE:
        return True
    with _RUNTIME_LOCK:
        if _RUNTIME_ACTIVE:
            return True
        if _RUNTIME_ACTIVATING:
            return False
        _RUNTIME_ACTIVATING = True
        try:
            from .gpu_preflight import install_gpu_preflight_patch
            install_gpu_preflight_patch()
            from .comfy_kitchen_multigpu import install_comfy_kitchen_multigpu_dlpack_guard
            install_comfy_kitchen_multigpu_dlpack_guard()
            from .v0211_mode4 import install_snapshot_runtime_patch
            install_snapshot_runtime_patch()
            from .v0211_vae_guard import install_dual_vae_identity_guard
            install_dual_vae_identity_guard()
            _RUNTIME_ACTIVE = True
            print("[H3VM] v0.21.1 runtime activated lazily | standard workflows untouched", flush=True)
            return True
        finally:
            _RUNTIME_ACTIVATING = False
