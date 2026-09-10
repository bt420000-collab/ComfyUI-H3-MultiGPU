"""H3VM rc6 runtime package. Heavy CUDA/Comfy imports remain lazy where possible."""

# Install the logical-device resolver before Core/loader compatibility wrappers
# import and bind loader_base private helpers.
from .gpu_preflight import install_gpu_preflight_patch as _install_gpu_preflight_patch
_install_gpu_preflight_patch()
del _install_gpu_preflight_patch

# Windows + multi-GPU + comfy-kitchen CUDA can enter a state where a tensor is
# physically on cuda:1 while the thread current device remains cuda:0. PyTorch
# then rejects comfy-kitchen's DLPack export. Install the narrow, idempotent
# compatibility guard before H3VM starts constructing/loading GPU islands.
from .comfy_kitchen_multigpu import (
    install_comfy_kitchen_multigpu_dlpack_guard as _install_ck_multigpu_guard,
)
_install_ck_multigpu_guard()
del _install_ck_multigpu_guard

# Lab branch only: opt-in equal-card planner for 16G+16G (and other near-identical
# pairs). The environment variable gate keeps stock rc6 behavior unchanged unless
# a tester explicitly enables H3VM_DUAL16_SYMMETRIC_LAB=1.
from .dual16g_lab import install_dual16g_symmetric_relay_patch as _install_dual16g_lab
_install_dual16g_lab()
del _install_dual16g_lab

from .core_adapter import install_runtime_bridge as _install_runtime_bridge
_install_runtime_bridge()
del _install_runtime_bridge
