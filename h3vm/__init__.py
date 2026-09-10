"""H3 VRAM Master runtime package. Heavy CUDA/Comfy imports remain lazy where possible."""

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

from .core_adapter import install_runtime_bridge as _install_runtime_bridge
_install_runtime_bridge()
del _install_runtime_bridge

# H3VM now means H3 VRAM Master. The fusion overlay stays deliberately thin:
# it adds hardware-aware planning around the proven public backends instead of
# replacing the heavy execution algorithms wholesale.
from .vram_master_fusion import install_vram_master_fusion as _install_vram_master_fusion
_install_vram_master_fusion()
del _install_vram_master_fusion
