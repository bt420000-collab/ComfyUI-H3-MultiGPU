"""H3VM rc5 runtime package. Heavy CUDA/Comfy imports remain lazy."""

# Install the logical-device resolver before Core/loader compatibility wrappers
# import and bind loader_base private helpers.
from .gpu_preflight import install_gpu_preflight_patch as _install_gpu_preflight_patch
_install_gpu_preflight_patch()
del _install_gpu_preflight_patch

from .core_adapter import install_runtime_bridge as _install_runtime_bridge
_install_runtime_bridge()
del _install_runtime_bridge
