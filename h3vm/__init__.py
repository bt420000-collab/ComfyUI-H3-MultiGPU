"""H3VM rc5 runtime package. Heavy CUDA/Comfy imports remain lazy."""
from .core_adapter import install_runtime_bridge as _install_runtime_bridge
_install_runtime_bridge()
del _install_runtime_bridge
