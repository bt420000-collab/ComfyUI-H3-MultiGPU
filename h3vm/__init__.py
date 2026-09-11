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

# Upgrade explicit neutral_pinned TransportEngine moves to a bounded two-slot
# segmented pipeline. It reuses the existing ring allocation and falls back to
# the legacy blocking bounce if the platform rejects the pipelined path.
from .segmented_relay import install_segmented_relay_patch as _install_segmented_relay_patch
_install_segmented_relay_patch()
del _install_segmented_relay_patch

# Host attention used to perform six independent RAM crossings for q/k/v and
# their scales. Packetize the helper shard into one aligned transient transfer.
from .attention_relay_packet import (
    install_attention_relay_packet_patch as _install_attention_relay_packet_patch,
)
_install_attention_relay_packet_patch()
del _install_attention_relay_packet_patch

from .core_adapter import install_runtime_bridge as _install_runtime_bridge
_install_runtime_bridge()
del _install_runtime_bridge

# H3VM now means H3 VRAM Master. The fusion overlay stays deliberately thin:
# it adds hardware-aware planning around the proven public backends instead of
# replacing the heavy execution algorithms wholesale.
from .vram_master_fusion import install_vram_master_fusion as _install_vram_master_fusion
_install_vram_master_fusion()
del _install_vram_master_fusion

# Capacity keeps its exact legacy path as an automatic fallback. Supported
# quantized QKV layers are sliced while still quantized, eliminating the old
# full dequantize -> slice -> F.linear detour for those calls.
from .capacity_quantized_qkv import (
    install_capacity_quantized_qkv_patch as _install_capacity_quantized_qkv_patch,
)
_install_capacity_quantized_qkv_patch()
del _install_capacity_quantized_qkv_patch
