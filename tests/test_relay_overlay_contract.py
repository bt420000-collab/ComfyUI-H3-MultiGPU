from h3vm.attention_parallel import H3VMRelayAttentionParallel
from h3vm.attention_relay_packet import _PATCH_MARKER as PACKET_MARKER
from h3vm.global_memory import TransportEngine
from h3vm.segmented_relay import _PATCH_MARKER as SEGMENT_MARKER

assert getattr(H3VMRelayAttentionParallel._packed_head_slice_host, PACKET_MARKER, False) is True
assert getattr(TransportEngine.move_tensor, SEGMENT_MARKER, False) is True

print("H3 VRAM Master relay overlay contract: OK")
