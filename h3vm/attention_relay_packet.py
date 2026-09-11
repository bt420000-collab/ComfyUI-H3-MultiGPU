from __future__ import annotations

"""Packetized host relay overlay for H3VM attention.

The legacy host path performs six TransportEngine moves for one helper head
shard: q/k/v plus q/k/v scales.  On Windows no-P2P this multiplies Python,
allocator and host-transfer setup overhead.  The VRAM Master overlay combines
those tensors into one aligned packet and performs one host transfer.

Any incompatibility falls back to the exact legacy method for that call.
"""

import logging
import os

LOG = logging.getLogger("H3VM")
_PATCH_MARKER = "_h3vm_attention_relay_packet_v1"
_WARNED = set()
_LOGGED = set()


def _min_payload_bytes() -> int:
    raw = os.environ.get("H3VM_RELAY_PACKET_MIN_KIB", "256")
    try:
        return max(0, int(float(raw) * 1024.0))
    except Exception:
        return 256 * 1024


def _payload_bytes(values) -> int:
    return sum(int(v.numel() * v.element_size()) for v in values)


def install_attention_relay_packet_patch() -> bool:
    if os.environ.get("H3VM_DISABLE_RELAY_PACKET", "0") == "1":
        return False

    from .attention_parallel import H3VMRelayAttentionParallel
    current = H3VMRelayAttentionParallel._packed_head_slice_host
    if getattr(current, _PATCH_MARKER, False):
        return True
    original = current

    def packed_head_slice_host(self, quantized, start, end, device):
        import comfy_kitchen

        if self.host_engine is None:
            return original(self, quantized, start, end, device)

        try:
            batch, heads, _, head_dim = quantized.q.shape
            padded_length = int(quantized.v.shape[-1])
            vv = quantized.v.view(batch, heads, head_dim, padded_length)[:, start:end].contiguous()
            v_scale = quantized.v_scale.view(batch, heads, head_dim)[:, start:end].contiguous()
            values = (
                quantized.q[:, start:end].contiguous(),
                quantized.k[:, start:end].contiguous(),
                vv.reshape(-1, padded_length).contiguous(),
                quantized.q_scale[:, start:end].contiguous(),
                quantized.k_scale[:, start:end].contiguous(),
                v_scale.reshape(-1).contiguous(),
            )
            payload = _payload_bytes(values)
            if payload < _min_payload_bytes():
                return original(self, quantized, start, end, device)

            from .relay_packet import move_tensor_group
            moved, info = move_tensor_group(self.host_engine, values, device, alignment=16)
            q, k, v, q_scale, k_scale, vs = moved

            key = id(self)
            if key not in _LOGGED:
                _LOGGED.add(key)
                LOG.info(
                    "H3VM VRAM MASTER RELAY PACKET ACTIVE | tensors=%d payload=%.2fMiB packet=%.2fMiB padding=%dB transport=%s",
                    info.tensor_count, info.payload_bytes / (1024 ** 2), info.packet_bytes / (1024 ** 2),
                    info.padding_bytes, getattr(self, "transport", None),
                )

            return comfy_kitchen.PrequantizedInt8Attention(
                q=q,
                k=k,
                v=v,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=vs,
                original_head_dim=quantized.original_head_dim,
                input_dtype=quantized.input_dtype,
                attention_scale=quantized.attention_scale,
                cta_k=quantized.cta_k,
                attn_mask=None,
            )
        except Exception as exc:
            key = id(self)
            if key not in _WARNED:
                _WARNED.add(key)
                LOG.warning("H3VM relay packet fallback to legacy six-move path | %r", exc)
            return original(self, quantized, start, end, device)

    setattr(packed_head_slice_host, _PATCH_MARKER, True)
    H3VMRelayAttentionParallel._h3vm_original_packed_head_slice_host = original
    H3VMRelayAttentionParallel._packed_head_slice_host = packed_head_slice_host
    LOG.info("H3VM VRAM Master relay packet overlay installed | 6 host moves -> 1 packet move")
    return True
