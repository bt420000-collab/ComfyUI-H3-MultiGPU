from __future__ import annotations

"""Single-packet codec for H3 VRAM Master host relay.

Windows/no-P2P attention used to move q, k, v and three scale tensors through
host RAM as six independent transfers.  This module packs same-device tensors
into one aligned uint8 packet, moves that packet once, and reconstructs zero-copy
views on the destination GPU.

The packet is a transient transport object only.  It never stores model weights
and it does not change H3 arithmetic.
"""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TensorPacketSpec:
    offset_bytes: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: object


@dataclass(frozen=True)
class PacketInfo:
    tensor_count: int
    payload_bytes: int
    packet_bytes: int
    alignment: int

    @property
    def padding_bytes(self) -> int:
        return int(self.packet_bytes - self.payload_bytes)


def _align_up(value: int, alignment: int) -> int:
    value = int(value)
    alignment = max(1, int(alignment))
    return ((value + alignment - 1) // alignment) * alignment


def aligned_layout(sizes: Iterable[int], alignment: int = 16) -> tuple[tuple[int, ...], int]:
    """Return aligned byte offsets and final packet size without importing torch."""
    alignment = max(1, int(alignment))
    offsets = []
    cursor = 0
    for raw in sizes:
        size = int(raw)
        if size < 0:
            raise ValueError("tensor byte sizes must be non-negative")
        cursor = _align_up(cursor, alignment)
        offsets.append(cursor)
        cursor += size
    return tuple(offsets), _align_up(cursor, alignment)


def tensor_nbytes(tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def pack_tensors(tensors, *, alignment: int = 16):
    """Pack contiguous byte images of tensors from one device into one uint8 tensor."""
    import torch

    values = tuple(tensors)
    if not values:
        raise ValueError("pack_tensors requires at least one tensor")
    if not all(torch.is_tensor(t) for t in values):
        raise TypeError("pack_tensors accepts tensors only")

    device = values[0].device
    if any(t.device != device for t in values):
        raise ValueError("all packet tensors must live on the same source device")

    contiguous = tuple(t if t.is_contiguous() else t.contiguous() for t in values)
    sizes = tuple(tensor_nbytes(t) for t in contiguous)
    offsets, packet_bytes = aligned_layout(sizes, alignment=alignment)

    with torch.device(device):
        packet = torch.empty(packet_bytes, dtype=torch.uint8, device=device)
    specs = []
    for t, off, nbytes in zip(contiguous, offsets, sizes):
        if nbytes:
            packet.narrow(0, int(off), int(nbytes)).copy_(t.view(torch.uint8).reshape(-1))
        specs.append(TensorPacketSpec(
            offset_bytes=int(off), nbytes=int(nbytes), shape=tuple(int(x) for x in t.shape), dtype=t.dtype,
        ))

    return packet, tuple(specs), PacketInfo(
        tensor_count=len(values), payload_bytes=sum(sizes), packet_bytes=int(packet_bytes), alignment=int(alignment),
    )


def unpack_tensors(packet, specs):
    """Reconstruct contiguous typed views. Returned tensors alias packet storage."""
    import torch

    if not torch.is_tensor(packet) or packet.dtype is not torch.uint8 or packet.ndim != 1:
        raise TypeError("packet must be a flat torch.uint8 tensor")
    outputs = []
    for spec in specs:
        segment = packet.narrow(0, int(spec.offset_bytes), int(spec.nbytes))
        if int(spec.nbytes) == 0:
            out = torch.empty(tuple(spec.shape), dtype=spec.dtype, device=packet.device)
        else:
            out = segment.view(spec.dtype).reshape(tuple(spec.shape))
        outputs.append(out)
    return tuple(outputs)


def move_tensor_group(engine, tensors, dst_device, *, mode=None, alignment: int = 16):
    """Pack -> one TransportEngine move -> zero-copy unpack on destination."""
    packet, specs, info = pack_tensors(tensors, alignment=alignment)
    moved = engine.move_tensor(packet, dst_device, mode=mode)
    outputs = unpack_tensors(moved, specs)
    return outputs, info
