from __future__ import annotations

"""Executable two-GPU exchange primitives for H3 VRAM Master Exact-SP Lab.

This is intentionally below the model/block hook layer.  It implements the two
communication operations an exact two-rank sequence/head-parallel H3 block needs:

1. gather hidden token shards so each GPU can run its local QKV head shard over
   the full sequence;
2. redistribute local attention-head outputs back to token ownership before the
   original full OutProj/MLP runs on each token shard.

No worker processes, PM concepts or project scheduling live here.
"""

from dataclasses import dataclass
import time

from .sequence_partition import SequencePartition


@dataclass
class ExactSPExchangeStats:
    hidden_calls: int = 0
    head_calls: int = 0
    bytes_moved: int = 0
    seconds: float = 0.0

    def as_dict(self):
        gbps = (self.bytes_moved / max(self.seconds, 1e-9)) / 1e9 if self.bytes_moved else 0.0
        return {
            "hidden_calls": int(self.hidden_calls),
            "head_calls": int(self.head_calls),
            "bytes_moved": int(self.bytes_moved),
            "seconds": float(self.seconds),
            "effective_gbps": float(gbps),
        }


def _tensor_bytes(tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _slice_axis(tensor, axis: int, start: int, end: int):
    axis = int(axis)
    if axis < 0:
        axis += tensor.ndim
    if not 0 <= axis < tensor.ndim:
        raise ValueError(f"axis out of range: {axis} for ndim={tensor.ndim}")
    if not 0 <= int(start) <= int(end) <= int(tensor.shape[axis]):
        raise ValueError(f"invalid slice [{start}:{end}] for axis size {tensor.shape[axis]}")
    return tensor.narrow(axis, int(start), int(end) - int(start))


class ExactSPExchangeFabric:
    """Exact tensor exchange using H3VM TransportEngine semantics."""

    def __init__(self, primary, secondary, *, transport_engine):
        import torch
        self.primary = torch.device(primary)
        self.secondary = torch.device(secondary)
        self.transport = transport_engine
        self.stats = ExactSPExchangeStats()

    def _move(self, tensor, dst):
        t0 = time.perf_counter()
        out = self.transport.move_tensor(tensor, dst)
        dt = time.perf_counter() - t0
        self.stats.bytes_moved += _tensor_bytes(tensor)
        self.stats.seconds += dt
        return out

    @staticmethod
    def _validate_partition(partition: SequencePartition, primary_local, secondary_local, *, seq_axis=0):
        seq_axis = int(seq_axis)
        if seq_axis < 0:
            seq_axis += primary_local.ndim
        if primary_local.ndim != secondary_local.ndim:
            raise ValueError("primary/secondary tensors must have the same rank")
        if int(primary_local.shape[seq_axis]) != int(partition.counts[0]):
            raise ValueError("primary local sequence does not match partition")
        if int(secondary_local.shape[seq_axis]) != int(partition.counts[1]):
            raise ValueError("secondary local sequence does not match partition")
        for axis, (a, b) in enumerate(zip(primary_local.shape, secondary_local.shape)):
            if axis != seq_axis and int(a) != int(b):
                raise ValueError("non-sequence dimensions must match")
        return seq_axis

    def gather_hidden(self, primary_local, secondary_local, partition: SequencePartition, *, seq_axis=0):
        """Return full sequence on both GPUs while preserving primary-then-secondary order."""
        import torch

        seq_axis = self._validate_partition(partition, primary_local, secondary_local, seq_axis=seq_axis)
        if primary_local.device != self.primary or secondary_local.device != self.secondary:
            raise ValueError("hidden shards are not on the configured devices")

        secondary_on_primary = self._move(secondary_local.contiguous(), self.primary)
        primary_on_secondary = self._move(primary_local.contiguous(), self.secondary)
        full_primary = torch.cat((primary_local, secondary_on_primary), dim=seq_axis)
        full_secondary = torch.cat((primary_on_secondary, secondary_local), dim=seq_axis)
        self.stats.hidden_calls += 1
        return full_primary, full_secondary

    def redistribute_heads(self, primary_heads_full, secondary_heads_full,
                           partition: SequencePartition, *, seq_axis=-2, head_axis=1):
        """Head-sharded full-sequence attention -> full-head local-token shards.

        ``primary_heads_full`` contains the lower/first head range for all tokens.
        ``secondary_heads_full`` contains the upper/second head range for all tokens.
        The returned tensors keep token ownership but restore original head order.
        """
        import torch

        if primary_heads_full.device != self.primary or secondary_heads_full.device != self.secondary:
            raise ValueError("attention head shards are not on the configured devices")
        if primary_heads_full.ndim != secondary_heads_full.ndim:
            raise ValueError("head shards must have the same rank")

        sa = int(seq_axis)
        ha = int(head_axis)
        if sa < 0:
            sa += primary_heads_full.ndim
        if ha < 0:
            ha += primary_heads_full.ndim
        if sa == ha:
            raise ValueError("sequence and head axes must differ")
        total = int(sum(partition.counts))
        if int(primary_heads_full.shape[sa]) != total or int(secondary_heads_full.shape[sa]) != total:
            raise ValueError("full-sequence head shards do not match partition total")
        for axis, (a, b) in enumerate(zip(primary_heads_full.shape, secondary_heads_full.shape)):
            if axis not in (ha,) and int(a) != int(b):
                raise ValueError("head-shard tensors differ outside the head axis")

        p_count, s_count = (int(partition.counts[0]), int(partition.counts[1]))
        p0, p1 = 0, p_count
        s0, s1 = p_count, p_count + s_count

        primary_local_heads = _slice_axis(primary_heads_full, sa, p0, p1).contiguous()
        helper_for_primary = _slice_axis(secondary_heads_full, sa, p0, p1).contiguous()
        helper_for_primary = self._move(helper_for_primary, self.primary)
        primary_tokens_full_heads = torch.cat((primary_local_heads, helper_for_primary), dim=ha)

        root_for_secondary = _slice_axis(primary_heads_full, sa, s0, s1).contiguous()
        root_for_secondary = self._move(root_for_secondary, self.secondary)
        secondary_local_heads = _slice_axis(secondary_heads_full, sa, s0, s1).contiguous()
        secondary_tokens_full_heads = torch.cat((root_for_secondary, secondary_local_heads), dim=ha)

        self.stats.head_calls += 1
        return primary_tokens_full_heads, secondary_tokens_full_heads
