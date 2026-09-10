from __future__ import annotations

"""Exact sequence/head-parallel planning for H3 VRAM Master.

This file does not start worker processes and knows nothing about PM. It is a
single-inference compute plan: sequence ownership, head ownership, and expected
cross-device traffic. The execution backend can consume the plan later.
"""

from dataclasses import dataclass

from .compute_planner import PairProfile, RuntimePlan, build_runtime_plan
from .sequence_partition import SequencePartition


@dataclass(frozen=True)
class ExactSPPlan:
    runtime: RuntimePlan
    sequence: SequencePartition
    head_counts: tuple[int, int]
    hidden_exchange_bytes_per_block: int
    head_exchange_bytes_per_block: int
    total_exchange_bytes_per_block: int
    recommended: bool
    recommendation_reason: str

    @property
    def primary_heads(self) -> int:
        return self.head_counts[0]

    @property
    def secondary_heads(self) -> int:
        return self.head_counts[1]


def estimate_exchange_bytes(*, seq_len: int, hidden_dim: int, heads: int,
                            head_dim: int, element_size: int,
                            primary_fraction: float) -> tuple[int, int, int]:
    """Estimate the two mandatory Exact-SP communication phases per H3 block.

    1) hidden exchange: each side receives the peer's token rows so its local
       QKV head shard can attend over the full sequence;
    2) head-output exchange: each side receives peer head outputs for its local
       token rows before applying the original full OutProj.

    The estimate is bidirectional aggregate bytes, not bus transaction bytes.
    """
    seq_len = int(seq_len)
    hidden_dim = int(hidden_dim)
    heads = int(heads)
    head_dim = int(head_dim)
    element_size = int(element_size)
    if min(seq_len, hidden_dim, heads, head_dim, element_size) <= 0:
        raise ValueError("all dimensions must be positive")

    sequence = SequencePartition.dual_ratio(seq_len, primary_fraction)
    p_tokens, s_tokens = sequence.counts

    # Each direction sends its local hidden rows exactly once.
    hidden_bytes = (p_tokens + s_tokens) * hidden_dim * element_size

    # After local-head attention over the full sequence, each side sends the
    # output heads needed for the peer's local token rows. Summed over both
    # directions this equals one full sequence x all-head activation.
    head_bytes = seq_len * heads * head_dim * element_size
    total = hidden_bytes + head_bytes
    return int(hidden_bytes), int(head_bytes), int(total)


def build_exact_sp_plan(pair: PairProfile, *, seq_len: int, hidden_dim: int,
                        heads: int = 56, head_dim: int = 128,
                        element_size: int = 2, manual_ratio=None,
                        host_relay_gbps: float | None = None) -> ExactSPPlan:
    runtime = build_runtime_plan(
        pair,
        backend="EXACT_SP",
        manual_ratio=manual_ratio,
        total_heads=int(heads),
    )
    sequence = SequencePartition.dual_ratio(seq_len, runtime.primary_fraction)
    head_counts = (runtime.attention_primary_heads, runtime.attention_secondary_heads)
    hb, ab, total = estimate_exchange_bytes(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        heads=heads,
        head_dim=head_dim,
        element_size=element_size,
        primary_fraction=runtime.primary_fraction,
    )

    if runtime.exact_sp_tier == "READY":
        recommended = True
        why = "bidirectional P2P available"
    elif host_relay_gbps is None:
        recommended = False
        why = "host-relay bandwidth not measured; keep Exact-SP in Lab"
    else:
        # Exact-SP performs two exchanges every transformer block. On host-only
        # systems, a low measured relay rate can erase compute parallel gains.
        # 12 GB/s is deliberately conservative and remains a lab threshold, not
        # a universal hardware requirement.
        recommended = float(host_relay_gbps) >= 12.0
        why = (
            f"measured host relay {float(host_relay_gbps):.2f} GB/s "
            + ("passes" if recommended else "fails")
            + " the 12 GB/s lab threshold"
        )

    return ExactSPPlan(
        runtime=runtime,
        sequence=sequence,
        head_counts=head_counts,
        hidden_exchange_bytes_per_block=hb,
        head_exchange_bytes_per_block=ab,
        total_exchange_bytes_per_block=total,
        recommended=bool(recommended),
        recommendation_reason=why,
    )
