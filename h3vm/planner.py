from __future__ import annotations
from dataclasses import dataclass

GIB = 1024 ** 3

@dataclass(frozen=True)
class BlockStat:
    index: int
    storage_bytes: int

@dataclass(frozen=True)
class Placement:
    secondary_count: int
    secondary_bytes: int
    primary_block_bytes: int
    primary_fixed_bytes: int
    primary_capacity: int
    secondary_capacity: int

    @property
    def fast_pool_fit(self) -> bool:
        return (self.primary_block_bytes + self.primary_fixed_bytes <= self.primary_capacity
                and self.secondary_bytes <= self.secondary_capacity)

    def describe(self) -> str:
        return (
            f"secondary=0-{self.secondary_count-1} ({self.secondary_count}), "
            f"primary={self.secondary_count}-49, fast_pool_fit={self.fast_pool_fit}"
        )


def _capacities(primary_total, secondary_total, primary_reserve_gib, secondary_reserve_gib):
    return (
        max(0, int(primary_total - primary_reserve_gib * GIB)),
        max(0, int(secondary_total - secondary_reserve_gib * GIB)),
    )


def choose_contiguous_prefix(blocks, fixed_primary_bytes: int, primary_total: int, secondary_total: int,
                             primary_reserve_gib: float, secondary_reserve_gib: float,
                             policy: str = "balanced") -> Placement:
    pcap, scap = _capacities(primary_total, secondary_total, primary_reserve_gib, secondary_reserve_gib)
    total_blocks = sum(b.storage_bytes for b in blocks)
    candidates = []
    running = 0
    for n in range(1, len(blocks)):
        running += blocks[n-1].storage_bytes
        sbytes = running
        pbytes = total_blocks - sbytes
        p_used = fixed_primary_bytes + pbytes
        if sbytes <= scap and p_used <= pcap:
            p_head = (pcap - p_used) / max(1, pcap)
            s_head = (scap - sbytes) / max(1, scap)
            candidates.append((n, sbytes, pbytes, p_head, s_head))
    if not candidates:
        raise RuntimeError(
            "H3VM: H3 weights do not fit the requested two-GPU fast pool with the selected reserves. "
            "RAM overflow is intentionally disabled in this milestone."
        )

    if policy == "speed_first":
        # The secondary card is the smaller/slower card in the target 16G+8G setup.
        # Use the minimum number of secondary blocks that makes the primary fit.
        n, sbytes, pbytes, _, _ = min(candidates, key=lambda x: x[0])
    else:
        # Legacy Dev4 behavior: maximize the weaker normalized headroom.
        n, sbytes, pbytes, _, _ = max(
            candidates,
            key=lambda x: (min(x[3], x[4]), -abs(x[0] - len(blocks)//3)),
        )
    return Placement(n, sbytes, pbytes, fixed_primary_bytes, pcap, scap)
