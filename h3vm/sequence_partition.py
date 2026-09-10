from __future__ import annotations

from dataclasses import dataclass


def _balanced_counts(total: int, parts: int) -> tuple[int, ...]:
    if total < 1 or parts < 1 or parts > total:
        raise ValueError("invalid total/parts")
    base, extra = divmod(int(total), int(parts))
    return tuple(base + (i < extra) for i in range(parts))


@dataclass(frozen=True)
class SequencePartition:
    """Device-agnostic partition of one H3 packed sequence.

    The class deliberately knows nothing about CUDA, NCCL, workers, or PM. It is
    only row arithmetic shared by future Exact-SP and existing token-row paths.
    """

    total: int
    counts: tuple[int, ...]

    def __post_init__(self):
        total = int(self.total)
        counts = tuple(int(x) for x in self.counts)
        if total < 1 or not counts or any(x < 1 for x in counts):
            raise ValueError("partition counts must be positive")
        if sum(counts) != total:
            raise ValueError(f"partition counts sum to {sum(counts)}, expected {total}")
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "counts", counts)

    @classmethod
    def balanced(cls, total: int, parts: int = 2):
        return cls(int(total), _balanced_counts(int(total), int(parts)))

    @classmethod
    def dual_ratio(cls, total: int, primary_fraction: float = 0.5, *, alignment: int = 1):
        total = int(total)
        alignment = max(1, int(alignment))
        fraction = float(primary_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("primary_fraction must be between 0 and 1")
        if total < 2:
            raise ValueError("dual partition requires at least two rows")

        raw = round(total * fraction)
        if alignment > 1:
            raw = round(raw / alignment) * alignment
        primary = min(total - 1, max(1, int(raw)))
        secondary = total - primary
        return cls(total, (primary, secondary))

    @property
    def parts(self) -> int:
        return len(self.counts)

    def bounds(self, rank: int) -> tuple[int, int]:
        rank = int(rank)
        if not 0 <= rank < self.parts:
            raise IndexError(f"rank {rank} outside 0..{self.parts - 1}")
        start = sum(self.counts[:rank])
        return start, start + self.counts[rank]

    def size(self, rank: int) -> int:
        return self.counts[int(rank)]

    def peer(self, rank: int) -> int:
        if self.parts != 2:
            raise ValueError("peer() is only defined for two-way partitions")
        rank = int(rank)
        if rank not in (0, 1):
            raise IndexError("dual rank must be 0 or 1")
        return 1 - rank

    def localize_segments(self, segments, rank: int):
        """Clip global [start, stop, ...] segments into rank-local row coordinates."""
        begin, end = self.bounds(rank)
        out = []
        for segment in segments:
            if len(segment) < 2:
                raise ValueError(f"invalid segment: {segment!r}")
            start, stop, *tail = segment
            left = max(int(start), begin)
            right = min(int(stop), end)
            if left < right:
                out.append((left - begin, right - begin, *tail))
        return out

    def split_lengths(self) -> tuple[int, ...]:
        return self.counts
