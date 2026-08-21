from __future__ import annotations


def balanced_counts(heads: int, n: int) -> list[int]:
    if heads < 1 or n < 1 or n > heads:
        raise ValueError("invalid heads/device count")
    base, extra = divmod(heads, n)
    return [base + (i < extra) for i in range(n)]


def weighted_counts(heads: int, weights: list[float]) -> list[int]:
    """Integer proportional split with at least one head per device."""
    n = len(weights)
    if heads < n or n < 1:
        raise ValueError("not enough heads for devices")
    weights = [max(1e-6, float(w)) for w in weights]
    total = sum(weights)
    raw = [heads * w / total for w in weights]
    counts = [int(x) for x in raw]
    left = heads - sum(counts)
    order = sorted(range(n), key=lambda i: (raw[i] - counts[i], weights[i]), reverse=True)
    for i in order[:left]:
        counts[i] += 1

    # Keep every participating device alive. In normal H3 use (56 heads, 2 GPUs)
    # this is never needed, but it makes the helper robust to extreme weights.
    for i in range(n):
        if counts[i] > 0:
            continue
        donor = max(range(n), key=lambda j: counts[j])
        if counts[donor] <= 1:
            raise ValueError("cannot assign at least one head per device")
        counts[donor] -= 1
        counts[i] = 1
    return counts


def counts_to_ranges(counts: list[int]) -> list[tuple[int, int]]:
    out = []
    start = 0
    for count in counts:
        out.append((start, start + int(count)))
        start += int(count)
    return out
