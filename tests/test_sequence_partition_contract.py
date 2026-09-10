from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h3vm_sequence_partition_contract",
    ROOT / "h3vm" / "sequence_partition.py",
)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)
SequencePartition = MOD.SequencePartition


p = SequencePartition.balanced(101, 2)
assert p.counts == (51, 50)
assert p.bounds(0) == (0, 51)
assert p.bounds(1) == (51, 101)
assert p.peer(0) == 1 and p.peer(1) == 0

p = SequencePartition.dual_ratio(100, 0.60)
assert p.counts == (60, 40)

p = SequencePartition.dual_ratio(103, 0.50, alignment=8)
assert sum(p.counts) == 103
assert p.counts[0] == 48

segments = [
    (0, 10, "text"),
    (10, 70, "video"),
    (70, 101, "audio"),
]
p = SequencePartition.balanced(101, 2)
assert p.localize_segments(segments, 0) == [
    (0, 10, "text"),
    (10, 51, "video"),
]
assert p.localize_segments(segments, 1) == [
    (0, 19, "video"),
    (19, 50, "audio"),
]

try:
    SequencePartition(10, (4, 5))
except ValueError:
    pass
else:
    raise AssertionError("invalid partition sum must fail")

print("sequence partition contract: OK")