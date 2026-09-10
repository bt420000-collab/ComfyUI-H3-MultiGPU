from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h3vm_quant_shard_contract",
    ROOT / "h3vm" / "quant_shard.py",
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

ranges = MOD.qkv_head_row_ranges(56, 128, 0, 28)
assert ranges == (
    (0, 3584),
    (7168, 10752),
    (14336, 17920),
)

ranges = MOD.qkv_head_row_ranges(56, 128, 28, 56)
assert ranges == (
    (3584, 7168),
    (10752, 14336),
    (17920, 21504),
)

covered = []
for start, stop in MOD.qkv_head_row_ranges(56, 128, 7, 19):
    covered.append(stop - start)
assert covered == [1536, 1536, 1536]

for bad in [(-1, 2), (2, 2), (0, 57)]:
    try:
        MOD.qkv_head_row_ranges(56, 128, bad[0], bad[1])
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid head range must fail: {bad}")

print("quant shard contract: OK")
