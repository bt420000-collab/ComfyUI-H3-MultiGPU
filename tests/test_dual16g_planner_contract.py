from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h3vm_dual16g_lab_contract",
    ROOT / "h3vm" / "dual16g_lab.py",
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

assert MOD.near_symmetric_values(16.0, 16.0)
assert MOD.near_symmetric_values(16.0, 15.5)
assert not MOD.near_symmetric_values(16.0, 12.0)

assert MOD.symmetric_head_counts(56) == [28, 28]
assert MOD.symmetric_head_counts(55) == [28, 27]

assert MOD.manual_dual_head_counts(56, 0.50) == [28, 28]
assert MOD.manual_dual_head_counts(56, 0.60) == [34, 22]
assert MOD.manual_dual_head_counts(56, 0.40) == [22, 34]

for bad in (0.0, 1.0, -0.1, 1.1):
    try:
        MOD.manual_dual_head_counts(56, bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid manual fraction must fail: {bad}")

print("dual16g planner contract: OK")
