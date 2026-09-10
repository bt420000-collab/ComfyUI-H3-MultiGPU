from __future__ import annotations

import importlib.util
import os
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

quiet = MOD.symmetric_policy_overrides("DUAL_QUIET", primary_fraction=0.56)
assert quiet["attention_head_balance"] == "balanced"
assert quiet["attention_helper_head_cap"] == 28
assert quiet["mlp_primary_fraction"] == 0.56
assert quiet["critical_path_min_primary_fraction"] == 0.52
assert quiet["critical_path_max_primary_fraction"] == 0.62

capacity = MOD.symmetric_policy_overrides("DUAL_CAPACITY", primary_fraction=0.54)
assert capacity["capacity_helper_heads"] == 28
assert capacity["mlp_primary_fraction"] == 0.54
assert capacity["critical_path_min_primary_fraction"] == 0.54
assert capacity["critical_path_max_primary_fraction"] == 0.54

old = os.environ.get("H3VM_DUAL16_PRIMARY_FRACTION")
try:
    os.environ["H3VM_DUAL16_PRIMARY_FRACTION"] = "0.50"
    assert MOD.requested_primary_fraction() == 0.50
    os.environ["H3VM_DUAL16_PRIMARY_FRACTION"] = "0.60"
    assert MOD.requested_primary_fraction() == 0.60
    os.environ["H3VM_DUAL16_PRIMARY_FRACTION"] = "0.61"
    try:
        MOD.requested_primary_fraction()
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range lab fraction must fail")
finally:
    if old is None:
        os.environ.pop("H3VM_DUAL16_PRIMARY_FRACTION", None)
    else:
        os.environ["H3VM_DUAL16_PRIMARY_FRACTION"] = old

for bad in (0.0, 1.0, -0.1, 1.1):
    try:
        MOD.manual_dual_head_counts(56, bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid manual fraction must fail: {bad}")

print("dual16g planner contract: OK")
