from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dual16g_log_summary_contract",
    ROOT / "scripts" / "dual16g_log_summary.py",
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

sample = """
[H3VM DUAL16 LAB] symmetric policy ACTIVE | mode=DUAL_QUIET MLP=56/44 ATTN=28/28 | manual UI remains hidden
H3VM Dev14 CPM attn #1 | root-first transport=host_pinned heads=[28, 28] | root=10.00ms stage=4.00ms helper=9.00ms return=3.00ms
H3VM Dev14.1 CPM MLP #1 | block=0 rows(root/helper)=560/440 | root=12.0ms shadow=10.0ms [stage 1.0+1.0 compute 6.0 return 1.0+1.0] | slack=+2.0ms stall=0.0ms | root_fraction 0.56->0.56 action=hold
"""

out = MOD.summarize(sample)
assert out["policy_events"][0]["mlp"] == [56, 44]
assert out["policy_events"][0]["attention"] == [28, 28]
assert out["attention"]["samples"] == 1
assert out["attention"]["stage_ms"]["median"] == 4.0
assert out["mlp"]["samples"] == 1
assert out["mlp"]["slack_ms"]["median"] == 2.0
assert out["mlp"]["actions"] == {"hold": 1}
assert out["hint"] == "MLP split is near the measured relay/compute balance region"

print("dual16g log summary contract: OK")
