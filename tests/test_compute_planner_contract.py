from h3vm.compute_planner import (
    DeviceProfile,
    PairProfile,
    build_runtime_plan,
    parse_primary_fraction,
)

assert parse_primary_fraction("54/46") == 0.54
assert parse_primary_fraction("54") == 0.54
assert parse_primary_fraction("0.54") == 0.54
assert parse_primary_fraction("AUTO", default=0.56) == 0.56

pair16 = PairProfile(
    DeviceProfile("cuda:0", "RTX 5060 Ti", 16.0, 36),
    DeviceProfile("cuda:1", "RTX 5060 Ti", 16.0, 36),
    False, False, "win32",
)
plan = build_runtime_plan(pair16, backend="MODE4")
assert pair16.near_symmetric is True
assert (plan.mode4_primary_blocks, plan.mode4_secondary_blocks) == (25, 25)
assert plan.transport == "host_pageable"
assert plan.exact_sp_tier == "LAB_HOST_RELAY"

pair168 = PairProfile(
    DeviceProfile("cuda:0", "GPU 16G", 16.0, 36),
    DeviceProfile("cuda:1", "GPU 8G", 8.0, 24),
    False, False, "win32",
)
plan = build_runtime_plan(pair168, backend="MODE4")
assert pair168.near_symmetric is False
assert (plan.mode4_primary_blocks, plan.mode4_secondary_blocks) == (28, 22)

manual = build_runtime_plan(pair16, backend="MODE4", manual_ratio="60/40")
assert (manual.mode4_primary_blocks, manual.mode4_secondary_blocks) == (30, 20)
assert manual.manual_ratio is True

p2p = PairProfile(
    DeviceProfile("cuda:0", "GPU A", 24.0, 60),
    DeviceProfile("cuda:1", "GPU B", 24.0, 60),
    True, True, "linux",
)
exact = build_runtime_plan(p2p, backend="EXACT_SP")
assert exact.transport == "p2p"
assert exact.exact_sp_tier == "READY"
assert (exact.attention_primary_heads, exact.attention_secondary_heads) == (28, 28)

print("H3 VRAM Master compute planner contract: OK")
