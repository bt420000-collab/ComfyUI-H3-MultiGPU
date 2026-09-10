from h3vm.compute_planner import DeviceProfile, PairProfile
from h3vm.exact_sp_lab import build_exact_sp_plan, estimate_exchange_bytes

pair = PairProfile(
    DeviceProfile("cuda:0", "GPU0", 16.0, 36),
    DeviceProfile("cuda:1", "GPU1", 16.0, 36),
    False, False, "win32",
)

hb, ab, total = estimate_exchange_bytes(
    seq_len=100,
    hidden_dim=64,
    heads=8,
    head_dim=8,
    element_size=2,
    primary_fraction=0.50,
)
assert hb == 100 * 64 * 2
assert ab == 100 * 8 * 8 * 2
assert total == hb + ab

plan = build_exact_sp_plan(
    pair,
    seq_len=101,
    hidden_dim=5376,
    heads=56,
    head_dim=128,
    element_size=2,
)
assert plan.sequence.counts == (51, 50)
assert plan.head_counts == (28, 28)
assert plan.recommended is False
assert "Lab" in plan.recommendation_reason

fast_host = build_exact_sp_plan(
    pair,
    seq_len=101,
    hidden_dim=5376,
    heads=56,
    head_dim=128,
    element_size=2,
    host_relay_gbps=14.0,
)
assert fast_host.recommended is True

print("H3 VRAM Master Exact-SP lab contract: OK")
