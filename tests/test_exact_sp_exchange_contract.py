from h3vm.exact_sp_exchange import ExactSPExchangeStats

s = ExactSPExchangeStats(hidden_calls=2, head_calls=3, bytes_moved=2_000_000_000, seconds=1.0)
r = s.as_dict()
assert r["hidden_calls"] == 2
assert r["head_calls"] == 3
assert r["bytes_moved"] == 2_000_000_000
assert abs(r["effective_gbps"] - 2.0) < 1e-12

print("H3 VRAM Master Exact-SP exchange contract: OK")
