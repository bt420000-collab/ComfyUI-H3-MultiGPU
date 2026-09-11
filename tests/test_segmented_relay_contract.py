from h3vm.segmented_relay import pipeline_chunk_plan

assert pipeline_chunk_plan(0, 32) == ()
assert pipeline_chunk_plan(10, 32) == (10,)
assert pipeline_chunk_plan(100, 32) == (32, 32, 32, 4)
assert sum(pipeline_chunk_plan(1025, 256)) == 1025

print("H3 VRAM Master segmented relay contract: OK")
