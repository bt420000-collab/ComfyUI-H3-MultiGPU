# H3 VRAM Master Fusion Roadmap

This lab evolves the public H3VM runtime without importing private PM/task orchestration.

## Runtime layers

1. **Planner**: one device/capability description and one compute plan per run.
2. **Memory/transport**: VRAM placement, RAM backing, host relay, transfer telemetry.
3. **Compute primitives**: sequence partition, quantized QKV head shard, token-row shard.
4. **Backends**:
   - Single: compatibility baseline.
   - Capacity: exact, capacity-first execution.
   - Mode4: speculative Snapshot Islands speed backend.
   - Exact-SP Lab: exact sequence/head parallel backend under validation.
5. **Postprocess**: dual-GPU temporal Video VAE.

## Promotion rules

- No private PM, project lifecycle, lease, job queue, P1/P2/P3 scheduling, or artifact routing in public runtime.
- Existing Mode4 behavior is not silently replaced by Exact-SP.
- Every exact-path change must pass tensor parity before production promotion.
- User-facing compute-ratio controls are added only after validated ranges exist on real hardware.
- The public branch remains derived from the public GitHub base. Local production code is reference material only.
