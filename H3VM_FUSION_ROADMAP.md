# H3 VRAM Master Fusion Roadmap

This roadmap evolves the public H3VM runtime without importing private PM/task orchestration.

The roadmap is subordinate to the product and engineering charter:

> [H3VM_PRODUCT_CHARTER.md](H3VM_PRODUCT_CHARTER.md)

The stable objective is not “support more multi-GPU techniques.”

The stable objective is:

> **Organize the compute, VRAM, and host memory available in a normal machine so MiniMax H3 can complete the current workload in the most useful way.**

## Runtime layers

1. **Planner**
   - One hardware/capability description per run.
   - One explicit execution plan per backend.
   - Increasingly hardware-aware rather than profile-table driven.

2. **Memory / transport**
   - VRAM placement and dynamic residency.
   - RAM backing.
   - Host relay / P2P transport.
   - Transfer telemetry and measured-bandwidth policy.

3. **Compute primitives**
   - Sequence partition.
   - Quantized QKV head shard.
   - Token-row shard.
   - Exact exchange / redistribution primitives.

4. **Product backends**
   - **Single GPU**: compatibility baseline.
   - **Dual-GPU Cooperative**: balanced daily execution.
   - **Dual-GPU Capacity**: capacity-first exact execution.
   - **Dual-GPU Fast**: speed-first speculative / staggered execution such as Mode4.
   - **Dual-GPU Native**: exact sequence/head-parallel execution such as Exact-SP.

5. **Postprocess**
   - Dual-GPU temporal Video VAE.
   - Future postprocess work must still satisfy measurable product benefit.

## Promotion rules

A lab technique is promoted only when it satisfies the charter and provides measurable user value.

### Required before promotion

- Real MiniMax H3 workload, not synthetic-only success.
- Clear benefit in at least one dimension: speed, capacity, quality/exactness, compatibility, or usability.
- Tensor / output correctness gate appropriate to the backend.
- Wall-time measurement for performance claims.
- VRAM / transport measurement when memory or relay behavior changes.
- Safe fallback or fail-closed behavior.
- UI wording that describes the user goal rather than exposing unnecessary research terminology.
- Documentation that matches actual runtime behavior.

### Explicit non-goals

- “GPU1 is busier” is not a promotion criterion.
- A new paper / algorithm does not automatically deserve a new public mode.
- Existing Mode4 behavior is not silently replaced by Exact-SP.
- Unsupported LoRA / hooks / patches are not silently ignored.
- Public H3VM does not absorb private PM, project lifecycle, lease, job queue, P1/P2/P3 scheduling, or artifact routing.

## Planner direction

The long-term Planner should progressively consider:

- GPU VRAM capacity
- GPU compute scale
- device symmetry
- CUDA P2P availability
- measured host-relay bandwidth
- quantization / model format
- resolution, duration, and step count
- requested product goal: compatibility, cooperation, capacity, speed, or exactness

The goal is not to expose an ever-growing tuning matrix.

The goal is:

> **Let H3VM learn the user's machine.**

## CPU control-plane direction

For the next generation of speed-oriented execution, CPU work should not be treated as mere launch overhead.

A CPU control plane may handle:

- observation
- estimation
- policy selection
- prediction quality / drift checks
- correction decisions
- transfer selection
- relay compression / coding
- safe fallback

The stable division of labor is:

> **GPUs perform large-scale parallel arithmetic; the CPU decides what is worth computing and moving.**

## UI policy

Technical names remain available in logs and developer documentation.

The normal UI should prefer durable product language:

- Single GPU
- Dual-GPU Cooperative
- Dual-GPU Capacity
- Dual-GPU Fast
- Dual-GPU Native

A better future implementation should replace the backend behind one of these goals rather than automatically adding another research-named mode.
