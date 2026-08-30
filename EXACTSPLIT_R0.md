# H3VM ExactSplit R0 Research Contract

Status: research-only. This work must stay isolated from production Mode3/Mode4 until the exactness and speed gates pass.

## Goal

Build a sampler-agnostic two-GPU backend that hard-shards one H3 forward across two GPUs while preserving exact forward semantics.

The distributed layer MUST NOT know or infer diffusion schedule semantics. In particular it must not branch on:

- diffusion step index
- sigma history
- PDD E/P state
- refresh cadence
- previous-step features
- predictor state
- stale feature availability

`EEEEEEEE`, vanilla 20-step H3, Turbo, or another scheduler must all appear to ExactSplit as repeated independent calls to the same exact H3 forward contract.

## Non-negotiable invariants

1. Exact only. No stale features, prediction, temporal reuse, cache approximation, or cross-step speculation.
2. Weight sharding must be physical. Do not solve 16+16 GB by duplicating the full DiT on both GPUs.
3. The sampler/PDD layer remains above ExactSplit and does not need special integration.
4. Production Mode3 and Mode4 code paths remain untouched during R0.
5. No hidden BF16 dequantized full-weight copies.
6. Cross-GPU synchronization must be coarse enough that PCIe does not reproduce Mode3's fine-grained critical-path penalty.

## Existing assets to reuse

- `h3vm/attention_parallel.py`: exact attention head split and Windows host-relay transport experiments.
- `h3vm/tp_mlp.py`: exact quantized FC1 row sharding without BF16 materialization.
- `h3vm/global_memory.py`: transport probing and host-neutral transfer engine.
- `h3vm/capacity_mode.py`: existing capacity ownership and residency logic, to be mined for placement ideas only.

## R0 hypothesis

The likely viable architecture is not pure Ulysses and not Mode3-style fine-grained operator ping-pong.

Candidate:

- physically shard large weights across the two 16 GB GPUs;
- keep each shard resident;
- perform exact intra-forward parallel work;
- minimize synchronization count per transformer block;
- prefer reduce-scatter/all-gather style boundaries when mathematically valid;
- batch/merge communication so each transfer carries a large payload;
- overlap independent local compute with transport whenever Windows/WDDM permits it.

## Architecture families to test

### A. Fine TP baseline

Split QKV/MLP projections inside every block. This is closest to existing Mode3 and serves as the known-slow control.

Expected problem: too many synchronization points per block.

### B. Coarse block ownership

GPU0 and GPU1 own disjoint contiguous groups of complete blocks. Hidden state crosses devices only at ownership boundaries.

Expected benefit: excellent memory efficiency and very low transfer count.

Expected problem: without temporal/sequence microbatching, one GPU waits while the other computes.

### C. Coarse block ownership + sequence microbatch pipeline

Split the video token/temporal sequence into exact microbatches that can flow through two block stages while preserving attention correctness. This is only allowed where the attention dependency graph makes the split exact; no stale cross-microbatch KV is permitted.

Expected benefit: combines weight sharding with compute overlap.

Risk: full global self-attention can make naive sequence microbatching mathematically invalid. R0 must prove the exact dependency boundary before implementation.

### D. Hybrid exact TP

Use head/row sharding only for the expensive sublayers where compute/communication ratio is favorable, and use block ownership elsewhere. This may be the best Windows/no-P2P compromise.

## R0 gates

### Gate 1: transport envelope

Measure both directions for realistic activation payload sizes using:

- direct D2D fallback
- pageable host relay
- explicit pinned host relay

### Gate 2: dual-compute concurrency

Verify two independent GEMM/attention-class workloads actually overlap on both GPUs in one Python process under the current Windows driver.

### Gate 3: break-even model

For each candidate synchronization boundary, compute:

`useful parallel compute saved > transfer + synchronization overhead`

Reject any shard boundary that cannot beat single-GPU execution by at least 10% in microbench before integration.

### Gate 4: exactness

Compare single-GPU and ExactSplit forward outputs on identical inputs. R0 target:

- same shape/dtype
- finite outputs
- no semantic scheduling dependency
- numerical delta bounded to the expected quantized/floating execution-order tolerance

### Gate 5: PDD transparency

Run PDD `EEEEEEEE` through the same H3 forward entrypoint with ExactSplit enabled. ExactSplit must not inspect or mutate the E schedule.

## R0 deliverables

1. `research/exactsplit_probe.py` transport and concurrency probe.
2. Measured report from the user's 16 GB + 16 GB identical GPU machine.
3. From those measurements, select one architecture family for R1 runtime implementation.

## Stop conditions

Do not integrate into the main loader if any of the following is true:

- exactness requires stale/predicted features;
- full model duplication becomes necessary;
- transport occupies the majority of the critical path;
- the selected design needs PDD/sampler-specific branching;
- the implementation regresses existing Mode3/Mode4 behavior.
