# H3 VRAM Master (H3VM)

Public product name: **ComfyUI-H3-VRAM-Master**

Internal/runtime acronym: **H3VM = H3 VRAM Master**

The public repository may keep its historical GitHub slug during lab development. The product and plugin/package name is **ComfyUI-H3-VRAM-Master**.

## Product sentence

> **You choose the shot. H3VM decides how the machine runs it.**

H3 VRAM Master is the VRAM and multi-GPU execution manager for local MiniMax H3 generation.

Its purpose is not to collect multi-GPU techniques. Its purpose is to organize the GPU compute, VRAM, and host memory available in a machine so H3 can run in the most useful way for the current workload.

## Stable product goals

Every user-facing mode should correspond to a durable user goal rather than a research term:

- **Single GPU** — compatibility and baseline.
- **Dual-GPU Cooperative** — balanced daily multi-GPU execution.
- **Dual-GPU Capacity** — make workloads fit when one GPU cannot.
- **Dual-GPU Fast** — prioritize generation speed.
- **Dual-GPU Native** — exact cooperative model execution.

Internal implementations may change without forcing a new product mode.

Examples:

- Exact-SP is an implementation of **Dual-GPU Native**.
- Snapshot / Predictor is an implementation of **Dual-GPU Fast**.
- Host Relay is an implementation detail of cross-GPU transport.
- Quantized QKV sharding is an implementation detail of memory/compute reduction.

## Product boundary

H3VM is the public execution/runtime layer for MiniMax H3. It may contain:

- VRAM placement and dynamic residency
- multi-GPU compute
- hardware-aware planning
- host relay and transport
- attention / QKV / MLP sharding
- sequence partitioning
- Capacity
- Mode4
- Exact-SP / native parallel work
- dual-GPU Video VAE
- telemetry and safe fallback logic

Private PM/task orchestration remains outside this repository and must not be imported as part of the runtime or branding.

## Development authority

The long-term product and engineering authority for this project is:

> [H3VM_PRODUCT_CHARTER.md](H3VM_PRODUCT_CHARTER.md)

When a new implementation conflicts with the charter, the implementation changes. The charter does not drift merely because a new technique appeared.
