# H3 VRAM Master for ComfyUI

**H3VM = H3 VRAM Master**

[中文说明](README.md) | **English**

H3 VRAM Master is a **VRAM and multi-GPU execution manager for local MiniMax H3 generation**.

It is not just a dual-GPU loader, and its goal is not to make GPU1 look busy.

Its job is to organize the compute, VRAM, and host memory already available in a normal workstation, then choose the most appropriate execution strategy for the current H3 workload.

> **You choose the shot. H3VM decides how the machine runs it.**

---

## Why H3VM exists

Local H3 workloads often have unused hardware rather than no hardware:

- one GPU is saturated while another is nearly idle;
- two cards have enough combined VRAM, but the model still has to fit on one card;
- cross-GPU traffic can cost more than the compute it was meant to save;
- many Windows consumer systems have no NVLink or CUDA P2P;
- a fixed 50/50 split makes little sense for mixed-VRAM or asymmetric GPU pairs.

H3VM is built around one practical question:

> **What is the best way to run this workload on this machine?**

---

## Five product modes

The UI should describe the user goal first. Research names belong in logs and advanced information.

| Product mode | User goal | Internal name |
|---|---|---|
| **Single GPU** | compatibility and baseline | `SINGLE_GPU` |
| **Dual-GPU Cooperative** | balanced daily multi-GPU use | `DUAL_QUIET` |
| **Dual-GPU Capacity** | make workloads fit when one GPU cannot | `DUAL_CAPACITY` |
| **Dual-GPU Fast** | prioritize generation speed | `DUAL_SYNC_ACCEL / Mode4` |
| **Dual-GPU Native** | exact cooperative model execution | `DUAL_EXACT_SP` |

### Single GPU

The compatibility, quality, performance, and VRAM baseline.

### Dual-GPU Cooperative

The general-purpose daily path. Two GPUs share useful work while H3VM balances wall time, VRAM pressure, and stability.

### Dual-GPU Capacity

The success criterion is simple:

> **A workload that OOMs on one card can complete using both GPUs and host memory.**

### Dual-GPU Fast

The speed-first path. Different stages can advance out of phase so the secondary GPU prepares useful future work instead of waiting for the primary GPU.

This path may use prediction / approximate execution and is intended for fast auditions, batch generation, and throughput-sensitive work.

### Dual-GPU Native

Both GPUs cooperate on the current exact model computation instead of predicting future state.

This is the Exact-SP / native parallel direction and remains subject to strict parity and real-H3 promotion gates.

---

## No NVLink required

Windows consumer GPUs without CUDA P2P are a first-class target, not an edge case.

When direct GPU communication is available, H3VM can use it. When it is not, H3VM can relay data through host memory and choose a transport strategy appropriate to the measured hardware path.

The user should not need to become a PCIe topology expert just to use two GPUs.

---

## H3VM manages memory, not only GPU count

The runtime is designed to coordinate:

- model residency;
- dynamic VRAM loading and eviction;
- RAM backing;
- quantized partial computation;
- cross-GPU transport;
- per-device reserve and hot-cache policy;
- dual-GPU Video VAE work;
- fail-safe fallback paths.

The real question is not merely “How do I use two GPUs?”

It is:

> **How should the whole machine serve H3?**

---

## Two ways to use H3VM

### H3 VRAM Master Loader

The integrated controller for direct generation and quick A/B testing.

It can own common generation controls together with H3VM execution strategy.

### H3 VRAM Master Core

A narrow execution boundary for existing production workflows:

```text
MODEL
  |
  v
H3VM Core
  |
  v
MODEL
```

The surrounding workflow keeps ownership of prompt, seed, resolution, sampler, sigmas, and actual sampling steps.

This allows multi-GPU execution without rebuilding the entire workflow around H3VM.

---

## Long-term direction: a hardware-aware Planner

H3VM should move away from permanent hard-coded tuning tables and toward a Planner that understands the machine.

The Planner can progressively consider:

- VRAM capacity;
- GPU compute scale;
- symmetry between devices;
- CUDA P2P availability;
- measured host-relay bandwidth;
- model / quantization format;
- resolution, duration, and step count;
- the current goal: speed, capacity, or exactness.

The goal is not to teach every user multi-GPU tuning.

> **The goal is for H3VM to learn the user's machine.**

---

## Development philosophy

H3VM is not a collection of multi-GPU buzzwords.

A new technique should enter the mainline only when it provides at least one measurable benefit:

1. **Speed** — lower real wall time.
2. **Capacity** — complete workloads that do not fit on one GPU.
3. **Quality / exactness** — more reliable computation at comparable cost.
4. **Compatibility** — make ordinary hardware configurations usable.
5. **Usability** — less workflow surgery, less manual tuning, clearer behavior.

**A busier GPU1 is not proof of an optimization.**

The long-term product and engineering charter is documented here:

> [H3VM_PRODUCT_CHARTER.md](H3VM_PRODUCT_CHARTER.md)

Implementation details may change quickly. The product goal should not drift with them.

---

## Installation

Place the plugin directory at:

```text
ComfyUI/custom_nodes/ComfyUI-H3-VRAM-Master
```

Install missing dependencies from `requirements.txt`, make sure the target CUDA devices are visible to the current ComfyUI/Python process, and restart ComfyUI.

When upgrading, remove or move aside older `ComfyUI-H3-MultiGPU` / H3VM plugin folders so duplicate runtime bridges and nodes cannot load together.

---

## Recommended validation order

Keep prompt, seed, model, LoRA, resolution, and other generation conditions fixed when comparing modes.

1. **Single GPU** — establish a baseline.
2. **Dual-GPU Cooperative** — verify general multi-GPU benefit.
3. **Dual-GPU Capacity** — validate near or beyond the single-GPU VRAM limit.
4. **Dual-GPU Fast** — compare speed against visual stability.
5. **Dual-GPU Native** — validate the exact dual-GPU path under its current preview contract.

For performance modes, the important metrics are **wall time and the primary critical path**.

For Capacity mode, the first success criterion is:

> **single GPU fails to fit, multi-GPU H3VM completes.**

---

## Project boundary

Public H3VM focuses on the execution layer:

- VRAM management
- multi-GPU compute
- hardware-aware planning
- host relay / transport
- Attention / QKV / MLP sharding
- Capacity
- Mode4
- Exact-SP
- Dual Video VAE
- telemetry

Higher-level production orchestration such as job queues, worker leases, project lifecycle, P1/P2/P3 scheduling, and artifact routing belongs outside the public runtime.

---

## Technical references

- [H3VM_PRODUCT_CHARTER.md](H3VM_PRODUCT_CHARTER.md) — product and engineering charter
- [H3VM_FUSION_ROADMAP.md](H3VM_FUSION_ROADMAP.md) — runtime fusion roadmap
- [BRAND_H3_VRAM_MASTER.md](BRAND_H3_VRAM_MASTER.md) — naming and public boundary
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — third-party technology and license notices

## Author

- GitHub: `bt420000-collab`
- Bilibili: https://space.bilibili.com/16826253

## License

See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
