# H3 VRAM Master for ComfyUI

**H3VM = H3 VRAM Master**

[中文说明](README.md) | **English**

H3 VRAM Master is a **VRAM and multi-GPU execution manager for local MiniMax H3 generation**.

It is not just a dual-GPU loader, and its goal is not to make a second GPU look busy.

Its purpose is to organize the GPU compute, VRAM, and host memory already available in a normal workstation, then choose the most useful execution strategy for the current H3 workload.

> **You choose the shot. H3VM decides how the machine runs it.**

Current stable release: **v0.20.0-rc6**

---

## Why H3VM exists

Local H3 systems often have unused hardware rather than no hardware:

- one GPU is saturated while another is nearly idle;
- two cards have enough combined VRAM, but the model still has to fit on one card;
- cross-GPU traffic can cost more than the compute it was meant to save;
- many Windows consumer systems have no NVLink or CUDA P2P;
- a fixed split makes little sense for asymmetric GPU pairs.

H3VM is built around one practical question:

> **What is the best way to run this workload on this machine?**

---

## Current stable execution modes

### Single GPU

Compatibility and baseline mode.

Internal name: `SINGLE_GPU`

### Dual-GPU Cooperative

The daily-use path. Two GPUs share useful work while H3VM balances wall time, VRAM pressure, and stability.

Internal name: `DUAL_QUIET`

### Dual-GPU Capacity

Capacity-first mode for workloads that do not fit comfortably on one GPU.

Internal name: `DUAL_CAPACITY`

The first success criterion is simple:

> **single GPU does not fit, multi-GPU H3VM completes.**

### Dual-GPU Fast

The speed-first Mode4 / FullThrottle path. It may use snapshot and prediction-based execution so the two GPUs can advance useful work out of phase.

Internal name: `DUAL_SYNC_ACCEL`

> The development line also contains a **Dual-GPU Native / Exact-SP** direction. It remains a promotion candidate until it passes real-H3 parity, wall-time, VRAM, and transport gates.

---

## No NVLink required

Windows consumer GPUs without CUDA P2P are a first-class target.

When direct GPU communication is available, H3VM can use it. When it is not, H3VM can relay data through host memory.

The goal is not to turn every user into a PCIe topology expert.

> **Use the direct road when it exists. Use the host-memory road when it does not. Avoid traffic jams either way.**

---

## H3VM manages memory, not only GPU count

The runtime is intended to coordinate:

- model residency;
- dynamic VRAM loading and eviction;
- RAM backing;
- partial / quantized computation;
- cross-GPU transport;
- per-device reserve and cache policy;
- dual-GPU Video VAE work;
- safe fallback paths.

The real question is not merely “How do I use two GPUs?”

It is:

> **How should the whole machine serve H3?**

---

## Two ways to use H3VM

### H3VM Master Loader

An integrated controller for direct generation and quick A/B testing.

### H3VM Core

A narrow execution boundary for existing workflows:

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

The goal is simple:

> **You should not have to rebuild your workflow just to use multiple GPUs.**

---

## Development philosophy

H3VM is not a collection of multi-GPU buzzwords.

A new technique belongs in the mainline only when it provides at least one measurable benefit:

1. **Speed** — lower real wall time.
2. **Capacity** — complete workloads that do not fit on one GPU.
3. **Quality / exactness** — more reliable computation at comparable cost.
4. **Compatibility** — make ordinary hardware configurations usable.
5. **Usability** — less workflow surgery, less manual tuning, clearer behavior.

**A busier GPU1 is not proof of an optimization.**

The long-term product and engineering charter is here:

> [H3VM_PRODUCT_CHARTER.md](H3VM_PRODUCT_CHARTER.md)

Implementation details may change quickly. The product direction should not drift with them.

---

## Long-term direction: a hardware-aware Planner

H3VM should progressively replace fixed tuning tables with a Planner that understands the machine.

It can consider:

- VRAM capacity;
- GPU compute scale;
- symmetry between devices;
- CUDA P2P availability;
- measured host-relay bandwidth;
- model / quantization format;
- resolution, duration, and step count;
- the current goal: speed, capacity, or exactness.

> **The goal is not to teach users multi-GPU tuning. The goal is for H3VM to learn the user's machine.**

---

## Installation

Place the plugin directory under `ComfyUI/custom_nodes/`, install missing dependencies from `requirements.txt`, make sure the target CUDA devices are visible to the current ComfyUI/Python process, and restart ComfyUI.

When upgrading, move aside older H3VM plugin folders so duplicate nodes and runtime bridges cannot load together.

Identical GPU models are supported; H3VM cares about distinct CUDA logical devices, not different marketing names.

---

## Recommended validation

Keep prompt, seed, model, LoRA, resolution, and other generation conditions fixed while comparing modes.

For performance modes, measure **wall time and the primary critical path**.

For Capacity mode, the first success criterion is:

> **single GPU fails to fit, multi-GPU H3VM completes.**

---

## Public runtime boundary

Public H3VM focuses on the execution layer: VRAM management, multi-GPU compute, planning, host relay, Attention / QKV / MLP sharding, Capacity, Mode4, Exact-SP research, Dual Video VAE, and telemetry.

Higher-level production orchestration such as job queues, worker leases, project lifecycle, P1/P2/P3 scheduling, and artifact routing belongs outside the public runtime.

---

## Author

- GitHub: `bt420000-collab`
- Bilibili: https://space.bilibili.com/16826253

## License

See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
