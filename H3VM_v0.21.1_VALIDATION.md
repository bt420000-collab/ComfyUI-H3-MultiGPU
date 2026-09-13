# H3VM v0.21.1 Validation

Release contract checks:

- Public nodes: H3VM Core + H3VM Video VAE only.
- H3VM Core owns the Dual-GPU VAE switch and emits `H3VM_VAE_POLICY`.
- H3VM Video VAE consumes Core policy; no duplicate VAE acceleration switch exists on the VAE node.
- Policy OFF and single-GPU Core mode use native `vae.decode()`.
- Mode4 registers an `OUTER_SAMPLE` lifecycle wrapper; begin/end reset runtime state independently of `expected_steps_hint`.
- A mismatched steps hint cannot reset Mode4 or exact runtimes in the middle of a real Sampler run.
- Consecutive same-shape prompts therefore do not rely on tensor-shape changes to clear stale mailbox/predictor state.
- Predictor timestep startup keeps coordinate history uncommitted until a real timestep is available instead of creating a normal `step_index -> timestep` reset cycle.
- Dual Video VAE rejects an obvious same-architecture / different-weight mismatch when lightweight signatures are available.
- UI localization is label-only: no widget reordering / serialization override.
- Plugin discovery remains side-effect free; H3VM runtime activation is lazy.
- Final reference workflow wires Core mode + VAE policy into H3VM Video VAE, uses BasicScheduler=20, and aligns `expected_steps_hint=20`.
- Static/API compatibility baseline: ComfyUI v0.35.0 interfaces.

Automated release regression tests cover the lifecycle/hint boundary, predictor coordinate startup, wrapper registration and dual-VAE identity guard.

Hardware runtime validation still requires an actual ComfyUI run on target GPUs. Recommended smoke matrix before publishing a hardware-validated build:

1. Mode4: standard 20-step reference workflow, run two consecutive same-shape prompts.
2. Mode4: 4 / 8 / 20-step runs with matching and deliberately mismatched hints.
3. Predictor: verify no normal first-call `step_index -> timestep` warning pair.
4. Video VAE: dual decode with matching weights; verify mismatched weights fail clearly.
5. Compiler ON/OFF A/B on the target ComfyUI build and GPU pair.
