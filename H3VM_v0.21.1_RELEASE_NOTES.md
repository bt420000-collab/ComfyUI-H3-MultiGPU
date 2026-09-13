# H3VM v0.21.1 Release Notes

v0.21.1 is a product-surface cleanup and compatibility release: fewer modes, stronger isolation, a focused Core engine, new Fast-mode prediction choices, and adjustable secondary-GPU participation.

## Main changes

1. **Improved compatibility with newer ComfyUI multi-GPU startup behavior.** Dual-GPU users should launch with `--cuda-device all`; H3VM does not require additional plugin-specific launch flags as long as both target CUDA devices remain visible.
2. **Independent / lazy runtime isolation.** Installing H3VM no longer means unrelated workflows should inherit its runtime behavior. H3VM activates when its execution path is actually used.
3. **Product modes reduced to three practical dual-GPU modes:** Fast, Capacity and Background. Single GPU remains available by disabling multi-GPU support.
4. **Fast mode adds multiple predictors.** SPECTRAL is the new default fast predictor; LINEAR remains available for content-dependent compatibility and A/B testing.
5. **Adjustable secondary-GPU participation.** 100% / 75% / 50% / 25% / custom 1–100% supports equal and heterogeneous GPU combinations.
6. **Removed the compatibility-heavy all-in-one Master Loader from the public surface.** H3VM is now a focused Loader-replacement execution layer: existing Model Loader -> H3VM Core -> existing workflow.

## Under the hood

- Mode4 run boundaries are owned by ComfyUI `OUTER_SAMPLE`; `expected_steps_hint` is scheduling guidance only.
- Predictor startup avoids committing temporary step-index coordinate history before a real timestep is available.
- Dual Video VAE adds best-effort same-weight identity validation.
- Bundled MiniMax H3 reference workflow is standardized at 20 Steps / Hint 20.
- Newer ComfyUI heterogeneous-GPU visibility and startup preflight fixes are retained.
- Public node surface remains two nodes: H3VM Core + optional H3VM Video VAE.

## Recommended first run

- Start ComfyUI with `--cuda-device all` for dual-GPU use.
- Insert H3VM Core after the existing Model Loader.
- Use **Multi-GPU Fast + SPECTRAL + 100% participation** first.
- Use Capacity for VRAM-limited workloads and Background when the PC also needs to remain responsive for browsing, editing or other foreground work.
