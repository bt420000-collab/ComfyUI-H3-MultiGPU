# H3 VRAM Master v0.21.1 for ComfyUI

**H3VM = H3 VRAM Master**  
[中文](README.md) | **English**

H3VM is a VRAM and multi-GPU execution engine for local **MiniMax H3** generation.

> **You keep your workflow. H3VM schedules the GPUs.**

v0.21.1 focuses on a smaller public surface, stronger isolation, newer ComfyUI compatibility, and more practical controls for heterogeneous GPU pairs.

## v0.21.1 highlights

### Newer ComfyUI multi-GPU compatibility

For dual-GPU use, start ComfyUI with both GPUs visible:

```bash
--cuda-device all
```

H3VM does not require a separate set of plugin-specific startup flags beyond making both target CUDA devices visible. Existing ComfyUI flags can generally remain as-is.

Identical and heterogeneous GPU pairs are supported; model names and VRAM sizes do not need to match.

### Isolated lazy runtime

H3VM runtime activates only when an H3VM execution path is used. Standard Prompt, Seed, resolution, duration, Sampler, Sigmas, actual Steps, CLIP/text encoder and normal ComfyUI workflow ownership remain unchanged.

### Three practical dual-GPU modes

- **Multi-GPU Fast**: speed-first execution.
- **Multi-GPU Capacity**: VRAM-first execution for workloads that may OOM on one card.
- **Multi-GPU Background**: leaves more GPU headroom for desktop, browser, editing and other foreground work.

Disable multi-GPU support to use the normal single-GPU path.

### Multiple predictors for Fast mode

- **SPECTRAL**: new default fast predictor with a higher performance ceiling.
- **LINEAR**: retained as a compatibility / A-B comparison route.

Different motion and scene content can use different prediction behavior instead of forcing one predictor onto every video.

### Adjustable secondary-GPU participation

Presets: `100% / 75% / 50% / 25% / custom 1–100%`.

This lets H3VM adapt to equal cards, asymmetric cards, 16 GB + 8 GB, 16 GB + 16 GB and other real-world combinations.

### Focused Loader-replacement engine

The compatibility-heavy all-in-one Master Loader is removed from the public surface. v0.21.1 exposes only:

- **H3VM Core** for MODEL execution / VRAM / multi-GPU scheduling.
- **H3VM Video VAE** for optional dual-GPU MiniMax H3 Video VAE decoding.

Connect H3VM Core after your existing Model Loader:

```text
Model Loader -> H3VM Core -> your existing sampler / workflow
```

Your original workflow keeps ownership of Prompt, Seed, resolution, Sampler and Steps.

## Quick start

1. Put this repository under `ComfyUI/custom_nodes/`.
2. For dual-GPU use, start ComfyUI with `--cuda-device all`.
3. Insert `H3VM Core` after your existing Model Loader.
4. Start with **Multi-GPU Fast + SPECTRAL + 100% secondary participation**.
5. Standard MiniMax H3: **20 Steps / H3VM Hint 20**.
6. Use Capacity when VRAM is the bottleneck; use Background when the same PC must stay responsive for other tasks.

## Under-the-hood stability work

- Mode4 run boundaries now follow the ComfyUI `OUTER_SAMPLE` lifecycle; Steps Hint is scheduling guidance only.
- Predictor startup no longer commits a temporary coordinate source before real timestep data is available.
- Dual Video VAE adds best-effort same-weight identity validation.
- The bundled reference workflow is aligned to **20 Steps / Hint 20**.
- Newer ComfyUI visibility/preflight fixes for heterogeneous GPU pairs are retained.

## Reference workflow

`example_workflows/01_H3VM_Core_Reference_Workflow.json`

## Compatibility baseline

v0.21.1 was release-checked against the current **ComfyUI v0.35.0** API surface.

Windows consumer dual-GPU systems without NVLink / CUDA P2P remain a primary H3VM target.

## Author

- GitHub: `bt420000-collab`
- Bilibili: https://space.bilibili.com/16826253

## License

See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
