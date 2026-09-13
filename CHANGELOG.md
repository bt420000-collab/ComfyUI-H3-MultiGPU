# Changelog

## v0.21.1

### User-facing changes

- Improved newer-ComfyUI dual-GPU compatibility and startup preflight; dual-GPU launch target is `--cuda-device all`.
- Added lazy / isolated H3VM runtime activation to reduce cross-workflow interference.
- Simplified public dual-GPU modes to **Fast / Capacity / Background**; disabling multi-GPU returns to Single GPU.
- Added **SPECTRAL** fast prediction alongside **LINEAR** for content-dependent Fast-mode behavior.
- Added adjustable secondary-GPU participation: **100% / 75% / 50% / 25% / custom 1–100%**.
- Removed the compatibility-heavy all-in-one Master Loader from the public surface; H3VM Core is now the focused Model-Loader replacement engine.
- Public surface: **H3VM Core + H3VM Video VAE**.

### Stability and release hardening

- Mode4 sampling boundaries now follow ComfyUI `OUTER_SAMPLE`; Steps Hint is no longer a correctness/reset boundary.
- Predictor startup avoids temporary coordinate-source history when real timestep is initially unavailable.
- Dual Video VAE adds best-effort primary/secondary VAE identity validation.
- Reference workflow standardized to **20 Steps / Hint 20**.
- Retains heterogeneous-GPU visibility / preflight fixes for newer ComfyUI startup behavior.

## v0.20.0-rc6 and earlier

Historical development releases established Windows consumer dual-GPU support, host-relay execution, Capacity mode, Mode4 acceleration, Core integration and multi-GPU compatibility guards.
