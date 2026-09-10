# Local dual-16G reference boundary

The uploaded local package is a technical reference only. Public development continues from the GitHub public baseline.

Validated facts extracted from the local package that may be used as public engineering evidence:

- Local package version string: `0.20.2-rc3-production-dual16-mode4-win-pageable-ck-dlpack-guard.mode3-split-lab1`.
- Public-facing Mode4/Stock FullThrottle local tuning uses a symmetric `25/25` Snapshot-Islands split with `secondary_blocks_target=25`.
- Predictor remains linear with beta `0.75`, exact-last-step enabled, 2 GiB secondary RAM backing.
- Windows multi-GPU path includes an explicit pageable-host safety policy instead of assuming pinned host transfers are safe.
- The local package still contains historical asymmetric 16G+8G experiments in Quiet/Capacity/older Lab paths. Those settings are not to be treated as the validated dual-16G answer.

Private/project-specific orchestration, PM behavior, project routing, pass scheduling, or other local-only product wiring must not be copied into the public repository.

Development rule for this branch:

1. GitHub public `main` remains the implementation base.
2. Local code is evidence/reference, not a source tree to merge wholesale.
3. Only isolated, generally useful runtime improvements are eligible for public-porting.
4. Any public change must be independently reviewed, tested, and attributable to public-compatible code/ideas.
