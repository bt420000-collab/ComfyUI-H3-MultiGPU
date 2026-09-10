# H3 VRAM Master (H3VM)

Public product name: **ComfyUI-H3-VRAM-Master**

Internal/runtime acronym: **H3VM = H3 VRAM Master**

The public repository remains on its existing GitHub slug during lab development. The plugin/package folder for the next release should be `ComfyUI-H3-VRAM-Master`.

## Product boundary

H3VM is the public execution/runtime layer for MiniMax H3. It may contain VRAM placement, multi-GPU compute, host relay, attention/QKV sharding, sequence partitioning, Capacity, Mode4, exact/SP experiments, and dual-GPU VAE work.

Private PM/task orchestration remains outside this repository and must not be imported as part of the rebrand or runtime fusion.
