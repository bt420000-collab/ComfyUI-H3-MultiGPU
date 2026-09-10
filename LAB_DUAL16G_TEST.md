# H3VM Dual-16G Lab Validation

Branch: `lab/dual16g-exact-sp-v1`

This lab targets two near-identical 16 GB CUDA GPUs on the public H3VM runtime.
It does **not** contain PM/task scheduling code and it does **not** change Mode4.
The stable `main` branch remains untouched while this matrix is being validated.

## What this lab changes

When `H3VM_DUAL16_SYMMETRIC_LAB=1` and the selected two GPUs are near-symmetric
(within 8% total VRAM and SM count), public `DUAL_QUIET` / `DUAL_CAPACITY`
execution policy changes from the old asymmetric 16G+8G assumptions to:

- H3 Attention target: `28 / 28` heads.
- MLP token-row starting split: `56 / 44` by default.
- Hidden local sweep range: primary fraction `0.50 .. 0.60`.
- Existing runtime memory/free-VRAM safety checks remain active.
- No public UI control is added yet.

The starting MLP split is intentionally **not** forced to 50/50. On a Windows
host-relay path, equal GPU compute capacity does not imply equal end-to-end work:
GPU1 pays staging/return overhead, so the best wall-time split may remain biased
toward GPU0.

## 0. Update the lab branch

From the public plugin folder:

```powershell
git fetch origin
git checkout lab/dual16g-exact-sp-v1
git pull origin lab/dual16g-exact-sp-v1
```

Do not merge this branch into `main` during testing.

## 1. Hardware probe

Use the same Python environment that runs ComfyUI:

```powershell
G:\ComfyUI-MiniMax-H3\python_embeded\python.exe .\scripts\dual16g_lab_probe.py
```

Expected for a true symmetric pair:

```text
"near_symmetric": true
"recommended_h3_heads": [28, 28]
"recommended_mlp_primary_fraction": 0.56
```

Record the GPU names, reported VRAM, SM count and P2P result. P2P may remain
false; this lab is designed to keep H3VM's host-aware transport path.

## 2. Real quantized-QKV parity gate

Before direct quantized QKV slicing is allowed into a production hot path, test
one real H3 Attention QKV layer against two quantized head shards.

Example:

```powershell
G:\ComfyUI-MiniMax-H3\python_embeded\python.exe .\scripts\qkv_quantized_parity.py `
  --comfy-root G:\ComfyUI-MiniMax-H3\ComfyUI `
  --model "G:\ComfyUI-MiniMax-H3\ComfyUI\models\diffusion_models\YOUR_H3_MODEL.safetensors" `
  --device cuda:0 `
  --sequence 64
```

Required result:

```text
PARITY: PASS
```

Also record `max_abs`, `mean_abs` and `max_rel`. If this gate fails, do not wire
`h3vm/quant_shard.py` into Capacity or future Exact-SP execution yet.

## 3. Baseline run, lab OFF

Start ComfyUI normally with both CUDA devices visible. Leave the lab flag unset:

```powershell
Remove-Item Env:H3VM_DUAL16_SYMMETRIC_LAB -ErrorAction SilentlyContinue
Remove-Item Env:H3VM_DUAL16_PRIMARY_FRACTION -ErrorAction SilentlyContinue
```

Run the same H3 workflow used for all later A/B tests. Use:

- identical model and LoRA;
- identical Prompt and Seed;
- identical resolution and frame count;
- identical sampler / sigmas / steps;
- `DUAL_QUIET` first;
- telemetry enabled.

Run once as warm-up, then record three warm runs. Use median wall time rather
than the fastest single run.

## 4. Symmetric 56/44 run

Before launching ComfyUI from the same PowerShell session:

```powershell
$env:H3VM_DUAL16_SYMMETRIC_LAB="1"
$env:H3VM_DUAL16_PRIMARY_FRACTION="0.56"
```

Start ComfyUI with the usual launcher. The console must contain:

```text
[H3VM DUAL16 LAB] symmetric policy ACTIVE | mode=DUAL_QUIET MLP=56/44 ATTN=28/28
```

Again run one warm-up plus three measured runs.

## 5. MLP ratio sweep

Keep Attention at `28/28` for this first matrix. Sweep only MLP row ownership so
communication overhead is not mixed with a second variable.

Recommended order:

```text
0.60  -> 60/40
0.58  -> 58/42
0.56  -> 56/44
0.54  -> 54/46
0.52  -> 52/48
0.50  -> 50/50
```

For each value, set the environment variable **before launching ComfyUI**:

```powershell
$env:H3VM_DUAL16_SYMMETRIC_LAB="1"
$env:H3VM_DUAL16_PRIMARY_FRACTION="0.54"
```

Restart ComfyUI for a clean comparison, warm once, then measure three runs.
Do not change Prompt/Seed/resolution/length/steps between ratios.

## 6. What to record

For each ratio record:

| Field | Why it matters |
| --- | --- |
| Median denoise / sampling wall time | Primary speed metric |
| End-to-end wall time | Detect VAE/loader overhead masking DiT gains |
| GPU0 / GPU1 peak dedicated VRAM | Safety margin |
| GPU0 / GPU1 utilization pattern | Detect idle helper / overloaded root |
| Attention root/stage/helper/return telemetry | Find relay tax |
| MLP root/helper/slack/stall telemetry | Find the correct split |
| OOM / fallback / disabled helper events | Reject unstable ratios |
| Output quality / audio result | Exact paths should not introduce quality drift |

A ratio is not a winner merely because GPU utilization looks balanced. The
winner is the fastest stable wall time with adequate VRAM headroom and unchanged
output semantics.

## 7. Capacity mode second

Only after `DUAL_QUIET` is stable, repeat a smaller matrix in `DUAL_CAPACITY`:

```text
0.56
0.54
0.52
0.50
```

In this lab Capacity also targets 28 helper heads. Capacity remains a
capacity-first exact mode, so a slower wall time can still be acceptable if it
materially lowers peak VRAM and permits a workload that otherwise OOMs.

## 8. Mode4 is a control, not part of this patch

`DUAL_SYNC_ACCEL` / Mode4 remains unchanged. Run the same stock Mode4 workload
only as an external performance reference. Do not interpret changes in this lab
as Mode4 changes.

## 9. Promotion gate

Do not expose a public compute-ratio control until all of these are true:

1. Existing rc6 contract CI stays green.
2. Quantized QKV real-model parity passes.
3. At least one symmetric policy beats or matches the old 16G+8G-derived public
   policy on dual 16 GB hardware without reducing stability.
4. The winning region is repeatable across more than one resolution / length.
5. Capacity and Quiet behavior remain clearly separated.
6. Mode4 behavior is unchanged.

## Future public manual tuning control

The lab already contains the arithmetic hook for manual head ratios, but it is
not exposed in the UI. After validation, the preferred public design is a small
human-readable selector rather than dozens of raw engineering knobs, for
example:

```text
AUTO | 自动
50/50 | 对称双卡
54/46 | 轻偏主卡
56/44 | 主卡优先
60/40 | 高通信开销
CUSTOM | 高级
```

The public control should translate into a validated policy for MLP rows and,
only if testing proves useful, discrete Attention head counts. It should never
blindly equate a 56/44 MLP split with a 56/44 Attention split because the two
operations have different communication costs.
