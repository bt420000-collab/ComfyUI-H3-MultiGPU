# H3VM LAB: RAM-Native Compute v1

**Status: standalone feasibility test; not merged into the public H3VM runtime.**
**Goal: single-GPU, Windows-native, CUDA-mapped RAM computation for a future long-video VRAM-saving backend.**

## Hypothesis

Keep an H3-like weight matrix physically resident in system RAM. Pass a
CUDA device-addressable pointer to a GPU kernel, while input and output activations
remain in VRAM. The kernel reads weight data as needed over PCIe without an
explicit host-to-device upload of the full weight matrix.

This is not memory bandwidth magic: GPU accesses to remote host memory still
use PCIe. It is a different memory residency policy, not a substitute for GDDR.
The experiment is primarily about capacity and correct execution, not speed.

## First lab delivered

- bench_mapped_gemm.py: standalone CUDA Python + NVRTC tile-GEMM benchmark.
- The same simple FP32 GEMM kernel executes against three weight-storage modes:
  1. mapped_ram: W in mapped pinned system RAM; no separate VRAM W allocation.
  2. vram_resident: W copied once into VRAM; kernel reads resident W.
  3. copy_each_iter: W uploaded to VRAM for each iteration, then kernel runs.
- X and Y always stay on the GPU; the weight pointer is the only variable.
- Reports milliseconds per iteration, an estimate of weight-allocation impact on
  VRAM, numeric parity and a JSON summary.
- A small default benchmark and a 256 MiB pinned-RAM safety cap avoid locking
  very large sections of host memory accidentally.
- No H3 model, H3 sampler, quantized kernel, or ComfyUI runtime is modified.

## Windows instructions

Use a separate venv to avoid changing production ComfyUI dependencies:

~~~powershell
cd <your-clone-of-ComfyUI-H3-MultiGPU>
git fetch origin
git switch lab/ram-native-compute-v1

py -3.13 -m venv .venv-ramlab
.\.venv-ramlab\Scripts\python.exe -m pip install -r labs\ram-native-compute\requirements.txt

.\.venv-ramlab\Scripts\python.exe labs\ram-native-compute\bench_mapped_gemm.py
.\.venv-ramlab\Scripts\python.exe labs\ram-native-compute\bench_mapped_gemm.py --m 5376 --k 5376 --n 64 --iters 4
~~~

Requires an NVIDIA CUDA-capable GPU, a compatible official driver, supported
CUDA 13 CUDA Python wheels, and the CUDA runtime/compiler libraries required by
cuda.core.Program (NVRTC). If compilation fails because NVRTC is missing,
install the matching official CUDA Toolkit; do not patch the display driver.

Default mapped W is 16 MiB (2048 x 2048, FP32). The 5376 x 5376 experiment
uses 110.25 MiB pinned RAM for W, still under the default limit.

Do not run on a machine with insufficient available system RAM. Pinned
host allocations cannot be freely paged out and must be bounded. The script frees
both host and GPU memory in finally. Do not concurrently run the production
H3VM/ComfyUI workload during the first benchmark.

## Interpretation / pass criteria

- P0 functional pass: CUDA kernel reads mapped host W and produces the same
  result as an identical kernel reading W from device VRAM.
- P0 residency pass: mapped path does not allocate a second full-sized VRAM
  W buffer. X, Y, CUDA contexts, compilation caches and kernel scratch remain
  real VRAM costs.
- P0 speed observation: record time for all three modes. There is no
  speed threshold for a capacity-first prototype.
- Not proven by P0: compatibility with H3 INT8/FP8 ConvRot Linear, Sage,
  LoRA, cuBLAS, torch Tensor wrapping, DynamicVRAM, long-video activations, or
  whole-H3 generation. The tiled FP32 kernel is deliberately simple.

The metric to optimize later is maximum complete video length at fixed VRAM,
with acceptable wall time and correct H3 outputs, not peak memory bandwidth.

## Suggested next experiments

1. P1: one real H3 Linear. Identify a production H3 fc1 or fc2
   operation, compare native quantized kernel requirements to a mapped-memory
   pointer. Introduce an experimental backend only after parity tests.
2. P2: bounded per-block residency. Reserve dynamic VRAM for activations;
   keep selected cold weights mapped in host RAM, with optional device cache for
   frequently reused sub-tiles. No hidden whole-weight copy.
3. P3: long-video capacity. Combine with QKV/MLP activation chunking and
   shape-aware VRAM budgets. Compare completed frame counts and end-to-end time.
4. P4: optional CPU calculations. Only for tasks not on the GPU critical path;
   profile host memory bandwidth contention with mapped CUDA loads.
5. Preserve existing H3VM Core and stable single-/multi-GPU algorithms.
   The mapped backend stays opt-in until measured.

## Platform constraints and references

- Uses cudaHostAllocMapped + cudaHostGetDevicePointer from unmodified CUDA.
- CUDA mapped host memory is page-locked and accessed over the CPU-GPU
  interconnect on a discrete GPU. No P2P, iGPU, WSL or driver patch needed.
- Windows WDDM shared GPU memory and a CUDA-mapped host allocation are not
  equivalent to Spark hardware-coherent SoC unified memory.
- This lab does not modify WDDM or attempt driver feature unlocking.

References:
- NVIDIA CUDA Memory Guide:
  https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html
- NVIDIA CUDA Python simpleZeroCopy sample (API reference, not copied):
  https://github.com/NVIDIA/cuda-samples/tree/master/python/2_CoreConcepts/simpleZeroCopy
- H3VM product charter: ../../H3VM_PRODUCT_CHARTER.md
