#!/usr/bin/env python3
"""H3VM RAM-Native Compute LAB: mapped host weights, CUDA GEMM, no weight upload.

This is a standalone, correctness-first hardware experiment, NOT an H3 runtime.
The CUDA kernel is intentionally simple and is NOT a cuBLAS/INT8 benchmark.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import time

import numpy as np
from cuda.bindings import runtime as rt
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch


# This kernel receives an address that refers to HOST RAM for W in mapped mode.
# X and Y remain in actual GPU memory. It never copies the mapped W into VRAM.
KERNEL = r"""
extern "C" __global__
void mapped_weight_gemm(float* y, const float* w, const float* x,
                        int m, int k, int n) {
    __shared__ float wt[16][16];
    __shared__ float xt[16][16];
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float acc = 0.0f;
    for (int base = 0; base < k; base += 16) {
        int wk = base + threadIdx.x;
        int xk = base + threadIdx.y;
        wt[threadIdx.y][threadIdx.x] =
            (row < m && wk < k) ? w[(size_t)row * k + wk] : 0.0f;
        xt[threadIdx.y][threadIdx.x] =
            (col < n && xk < k) ? x[(size_t)xk * n + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            acc = fmaf(wt[threadIdx.y][t], xt[t][threadIdx.x], acc);
        }
        __syncthreads();
    }
    if (row < m && col < n) y[(size_t)row * n + col] = acc;
}
"""


def cuda(call, *args):
    result = call(*args)
    if result[0] != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{call.__name__} failed: {result[0]}")
    return result[1] if len(result) > 1 else None


def mem_free_bytes() -> int:
    return int(cuda(rt.cudaMemGetInfo)[0])


def copy_h2d(dst: int, src: int, size: int) -> None:
    cuda(rt.cudaMemcpy, int(dst), int(src), int(size),
         rt.cudaMemcpyKind.cudaMemcpyHostToDevice)


def copy_d2h(dst: int, src: int, size: int) -> None:
    cuda(rt.cudaMemcpy, int(dst), int(src), int(size),
         rt.cudaMemcpyKind.cudaMemcpyDeviceToHost)


def run(args) -> dict:
    if min(args.m, args.k, args.n, args.iters) <= 0:
        raise ValueError("m, k, n and iters must be positive")
    if max(args.m, args.k, args.n) > 32768:
        raise ValueError("max dimension is 32768 in this exploratory kernel")
    w_bytes = args.m * args.k * np.dtype(np.float32).itemsize
    if w_bytes > args.max_mapped_mb * 1024 * 1024:
        raise ValueError(
            f"Weight is {w_bytes / 1048576:.1f} MiB; exceeds the "
            f"{args.max_mapped_mb} MiB pinned-RAM safety cap"
        )
    x_bytes = args.k * args.n * 4
    y_bytes = args.m * args.n * 4
    device = Device(args.device)
    device.set_current()
    stream = device.create_stream()
    d_ptrs = []
    h_weight = None

    try:
        # Compiling first prevents first-use JIT time from skewing the results.
        prog = Program(
            KERNEL, code_type="c++",
            options=ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
        )
        kernel = prog.compile("cubin").get_kernel("mapped_weight_gemm")
        config = LaunchConfig(
            grid=((args.n + 15) // 16, (args.m + 15) // 16, 1),
            block=(16, 16, 1)
        )
        # cudaHostAllocMapped: RAM physically stays on host.
        h_weight = int(cuda(
            rt.cudaHostAlloc, w_bytes,
            rt.cudaHostAllocMapped | rt.cudaHostAllocPortable
        ))
        d_weight_mapped = int(cuda(rt.cudaHostGetDevicePointer, h_weight, 0))
        weight = np.ctypeslib.as_array(
            (ctypes.c_float * (args.m * args.k)).from_address(h_weight)
        )
        rng = np.random.default_rng(42)
        weight[:] = rng.standard_normal(weight.size).astype(np.float32) / math.sqrt(args.k)

        x_host = rng.standard_normal((args.k, args.n)).astype(np.float32)
        x_gpu = int(cuda(rt.cudaMalloc, x_bytes))
        d_ptrs.append(x_gpu)
        y_gpu = int(cuda(rt.cudaMalloc, y_bytes))
        d_ptrs.append(y_gpu)
        copy_h2d(x_gpu, x_host.ctypes.data, x_bytes)
        stream.sync()
        free_before_device_weight = mem_free_bytes()

        def one(weight_addr: int) -> None:
            launch(
                stream, config, kernel,
                y_gpu, int(weight_addr), x_gpu,
                np.int32(args.m), np.int32(args.k), np.int32(args.n)
            )

        def read_output() -> np.ndarray:
            stream.sync()
            result = np.empty((args.m, args.n), dtype=np.float32)
            copy_d2h(result.ctypes.data, y_gpu, y_bytes)
            return result

        def time_case(name: str, weight_addr: int, copy_each_time: bool = False):
            for _ in range(2):
                if copy_each_time:
                    copy_h2d(weight_addr, h_weight, w_bytes)
                one(weight_addr)
                stream.sync()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                if copy_each_time:
                    copy_h2d(weight_addr, h_weight, w_bytes)
                one(weight_addr)
                stream.sync()
            ms = (time.perf_counter() - t0) * 1000 / args.iters
            print(f"{name:16s} {ms:12.3f} ms/iteration", flush=True)
            return ms, read_output()

        mapped_ms, mapped_output = time_case("mapped_ram", d_weight_mapped)

        # Comparators intentionally allocate a SECOND copy of W in VRAM.
        # In production RAM-native mode this cudaMalloc/cudaMemcpy path is absent.
        device_weight = int(cuda(rt.cudaMalloc, w_bytes))
        d_ptrs.append(device_weight)
        copy_h2d(device_weight, h_weight, w_bytes)
        stream.sync()
        free_after_device_weight = mem_free_bytes()
        resident_ms, resident_output = time_case("vram_resident", device_weight)
        staged_ms, staged_output = time_case("copy_each_iter", device_weight, True)

        max_abs = float(np.max(np.abs(mapped_output - resident_output)))
        correct = bool(np.allclose(
            mapped_output, resident_output, rtol=2e-4, atol=2e-4
        ) and np.allclose(
            staged_output, resident_output, rtol=2e-4, atol=2e-4
        ))
        report = {
            "experiment": "H3VM_RAM_NATIVE_COMPUTE_V1",
            "status": "PASS" if correct else "FAIL_PARITY",
            "gpu": str(device.name),
            "m": args.m, "k": args.k, "n": args.n,
            "weight_mib": round(w_bytes / (1024 ** 2), 2),
            "mapped_weight_device_allocation_bytes": 0,
            "vram_weight_extra_mib_measured": round(
                max(0, free_before_device_weight - free_after_device_weight)
                / (1024 ** 2), 2
            ),
            "mapped_ram_ms": round(mapped_ms, 4),
            "vram_resident_ms": round(resident_ms, 4),
            "copy_each_iter_ms": round(staged_ms, 4),
            "max_abs_error": max_abs,
            "note": "FP32 toy GEMM kernel; NOT H3 quality or performance proof"
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not correct:
            raise RuntimeError(f"Parity check failed: max_abs_error={max_abs}")
        return report
    finally:
        try:
            stream.sync()
        except Exception:
            pass
        for ptr in reversed(d_ptrs):
            rt.cudaFree(int(ptr))
        if h_weight is not None:
            rt.cudaFreeHost(int(h_weight))
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=64, help="token columns")
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--max-mapped-mb", type=int, default=256)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
