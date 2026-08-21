"""Dev10 dual-GPU temporal-chunk decoder for MiniMax H3 Video VAE.

Design goals:
- Keep ComfyUI/MiniMax H3's native temporal chunk boundaries and overlap semantics.
- Decode independent temporal chunks on two isolated VAE instances, one per GPU.
- Never require GPU0<->GPU1 tensor copies; raw chunk outputs return to CPU and are merged there.
- Explicitly unload/demote the H3 working set before bringing two Video VAE instances online.

This is an experimental post-process accelerator. It intentionally targets only the
MiniMaxH3VideoVAE implementation with comfy_has_chunked_io=True.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChunkSpec:
    index: int
    start: int
    end: int


_SECONDARY_CACHE: dict[tuple[str, str, str], Any] = {}
_SECONDARY_CACHE_LOCK = threading.Lock()


def _resolve_device(label: str):
    import torch
    if label.startswith("gpu:"):
        return torch.device("cuda", int(label.split(":", 1)[1]))
    if label.startswith("cuda:"):
        return torch.device(label)
    raise ValueError(f"Unsupported CUDA device label: {label}")


def _load_secondary_vae(vae_name: str, device, dtype):
    """Load/cache an independent VAE object whose patcher owns the secondary GPU."""
    import comfy.sd
    import comfy.utils
    import folder_paths

    key = (str(vae_name), str(device), str(dtype))
    with _SECONDARY_CACHE_LOCK:
        cached = _SECONDARY_CACHE.get(key)
        if cached is not None:
            return cached

        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd, metadata = comfy.utils.load_torch_file(
            path, safe_load=True, return_metadata=True
        )
        secondary = comfy.sd.VAE(sd=sd, device=device, dtype=dtype, metadata=metadata)
        _SECONDARY_CACHE[key] = secondary
        return secondary


def _validate_h3_video_vae(vae, secondary=None):
    fs = getattr(vae, "first_stage_model", None)
    if fs is None:
        raise RuntimeError("H3VM Dev10: primary VAE has no first_stage_model")
    if type(fs).__name__ != "MiniMaxH3VideoVAE":
        raise RuntimeError(
            "H3VM Dev10 Dual Video VAE only supports MiniMaxH3VideoVAE; "
            f"got {type(fs).__name__}."
        )
    if not getattr(fs, "comfy_has_chunked_io", False):
        raise RuntimeError("H3VM Dev10 requires comfy_has_chunked_io=True")
    for name in (
        "_decode_temporal_chunks", "decode_output_shape", "_adaptive_decode",
        "tokens_chunk_size", "token_overlap", "vae_ratio_t", "frame_pre_padding",
        "frame_overlap", "token_drop", "_finalize_pixels",
    ):
        if not hasattr(fs, name):
            raise RuntimeError(f"H3VM Dev10: MiniMax H3 VAE API missing {name}")
    if secondary is not None:
        fs2 = getattr(secondary, "first_stage_model", None)
        if type(fs2).__name__ != "MiniMaxH3VideoVAE":
            raise RuntimeError("H3VM Dev10: secondary VAE type mismatch")
    return fs


def _cleanup_h3_runtime():
    """Release stale H3 prefetch/residency before dual-VAE decode."""
    import comfy.model_management as mm
    try:
        import comfy.model_prefetch as mp
        mp.cleanup_prefetch_queues()
    except Exception as e:
        logging.warning("H3VM Dev10 prefetch cleanup warning: %s", e)
    # At this point denoising has completed. Releasing loaded model residency gives
    # the 8 GB card a clean runway for the second 4.9 GB Video VAE.
    try:
        mm.unload_all_models()
    except Exception as e:
        logging.warning("H3VM Dev10 unload_all_models warning: %s", e)
    try:
        mm.soft_empty_cache()
    except Exception:
        pass



def _post_cleanup_dual_vae(drop_secondary_cache: bool = True):
    """Return both GPUs to a clean ComfyUI residency state after decode.

    Dev10 intentionally cached the secondary VAE object for speed. On repeated
    H3 runs that persistent patcher/reference can keep the 8G card too close to
    the cliff. Dev11 defaults to a clean hand-back instead.
    """
    import gc
    import comfy.model_management as mm
    try:
        mm.unload_all_models()
    except Exception as e:
        logging.warning("H3VM Dev11 post-VAE unload warning: %s", e)
    try:
        mm.soft_empty_cache()
    except Exception:
        pass
    if drop_secondary_cache:
        with _SECONDARY_CACHE_LOCK:
            _SECONDARY_CACHE.clear()
    gc.collect()
    try:
        mm.soft_empty_cache()
    except Exception:
        pass

def _blend_cpu(a, b, blend_extent: int, dim: int):
    import torch
    blend_extent = min(a.shape[dim], b.shape[dim], int(blend_extent))
    if blend_extent <= 0:
        return b
    positions = torch.arange(blend_extent, device=b.device, dtype=torch.float32)
    weight_a = 1.0 - positions / blend_extent
    weight_b = positions / blend_extent
    shape = [1] * a.ndim
    shape[dim] = blend_extent
    weight_a = weight_a.view(shape)
    weight_b = weight_b.view(shape)
    slice_a = [slice(None)] * a.ndim
    slice_b = [slice(None)] * b.ndim
    slice_a[dim] = slice(-blend_extent, None)
    slice_b[dim] = slice(0, blend_extent)
    # Perform overlap math in fp32 on CPU. The source decoder commonly produces
    # fp16; fp32 CPU blend avoids slow/fragile half arithmetic and is visually
    # equivalent to the reference overlap blend.
    blended = a[tuple(slice_a)].float() * weight_a + b[tuple(slice_b)].float() * weight_b
    if blend_extent < b.shape[dim]:
        rest = [slice(None)] * b.ndim
        rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(rest)].float()], dim=dim)
    return blended


def _finalize_cpu(fs, part):
    """Mirror MiniMaxH3VideoVAE._finalize_pixels on CPU in fp32."""
    import torch
    std = fs.pixel_std.detach().to(device="cpu", dtype=torch.float32)
    mean = fs.pixel_mean.detach().to(device="cpu", dtype=torch.float32)
    return (part.float() * std).add_(mean).clamp_(0.0, 1.0)


def _denormalize_latent_cpu(fs, z):
    import torch
    z = z.detach().to("cpu")
    mean = fs.latents_mean.detach().to(device="cpu", dtype=z.dtype).view(1, -1, 1, 1, 1)
    std = fs.latents_std.detach().to(device="cpu", dtype=z.dtype).view(1, -1, 1, 1, 1)
    return z * std + mean


def _prepare_chunks(fs, z):
    import torch
    original_shape = tuple(z.shape)
    pad_tokens, num_chunks = fs._decode_temporal_chunks(z.shape[2])
    if pad_tokens > 0:
        pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
        z = torch.cat([z, pad_z], dim=2)
    specs = []
    for i in range(num_chunks):
        start = i * fs.tokens_chunk_size
        end = start + fs.tokens_chunk_size + fs.token_overlap
        specs.append(ChunkSpec(i, start, end))
    return z, original_shape, pad_tokens, specs


def _load_vae_to_gpu(vae, representative_shape):
    import comfy.model_management as mm
    memory_used = int(vae.memory_used_decode(representative_shape, vae.vae_dtype))
    with mm.cuda_device_context(vae.device):
        mm.load_models_gpu([vae.patcher], memory_required=memory_used, force_full_load=vae.disable_offload)
    return memory_used


def _worker(name, vae, z_cpu, jobs, results, errors, stats, lock):
    import torch
    import comfy.model_management as mm

    device = vae.device
    fs = vae.first_stage_model
    local = []
    try:
        # PyTorch grad/inference state is thread-local. ComfyUI enters the node from an
        # inference/no-grad execution context, but newly spawned Python workers do not
        # inherit that state. Without this guard, an inference tensor from the sampler
        # can reach an autograd-enabled worker and RMSNorm attempts to save it for
        # backward, raising "Inference tensors cannot be saved for backward".
        # VAE decode is inference-only, so disable grad explicitly inside EACH worker.
        with torch.no_grad():
            with mm.cuda_device_context(device):
                while True:
                    try:
                        spec = jobs.get_nowait()
                    except queue.Empty:
                        break
                    t0 = time.perf_counter()
                    clip_z = z_cpu[:, :, spec.start:spec.end, :, :].to(
                        device=device, dtype=vae.vae_dtype, non_blocking=False
                    )
                    # Keep all heavy ViT3D decoder work local to this accelerator.
                    clip_dec = fs._adaptive_decode(clip_z)
                    torch.cuda.synchronize(device)
                    # No D2D: each island commits its raw decoded chunk to CPU.
                    out_cpu = clip_dec.detach().to(device="cpu", copy=True).contiguous()
                    dt = time.perf_counter() - t0
                    with lock:
                        results[spec.index] = out_cpu
                        local.append((spec.index, dt, tuple(out_cpu.shape)))
                    del clip_dec, clip_z
                    jobs.task_done()
    except BaseException as e:
        with lock:
            errors.append((name, e))
    finally:
        with lock:
            stats[name] = local


def _merge_chunks(fs, raw_results, original_shape):
    import torch

    output_shape = fs.decode_output_shape(original_shape)
    dec = torch.empty(output_shape, dtype=torch.float32, device="cpu")
    chunk_dec_frames = fs.tokens_chunk_size * fs.vae_ratio_t
    split_count = int(fs.token_drop > 0) + 1
    dec_overlap = None
    write_pos = 0

    def write_part(part):
        nonlocal write_pos
        if part.shape[2] <= 0:
            return
        part = _finalize_cpu(fs, part)
        copy_frames = min(part.shape[2], max(0, dec.shape[2] - write_pos))
        if copy_frames > 0:
            dec[:, :, write_pos:write_pos + copy_frames].copy_(part[:, :, :copy_frames])
            write_pos += copy_frames

    for i in range(len(raw_results)):
        clip_dec = raw_results[i]
        if clip_dec is None:
            raise RuntimeError(f"H3VM Dev10 missing decoded chunk {i}")
        for j in range(split_count):
            f_start = j * chunk_dec_frames
            f_end = min(f_start + chunk_dec_frames, clip_dec.shape[2])
            part = clip_dec[:, :, f_start:f_end]
            part = part[:, :, fs.frame_pre_padding:]
            if j == 0:
                if dec_overlap is not None:
                    part = _blend_cpu(dec_overlap, part, fs.frame_overlap, dim=-3)
                    dec_overlap = None
                write_part(part)
            else:
                dec_overlap = part.contiguous()
        if i == len(raw_results) - 1 and dec_overlap is not None:
            write_part(dec_overlap)
            dec_overlap = None
        raw_results[i] = None

    return dec


def dual_decode_h3_video(
    vae,
    samples,
    vae_name: str,
    primary_device: str = "gpu:0",
    secondary_device: str = "gpu:1",
    cleanup_h3: bool = True,
    telemetry: bool = True,
    post_cleanup: bool = True,
):
    import torch

    latent = samples["samples"]
    if getattr(latent, "is_nested", False):
        latent = latent.unbind()[0]

    primary = _resolve_device(primary_device)
    secondary_dev = _resolve_device(secondary_device)
    if primary == secondary_dev:
        raise ValueError("H3VM Dev10 requires two different CUDA devices")
    if str(getattr(vae, "device", "")) != str(primary):
        logging.warning(
            "H3VM Dev10 primary VAE load device is %s but node requested %s; using VAE's device.",
            getattr(vae, "device", None), primary,
        )
        primary = vae.device

    fs0 = _validate_h3_video_vae(vae)
    if latent.ndim != 5 or latent.shape[2] <= 1:
        logging.info("H3VM Dev10 fallback: latent is not a multi-frame H3 video")
        images = vae.decode(latent)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

    t_all = time.perf_counter()
    z_cpu = _denormalize_latent_cpu(fs0, latent)
    z_cpu, original_shape, pad_tokens, specs = _prepare_chunks(fs0, z_cpu)
    if len(specs) < 2:
        logging.info("H3VM Dev10 fallback: only %d temporal chunk", len(specs))
        images = vae.decode(latent)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

    if cleanup_h3:
        _cleanup_h3_runtime()

    secondary_vae = _load_secondary_vae(vae_name, secondary_dev, vae.vae_dtype)
    _validate_h3_video_vae(vae, secondary_vae)

    # Reserve/load against one real temporal chunk, not the full 15/30 s sequence.
    max_chunk_len = max(s.end - s.start for s in specs)
    representative = list(original_shape)
    representative[2] = max_chunk_len
    mem0 = _load_vae_to_gpu(vae, tuple(representative))
    mem1 = _load_vae_to_gpu(secondary_vae, tuple(representative))

    jobs = queue.Queue()
    for spec in specs:
        jobs.put(spec)
    results = [None] * len(specs)
    errors = []
    stats = {}
    lock = threading.Lock()

    t_decode = time.perf_counter()
    threads = [
        threading.Thread(
            target=_worker,
            args=("gpu0", vae, z_cpu, jobs, results, errors, stats, lock),
            daemon=True,
        ),
        threading.Thread(
            target=_worker,
            args=("gpu1", secondary_vae, z_cpu, jobs, results, errors, stats, lock),
            daemon=True,
        ),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    decode_s = time.perf_counter() - t_decode

    if errors:
        who, err = errors[0]
        if post_cleanup:
            _post_cleanup_dual_vae(drop_secondary_cache=True)
        raise RuntimeError(f"H3VM Dev10 dual Video VAE worker {who} failed: {err}") from err

    t_merge = time.perf_counter()
    decoded = _merge_chunks(fs0, results, original_shape)
    merge_s = time.perf_counter() - t_merge

    images = decoded.movedim(1, -1)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

    if telemetry:
        c0 = stats.get("gpu0", [])
        c1 = stats.get("gpu1", [])
        s0 = sum(x[1] for x in c0)
        s1 = sum(x[1] for x in c1)
        logging.info(
            "H3VM Dev10 DUAL VIDEO VAE | chunks=%d pad_tokens=%d assignment=%d/%d | "
            "gpu0_work=%.2fs gpu1_work=%.2fs wall_decode=%.2fs merge=%.2fs total=%.2fs | "
            "reserve_est=%.2f/%.2fGiB | NO-D2D",
            len(specs), pad_tokens, len(c0), len(c1), s0, s1, decode_s, merge_s,
            time.perf_counter() - t_all, mem0 / (1024**3), mem1 / (1024**3),
        )
        logging.info(
            "H3VM Dev10 chunk map | gpu0=%s | gpu1=%s",
            [x[0] for x in c0], [x[0] for x in c1],
        )

    if post_cleanup:
        _post_cleanup_dual_vae(drop_secondary_cache=True)
        logging.info("H3VM Dev11 POST-VAE CLEAN | all model residency unloaded; secondary VAE cache dropped")

    return images
