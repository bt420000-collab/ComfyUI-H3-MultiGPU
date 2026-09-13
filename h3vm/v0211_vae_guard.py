"""v0.21.1 lightweight same-weight guard for Dual Video VAE."""
from __future__ import annotations

import hashlib
import struct

_PATCHED = False


def _model_weight_signature(model, max_tensors: int = 6):
    try:
        import torch
    except Exception:
        return None
    digest = hashlib.sha256()
    sampled = 0
    try:
        params = model.named_parameters(recurse=True)
    except Exception:
        return None
    for name, param in params:
        try:
            if not torch.is_tensor(param) or getattr(param, "is_meta", False) or param.numel() <= 0:
                continue
            flat = param.detach().reshape(-1)
            n = int(flat.numel())
            indices = sorted(set((0, n // 2, n - 1)))
            values = flat[indices].to(device="cpu", dtype=torch.float32).tolist()
            digest.update(str(name).encode("utf-8", errors="replace"))
            digest.update(str(tuple(param.shape)).encode("ascii", errors="replace"))
            for value in values:
                digest.update(struct.pack("<f", float(value)))
            sampled += 1
            if sampled >= max(1, int(max_tensors)):
                break
        except Exception:
            continue
    return digest.hexdigest() if sampled else None


def _assert_matching(primary_vae, secondary_vae, vae_name: str):
    primary = getattr(primary_vae, "first_stage_model", None)
    secondary = getattr(secondary_vae, "first_stage_model", None)
    sig_primary = _model_weight_signature(primary) if primary is not None else None
    sig_secondary = _model_weight_signature(secondary) if secondary is not None else None
    if sig_primary is not None and sig_secondary is not None and sig_primary != sig_secondary:
        raise RuntimeError(
            "H3VM Dual Video VAE weight mismatch: the connected primary VAE does not "
            f"match vae_name={vae_name!r}. Select the same MiniMax H3 Video VAE file "
            "as the VAE Loader feeding this node."
        )
    return bool(sig_primary is not None and sig_secondary is not None)


def install_dual_vae_identity_guard():
    global _PATCHED
    if _PATCHED:
        return True
    from . import dual_video_vae as mod
    original = mod.dual_decode_h3_video
    if getattr(original, "_h3vm_v0211_identity_guard", False):
        _PATCHED = True
        return True

    def guarded(vae, samples, vae_name, primary_device="gpu:0", secondary_device="gpu:1", cleanup_h3=True, telemetry=True, post_cleanup=True):
        secondary_dev = mod._resolve_device(secondary_device)
        secondary_vae = mod._load_secondary_vae(str(vae_name), secondary_dev, vae.vae_dtype)
        verified = _assert_matching(vae, secondary_vae, str(vae_name))
        if telemetry and verified:
            import logging
            logging.info("H3VM v0.21.1 Dual Video VAE same-weight signature verified")
        return original(
            vae=vae, samples=samples, vae_name=vae_name,
            primary_device=primary_device, secondary_device=secondary_device,
            cleanup_h3=cleanup_h3, telemetry=telemetry, post_cleanup=post_cleanup,
        )

    guarded._h3vm_v0211_identity_guard = True
    guarded._h3vm_original = original
    mod.dual_decode_h3_video = guarded
    _PATCHED = True
    return True
