from __future__ import annotations
import time


def can_peer(a, b):
    import torch
    a = torch.device(a)
    b = torch.device(b)
    if a.type != "cuda" or b.type != "cuda" or a.index is None or b.index is None:
        return False, False
    fn = getattr(torch.cuda, "can_device_access_peer", None)
    if fn is None:
        return False, False
    try:
        return bool(fn(a.index, b.index)), bool(fn(b.index, a.index))
    except Exception:
        return False, False


def _one_way_copy(src_device, dst_device, size_mb=64, repeats=4):
    """Measure the exact cross-device copy primitive H3VM uses.

    This intentionally works even when cudaDeviceCanAccessPeer() is false.
    PyTorch/CUDA may route the copy through a driver-managed fallback path.
    No user-created pinned host buffer is used here.
    """
    import torch
    src_device = torch.device(src_device)
    dst_device = torch.device(dst_device)
    n = int(size_mb * 1024 * 1024)

    with torch.cuda.device(src_device):
        src = torch.empty(n, dtype=torch.uint8, device=src_device)
        src.fill_(0x5A)
    with torch.cuda.device(dst_device):
        dst = torch.empty(n, dtype=torch.uint8, device=dst_device)

    torch.cuda.synchronize(src_device)
    torch.cuda.synchronize(dst_device)

    # Warm-up the route once. The same operation is already used by the
    # physical shard for the packed hidden-state handoff.
    with torch.cuda.device(dst_device):
        dst.copy_(src, non_blocking=False)
    torch.cuda.synchronize(dst_device)

    t0 = time.perf_counter()
    for _ in range(repeats):
        with torch.cuda.device(dst_device):
            dst.copy_(src, non_blocking=False)
    torch.cuda.synchronize(dst_device)
    dt = max(1e-9, time.perf_counter() - t0)
    gbps = (n * repeats) / dt / 1e9

    del src, dst
    for d in (src_device, dst_device):
        try:
            torch.cuda.synchronize(d)
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        except Exception:
            pass
    return gbps


def probe_copy_paths(a, b, size_mb=64, repeats=4):
    """Probe direct-P2P capability *and* practical fallback D2D bandwidth."""
    import torch
    a = torch.device(a)
    b = torch.device(b)
    ab, ba = can_peer(a, b)
    result = {
        "peer_ab": ab,
        "peer_ba": ba,
        "copy_gbps_ab": None,
        "copy_gbps_ba": None,
        "size_mb": int(size_mb),
        "repeats": int(repeats),
        "error_ab": None,
        "error_ba": None,
    }

    try:
        result["copy_gbps_ab"] = _one_way_copy(a, b, size_mb=size_mb, repeats=repeats)
    except Exception as e:
        result["error_ab"] = repr(e)
    try:
        result["copy_gbps_ba"] = _one_way_copy(b, a, size_mb=size_mb, repeats=repeats)
    except Exception as e:
        result["error_ba"] = repr(e)
    return result


# Backward-compatible Dev5 name.
def probe_peer_copy(a, b, size_mb=32, repeats=4):
    r = probe_copy_paths(a, b, size_mb=size_mb, repeats=repeats)
    return {
        "peer_ab": r["peer_ab"],
        "peer_ba": r["peer_ba"],
        "gbps_ab": r["copy_gbps_ab"] if r["peer_ab"] else None,
        "gbps_ba": r["copy_gbps_ba"] if r["peer_ba"] else None,
        "error": r["error_ab"] or r["error_ba"],
    }
