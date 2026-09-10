from __future__ import annotations

"""Runtime fusion overlay for H3 VRAM Master.

The public rc6 algorithms remain in their existing modules. This overlay adds a
small hardware-aware planning boundary around Mode4 so a validated symmetric
25/25 dual-16G profile does not accidentally become the default for asymmetric
16G+8G users. Other backends are left untouched in this first fusion stage.
"""

from contextlib import contextmanager
import logging
import os

LOG = logging.getLogger("H3VM")
_PATCH_MARKER = "_h3vm_vram_master_fusion_v1"


def _mode4_plan(config):
    from .compute_planner import build_runtime_plan, probe_cuda_pair
    from .loader import _resolve_device

    primary = _resolve_device(config.primary_device)
    secondary = _resolve_device(config.secondary_device)
    pair = probe_cuda_pair(str(primary), str(secondary))
    plan = build_runtime_plan(pair, backend="MODE4")
    return pair, plan


@contextmanager
def _override_snapshot_split(secondary_blocks: int):
    """Temporarily override only the public Mode4 builder's block target."""
    from . import loader

    original = loader.build_h3_snapshot_islands_full_throttle

    def planned_builder(*args, **kwargs):
        kwargs["secondary_blocks_target"] = int(secondary_blocks)
        return original(*args, **kwargs)

    loader.build_h3_snapshot_islands_full_throttle = planned_builder
    try:
        yield
    finally:
        loader.build_h3_snapshot_islands_full_throttle = original


def install_vram_master_fusion() -> bool:
    """Install the idempotent public-runtime fusion overlay."""
    if os.environ.get("H3VM_DISABLE_VRAM_MASTER_FUSION", "0") == "1":
        return False

    from . import core_adapter as core

    current = core.build_asset_model
    if getattr(current, _PATCH_MARKER, False):
        return True

    original_build = core.build_asset_model
    original_adapt = core.adapt_model
    lock = getattr(core, "_CORE_BUILD_LOCK", None)

    def _planned_call(fn, config, *args, **kwargs):
        try:
            mode = core._normalize_mode(config.mode)
        except Exception:
            return fn(*args, config=config, **kwargs)
        if mode != "DUAL_SYNC_ACCEL":
            return fn(*args, config=config, **kwargs)

        try:
            pair, plan = _mode4_plan(config)
        except Exception as exc:
            # CI, CPU-only imports, or unusual launchers may not expose CUDA
            # properties at planning time. Preserve rc6 behavior rather than
            # turning the planner into a new startup dependency.
            LOG.debug("H3VM VRAM Master Mode4 planner fallback: %s", exc)
            return fn(*args, config=config, **kwargs)

        LOG.info(
            "H3VM VRAM MASTER plan | Mode4 | %s / %s | symmetric=%s | %s | reason=%s",
            pair.primary.name, pair.secondary.name, pair.near_symmetric,
            plan.summary(), plan.reason,
        )
        manager = lock if lock is not None else _NullLock()
        with manager:
            with _override_snapshot_split(plan.mode4_secondary_blocks):
                return fn(*args, config=config, **kwargs)

    def build_asset_model(*args, config, **kwargs):
        return _planned_call(original_build, config, *args, **kwargs)

    def adapt_model(model, *, config):
        return _planned_call(original_adapt, config, model)

    setattr(build_asset_model, _PATCH_MARKER, True)
    setattr(adapt_model, _PATCH_MARKER, True)
    core._h3vm_vram_master_original_build_asset_model = original_build
    core._h3vm_vram_master_original_adapt_model = original_adapt
    core.build_asset_model = build_asset_model
    core.adapt_model = adapt_model
    LOG.info("H3VM VRAM Master fusion installed | Mode4 hardware-aware planner active")
    return True


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
