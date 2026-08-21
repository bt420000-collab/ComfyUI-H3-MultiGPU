"""Larry MiniMax-H3 Turbo compatibility for H3VM Snapshot Compute Islands.

Runtime-only module. This file intentionally imports Comfy/Torch only when the
Turbo H3VM loader executes, keeping the custom-node package import-safe.

The stock Larry bypass adapter assumes the default Comfy CUDA device for LoRA
weights. H3VM has real blocks on both cuda:0 and cuda:1, so this shim builds
separate injection managers for each island and pins each adapter's compute
weights to the island that owns the base module. Int8-fused MLP fc2 LoRAs keep
Larry's merge path because that kernel bypasses module.forward.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import sys
import types
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger(__name__)


def _load_larry_module():
    """Reuse the loaded Larry module when possible, otherwise load its __init__.py.

    H3VM deliberately depends on the user's installed Larry node for the exact
    Turbo v4 adapter semantics and bundled E-grid instead of vendoring a stale
    private copy.
    """
    required = ("MiniMaxH3TurboLoRA", "_FrugalLoRA", "_int8_fused_fc2", "_egrid", "_unique_t", "_interp_egrid")
    for mod in list(sys.modules.values()):
        try:
            path = str(getattr(mod, "__file__", "") or "").replace("\\", "/")
            if "ComfyUI-MiniMax-H3-Turbo" in path and all(hasattr(mod, x) for x in required):
                return mod
        except Exception:
            continue

    custom_nodes = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    init_py = os.path.join(custom_nodes, "ComfyUI-MiniMax-H3-Turbo", "__init__.py")
    if not os.path.isfile(init_py):
        raise RuntimeError(
            "H3VM Turbo 22/28 requires the sibling custom node "
            "ComfyUI-MiniMax-H3-Turbo. Install/update Larry's node first."
        )
    name = "_h3vm_larry_turbo_runtime"
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(name, init_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"H3VM could not import Larry Turbo node from {init_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise RuntimeError(
            "Installed ComfyUI-MiniMax-H3-Turbo is too old for H3VM compatibility. "
            "Missing helpers: " + ", ".join(missing)
        )
    return mod


@dataclass
class TurboPlan:
    larry: Any
    lora: dict
    lora_name: str
    strength: float
    low_vram: bool
    pruned: bool
    modules: list[str]
    backbone: list[str]
    adaln: list[str]
    fused_fc2: set[str]
    adaln_bases: dict[str, Any]


def prepare_turbo_plan(patcher, dm, lora_name: str, strength: float, low_vram: bool = False) -> TurboPlan:
    import folder_paths
    import comfy.utils

    larry = _load_larry_module()
    path = folder_paths.get_full_path("loras", lora_name)
    if path is None:
        raise RuntimeError(f"H3VM Turbo cannot find LoRA: {lora_name}")
    raw_lora = comfy.utils.load_torch_file(path, safe_load=True)
    # Larry's original files use `blocks.*`, while official LightX2V ComfyUI
    # files already use `diffusion_model.blocks.*`.  H3VM's internal island
    # mapping is canonicalized to the former.  Strip exactly one ComfyUI model
    # prefix from every tensor/alpha key so both families share one loader.
    prefixed = sum(1 for k in raw_lora if str(k).startswith("diffusion_model."))
    if prefixed:
        lora = {
            (str(k)[len("diffusion_model."):] if str(k).startswith("diffusion_model.") else str(k)): v
            for k, v in raw_lora.items()
        }
        LOG.info("H3VM Turbo namespace normalize | lora=%s stripped_diffusion_model=%d/%d",
                 lora_name, prefixed, len(raw_lora))
    else:
        lora = raw_lora
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora if ".lora_" in k})
    if not modules:
        raise RuntimeError(f"H3VM Turbo found no LoRA modules in {lora_name}")

    pruned = bool(getattr(dm, "use_adaln_curves", False))
    if pruned:
        backbone = [m for m in modules if "adaln_proj" not in m]
        adaln = [m for m in modules if "adaln_proj" in m]
    else:
        backbone, adaln = modules, []

    fused_fc2 = set(larry._int8_fused_fc2(dm, backbone)) if not low_vram else set()
    adaln_bases = {}
    if adaln:
        for name in adaln:
            parent = name.rsplit(".linear", 1)[0]
            try:
                adaln_bases[name] = comfy.utils.get_attr(dm, parent)
            except Exception as exc:
                raise RuntimeError(f"H3VM Turbo cannot resolve AdaLN module {parent}: {exc}") from exc

    LOG.info(
        "H3VM Turbo plan | lora=%s strength=%.3f base=%s mode=%s | modules=%d backbone=%d adaln=%d fused_fc2=%d",
        lora_name, float(strength), "pruned" if pruned else "full",
        "merge" if low_vram else "bypass_sharp", len(modules), len(backbone), len(adaln), len(fused_fc2),
    )
    return TurboPlan(
        larry=larry, lora=lora, lora_name=lora_name, strength=float(strength),
        low_vram=bool(low_vram), pruned=pruned, modules=modules,
        backbone=backbone, adaln=adaln, fused_fc2=fused_fc2,
        adaln_bases=adaln_bases,
    )


def _block_index(module_name: str):
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _owned(modules: list[str], split_count: int, owner: str) -> list[str]:
    out = []
    for m in modules:
        idx = _block_index(m)
        if owner == "prefix" and idx is not None and idx < split_count:
            out.append(m)
        elif owner == "tail" and idx is not None and idx >= split_count:
            out.append(m)
        elif owner == "main" and idx is None:
            out.append(m)
    return out


def _force_manager_device(manager, target_device):
    """Make Larry/Comfy bypass hooks place adapter weights on their island GPU."""
    for hook in getattr(manager, "hooks", []):
        cls_move = type(hook)._move_adapter_weights_to_device

        def forced(self, _requested_device, dtype=None, _target=target_device, _move=cls_move):
            return _move(self, _target, dtype)

        hook._move_adapter_weights_to_device = types.MethodType(forced, hook)


def _apply_bypass(patcher, plan: TurboPlan, modules: list[str], target_device, key: str) -> int:
    if not modules:
        return 0
    import comfy.lora
    import comfy.weight_adapter

    key_map = {m: f"diffusion_model.{m}.weight" for m in modules}
    loaded = comfy.lora.load_lora(plan.lora, key_map, log_missing=False)
    manager = comfy.weight_adapter.BypassInjectionManager()
    sd_keys = set(patcher.model.state_dict().keys())
    n = 0
    for weight_key, adapter in loaded.items():
        if weight_key not in sd_keys:
            continue
        if isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
            adapter = plan.larry._FrugalLoRA(adapter.loaded_keys, adapter.weights)
        elif not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
            continue
        manager.add_adapter(weight_key, adapter, strength=plan.strength)
        n += 1
    injections = manager.create_injections(patcher.model)
    _force_manager_device(manager, target_device)
    if manager.get_hook_count() > 0:
        patcher.set_injections(key, injections)
    return n


def _apply_merge(patcher, plan: TurboPlan, modules: list[str]) -> int:
    if not modules:
        return 0
    return int(plan.larry._apply_merge_lora(patcher, plan.lora, modules, plan.strength))


def _larry_unique_t_compat(plan: TurboPlan, timestep, shift_v: float, shift_a: float, payload: dict):
    """Call Larry's private ``_unique_t`` across known API generations.

    Larry's Turbo node has changed this helper while tracking H3 conditioning
    semantics:
      * old preview: ``(..., has_vis_cond)``
      * ref-audio fix: ``(..., has_vis_cond, has_aud_cond[, vis_aug, aud_aug])``
      * newer curve implementation: ``(..., payload)``

    H3VM intentionally reuses the installed Larry node, so do not freeze one
    private signature here.  Derive the condition flags from H3's PackedLayout
    and dispatch by parameter name.
    """
    fn = plan.larry._unique_t
    layout = payload.get("layout")
    segments = list(getattr(layout, "segments", ()) or ())
    refs = payload.get("refs") or ()

    if segments:
        has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in segments)
        # Comfy revisions have used both cond_audio and ref_audio segment names.
        has_aud_cond = any(k in ("cond_audio", "ref_audio") for _, _, k in segments)
    else:
        has_vis_cond = bool(payload.get("keyframes")) or any(
            isinstance(r, dict) and r.get("kind") in ("image", "video", "video_audio")
            for r in refs
        )
        has_aud_cond = bool(payload.get("cond_audio_latents")) or any(
            isinstance(r, dict) and r.get("kind") in ("audio", "video_audio")
            for r in refs
        )

    vis_aug = float(payload.get("visual_cond_noise_aug", 0.999))
    aud_aug = float(payload.get("audio_cond_noise_aug", 1.0))

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}

    if "payload" in params:
        return fn(timestep, shift_v, shift_a, payload)
    if "has_aud_cond" in params:
        kwargs = {}
        if "vis_aug" in params:
            kwargs["vis_aug"] = vis_aug
        if "aud_aug" in params:
            kwargs["aud_aug"] = aud_aug
        return fn(timestep, shift_v, shift_a, has_vis_cond, has_aud_cond, **kwargs)
    if "has_vis_cond" in params or len(params) == 4:
        return fn(timestep, shift_v, shift_a, has_vis_cond)

    # Last-resort compatibility for wrappers that hide their signature.  Only
    # retry arity mismatches; a TypeError raised *inside* Larry should surface.
    try:
        return fn(timestep, shift_v, shift_a, has_vis_cond, has_aud_cond, vis_aug, aud_aug)
    except TypeError as exc7:
        if "argument" not in str(exc7) and "positional" not in str(exc7):
            raise
    try:
        return fn(timestep, shift_v, shift_a, has_vis_cond, has_aud_cond)
    except TypeError as exc5:
        if "argument" not in str(exc5) and "positional" not in str(exc5):
            raise
    return fn(timestep, shift_v, shift_a, has_vis_cond)


def _install_cycle_free_adaln(main_patcher, dm, plan: TurboPlan, split_count: int, primary_device, secondary_device, prefix_device=None, tail_device=None):
    """Pruned-base AdaLN Turbo delta without object patches on proxy block paths.

    Larry normally installs an object patch on blocks.N.adaln_proj.forward. H3VM
    replaces those block paths with proxies, so the object patch would no longer
    resolve. We patch the *private original module object* directly and keep the
    same E-grid math. The closure intentionally does not capture the parent module,
    avoiding the parent -> forward closure -> parent cycle that would upset H3VM's
    repeated-run lifecycle.
    """
    if not plan.adaln:
        return 0
    import torch
    import torch.nn.functional as F
    import comfy.patcher_extension

    E = plan.larry._egrid()
    shared = {"silu_temb": None}
    shift_v = float(getattr(dm, "sigma_shift_video", plan.larry.SHIFT_V))
    shift_a = float(getattr(dm, "sigma_shift_audio", plan.larry.SHIFT_A))
    tt = getattr(dm, "adaln_t_table", None)
    if tt is not None and tt.shape[0] != E.shape[0]:
        tt = None

    def wrap(executor, *args, **kwargs):
        ts = args[1] if len(args) > 1 else kwargs.get("timestep")
        ctx = args[2] if len(args) > 2 else kwargs.get("context")
        payload = kwargs.get("minimax_payload") or {}
        us = _larry_unique_t_compat(plan, ts, shift_v, shift_a, payload)
        shared["silu_temb"] = plan.larry._interp_egrid(us, E, ctx.device, ctx.dtype)
        return executor(*args, **kwargs)

    main_patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "h3vm_turbo_adaln",
        wrap,
    )

    if prefix_device is None:
        prefix_device = secondary_device
    if tail_device is None:
        tail_device = primary_device

    n = 0
    for name, base in plan.adaln_bases.items():
        a_cpu = plan.lora[name + ".lora_A.weight"]
        b_cpu = plan.lora[name + ".lora_B.weight"] * plan.strength
        linear = base.linear
        apply_silu = bool(base.apply_silu)
        modalities = int(base.modalities)
        expand = int(base.expand)
        hidden = int(base.hidden)
        idx = _block_index(name)
        owner_device = prefix_device if idx is not None and idx < split_count else tail_device

        # Tiny A/B matrices may stay resident on the owning island; unlike the
        # full base weights this is a small, bounded Turbo-only cost and avoids a
        # CPU->GPU copy at every AdaLN call.
        try:
            a_dev = a_cpu.to(owner_device)
            b_dev = b_cpu.to(owner_device)
        except Exception:
            a_dev, b_dev = a_cpu, b_cpu

        def forward(t_emb, _linear=linear, _apply=apply_silu, _modalities=modalities,
                    _expand=expand, _hidden=hidden, _a=a_dev, _b=b_dev, _tt=tt,
                    _E=E, _shared=shared):
            x = _linear(F.silu(t_emb) if _apply else t_emb)
            st = None
            # Match Larry's curve-mode mapping exactly when possible.
            if _tt is not None and not _apply:
                try:
                    tb = _tt.to(t_emb.device, torch.float32)
                    idxs = torch.cdist(t_emb.detach().float(), tb).argmin(dim=1)
                    st = _E.to(t_emb.device)[idxs]
                except Exception:
                    st = None
            if st is None:
                st = _shared.get("silu_temb")
            if st is not None and st.shape[0] == x.shape[0]:
                av = _a.to(x.device, x.dtype)
                bv = _b.to(x.device, x.dtype)
                sv = st.to(x.device, x.dtype)
                x = x + (bv @ (av @ sv.T)).T
            x = x.view(x.shape[0] * _modalities, _expand * _hidden)
            return x.chunk(_expand, dim=-1)

        base.forward = forward
        n += 1
    return n


def apply_turbo_plan_to_islands(*, plan: TurboPlan, main_patcher, dm,
                                prefix_patcher, tail_patcher, split_count: int,
                                primary_device, secondary_device, prefix_device=None, tail_device=None):
    """Apply one Larry Turbo LoRA across main/prefix/tail H3VM patchers."""
    if prefix_device is None:
        prefix_device = secondary_device
    if tail_device is None:
        tail_device = primary_device

    backbone = list(plan.backbone)
    prefix = _owned(backbone, split_count, "prefix")
    tail = _owned(backbone, split_count, "tail")
    main = _owned(backbone, split_count, "main")

    counts = {"prefix_bypass": 0, "tail_bypass": 0, "main_bypass": 0,
              "prefix_merge": 0, "tail_merge": 0, "main_merge": 0,
              "adaln": 0}

    if plan.low_vram:
        counts["prefix_merge"] = _apply_merge(prefix_patcher, plan, prefix)
        counts["tail_merge"] = _apply_merge(tail_patcher, plan, tail)
        counts["main_merge"] = _apply_merge(main_patcher, plan, main)
    else:
        prefix_fc2 = sorted(set(prefix) & plan.fused_fc2)
        tail_fc2 = sorted(set(tail) & plan.fused_fc2)
        main_fc2 = sorted(set(main) & plan.fused_fc2)
        counts["prefix_merge"] = _apply_merge(prefix_patcher, plan, prefix_fc2)
        counts["tail_merge"] = _apply_merge(tail_patcher, plan, tail_fc2)
        counts["main_merge"] = _apply_merge(main_patcher, plan, main_fc2)
        counts["prefix_bypass"] = _apply_bypass(
            prefix_patcher, plan, [m for m in prefix if m not in plan.fused_fc2],
            prefix_device, "h3vm_turbo_bypass_prefix")
        counts["tail_bypass"] = _apply_bypass(
            tail_patcher, plan, [m for m in tail if m not in plan.fused_fc2],
            tail_device, "h3vm_turbo_bypass_tail")
        counts["main_bypass"] = _apply_bypass(
            main_patcher, plan, [m for m in main if m not in plan.fused_fc2],
            primary_device, "h3vm_turbo_bypass_main")

    if plan.pruned and plan.adaln:
        counts["adaln"] = _install_cycle_free_adaln(
            main_patcher, dm, plan, split_count, primary_device, secondary_device, prefix_device, tail_device)

    LOG.info(
        "H3VM Turbo islands applied | split=%d/%d | prefix bypass/merge=%d/%d | "
        "tail bypass/merge=%d/%d | main bypass/merge=%d/%d | adaln=%d",
        split_count, 50 - split_count,
        counts["prefix_bypass"], counts["prefix_merge"],
        counts["tail_bypass"], counts["tail_merge"],
        counts["main_bypass"], counts["main_merge"], counts["adaln"],
    )
    return counts


def _install_cycle_free_adaln_streaming(main_patcher, dm, plan: TurboPlan, owner_device_for_block):
    """Dev12 variant of the AdaLN compatibility shim for non-contiguous islands."""
    if not plan.adaln:
        return 0
    import torch
    import torch.nn.functional as F
    import comfy.patcher_extension

    E = plan.larry._egrid()
    shared = {"silu_temb": None}
    shift_v = float(getattr(dm, "sigma_shift_video", plan.larry.SHIFT_V))
    shift_a = float(getattr(dm, "sigma_shift_audio", plan.larry.SHIFT_A))
    tt = getattr(dm, "adaln_t_table", None)
    if tt is not None and tt.shape[0] != E.shape[0]:
        tt = None

    def wrap(executor, *args, **kwargs):
        ts = args[1] if len(args) > 1 else kwargs.get("timestep")
        ctx = args[2] if len(args) > 2 else kwargs.get("context")
        payload = kwargs.get("minimax_payload") or {}
        us = _larry_unique_t_compat(plan, ts, shift_v, shift_a, payload)
        shared["silu_temb"] = plan.larry._interp_egrid(us, E, ctx.device, ctx.dtype)
        return executor(*args, **kwargs)

    main_patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "h3vm_turbo_adaln_dev12_streaming",
        wrap,
    )

    n = 0
    for name, base in plan.adaln_bases.items():
        a_cpu = plan.lora[name + ".lora_A.weight"]
        b_cpu = plan.lora[name + ".lora_B.weight"] * plan.strength
        linear = base.linear
        apply_silu = bool(base.apply_silu)
        modalities = int(base.modalities)
        expand = int(base.expand)
        hidden = int(base.hidden)
        idx = _block_index(name)
        owner_device = owner_device_for_block(idx) if idx is not None else None

        # Keep the tiny adapter matrices close to the block that owns them. If a
        # backend refuses this move, fall back to host tensors and copy on demand.
        try:
            a_dev = a_cpu.to(owner_device) if owner_device is not None else a_cpu
            b_dev = b_cpu.to(owner_device) if owner_device is not None else b_cpu
        except Exception:
            a_dev, b_dev = a_cpu, b_cpu

        def forward(t_emb, _linear=linear, _apply=apply_silu, _modalities=modalities,
                    _expand=expand, _hidden=hidden, _a=a_dev, _b=b_dev, _tt=tt,
                    _E=E, _shared=shared):
            x = _linear(F.silu(t_emb) if _apply else t_emb)
            st = None
            if _tt is not None and not _apply:
                try:
                    tb = _tt.to(t_emb.device, torch.float32)
                    idxs = torch.cdist(t_emb.detach().float(), tb).argmin(dim=1)
                    st = _E.to(t_emb.device)[idxs]
                except Exception:
                    st = None
            if st is None:
                st = _shared.get("silu_temb")
            if st is not None and st.shape[0] == x.shape[0]:
                av = _a.to(x.device, x.dtype)
                bv = _b.to(x.device, x.dtype)
                sv = st.to(x.device, x.dtype)
                x = x + (bv @ (av @ sv.T)).T
            x = x.view(x.shape[0] * _modalities, _expand * _hidden)
            return x.chunk(_expand, dim=-1)

        base.forward = forward
        n += 1
    return n


def apply_turbo_plan_to_streaming_islands(*, plan: TurboPlan, main_patcher, dm,
                                           primary_island_patcher, secondary_island_patcher,
                                           owner_map: dict, primary_device, secondary_device):
    """Apply Larry Turbo to Dev12's non-contiguous striped block islands."""
    backbone = list(plan.backbone)
    primary_modules = []
    secondary_modules = []
    main_modules = []
    for name in backbone:
        idx = _block_index(name)
        if idx is None:
            main_modules.append(name)
        elif int(owner_map[int(idx)]) == 0:
            primary_modules.append(name)
        else:
            secondary_modules.append(name)

    counts = {
        "primary_bypass": 0, "secondary_bypass": 0, "main_bypass": 0,
        "primary_merge": 0, "secondary_merge": 0, "main_merge": 0,
        "adaln": 0,
    }

    if plan.low_vram:
        counts["primary_merge"] = _apply_merge(primary_island_patcher, plan, primary_modules)
        counts["secondary_merge"] = _apply_merge(secondary_island_patcher, plan, secondary_modules)
        counts["main_merge"] = _apply_merge(main_patcher, plan, main_modules)
    else:
        p_fc2 = sorted(set(primary_modules) & plan.fused_fc2)
        s_fc2 = sorted(set(secondary_modules) & plan.fused_fc2)
        m_fc2 = sorted(set(main_modules) & plan.fused_fc2)
        counts["primary_merge"] = _apply_merge(primary_island_patcher, plan, p_fc2)
        counts["secondary_merge"] = _apply_merge(secondary_island_patcher, plan, s_fc2)
        counts["main_merge"] = _apply_merge(main_patcher, plan, m_fc2)
        counts["primary_bypass"] = _apply_bypass(
            primary_island_patcher, plan,
            [m for m in primary_modules if m not in plan.fused_fc2],
            primary_device, "h3vm_turbo_bypass_dev12_primary")
        counts["secondary_bypass"] = _apply_bypass(
            secondary_island_patcher, plan,
            [m for m in secondary_modules if m not in plan.fused_fc2],
            secondary_device, "h3vm_turbo_bypass_dev12_secondary")
        counts["main_bypass"] = _apply_bypass(
            main_patcher, plan,
            [m for m in main_modules if m not in plan.fused_fc2],
            primary_device, "h3vm_turbo_bypass_dev12_main")

    if plan.pruned and plan.adaln:
        def owner_device_for_block(idx):
            return primary_device if int(owner_map[int(idx)]) == 0 else secondary_device
        counts["adaln"] = _install_cycle_free_adaln_streaming(
            main_patcher, dm, plan, owner_device_for_block)

    LOG.info(
        "H3VM Turbo Dev12 streaming applied | primary modules=%d secondary=%d main=%d | "
        "bypass=%d/%d/%d merge=%d/%d/%d adaln=%d",
        len(primary_modules), len(secondary_modules), len(main_modules),
        counts["primary_bypass"], counts["secondary_bypass"], counts["main_bypass"],
        counts["primary_merge"], counts["secondary_merge"], counts["main_merge"], counts["adaln"],
    )
    return counts


def apply_turbo_plan_to_mlp_helpers(*, plan: TurboPlan,
                                     primary_helper_patcher, secondary_helper_patcher,
                                     owner_map: dict, primary_device, secondary_device,
                                     helper_indices=None):
    """Mirror Larry Turbo's MLP LoRA semantics onto Dev12.3 helper MLP views.

    Primary helper executes blocks physically owned by secondary, and vice versa.
    Only MLP linears are present in these storage-only helper roots.  FC1 follows
    Larry's runtime bypass path; fused INT8 FC2 follows the same merge/weight-
    function path as the owner model.  Thus token-parallel rows use identical
    Turbo arithmetic regardless of which GPU computes them.
    """
    mlp_modules = [m for m in plan.backbone if ".mlp.fc" in m]
    allowed = None if helper_indices is None else {int(i) for i in helper_indices}
    pmods, smods = [], []
    for name in mlp_modules:
        idx = _block_index(name)
        if idx is None:
            continue
        if allowed is not None and int(idx) not in allowed:
            continue
        # The helper is always opposite the whole-block owner.
        if int(owner_map[int(idx)]) == 1:
            pmods.append(name)  # secondary-owned block -> primary helper
        else:
            smods.append(name)  # primary-owned block -> secondary helper

    out = {"primary_bypass": 0, "primary_merge": 0,
           "secondary_bypass": 0, "secondary_merge": 0}

    def apply_one(patcher, modules, device, prefix):
        if patcher is None or not modules:
            return 0, 0
        if plan.low_vram:
            return 0, _apply_merge(patcher, plan, modules)
        fused = sorted(set(modules) & plan.fused_fc2)
        bypass = [m for m in modules if m not in plan.fused_fc2]
        nm = _apply_merge(patcher, plan, fused)
        nb = _apply_bypass(patcher, plan, bypass, device, prefix)
        return nb, nm

    out["primary_bypass"], out["primary_merge"] = apply_one(
        primary_helper_patcher, pmods, primary_device, "h3vm_turbo_dev12_3_mlp_primary_helper")
    out["secondary_bypass"], out["secondary_merge"] = apply_one(
        secondary_helper_patcher, smods, secondary_device, "h3vm_turbo_dev12_3_mlp_secondary_helper")

    LOG.info(
        "H3VM Turbo Dev12.3 MLP helpers | primary modules=%d bypass/merge=%d/%d | "
        "secondary modules=%d bypass/merge=%d/%d",
        len(pmods), out["primary_bypass"], out["primary_merge"],
        len(smods), out["secondary_bypass"], out["secondary_merge"],
    )
    return out
def apply_turbo_plan_to_attention_helpers(*, plan: TurboPlan,
                                            primary_helper_patcher, secondary_helper_patcher,
                                            owner_map: dict, primary_device, secondary_device,
                                            helper_indices=None):
    """Replay Larry Turbo attention-projection patches on Dev19 helper packets."""
    attention_modules = [m for m in plan.backbone if ".attn.qkv_proj" in m or ".attn.out_proj" in m]
    allowed = None if helper_indices is None else {int(i) for i in helper_indices}
    pmods, smods = [], []
    for name in attention_modules:
        idx = _block_index(name)
        if idx is None:
            continue
        if allowed is not None and int(idx) not in allowed:
            continue
        if int(owner_map[int(idx)]) == 1:
            pmods.append(name)
        else:
            smods.append(name)

    out = {"primary_bypass": 0, "primary_merge": 0,
           "secondary_bypass": 0, "secondary_merge": 0}

    def apply_one(patcher, modules, device, prefix):
        if patcher is None or not modules:
            return 0, 0
        if plan.low_vram:
            return 0, _apply_merge(patcher, plan, modules)
        # Attention projections are not Larry's fused FC2 path. Keep the same
        # runtime-bypass semantics as the owner model so dense materialization
        # includes the exact LoRA delta before Dev19 slices head channels.
        nb = _apply_bypass(patcher, plan, modules, device, prefix)
        return nb, 0

    out["primary_bypass"], out["primary_merge"] = apply_one(
        primary_helper_patcher, pmods, primary_device, "h3vm_turbo_dev19_attn_primary_helper")
    out["secondary_bypass"], out["secondary_merge"] = apply_one(
        secondary_helper_patcher, smods, secondary_device, "h3vm_turbo_dev19_attn_secondary_helper")

    LOG.info(
        "H3VM Turbo Dev19 attention helpers | primary modules=%d bypass/merge=%d/%d | "
        "secondary modules=%d bypass/merge=%d/%d",
        len(pmods), out["primary_bypass"], out["primary_merge"],
        len(smods), out["secondary_bypass"], out["secondary_merge"],
    )
    return out

