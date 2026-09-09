"""H3VM dual-GPU visibility and logical-device preflight.

This module intentionally patches the small device-selection boundary instead of
editing the frozen heavy runtime in ``loader_base.py``.  H3VM device options are
logical CUDA indices as seen by the current ComfyUI/Python process.  Therefore a
``CUDA_VISIBLE_DEVICES`` mask may renumber physical GPUs, and identical GPU model
names are perfectly valid as long as two different logical devices are visible.
"""
from __future__ import annotations

import os
import subprocess

_PATCH_FLAG = "_h3vm_gpu_preflight_v2_installed"
_LOGGED_PAIRS: set[tuple[int, int]] = set()


def _explicit_cuda_index(option):
    """Return an explicit logical CUDA index for gpu:N / cuda:N options."""
    if isinstance(option, int):
        return option if option >= 0 else None
    if isinstance(option, str):
        text = option.strip().lower()
        for prefix in ("gpu:", "cuda:"):
            if text.startswith(prefix):
                suffix = text.split(":", 1)[1].strip()
                if suffix.isdigit():
                    return int(suffix)
    return None


def _resolve_device(option):
    """Resolve H3VM's explicit GPU choices as PyTorch-logical CUDA devices.

    Explicit ``gpu:N`` / ``cuda:N`` values bypass ComfyUI's resolver so two
    same-model cards cannot collapse onto a preferred/default device.  Other
    option forms still delegate to ComfyUI for compatibility.
    """
    import torch

    index = _explicit_cuda_index(option)
    if index is not None:
        return torch.device(f"cuda:{index}")
    if isinstance(option, torch.device):
        return option

    import comfy.model_management

    try:
        resolved = comfy.model_management.resolve_gpu_device_option(option)
    except Exception:
        resolved = None
    if resolved is None:
        raise RuntimeError(f"H3VM cannot resolve device {option!r}")
    return torch.device(resolved)


def _visible_cuda_inventory(torch_module):
    if not bool(torch_module.cuda.is_available()):
        return []
    try:
        count = int(torch_module.cuda.device_count())
    except Exception:
        count = 0

    inventory = []
    for index in range(count):
        name = "unknown"
        total_gib = None
        try:
            name = str(torch_module.cuda.get_device_name(index))
        except Exception:
            pass
        try:
            total_gib = float(torch_module.cuda.get_device_properties(index).total_memory) / (1024 ** 3)
        except Exception:
            pass
        inventory.append((index, name, total_gib))
    return inventory


def _format_visible(inventory):
    if not inventory:
        return "  (none)"
    lines = []
    for index, name, total_gib in inventory:
        memory = f", {total_gib:.1f} GiB" if total_gib is not None else ""
        lines.append(f"  cuda:{index} = {name}{memory}")
    return "\n".join(lines)


def _nvidia_smi_inventory():
    """Best-effort physical inventory used only when PyTorch sees <2 GPUs."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return ()
    if proc.returncode != 0:
        return ()
    return tuple(line.strip() for line in proc.stdout.splitlines() if line.strip())


def _common_preflight(primary_device, secondary_device):
    import torch

    cuda_available = bool(torch.cuda.is_available())
    inventory = _visible_cuda_inventory(torch)
    visible_count = len(inventory)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    nvidia_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES")

    if not cuda_available or visible_count < 2:
        physical = _nvidia_smi_inventory()
        message = [
            "H3VM dual-GPU preflight failed.",
            f"PyTorch CUDA available: {cuda_available}",
            f"PyTorch-visible CUDA devices: {visible_count}",
            f"CUDA_VISIBLE_DEVICES={cuda_visible!r}",
            f"NVIDIA_VISIBLE_DEVICES={nvidia_visible!r}",
            "Visible devices:",
            _format_visible(inventory),
        ]
        if physical:
            message.append("System GPUs reported by nvidia-smi:")
            message.extend(f"  {row}" for row in physical)
        message.extend(
            [
                "H3VM supports identical GPU model names (for example two RTX 3080s).",
                "The two selected GPUs must both be visible to this ComfyUI/Python process.",
            ]
        )
        if cuda_visible not in (None, "", "-1"):
            message.append(
                "CUDA_VISIBLE_DEVICES is set. Restart ComfyUI with both target GPUs exposed "
                "(for example CUDA_VISIBLE_DEVICES=0,1), then select gpu:0 + gpu:1 inside H3VM."
            )
        else:
            message.append(
                "Verify the NVIDIA driver/startup environment, restart ComfyUI, and confirm "
                "torch.cuda.device_count() is at least 2 before using a dual-GPU mode."
            )
        raise RuntimeError("\n".join(message))

    primary = _resolve_device(primary_device)
    secondary = _resolve_device(secondary_device)

    for label, device in (("primary", primary), ("secondary", secondary)):
        if device.type != "cuda":
            raise RuntimeError(f"H3VM {label} device must be CUDA, got {device}")
        if device.index is None or device.index < 0 or device.index >= visible_count:
            raise RuntimeError(
                f"H3VM {label} device {device} is outside the {visible_count} CUDA devices visible "
                "to this ComfyUI process. H3VM uses logical indices after CUDA_VISIBLE_DEVICES."
            )

    if primary == secondary:
        raise RuntimeError(
            f"H3VM needs two different logical CUDA devices, got {primary} and {secondary}. "
            "Identical GPU model names are supported; choose two different indices such as gpu:0 and gpu:1."
        )

    pair = (int(primary.index), int(secondary.index))
    if pair not in _LOGGED_PAIRS:
        primary_name = str(torch.cuda.get_device_name(primary))
        secondary_name = str(torch.cuda.get_device_name(secondary))
        same_model = primary_name == secondary_name
        print(
            f"[H3VM GPU PREFLIGHT] visible={visible_count} | "
            f"primary={primary} ({primary_name}) | secondary={secondary} ({secondary_name}) | "
            f"identical_models={'yes' if same_model else 'no'} | "
            f"CUDA_VISIBLE_DEVICES={cuda_visible!r}",
            flush=True,
        )
        _LOGGED_PAIRS.add(pair)

    return primary, secondary


def install_gpu_preflight_patch():
    """Install the improved resolver/preflight before Core/loader wrappers bind."""
    from . import loader_base

    if bool(getattr(loader_base, _PATCH_FLAG, False)):
        return
    loader_base._resolve_device = _resolve_device
    loader_base._common_preflight = _common_preflight
    setattr(loader_base, _PATCH_FLAG, True)
