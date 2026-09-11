"""H3VM dual-GPU visibility and logical-device preflight.

This module intentionally patches the small device-selection boundary instead of
editing the frozen heavy runtime in ``loader_base.py``. H3VM device options are
logical CUDA indices as seen by the current ComfyUI/Python process. Therefore a
``CUDA_VISIBLE_DEVICES`` mask may renumber or hide physical GPUs. GPU model names
and VRAM sizes do not need to match; what matters is that two distinct logical
CUDA devices are visible to the running process.
"""
from __future__ import annotations

import os
import subprocess
import sys

_PATCH_FLAG = "_h3vm_gpu_preflight_v3_installed"
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


def _startup_cuda_device_arg(argv=None):
    """Return ComfyUI's --cuda-device value when the process was pinned at launch."""
    items = list(sys.argv if argv is None else argv)
    for index, item in enumerate(items):
        text = str(item).strip()
        if text == "--cuda-device" and index + 1 < len(items):
            return str(items[index + 1]).strip()
        if text.startswith("--cuda-device="):
            return text.split("=", 1)[1].strip()
    return None


def _resolve_device(option):
    """Resolve H3VM's explicit GPU choices as PyTorch-logical CUDA devices.

    Explicit ``gpu:N`` / ``cuda:N`` values bypass ComfyUI's resolver so device
    selections remain stable even for identical cards. Other option forms still
    delegate to ComfyUI for compatibility.
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


def _visibility_remediation(cuda_visible, physical):
    """Build actionable startup guidance without pretending a live process can unmask CUDA."""
    lines = []
    launch_cuda_device = _startup_cuda_device_arg()

    if len(physical) >= 2:
        lines.extend(
            [
                f"nvidia-smi reports {len(physical)} physical NVIDIA GPUs, but PyTorch sees fewer than two.",
                "This is a startup visibility/masking problem, not a GPU-model compatibility failure.",
                "H3VM supports both identical and heterogeneous GPU pairs; model names and VRAM sizes may differ.",
            ]
        )
    else:
        lines.append(
            "H3VM supports both identical and heterogeneous GPU pairs, but two CUDA logical devices must be visible."
        )

    if launch_cuda_device not in (None, "", "all"):
        lines.append(
            f"ComfyUI startup pin detected: --cuda-device {launch_cuda_device}. "
            "Remove the single-GPU pin or start ComfyUI with --cuda-device all, then restart."
        )

    if cuda_visible not in (None, "", "-1"):
        lines.append(
            "CUDA_VISIBLE_DEVICES is set. H3VM cannot unmask a GPU after PyTorch has started. "
            "Expose both target physical GPUs before launch (for example CUDA_VISIBLE_DEVICES=0,1), then restart ComfyUI."
        )
    elif launch_cuda_device in (None, "", "all"):
        lines.append(
            "Verify the NVIDIA driver/startup environment, restart ComfyUI, and confirm torch.cuda.device_count() >= 2."
        )

    lines.append(
        "Do not select gpu:1 until the startup/preflight log shows at least two PyTorch-visible CUDA devices."
    )
    return lines


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
        message.extend(_visibility_remediation(cuda_visible, physical))
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
            "GPU model names may be identical or different; choose two distinct indices such as gpu:0 and gpu:1."
        )

    pair = (int(primary.index), int(secondary.index))
    if pair not in _LOGGED_PAIRS:
        primary_name = str(torch.cuda.get_device_name(primary))
        secondary_name = str(torch.cuda.get_device_name(secondary))
        pair_kind = "identical-model" if primary_name == secondary_name else "heterogeneous"
        print(
            f"[H3VM GPU PREFLIGHT] visible={visible_count} | "
            f"primary={primary} ({primary_name}) | secondary={secondary} ({secondary_name}) | "
            f"pair={pair_kind} | CUDA_VISIBLE_DEVICES={cuda_visible!r}",
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
