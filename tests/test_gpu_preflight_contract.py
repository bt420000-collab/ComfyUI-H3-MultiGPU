from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types


MODULE_PATH = Path(__file__).resolve().parents[1] / "h3vm" / "gpu_preflight.py"
spec = importlib.util.spec_from_file_location("h3vm_gpu_preflight_contract", MODULE_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


assert mod._explicit_cuda_index("gpu:0") == 0
assert mod._explicit_cuda_index("GPU:1") == 1
assert mod._explicit_cuda_index("cuda:12") == 12
assert mod._explicit_cuda_index(3) == 3
assert mod._explicit_cuda_index(-1) is None
assert mod._explicit_cuda_index("gpu:x") is None
assert mod._explicit_cuda_index("default") is None

assert mod._startup_cuda_device_arg(["main.py", "--cuda-device", "0"]) == "0"
assert mod._startup_cuda_device_arg(["main.py", "--cuda-device=1"]) == "1"
assert mod._startup_cuda_device_arg(["main.py", "--cuda-device", "all"]) == "all"
assert mod._startup_cuda_device_arg(["main.py"]) is None

visible = mod._format_visible([
    (0, "NVIDIA GeForce RTX 4060 Ti", 16.0),
    (1, "NVIDIA CMP 40HX", 8.0),
])
assert "RTX 4060 Ti" in visible
assert "CMP 40HX" in visible
assert "cuda:0" in visible and "cuda:1" in visible


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def get_device_name(index):
        assert index == 0
        return "NVIDIA GeForce RTX 4060 Ti"

    @staticmethod
    def get_device_properties(index):
        assert index == 0
        return types.SimpleNamespace(total_memory=16 * 1024**3)


old_torch = sys.modules.get("torch")
old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
old_smi = mod._nvidia_smi_inventory
old_argv = sys.argv
try:
    sys.modules["torch"] = types.SimpleNamespace(cuda=_FakeCuda())
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sys.argv = ["main.py", "--cuda-device", "0"]
    mod._nvidia_smi_inventory = lambda: (
        "0, NVIDIA GeForce RTX 4060 Ti, 16384",
        "1, NVIDIA CMP 40HX, 8192",
    )
    try:
        mod._common_preflight("gpu:0", "gpu:1")
        raise AssertionError("preflight should fail when only one logical CUDA device is visible")
    except RuntimeError as exc:
        text = str(exc)
        assert "PyTorch-visible CUDA devices: 1" in text
        assert "CUDA_VISIBLE_DEVICES='0'" in text
        assert "RTX 4060 Ti" in text
        assert "CMP 40HX" in text
        assert "heterogeneous GPU pairs" in text
        assert "startup visibility/masking problem" in text
        assert "--cuda-device 0" in text
        assert "--cuda-device all" in text
        assert "cannot unmask a GPU after PyTorch has started" in text
        assert "CUDA_VISIBLE_DEVICES=0,1" in text
finally:
    mod._nvidia_smi_inventory = old_smi
    sys.argv = old_argv
    if old_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = old_torch
    if old_cuda_visible is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible

print("gpu preflight contract: ok")
