from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types


MODULE_PATH = Path(__file__).resolve().parents[1] / "h3vm" / "comfy_kitchen_multigpu.py"
spec = importlib.util.spec_from_file_location("h3vm_ck_multigpu_contract", MODULE_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class _FakeCuda:
    current = 0
    switches = []

    @classmethod
    def is_available(cls):
        return True

    @classmethod
    def device_count(cls):
        return 2

    @classmethod
    def current_device(cls):
        return cls.current

    @classmethod
    def set_device(cls, index):
        cls.switches.append((cls.current, int(index)))
        cls.current = int(index)


class _FakeTensor:
    is_cuda = True

    def __init__(self, index):
        self.device = types.SimpleNamespace(index=int(index))


old_platform = mod.sys.platform
old_torch = sys.modules.get("torch")
old_modules = {name: sys.modules.get(name) for name in (
    "comfy_kitchen", "comfy_kitchen.backends", "comfy_kitchen.backends.cuda"
)}
old_optout = os.environ.get("H3VM_DISABLE_CK_MULTIGPU_GUARD")

try:
    mod.sys.platform = "win32"
    sys.modules["torch"] = types.SimpleNamespace(cuda=_FakeCuda)

    pkg = types.ModuleType("comfy_kitchen")
    pkg.__path__ = []
    backends = types.ModuleType("comfy_kitchen.backends")
    backends.__path__ = []
    ck_cuda = types.ModuleType("comfy_kitchen.backends.cuda")

    def _original(tensor):
        assert _FakeCuda.current == tensor.device.index
        return ("capsule", tensor.device.index)

    ck_cuda._wrap_for_dlpack = _original
    sys.modules["comfy_kitchen"] = pkg
    sys.modules["comfy_kitchen.backends"] = backends
    sys.modules["comfy_kitchen.backends.cuda"] = ck_cuda

    assert mod.install_comfy_kitchen_multigpu_dlpack_guard() is True
    wrapped = ck_cuda._wrap_for_dlpack
    assert wrapped(_FakeTensor(1)) == ("capsule", 1)
    assert _FakeCuda.current == 1
    assert _FakeCuda.switches == [(0, 1)]

    # Reinstall is idempotent and does not stack wrappers.
    assert mod.install_comfy_kitchen_multigpu_dlpack_guard() is True
    assert ck_cuda._wrap_for_dlpack is wrapped

    # A tensor on the other GPU moves the current device back as needed.
    assert wrapped(_FakeTensor(0)) == ("capsule", 0)
    assert _FakeCuda.current == 0
    assert _FakeCuda.switches[-1] == (1, 0)

    os.environ["H3VM_DISABLE_CK_MULTIGPU_GUARD"] = "1"
    assert mod.install_comfy_kitchen_multigpu_dlpack_guard() is False
finally:
    mod.sys.platform = old_platform
    if old_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = old_torch
    for name, value in old_modules.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value
    if old_optout is None:
        os.environ.pop("H3VM_DISABLE_CK_MULTIGPU_GUARD", None)
    else:
        os.environ["H3VM_DISABLE_CK_MULTIGPU_GUARD"] = old_optout

print("comfy-kitchen multi-GPU guard contract: ok")
