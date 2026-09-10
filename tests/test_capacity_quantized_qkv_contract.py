from h3vm.capacity_mode import CapacityFabric
from h3vm.capacity_quantized_qkv import _PATCH_MARKER, _looks_quantized

assert getattr(CapacityFabric.execute_capacity_attention, _PATCH_MARKER, False) is True

class FakeQuantizedTensor:
    pass

fake = FakeQuantizedTensor()
fake.__class__.__name__ = "FakeQuantizedTensor"
assert _looks_quantized(fake) is False

class QuantizedTensor:
    def __init__(self):
        self._qdata = object()

assert _looks_quantized(QuantizedTensor()) is True
print("H3 VRAM Master Capacity quantized-QKV contract: OK")
