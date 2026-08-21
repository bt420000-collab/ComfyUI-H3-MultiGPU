import sys
from types import ModuleType, SimpleNamespace

import h3vm.core_adapter as core
from h3vm.core_adapter import H3VMCoreConfig, _copy_weight_patch_state, execution_kwargs


def test_static_mode_policy():
    single = execution_kwargs(H3VMCoreConfig(mode="SINGLE_GPU"))
    assert single["mlp_token_parallel"] is False
    assert single["attention_mode"] == "off"
    assert single["single_root"] is True

    quiet = execution_kwargs(H3VMCoreConfig(mode="DUAL_QUIET"))
    assert quiet["mlp_primary_fraction"] == 0.68
    assert quiet["attention_mode"] == "force_host"
    assert quiet["critical_path_post_attention_island"] is True
    assert quiet["sidecar_mlp_blocks"] == 50


def test_weight_patch_mirror_isolated_lists():
    marker = object()
    source = SimpleNamespace(
        patches={"diffusion_model.blocks.0.mlp.fc1.weight": [marker]},
        patches_uuid="same-patch-set",
        force_cast_weights=True,
    )
    target = SimpleNamespace(patches={}, patches_uuid=None, force_cast_weights=False)
    count = _copy_weight_patch_state(source, target)
    assert count == 1
    assert target.patches_uuid == source.patches_uuid
    assert target.force_cast_weights is True
    assert target.patches is not source.patches
    assert target.patches["diffusion_model.blocks.0.mlp.fc1.weight"] is not source.patches[
        "diffusion_model.blocks.0.mlp.fc1.weight"
    ]
    assert target.patches["diffusion_model.blocks.0.mlp.fc1.weight"][0] is marker


def test_runtime_bridge_rebinds_existing_capacity_loader():
    class DummyLoader:
        def load(self):
            return "old"

    fake_parent = ModuleType("_h3vm_contract_parent")
    fake_parent.H3VMCapacityModeTurboLoader = DummyLoader
    sys.modules[fake_parent.__name__] = fake_parent
    old_package = core.__package__
    try:
        core.__package__ = fake_parent.__name__ + ".h3vm"
        assert core.install_runtime_bridge() is True
        assert DummyLoader._h3vm_core_bridged is True
        assert hasattr(DummyLoader, "_h3vm_original_load")
        assert DummyLoader.load is not DummyLoader._h3vm_original_load
    finally:
        core.__package__ = old_package
        sys.modules.pop(fake_parent.__name__, None)


if __name__ == "__main__":
    test_static_mode_policy()
    test_weight_patch_mirror_isolated_lists()
    test_runtime_bridge_rebinds_existing_capacity_loader()
    print("H3VM Core contract tests passed")
