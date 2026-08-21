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


def test_asset_builder_uses_shared_mode_policy_without_comfy_runtime():
    import h3vm.loader as loader

    calls = []
    original_builder = loader.build_h3_streaming_exact_turbo

    def fake_builder(**kwargs):
        calls.append(kwargs)
        return kwargs

    loader.build_h3_streaming_exact_turbo = fake_builder
    try:
        quiet = core.build_asset_model(
            unet_name="unit-test-h3",
            turbo_lora_name="unit-test-lora",
            config=H3VMCoreConfig(mode="DUAL_QUIET", expected_steps=4),
        )
        assert quiet["attention_mode"] == "force_host"
        assert quiet["mlp_primary_fraction"] == 0.68
        assert quiet["critical_path_post_attention_island"] is True
        assert quiet["sidecar_mlp_blocks"] == 50
        assert quiet["expected_steps"] == 4

        single = core.build_asset_model(
            unet_name="unit-test-h3",
            turbo_lora_name="unit-test-lora",
            config=H3VMCoreConfig(mode="SINGLE_GPU", expected_steps=8),
        )
        assert single["attention_mode"] == "off"
        assert single["mlp_token_parallel"] is False
        assert single["expected_steps"] == 8
    finally:
        loader.build_h3_streaming_exact_turbo = original_builder

    assert len(calls) == 2


def test_prebuilt_bridge_copies_and_restores_hooks():
    import h3vm.loader as loader
    import h3vm.turbo_compat as turbo

    private = SimpleNamespace(
        patches={"diffusion_model.blocks.0.mlp.fc1.weight": ["PATCH"]},
        patches_uuid="uuid1",
        force_cast_weights=True,
    )
    original_load = loader._load_private_h3
    original_prepare = turbo.prepare_turbo_plan
    original_stream = turbo.apply_turbo_plan_to_streaming_islands
    original_mlp = turbo.apply_turbo_plan_to_mlp_helpers
    original_attention = turbo.apply_turbo_plan_to_attention_helpers

    with core._prebuilt_streaming_bridge(private):
        assert loader._load_private_h3 is not original_load
        assert turbo.prepare_turbo_plan is not original_prepare

        main = SimpleNamespace(
            patches=private.patches,
            patches_uuid=private.patches_uuid,
            force_cast_weights=True,
        )
        primary = SimpleNamespace(patches={}, patches_uuid=None, force_cast_weights=False)
        secondary = SimpleNamespace(patches={}, patches_uuid=None, force_cast_weights=False)
        result = turbo.apply_turbo_plan_to_streaming_islands(
            plan=object(), main_patcher=main, dm=None,
            primary_island_patcher=primary,
            secondary_island_patcher=secondary,
            owner_map={}, primary_device=None, secondary_device=None,
        )
        assert result["patch_entries"] == 2
        assert primary.patches_uuid == "uuid1"
        assert secondary.patches_uuid == "uuid1"

        helper0 = SimpleNamespace(patches={}, patches_uuid=None, force_cast_weights=False)
        helper1 = SimpleNamespace(patches={}, patches_uuid=None, force_cast_weights=False)
        helper_result = turbo.apply_turbo_plan_to_mlp_helpers(
            plan=object(), primary_helper_patcher=helper0,
            secondary_helper_patcher=helper1, owner_map={},
            primary_device=None, secondary_device=None, helper_indices=[0],
        )
        assert helper_result["patch_entries"] == 2

    assert loader._load_private_h3 is original_load
    assert turbo.prepare_turbo_plan is original_prepare
    assert turbo.apply_turbo_plan_to_streaming_islands is original_stream
    assert turbo.apply_turbo_plan_to_mlp_helpers is original_mlp
    assert turbo.apply_turbo_plan_to_attention_helpers is original_attention


if __name__ == "__main__":
    test_static_mode_policy()
    test_weight_patch_mirror_isolated_lists()
    test_runtime_bridge_rebinds_existing_capacity_loader()
    test_asset_builder_uses_shared_mode_policy_without_comfy_runtime()
    test_prebuilt_bridge_copies_and_restores_hooks()
    print("H3VM Core contract tests passed")
