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


def test_mode4_asset_route_uses_stock_full_throttle_builder():
    import h3vm.loader as loader_mod

    calls = {}
    original = loader_mod.build_h3_snapshot_islands_full_throttle

    def fake_builder(**kwargs):
        calls.update(kwargs)
        return "mode4-model"

    loader_mod.build_h3_snapshot_islands_full_throttle = fake_builder
    try:
        out = core.build_asset_model(
            unet_name="clean_h3.safetensors",
            turbo_lora_name="ignored_turbo.safetensors",
            turbo_strength=1.7,
            turbo_low_vram=False,
            config=H3VMCoreConfig(mode="DUAL_SYNC_ACCEL", telemetry=False),
        )
        assert out == "mode4-model"
        assert calls["unet_name"] == "clean_h3.safetensors"
        assert calls["secondary_blocks_target"] == 22
        assert calls["expected_steps"] == 20
        assert calls["predictor_mode"] == "linear"
        assert calls["predictor_beta"] == 0.75
        assert calls["secondary_ram_backing_gb"] == 2.0
        assert calls["turbo_lora_name"] is None
        assert calls["safe_profile"] is True
        assert calls["host_feeder"] == "bounded_pinned"
        assert calls["pinned_mailbox_mb"] == 1024
    finally:
        loader_mod.build_h3_snapshot_islands_full_throttle = original


def test_mode4_auto_falls_back_when_comfy_global_pinned_is_enabled():
    fake_comfy = ModuleType("comfy")
    fake_cli = ModuleType("comfy.cli_args")
    fake_cli.args = SimpleNamespace(disable_pinned_memory=False)
    old_comfy = sys.modules.get("comfy")
    old_cli = sys.modules.get("comfy.cli_args")
    sys.modules["comfy"] = fake_comfy
    sys.modules["comfy.cli_args"] = fake_cli
    try:
        policy = core._mode4_fullthrottle_host_policy()
        assert policy["safe_profile"] is False
        assert policy["host_feeder"] == "pageable"
        assert policy["pinned_mailbox_mb"] == 0
    finally:
        if old_comfy is None:
            sys.modules.pop("comfy", None)
        else:
            sys.modules["comfy"] = old_comfy
        if old_cli is None:
            sys.modules.pop("comfy.cli_args", None)
        else:
            sys.modules["comfy.cli_args"] = old_cli



if __name__ == "__main__":
    test_static_mode_policy()
    test_weight_patch_mirror_isolated_lists()
    test_runtime_bridge_rebinds_existing_capacity_loader()
    test_mode4_asset_route_uses_stock_full_throttle_builder()
    test_mode4_auto_falls_back_when_comfy_global_pinned_is_enabled()
    print("H3VM Core contract tests passed")
