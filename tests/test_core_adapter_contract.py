from types import SimpleNamespace

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


if __name__ == "__main__":
    test_static_mode_policy()
    test_weight_patch_mirror_isolated_lists()
    print("H3VM Core contract tests passed")
