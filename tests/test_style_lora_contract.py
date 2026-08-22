from types import SimpleNamespace

from h3vm import core_adapter as core
from h3vm.core_adapter import H3VMCoreConfig
from h3vm.style_lora import (
    StyleLoRASpec,
    append_style_loras,
    is_acceleration_lora,
    merge_weight_patch_state,
    filter_weight_patch_state_inplace,
    normalize_style_lora_stack,
)


def test_stack_is_appendable_and_preserves_order():
    a = append_style_loras(None, [("style-a.safetensors", 1.0), ("style-b.safetensors", 0.5)])
    b = append_style_loras(a, [("char-c.safetensors", 0.7)])
    assert [x.name for x in b] == ["style-a.safetensors", "style-b.safetensors", "char-c.safetensors"]
    assert [x.strength for x in b] == [1.0, 0.5, 0.7]


def test_known_accelerators_rejected_from_style_path():
    assert is_acceleration_lora("minimax_h3_turbo_v4_step600_ema.safetensors")
    assert is_acceleration_lora("minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")
    try:
        normalize_style_lora_stack([("minimax_h3_turbo_v4_step600_ema.safetensors", 1.0)])
    except RuntimeError as e:
        assert "accelerator LoRA" in str(e)
    else:
        raise AssertionError("accelerator LoRA was accepted as an ordinary style LoRA")


def test_patch_replay_filters_to_owned_weights():
    class Model:
        def state_dict(self):
            return {
                "diffusion_model.blocks.0.mlp.fc1.weight": object(),
                "diffusion_model.blocks.0.mlp.fc2.weight": object(),
            }

    target = SimpleNamespace(model=Model(), patches={})
    source = {
        "diffusion_model.blocks.0.mlp.fc1.weight": [(1,)],
        "diffusion_model.blocks.9.attn.q_proj.weight": [(2,)],
    }
    n = merge_weight_patch_state(target, source)
    assert n == 1
    assert list(target.patches) == ["diffusion_model.blocks.0.mlp.fc1.weight"]




def test_main_patch_state_can_be_trimmed_after_proxy_partition():
    class Model:
        def state_dict(self):
            return {"diffusion_model.final_layer.weight": object()}

    target = SimpleNamespace(model=Model(), patches={
        "diffusion_model.final_layer.weight": [(1,)],
        "diffusion_model.blocks.0.mlp.fc1.weight": [(2,)],
    })
    n = filter_weight_patch_state_inplace(target)
    assert n == 1
    assert list(target.patches) == ["diffusion_model.final_layer.weight"]


def test_asset_path_forwards_style_stack_in_all_modes():
    import h3vm.loader as loader_mod

    style_stack = (StyleLoRASpec("style-a.safetensors", 0.8),)
    old_stream = loader_mod.build_h3_streaming_exact_turbo
    old_snapshot = loader_mod.build_h3_snapshot_islands_full_throttle
    old_plan = core._capacity_vram_plan
    calls = []

    def fake_stream(**kwargs):
        calls.append(("stream", kwargs))
        return kwargs

    def fake_snapshot(**kwargs):
        calls.append(("snapshot", kwargs))
        return kwargs

    def fake_capacity_plan(profile, primary, secondary):
        return {
            "profile": profile,
            "primary_runtime_reserve_gb": 6.0,
            "secondary_runtime_reserve_gb": 3.0,
            "primary_hot_cache_gb": 3.0,
            "secondary_hot_cache_gb": 2.0,
            "single_root_trim_interval": 1,
            "mlp_primary_fraction": 0.68,
            "capacity_helper_heads": 16,
        }

    loader_mod.build_h3_streaming_exact_turbo = fake_stream
    loader_mod.build_h3_snapshot_islands_full_throttle = fake_snapshot
    core._capacity_vram_plan = fake_capacity_plan
    try:
        for mode in ("SINGLE_GPU", "DUAL_QUIET", "DUAL_CAPACITY", "DUAL_SYNC_ACCEL"):
            core.build_asset_model(
                unet_name="h3.safetensors",
                turbo_lora_name="turbo.safetensors",
                turbo_strength=1.0,
                config=H3VMCoreConfig(mode=mode, telemetry=False),
                style_lora_stack=style_stack,
            )
        assert len(calls) == 4
        for kind, kwargs in calls:
            assert kwargs["style_lora_stack"] == style_stack
        mode4 = calls[-1]
        assert mode4[0] == "snapshot"
        assert mode4[1]["turbo_lora_name"] is None
    finally:
        loader_mod.build_h3_streaming_exact_turbo = old_stream
        loader_mod.build_h3_snapshot_islands_full_throttle = old_snapshot
        core._capacity_vram_plan = old_plan


if __name__ == "__main__":
    test_stack_is_appendable_and_preserves_order()
    test_known_accelerators_rejected_from_style_path()
    test_patch_replay_filters_to_owned_weights()
    test_main_patch_state_can_be_trimmed_after_proxy_partition()
    test_asset_path_forwards_style_stack_in_all_modes()
    print("H3VM Style LoRA contract tests passed")
