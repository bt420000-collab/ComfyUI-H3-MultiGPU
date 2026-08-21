from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {n}")
    return text.replace(old, new, 1)


# --- h3vm/loader.py ---------------------------------------------------------
path = Path("h3vm/loader.py")
text = path.read_text(encoding="utf-8")

text = replace_once(
    text,
    '                                   capacity_helper_heads: int = 16,\n'
    '                                   capacity_attention_kernel: str = "INT8_CURRENT"):',
    '                                   capacity_helper_heads: int = 16,\n'
    '                                   capacity_attention_kernel: str = "INT8_CURRENT",\n'
    '                                   _prebuilt_patcher=None):',
    "streaming signature",
)

text = replace_once(
    text,
    '    patcher, dm = _load_private_h3(unet_name, primary, safe_profile)\n\n'
    '    from .turbo_compat import prepare_turbo_plan\n'
    '    turbo_plan = prepare_turbo_plan(\n'
    '        patcher, dm, str(turbo_lora_name), float(turbo_strength), bool(turbo_low_vram)\n'
    '    )\n',
    '    if _prebuilt_patcher is None:\n'
    '        patcher, dm = _load_private_h3(unet_name, primary, safe_profile)\n'
    '        from .turbo_compat import prepare_turbo_plan\n'
    '        turbo_plan = prepare_turbo_plan(\n'
    '            patcher, dm, str(turbo_lora_name), float(turbo_strength), bool(turbo_low_vram)\n'
    '        )\n'
    '    else:\n'
    '        # Generic Core path: the caller already produced a private deepclone.\n'
    '        # Standard ModelPatcher weight patches are mirrored onto every H3VM\n'
    '        # island/helper patcher below. Runtime injections/object patches are\n'
    '        # rejected by core_adapter before reaching this point.\n'
    '        patcher = _prebuilt_patcher\n'
    '        dm = _validate_h3(patcher)\n'
    '        turbo_plan = None\n'
    '        LOG.info(\n'
    '            "H3VM Core prebuilt source | patches=%d | load_device=%s",\n'
    '            sum(len(v) for v in getattr(patcher, "patches", {}).values()),\n'
    '            getattr(patcher, "load_device", None),\n'
    '        )\n',
    "private load to prebuilt source",
)

old_patchers = '''    Patcher = patcher.__class__
    primary_island_patcher = Patcher(
        primary_root,
        load_device=primary,
        offload_device=torch.device("cpu"),
        size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )
    secondary_island_patcher = None if secondary_root is None else Patcher(
        secondary_root,
        load_device=secondary,
        offload_device=torch.device("cpu"),
        size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )

    primary_mlp_helper_patcher = None
    secondary_mlp_helper_patcher = None
    primary_mlp_helper_root = None
    secondary_mlp_helper_root = None
    if bool(mlp_token_parallel):
        # Never create empty helper patchers either.  In Dev13 all whole blocks
        # belong to GPU0, so only GPU1 needs the mirrored helper MLP tree.
        if primary_mlp_helpers:
            primary_mlp_helper_root = make_island_root(
                patcher.model, dm, primary_mlp_helpers, len(original_blocks), "dev12.3-primary-mlp-helper"
            )
            primary_mlp_helper_patcher = Patcher(
                primary_mlp_helper_root, load_device=primary, offload_device=torch.device("cpu"), size=0,
                weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
            )
        if secondary_mlp_helpers:
            secondary_mlp_helper_root = make_island_root(
                patcher.model, dm, secondary_mlp_helpers, len(original_blocks), "dev12.3-secondary-mlp-helper"
            )
            secondary_mlp_helper_patcher = Patcher(
                secondary_mlp_helper_root, load_device=secondary, offload_device=torch.device("cpu"), size=0,
                weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
            )
'''

new_patchers = '''    Patcher = patcher.__class__

    def _new_streaming_patcher(root, device):
        out = Patcher(
            root,
            load_device=device,
            offload_device=torch.device("cpu"),
            size=0,
            weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
        )
        if _prebuilt_patcher is not None:
            # Mirror ordinary ModelPatcher weight-patch state onto the subset
            # tree. Extra keys are harmless: ModelPatcher only applies patches
            # for weights present in this root. This is the exact mechanism that
            # lets a standard upstream LoraLoaderModelOnly feed H3VM Core.
            out.patches = {k: list(v) for k, v in getattr(patcher, "patches", {}).items()}
            out.patches_uuid = getattr(patcher, "patches_uuid", out.patches_uuid)
            out.force_cast_weights = bool(getattr(patcher, "force_cast_weights", False))
        return out

    primary_island_patcher = _new_streaming_patcher(primary_root, primary)
    secondary_island_patcher = None if secondary_root is None else _new_streaming_patcher(
        secondary_root, secondary
    )

    primary_mlp_helper_patcher = None
    secondary_mlp_helper_patcher = None
    primary_mlp_helper_root = None
    secondary_mlp_helper_root = None
    if bool(mlp_token_parallel):
        # Never create empty helper patchers either.  In Dev13 all whole blocks
        # belong to GPU0, so only GPU1 needs the mirrored helper MLP tree.
        if primary_mlp_helpers:
            primary_mlp_helper_root = make_island_root(
                patcher.model, dm, primary_mlp_helpers, len(original_blocks), "dev12.3-primary-mlp-helper"
            )
            primary_mlp_helper_patcher = _new_streaming_patcher(primary_mlp_helper_root, primary)
        if secondary_mlp_helpers:
            secondary_mlp_helper_root = make_island_root(
                patcher.model, dm, secondary_mlp_helpers, len(original_blocks), "dev12.3-secondary-mlp-helper"
            )
            secondary_mlp_helper_patcher = _new_streaming_patcher(secondary_mlp_helper_root, secondary)
'''
text = replace_once(text, old_patchers, new_patchers, "streaming patcher factory")

old_turbo = '''    from .turbo_compat import apply_turbo_plan_to_streaming_islands
    turbo_counts = apply_turbo_plan_to_streaming_islands(
        plan=turbo_plan,
        main_patcher=patcher,
        dm=dm,
        primary_island_patcher=primary_island_patcher,
        secondary_island_patcher=secondary_island_patcher,
        owner_map=owner_map,
        primary_device=primary,
        secondary_device=secondary,
    )

    mlp_helper_turbo_counts = None
    if bool(mlp_token_parallel):
        from .turbo_compat import apply_turbo_plan_to_mlp_helpers
        mlp_helper_turbo_counts = apply_turbo_plan_to_mlp_helpers(
            plan=turbo_plan,
            primary_helper_patcher=primary_mlp_helper_patcher,
            secondary_helper_patcher=secondary_mlp_helper_patcher,
            owner_map=owner_map,
            primary_device=primary,
            secondary_device=secondary,
            helper_indices=helper_indices,
        )
        if bool(capacity_mode):
            from .turbo_compat import apply_turbo_plan_to_attention_helpers
            apply_turbo_plan_to_attention_helpers(
                plan=turbo_plan,
                primary_helper_patcher=primary_mlp_helper_patcher,
                secondary_helper_patcher=secondary_mlp_helper_patcher,
                owner_map=owner_map,
                primary_device=primary,
                secondary_device=secondary,
                helper_indices=helper_indices,
            )
'''

new_turbo = '''    turbo_counts = None
    mlp_helper_turbo_counts = None
    if turbo_plan is not None:
        # Integrated Master path keeps the already-validated Larry/LightX
        # compatibility layer, including device-aware bypass adapters.
        from .turbo_compat import apply_turbo_plan_to_streaming_islands
        turbo_counts = apply_turbo_plan_to_streaming_islands(
            plan=turbo_plan,
            main_patcher=patcher,
            dm=dm,
            primary_island_patcher=primary_island_patcher,
            secondary_island_patcher=secondary_island_patcher,
            owner_map=owner_map,
            primary_device=primary,
            secondary_device=secondary,
        )

        if bool(mlp_token_parallel):
            from .turbo_compat import apply_turbo_plan_to_mlp_helpers
            mlp_helper_turbo_counts = apply_turbo_plan_to_mlp_helpers(
                plan=turbo_plan,
                primary_helper_patcher=primary_mlp_helper_patcher,
                secondary_helper_patcher=secondary_mlp_helper_patcher,
                owner_map=owner_map,
                primary_device=primary,
                secondary_device=secondary,
                helper_indices=helper_indices,
            )
            if bool(capacity_mode):
                from .turbo_compat import apply_turbo_plan_to_attention_helpers
                apply_turbo_plan_to_attention_helpers(
                    plan=turbo_plan,
                    primary_helper_patcher=primary_mlp_helper_patcher,
                    secondary_helper_patcher=secondary_mlp_helper_patcher,
                    owner_map=owner_map,
                    primary_device=primary,
                    secondary_device=secondary,
                    helper_indices=helper_indices,
                )
    elif _prebuilt_patcher is not None:
        turbo_counts = {
            "source": "prebuilt_weight_patches",
            "patch_entries": sum(len(v) for v in getattr(patcher, "patches", {}).values()),
        }
'''
text = replace_once(text, old_turbo, new_turbo, "conditional turbo replay")

# Record where the model came from in telemetry/config.
text = replace_once(
    text,
    '        "turbo_counts": turbo_counts,\n        "attention_mode": attention_mode,',
    '        "turbo_counts": turbo_counts,\n'
    '        "model_source": "prebuilt_model" if _prebuilt_patcher is not None else "asset_loader",\n'
    '        "attention_mode": attention_mode,',
    "model source telemetry",
)

path.write_text(text, encoding="utf-8")


# --- package __init__.py: Master Loader now calls shared Core ----------------
path = Path("__init__.py")
text = path.read_text(encoding="utf-8")

old_master = '''        model = H3VMCapacityModeTurboLoader().load(
            mode=mode, unet_name=str(unet_name), lora_name=str(turbo_lora),
            strength=float(lora_strength), low_vram=False,
            primary_device="gpu:0", secondary_device="gpu:1",
            capacity_vram_profile=str(capacity_vram_profile),
            capacity_mlp_chunk_rows=4096, capacity_outproj_chunk_rows=4096,
            capacity_helper_heads=16, capacity_attention_kernel="INT8_CURRENT",
            telemetry=bool(telemetry), expected_steps=int(step_count),
        )[0]
'''
new_master = '''        from .h3vm.core_adapter import H3VMCoreConfig, build_asset_model
        model = build_asset_model(
            unet_name=str(unet_name),
            turbo_lora_name=str(turbo_lora),
            turbo_strength=float(lora_strength),
            turbo_low_vram=False,
            config=H3VMCoreConfig(
                mode=mode,
                primary_device="gpu:0",
                secondary_device="gpu:1",
                capacity_vram_profile=str(capacity_vram_profile),
                expected_steps=int(step_count),
                telemetry=bool(telemetry),
                capacity_mlp_chunk_rows=4096,
                capacity_outproj_chunk_rows=4096,
                capacity_attention_kernel="INT8_CURRENT",
            ),
        )
'''
text = replace_once(text, old_master, new_master, "Master to shared Core")
text = text.replace(
    'print("[H3VM v0.20.0-rc1] Multi-GPU Loader ready | Single / Dual Quiet / Dual Capacity", flush=True)',
    'print("[H3VM v0.21.0-rc1] Shared Core ready | Master + generic MODEL adapter", flush=True)',
)
path.write_text(text, encoding="utf-8")


# --- VERSION / README / CHANGELOG -------------------------------------------
Path("VERSION").write_text("0.21.0-rc1\n", encoding="utf-8")

path = Path("README.md")
text = path.read_text(encoding="utf-8")
text = text.replace("> 当前版本：v0.20.0-rc1", "> 当前版本：v0.21.0-rc1")
marker = "## 安装\n"
section = '''## H3VM Core 通用 MODEL 接口\n\n从 v0.21.0-rc1 开始，双卡执行内核与 Master Loader 解耦。Master Loader 仍保留模型、Turbo LoRA、步数、时长、分辨率、Seed、Prompt 等一体化功能，但底层模式策略改为调用同一个 `h3vm.core_adapter`。\n\n高级工作流可以使用内部 API `adapt_model(model, config=H3VMCoreConfig(...))` 把已经由上游组装好的标准 ComfyUI `MODEL` 接入 H3VM。当前首版支持 clean H3 与标准 ModelPatcher weight-patch LoRA（常见 `LoraLoaderModelOnly` 路径）。\n\n为了避免“能跑但数学已经错了”，当前遇到 runtime injections、object patches、hook patches 或 weight-wrapper patches 会明确拒绝，不会静默降级。Larry 这类 injection/bypass Turbo 在 Master Loader 内仍继续走已经验证的 H3VM 专用兼容路径。后续会继续补 device-aware helper injection remap。\n\n这意味着项目现在分成两层：\n\n- **H3VM Core**：MODEL → H3VM execution → MODEL，负责显卡模式、调度、显存驻留与 telemetry。\n- **H3VM Master Loader**：面向普通用户的一体化前端，完整保留现有整合功能。\n\n'''
if section not in text:
    if marker not in text:
        raise RuntimeError("README install marker missing")
    text = text.replace(marker, section + marker, 1)
path.write_text(text, encoding="utf-8")

path = Path("CHANGELOG.md")
text = path.read_text(encoding="utf-8")
entry = '''## v0.21.0-rc1\n\n- Added shared `h3vm.core_adapter` execution policy used by the product Master Loader.\n- Added internal generic prebuilt MODEL adapter using ComfyUI `deepclone_multigpu()`.\n- Standard ModelPatcher weight patches are mirrored onto H3VM streaming/block/helper patchers.\n- Injection/object/hook/weight-wrapper MODEL states fail closed until exact per-device remapping is implemented.\n- Existing integrated Larry/LightX Master path remains intact and continues using the validated Turbo compatibility layer.\n\n'''
if not text.startswith("## v0.21.0-rc1"):
    text = entry + text
path.write_text(text, encoding="utf-8")


# --- lightweight contract test ----------------------------------------------
tests = Path("tests")
tests.mkdir(exist_ok=True)
Path("tests/test_core_adapter_contract.py").write_text('''from h3vm.core_adapter import H3VMCoreConfig, execution_kwargs\n\n\ndef main():\n    single = execution_kwargs(H3VMCoreConfig(mode="SINGLE_GPU"))\n    assert single["mlp_token_parallel"] is False\n    assert single["attention_mode"] == "off"\n    quiet = execution_kwargs(H3VMCoreConfig(mode="DUAL_QUIET"))\n    assert quiet["mlp_primary_fraction"] == 0.68\n    assert quiet["critical_path_post_attention_island"] is True\n    assert quiet["attention_mode"] == "force_host"\n    print("core_adapter contract OK")\n\n\nif __name__ == "__main__":\n    main()\n''', encoding="utf-8")

print("H3VM v0.21.0-rc1 core refactor staged")
