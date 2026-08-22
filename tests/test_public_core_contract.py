from pathlib import Path


def _root():
    return Path(__file__).resolve().parents[1]


def test_public_core_node_is_registered_and_business_neutral():
    text = (_root() / "__init__.py").read_text(encoding="utf-8")
    assert '"H3VMCoreEngine": H3VMCoreEngine' in text
    start = text.index("class H3VMCoreEngine")
    end = text.index("class H3VMMasterLoader")
    node = text[start:end]
    assert '"model": ("MODEL",)' in node
    assert "expected_steps_hint" in node
    for forbidden in ("turbo_lora", "style_lora_stack", "prompt", "seed", "resolution", "sampler", "sigmas"):
        assert f'"{forbidden}"' not in node


def test_master_loader_remains_integrated_cockpit():
    text = (_root() / "__init__.py").read_text(encoding="utf-8")
    assert "class H3VMMasterLoader(_base.H3VMMasterLoader)" in text
    start = text.index("class H3VMMasterLoader")
    master = text[start:]
    for expected in (
        "unet_name", "turbo_lora", "steps", "duration_seconds", "resolution", "seed", "prompt", "style_lora_stack",
    ):
        assert expected in master


def test_prebuilt_mode4_uses_steps_hint_not_stock_20_contract():
    text = (_root() / "h3vm" / "core_adapter.py").read_text(encoding="utf-8")
    marker = "[H3VM CORE] prebuilt Mode4 Snapshot engine |"
    i = text.index(marker)
    block = text[i:i+3200]
    assert "expected_steps=max(1, int(config.expected_steps))" in block
    assert "turbo_lora_name=None" in block


if __name__ == "__main__":
    test_public_core_node_is_registered_and_business_neutral()
    test_master_loader_remains_integrated_cockpit()
    test_prebuilt_mode4_uses_steps_hint_not_stock_20_contract()
    print("H3VM Public Core contract tests passed")
