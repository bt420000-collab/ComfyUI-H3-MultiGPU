from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
brand = (ROOT / "BRAND_H3_VRAM_MASTER.md").read_text(encoding="utf-8")
roadmap = (ROOT / "H3VM_FUSION_ROADMAP.md").read_text(encoding="utf-8")

assert "ComfyUI-H3-VRAM-Master" in brand
assert "H3VM = H3 VRAM Master" in brand
assert "Private PM" in brand
assert "Exact-SP" in roadmap
assert "Mode4" in roadmap
print("H3 VRAM Master brand contract: OK")
