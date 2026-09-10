from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
name = (ROOT / "H3VM_PRODUCT_NAME").read_text(encoding="utf-8").strip()
assert name == "ComfyUI-H3-VRAM-Master"
print("H3VM product name contract: OK")
