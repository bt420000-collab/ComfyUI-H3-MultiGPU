from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / "H3VM_FUSION_ROADMAP.md").read_text(encoding="utf-8")
for required in ("Planner", "Capacity", "Mode4", "Exact-SP Lab", "dual-GPU temporal Video VAE"):
    assert required in text, required
print("H3VM fusion docs contract: OK")
