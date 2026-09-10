from __future__ import annotations

import json


def main():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    count = int(torch.cuda.device_count())
    if count < 2:
        raise SystemExit(f"Need at least two visible CUDA devices, got {count}")

    rows = []
    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        rows.append({
            "index": idx,
            "name": torch.cuda.get_device_name(idx),
            "vram_gib": round(int(props.total_memory) / (1024 ** 3), 3),
            "sm": int(props.multi_processor_count),
        })

    peer = {"0_to_1": None, "1_to_0": None}
    fn = getattr(torch.cuda, "can_device_access_peer", None)
    if fn is not None:
        try:
            peer["0_to_1"] = bool(fn(0, 1))
            peer["1_to_0"] = bool(fn(1, 0))
        except Exception as exc:
            peer["error"] = repr(exc)

    print(json.dumps({"devices": rows, "p2p": peer}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()