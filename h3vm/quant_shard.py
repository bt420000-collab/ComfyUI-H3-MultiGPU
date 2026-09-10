from __future__ import annotations

"""Exact row/head sharding helpers for H3 QKV projections.

The quantized-row slicing approach is adapted from the public MIT-licensed
Kaihui-AMD/ComfyUI-MiniMaxH3-MultiGPU project. H3VM keeps the logic isolated
here so Capacity, future Exact-SP, and parity tests share one implementation.
"""

import copy
import dataclasses


def qkv_head_row_ranges(heads: int, head_dim: int, start_head: int, end_head: int):
    """Return the three contiguous Q/K/V output-row ranges for one head slice."""
    heads = int(heads)
    head_dim = int(head_dim)
    start_head = int(start_head)
    end_head = int(end_head)
    if heads < 1 or head_dim < 1:
        raise ValueError("heads and head_dim must be positive")
    if not 0 <= start_head < end_head <= heads:
        raise ValueError(
            f"invalid head range [{start_head}, {end_head}) for {heads} heads"
        )
    inner = heads * head_dim
    start = start_head * head_dim
    stop = end_head * head_dim
    return (
        (start, stop),
        (inner + start, inner + stop),
        (2 * inner + start, 2 * inner + stop),
    )


def qkv_head_row_indices(heads: int, head_dim: int, start_head: int, end_head: int, *, device=None):
    """Materialize row indices for QKV head slicing on the requested torch device."""
    import torch

    parts = [
        torch.arange(start, stop, device=device)
        for start, stop in qkv_head_row_ranges(heads, head_dim, start_head, end_head)
    ]
    return torch.cat(parts)


def _is_quantized_tensor(weight) -> bool:
    return type(weight).__name__ == "QuantizedTensor" and hasattr(weight, "_qdata")


def _replace_quant_params(weight, shape, *, index=None):
    updates = {"orig_shape": tuple(shape)}
    scale = getattr(getattr(weight, "_params", None), "scale", None)
    if index is not None:
        try:
            import torch

            if isinstance(scale, torch.Tensor) and scale.ndim > 0 and scale.numel() == weight.shape[0]:
                updates["scale"] = scale.reshape(-1)[index.to(scale.device)].contiguous()
        except Exception:
            pass
    return dataclasses.replace(weight._params, **updates)


def _wrap_parameter(weight):
    import torch

    return torch.nn.Parameter(weight, requires_grad=False)


def shard_linear_rows(linear, index):
    """Clone a Linear-like module while retaining only selected output rows.

    QuantizedTensor weights stay quantized. Per-output-channel scales are sliced
    with the same row index, avoiding a full dequantize -> slice -> F.linear path.
    """
    result = copy.copy(linear)
    result._parameters = dict(linear._parameters)

    index = index.to(linear.weight.device)
    weight = linear.weight
    if _is_quantized_tensor(weight):
        qdata = weight._qdata[index].contiguous()
        params = _replace_quant_params(
            weight,
            (index.numel(), weight.shape[1]),
            index=index,
        )
        sharded_weight = type(weight)(qdata, weight._layout_cls, params)
    else:
        sharded_weight = weight[index].contiguous()
    result._parameters["weight"] = _wrap_parameter(sharded_weight)

    if linear.bias is not None:
        result._parameters["bias"] = _wrap_parameter(linear.bias[index].contiguous())
    result.out_features = int(index.numel())
    if hasattr(result, "_orig_shape"):
        result._orig_shape = (result.out_features, result.in_features)
    return result


def shard_qkv_heads(linear, heads: int, head_dim: int, start_head: int, end_head: int):
    """Create an exact QKV projection containing only [start_head, end_head)."""
    index = qkv_head_row_indices(
        heads,
        head_dim,
        start_head,
        end_head,
        device=linear.weight.device,
    )
    return shard_linear_rows(linear, index)