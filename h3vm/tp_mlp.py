from __future__ import annotations

import copy
import dataclasses
import gc
import logging
import time
import weakref

LOG = logging.getLogger("H3VM")
_LAYOUT_CONVROT = "TensorCoreConvRotW4A4Layout"
_LAYOUT_INT8 = "TensorWiseINT8Layout"
_SUPPORTED_FC1_LAYOUTS = {_LAYOUT_CONVROT, _LAYOUT_INT8}


def _require_shardable_fc1(weight, label):
    """Validate a row-shardable quantized FC1 without dequantizing it.

    H3 pruned checkpoints are not uniform at the module level: the filename may
    advertise int8/convrot while individual linears can be TensorWiseINT8Layout.
    FC1 row sharding is exact for both supported layouts because both shards
    consume the same full input K dimension and only split independent output
    rows. ConvRot, when present, rotates K only and therefore does not couple
    output rows.
    """
    if not (hasattr(weight, "_qdata") and hasattr(weight, "_params")):
        raise RuntimeError(
            f"H3VM Dev8.2 tensor-parallel FC1 requires a supported quantized {label}; "
            f"got {type(weight).__name__}. Refusing BF16 materialization/duplication."
        )
    layout = getattr(weight, "_layout_cls", None)
    p = weight._params
    q = weight._qdata
    if layout not in _SUPPORTED_FC1_LAYOUTS or bool(getattr(p, "transposed", False)):
        raise RuntimeError(
            f"H3VM Dev8.2 tensor-parallel FC1 supports {_LAYOUT_CONVROT} and {_LAYOUT_INT8}; "
            f"got {layout}. Refusing BF16 materialization/duplication."
        )
    if q.ndim != 2 or len(tuple(p.orig_shape)) != 2:
        raise RuntimeError(f"H3VM Dev8.2 invalid quantized geometry for {label}: q={tuple(q.shape)} orig={p.orig_shape}")

    out_features, in_features = map(int, p.orig_shape)
    if int(q.shape[0]) != out_features:
        raise RuntimeError(
            f"H3VM Dev8.2 output-row contract mismatch for {label}: q={tuple(q.shape)} orig={p.orig_shape}"
        )
    if layout == _LAYOUT_CONVROT:
        if int(q.shape[1]) * 2 != in_features:
            raise RuntimeError(
                f"H3VM Dev8.2 packed ConvRot contract mismatch for {label}: q={tuple(q.shape)} orig={p.orig_shape}"
            )
    else:
        if tuple(map(int, q.shape)) != (out_features, in_features):
            raise RuntimeError(
                f"H3VM Dev8.2 INT8 contract mismatch for {label}: q={tuple(q.shape)} orig={p.orig_shape}"
            )
        if not bool(getattr(p, "is_weight", True)):
            raise RuntimeError(f"H3VM Dev8.2 {label} is TensorWiseINT8Layout but is_weight=False")

    scale = p.scale
    scale_mode = None
    if int(scale.numel()) == 1:
        scale_mode = "tensorwise"
    elif scale.ndim >= 1 and int(scale.shape[0]) == out_features:
        scale_mode = "rowwise"
    else:
        raise RuntimeError(
            f"H3VM Dev8.2 unsupported scale geometry for {label}: scale={tuple(scale.shape)} orig={p.orig_shape}"
        )
    return q, p, layout, scale_mode


def _slice_row_scale(scale, ffn, ps, *, primary):
    """Slice FC1 quant scales in the same [gate | up] row pattern as qdata."""
    import torch
    if int(scale.numel()) == 1:
        return scale.clone()
    if primary:
        return torch.cat((scale[:ps], scale[ffn:ffn + ps]), dim=0).contiguous()
    return torch.cat((scale[ps:ffn], scale[ffn + ps:2 * ffn]), dim=0).contiguous()


def _qt(qdata, params, *, layout, orig_shape, scale):
    from comfy.quant_ops import QuantizedTensor
    new_params = dataclasses.replace(
        params,
        orig_shape=tuple(int(x) for x in orig_shape),
        scale=scale,
        transposed=False,
    )
    return QuantizedTensor(qdata, layout, new_params)

def _clone_linear(src, weight, *, in_features, out_features):
    """Clone the Comfy Linear shell without cloning its original full weight."""
    import torch

    m = copy.copy(src)
    m._parameters = dict(getattr(src, "_parameters", {}))
    m._buffers = dict(getattr(src, "_buffers", {}))
    m._modules = dict(getattr(src, "_modules", {}))
    m.in_features = int(in_features)
    m.out_features = int(out_features)
    m.weight = torch.nn.Parameter(weight, requires_grad=False)
    m.bias = None
    if hasattr(m, "weight_function"):
        m.weight_function = list(getattr(src, "weight_function", []))
    if hasattr(m, "bias_function"):
        m.bias_function = list(getattr(src, "bias_function", []))
    for name in ("_v", "_v_weight", "_v_bias", "_v_signature", "_prefetch"):
        if hasattr(m, name):
            try:
                delattr(m, name)
            except Exception:
                pass
    return m


def _split_fc1(old_mlp, primary_groups):
    """Exactly row-shard H3's ConvRot FC1 while preserving SwiGLU pairing.

    FC1 is hidden -> [gate_all | up_all]. ConvRot W4A4 weight quantization is
    rowwise and FC1 sees the *same full input* on both accelerators. Therefore
    splitting output rows does not change activation quantization semantics.

    FC2 is deliberately kept whole on the block owner in Dev8. K-sharding FC2
    would make each shard derive its own activation scale, which is not the same
    arithmetic as the original full ConvRot linear. That optimization is deferred
    until H3VM has a shared-scale/custom-kernel path.
    """
    import torch

    fc1, fc2 = old_mlp.fc1, old_mlp.fc2
    q1, p1, fc1_layout, fc1_scale_mode = _require_shardable_fc1(fc1.weight, "fc1.weight")
    ffn = int(fc2.in_features)
    hidden = int(fc2.out_features)
    if int(fc1.in_features) != hidden or int(fc1.out_features) != 2 * ffn:
        raise RuntimeError(
            f"H3VM Dev8.2 unexpected H3 MLP geometry: fc1={fc1.in_features}->{fc1.out_features}, "
            f"fc2={fc2.in_features}->{fc2.out_features}"
        )
    if tuple(p1.orig_shape) != (2 * ffn, hidden):
        raise RuntimeError(
            f"H3VM Dev8.2 FC1 quantized shape mismatch: {p1.orig_shape}, expected {(2 * ffn, hidden)}"
        )

    # We use FC2's ConvRot group when available because the user-facing split is
    # expressed in the H3 FFN's natural 256-channel groups. FC1 row slicing itself
    # does not require group alignment, but keeping one common geometry makes the
    # future shared-scale FC2 path compatible with this placement.
    group = int(getattr(getattr(fc2.weight, "_params", None), "convrot_groupsize", 256))
    if group <= 0 or ffn % group:
        raise RuntimeError(f"H3VM Dev8.2 ffn={ffn} is not divisible by group={group}")
    groups = ffn // group
    pg = max(1, min(groups - 1, int(primary_groups)))
    ps = pg * group
    ss = ffn - ps

    # Packed FC1 rows are independent. Preserve [gate | up] inside each shard.
    p_q = torch.cat((q1[:ps], q1[ffn:ffn + ps]), dim=0).contiguous()
    p_s = _slice_row_scale(p1.scale, ffn, ps, primary=True)
    s_q = torch.cat((q1[ps:ffn], q1[ffn + ps:2 * ffn]), dim=0).contiguous()
    s_s = _slice_row_scale(p1.scale, ffn, ps, primary=False)

    p_qt = _qt(p_q, p1, layout=fc1_layout, orig_shape=(2 * ps, hidden), scale=p_s)
    s_qt = _qt(s_q, p1, layout=fc1_layout, orig_shape=(2 * ss, hidden), scale=s_s)
    p_fc1 = _clone_linear(fc1, p_qt, in_features=hidden, out_features=2 * ps)
    s_fc1 = _clone_linear(fc1, s_qt, in_features=hidden, out_features=2 * ss)

    return (
        H3TPFC1Shard(p_fc1, ps),
        H3TPFC1Shard(s_fc1, ss),
        fc2,
        {
            "hidden": hidden,
            "ffn": ffn,
            "group": group,
            "groups": groups,
            "primary_groups": pg,
            "primary_ffn": ps,
            "secondary_ffn": ss,
            "fc1_layout": fc1_layout,
            "fc1_scale_mode": fc1_scale_mode,
            "fc1_convrot": bool(getattr(p1, "convrot", fc1_layout == _LAYOUT_CONVROT)),
            "math": "exact_fc1_row_shard_fc2_owner",
        },
    )


class H3TPFC1Shard:
    """Lazy torch Module factory for one physical FC1 row shard."""

    def __new__(cls, fc1, ffn):
        import torch

        class _Shard(torch.nn.Module):
            def __init__(self, layer, n):
                super().__init__()
                self.fc1 = layer
                self.ffn = int(n)

            def forward(self, x):
                return self.fc1(x)

        return _Shard(fc1, ffn)


def _transport_snapshot(space):
    s = space.transport.stats
    return int(s["moves"]), int(s["bytes"]), float(s["seconds"])


def _used_gib(device):
    import torch
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / (1024 ** 3)


def _ram_gib():
    try:
        import psutil
        v = psutil.virtual_memory()
        return v.used / (1024 ** 3), v.available / (1024 ** 3)
    except Exception:
        return 0.0, 0.0


class TensorParallelFabric:
    def __init__(self, *, space, primary, secondary, tp_blocks, primary_groups,
                 tp_prefetch=True, telemetry=True):
        self.space = space
        self.primary = primary
        self.secondary = secondary
        self.tp_blocks = tuple(sorted(int(x) for x in tp_blocks))
        self.primary_groups = int(primary_groups)
        self.tp_prefetch = bool(tp_prefetch)
        self.telemetry = bool(telemetry)
        self.calls_per_step = max(1, len(self.tp_blocks))
        self.calls = 0
        self.step = 0
        self.acc = self._empty_acc()
        self.peak_primary = 0.0
        self.peak_secondary = 0.0

    @staticmethod
    def _empty_acc():
        return {
            "wall_ms": 0.0,
            "stage_ms": 0.0,
            "gather_ms": 0.0,
            "local_fc1_ms": 0.0,
            "remote_fc1_ms": 0.0,
            "fc2_ms": 0.0,
            "moves": 0,
            "bytes": 0,
            "transport_ms": 0.0,
        }

    def record(self, **r):
        self.calls += 1
        for k in self.acc:
            if k in r:
                self.acc[k] += r[k]
        try:
            self.peak_primary = max(self.peak_primary, _used_gib(self.primary))
            self.peak_secondary = max(self.peak_secondary, _used_gib(self.secondary))
        except Exception:
            pass
        if self.calls % self.calls_per_step:
            return
        self.step += 1
        a = self.acc
        if self.telemetry and (self.step <= 3 or self.step % 10 == 0):
            n = self.calls_per_step
            fc1_serial = a["local_fc1_ms"] + a["remote_fc1_ms"]
            fc1_parallel_floor = max(a["local_fc1_ms"], a["remote_fc1_ms"])
            fc1_overlap_ceiling = fc1_serial / max(fc1_parallel_floor, 1e-9)
            payload_gbps = (a["bytes"] / 1e9) / max(a["transport_ms"] / 1000.0, 1e-9)
            used, avail = _ram_gib()
            LOG.info(
                "H3VM Dev8.2 TP step #%d | blocks=%s | avg MLP wall=%.1fms stage=%.1fms gather=%.1fms fc2=%.1fms | "
                "FC1 cuda local=%.1fms remote=%.1fms overlap_ceiling=%.2fx | "
                "global moves=%d payload=%.1fMiB transport=%.1fms %.2fGB/s | "
                "peak %s=%.2fGiB %s=%.2fGiB | RAM used=%.1fGiB avail=%.1fGiB",
                self.step, list(self.tp_blocks),
                a["wall_ms"] / n, a["stage_ms"] / n, a["gather_ms"] / n, a["fc2_ms"] / n,
                a["local_fc1_ms"] / n, a["remote_fc1_ms"] / n, fc1_overlap_ceiling,
                int(a["moves"]), a["bytes"] / (1024 ** 2), a["transport_ms"], payload_gbps,
                self.primary, self.peak_primary, self.secondary, self.peak_secondary,
                used, avail,
            )
        self.acc = self._empty_acc()


def make_tp_mlp(local_fc1, remote_fc1, fc2, *, local_is_primary,
                local_device, remote_device, space, fabric, block_index, tp_prefetch=True):
    import torch

    class H3TensorParallelMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.local_fc1 = local_fc1
            self.fc2 = fc2  # Full original FC2 stays on the whole-block owner.
            object.__setattr__(self, "_remote_ref", weakref.ref(remote_fc1))
            object.__setattr__(self, "_space", space)
            object.__setattr__(self, "_fabric", fabric)
            self.local_is_primary = bool(local_is_primary)
            self.local_device = local_device
            self.remote_device = remote_device
            self.block_index = int(block_index)
            self.tp_prefetch = bool(tp_prefetch)
            self._prefetch_logged = False

        def _prefetch_remote(self, remote):
            if not self.tp_prefetch:
                return None
            try:
                import comfy.model_prefetch
                opts = {"prefetch_dynamic_vbars": True}
                q = comfy.model_prefetch.make_prefetch_queue([remote], self.remote_device, opts)
                if q is not None:
                    comfy.model_prefetch.prefetch_queue_pop(q, self.remote_device, remote)
                if not self._prefetch_logged:
                    LOG.info(
                        "H3VM Dev8.2 TP helper prefetch | block=%d owner=%s helper=%s active=%s",
                        self.block_index, self.local_device, self.remote_device, q is not None,
                    )
                    self._prefetch_logged = True
                return q
            except Exception as e:
                if not self._prefetch_logged:
                    LOG.warning("H3VM Dev8.2 TP helper prefetch disabled for block %d: %r", self.block_index, e)
                    self._prefetch_logged = True
                return None

        def forward(self, x):
            import torch
            import comfy.model_management
            import comfy.ops

            remote = self._remote_ref()
            if remote is None:
                raise RuntimeError(f"H3VM Dev8.2 remote FC1 shard for block {self.block_index} was released")
            if x.device != self.local_device:
                raise RuntimeError(
                    f"H3VM Dev8.2 block {self.block_index} owner mismatch: x={x.device}, expected {self.local_device}"
                )

            mode = self._space.transport.selected_mode
            neutral = mode.startswith("neutral_")
            stat0 = _transport_snapshot(self._space)
            wall0 = time.perf_counter()

            # Helper needs a read replica of the same full FC1 input. FC1 row
            # sharding is exact because both shards derive activation scales from
            # the same x, while their output rows are disjoint.
            t0 = time.perf_counter()
            gx = self._space.adopt(x, name=f"tp.b{self.block_index}.fc1.in", ram_backing=neutral)
            try:
                xr = gx.acquire(self.remote_device, prefer_neutral=neutral)
            finally:
                self._space.drop(gx)
            stage_ms = (time.perf_counter() - t0) * 1000.0

            q = self._prefetch_remote(remote)
            r0 = torch.cuda.Event(enable_timing=True)
            r1 = torch.cuda.Event(enable_timing=True)
            l0 = torch.cuda.Event(enable_timing=True)
            l1 = torch.cuda.Event(enable_timing=True)

            # Queue on two independent CUDA devices. No P2P is required.
            with comfy.model_management.cuda_device_context(self.remote_device):
                r0.record()
                zr = remote(xr)
                r1.record()
                if q is not None:
                    try:
                        import comfy.model_prefetch
                        comfy.model_prefetch.prefetch_queue_pop(q, self.remote_device, None)
                    except Exception:
                        pass

            with comfy.model_management.cuda_device_context(self.local_device):
                l0.record()
                zl = self.local_fc1(x)
                l1.record()

            # Return only the helper FC1 rows, not a full hidden-size partial.
            t1 = time.perf_counter()
            gz = self._space.adopt(zr, name=f"tp.b{self.block_index}.fc1.rows", ram_backing=neutral)
            try:
                zr_local = gz.acquire(self.local_device, prefer_neutral=neutral)
            finally:
                self._space.drop(gz)
            torch.cuda.synchronize(self.remote_device)
            gather_ms = (time.perf_counter() - t1) * 1000.0

            # Restore the original FC1 row order: [gate_all | up_all]. Then run
            # the untouched full FC2 on the block owner, preserving original H3
            # ConvRot activation-quantization semantics.
            with comfy.model_management.cuda_device_context(self.local_device):
                gp, up = (zl.chunk(2, dim=-1) if self.local_is_primary else zr_local.chunk(2, dim=-1))
                gs, us = (zr_local.chunk(2, dim=-1) if self.local_is_primary else zl.chunk(2, dim=-1))
                z = torch.cat((gp, gs, up, us), dim=-1)
                f0 = torch.cuda.Event(enable_timing=True)
                f1 = torch.cuda.Event(enable_timing=True)
                f0.record()
                out = comfy.ops.linear_input_act(self.fc2, z, "swiglu")
                f1.record()
                torch.cuda.synchronize(self.local_device)

            local_fc1_ms = float(l0.elapsed_time(l1))
            remote_fc1_ms = float(r0.elapsed_time(r1))
            fc2_ms = float(f0.elapsed_time(f1))
            wall_ms = (time.perf_counter() - wall0) * 1000.0
            stat1 = _transport_snapshot(self._space)
            self._fabric.record(
                wall_ms=wall_ms,
                stage_ms=stage_ms,
                gather_ms=gather_ms,
                local_fc1_ms=local_fc1_ms,
                remote_fc1_ms=remote_fc1_ms,
                fc2_ms=fc2_ms,
                moves=stat1[0] - stat0[0],
                bytes=stat1[1] - stat0[1],
                transport_ms=max(0.0, stat1[2] - stat0[2]) * 1000.0,
            )
            return out

    return H3TensorParallelMLP()


def make_helper_root(modules, label):
    import torch

    class H3TPHelperRoot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.helpers = torch.nn.ModuleDict({str(k): v for k, v in modules.items()})
            self.label = str(label)

        def memory_required(self, input_shape=None):
            return 0

    return H3TPHelperRoot()


def select_tp_blocks(scope, split_count, total_blocks=50):
    c = int(split_count)
    n = int(total_blocks)
    scope = str(scope)
    if scope == "off":
        return []
    if scope == "boundary_2":
        return sorted({max(0, c - 1), min(n - 1, c)})
    if scope == "boundary_6":
        return list(range(max(0, c - 3), min(n, c + 3)))
    if scope == "all_blocks":
        return list(range(n))
    raise ValueError(f"Unknown H3VM Dev8.2 tp_scope: {scope}")



def _storage_bytes(tensor):
    return int(tensor.numel()) * int(tensor.element_size())


def estimate_fc1_shard_storage(old_mlp, primary_groups):
    """Estimate native-quantized FC1 shard storage without allocating shard tensors.

    The estimate follows the exact row slicing used by ``_split_fc1``. It counts
    qdata plus quantization scale tensors. This is intentionally slightly more
    conservative than ComfyUI's registered-weight accounting for layouts where
    scale metadata is not charged separately.
    """
    q1, p1, layout, scale_mode = _require_shardable_fc1(old_mlp.fc1.weight, "fc1.weight")
    fc2 = old_mlp.fc2
    ffn = int(fc2.in_features)
    hidden = int(fc2.out_features)
    group = int(getattr(getattr(fc2.weight, "_params", None), "convrot_groupsize", 256))
    if group <= 0 or ffn % group:
        raise RuntimeError(f"H3VM Dev8.2 ffn={ffn} is not divisible by group={group}")
    groups = ffn // group
    pg = max(1, min(groups - 1, int(primary_groups)))
    ps = pg * group
    ss = ffn - ps

    full_rows = 2 * ffn
    p_rows = 2 * ps
    s_rows = 2 * ss
    q_row_bytes = _storage_bytes(q1) // max(1, full_rows)

    scale = p1.scale
    scale_full_bytes = _storage_bytes(scale)
    if int(scale.numel()) == 1:
        # Each physical shard owns a tiny scalar metadata copy.
        p_scale_bytes = scale_full_bytes
        s_scale_bytes = scale_full_bytes
    else:
        if int(scale.shape[0]) != full_rows:
            raise RuntimeError(
                f"H3VM Dev8.2 unsupported scale geometry during planner: scale={tuple(scale.shape)} rows={full_rows}"
            )
        scale_row_bytes = scale_full_bytes // max(1, full_rows)
        p_scale_bytes = p_rows * scale_row_bytes
        s_scale_bytes = s_rows * scale_row_bytes

    p_bytes = p_rows * q_row_bytes + p_scale_bytes
    s_bytes = s_rows * q_row_bytes + s_scale_bytes
    full_bytes = _storage_bytes(q1) + scale_full_bytes
    return {
        "primary_bytes": int(p_bytes),
        "secondary_bytes": int(s_bytes),
        "full_bytes": int(full_bytes),
        "groups": int(groups),
        "group": int(group),
        "primary_groups": int(pg),
        "primary_ffn": int(ps),
        "secondary_ffn": int(ss),
        "layout": layout,
        "scale_mode": scale_mode,
        "hidden": hidden,
        "ffn": ffn,
    }


def plan_tp_primary_groups(*, blocks, split_count, tp_scope, requested_groups,
                           base_primary_bytes, base_secondary_bytes,
                           primary_capacity, secondary_capacity,
                           safety_margin_bytes=4 * 1024 * 1024):
    """Auto-fit the FC1 row split into both GPU reserve budgets before mutation.

    The previous boundary probe exposed an important interaction: the whole-block placement can fit
    with only a few MiB of headroom, while boundary tensor parallelism moves a
    small net slice from one card to the other. Rather than failing after model
    surgery, Dev8.2 treats ``primary_ffn_groups`` as a performance target and
    finds the nearest safe native-quantized split first.
    """
    tp_blocks = select_tp_blocks(tp_scope, split_count, len(blocks))
    if not tp_blocks:
        return {
            "requested_groups": int(requested_groups),
            "effective_groups": int(requested_groups),
            "primary_bytes": int(base_primary_bytes),
            "secondary_bytes": int(base_secondary_bytes),
            "margin_bytes": int(safety_margin_bytes),
            "tp_blocks": [],
        }

    first = estimate_fc1_shard_storage(blocks[tp_blocks[0]].mlp, requested_groups)
    groups = int(first["groups"])
    requested = max(1, min(groups - 1, int(requested_groups)))

    # Prefer the user's requested speed bias, then the nearest alternative.
    # On equal distance, prefer more work on the faster primary GPU.
    candidates = sorted(range(1, groups), key=lambda g: (abs(g - requested), -g))
    best_failure = None
    for pg in candidates:
        p_bytes = int(base_primary_bytes)
        s_bytes = int(base_secondary_bytes)
        for i in tp_blocks:
            e = estimate_fc1_shard_storage(blocks[i].mlp, pg)
            pb, sb, full = e["primary_bytes"], e["secondary_bytes"], e["full_bytes"]
            if i < int(split_count):
                # This block was wholly on secondary: primary gains p shard,
                # secondary replaces full FC1 with the s shard.
                p_bytes += pb
                s_bytes += sb - full
            else:
                # This block was wholly on primary: primary replaces full FC1
                # with p shard and secondary gains the s helper shard.
                p_bytes += pb - full
                s_bytes += sb

        p_ok = p_bytes <= int(primary_capacity) - int(safety_margin_bytes)
        s_ok = s_bytes <= int(secondary_capacity) - int(safety_margin_bytes)
        if p_ok and s_ok:
            return {
                "requested_groups": requested,
                "effective_groups": int(pg),
                "primary_bytes": int(p_bytes),
                "secondary_bytes": int(s_bytes),
                "margin_bytes": int(safety_margin_bytes),
                "tp_blocks": list(tp_blocks),
            }
        score = max(
            p_bytes - (int(primary_capacity) - int(safety_margin_bytes)),
            s_bytes - (int(secondary_capacity) - int(safety_margin_bytes)),
        )
        if best_failure is None or score < best_failure[0]:
            best_failure = (score, pg, p_bytes, s_bytes)

    _, pg, p_bytes, s_bytes = best_failure
    raise RuntimeError(
        "H3VM Dev8.2 cannot fit any FC1 tensor-parallel row split inside the selected reserve budgets: "
        f"closest primary_groups={pg}, projected={p_bytes/(1024**3):.3f}+{s_bytes/(1024**3):.3f}GiB, "
        f"capacities={primary_capacity/(1024**3):.3f}+{secondary_capacity/(1024**3):.3f}GiB. "
        "Reduce TP scope or reserve requirements."
    )


def install_tp_mlp_fabric(*, blocks, split_count, primary, secondary, space,
                          tp_scope, primary_ffn_groups, tp_prefetch=True, telemetry=True):
    """Install exact FC1 tensor parallelism on selected H3 blocks."""
    tp_blocks = select_tp_blocks(tp_scope, split_count, len(blocks))
    fabric = TensorParallelFabric(
        space=space, primary=primary, secondary=secondary,
        tp_blocks=tp_blocks, primary_groups=primary_ffn_groups,
        tp_prefetch=tp_prefetch, telemetry=telemetry,
    )
    primary_helpers = {}
    secondary_helpers = {}
    geometry = None

    for ordinal, i in enumerate(tp_blocks):
        block = blocks[i]
        old = block.mlp
        p_fc1, s_fc1, fc2, g = _split_fc1(old, primary_ffn_groups)
        if geometry is None:
            geometry = g
        elif (g["ffn"], g["group"], g["primary_groups"]) != (
            geometry["ffn"], geometry["group"], geometry["primary_groups"]
        ):
            raise RuntimeError("H3VM Dev8.2 found non-uniform MLP geometry across H3 blocks")

        if i < int(split_count):
            # Whole block owner is secondary. Secondary keeps the smaller FC1
            # row shard while the faster primary computes the larger helper part.
            primary_helpers[f"b{i:02d}"] = p_fc1
            block.mlp = make_tp_mlp(
                s_fc1, p_fc1, fc2,
                local_is_primary=False,
                local_device=secondary, remote_device=primary,
                space=space, fabric=fabric, block_index=i,
                tp_prefetch=tp_prefetch,
            )
        else:
            secondary_helpers[f"b{i:02d}"] = s_fc1
            block.mlp = make_tp_mlp(
                p_fc1, s_fc1, fc2,
                local_is_primary=True,
                local_device=primary, remote_device=secondary,
                space=space, fabric=fabric, block_index=i,
                tp_prefetch=tp_prefetch,
            )

        del old
        if ordinal % 4 == 3:
            gc.collect()

    if geometry is None:
        geometry = {
            "hidden": 0, "ffn": 0, "group": 0, "groups": 0,
            "primary_groups": int(primary_ffn_groups), "primary_ffn": 0,
            "secondary_ffn": 0, "math": "off",
        }
    return fabric, primary_helpers, secondary_helpers, geometry, tp_blocks
