from __future__ import annotations

import json
import logging
import time
import types
import weakref

from .mlp_token_parallel import clone_mlp_shared, _clone_linear_shared
from .post_attention_island import _clone_norm_shared, _mod_scale_shift_range, _mod_gate_range
from .rolling_helper_matrix import RollingCoverageMLPFabric
from .critical_path import CriticalPathMLPFabric

LOG = logging.getLogger("H3VM")


class _CapacityAttentionHelper:
    pass


def _clone_attention_shared(src_attn):
    import torch

    class H3AttentionHelper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = int(src_attn.heads)
            self.head_dim = int(src_attn.head_dim)
            self.qkv_proj = _clone_linear_shared(src_attn.qkv_proj)
            self.q_norm = _clone_norm_shared(src_attn.q_norm)
            self.k_norm = _clone_norm_shared(src_attn.k_norm)
            # Keep OutProj in the packet so Turbo patches remain path-compatible,
            # even though Capacity V1 runs the final OutProj on GPU0 in token chunks.
            self.out_proj = _clone_linear_shared(src_attn.out_proj)

        def forward(self, *args, **kwargs):
            raise RuntimeError("H3VM Capacity attention helper is storage-only")

    return H3AttentionHelper()


def build_capacity_helper_maps(blocks, owner_map, helper_indices):
    """Build one CPU-backed helper packet per H3 block.

    Each packet mirrors only the compute that can reduce activation pressure:
      * Norm2 + MLP for token-row partitioning;
      * QKV projection + Q/K norms for attention-head partitioning.

    Parameter names match the original block tree, so the existing Larry Turbo
    bypass/merge plan can be replayed exactly onto the helper patcher.
    """
    import torch

    selected = {int(i) for i in helper_indices}
    primary_helpers = {}
    secondary_helpers = {}
    helper_mlp_by_block = {}
    helper_attn_by_block = {}
    helper_packet_by_block = {}

    class H3CapacityHelperBlock(torch.nn.Module):
        def __init__(self, mlp, attn, index):
            super().__init__()
            self.mlp = mlp
            self.attn = attn
            self.h3vm_index = int(index)

        def forward(self, *args, **kwargs):
            raise RuntimeError("H3VM Capacity helper block is storage-only")

    for i, block in enumerate(blocks):
        if int(i) not in selected:
            continue
        mlp = clone_mlp_shared(block.mlp)
        mlp.norm2_helper = _clone_norm_shared(block.norm2)
        attn = _clone_attention_shared(block.attn)
        packet = H3CapacityHelperBlock(mlp, attn, i)
        helper_mlp_by_block[int(i)] = mlp
        helper_attn_by_block[int(i)] = attn
        helper_packet_by_block[int(i)] = packet
        if int(owner_map[int(i)]) == 0:
            secondary_helpers[int(i)] = packet
        else:
            primary_helpers[int(i)] = packet

    return (
        primary_helpers,
        secondary_helpers,
        helper_mlp_by_block,
        helper_attn_by_block,
        helper_packet_by_block,
    )


class CapacityFabric(RollingCoverageMLPFabric):
    """Exact capacity-first H3 execution fabric for asymmetric 16G+8G systems.

    Unlike the Dev14-Dev18 speed path, Capacity mode intentionally allows GPU0 to
    wait. Its job is to lower *peak activation residency*, not wall time.

    Memory rules:
      * all 50 blocks stay CPU-backed and execute on GPU0 one at a time;
      * GPU1 owns a transient mirrored helper packet for the current block;
      * attention QKV is partitioned by heads before the large QKV activation is
        materialized, so GPU0 never owns all 56 heads of QKV at once;
      * post-attention MLP is partitioned by token rows, then each side is further
        micro-chunked so the 2x-FFN SwiGLU intermediate is bounded;
      * final attention OutProj is token-chunked on GPU0 to avoid concatenating a
        sequence-sized all-head flat buffer.

    This path is exact with respect to the same dense projection arithmetic used
    by Dev19's validated QKV-only island. It changes placement/chunking only.
    """

    capacity_mode = True
    post_attention_island_mode = True
    rolling_helper_matrix_mode = False
    critical_path = False

    def __init__(self, *args, helper_attn_by_block=None, helper_packet_by_block=None,
                 mlp_chunk_rows=4096, outproj_chunk_rows=4096,
                 helper_heads=16, attention_kernel="INT8_CURRENT", **kwargs):
        # Build the base bookkeeping, but then disable every speed-path policy.
        kwargs["coverage_schedule"] = (50, 50, 50, 50)
        kwargs["adaptive_slack"] = False
        kwargs["resident_sidecar_packet"] = False
        super().__init__(*args, **kwargs)

        self.critical_path = False
        self.adaptive_slack = False
        self.primary_stall_budget_ms = 1.0e12
        self._disabled_blocks.clear()
        self.active_blocks = set(self._all_indices)
        self._coverage_target = len(self._all_indices)

        self.helper_attn_by_block = dict(helper_attn_by_block or {})
        self.helper_packet_by_block = dict(helper_packet_by_block or {})
        self.mlp_chunk_rows = max(256, int(mlp_chunk_rows))
        self.outproj_chunk_rows = max(256, int(outproj_chunk_rows))
        self.helper_heads = max(1, int(helper_heads))
        self.attention_kernel = str(attention_kernel).upper()
        if self.attention_kernel not in ("INT8_CURRENT", "OFFICIAL_OPTIMIZED"):
            raise ValueError(f"Unknown H3VM Capacity attention kernel: {attention_kernel}")

        self._helper_rope = None
        self._helper_rope_sig = None
        self._step_runtime = []
        self._attn_calls = 0
        self._attn_stage_ms = 0.0
        self._attn_return_ms = 0.0
        self._attn_root_ms = 0.0
        self._attn_helper_ms = 0.0
        self._chunked_mlp_calls = 0
        self._peak_root_free_mib = None
        self._peak_helper_free_mib = None

    def _reset_sample(self):
        self._step_runtime.clear()
        self._helper_rope = None
        self._helper_rope_sig = None

    # Capacity mode deliberately avoids one-ahead helper prefetch. Loading the
    # next mirrored packet while 100k-token activations are live defeats the
    # point of reserving VRAM for workspace. DynamicVRAM faults current weights
    # on demand and the runtime trims after every block.
    def prefetch_helper(self, index: int, transformer_options=None):
        return None

    def _consume_helper_prefetch(self, index: int):
        return None

    def _build_and_prime_prefetch(self, transformer_options=None, start_index=None):
        return None

    def _flush_helper_prefetch(self):
        self._helper_prefetch_q = None
        self._helper_prefetch_indices = []

    def _helper_has_runway(self, helper):
        # Fail by real allocator OOM rather than silently moving the whole high-res
        # block back to GPU0, which would destroy the capacity contract.
        return True

    def begin_step(self, step: int):
        # Bypass RollingCoverage's matrix/prefetch behavior but retain the base
        # counters used by runtime telemetry.
        CriticalPathMLPFabric.begin_step(self, int(step))
        self.active_blocks = set(self._all_indices)
        self._disabled_blocks.clear()
        self._coverage_target = len(self._all_indices)
        for idx in self._all_indices:
            self.block_primary_fraction[int(idx)] = self.primary_fraction
            self._miss_counts[int(idx)] = 0
            self._positive_slack_streak[int(idx)] = 0
        self._helper_prefetch_host_ms = 0.0
        self._helper_prefetch_prime_ms = 0.0
        self._helper_prefetch_calls = 0
        self._helper_prefetch_errors = 0
        self._helper_prefetch_broken = False
        self._helper_rope = None
        self._helper_rope_sig = None
        self._attn_calls = 0
        self._attn_stage_ms = 0.0
        self._attn_return_ms = 0.0
        self._attn_root_ms = 0.0
        self._attn_helper_ms = 0.0
        self._chunked_mlp_calls = 0
        self._peak_root_free_mib = None
        self._peak_helper_free_mib = None
        LOG.info(
            "H3VM CAPACITY step=%d | EXACT | blocks=50 | attention=QKV_HEAD_SHARD/%s | "
            "post_mlp=TOKEN_SHARD+CHUNK rows=%d | split=%.0f/%.0f | helper_heads<=%d",
            int(step), self.attention_kernel, self.mlp_chunk_rows,
            self.primary_fraction * 100.0, (1.0 - self.primary_fraction) * 100.0,
            self.helper_heads,
        )

    @staticmethod
    def _dense_linear_weight(linear, x):
        import torch
        import comfy.ops
        from comfy.quant_ops import QuantizedTensor

        w, b, off = comfy.ops.cast_bias_weight(
            linear, x, offloadable=True, compute_dtype=x.dtype, want_requant=False
        )
        if isinstance(w, QuantizedTensor):
            w = w.dequantize()
        if not isinstance(w, torch.Tensor) or w.ndim != 2:
            comfy.ops.uncast_bias_weight(linear, w, b, off)
            raise RuntimeError(
                f"H3VM Capacity unsupported projected weight type={type(w)!r} "
                f"shape={getattr(w, 'shape', None)}"
            )
        return w, b, off

    @staticmethod
    def _qkv_weight_slice(weight, bias, heads, head_dim, start, end):
        import torch
        inner = int(heads) * int(head_dim)
        a = int(start) * int(head_dim)
        b = int(end) * int(head_dim)
        if weight.shape[0] != 3 * inner:
            raise RuntimeError(
                f"H3VM Capacity QKV weight shape mismatch {tuple(weight.shape)} expected_out={3*inner}"
            )
        w = torch.cat(
            (weight[a:b], weight[inner+a:inner+b], weight[2*inner+a:2*inner+b]), dim=0
        )
        bb = None if bias is None else torch.cat(
            (bias[a:b], bias[inner+a:inner+b], bias[2*inner+a:2*inner+b]), dim=0
        )
        return w, bb

    def _qkv_norm_rope_attention(self, attn_like, qkv, rope_freqs, transformer_options, head_count):
        import comfy.model_management as mm
        import comfy.quant_ops
        import comfy_kitchen

        s = int(qkv.shape[0])
        hd = int(attn_like.head_dim)
        subinner = int(head_count) * hd
        q, k, v = qkv.split(subinner, dim=-1)
        v = v.view(s, int(head_count), hd)
        q = q.view(1, s, int(head_count), hd)
        k = k.view(1, s, int(head_count), hd)
        qw = mm.cast_to(attn_like.q_norm.weight, device=qkv.device)
        kw = mm.cast_to(attn_like.k_norm.weight, device=qkv.device)
        rot = int(rope_freqs.shape[-3]) * 2
        comfy.quant_ops.ck.rms_rope_split_half_(
            q, k, rope_freqs, qw, kw,
            epsilon=attn_like.q_norm.eps, rot_dim=rot,
        )
        q = q[0].transpose(0, 1).unsqueeze(0)
        k = k[0].transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        if self.attention_kernel == "OFFICIAL_OPTIMIZED":
            # Same ComfyUI attention dispatcher used by official MiniMax H3.
            # With --use-sage-attention this selects SageAttention; requesting
            # skip_output_reshape only keeps the per-head shape required by the
            # capacity path's chunked OutProj. The attention math itself is the
            # official optimized path rather than H3VM's INT8 prequantized path.
            from comfy.ldm.modules.attention import optimized_attention
            return optimized_attention(
                q, k, v, int(head_count), mask=None, skip_reshape=True,
                skip_output_reshape=True, transformer_options=transformer_options,
            )
        packed = comfy_kitchen.prequantize_int8_attention(q, k, v, scale=None, attn_mask=None)
        return comfy_kitchen.int8_attention_from_prequantized(packed)

    def _head_ranges(self, attn):
        heads = int(attn.heads)
        helper = min(self.helper_heads, heads - 1)
        # H3's validated asymmetric default is 40/16. Keep helper contiguous at
        # the tail so QKV slicing and final head order are exact.
        root = heads - helper
        return (0, root), (root, heads), [root, helper]

    def _sample_free(self):
        import torch
        mib = 1024 ** 2
        try:
            r = torch.cuda.mem_get_info(self.primary)[0] / mib
            h = torch.cuda.mem_get_info(self.secondary)[0] / mib
            self._peak_root_free_mib = r if self._peak_root_free_mib is None else min(self._peak_root_free_mib, r)
            self._peak_helper_free_mib = h if self._peak_helper_free_mib is None else min(self._peak_helper_free_mib, h)
        except Exception:
            pass

    def _ensure_helper_rope(self, rope_freqs, owner, helper, caller_stream):
        sig = (tuple(rope_freqs.shape), str(rope_freqs.dtype), int(rope_freqs.data_ptr()))
        if self._helper_rope is not None and self._helper_rope_sig == sig:
            return self._helper_rope
        remote, ready, _ = self._stage_to_helper(rope_freqs.contiguous(), owner, helper, caller_stream)
        ready.synchronize()
        self._helper_rope = remote
        self._helper_rope_sig = sig
        return remote

    def _chunked_outproj(self, attn, root_heads, helper_heads, seq):
        import torch
        rows = self.outproj_chunk_rows
        hidden = int(getattr(attn.out_proj, "out_features", 0) or 5376)
        out = torch.empty((int(seq), hidden), device=root_heads.device, dtype=root_heads.dtype)
        for start in range(0, int(seq), rows):
            end = min(int(seq), start + rows)
            # [1,H,S,D] -> only this token window is concatenated/flattened.
            heads = torch.cat(
                (root_heads[:, :, start:end, :], helper_heads[:, :, start:end, :]), dim=1
            )
            flat = heads.squeeze(0).transpose(0, 1).reshape(end - start, -1)
            part = attn.out_proj(flat)
            out[start:end].copy_(part)
            del heads, flat, part
        return out

    def execute_capacity_attention(self, index: int, attn, h, rope_freqs, transformer_options):
        import torch
        import torch.nn.functional as F
        import comfy.ops
        import comfy.model_management as mm

        index = int(index)
        owner = self.primary
        helper = self.secondary
        if h.device != owner:
            raise RuntimeError(f"H3VM Capacity attention requires root hidden on {owner}, got {h.device}")
        helper_attn = self.helper_attn_by_block.get(index)
        if helper_attn is None:
            raise RuntimeError(f"H3VM Capacity missing helper attention packet for block {index}")

        self._sample_free()
        caller_stream = torch.cuda.current_stream(owner)
        root_stream = self.compute_streams[owner]
        helper_stream = self.compute_streams[helper]
        root_stream.wait_stream(caller_stream)
        (r0, r1), (h0i, h1i), counts = self._head_ranges(attn)

        # Start root work first. Unlike Dev19, capacity mode does not judge this
        # path by overlap; the split exists to prevent all 56 QKV heads residing
        # on the 16G card at once.
        re0 = torch.cuda.Event(enable_timing=True)
        re1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(owner), torch.cuda.stream(root_stream), mm.cuda_device_context(owner):
            re0.record(root_stream)
            rw, rb, roff = self._dense_linear_weight(attn.qkv_proj, h)
            try:
                sw, sb = self._qkv_weight_slice(rw, rb, attn.heads, attn.head_dim, r0, r1)
                rqkv = F.linear(h, sw, sb)
                del sw, sb
            finally:
                comfy.ops.uncast_bias_weight(attn.qkv_proj, rw, rb, roff)
            root_heads = self._qkv_norm_rope_attention(
                attn, rqkv, rope_freqs, transformer_options, r1-r0
            )
            del rqkv
            re1.record(root_stream)

        t_stage = time.perf_counter()
        remote_h, remote_ready, _ = self._stage_to_helper(h.contiguous(), owner, helper, caller_stream)
        helper_rope = self._ensure_helper_rope(rope_freqs, owner, helper, caller_stream)
        self._attn_stage_ms += (time.perf_counter() - t_stage) * 1000.0

        he0 = torch.cuda.Event(enable_timing=True)
        he1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(helper), torch.cuda.stream(helper_stream), mm.cuda_device_context(helper):
            helper_stream.wait_event(remote_ready)
            he0.record(helper_stream)
            hw, hb, hoff = self._dense_linear_weight(helper_attn.qkv_proj, remote_h)
            try:
                sw, sb = self._qkv_weight_slice(
                    hw, hb, helper_attn.heads, helper_attn.head_dim, h0i, h1i
                )
                hqkv = F.linear(remote_h, sw, sb)
                del sw, sb
            finally:
                comfy.ops.uncast_bias_weight(helper_attn.qkv_proj, hw, hb, hoff)
            helper_heads = self._qkv_norm_rope_attention(
                helper_attn, hqkv, helper_rope, transformer_options, h1i-h0i
            )
            del hqkv, remote_h
            he1.record(helper_stream)

        t_ret = time.perf_counter()
        returned, returned_ready, _ = self._return_to_owner(helper_heads, owner, helper, he1)
        self._attn_return_ms += (time.perf_counter() - t_ret) * 1000.0

        caller_stream.wait_event(re1)
        caller_stream.wait_event(returned_ready)
        out = self._chunked_outproj(attn, root_heads, returned, h.shape[0])
        out.record_stream(caller_stream)
        root_heads.record_stream(caller_stream)
        returned.record_stream(caller_stream)

        try:
            self._attn_root_ms += float(re0.elapsed_time(re1))
            self._attn_helper_ms += float(he0.elapsed_time(he1))
        except Exception:
            pass
        self._attn_calls += 1
        if self._attn_calls <= 2 or self._attn_calls % 10 == 0:
            LOG.info(
                "H3VM CAPACITY ATTN #%d block=%d heads=%s seq=%d | stage_acc=%.1fms "
                "return_acc=%.1fms root_acc=%.1fms helper_acc=%.1fms | outproj_chunk=%d",
                self._attn_calls, index, counts, int(h.shape[0]),
                self._attn_stage_ms, self._attn_return_ms,
                self._attn_root_ms, self._attn_helper_ms,
                self.outproj_chunk_rows,
            )
        self._sample_free()
        return out

    def _run_post_slice(self, *, block, helper_mlp, x_slice, shift, scale, gate,
                        segments, global_start: int, helper_side: bool):
        import torch
        norm = helper_mlp.norm2_helper if helper_side else block.norm2
        mlp = helper_mlp if helper_side else block.mlp
        rows = self.mlp_chunk_rows
        out = torch.empty_like(x_slice)
        for off in range(0, int(x_slice.shape[0]), rows):
            end = min(int(x_slice.shape[0]), off + rows)
            xs = x_slice[off:end]
            h = norm(xs)
            _mod_scale_shift_range(h, shift, scale, segments, global_start + off)
            y = self._run_mlp(mlp, h)
            part = _mod_gate_range(xs, gate, y, segments, global_start + off)
            out[off:end].copy_(part)
            del h, y, part
            self._chunked_mlp_calls += 1
        return out

    def execute_post(self, index: int, block, x, shift_mlp, scale_mlp, gate_mlp, mod_segments):
        """Capacity version of Dev18 post-attention row island.

        It keeps the same exact token split but never fails open to full-GPU0 MLP.
        """
        import torch
        import comfy.model_management as mm

        index = int(index)
        seq = int(x.shape[0])
        owner = self.owner_device(index)
        helper = self.helper_device(index)
        if index not in self.active_blocks or seq < self.min_sequence_length:
            # Small sequences do not need pooled capacity. Chunk locally anyway so
            # SINGLE/Dual behavior does not unexpectedly allocate a full FFN tensor.
            return self._run_post_slice(
                block=block, helper_mlp=self.helper_mlp_by_block.get(index, block.mlp),
                x_slice=x, shift=shift_mlp, scale=scale_mlp, gate=gate_mlp,
                segments=mod_segments, global_start=0, helper_side=False,
            )
        if x.device != owner:
            raise RuntimeError(f"H3VM Capacity owner mismatch block={index}: x={x.device} owner={owner}")
        if not self.sidecar_ready():
            raise RuntimeError("H3VM Capacity helper GPU is not ready; refusing root-only fallback")

        helper_mlp = self.helper_mlp_by_block[index]
        cut, primary_fraction = self._split(index, seq)
        local_x = x[:cut]
        remote_x = x[cut:]
        local_start = 0
        remote_start = cut

        caller_stream = torch.cuda.current_stream(owner)
        owner_stream = self.compute_streams[owner]
        helper_stream = self.compute_streams[helper]
        owner_stream.wait_stream(caller_stream)

        o0 = torch.cuda.Event(enable_timing=True)
        o1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(owner), torch.cuda.stream(owner_stream), mm.cuda_device_context(owner):
            o0.record(owner_stream)
            local_out = self._run_post_slice(
                block=block, helper_mlp=helper_mlp, x_slice=local_x,
                shift=shift_mlp, scale=scale_mlp, gate=gate_mlp,
                segments=mod_segments, global_start=local_start, helper_side=False,
            )
            o1.record(owner_stream)

        mods = torch.stack((shift_mlp, scale_mlp, gate_mlp), dim=0).contiguous()
        host_x = self._host_buffer("capacity_stage_x", remote_x)
        host_m = self._host_buffer("capacity_stage_mods", mods)
        d2h = self.d2h_streams[owner]
        h2d = self.h2d_streams[helper]
        d0 = torch.cuda.Event(enable_timing=True); d1 = torch.cuda.Event(enable_timing=True)
        s0 = torch.cuda.Event(enable_timing=True); s1 = torch.cuda.Event(enable_timing=True)
        ready = torch.cuda.Event()
        d2h.wait_stream(caller_stream)
        with torch.cuda.device(owner), torch.cuda.stream(d2h):
            d0.record(d2h)
            host_x.copy_(remote_x, non_blocking=self._pinned_available)
            host_m.copy_(mods, non_blocking=self._pinned_available)
            d1.record(d2h)
        d1.synchronize()
        remote_in = torch.empty_like(remote_x, device=helper)
        remote_mods = torch.empty_like(mods, device=helper)
        with torch.cuda.device(helper), torch.cuda.stream(h2d):
            s0.record(h2d)
            remote_in.copy_(host_x, non_blocking=self._pinned_available)
            remote_mods.copy_(host_m, non_blocking=self._pinned_available)
            s1.record(h2d)
            ready.record(h2d)

        h0 = torch.cuda.Event(enable_timing=True)
        h1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(helper), torch.cuda.stream(helper_stream), mm.cuda_device_context(helper):
            helper_stream.wait_event(ready)
            h0.record(helper_stream)
            remote_out = self._run_post_slice(
                block=block, helper_mlp=helper_mlp, x_slice=remote_in,
                shift=remote_mods[0], scale=remote_mods[1], gate=remote_mods[2],
                segments=mod_segments, global_start=remote_start, helper_side=True,
            )
            h1.record(helper_stream)

        returned, returned_ready, return_events = self._return_to_owner(remote_out, owner, helper, h1)
        caller_stream.wait_event(o1)
        caller_stream.wait_event(returned_ready)
        out = torch.cat((local_out, returned), dim=0)
        local_out.record_stream(caller_stream)
        returned.record_stream(caller_stream)
        out.record_stream(caller_stream)

        self._pending[index] = {
            "owner": owner,
            "helper": helper,
            "cut": cut,
            "seq": seq,
            "owner_events": (o0, o1),
            "helper_events": (h0, h1),
            "stage_events": (d0, d1, s0, s1),
            "return_events": return_events,
            "shadow_enqueue_ms": 0.0,
            "primary_fraction": primary_fraction,
            "capacity": True,
        }
        self._sample_free()
        return out

    def execute_block(self, index: int, block, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        transformer_options = transformer_options or {}
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        h = block.norm1(x)
        _mod_scale_shift_range(h, shift_msa, scale_msa, mod_segments, 0)
        attn_out = self.execute_capacity_attention(
            index, block.attn, h, rope_freqs, transformer_options
        )
        x = _mod_gate_range(x, gate_msa, attn_out, mod_segments, 0)
        del h, attn_out
        return self.execute_post(index, block, x, shift_mlp, scale_mlp, gate_mlp, mod_segments)

    def on_block_complete(self, index: int):
        """Retire telemetry only; never disable helper work because it is 'late'."""
        index = int(index)
        rec = self._pending.pop(index, None)
        if rec is None:
            return
        owner_ms = self._elapsed(rec["owner_events"])
        helper_ms = self._elapsed(rec["helper_events"])
        stage_d2h = self._elapsed(rec["stage_events"][:2])
        stage_h2d = self._elapsed(rec["stage_events"][2:])
        ret_d2h = self._elapsed(rec["return_events"][:2])
        ret_h2d = self._elapsed(rec["return_events"][2:])
        shadow_ms = stage_d2h + stage_h2d + helper_ms + ret_d2h + ret_h2d
        slack_ms = owner_ms - shadow_ms
        stall_est = max(0.0, -slack_ms)
        self.calls += 1
        self._step_calls += 1
        self._step_owner_ms += owner_ms
        self._step_shadow_ms += shadow_ms
        self._step_stall_est_ms += stall_est
        self._step_slack_ms.append(slack_ms)
        if self.telemetry and (self.calls <= 2 or self.calls % 10 == 0):
            LOG.info(
                "H3VM CAPACITY MLP #%d block=%d rows=%d/%d | root=%.1fms helper_path=%.1fms "
                "stage=%.1f+%.1f compute=%.1f return=%.1f+%.1f | chunks=%d",
                self.calls, index, rec["cut"], rec["seq"] - rec["cut"],
                owner_ms, shadow_ms, stage_d2h, stage_h2d, helper_ms,
                ret_d2h, ret_h2d, self._chunked_mlp_calls,
            )

    def step_summary(self):
        # Keep runtime's expected CPM fields but reinterpret stalls as telemetry,
        # not as a scheduling failure.
        fracs = [self.primary_fraction]
        return {
            "calls": self._step_calls,
            "slack_min_ms": min(self._step_slack_ms) if self._step_slack_ms else 0.0,
            "slack_avg_ms": (sum(self._step_slack_ms) / len(self._step_slack_ms)) if self._step_slack_ms else 0.0,
            "primary_stall_est_ms": self._step_stall_est_ms,
            "root_compute_ms": self._step_owner_ms,
            "shadow_path_ms": self._step_shadow_ms,
            "disabled": 0,
            "skipped": 0,
            "startup_skipped": 0,
            "adjust_up": 0,
            "adjust_down": 0,
            "fraction_min": self.primary_fraction,
            "fraction_avg": self.primary_fraction,
            "fraction_max": self.primary_fraction,
        }

    def record_step_runtime(self, **rec):
        item = {
            "step": int(self._step),
            "mode": "DUAL_CAPACITY",
            "attention_calls": int(self._attn_calls),
            "attention_stage_ms_total": round(float(self._attn_stage_ms), 3),
            "attention_return_ms_total": round(float(self._attn_return_ms), 3),
            "attention_root_ms_total": round(float(self._attn_root_ms), 3),
            "attention_helper_ms_total": round(float(self._attn_helper_ms), 3),
            "mlp_microchunks": int(self._chunked_mlp_calls),
            "mlp_chunk_rows": int(self.mlp_chunk_rows),
            "outproj_chunk_rows": int(self.outproj_chunk_rows),
            "helper_heads": int(self.helper_heads),
            "attention_kernel": str(self.attention_kernel),
            "primary_fraction": float(self.primary_fraction),
            "min_root_free_mib": None if self._peak_root_free_mib is None else round(float(self._peak_root_free_mib), 1),
            "min_helper_free_mib": None if self._peak_helper_free_mib is None else round(float(self._peak_helper_free_mib), 1),
        }
        item.update({k: round(float(v), 3) for k, v in rec.items()})
        self._step_runtime.append(item)
        LOG.info("H3VM CAPACITY STEP RESULT %s", json.dumps(item, separators=(",", ":")))

    def close(self):
        self._helper_rope = None
        self.helper_attn_by_block.clear()
        self.helper_packet_by_block.clear()
        self._step_runtime.clear()
        super().close()


def install_capacity_mode(blocks, fabric: CapacityFabric):
    installed = 0
    fref = weakref.ref(fabric)
    for i, block in enumerate(blocks):
        if hasattr(block, "_h3vm_capacity_original_forward"):
            continue
        block._h3vm_capacity_original_forward = block.forward

        def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None,
                    _idx=int(i), _fref=fref, **kwargs):
            f = _fref()
            if f is None or kwargs:
                return self._h3vm_capacity_original_forward(
                    x, t_emb, mod_segments, rope_freqs,
                    transformer_options={} if transformer_options is None else transformer_options,
                    **kwargs,
                )
            return f.execute_block(
                _idx, self, x, t_emb, mod_segments, rope_freqs,
                {} if transformer_options is None else transformer_options,
            )

        block.forward = types.MethodType(forward, block)
        installed += 1
    return installed
