from __future__ import annotations

import copy
import json
import logging
import time
import types
import weakref

from .mlp_token_parallel import clone_mlp_shared, make_helper_block
from .rolling_helper_matrix import RollingCoverageMLPFabric

LOG = logging.getLogger("H3VM")


def _clone_norm_shared(src):
    """Clone a tiny RMSNorm shell while sharing CPU backing storage.

    The helper norm is registered under the same rolling DynamicVRAM helper
    module as FC1/FC2. The tensor is tiny (~5k values per block), but keeping it
    in the helper tree gives the queue one atomic post-attention work packet.
    """
    import torch

    m = copy.copy(src)
    m._parameters = dict(getattr(src, "_parameters", {}))
    m._buffers = dict(getattr(src, "_buffers", {}))
    m._modules = dict(getattr(src, "_modules", {}))
    if getattr(src, "weight", None) is not None:
        m.weight = torch.nn.Parameter(src.weight, requires_grad=False)
    if getattr(src, "bias", None) is not None:
        m.bias = torch.nn.Parameter(src.bias, requires_grad=False)
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


def build_post_attention_helper_maps(blocks, owner_map, helper_indices):
    """Build the Dev18 rolling helper universe.

    Canonical helper paths stay ``blocks.N.mlp.fc1/fc2`` so Larry Turbo LoRA
    patches keep resolving exactly as in Dev17. The helper norm is simply an
    extra child of the MLP helper packet and therefore rides the same prefetch
    queue without another residency mechanism.
    """
    selected = {int(i) for i in helper_indices}
    primary_helpers = {}
    secondary_helpers = {}
    helper_mlp_by_block = {}
    for i, block in enumerate(blocks):
        if int(i) not in selected:
            continue
        helper_mlp = clone_mlp_shared(block.mlp)
        helper_mlp.norm2_helper = _clone_norm_shared(block.norm2)
        helper_mlp_by_block[int(i)] = helper_mlp
        hb = make_helper_block(helper_mlp, i)
        if int(owner_map[int(i)]) == 0:
            secondary_helpers[int(i)] = hb
        else:
            primary_helpers[int(i)] = hb
    return primary_helpers, secondary_helpers, helper_mlp_by_block


def _mod_scale_shift_range(h, shift, scale, segments, global_start: int):
    """Exact H3 scale/shift on an arbitrary contiguous token-row slice."""
    start = int(global_start)
    stop = start + int(h.shape[0])
    for a, b, row in segments:
        aa = max(int(a), start)
        bb = min(int(b), stop)
        if aa >= bb:
            continue
        la = aa - start
        lb = bb - start
        h[la:lb].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
    return h


def _mod_gate_range(x, gate, other, segments, global_start: int):
    """Exact H3 gated residual on an arbitrary contiguous token-row slice."""
    start = int(global_start)
    stop = start + int(x.shape[0])
    for a, b, row in segments:
        aa = max(int(a), start)
        bb = min(int(b), stop)
        if aa >= bb:
            continue
        la = aa - start
        lb = bb - start
        x[la:lb].addcmul_(other[la:lb], gate[row].to(x.dtype))
    return x


class PostAttentionRowIslandFabric(RollingCoverageMLPFabric):
    """Dev18 A/B experiment: Dev17 MLP-only vs a larger post-attention row island.

    Step1: warmup MLP-only baseline (50 blocks, 68/32)
    Step2: steady MLP-only baseline (50 blocks, 68/32)
    Step3: post-attention island (Norm2 -> mod -> MLP -> gate/residual)
    Step4: repeat post-attention island

    Cross-GPU contract remains ONE-IN / ONE-OUT for the row state. Modulation
    tensors are a tiny side packet and do not contain sequence-sized activations.
    GPU0 attention stays untouched and remains the critical spine.
    """

    post_attention_island_mode = True
    rolling_helper_matrix_mode = True

    def __init__(self, *args, **kwargs):
        kwargs["coverage_schedule"] = (50, 50, 50, 50)
        kwargs["primary_fraction"] = 0.68
        kwargs["adaptive_slack"] = True
        kwargs["min_primary_fraction"] = 0.68
        kwargs["max_primary_fraction"] = 0.72
        kwargs["target_slack_ms"] = 9999.0
        kwargs["harvest_confirmations"] = 99
        super().__init__(*args, **kwargs)
        self._phase = "WARMUP_BASELINE"
        self._island_enabled = False
        self._phase_runtime = []

    def _reset_sample(self):
        super()._reset_sample()
        self._phase_runtime.clear()
        self._phase = "WARMUP_BASELINE"
        self._island_enabled = False

    def begin_step(self, step: int):
        if int(step) <= 1 and self._phase_runtime:
            self._phase_runtime.clear()
        super().begin_step(step)
        s = int(step)
        if s <= 1:
            self._phase = "WARMUP_BASELINE"
            self._island_enabled = False
        elif s == 2:
            self._phase = "MLP_ONLY_BASELINE"
            self._island_enabled = False
        elif s == 3:
            self._phase = "POST_ATTN_ISLAND"
            self._island_enabled = True
        else:
            self._phase = "POST_ATTN_ISLAND_REPEAT"
            self._island_enabled = True
        # Coverage experiment base resets the same 68/32 split for all blocks.
        LOG.info(
            "H3VM DEV18 PLAN step=%d phase=%s | coverage=50/50 root/helper=68/32 | "
            "post_island=%s contract=ONE-ROW-IN/ONE-ROW-OUT",
            s, self._phase, self._island_enabled,
        )

    def _stage_post_packet(self, remote_x, mods, owner, helper, caller_stream):
        """One synchronized host-relay runway for row-state + tiny modulation packet."""
        import torch

        host_x = self._host_buffer("dev18_stage_x", remote_x)
        host_m = self._host_buffer("dev18_stage_mods", mods)
        d2h = self.d2h_streams[owner]
        h2d = self.h2d_streams[helper]

        d0 = torch.cuda.Event(enable_timing=True)
        d1 = torch.cuda.Event(enable_timing=True)
        h0 = torch.cuda.Event(enable_timing=True)
        h1 = torch.cuda.Event(enable_timing=True)
        ready = torch.cuda.Event()

        d2h.wait_stream(caller_stream)
        with torch.cuda.device(owner), torch.cuda.stream(d2h):
            d0.record(d2h)
            host_x.copy_(remote_x, non_blocking=self._pinned_available)
            host_m.copy_(mods, non_blocking=self._pinned_available)
            d1.record(d2h)
        d1.synchronize()

        remote = torch.empty_like(remote_x, device=helper)
        remote_mods = torch.empty_like(mods, device=helper)
        with torch.cuda.device(helper), torch.cuda.stream(h2d):
            h0.record(h2d)
            remote.copy_(host_x, non_blocking=self._pinned_available)
            remote_mods.copy_(host_m, non_blocking=self._pinned_available)
            h1.record(h2d)
            ready.record(h2d)
        return remote, remote_mods, ready, (d0, d1, h0, h1)

    def _run_post_slice(self, *, block, helper_mlp, x_slice, shift, scale, gate,
                        segments, global_start: int, helper_side: bool):
        norm = helper_mlp.norm2_helper if helper_side else block.norm2
        h = norm(x_slice)
        _mod_scale_shift_range(h, shift, scale, segments, global_start)
        y = self._run_mlp(helper_mlp if helper_side else block.mlp, h)
        return _mod_gate_range(x_slice, gate, y, segments, global_start)

    def execute_post(self, index: int, block, x, shift_mlp, scale_mlp, gate_mlp, mod_segments):
        import torch
        import comfy.model_management as mm

        index = int(index)
        seq = int(x.shape[0])
        owner = self.owner_device(index)
        helper = self.helper_device(index)

        if (
            index not in self.active_blocks
            or index in self._disabled_blocks
            or seq < self.min_sequence_length
            or not self.sidecar_ready()
            or not self._helper_has_runway(helper)
        ):
            h = block.norm2(x)
            _mod_scale_shift_range(h, shift_mlp, scale_mlp, mod_segments, 0)
            y = self._run_mlp(block.mlp, h)
            return _mod_gate_range(x, gate_mlp, y, mod_segments, 0)
        if x.device != owner:
            raise RuntimeError(f"H3VM Dev18 owner mismatch block={index}: x={x.device} owner={owner}")

        # Consume the same all-50 rolling helper queue proven by Dev17. The
        # helper MLP packet now also contains the tiny norm2 helper.
        self._consume_helper_prefetch(index)
        helper_mlp = self.helper_mlp_by_block[index]
        cut, primary_fraction = self._split(index, seq)

        if owner == self.primary:
            local_x = x[:cut]
            remote_x = x[cut:]
            local_start = 0
            remote_start = cut
            remote_first = False
        else:
            remote_x = x[:cut]
            local_x = x[cut:]
            remote_start = 0
            local_start = cut
            remote_first = True

        caller_stream = torch.cuda.current_stream(owner)
        owner_stream = self.compute_streams[owner]
        helper_stream = self.compute_streams[helper]
        owner_stream.wait_stream(caller_stream)

        # Launch the complete local post-attention slice first. GPU1 logistics
        # starts only after GPU0 has work in flight.
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
        remote_in, remote_mods, remote_ready, stage_events = self._stage_post_packet(
            remote_x.contiguous(), mods, owner, helper, caller_stream
        )

        h0 = torch.cuda.Event(enable_timing=True)
        h1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(helper), torch.cuda.stream(helper_stream), mm.cuda_device_context(helper):
            helper_stream.wait_event(remote_ready)
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
        if remote_first:
            out = torch.cat((returned, local_out), dim=0)
        else:
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
            "stage_events": stage_events,
            "return_events": return_events,
            "shadow_enqueue_ms": 0.0,
            "primary_fraction": primary_fraction,
            "dev18_kind": "post_attention_island",
        }
        return out

    def execute_block(self, index: int, block, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        """Execute the exact H3 DiT block, changing only the post-attention row placement."""
        transformer_options = transformer_options or {}
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)

        h = block.norm1(x)
        _mod_scale_shift_range(h, shift_msa, scale_msa, mod_segments, 0)
        attn_out = block.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
        x = _mod_gate_range(x, gate_msa, attn_out, mod_segments, 0)

        if not self._island_enabled:
            # Exact Dev17 baseline inside the same block wrapper: Norm2/modulation
            # remain on GPU0 and only FC1->SwiGLU->FC2 rows are split.
            h2 = block.norm2(x)
            _mod_scale_shift_range(h2, shift_mlp, scale_mlp, mod_segments, 0)
            y = self.execute(index, block.mlp, h2)
            return _mod_gate_range(x, gate_mlp, y, mod_segments, 0)

        return self.execute_post(index, block, x, shift_mlp, scale_mlp, gate_mlp, mod_segments)

    def record_step_runtime(self, **rec):
        item = {
            "step": int(self._step),
            "phase": self._phase,
            "post_island": bool(self._island_enabled),
            "coverage": 50,
            "actual_sidecar_calls": int(self._step_calls),
            "helper_prefetch_host_ms": round(float(self._helper_prefetch_host_ms), 3),
            "helper_prefetch_prime_ms": round(float(self._helper_prefetch_prime_ms), 3),
            "helper_prefetch_calls": int(self._helper_prefetch_calls),
            "helper_prefetch_errors": int(self._helper_prefetch_errors),
            "primary_wait_helper_est_ms": round(float(self._step_stall_est_ms), 3),
            "slack_avg_ms": round(float(sum(self._step_slack_ms) / len(self._step_slack_ms)), 3) if self._step_slack_ms else 0.0,
        }
        item.update({k: round(float(v), 3) for k, v in rec.items()})
        self._phase_runtime.append(item)

        if int(self._step) >= 4:
            baseline = next((x for x in self._phase_runtime if x.get("phase") == "MLP_ONLY_BASELINE"), None)
            islands = [x for x in self._phase_runtime if x.get("post_island") and x.get("actual_sidecar_calls", 0) > 0]
            best_island = min(islands, key=lambda x: x.get("wall_ms", 1e30)) if islands else None
            comparison = None
            if baseline is not None and best_island is not None:
                comparison = {
                    "baseline_wall_ms": baseline.get("wall_ms"),
                    "best_island_wall_ms": best_island.get("wall_ms"),
                    "wall_delta_ms": round(float(best_island.get("wall_ms", 0.0) - baseline.get("wall_ms", 0.0)), 3),
                    "baseline_primary_compute_ms": baseline.get("primary_compute_ms"),
                    "best_island_primary_compute_ms": best_island.get("primary_compute_ms"),
                    "primary_compute_delta_ms": round(float(best_island.get("primary_compute_ms", 0.0) - baseline.get("primary_compute_ms", 0.0)), 3),
                    "best_island_primary_wait_helper_ms": best_island.get("primary_wait_helper_est_ms"),
                }
            verdict = {
                "mode": "DEV18_POST_ATTENTION_ROW_ISLAND_AB",
                "contract": "GPU0_ATTN_SPINE__ALL50_ROLLING__POST_ATTN_ONE_ROW_IN_ONE_ROW_OUT",
                "fixed_root_fraction": 0.68,
                "steps": self._phase_runtime,
                "comparison": comparison,
            }
            LOG.info("H3VM DEV18 FINAL RESULT %s", json.dumps(verdict, separators=(",", ":")))

    def close(self):
        self._phase_runtime.clear()
        super().close()


def install_post_attention_island(blocks, fabric: PostAttentionRowIslandFabric):
    """Patch DiTBlock.forward instead of MLP.forward.

    Unexpected/new forward signatures fail open to the original ComfyUI block,
    which protects compatibility if upstream adds a precomputed argument path.
    """
    installed = 0
    fref = weakref.ref(fabric)
    for i, block in enumerate(blocks):
        if hasattr(block, "_h3vm_dev18_original_forward"):
            continue
        block._h3vm_dev18_original_forward = block.forward

        def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None,
                    _idx=int(i), _fref=fref, **kwargs):
            f = _fref()
            if f is None or kwargs:
                return self._h3vm_dev18_original_forward(
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
