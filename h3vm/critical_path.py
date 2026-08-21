from __future__ import annotations

import logging
import math
import time

LOG = logging.getLogger("H3VM")
MIB = 1024 ** 2


def select_sidecar_blocks(total_blocks: int, count: int, start_block: int = 0):
    """Deterministically spread a fixed sidecar work package across the network.

    Dev14.1 can reserve a short *launch runway* at the beginning of the H3 graph.
    GPU0 starts the critical path immediately while GPU1 prepares its resident
    sidecar package in the background.  Sidecar MLP work is then spread evenly
    over the remaining blocks. Whole-block ownership never leaves the primary.
    """
    total_blocks = max(1, int(total_blocks))
    start_block = max(0, min(int(start_block), total_blocks - 1))
    available = total_blocks - start_block
    count = max(0, min(int(count), available))
    if count == 0:
        return []
    out = []
    for k in range(count):
        idx = start_block + int(round((k + 0.5) * available / count - 0.5))
        idx = max(start_block, min(total_blocks - 1, idx))
        if idx not in out:
            out.append(idx)
    if len(out) < count:
        for idx in range(start_block, total_blocks):
            if idx not in out:
                out.append(idx)
                if len(out) == count:
                    break
    return sorted(out)


def build_selected_helper_maps(blocks, owner_map, helper_indices):
    """Build MLP mirrors only for the fixed sidecar work package."""
    from .mlp_token_parallel import clone_mlp_shared, make_helper_block

    selected = {int(i) for i in helper_indices}
    primary_helpers = {}
    secondary_helpers = {}
    helper_mlp_by_block = {}
    for i, block in enumerate(blocks):
        if int(i) not in selected:
            continue
        helper_mlp = clone_mlp_shared(block.mlp)
        helper_mlp_by_block[int(i)] = helper_mlp
        hb = make_helper_block(helper_mlp, i)
        if int(owner_map[int(i)]) == 0:
            secondary_helpers[int(i)] = hb
        else:
            primary_helpers[int(i)] = hb
    return primary_helpers, secondary_helpers, helper_mlp_by_block


class CriticalPathMLPFabric:
    """Dev14 exact MLP sidecar scheduler.

    Primary rule: the root CUDA work is launched *before* any helper staging.
    The secondary is allowed to use only a fixed, resident MLP package.  Its
    input DMA, helper compute and result DMA are shadow work.  The root never
    waits for helper preparation.  The only dependency is the mathematically
    required merge point.

    Exactness is unchanged because MLP rows/tokens are independent and are
    concatenated back in original token order.
    """

    def __init__(self, *, primary, secondary, owner_map, helper_mlp_by_block,
                 primary_fraction=0.72, min_sequence_length=8192,
                 helper_safety_mb=768, primary_stall_budget_ms=1.0,
                 adaptive_slack=True, min_primary_fraction=0.68,
                 max_primary_fraction=0.84, fraction_step=0.02,
                 target_slack_ms=8.0, resident_sidecar_packet=False, harvest_confirmations=1, telemetry=True):
        import torch

        self.critical_path = True
        self.primary = torch.device(primary)
        self.secondary = torch.device(secondary)
        self.owner_map = dict(owner_map)
        self.helper_mlp_by_block = dict(helper_mlp_by_block)
        self.active_blocks = set(self.helper_mlp_by_block)
        self.primary_fraction = min(0.90, max(0.55, float(primary_fraction)))
        self.min_primary_fraction = min(self.primary_fraction, max(0.55, float(min_primary_fraction)))
        self.max_primary_fraction = max(self.primary_fraction, min(0.92, float(max_primary_fraction)))
        self.fraction_step = max(0.005, min(0.08, float(fraction_step)))
        self.target_slack_ms = max(0.0, float(target_slack_ms))
        self.adaptive_slack = bool(adaptive_slack)
        self.resident_sidecar_packet = bool(resident_sidecar_packet)
        # Dev15: do not chase one noisy helper sample. A deeper cut is allowed
        # only after repeated positive slack on the same block; deadline misses
        # still return work to GPU0 immediately.
        self.harvest_confirmations = max(1, int(harvest_confirmations))
        self._positive_slack_streak = {int(i): 0 for i in self.active_blocks}
        self.block_primary_fraction = {int(i): self.primary_fraction for i in self.active_blocks}
        self._miss_counts = {int(i): 0 for i in self.active_blocks}
        self._sidecar_ready = True
        self._warmup_ms = 0.0
        self.min_sequence_length = max(1, int(min_sequence_length))
        self.helper_safety_mb = max(128, int(helper_safety_mb))
        self.primary_stall_budget_ms = max(0.0, float(primary_stall_budget_ms))
        self.telemetry = bool(telemetry)

        self.compute_streams = {
            self.primary: torch.cuda.Stream(device=self.primary),
            self.secondary: torch.cuda.Stream(device=self.secondary),
        }
        self.d2h_streams = {
            self.primary: torch.cuda.Stream(device=self.primary),
            self.secondary: torch.cuda.Stream(device=self.secondary),
        }
        self.h2d_streams = {
            self.primary: torch.cuda.Stream(device=self.primary),
            self.secondary: torch.cuda.Stream(device=self.secondary),
        }

        self._host_buffers = {}
        self._pinned_available = True
        self._pending = {}
        self._disabled_blocks = set()
        self._step = 0
        self._step_calls = 0
        self._step_stall_est_ms = 0.0
        self._step_slack_ms = []
        self._step_owner_ms = 0.0
        self._step_shadow_ms = 0.0
        self._step_skipped = 0
        self._step_startup_skipped = 0
        self._step_adjust_up = 0
        self._step_adjust_down = 0
        self.calls = 0

    @staticmethod
    def _run_mlp(mlp, x):
        import comfy.ops
        return comfy.ops.linear_input_act(mlp.fc2, mlp.fc1(x), "swiglu")

    def begin_step(self, step: int):
        self._step = int(step)
        self._step_calls = 0
        self._step_stall_est_ms = 0.0
        self._step_slack_ms = []
        self._step_owner_ms = 0.0
        self._step_shadow_ms = 0.0
        self._step_skipped = 0
        self._step_startup_skipped = 0
        self._step_adjust_up = 0
        self._step_adjust_down = 0

    def set_sidecar_ready(self, ready: bool):
        self._sidecar_ready = bool(ready)

    def sidecar_ready(self):
        return bool(self._sidecar_ready)

    def warmup_sidecar(self, rows: int = 128):
        """Warm helper kernels off the GPU0 critical line.

        Dev14.1 keeps the historical one-module warmup. Dev14.2 marks the helper
        patcher as a bounded resident packet and warms every selected MLP with a
        tiny row batch only after the packet is fully resident.
        """
        if not self.helper_mlp_by_block:
            return 0.0
        import torch
        import comfy.model_management as mm
        indices = sorted(self.helper_mlp_by_block)
        if self.resident_sidecar_packet:
            rows = 32
            warm_indices = indices
            tag = "Dev14.2 SIDECAR packet warmup"
        else:
            rows = max(16, min(int(rows), 256))
            warm_indices = indices[:1]
            tag = "Dev14.1 SIDECAR warmup"
        first = self.helper_mlp_by_block[warm_indices[0]]
        hidden = int(getattr(getattr(first, "fc1", None), "in_features", 0) or 5376)
        stream = self.compute_streams[self.secondary]
        t0 = time.perf_counter()
        try:
            with torch.cuda.device(self.secondary), torch.cuda.stream(stream), mm.cuda_device_context(self.secondary):
                x = torch.zeros((rows, hidden), device=self.secondary, dtype=torch.bfloat16)
                for idx in warm_indices:
                    y = self._run_mlp(self.helper_mlp_by_block[idx], x)
                    del y
            stream.synchronize()
            del x
            self._warmup_ms = (time.perf_counter() - t0) * 1000.0
            if self.resident_sidecar_packet:
                LOG.info(
                    "H3VM Dev14.2 SIDECAR packet warmup | modules=%d rows=%d hidden=%d wall=%.1fms",
                    len(warm_indices), rows, hidden, self._warmup_ms,
                )
            else:
                LOG.info(
                    "H3VM Dev14.1 SIDECAR warmup | rows=%d hidden=%d wall=%.1fms",
                    rows, hidden, self._warmup_ms,
                )
            return self._warmup_ms
        except Exception as exc:
            LOG.warning("H3VM %s skipped: %r", tag, exc)
            return 0.0

    def prefetch_helper(self, index: int, transformer_options=None):
        # Dev14 intentionally has no per-block helper prefetch.  The selected
        # helper patcher is prepared once as a bounded resident sidecar package.
        return None

    def helper_device(self, index: int):
        return self.secondary if int(self.owner_map[int(index)]) == 0 else self.primary

    def owner_device(self, index: int):
        return self.primary if int(self.owner_map[int(index)]) == 0 else self.secondary

    def _split(self, index: int, seq: int):
        frac = float(self.block_primary_fraction.get(int(index), self.primary_fraction))
        cut = int(round(seq * frac / 128.0)) * 128
        cut = max(128, min(seq - 128, cut)) if seq >= 256 else max(1, seq // 2)
        return cut, frac

    def _helper_has_runway(self, helper):
        import torch
        try:
            free, _ = torch.cuda.mem_get_info(helper)
            return int(free) > self.helper_safety_mb * MIB
        except Exception:
            return True

    def _host_buffer(self, key, like):
        import torch
        shape = tuple(like.shape)
        dtype = like.dtype
        cache_key = (str(key), shape, str(dtype))
        buf = self._host_buffers.get(cache_key)
        if buf is not None:
            return buf
        try:
            buf = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        except Exception:
            self._pinned_available = False
            buf = torch.empty(shape, dtype=dtype, device="cpu")
        self._host_buffers[cache_key] = buf
        return buf

    def _stage_to_helper(self, remote_x, owner, helper, caller_stream):
        """GPU0->host->GPU1 after primary compute has already launched."""
        import torch
        host = self._host_buffer("stage", remote_x)
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
            host.copy_(remote_x, non_blocking=self._pinned_available)
            d1.record(d2h)
        # CPU waits only for the transfer runway.  The primary compute stream is
        # independent and keeps running during this wait.
        d1.synchronize()

        remote = torch.empty_like(remote_x, device=helper)
        with torch.cuda.device(helper), torch.cuda.stream(h2d):
            h0.record(h2d)
            remote.copy_(host, non_blocking=self._pinned_available)
            h1.record(h2d)
            ready.record(h2d)
        return remote, ready, (d0, d1, h0, h1)

    def _return_to_owner(self, remote_out, owner, helper, helper_done):
        """GPU1->host->GPU0 without a device-wide primary synchronize."""
        import torch
        host = self._host_buffer("return", remote_out)
        d2h = self.d2h_streams[helper]
        h2d = self.h2d_streams[owner]

        d0 = torch.cuda.Event(enable_timing=True)
        d1 = torch.cuda.Event(enable_timing=True)
        h0 = torch.cuda.Event(enable_timing=True)
        h1 = torch.cuda.Event(enable_timing=True)
        ready = torch.cuda.Event()

        d2h.wait_event(helper_done)
        with torch.cuda.device(helper), torch.cuda.stream(d2h):
            d0.record(d2h)
            host.copy_(remote_out, non_blocking=self._pinned_available)
            d1.record(d2h)
        # Again, only the sidecar transfer thread is waited on. GPU0 local MLP
        # continues on its own compute stream.
        d1.synchronize()

        returned = torch.empty_like(remote_out, device=owner)
        with torch.cuda.device(owner), torch.cuda.stream(h2d):
            h0.record(h2d)
            returned.copy_(host, non_blocking=self._pinned_available)
            h1.record(h2d)
            ready.record(h2d)
        return returned, ready, (d0, d1, h0, h1)

    def execute(self, index: int, owner_mlp, x):
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
            or not self._sidecar_ready
            or not self._helper_has_runway(helper)
        ):
            if index in self.active_blocks:
                self._step_skipped += 1
                if not self._sidecar_ready:
                    self._step_startup_skipped += 1
            return self._run_mlp(owner_mlp, x)
        if x.device != owner:
            raise RuntimeError(f"H3VM Dev14 MLP owner mismatch block={index}: x={x.device} owner={owner}")

        helper_mlp = self.helper_mlp_by_block[index]
        cut, primary_fraction = self._split(index, seq)
        x_primary = x[:cut]
        x_secondary = x[cut:]
        if owner == self.primary:
            local_x, remote_x = x_primary, x_secondary
            remote_first = False
        else:
            remote_x, local_x = x_primary, x_secondary
            remote_first = True

        caller_stream = torch.cuda.current_stream(owner)
        owner_stream = self.compute_streams[owner]
        helper_stream = self.compute_streams[helper]
        owner_stream.wait_stream(caller_stream)

        # Critical-path rule #1: launch the primary CUDA work first. Nothing on
        # the sidecar path may delay this launch.
        o0 = torch.cuda.Event(enable_timing=True)
        o1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(owner), torch.cuda.stream(owner_stream), mm.cuda_device_context(owner):
            o0.record(owner_stream)
            local_out = self._run_mlp(owner_mlp, local_x)
            o1.record(owner_stream)

        # Shadow path starts only after primary work is already in flight.
        t_shadow = time.perf_counter()
        remote_in, remote_ready, stage_events = self._stage_to_helper(
            remote_x.contiguous(), owner, helper, caller_stream
        )

        h0 = torch.cuda.Event(enable_timing=True)
        h1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(helper), torch.cuda.stream(helper_stream), mm.cuda_device_context(helper):
            helper_stream.wait_event(remote_ready)
            h0.record(helper_stream)
            remote_out = helper_mlp(remote_in)
            h1.record(helper_stream)

        returned, returned_ready, return_events = self._return_to_owner(
            remote_out, owner, helper, h1
        )
        shadow_enqueue_ms = (time.perf_counter() - t_shadow) * 1000.0

        # Critical-path rule #2: the only legal dependency is the exact merge.
        # CUDA stream waits preserve ordering without a host/device-wide sync.
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
            "shadow_enqueue_ms": shadow_enqueue_ms,
            "primary_fraction": primary_fraction,
        }
        return out

    @staticmethod
    def _elapsed(pair):
        a, b = pair
        try:
            return float(a.elapsed_time(b))
        except Exception:
            return 0.0

    def on_block_complete(self, index: int):
        """Called after the root block has completed; safe telemetry point."""
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

        # Dev14.1 Slack Harvester: positive CPM slack is budget that can be
        # converted into less GPU0 work on the *next* denoise step. Deadline
        # misses immediately move work back to GPU0. A block is disabled only
        # after two consecutive misses at the conservative end of the range.
        frac_now = float(rec.get("primary_fraction", self.primary_fraction))
        frac_next = frac_now
        action = "hold"
        if self.adaptive_slack:
            if stall_est > self.primary_stall_budget_ms:
                frac_next = min(self.max_primary_fraction, frac_now + self.fraction_step)
                self._miss_counts[index] = int(self._miss_counts.get(index, 0)) + 1
                self._positive_slack_streak[index] = 0
                self._step_adjust_up += int(frac_next > frac_now)
                action = "return_to_root"
                if self._miss_counts[index] >= 2 and frac_next >= self.max_primary_fraction - 1e-9:
                    self._disabled_blocks.add(index)
                    action = "disable"
            elif slack_ms > self.target_slack_ms + 4.0:
                self._miss_counts[index] = 0
                streak = int(self._positive_slack_streak.get(index, 0)) + 1
                self._positive_slack_streak[index] = streak
                # The first 68->66 cut is cheap and already validated. Deeper
                # cuts require repeated proof so bimodal helper kernels cannot
                # yank the critical line back and forth every denoise step.
                needed = 1 if frac_now >= 0.68 - 1e-9 else self.harvest_confirmations
                if streak >= needed:
                    frac_next = max(self.min_primary_fraction, frac_now - self.fraction_step)
                    self._positive_slack_streak[index] = 0
                    self._step_adjust_down += int(frac_next < frac_now)
                    action = "harvest"
                else:
                    action = "confirm_slack"
            else:
                self._miss_counts[index] = 0
                self._positive_slack_streak[index] = 0
        elif stall_est > self.primary_stall_budget_ms:
            self._disabled_blocks.add(index)
            action = "disable"
        self.block_primary_fraction[index] = frac_next

        if stall_est > self.primary_stall_budget_ms:
            LOG.warning(
                "H3VM Dev14.1 PRIMARY PATH GUARD | block=%d action=%s | owner=%.1fms shadow=%.1fms "
                "slack=%.1fms stall_est=%.1fms | root_fraction %.2f->%.2f",
                index, action, owner_ms, shadow_ms, slack_ms, stall_est, frac_now, frac_next,
            )

        if self.telemetry and (self.calls <= 4 or stall_est > self.primary_stall_budget_ms or action == "harvest"):
            LOG.info(
                "H3VM Dev14.1 CPM MLP #%d | block=%d rows(root/helper)=%d/%d | "
                "root=%.1fms shadow=%.1fms [stage %.1f+%.1f compute %.1f return %.1f+%.1f] | "
                "slack=%+.1fms stall=%.1fms | root_fraction %.2f->%.2f action=%s",
                self.calls, index, rec["cut"], rec["seq"] - rec["cut"],
                owner_ms, shadow_ms, stage_d2h, stage_h2d, helper_ms, ret_d2h, ret_h2d,
                slack_ms, stall_est, frac_now, frac_next, action,
            )

    def step_summary(self):
        if not self._step_slack_ms:
            return {
                "calls": 0, "slack_min_ms": 0.0, "slack_avg_ms": 0.0,
                "primary_stall_est_ms": 0.0, "disabled": len(self._disabled_blocks),
                "skipped": self._step_skipped, "startup_skipped": self._step_startup_skipped,
                "adjust_up": self._step_adjust_up, "adjust_down": self._step_adjust_down,
                "fraction_min": self.primary_fraction, "fraction_avg": self.primary_fraction,
                "fraction_max": self.primary_fraction,
            }
        fracs = [float(self.block_primary_fraction.get(i, self.primary_fraction)) for i in self.active_blocks if i not in self._disabled_blocks]
        return {
            "calls": self._step_calls,
            "slack_min_ms": min(self._step_slack_ms),
            "slack_avg_ms": sum(self._step_slack_ms) / len(self._step_slack_ms),
            "primary_stall_est_ms": self._step_stall_est_ms,
            "root_compute_ms": self._step_owner_ms,
            "shadow_path_ms": self._step_shadow_ms,
            "disabled": len(self._disabled_blocks),
            "skipped": self._step_skipped,
            "startup_skipped": self._step_startup_skipped,
            "adjust_up": self._step_adjust_up,
            "adjust_down": self._step_adjust_down,
            "fraction_min": min(fracs) if fracs else self.primary_fraction,
            "fraction_avg": (sum(fracs) / len(fracs)) if fracs else self.primary_fraction,
            "fraction_max": max(fracs) if fracs else self.primary_fraction,
        }

    def close(self):
        self._pending.clear()
        self._host_buffers.clear()
        self.helper_mlp_by_block.clear()
