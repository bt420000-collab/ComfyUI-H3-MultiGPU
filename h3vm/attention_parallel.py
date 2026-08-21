"""Two-GPU exact H3 attention-head parallelism.

Derived from AesSedai/ComfyUI-MiniMaxH3-Parallel (MIT, Copyright (c) 2026 AesSedai).
Dev5 changes:
- restrict helpers to the H3VM-selected primary/secondary pair;
- cache device order separately for either root GPU;
- support SM-weighted head ranges for heterogeneous 5060 Ti + 5060 style pairs;
- auto fallback mode for a reliable H3VM loader.
"""
from __future__ import annotations
import logging

LOG = logging.getLogger("H3VM")

from .head_plan import balanced_counts, weighted_counts, counts_to_ranges


class H3VMTwoGPUAttentionParallel:
    __slots__ = (
        "pair", "mode", "head_balance", "min_sequence_length", "streams",
        "enabled", "disable_reason", "_logged_roots",
    )

    def __init__(self, primary_device, secondary_device, *, mode="auto",
                 head_balance="sm_weighted", min_sequence_length=16384):
        import torch
        self.pair = (torch.device(primary_device), torch.device(secondary_device))
        self.mode = mode
        self.head_balance = head_balance
        self.min_sequence_length = int(min_sequence_length)
        self.streams = {}
        self.enabled = False
        self.disable_reason = None
        self._logged_roots = set()
        self._preflight()

    def _preflight(self):
        import torch
        if self.mode == "off":
            self.disable_reason = "attention_mode=off"
            return
        try:
            import comfy_kitchen
        except Exception as e:
            self.disable_reason = f"comfy_kitchen import failed: {e}"
            if self.mode == "force_2gpu":
                raise RuntimeError(self.disable_reason) from e
            return

        a, b = self.pair
        failures = []
        for d in (a, b):
            try:
                if not comfy_kitchen.int8_attention_is_available(d):
                    failures.append(f"CK INT8 attention unavailable on {d}")
            except Exception as e:
                failures.append(f"CK INT8 check failed on {d}: {e}")
        fn = getattr(torch.cuda, "can_device_access_peer", None)
        if fn is None:
            failures.append("torch.cuda.can_device_access_peer unavailable")
        else:
            try:
                if not fn(a.index, b.index):
                    failures.append(f"P2P {a}->{b} unavailable")
                if not fn(b.index, a.index):
                    failures.append(f"P2P {b}->{a} unavailable")
            except Exception as e:
                failures.append(f"P2P query failed: {e}")

        if failures:
            self.disable_reason = "; ".join(failures)
            if self.mode == "force_2gpu":
                raise RuntimeError("H3VM Dev5 forced attention parallel preflight failed: " + self.disable_reason)
            LOG.warning("H3VM Dev5 attention parallel auto-disabled | %s", self.disable_reason)
            return

        for d in (a, b):
            self.streams[d] = torch.cuda.Stream(device=d)
        self.enabled = True

    def _devices(self, root_device):
        import torch
        root = torch.device(root_device)
        if root.type != "cuda":
            raise RuntimeError("H3VM attention parallel requires CUDA root")
        if root.index is None:
            root = torch.device("cuda", torch.cuda.current_device())
        a, b = self.pair
        if root == a:
            return [a, b]
        if root == b:
            return [b, a]
        raise RuntimeError(f"H3VM attention root {root} is outside configured pair {self.pair}")

    def _ranges(self, heads, devices):
        import torch
        if self.head_balance == "sm_weighted":
            weights = [max(1, torch.cuda.get_device_properties(d).multi_processor_count) for d in devices]
            counts = weighted_counts(heads, weights)
        else:
            counts = balanced_counts(heads, len(devices))
        return counts_to_ranges(counts), counts

    def __call__(self, _, *args, **kwargs):
        from comfy.ldm.modules.attention import attention_comfy_kitchen_int8
        kwargs["_inside_attn_wrapper"] = True
        return attention_comfy_kitchen_int8(*args, **kwargs)

    def container_function(self, q, k, v, heads, mask=None, attn_precision=None,
                           skip_reshape=False, skip_output_reshape=False, **kwargs):
        import torch
        import comfy_kitchen
        from comfy.ldm.modules.attention import _comfy_kitchen_int8_inputs

        q = q.take()
        k = k.take()
        v = v.take()
        q, k, v, mask, batch, dim_head = _comfy_kitchen_int8_inputs(
            q, k, v, heads, mask, skip_reshape, kwargs.get("enable_gqa", False)
        )
        if not self.enabled:
            quantized = comfy_kitchen.prequantize_int8_attention(
                q, k, v, scale=kwargs.get("scale", None), attn_mask=mask,
            )
            del q, k, v
            out = comfy_kitchen.int8_attention_from_prequantized(quantized)
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(batch, -1, heads * dim_head)
            return out

        devices = self._devices(q.device)
        use_two = self.mode == "force_2gpu" or q.shape[2] >= self.min_sequence_length
        if not use_two:
            devices = devices[:1]
        if len(devices) == 1:
            quantized = comfy_kitchen.prequantize_int8_attention(
                q, k, v, scale=kwargs.get("scale", None), attn_mask=mask,
            )
            del q, k, v
            out = comfy_kitchen.int8_attention_from_prequantized(quantized)
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(batch, -1, heads * dim_head)
            return out

        if mask is not None:
            raise RuntimeError("H3VM Dev5 attention parallel does not support attention masks")
        if q.shape[1] != heads or k.shape[1] != heads:
            raise RuntimeError(f"H3VM attention head mismatch q={q.shape[1]} k={k.shape[1]} expected={heads}")

        ranges, counts = self._ranges(heads, devices)
        root_device = devices[0]
        if root_device not in self._logged_roots:
            names = [torch.cuda.get_device_name(d) for d in devices]
            LOG.info(
                "H3VM Dev5 hybrid attention ACTIVE | root=%s helper=%s | heads=%s | balance=%s | devices=%s",
                devices[0], devices[1], counts, self.head_balance, names,
            )
            self._logged_roots.add(root_device)

        caller_stream = torch.cuda.current_stream(root_device)
        root_stream = self.streams[root_device]
        input_ready = torch.cuda.Event()
        input_ready.record(caller_stream)
        root_stream.wait_event(input_ready)
        with torch.cuda.device(root_device), torch.cuda.stream(root_stream):
            quantized = comfy_kitchen.prequantize_int8_attention(
                q, k, v, scale=kwargs.get("scale", None), attn_mask=None,
            )
            q.record_stream(root_stream)
            k.record_stream(root_stream)
            v.record_stream(root_stream)
            output = torch.empty(
                batch, heads, quantized.q.shape[2], quantized.original_head_dim,
                dtype=quantized.input_dtype, device=root_device,
            )
            packed_ready = torch.cuda.Event()
            packed_ready.record(root_stream)
        del q, k, v

        def packed_head_slice(quantized, start, end, device):
            batch_, heads_, _, head_dim_ = quantized.q.shape
            padded_length = quantized.v.shape[-1]
            vv = quantized.v.view(batch_, heads_, head_dim_, padded_length)[:, start:end]
            v_scale = quantized.v_scale.view(batch_, heads_, head_dim_)[:, start:end]
            return comfy_kitchen.PrequantizedInt8Attention(
                q=quantized.q[:, start:end].to(device, non_blocking=True).contiguous(),
                k=quantized.k[:, start:end].to(device, non_blocking=True).contiguous(),
                v=vv.reshape(-1, padded_length).to(device, non_blocking=True),
                q_scale=quantized.q_scale[:, start:end].to(device, non_blocking=True).contiguous(),
                k_scale=quantized.k_scale[:, start:end].to(device, non_blocking=True).contiguous(),
                v_scale=v_scale.reshape(-1).to(device, non_blocking=True),
                original_head_dim=quantized.original_head_dim,
                input_dtype=quantized.input_dtype,
                attention_scale=quantized.attention_scale,
                cta_k=quantized.cta_k,
                attn_mask=None,
            )

        worker_shards = {}
        worker_outputs = {}
        worker_done = {}
        try:
            for device, (start, end) in zip(devices[1:], ranges[1:]):
                with torch.cuda.device(device):
                    stream = self.streams[device]
                    stream.wait_event(packed_ready)
                    with torch.cuda.stream(stream):
                        worker_shards[device] = packed_head_slice(quantized, start, end, device)

            for device, (start, end) in zip(devices[1:], ranges[1:]):
                with torch.cuda.device(device), torch.cuda.stream(self.streams[device]):
                    worker_outputs[device] = comfy_kitchen.int8_attention_from_prequantized(worker_shards[device])

            start, end = ranges[0]
            with torch.cuda.device(root_device), torch.cuda.stream(root_stream):
                local = packed_head_slice(quantized, start, end, root_device)
                local_output = comfy_kitchen.int8_attention_from_prequantized(local)
                output[:, start:end].copy_(local_output)

            for device, (start, end) in zip(devices[1:], ranges[1:]):
                with torch.cuda.device(device), torch.cuda.stream(self.streams[device]):
                    output[:, start:end].copy_(worker_outputs[device], non_blocking=True)
                    done = torch.cuda.Event()
                    done.record(self.streams[device])
                    worker_done[device] = done

            with torch.cuda.device(root_device), torch.cuda.stream(root_stream):
                for done in worker_done.values():
                    root_stream.wait_event(done)
                if not skip_output_reshape:
                    output = output.transpose(1, 2).reshape(batch, -1, heads * dim_head)
                finished = torch.cuda.Event()
                finished.record(root_stream)
            caller_stream.wait_event(finished)
            output.record_stream(caller_stream)
            return output
        except Exception:
            for device in devices:
                try:
                    torch.cuda.synchronize(device)
                except RuntimeError:
                    pass
            raise


class H3VMRelayAttentionParallel(H3VMTwoGPUAttentionParallel):
    """Dev6 transport-aware exact head parallelism.

    When CUDA peer access is unavailable, this class can still use the ordinary
    cross-device copy path that PyTorch already uses for H3VM's span handoff.
    The relay path is deliberately conservative:
    - no cross-device CUDA event dependency;
    - staging copies are complete before dual-device attention launches;
    - root/helper attention kernels then overlap;
    - helper output is copied back only after both local computations finish.

    This is an experimental speed path, but loader-level gating keeps it disabled
    automatically when the measured fallback copy bandwidth is below threshold.
    """

    __slots__ = (
        "link_info", "relay_min_gbps", "transport", "helper_head_cap",
        "helper_safety_mb", "_relay_calls", "host_engine", "host_ring_mb", "primary_first",
    )

    def __init__(self, primary_device, secondary_device, *, mode="relay_auto",
                 head_balance="sm_weighted", min_sequence_length=16384,
                 link_info=None, relay_min_gbps=4.0, helper_head_cap=24,
                 helper_safety_mb=512, host_ring_mb=64, primary_first=False):
        self.link_info = link_info or {}
        self.relay_min_gbps = float(relay_min_gbps)
        self.transport = None
        self.helper_head_cap = int(helper_head_cap)
        self.helper_safety_mb = int(helper_safety_mb)
        self._relay_calls = 0
        self.host_engine = None
        self.host_ring_mb = max(16, int(host_ring_mb))
        self.primary_first = bool(primary_first)
        super().__init__(
            primary_device, secondary_device,
            mode=mode,
            head_balance=head_balance,
            min_sequence_length=min_sequence_length,
        )

    def _preflight(self):
        import torch
        if self.mode == "off":
            self.disable_reason = "attention_mode=off"
            return

        try:
            import comfy_kitchen
        except Exception as e:
            self.disable_reason = f"comfy_kitchen import failed: {e}"
            if self.mode in ("force_2gpu", "force_relay", "force_host"):
                raise RuntimeError(self.disable_reason) from e
            return

        a, b = self.pair
        failures = []
        for d in (a, b):
            try:
                if not comfy_kitchen.int8_attention_is_available(d):
                    failures.append(f"CK INT8 attention unavailable on {d}")
            except Exception as e:
                failures.append(f"CK INT8 check failed on {d}: {e}")
        if failures:
            self.disable_reason = "; ".join(failures)
            if self.mode in ("force_2gpu", "force_relay", "force_host"):
                raise RuntimeError("H3VM Dev6 attention preflight failed: " + self.disable_reason)
            LOG.warning("H3VM Dev6 attention auto-disabled | %s", self.disable_reason)
            return

        p2p_ab = bool(self.link_info.get("peer_ab", False))
        p2p_ba = bool(self.link_info.get("peer_ba", False))
        bw_ab = self.link_info.get("copy_gbps_ab")
        bw_ba = self.link_info.get("copy_gbps_ba")
        relay_ok = (
            bw_ab is not None and bw_ba is not None
            and float(bw_ab) >= self.relay_min_gbps
            and float(bw_ba) >= self.relay_min_gbps
        )

        if self.mode == "p2p_only":
            if not (p2p_ab and p2p_ba):
                self.disable_reason = "p2p_only requested but bidirectional P2P unavailable"
                LOG.warning("H3VM Dev6 attention disabled | %s", self.disable_reason)
                return
            self.transport = "p2p"
        elif self.mode in ("host_auto", "force_host"):
            # Dev12.2: bypass the Windows/WDDM fallback D2D route entirely.
            # H3VM owns a tiny bounded host DMA runway even while ComfyUI global
            # pinned memory remains disabled. This is a transfer cache, never a
            # model-weight warehouse.
            if p2p_ab and p2p_ba and self.mode == "host_auto":
                self.transport = "p2p"
            else:
                from .global_memory import TransportEngine
                engine = None
                pinned_error = None
                try:
                    engine = TransportEngine(
                        a, b, mode="neutral_pinned", benchmark_mb=32, benchmark_repeats=3,
                        allow_explicit_pinned=True, pinned_ring_mb=self.host_ring_mb,
                        pinned_ring_slots=2, host_only=True,
                    )
                    bench = engine.benchmark
                    pp = [bench.get("neutral_pinned", {}).get(x) for x in ("ab", "ba")]
                    pg = [bench.get("neutral_pageable", {}).get(x) for x in ("ab", "ba")]
                    pp_min = min(float(x) for x in pp if x is not None) if all(x is not None for x in pp) else 0.0
                    pg_min = min(float(x) for x in pg if x is not None) if all(x is not None for x in pg) else 0.0
                    if pg_min > pp_min * 1.05:
                        engine.selected_mode = "neutral_pageable"
                        engine.ring = None
                    self.host_engine = engine
                    self.transport = "host_pinned" if engine.selected_mode == "neutral_pinned" else "host_pageable"
                except Exception as e:
                    pinned_error = repr(e)
                    try:
                        engine = TransportEngine(
                            a, b, mode="neutral_pageable", benchmark_mb=32, benchmark_repeats=3,
                            allow_explicit_pinned=False, pinned_ring_mb=0,
                            pinned_ring_slots=1, host_only=True,
                        )
                        self.host_engine = engine
                        self.transport = "host_pageable"
                    except Exception as e2:
                        self.disable_reason = f"host relay setup failed pinned={pinned_error} pageable={e2!r}"
                        if self.mode == "force_host":
                            raise RuntimeError(self.disable_reason) from e2
                        LOG.warning("H3VM Dev12.2 host attention disabled | %s", self.disable_reason)
                        return
        elif self.mode == "force_relay":
            if not relay_ok:
                raise RuntimeError(
                    "H3VM Dev6 forced relay requested but fallback D2D copy benchmark "
                    f"is below {self.relay_min_gbps:.1f} GB/s or failed: "
                    f"ab={bw_ab} ba={bw_ba}"
                )
            self.transport = "relay"
        elif p2p_ab and p2p_ba:
            self.transport = "p2p"
        elif relay_ok:
            self.transport = "relay"
        else:
            self.disable_reason = (
                "no bidirectional P2P and fallback D2D copy path is below relay threshold "
                f"({self.relay_min_gbps:.1f} GB/s): ab={bw_ab} ba={bw_ba}"
            )
            LOG.warning("H3VM Dev6 attention auto-disabled | %s", self.disable_reason)
            return

        for d in (a, b):
            self.streams[d] = torch.cuda.Stream(device=d)
        self.enabled = True
        if self.host_engine is not None:
            hb = self.host_engine.benchmark
            LOG.info(
                "H3VM Dev12.2 host attention transport | transport=%s ring=%dMiB | "
                "pageable=%.2f/%.2fGB/s pinned=%.2f/%.2fGB/s",
                self.transport, self.host_ring_mb,
                float(hb.get("neutral_pageable", {}).get("ab") or 0.0),
                float(hb.get("neutral_pageable", {}).get("ba") or 0.0),
                float(hb.get("neutral_pinned", {}).get("ab") or 0.0),
                float(hb.get("neutral_pinned", {}).get("ba") or 0.0),
            )
        else:
            LOG.info(
                "H3VM Dev6 attention transport selected | transport=%s | p2p=%s/%s | "
                "copy=%.2f/%.2f GB/s | threshold=%.2f",
                self.transport, p2p_ab, p2p_ba,
                float(bw_ab or 0.0), float(bw_ba or 0.0), self.relay_min_gbps,
            )

    def _relay_counts(self, heads, devices, seq_len, dim_head):
        """Heterogeneous split with an explicit cap on the physical small GPU.

        ``devices`` is ordered [root, helper] and flips every striped ownership
        handoff.  The older relay code capped ``devices[1]`` unconditionally,
        which accidentally starved the 16G card whenever the 8G card happened
        to be the root.  Dev12.1 instead caps the configured *secondary* device
        regardless of root/helper role, while preserving an SM-weighted target.
        """
        import torch
        if self.head_balance == "sm_weighted":
            weights = [max(1, torch.cuda.get_device_properties(d).multi_processor_count) for d in devices]
            desired = weighted_counts(heads, weights)
        else:
            desired = balanced_counts(heads, len(devices))

        counts = [int(desired[0]), int(desired[1])]
        small_device = self.pair[1]
        small_idx = 0 if devices[0] == small_device else 1
        other_idx = 1 - small_idx

        # The 8G card gets a deterministic head ceiling.  Any clipped heads are
        # reassigned to the 16G card, so root orientation cannot invert the load
        # balance.  helper_head_cap is kept as the public node parameter for
        # backward compatibility, but semantically it is the small-GPU cap here.
        small_cap = max(1, min(int(self.helper_head_cap), heads - 1))
        if counts[small_idx] > small_cap:
            spill = counts[small_idx] - small_cap
            counts[small_idx] -= spill
            counts[other_idx] += spill

        # Only a helper receives an extra packed Q/K/V shard.  If the helper is
        # the 8G GPU, cap that shard again using *physical free* memory.
        helper = devices[1]
        if helper == small_device:
            per_head = max(1, int(seq_len) * int(dim_head) * 10)
            try:
                free, _ = torch.cuda.mem_get_info(helper)
                budget = max(0, int(free) - self.helper_safety_mb * 1024 * 1024)
                max_by_mem = max(1, budget // per_head)
            except Exception:
                max_by_mem = heads - 1
            helper_count = min(counts[1], max(1, int(max_by_mem)), heads - 1)
            if helper_count < counts[1]:
                spill = counts[1] - helper_count
                counts[1] = helper_count
                counts[0] += spill

        # Keep both CUDA engines participating.
        if counts[0] < 1:
            counts[0], counts[1] = 1, heads - 1
        if counts[1] < 1:
            counts[1], counts[0] = 1, heads - 1
        return counts

    @staticmethod
    def _packed_head_slice(quantized, start, end, device, *, non_blocking):
        import comfy_kitchen
        batch, heads, _, head_dim = quantized.q.shape
        padded_length = quantized.v.shape[-1]
        v = quantized.v.view(batch, heads, head_dim, padded_length)[:, start:end]
        v_scale = quantized.v_scale.view(batch, heads, head_dim)[:, start:end]
        return comfy_kitchen.PrequantizedInt8Attention(
            q=quantized.q[:, start:end].to(device, non_blocking=non_blocking).contiguous(),
            k=quantized.k[:, start:end].to(device, non_blocking=non_blocking).contiguous(),
            v=v.reshape(-1, padded_length).to(device, non_blocking=non_blocking),
            q_scale=quantized.q_scale[:, start:end].to(device, non_blocking=non_blocking).contiguous(),
            k_scale=quantized.k_scale[:, start:end].to(device, non_blocking=non_blocking).contiguous(),
            v_scale=v_scale.reshape(-1).to(device, non_blocking=non_blocking),
            original_head_dim=quantized.original_head_dim,
            input_dtype=quantized.input_dtype,
            attention_scale=quantized.attention_scale,
            cta_k=quantized.cta_k,
            attn_mask=None,
        )

    def _packed_head_slice_host(self, quantized, start, end, device):
        """Move one quantized head range through H3VM-owned host RAM."""
        import comfy_kitchen
        if self.host_engine is None:
            raise RuntimeError("H3VM Dev12.2 host attention has no transport engine")
        batch, heads, _, head_dim = quantized.q.shape
        padded_length = quantized.v.shape[-1]
        v = quantized.v.view(batch, heads, head_dim, padded_length)[:, start:end].contiguous()
        v_scale = quantized.v_scale.view(batch, heads, head_dim)[:, start:end].contiguous()
        move = self.host_engine.move_tensor
        return comfy_kitchen.PrequantizedInt8Attention(
            q=move(quantized.q[:, start:end].contiguous(), device),
            k=move(quantized.k[:, start:end].contiguous(), device),
            v=move(v.reshape(-1, padded_length), device),
            q_scale=move(quantized.q_scale[:, start:end].contiguous(), device),
            k_scale=move(quantized.k_scale[:, start:end].contiguous(), device),
            v_scale=move(v_scale.reshape(-1), device),
            original_head_dim=quantized.original_head_dim,
            input_dtype=quantized.input_dtype,
            attention_scale=quantized.attention_scale,
            cta_k=quantized.cta_k,
            attn_mask=None,
        )

    def _container_function_primary_first(self, q, k, v, heads, mask=None, attn_precision=None,
                                          skip_reshape=False, skip_output_reshape=False, **kwargs):
        """Dev14 critical-path relay: root attention launches before helper staging."""
        import time
        import torch
        import comfy_kitchen
        from comfy.ldm.modules.attention import _comfy_kitchen_int8_inputs

        q = q.take(); k = k.take(); v = v.take()
        q, k, v, mask, batch, dim_head = _comfy_kitchen_int8_inputs(
            q, k, v, heads, mask, skip_reshape, kwargs.get("enable_gqa", False)
        )
        use_two = self.mode in ("force_relay", "force_2gpu", "force_host") or q.shape[2] >= self.min_sequence_length
        if not self.enabled or not use_two:
            quantized = comfy_kitchen.prequantize_int8_attention(
                q, k, v, scale=kwargs.get("scale", None), attn_mask=mask,
            )
            del q, k, v
            out = comfy_kitchen.int8_attention_from_prequantized(quantized)
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(batch, -1, heads * dim_head)
            return out
        if mask is not None:
            raise RuntimeError("H3VM Dev14 primary-first relay attention does not support attention masks")
        if q.shape[1] != heads or k.shape[1] != heads:
            raise RuntimeError(f"H3VM relay attention head mismatch q={q.shape[1]} k={k.shape[1]} expected={heads}")

        devices = self._devices(q.device)
        root_device, helper_device = devices
        counts = self._relay_counts(heads, devices, q.shape[2], dim_head)
        ranges = counts_to_ranges(counts)
        if root_device not in self._logged_roots:
            LOG.info(
                "H3VM Dev14 PRIMARY-FIRST attention ACTIVE | root=%s helper=%s heads=%s seq=%d dim=%d",
                root_device, helper_device, counts, q.shape[2], dim_head,
            )
            self._logged_roots.add(root_device)

        root_stream = self.streams[root_device]
        helper_stream = self.streams[helper_device]
        caller_stream = torch.cuda.current_stream(root_device)
        root_stream.wait_stream(caller_stream)

        # Quantization is a shared prerequisite. Once packed QKV is ready, the
        # critical root attention launches immediately before any helper copy.
        with torch.cuda.device(root_device), torch.cuda.stream(root_stream):
            quantized = comfy_kitchen.prequantize_int8_attention(
                q, k, v, scale=kwargs.get("scale", None), attn_mask=None,
            )
            q.record_stream(root_stream); k.record_stream(root_stream); v.record_stream(root_stream)
        root_stream.synchronize()
        del q, k, v

        output = torch.empty(
            batch, heads, quantized.q.shape[2], quantized.original_head_dim,
            dtype=quantized.input_dtype, device=root_device,
        )
        root_start, root_end = ranges[0]
        re0 = torch.cuda.Event(enable_timing=True); re1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(root_device), torch.cuda.stream(root_stream):
            re0.record(root_stream)
            local = self._packed_head_slice(quantized, root_start, root_end, root_device, non_blocking=False)
            local_output = comfy_kitchen.int8_attention_from_prequantized(local)
            output[:, root_start:root_end].copy_(local_output)
            re1.record(root_stream)

        helper_start, helper_end = ranges[1]
        t_stage = time.perf_counter()
        if self.transport in ("host_pageable", "host_pinned"):
            worker_shard = self._packed_head_slice_host(quantized, helper_start, helper_end, helper_device)
        else:
            worker_shard = self._packed_head_slice(quantized, helper_start, helper_end, helper_device, non_blocking=False)
        torch.cuda.synchronize(helper_device)
        stage_ms = (time.perf_counter() - t_stage) * 1000.0

        he0 = torch.cuda.Event(enable_timing=True); he1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(helper_device), torch.cuda.stream(helper_stream):
            he0.record(helper_stream)
            worker_output = comfy_kitchen.int8_attention_from_prequantized(worker_shard)
            he1.record(helper_stream)
        helper_stream.synchronize()
        helper_ms = float(he0.elapsed_time(he1))

        # Dev15 REDLINE: helper return is an exact dependency, but it is not a
        # reason to device-synchronize GPU0. Enqueue the merge on root_stream and
        # make only the caller stream wait for that event. Root prefetch / other
        # independent streams remain free to run.
        t_return = time.perf_counter()
        merge_done = torch.cuda.Event()
        if self.transport in ("host_pageable", "host_pinned"):
            returned = self.host_engine.move_tensor(worker_output, root_device)
            with torch.cuda.device(root_device), torch.cuda.stream(root_stream):
                output[:, helper_start:helper_end].copy_(returned, non_blocking=True)
                merge_done.record(root_stream)
                returned.record_stream(root_stream)
        else:
            with torch.cuda.device(root_device), torch.cuda.stream(root_stream):
                output[:, helper_start:helper_end].copy_(worker_output, non_blocking=True)
                merge_done.record(root_stream)
                worker_output.record_stream(root_stream)
        return_ms = (time.perf_counter() - t_return) * 1000.0
        try:
            root_ms = float(re0.elapsed_time(re1))
        except Exception:
            # The root timing event may still be in flight; telemetry must never
            # put a synchronization fence back onto the critical line.
            root_ms = 0.0

        if not skip_output_reshape:
            output = output.transpose(1, 2).reshape(batch, -1, heads * dim_head)
        caller_stream.wait_event(merge_done)
        output.record_stream(caller_stream)

        self._relay_calls += 1
        if self._relay_calls <= 3 or self._relay_calls % 50 == 0:
            LOG.info(
                "H3VM Dev14 CPM attn #%d | root-first transport=%s heads=%s | root=%.2fms stage=%.2fms helper=%.2fms return=%.2fms",
                self._relay_calls, self.transport, counts, root_ms, stage_ms, helper_ms, return_ms,
            )
        return output

    def container_function(self, q, k, v, heads, mask=None, attn_precision=None,
                           skip_reshape=False, skip_output_reshape=False, **kwargs):
        if self.primary_first and self.transport in ("relay", "host_pageable", "host_pinned"):
            return self._container_function_primary_first(
                q, k, v, heads, mask=mask, attn_precision=attn_precision,
                skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs
            )
        if self.transport not in ("relay", "host_pageable", "host_pinned"):
            return super().container_function(
                q, k, v, heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        import time
        import torch
        import comfy_kitchen
        from comfy.ldm.modules.attention import _comfy_kitchen_int8_inputs

        q = q.take()
        k = k.take()
        v = v.take()
        q, k, v, mask, batch, dim_head = _comfy_kitchen_int8_inputs(
            q, k, v, heads, mask, skip_reshape, kwargs.get("enable_gqa", False)
        )

        use_two = self.mode in ("force_relay", "force_2gpu", "force_host") or q.shape[2] >= self.min_sequence_length
        if not self.enabled or not use_two:
            quantized = comfy_kitchen.prequantize_int8_attention(
                q, k, v, scale=kwargs.get("scale", None), attn_mask=mask,
            )
            del q, k, v
            out = comfy_kitchen.int8_attention_from_prequantized(quantized)
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(batch, -1, heads * dim_head)
            return out

        if mask is not None:
            raise RuntimeError("H3VM Dev6 relay attention does not support attention masks")
        if q.shape[1] != heads or k.shape[1] != heads:
            raise RuntimeError(f"H3VM relay attention head mismatch q={q.shape[1]} k={k.shape[1]} expected={heads}")

        devices = self._devices(q.device)
        root_device, helper_device = devices
        counts = self._relay_counts(heads, devices, q.shape[2], dim_head)
        ranges = counts_to_ranges(counts)

        if root_device not in self._logged_roots:
            LOG.info(
                "H3VM Dev12.1 RELAY attention ACTIVE | root=%s helper=%s | heads=%s | "
                "seq=%d dim=%d | helper_cap=%d | safety=%dMB",
                root_device, helper_device, counts, q.shape[2], dim_head,
                self.helper_head_cap, self.helper_safety_mb,
            )
            self._logged_roots.add(root_device)

        # 1. Quantize on root. Finish before fallback cross-device staging so no
        # cross-device event dependency is required.
        root_stream = self.streams[root_device]
        helper_stream = self.streams[helper_device]
        caller_stream = torch.cuda.current_stream(root_device)
        root_stream.wait_stream(caller_stream)

        with torch.cuda.device(root_device), torch.cuda.stream(root_stream):
            quantized = comfy_kitchen.prequantize_int8_attention(
                q, k, v, scale=kwargs.get("scale", None), attn_mask=None,
            )
            q.record_stream(root_stream)
            k.record_stream(root_stream)
            v.record_stream(root_stream)
        root_stream.synchronize()
        del q, k, v

        # 2. Stage only helper heads over the ordinary D2D fallback path.
        helper_start, helper_end = ranges[1]
        t_stage = time.perf_counter()
        if self.transport in ("host_pageable", "host_pinned"):
            worker_shard = self._packed_head_slice_host(
                quantized, helper_start, helper_end, helper_device
            )
        else:
            worker_shard = self._packed_head_slice(
                quantized, helper_start, helper_end, helper_device, non_blocking=False
            )
        torch.cuda.synchronize(helper_device)
        stage_ms = (time.perf_counter() - t_stage) * 1000.0

        output = torch.empty(
            batch, heads, quantized.q.shape[2], quantized.original_head_dim,
            dtype=quantized.input_dtype, device=root_device,
        )

        # 3. Launch helper and root attention. Launches are asynchronous and can
        # overlap even though the staging copy itself was conservative/synchronous.
        he0 = torch.cuda.Event(enable_timing=True)
        he1 = torch.cuda.Event(enable_timing=True)
        re0 = torch.cuda.Event(enable_timing=True)
        re1 = torch.cuda.Event(enable_timing=True)

        with torch.cuda.device(helper_device), torch.cuda.stream(helper_stream):
            he0.record(helper_stream)
            worker_output = comfy_kitchen.int8_attention_from_prequantized(worker_shard)
            he1.record(helper_stream)

        root_start, root_end = ranges[0]
        with torch.cuda.device(root_device), torch.cuda.stream(root_stream):
            re0.record(root_stream)
            local = self._packed_head_slice(
                quantized, root_start, root_end, root_device, non_blocking=False
            )
            local_output = comfy_kitchen.int8_attention_from_prequantized(local)
            output[:, root_start:root_end].copy_(local_output)
            re1.record(root_stream)

        helper_stream.synchronize()
        root_stream.synchronize()
        helper_ms = he0.elapsed_time(he1)
        root_ms = re0.elapsed_time(re1)

        # 4. Return helper result through the same measured fallback path.
        t_return = time.perf_counter()
        if self.transport in ("host_pageable", "host_pinned"):
            returned = self.host_engine.move_tensor(worker_output, root_device)
            with torch.cuda.device(root_device):
                output[:, helper_start:helper_end].copy_(returned, non_blocking=False)
            del returned
        else:
            with torch.cuda.device(root_device):
                output[:, helper_start:helper_end].copy_(worker_output, non_blocking=False)
        torch.cuda.synchronize(root_device)
        return_ms = (time.perf_counter() - t_return) * 1000.0

        if not skip_output_reshape:
            output = output.transpose(1, 2).reshape(batch, -1, heads * dim_head)

        caller_stream.wait_stream(root_stream)
        output.record_stream(caller_stream)

        self._relay_calls += 1
        if self._relay_calls <= 4 or self._relay_calls % 50 == 0:
            LOG.info(
                "H3VM Dev12.2 relay attn #%d | transport=%s root=%s helper=%s heads=%s | "
                "stage=%.2fms compute(root/helper)=%.2f/%.2fms return=%.2fms | "
                "parallel_compute_window=%.2fms",
                self._relay_calls, self.transport, root_device, helper_device, counts,
                stage_ms, root_ms, helper_ms, return_ms, max(root_ms, helper_ms),
            )

        return output
