from __future__ import annotations

import logging
import time

LOG = logging.getLogger("H3VM")
GIB = 1024 ** 3
MIB = 1024 ** 2


def _tensor_nbytes(value):
    try:
        import torch
        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return sum(_tensor_nbytes(x) for x in value)
    if isinstance(value, dict):
        return sum(_tensor_nbytes(x) for x in value.values())
    return 0


def _mem_snapshot(device):
    import torch
    free, total = torch.cuda.mem_get_info(device)
    return {
        "free": int(free),
        "total": int(total),
        "used": int(total - free),
        "alloc": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
    }


def _gib(v):
    return float(v) / GIB


def striped_owner_map(total_blocks: int, stripe_size: int):
    """Return block_index -> owner id (0 primary, 1 secondary)."""
    stripe_size = max(1, int(stripe_size))
    return {i: ((i // stripe_size) & 1) for i in range(int(total_blocks))}


class StreamingExactRuntime:
    """Dev12.1 RAM-first striped exact block executor.

    The mathematical block order is unchanged. Physical block ownership is striped
    across two DynamicVRAM patchers, while CPU RAM remains the backing store.

    Design goals:
      * keep the main/fixed H3 modules on the primary GPU;
      * initialize each block island as a low-residency DynamicVRAM model;
      * leave an explicit runtime/workspace reserve on each GPU;
      * use one-ahead VBAR prefetch only, never whole-island warmup;
      * trim old island residency at stripe boundaries;
      * move the exact activation through ordinary system RAM at device changes.

    This is a memory-system experiment first. It deliberately accepts extra PCIe
    traffic in exchange for proving that H3 block weights need not remain resident
    in either GPU's VRAM.
    """

    def __init__(self, primary_device, secondary_device, blocks, owner_map, *, space,
                 expected_steps=4, stripe_size=2,
                 primary_runtime_reserve_gb=5.0,
                 secondary_runtime_reserve_gb=4.0,
                 primary_hot_cache_gb=3.5,
                 secondary_hot_cache_gb=2.0,
                 one_ahead_prefetch=True,
                 trim_on_stripe_boundary=True,
                 telemetry=True,
                 hard_cleanup_after_sample=False,
                 single_root=False,
                 single_root_trim_interval=2,
                 critical_path_pipeline_window=1,
                 critical_path_rolling_retire=False):
        self.primary_device = primary_device
        self.secondary_device = secondary_device
        self.devices = {0: primary_device, 1: secondary_device}
        self.blocks = dict(blocks)
        self.owner_map = dict(owner_map)
        self.space = space
        self.expected_steps = max(1, int(expected_steps))
        self.stripe_size = max(1, int(stripe_size))
        self.runtime_reserve = {
            0: max(0.5, float(primary_runtime_reserve_gb)),
            1: max(0.5, float(secondary_runtime_reserve_gb)),
        }
        self.hot_cache = {
            0: max(0.0, float(primary_hot_cache_gb)),
            1: max(0.0, float(secondary_hot_cache_gb)),
        }
        self.one_ahead_prefetch = bool(one_ahead_prefetch)
        self.trim_on_stripe_boundary = bool(trim_on_stripe_boundary)
        self.telemetry = bool(telemetry)
        self.hard_cleanup_after_sample = bool(hard_cleanup_after_sample)
        self.single_root = bool(single_root)
        self.single_root_trim_interval = max(1, int(single_root_trim_interval))
        # Dev15: keep the CPU ahead of the GPU0 critical line. In critical-path
        # mode we retire CUDA work in small windows instead of inserting a
        # device-wide synchronize after every H3 block.
        self.critical_path_pipeline_window = max(1, int(critical_path_pipeline_window))
        self.critical_path_rolling_retire = bool(critical_path_rolling_retire)

        self.patchers = {0: None, 1: None}
        self.helper_patchers = {0: None, 1: None}
        self.mlp_fabric = None
        self.owner_blocks = {
            0: [i for i in sorted(self.blocks) if self.owner_map[i] == 0],
            1: [i for i in sorted(self.blocks) if self.owner_map[i] == 1],
        }
        self.prefetch_queues = {0: None, 1: None}
        self.prefetch_primed_after_stripe = {0: False, 1: False}

        self._step = 0
        self._step_started = None
        self._step_compute_ms = {0: 0.0, 1: 0.0}
        self._step_transfer_ms = 0.0
        self._step_trim_ms = {0: 0.0, 1: 0.0}
        self._step_trim_mib = {0: 0.0, 1: 0.0}
        self._step_boundary_count = 0
        self._peak = {0: 0.0, 1: 0.0}
        self._pipeline_pending = []
        self._step_prefetch_host_ms = 0.0
        self._step_barrier_ms = 0.0
        # Dev15.1 splits retirement wait into critical-stream event wait versus
        # rare device-wide maintenance barriers. Only the latter can stall
        # unrelated root prefetch/DMA work.
        self._step_event_wait_ms = 0.0
        self._step_device_barrier_ms = 0.0
        self._metadata_cache = {0: None, 1: None}
        self._sampling_generation = 0
        self._prepared = False
        self._warned_trim = False
        self._sidecar_prepare_thread = None
        self._sidecar_prepare_error = None
        self._pipeline_pending = []
        self._step_prefetch_host_ms = 0.0
        self._step_barrier_ms = 0.0
        self._step_event_wait_ms = 0.0
        self._step_device_barrier_ms = 0.0

    def bind_patchers(self, primary_patcher, secondary_patcher, primary_helper_patcher=None, secondary_helper_patcher=None):
        self.patchers[0] = primary_patcher
        self.patchers[1] = secondary_patcher
        self.helper_patchers[0] = primary_helper_patcher
        self.helper_patchers[1] = secondary_helper_patcher

    def bind_mlp_fabric(self, fabric):
        self.mlp_fabric = fabric

    def _sample_memory(self):
        for owner in (0, 1):
            try:
                snap = _mem_snapshot(self.devices[owner])
                self._peak[owner] = max(self._peak[owner], _gib(snap["used"]))
            except Exception:
                pass

    @staticmethod
    def _loaded_size(patcher):
        if patcher is None:
            return 0
        fn = getattr(patcher, "loaded_size", None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:
                return 0
        return 0

    def _trim_one(self, patcher, target_bytes: int, force_all: bool = False):
        if patcher is None:
            return 0, 0.0
        fn = getattr(patcher, "partially_unload", None)
        if not callable(fn):
            if not self._warned_trim:
                LOG.warning("H3VM Dev12 ModelPatcher lacks partially_unload(); hot-cache trimming disabled")
                self._warned_trim = True
            return 0, 0.0
        before = self._loaded_size(patcher)
        request = max(0, before - int(target_bytes))
        if force_all:
            request = max(request, int(1e15))
        if request <= 0:
            return 0, 0.0
        t0 = time.perf_counter()
        try:
            freed = int(fn(patcher.offload_device, request) or 0)
        except Exception as exc:
            LOG.warning("H3VM Dev12 trim warning device=%s: %r", patcher.load_device, exc)
            return 0, 0.0
        return freed, (time.perf_counter() - t0) * 1000.0

    def _trim_owner(self, owner: int, force_all: bool = False, account: bool = True):
        """Return cold block/helper DynamicVRAM weights to CPU.

        Dev12.3 keeps helper mirrors cold by default.  The ordinary block island
        alone receives the configured hot-cache allowance; helper MLP weights are
        streamed for the current block and can be discarded immediately.
        """
        total_freed = 0
        total_ms = 0.0
        # Dev13.1 FEED: in single-root mode the secondary has no whole-block
        # island at all, so its configured hot-cache budget belongs to the MLP
        # helper patcher.  Dev13.0 trimmed helper mirrors to zero every few
        # blocks, which kept the 8G coprocessor almost empty and forced repeated
        # RAM/VBAR faults.  Retain a bounded helper working set instead.
        helper_patcher = self.helper_patchers.get(owner)
        block_patcher = self.patchers.get(owner)
        if force_all:
            helper_target = 0
        elif self.single_root and helper_patcher is not None and block_patcher is None:
            helper_target = int(self.hot_cache[owner] * GIB)
        else:
            helper_target = 0
        freed, dt = self._trim_one(helper_patcher, helper_target, force_all=force_all)
        total_freed += freed; total_ms += dt
        target = 0 if force_all else int(self.hot_cache[owner] * GIB)
        freed, dt = self._trim_one(block_patcher, target, force_all=force_all)
        total_freed += freed; total_ms += dt
        if account:
            self._step_trim_ms[owner] += total_ms
            self._step_trim_mib[owner] += total_freed / MIB
        return total_freed

    def _prepare_island(self, owner: int, account_trim: bool = True):
        import comfy.model_management as mm
        patcher = self.patchers.get(owner)
        helper = self.helper_patchers.get(owner)
        models = [m for m in (patcher, helper) if m is not None]

        # Dev13H: a pure coprocessor device intentionally has no whole-block
        # patcher.  Do not manufacture/load a zero-parameter H3IslandRoot: AIMDO
        # still asks ModelPatcher for a VBAR and a zero-sized VBAR reservation can
        # fail on Windows/WDDM.  If Token-MLP is enabled, owner=1 contains only
        # the helper patcher and is prepared normally.
        if not models:
            LOG.info("H3VM Dev13H skip empty island owner=%d device=%s", owner, self.devices[owner])
            return

        # Repeated-run determinism: start each sample from a cold block/helper cache.
        self._trim_owner(owner, force_all=True, account=account_trim)
        reserve_bytes = int(self.runtime_reserve[owner] * GIB)
        mm.load_models_gpu(
            models,
            memory_required=0,
            minimum_memory_required=reserve_bytes,
            force_full_load=False,
        )
        # load_models_gpu may choose to opportunistically resident some dynamic
        # weights. Clamp that residency immediately to our explicit cache budget.
        self._trim_owner(owner, force_all=False, account=account_trim)

    def _prepare_sidecar_background(self):
        """Prepare GPU1 behind the GPU0 critical path.

        Dev14.1 keeps the DynamicVRAM helper package. Dev14.2 upgrades only its
        bounded sidecar packet to force-full residency because the package was
        deliberately sized to fit inside the 8GiB helper card. GPU0 never waits:
        until READY, selected MLP blocks execute root-only.
        """
        t0 = time.perf_counter()
        helper = self.helper_patchers.get(1)
        resident_packet = bool(
            self.mlp_fabric is not None
            and getattr(self.mlp_fabric, "resident_sidecar_packet", False)
        )
        try:
            if resident_packet:
                if helper is None:
                    raise RuntimeError("Dev14.2 resident sidecar requested without helper patcher")
                import comfy.model_management as mm
                self._trim_owner(1, force_all=True, account=False)
                reserve_bytes = int(self.runtime_reserve[1] * GIB)
                load_t0 = time.perf_counter()
                mm.load_models_gpu(
                    [helper],
                    memory_required=0,
                    minimum_memory_required=reserve_bytes,
                    force_full_load=True,
                )
                load_ms = (time.perf_counter() - load_t0) * 1000.0
                resident = self._loaded_size(helper)
                model_size_fn = getattr(helper, "model_size", None)
                model_size = int(model_size_fn()) if callable(model_size_fn) else resident
                ratio = (float(resident) / float(model_size)) if model_size > 0 else 1.0
                LOG.info(
                    "H3VM Dev14.2 SIDECAR PACKET resident | loaded=%.2f/%.2fGiB ratio=%.1f%% load=%.1fms reserve=%.2fGiB",
                    resident / GIB, model_size / GIB, ratio * 100.0, load_ms, self.runtime_reserve[1],
                )
            else:
                self._prepare_island(1, account_trim=False)

            warm = getattr(self.mlp_fabric, "warmup_sidecar", None) if self.mlp_fabric is not None else None
            if callable(warm):
                warm()
            ready = getattr(self.mlp_fabric, "set_sidecar_ready", None) if self.mlp_fabric is not None else None
            if callable(ready):
                ready(True)
            self._sidecar_prepare_error = None
            if resident_packet:
                LOG.info(
                    "H3VM Dev14.2 SIDECAR READY async | wall=%.1fms helper_resident=%.2fGiB | GPU0 never waited",
                    (time.perf_counter() - t0) * 1000.0, self._loaded_size(helper) / GIB,
                )
            else:
                LOG.info(
                    "H3VM Dev14.1 SIDECAR READY async | wall=%.1fms helper_resident=%.2fGiB",
                    (time.perf_counter() - t0) * 1000.0, self._loaded_size(helper) / GIB,
                )
        except Exception as exc:
            self._sidecar_prepare_error = repr(exc)
            ready = getattr(self.mlp_fabric, "set_sidecar_ready", None) if self.mlp_fabric is not None else None
            if callable(ready):
                ready(False)
            if resident_packet:
                LOG.warning("H3VM Dev14.2 resident sidecar prepare failed; GPU0 continues solo: %r", exc)
            else:
                LOG.warning("H3VM Dev14.1 async sidecar prepare failed; GPU0 continues solo: %r", exc)

    def _prepare_islands(self):
        # Dev14.1 startup CPM: GPU1 sidecar preparation is non-critical work.
        # Prepare GPU0 synchronously, then let GPU0 start the H3 graph while a
        # background thread loads/warms the fixed GPU1 helper package. Selected
        # MLP blocks simply run root-only until sidecar_ready becomes true.
        cp_async = bool(
            self.single_root
            and self.mlp_fabric is not None
            and getattr(self.mlp_fabric, "critical_path", False)
            and self.patchers.get(1) is None
            and self.helper_patchers.get(1) is not None
        )
        if cp_async:
            import threading
            t0 = time.perf_counter()
            self._prepare_island(0)
            primary_ms = (time.perf_counter() - t0) * 1000.0
            ready = getattr(self.mlp_fabric, "set_sidecar_ready", None)
            if callable(ready):
                ready(False)
            self._sidecar_prepare_error = None
            self._sidecar_prepare_thread = threading.Thread(
                target=self._prepare_sidecar_background,
                name="H3VM-Dev14.1-SidecarPrepare",
                daemon=True,
            )
            self._sidecar_prepare_thread.start()
            resident_packet = bool(getattr(self.mlp_fabric, "resident_sidecar_packet", False))
            LOG.info(
                "H3VM %s STARTUP CPM | primary_prepare=%.1fms | sidecar_prepare=ASYNC | GPU0 released immediately",
                "Dev14.2" if resident_packet else "Dev14.1", primary_ms,
            )
            self._prepared = True
            return

        t0 = time.perf_counter()
        self._prepare_island(0)
        p_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        self._prepare_island(1)
        s_ms = (time.perf_counter() - t1) * 1000.0
        LOG.info("H3VM startup prepare | primary=%.1fms secondary=%.1fms serial=%.1fms", p_ms, s_ms, p_ms + s_ms)
        self._prepared = True

    def _make_prefetch_queue(self, owner: int, transformer_options):
        if not self.one_ahead_prefetch:
            return None
        try:
            import comfy.model_prefetch as mp
            opts = dict(transformer_options or {})
            opts["prefetch_dynamic_vbars"] = True
            q = [self.blocks[i] for i in self.owner_blocks[owner]]
            out = mp.make_prefetch_queue(q, self.devices[owner], opts)
            LOG.debug(
                "H3VM Dev12 prefetch queue owner=%d device=%s active=%s blocks=%d",
                owner, self.devices[owner], out is not None, len(q),
            )
            return out
        except Exception as exc:
            LOG.warning("H3VM Dev12 prefetch disabled owner=%d: %r", owner, exc)
            return None

    def _prefetch_pop(self, owner: int, module):
        q = self.prefetch_queues.get(owner)
        if q is None:
            return
        t0 = time.perf_counter()
        try:
            import comfy.model_prefetch as mp
            mp.prefetch_queue_pop(q, self.devices[owner], module)
        except Exception as exc:
            LOG.warning("H3VM Dev12 prefetch pop warning owner=%d block=%s: %r", owner, getattr(module, 'h3vm_index', '?'), exc)
            self.prefetch_queues[owner] = None
        finally:
            if self._step_started is not None:
                self._step_prefetch_host_ms += (time.perf_counter() - t0) * 1000.0

    def _prime_next_stripe(self, owner: int):
        """Compatibility no-op.

        ComfyUI's prefetch queue is already a lookahead-1 pipeline: after an
        initial prime, each per-block pop waits for the current transfer and
        launches exactly the next owned block. Dev12.0 added an extra stripe-end
        pop, which over-consumed the [None] + blocks + [None] queue and produced
        the observed IndexError. Dev12.1 deliberately never advances it here.
        """
        return

    def _finish_prefetch(self, owner: int):
        q = self.prefetch_queues.get(owner)
        if q is None:
            return
        try:
            import comfy.model_prefetch as mp
            # With one initial prime + one pop per owned block, exactly one final
            # sentinel pair should remain. Only flush when queue[0] can still be
            # read after pop; otherwise let global cleanup retire it.
            if len(q) >= 2:
                mp.prefetch_queue_pop(q, self.devices[owner], None)
        except Exception as exc:
            LOG.debug("H3VM Dev12.1 prefetch finish owner=%d: %r", owner, exc)

    def _move_tree(self, value, dst):
        import torch
        if torch.is_tensor(value):
            if value.device == dst:
                return value
            host = value if value.device.type == "cpu" else self.space.transport.move_tensor(value, "cpu", mode="neutral_pageable")
            return self.space.transport.move_tensor(host, dst, mode="neutral_pageable")
        if isinstance(value, tuple):
            return tuple(self._move_tree(x, dst) for x in value)
        if isinstance(value, list):
            return [self._move_tree(x, dst) for x in value]
        if isinstance(value, dict):
            return {k: self._move_tree(x, dst) for k, x in value.items()}
        return value

    def _metadata_for_owner(self, owner, t_emb, mod_segments, rope_freqs):
        cached = self._metadata_cache.get(owner)
        if cached is not None:
            return cached
        dst = self.devices[owner]
        out = (
            self._move_tree(t_emb, dst),
            self._move_tree(mod_segments, dst),
            self._move_tree(rope_freqs, dst),
        )
        self._metadata_cache[owner] = out
        return out

    def sampling_begin(self):
        self._sampling_generation += 1
        self._step = 0
        self._metadata_cache = {0: None, 1: None}
        self.prefetch_queues = {0: None, 1: None}
        self.prefetch_primed_after_stripe = {0: False, 1: False}
        self._step_trim_ms = {0: 0.0, 1: 0.0}
        self._step_trim_mib = {0: 0.0, 1: 0.0}
        try:
            p = _mem_snapshot(self.primary_device)
            s = _mem_snapshot(self.secondary_device)
            LOG.info(
                "H3VM Dev12 PRE-FLIGHT gen=%d | %s free=%.2fGiB used=%.2fGiB | %s free=%.2fGiB used=%.2fGiB",
                self._sampling_generation,
                self.primary_device, _gib(p["free"]), _gib(p["used"]),
                self.secondary_device, _gib(s["free"]), _gib(s["used"]),
            )
        except Exception as exc:
            LOG.warning("H3VM Dev12 pre-flight telemetry unavailable: %r", exc)
        self._prepare_islands()

    def _begin_step(self, transformer_options):
        self._step += 1
        self._step_started = time.perf_counter()
        self._step_compute_ms = {0: 0.0, 1: 0.0}
        self._step_transfer_ms = 0.0
        self._step_trim_ms = {0: 0.0, 1: 0.0}
        self._step_trim_mib = {0: 0.0, 1: 0.0}
        self._step_boundary_count = 0
        self._peak = {0: 0.0, 1: 0.0}
        self._pipeline_pending = []
        self._step_prefetch_host_ms = 0.0
        self._step_barrier_ms = 0.0
        self._step_event_wait_ms = 0.0
        self._step_device_barrier_ms = 0.0
        self._metadata_cache = {0: None, 1: None}
        if self.mlp_fabric is not None:
            begin_step = getattr(self.mlp_fabric, "begin_step", None)
            if callable(begin_step):
                begin_step(self._step)
        self.prefetch_queues[0] = self._make_prefetch_queue(0, transformer_options)
        self.prefetch_queues[1] = self._make_prefetch_queue(1, transformer_options)
        self.prefetch_primed_after_stripe = {0: False, 1: False}
        # Prime each owner's first block exactly once. ComfyUI's queue shape is
        # [None] + blocks + [None]. After this prime, each executed block consumes
        # its ready entry and automatically launches the next owned block. This
        # is enough to overlap owner-A compute with owner-B DMA without any extra
        # stripe-end pop.
        for owner in (0, 1):
            if self.prefetch_queues[owner] is not None and self.owner_blocks[owner]:
                self._prefetch_pop(owner, self.blocks[self.owner_blocks[owner][0]])
                self.prefetch_primed_after_stripe[owner] = True
        self._sample_memory()

    def _critical_pipeline_active(self):
        return bool(
            self.single_root
            and self.mlp_fabric is not None
            and getattr(self.mlp_fabric, "critical_path", False)
            and self.critical_path_pipeline_window > 1
        )

    def _retire_critical_head(self):
        """Retire only the oldest critical-path block event.

        Dev15/15.1 retired a *batch* by waiting for the newest event and then
        clearing the whole pending list.  That creates a sawtooth queue: CPU
        runs ahead N blocks, then waits for block N to finish, then starts over.

        Dev15.2 keeps a true rolling CPM window.  Once more than ``window``
        blocks are outstanding, only the oldest event is retired.  The CPU stays
        a bounded number of blocks ahead of GPU0 while the critical stream never
        has to drain just because a bookkeeping window filled.
        """
        if not self._pipeline_pending:
            return
        rec = self._pipeline_pending.pop(0)
        t0 = time.perf_counter()
        rec["end"].synchronize()
        waited_ms = (time.perf_counter() - t0) * 1000.0
        self._step_barrier_ms += waited_ms
        self._step_event_wait_ms += waited_ms
        try:
            self._step_compute_ms[rec["owner"]] += float(rec["start"].elapsed_time(rec["end"]))
        except Exception:
            pass
        if self.mlp_fabric is not None:
            block_done = getattr(self.mlp_fabric, "on_block_complete", None)
            if callable(block_done):
                block_done(rec["index"])
        self._sample_memory()

    def _flush_critical_pipeline(self, force_device: bool = False):
        """Retire a bounded GPU0 window with the narrowest legal barrier.

        Dev15 used ``torch.cuda.synchronize(primary_device)`` for every pipeline
        retirement. That correctly preserved Exact semantics, but it also waited
        on unrelated DynamicVRAM/prefetch DMA and turned a scheduling checkpoint
        into a device-wide red light.

        Dev15.1 waits only for the final block event in the critical caller
        stream. A full-device barrier is reserved for cache-maintenance points
        where ``partially_unload`` may otherwise race an in-flight VBAR transfer.
        """
        if not self._pipeline_pending:
            return
        import torch
        pending = self._pipeline_pending
        t0 = time.perf_counter()
        if force_device:
            torch.cuda.synchronize(self.primary_device)
        else:
            # The final block event is downstream of Attention/MLP merge events,
            # so waiting on it retires the mathematical critical line without
            # draining independent copy/prefetch streams on the same GPU.
            pending[-1]["end"].synchronize()
        waited_ms = (time.perf_counter() - t0) * 1000.0
        self._step_barrier_ms += waited_ms
        if force_device:
            self._step_device_barrier_ms += waited_ms
        else:
            self._step_event_wait_ms += waited_ms
        self._pipeline_pending = []
        for rec in pending:
            try:
                self._step_compute_ms[rec["owner"]] += float(rec["start"].elapsed_time(rec["end"]))
            except Exception:
                pass
            if self.mlp_fabric is not None:
                block_done = getattr(self.mlp_fabric, "on_block_complete", None)
                if callable(block_done):
                    block_done(rec["index"])
        self._sample_memory()

    def execute(self, index, h, t_emb, mod_segments, rope_freqs, transformer_options=None):
        import torch
        import comfy.model_management as mm

        index = int(index)
        opts = transformer_options or {}
        if index == 0:
            self._begin_step(opts)

        owner = int(self.owner_map[index])
        dst = self.devices[owner]
        block = self.blocks[index]
        self._current_index = index

        # Exact activation handoff. We intentionally do not use D2D in Dev12.1.
        if getattr(h, "device", None) != dst:
            t0 = time.perf_counter()
            host = self.space.transport.move_tensor(h, "cpu", mode="neutral_pageable").contiguous()
            h = self.space.transport.move_tensor(host, dst, mode="neutral_pageable")
            self._step_transfer_ms += (time.perf_counter() - t0) * 1000.0
            self._step_boundary_count += 1

        t_emb_d, seg_d, rope_d = self._metadata_for_owner(owner, t_emb, mod_segments, rope_freqs)

        # Consume/advance the one-ahead queue. If the previous stripe primed this
        # owner, this pop also waits for that DMA and starts the following preload.
        self._prefetch_pop(owner, block)
        self.prefetch_primed_after_stripe[owner] = False

        if self.mlp_fabric is not None:
            self.mlp_fabric.prefetch_helper(index, opts)

        pipeline = self._critical_pipeline_active()
        if pipeline:
            # Dev15 REDLINE: record the root critical-line interval on the caller
            # stream but do not stop GPU0 after this block. The next loop can
            # immediately prepare future weights while CUDA finishes this one.
            caller = torch.cuda.current_stream(dst)
            b0 = torch.cuda.Event(enable_timing=True)
            b1 = torch.cuda.Event(enable_timing=True)
            b0.record(caller)
            with mm.cuda_device_context(dst):
                h = block(h, t_emb_d, seg_d, rope_d, transformer_options=opts)
            b1.record(caller)
            self._pipeline_pending.append({"index": index, "owner": owner, "start": b0, "end": b1})
        else:
            t1 = time.perf_counter()
            with mm.cuda_device_context(dst):
                h = block(h, t_emb_d, seg_d, rope_d, transformer_options=opts)
            torch.cuda.synchronize(dst)
            if self.mlp_fabric is not None:
                block_done = getattr(self.mlp_fabric, "on_block_complete", None)
                if callable(block_done):
                    block_done(index)
            self._step_compute_ms[owner] += (time.perf_counter() - t1) * 1000.0
            self._sample_memory()

        next_owner = self.owner_map.get(index + 1)
        stripe_ends = (next_owner is not None and int(next_owner) != owner) or index == max(self.blocks)
        if self.single_root:
            # Dev15: retire only at bounded CPM windows / cache-maintenance points.
            # This removes the historical 50 device-wide synchronizes per step.
            pending_limit = len(self._pipeline_pending) >= self.critical_path_pipeline_window
            trim_due = (((index + 1) % self.single_root_trim_interval) == 0) or index == max(self.blocks)
            if pipeline:
                if trim_due:
                    # Cache maintenance is the only legal full-device retirement.
                    self._flush_critical_pipeline(force_device=bool(trim_due))
                elif self.critical_path_rolling_retire:
                    # Dev15.2 rolling CPM: keep exactly ``window`` blocks in
                    # flight and retire only the oldest completed dependency.
                    # We intentionally use ``>`` so a 3-block window can stay
                    # fully populated instead of draining at block 3.
                    while len(self._pipeline_pending) > self.critical_path_pipeline_window:
                        self._retire_critical_head()
                elif pending_limit:
                    # Legacy Dev15/15.1 batch retirement.
                    self._flush_critical_pipeline(force_device=False)

            # Dev13: every H3 block remains on the physical primary. Keep both
            # the primary block cache and the secondary helper cache bounded
            # without any whole-hidden-state device handoff. The helper trim is
            # especially important because all 50 mirrored MLPs now live behind
            # the same secondary DynamicVRAM patcher.
            should_trim = (((index + 1) % self.single_root_trim_interval) == 0) or index == max(self.blocks)
            if should_trim and self.trim_on_stripe_boundary:
                self._trim_owner(0, force_all=False)
                self._trim_owner(1, force_all=False)
        elif stripe_ends:
            if self.trim_on_stripe_boundary:
                self._trim_owner(owner, force_all=False)
            # No explicit queue advance here. The per-block pop already launched
            # this owner's next block, so an extra pop would skip one entry.

        if index == max(self.blocks):
            # H3 final/fixed layers live on primary.
            if getattr(h, "device", None) != self.primary_device:
                t2 = time.perf_counter()
                host = self.space.transport.move_tensor(h, "cpu", mode="neutral_pageable")
                h = self.space.transport.move_tensor(host, self.primary_device, mode="neutral_pageable")
                self._step_transfer_ms += (time.perf_counter() - t2) * 1000.0
                self._step_boundary_count += 1

            wall_ms = (time.perf_counter() - self._step_started) * 1000.0
            if self.telemetry:
                helper_loaded_gib = self._loaded_size(self.helper_patchers.get(1)) / GIB
                LOG.info(
                    "H3VM %s EXACT step #%d | compute %s=%.1fms %s=%.1fms | host_transfer=%.1fms boundaries=%d | "
                    "trim=%.0f/%.0fMiB %.1f/%.1fms | wall=%.1fms | peak=%.2f/%.2fGiB | helper_resident=%.2fGiB | "
                    "prefetch_host=%.1fms barrier_wait=%.1fms pipeline=%d rolling=%s | event_wait=%.1fms device_barrier=%.1fms",
                    (("Dev18 POST-ATTN-ISLAND" if getattr(self.mlp_fabric, "post_attention_island_mode", False) else ("Dev16.0 PERSISTENT-WORKPOOL" if getattr(self.mlp_fabric, "workpool_mode", False) else ("Dev15.2 ROLLING-REDLINE" if self.critical_path_rolling_retire else "Dev15.1 RC-REDLINE"))) if self._critical_pipeline_active() else "Dev14 CRITICAL-PATH")
                    if (self.mlp_fabric is not None and getattr(self.mlp_fabric, "critical_path", False))
                    else ("Dev13 SINGLE-ROOT" if self.single_root else "Dev12"),
                    self._step,
                    self.primary_device, self._step_compute_ms[0],
                    self.secondary_device, self._step_compute_ms[1],
                    self._step_transfer_ms, self._step_boundary_count,
                    self._step_trim_mib[0], self._step_trim_mib[1],
                    self._step_trim_ms[0], self._step_trim_ms[1],
                    wall_ms, self._peak[0], self._peak[1], helper_loaded_gib,
                    self._step_prefetch_host_ms, self._step_barrier_ms, self.critical_path_pipeline_window,
                    bool(self.critical_path_rolling_retire), self._step_event_wait_ms, self._step_device_barrier_ms,
                )
                if self.mlp_fabric is not None:
                    summary_fn = getattr(self.mlp_fabric, "step_summary", None)
                    if callable(summary_fn):
                        cpm = summary_fn()
                        cpm_prefix = (
                            ("H3VM DEV16 WORKPOOL CPM STEP #%d | sidecar_calls=%d skipped=%d startup_skipped=%d | " if getattr(self.mlp_fabric, "workpool_mode", False) else ("H3VM Dev15.2 CPM STEP #%d | sidecar_calls=%d skipped=%d startup_skipped=%d | " if self.critical_path_rolling_retire else "H3VM Dev15.1 CPM STEP #%d | sidecar_calls=%d skipped=%d startup_skipped=%d | "))
                            if self._critical_pipeline_active() else
                            "H3VM Dev14 CPM STEP / Dev14.1 #%d | sidecar_calls=%d skipped=%d startup_skipped=%d | "
                        )
                        LOG.info(
                            cpm_prefix
                            + "slack min/avg=%+.1f/%+.1fms | primary_wait_helper_est=%.1fms disabled=%d | "
                            + "root_fraction min/avg/max=%.2f/%.2f/%.2f adjust(root/helper)=%d/%d",
                            self._step, int(cpm.get("calls", 0)), int(cpm.get("skipped", 0)), int(cpm.get("startup_skipped", 0)),
                            float(cpm.get("slack_min_ms", 0.0)), float(cpm.get("slack_avg_ms", 0.0)),
                            float(cpm.get("primary_stall_est_ms", 0.0)), int(cpm.get("disabled", 0)),
                            float(cpm.get("fraction_min", 0.0)), float(cpm.get("fraction_avg", 0.0)), float(cpm.get("fraction_max", 0.0)),
                            int(cpm.get("adjust_up", 0)), int(cpm.get("adjust_down", 0)),
                        )
                        if getattr(self.mlp_fabric, "workpool_mode", False):
                            LOG.info(
                                "H3VM DEV16 WORKPOOL STEP #%d | phase=%s ring_hit/miss=%d/%d ring_init=%.1fms | credit=%+d policy=%s",
                                self._step, str(cpm.get("phase", "?")), int(cpm.get("ring_hits", 0)), int(cpm.get("ring_misses", 0)),
                                float(cpm.get("ring_init_ms", 0.0)), int(cpm.get("credit", 0)), cpm.get("policy", {}),
                            )
                            if getattr(self.mlp_fabric, "ticket_workpool_mode", False):
                                LOG.info(
                                    "H3VM DEV16.1 TICKET STEP #%d | counts=%s rows=%s helper_ms=%s | cross_gpu=ONE-IN/ONE-OUT",
                                    self._step, cpm.get("ticket_counts", {}), cpm.get("ticket_rows", {}), cpm.get("ticket_helper_ms", {}),
                                )
                        if getattr(self.mlp_fabric, "post_attention_island_mode", False):
                            LOG.info(
                                "H3VM DEV18 STEP #%d | phase=%s post_island=%s coverage=%d actual_calls=%d | helper_prefetch_wait=%.1fms prime=%.1fms errors=%d",
                                self._step, str(getattr(self.mlp_fabric, "_phase", "?")), bool(getattr(self.mlp_fabric, "_island_enabled", False)),
                                int(cpm.get("coverage_target", 0)), int(cpm.get("calls", 0)),
                                float(cpm.get("helper_prefetch_host_ms", 0.0)), float(cpm.get("helper_prefetch_prime_ms", 0.0)),
                                int(cpm.get("helper_prefetch_errors", 0)),
                            )
                        elif getattr(self.mlp_fabric, "rolling_adaptive_load_mode", False):
                            LOG.info(
                                "H3VM DEV17.1 ADAPTIVE STEP #%d | phase=%s coverage=%d actual_calls=%d | root_plan=%s avg=%.3f | helper_prefetch_wait=%.1fms prime=%.1fms errors=%d",
                                self._step, str(cpm.get("phase", "?")), int(cpm.get("coverage_target", 0)), int(cpm.get("calls", 0)),
                                cpm.get("planned_fraction_hist", {}), float(cpm.get("planned_fraction_avg", 0.68)),
                                float(cpm.get("helper_prefetch_host_ms", 0.0)), float(cpm.get("helper_prefetch_prime_ms", 0.0)),
                                int(cpm.get("helper_prefetch_errors", 0)),
                            )
                        elif getattr(self.mlp_fabric, "rolling_helper_matrix_mode", False):
                            LOG.info(
                                "H3VM DEV17 ROLLING HELPER STEP #%d | coverage=%d actual_calls=%d | helper_prefetch_wait=%.1fms prime=%.1fms calls=%d errors=%d broken=%s",
                                self._step, int(cpm.get("coverage_target", 0)), int(cpm.get("calls", 0)),
                                float(cpm.get("helper_prefetch_host_ms", 0.0)), float(cpm.get("helper_prefetch_prime_ms", 0.0)),
                                int(cpm.get("helper_prefetch_calls", 0)), int(cpm.get("helper_prefetch_errors", 0)),
                                bool(cpm.get("helper_prefetch_broken", False)),
                            )

            if self.mlp_fabric is not None:
                record_step_runtime = getattr(self.mlp_fabric, "record_step_runtime", None)
                if callable(record_step_runtime):
                    record_step_runtime(
                        wall_ms=wall_ms, primary_compute_ms=self._step_compute_ms[0],
                        prefetch_host_ms=self._step_prefetch_host_ms, event_wait_ms=self._step_event_wait_ms,
                        device_barrier_ms=self._step_device_barrier_ms, barrier_wait_ms=self._step_barrier_ms,
                    )

            # Retire the final prefetched entry, then drop pins/queues.
            self._finish_prefetch(0)
            self._finish_prefetch(1)
            self._metadata_cache = {0: None, 1: None}
            try:
                import comfy.model_prefetch as mp
                mp.cleanup_prefetch_queues()
            except Exception:
                pass
            self.prefetch_queues = {0: None, 1: None}
            if self.expected_steps and self._step >= self.expected_steps:
                self._step = 0

        return h

    def sampling_end(self):
        try:
            self._flush_critical_pipeline()
        except Exception:
            pass
        try:
            import comfy.model_prefetch as mp
            mp.cleanup_prefetch_queues()
        except Exception:
            pass
        self.prefetch_queues = {0: None, 1: None}
        self._metadata_cache = {0: None, 1: None}
        sidecar_thread = getattr(self, "_sidecar_prepare_thread", None)
        if sidecar_thread is not None and sidecar_thread.is_alive():
            sidecar_thread.join(timeout=5.0)
        self._sidecar_prepare_thread = None
        if self.mlp_fabric is not None:
            final_summary = getattr(self.mlp_fabric, "final_summary", None)
            if callable(final_summary):
                try:
                    import json
                    LOG.info("H3VM DEV16 FINAL RESULT %s", json.dumps(final_summary(), ensure_ascii=False, separators=(",", ":")))
                except Exception as exc:
                    LOG.warning("H3VM DEV16 final summary failed: %r", exc)
        # The central RAM-first invariant: leave both block islands cold between
        # prompts so second Queue starts from essentially the same state.
        self._trim_owner(0, force_all=True)
        self._trim_owner(1, force_all=True)
        self._prepared = False
        self._step = 0

        if self.hard_cleanup_after_sample:
            try:
                import comfy.model_management as mm
                mm.soft_empty_cache()
            except Exception as exc:
                LOG.warning("H3VM Dev12 hard cleanup warning: %r", exc)

    def close(self):
        try:
            self.sampling_end()
        except Exception:
            pass
        self.patchers = {0: None, 1: None}
        self.helper_patchers = {0: None, 1: None}
        if self.mlp_fabric is not None:
            try:
                self.mlp_fabric.close()
            except Exception:
                pass
        self.mlp_fabric = None
        self.blocks.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class StreamingBlockProxyFactory:
    @staticmethod
    def make(runtime, index):
        import torch
        import weakref

        class H3StreamingBlockProxy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._runtime_ref = weakref.ref(runtime)
                self.index = int(index)

            def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
                rt = self._runtime_ref()
                if rt is None:
                    raise RuntimeError("H3VM Dev12 runtime released")
                return rt.execute(
                    self.index, x, t_emb, mod_segments, rope_freqs,
                    transformer_options=transformer_options,
                )

        return H3StreamingBlockProxy()
