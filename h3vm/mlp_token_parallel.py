from __future__ import annotations

import copy
import logging
import time
import types
import weakref

LOG = logging.getLogger("H3VM")
MIB = 1024 ** 2


def _clone_linear_shared(src):
    """Clone a Comfy Linear shell while sharing the immutable CPU weight storage.

    The Parameter objects are distinct so DynamicVRAM can virtualize them on
    different devices, but the initial CPU tensor storage is shared.  This makes
    the helper tree a second *view* of the model, not a second RAM copy.
    """
    import torch

    m = copy.copy(src)
    m._parameters = dict(getattr(src, "_parameters", {}))
    m._buffers = dict(getattr(src, "_buffers", {}))
    m._modules = dict(getattr(src, "_modules", {}))
    m.weight = torch.nn.Parameter(src.weight, requires_grad=False)
    if getattr(src, "bias", None) is not None:
        m.bias = torch.nn.Parameter(src.bias, requires_grad=False)
    else:
        m.bias = None
    if hasattr(m, "weight_function"):
        m.weight_function = list(getattr(src, "weight_function", []))
    if hasattr(m, "bias_function"):
        m.bias_function = list(getattr(src, "bias_function", []))
    # Runtime-only VBAR/prefetch state must never be shared between the owner and
    # helper module shells.
    for name in ("_v", "_v_weight", "_v_bias", "_v_signature", "_prefetch"):
        if hasattr(m, name):
            try:
                delattr(m, name)
            except Exception:
                pass
    return m


def clone_mlp_shared(src_mlp):
    import torch

    class H3TokenHelperMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = _clone_linear_shared(src_mlp.fc1)
            self.fc2 = _clone_linear_shared(src_mlp.fc2)

        def forward(self, x):
            import comfy.ops
            return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")

    return H3TokenHelperMLP()


def make_helper_block(mlp, index: int):
    import torch

    class H3MLPHelperBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mlp
            self.h3vm_index = int(index)

        def forward(self, *args, **kwargs):
            raise RuntimeError("H3VM Dev12.3 helper block is storage-only; call .mlp")

    return H3MLPHelperBlock()


def build_helper_maps(blocks, owner_map):
    """Create opposite-device MLP mirrors using shared CPU base storage.

    Returns:
      primary_helpers: blocks owned by secondary, executed on primary as helper
      secondary_helpers: blocks owned by primary, executed on secondary as helper
      helper_mlp_by_block: block index -> helper MLP module
    """
    primary_helpers = {}
    secondary_helpers = {}
    helper_mlp_by_block = {}
    for i, block in enumerate(blocks):
        helper_mlp = clone_mlp_shared(block.mlp)
        helper_mlp_by_block[int(i)] = helper_mlp
        hb = make_helper_block(helper_mlp, i)
        if int(owner_map[int(i)]) == 0:
            secondary_helpers[int(i)] = hb
        else:
            primary_helpers[int(i)] = hb
    return primary_helpers, secondary_helpers, helper_mlp_by_block


class TokenParallelMLPFabric:
    """Exact two-GPU token-parallel H3 MLP.

    H3's MLP is token-local.  We therefore partition only the sequence rows,
    keeping each token's full FC1 -> SwiGLU -> FC2 arithmetic on one GPU.  This
    preserves the fused INT8 per-row activation quantization semantics and avoids
    shipping the enormous FFN intermediate over PCIe.
    """

    def __init__(self, *, primary, secondary, owner_map, helper_mlp_by_block,
                 host_engine, primary_fraction=0.60, min_sequence_length=8192,
                 helper_safety_mb=1024, telemetry=True):
        import torch
        self.primary = torch.device(primary)
        self.secondary = torch.device(secondary)
        self.owner_map = dict(owner_map)
        self.helper_mlp_by_block = dict(helper_mlp_by_block)
        self.host_engine = host_engine
        self.primary_fraction = min(0.85, max(0.15, float(primary_fraction)))
        self.min_sequence_length = max(1, int(min_sequence_length))
        self.helper_safety_mb = max(128, int(helper_safety_mb))
        self.telemetry = bool(telemetry)
        self.streams = {
            self.primary: torch.cuda.Stream(device=self.primary),
            self.secondary: torch.cuda.Stream(device=self.secondary),
        }
        self.calls = 0
        self._logged = set()
        self._prefetch_logged = set()
        self._prefetch_queues = {}
        self._disabled_blocks = set()

    def helper_device(self, index: int):
        return self.secondary if int(self.owner_map[int(index)]) == 0 else self.primary

    def owner_device(self, index: int):
        return self.primary if int(self.owner_map[int(index)]) == 0 else self.secondary

    def _split(self, seq: int):
        # Keep a reasonably aligned M dimension for the fused kernels while
        # preserving both participants. Alignment changes only token assignment.
        cut = int(round(seq * self.primary_fraction / 128.0)) * 128
        return max(128, min(seq - 128, cut)) if seq >= 256 else max(1, seq // 2)

    def _helper_has_runway(self, helper):
        import torch
        try:
            free, _ = torch.cuda.mem_get_info(helper)
            return int(free) > self.helper_safety_mb * MIB
        except Exception:
            return True

    def prefetch_helper(self, index: int, transformer_options=None):
        """Best-effort helper MLP weight prefetch while owner enters the block.

        This runs before norm/attention, so helper FC1/FC2 weight DMA can overlap
        with the block's owner-side attention.  Any backend quirk simply disables
        prefetch for that block; correctness never depends on it.
        """
        if int(index) in self._disabled_blocks:
            return
        helper = self.helper_mlp_by_block.get(int(index))
        if helper is None:
            return
        device = self.helper_device(index)
        try:
            import comfy.model_prefetch as mp
            opts = dict(transformer_options or {})
            opts["prefetch_dynamic_vbars"] = True
            q = mp.make_prefetch_queue([helper], device, opts)
            if q is not None:
                mp.prefetch_queue_pop(q, device, helper)
                self._prefetch_queues[int(index)] = q
            if int(index) not in self._prefetch_logged:
                LOG.info(
                    "H3VM Dev12.3 MLP helper prefetch | block=%d helper=%s active=%s",
                    int(index), device, q is not None,
                )
                self._prefetch_logged.add(int(index))
        except Exception as exc:
            if int(index) not in self._prefetch_logged:
                LOG.warning("H3VM Dev12.3 MLP helper prefetch disabled block=%d: %r", int(index), exc)
                self._prefetch_logged.add(int(index))
            self._prefetch_queues.pop(int(index), None)

    def _finish_prefetch(self, index: int):
        q = self._prefetch_queues.pop(int(index), None)
        if q is None:
            return
        try:
            import comfy.model_prefetch as mp
            if len(q) >= 2:
                mp.prefetch_queue_pop(q, self.helper_device(index), None)
        except Exception:
            pass

    @staticmethod
    def _run_mlp(mlp, x):
        import comfy.ops
        return comfy.ops.linear_input_act(mlp.fc2, mlp.fc1(x), "swiglu")

    def execute(self, index: int, owner_mlp, x):
        import torch
        import comfy.model_management as mm

        index = int(index)
        seq = int(x.shape[0])
        owner = self.owner_device(index)
        helper = self.helper_device(index)
        if seq < self.min_sequence_length or index in self._disabled_blocks or not self._helper_has_runway(helper):
            return self._run_mlp(owner_mlp, x)
        if x.device != owner:
            raise RuntimeError(f"H3VM Dev12.3 MLP owner mismatch block={index}: x={x.device} owner={owner}")

        helper_mlp = self.helper_mlp_by_block[index]
        cut = self._split(seq)
        # Token order is [primary rows | secondary rows] regardless of which card
        # owns the current stripe.  This makes reconstruction a simple cat.
        x_primary = x[:cut]
        x_secondary = x[cut:]
        if owner == self.primary:
            local_x, remote_x = x_primary, x_secondary
            local_range = (0, cut)
            remote_range = (cut, seq)
        else:
            remote_x, local_x = x_primary, x_secondary
            remote_range = (0, cut)
            local_range = (cut, seq)

        t_stage = time.perf_counter()
        remote_x = self.host_engine.move_tensor(remote_x.contiguous(), helper)
        stage_ms = (time.perf_counter() - t_stage) * 1000.0

        owner_stream = self.streams[owner]
        helper_stream = self.streams[helper]
        caller_stream = torch.cuda.current_stream(owner)
        owner_stream.wait_stream(caller_stream)

        le0 = torch.cuda.Event(enable_timing=True)
        le1 = torch.cuda.Event(enable_timing=True)
        he0 = torch.cuda.Event(enable_timing=True)
        he1 = torch.cuda.Event(enable_timing=True)

        # Helper launches first, then owner. CUDA kernels execute concurrently on
        # physically separate devices once the helper input is staged.
        with torch.cuda.device(helper), torch.cuda.stream(helper_stream), mm.cuda_device_context(helper):
            he0.record(helper_stream)
            remote_out = helper_mlp(remote_x)
            he1.record(helper_stream)

        with torch.cuda.device(owner), torch.cuda.stream(owner_stream), mm.cuda_device_context(owner):
            le0.record(owner_stream)
            local_out = self._run_mlp(owner_mlp, local_x)
            le1.record(owner_stream)

        helper_stream.synchronize()
        owner_stream.synchronize()
        helper_ms = float(he0.elapsed_time(he1))
        owner_ms = float(le0.elapsed_time(le1))

        t_return = time.perf_counter()
        returned = self.host_engine.move_tensor(remote_out, owner)
        return_ms = (time.perf_counter() - t_return) * 1000.0

        # Restore original token order exactly.
        if remote_range[0] == 0:
            out = torch.cat((returned, local_out), dim=0)
        else:
            out = torch.cat((local_out, returned), dim=0)

        caller_stream.wait_stream(owner_stream)
        out.record_stream(caller_stream)
        self._finish_prefetch(index)

        self.calls += 1
        if self.telemetry and (self.calls <= 4 or self.calls % 50 == 0):
            LOG.info(
                "H3VM Dev12.3 token MLP #%d | block=%d owner=%s helper=%s rows(primary/secondary)=%d/%d | "
                "stage=%.1fms compute(owner/helper)=%.1f/%.1fms return=%.1fms | parallel_compute_window=%.1fms",
                self.calls, index, owner, helper, cut, seq-cut,
                stage_ms, owner_ms, helper_ms, return_ms, max(owner_ms, helper_ms),
            )
        return out

    def close(self):
        self._prefetch_queues.clear()
        self.helper_mlp_by_block.clear()


def install_token_parallel_mlp(blocks, fabric: TokenParallelMLPFabric):
    """Monkey-patch only MLP.forward; fc1/fc2 module paths stay canonical.

    Keeping the Linear modules at blocks.N.mlp.fc1/fc2 is essential: Larry's
    bypass injections and fused-fc2 merge patches continue to resolve exactly as
    in the baseline H3 Turbo runtime.
    """
    installed = 0
    for i, block in enumerate(blocks):
        mlp = block.mlp
        if hasattr(mlp, "_h3vm_dev12_3_original_forward"):
            continue
        mlp._h3vm_dev12_3_original_forward = mlp.forward
        fref = weakref.ref(fabric)

        def forward(self, x, _idx=int(i), _fref=fref):
            f = _fref()
            if f is None:
                import comfy.ops
                return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")
            return f.execute(_idx, self, x)

        mlp.forward = types.MethodType(forward, mlp)
        installed += 1
    return installed
