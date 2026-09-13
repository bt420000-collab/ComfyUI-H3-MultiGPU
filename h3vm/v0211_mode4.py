from __future__ import annotations

"""Small v0.21.1 compatibility layer for the proven Mode4 runtime.

It adds two release contracts without replacing the large historical runtime:
1) OUTER_SAMPLE owns run boundaries, not expected_steps_hint;
2) SPECTRAL uses a tiny CPU ridge-polynomial forecast over true host snapshots.
"""

import time

_PATCHED = False


def _scalar(value):
    try:
        import torch
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().reshape(-1).float().mean().item())
        return float(value)
    except Exception:
        return None


def _spectral_weights(coords, target, ridge=0.02):
    import torch
    if len(coords) < 2:
        return None
    degree = min(2, len(coords) - 1)
    vals = [float(x) for x in coords] + [float(target)]
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        return None
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    xs = [(x - center) / half for x in coords]
    xt = (float(target) - center) / half
    X = torch.tensor([[x ** p for p in range(degree + 1)] for x in xs], dtype=torch.float64)
    phi = torch.tensor([xt ** p for p in range(degree + 1)], dtype=torch.float64)
    A = X.T @ X + float(ridge) * torch.eye(degree + 1, dtype=torch.float64)
    try:
        proj = torch.linalg.solve(A, X.T)
    except Exception:
        proj = torch.linalg.pinv(A) @ X.T
    weights = (phi @ proj).tolist()
    total = float(sum(weights))
    if abs(total) > 1e-9:
        weights = [float(w) / total for w in weights]
    return [float(w) for w in weights]


def install_snapshot_runtime_patch():
    global _PATCHED
    if _PATCHED:
        return True
    from . import snapshot_islands as si
    cls = si.SnapshotIslandRuntime
    if getattr(cls, "_h3vm_v0211_patched", False):
        _PATCHED = True
        return True
    original_begin = cls._begin_prefix
    original_run_tail = cls._run_tail

    def sampling_begin(self):
        self._reset_sampling_run()
        self._h3vm_history = []
        self._h3vm_timestep = None

    def sampling_end(self):
        self._reset_sampling_run()
        self._h3vm_history = []
        self._h3vm_timestep = None

    def observe_timestep(self, timestep):
        value = _scalar(timestep)
        if value is not None:
            self._h3vm_timestep = value

    def begin_prefix(self, h, t_emb, mod_segments, rope_freqs, transformer_options):
        hist = getattr(self, "_h3vm_history", [])
        if hist and tuple(hist[-1][1].shape) != tuple(h.shape):
            self._h3vm_history = []
        return original_begin(self, h, t_emb, mod_segments, rope_freqs, transformer_options)

    def run_tail(self, snapshot_cpu, t_emb, mod_segments, rope_freqs, transformer_options, step,
                 predictor_prev_cpu=None, predictor_mode="stale", predictor_beta=0.75):
        requested = str(getattr(self, "_h3vm_predictor", predictor_mode)).lower()
        if "spectral" not in requested:
            return original_run_tail(self, snapshot_cpu, t_emb, mod_segments, rope_freqs, transformer_options,
                                     step, predictor_prev_cpu, "linear" if "linear" in requested else predictor_mode,
                                     predictor_beta)
        history = list(getattr(self, "_h3vm_history", []))
        target = getattr(self, "_h3vm_timestep", None)
        if target is None:
            target = float(step)
        usable = history[-4:]
        if len(usable) < 2:
            return original_run_tail(self, snapshot_cpu, t_emb, mod_segments, rope_freqs, transformer_options,
                                     step, predictor_prev_cpu, "linear", predictor_beta)
        coords = [x for x, _ in usable]
        weights = _spectral_weights(coords, target, getattr(self, "_h3vm_spectral_ridge", 0.02))
        if not weights:
            return original_run_tail(self, snapshot_cpu, t_emb, mod_segments, rope_freqs, transformer_options,
                                     step, predictor_prev_cpu, "linear", predictor_beta)
        t0 = time.perf_counter()
        predicted = None
        for (_, snap), w in zip(usable, weights):
            term = snap.mul(float(w))
            predicted = term if predicted is None else predicted.add(term)
        result = original_run_tail(self, predicted, t_emb, mod_segments, rope_freqs, transformer_options,
                                   step, None, "stale", predictor_beta)
        result["predictor_used"] = True
        result["predictor_ms"] = float(result.get("predictor_ms", 0.0)) + (time.perf_counter() - t0) * 1000.0
        result["predictor_kind"] = "spectral"
        return result

    def tail_execute(self, index, h, t_emb, mod_segments, rope_freqs, transformer_options=None):
        if index == self.first_tail:
            h = self._consume_tail(t_emb, mod_segments, rope_freqs, transformer_options)
        else:
            h = self._tail_output if self._tail_output is not None else h
        if index == self.last_tail:
            self._snapshot_prev_cpu = self._snapshot_cpu
            self._snapshot_cpu = self._pending_snapshot_cpu
            if self._snapshot_cpu is not None:
                coord = getattr(self, "_h3vm_timestep", None)
                if coord is None:
                    coord = float(self._step)
                history = list(getattr(self, "_h3vm_history", []))
                history.append((float(coord), self._snapshot_cpu))
                self._h3vm_history = history[-4:]
            self._reset_call_state()
        return h

    cls.sampling_begin = sampling_begin
    cls.sampling_end = sampling_end
    cls.observe_timestep = observe_timestep
    cls._begin_prefix = begin_prefix
    cls._run_tail = run_tail
    cls.tail_execute = tail_execute
    cls._h3vm_v0211_patched = True
    _PATCHED = True
    return True


def configure_mode4_patcher(patcher, requested_predictor="linear_raw", spectral_ridge=0.02):
    install_snapshot_runtime_patch()
    runtime = (getattr(patcher, "attachments", {}) or {}).get("h3vm_snapshot_runtime")
    if runtime is None:
        return patcher
    runtime._h3vm_predictor = str(requested_predictor)
    runtime._h3vm_spectral_ridge = float(spectral_ridge)
    runtime._h3vm_history = []
    runtime._h3vm_timestep = None
    try:
        import comfy.patcher_extension
        def coordinate_wrapper(executor, *args, **kwargs):
            ts = args[1] if len(args) > 1 else kwargs.get("timestep")
            runtime.observe_timestep(ts)
            return executor(*args, **kwargs)
        patcher.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                     "h3vm_v0211_scheduler_coordinate", coordinate_wrapper)
        def sample_wrapper(executor, *args, **kwargs):
            runtime.sampling_begin()
            try:
                return executor(*args, **kwargs)
            finally:
                runtime.sampling_end()
        patcher.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                     "h3vm_v0211_sampling_lifecycle", sample_wrapper)
    except Exception:
        pass
    return patcher
