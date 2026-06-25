"""CUDA graph runner for the INT8 two-kernel path.

Captures prep → quantize → GEMM into a replayable graph, eliminating
kernel launch overhead. One ``CudaGraphRunner`` per qualifying
``IntCrushOps.Linear`` (Triton available, no zero-point, no VBAR),
caching a graph per unique ``(M, N, K, dtype, has_bias)`` shape.

Usage::

    runner = CudaGraphRunner(layer)
    out = runner.run(x, bias, compute_dtype)
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch
import torch.nn.functional as F

from . import _triton_runtime as _tr
from ._quant_utils import rotate_activations

log = logging.getLogger(__name__)

__all__ = ["CudaGraphRunner"]


class _GraphEntry:
    """A cached CUDA graph plus its static buffers for one shape.

    All tensors written by the graph must stay referenced here for the
    graph's lifetime — dropping one lets the caching allocator reuse the
    memory, causing silent corruption on replay.
    """

    __slots__ = (
        "graph",
        "static_x",
        "static_x_int8",
        "static_s_a",
        "static_out",
        "bias_snapshot",
        "_no_clone",
    )

    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        static_x: torch.Tensor,
        static_x_int8: torch.Tensor,
        static_s_a: torch.Tensor,
        static_out: torch.Tensor,
        bias_snapshot: torch.Tensor | None,
        no_clone: bool = False,
    ) -> None:
        self.graph = graph
        self.static_x = static_x
        self.static_x_int8 = static_x_int8
        self.static_s_a = static_s_a
        self.static_out = static_out
        self.bias_snapshot = bias_snapshot
        self._no_clone = no_clone

    def replay(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Copy input into static buffer, replay graph, return output.

        With ``no_clone``, returns the static output buffer directly (no
        allocation) — caller must consume it before the next replay;
        safe in sequential inference.
        """
        self.static_x.copy_(x_2d)
        self.graph.replay()
        if self._no_clone:
            return self.static_out
        return self.static_out.clone()


class CudaGraphRunner:
    """Per-layer CUDA graph cache for the INT8 two-kernel path.

    Captures ``prep_activation → dynamic_quantize → fused_int8_gemm``
    into one graph per unique shape.  Stores its own static tensor refs
    and does not call back into ``layer._prep_activation()``.

    Args:
        layer: ``IntCrushOps.Linear`` module to read attributes from.
        safe_no_clone: If ``True`` (default), ``replay()`` returns the
            static output buffer without ``.clone()`` — saves one memcpy
            per replay; safe when the caller consumes the output before
            the next replay (normal sequential inference).  Set ``False``
            if the caller holds the tensor across calls.
        max_cached_shapes: Max distinct shapes cached per layer; least-
            recently-used entries are evicted (freeing their buffers).
    """

    _MAX_CACHED_SHAPES: int = 4

    def __init__(
        self,
        layer,
        safe_no_clone: bool = True,
        max_cached_shapes: int | None = None,
    ) -> None:
        # ── Static layer attributes (read once) ──
        self._w_int8: torch.Tensor = layer.weight._qdata
        self._s_w: torch.Tensor = layer.weight._params.scale
        self._smoothrot_factors: torch.Tensor | None = layer._intcrush_smoothrot_factors
        self._rot_need: bool = layer._intcrush_rot_need
        self._rot_size: int = layer._intcrush_rot_size
        self._w_in: int = layer._intcrush_w_in
        self._perm: torch.Tensor | None = layer._intcrush_perm
        self._smooth: torch.Tensor | None = layer._intcrush_smooth
        self._zp: torch.Tensor | None = layer.weight._params.zp
        self._safe_no_clone: bool = safe_no_clone
        self._max_cached: int = (
            max_cached_shapes if max_cached_shapes is not None
            else self._MAX_CACHED_SHAPES
        )

        # ── LRU cache: (M, N, K, dtype, has_bias) → _GraphEntry ──
        self._graphs: OrderedDict[tuple, _GraphEntry] = OrderedDict()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run the INT8 two-kernel path, using a cached graph when possible.

        Args:
            x: input ``[..., in_features]``
            bias: optional ``[out_features]``
            compute_dtype: output dtype (fp16 or bf16)

        Returns:
            2-D output ``[M, out_features]`` in *compute_dtype*;
            ``_finish_forward`` reshapes it back to match *x*.
        """
        # 1. Reshape to 2D (view, no CUDA op).
        x_2d = x.reshape(-1, x.shape[-1])
        M, K = x_2d.shape
        N = self._w_int8.shape[0]

        # 2. Zero-point not graphed — fall through to eager.
        if self._zp is not None:
            return self._run_eager(x_2d, bias, compute_dtype)

        # 3. Build cache key.
        key = (M, N, K, x.dtype, bias is not None)

        # 4. Lookup; replay on hit (after bias-pointer safety check).
        entry = self._graphs.get(key)
        if entry is not None:
            if entry.bias_snapshot is bias:
                self._graphs.move_to_end(key)
                return entry.replay(x_2d)
            # Bias pointer changed — invalidate and re-capture.
            del self._graphs[key]

        # 5. Miss: run eagerly (warms autotuner), then capture.
        out = self._run_eager(x_2d, bias, compute_dtype)
        self._capture(key, x_2d, bias, compute_dtype)

        # LRU eviction when over cap.
        while len(self._graphs) > self._max_cached:
            _, evicted = self._graphs.popitem(last=False)
            del evicted
            log.debug("[INT-Crush] Evicted oldest CUDA graph entry (cache full)")

        return out

    def invalidate(self) -> None:
        """Clear all cached graphs.

        Called on weight change (LoRA attach/detach, device move,
        weight replacement).  ``_ops.py`` usually sets
        ``self._graph_runner = None`` to destroy the runner entirely;
        use this when the runner should be kept.
        """
        self._graphs.clear()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _prep_and_quantize(
        self,
        x_2d: torch.Tensor,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
        *,
        s_a_out: torch.Tensor | None = None,
        x_int8_out: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full prep → quantize → GEMM path.

        When ``s_a_out``/``x_int8_out``/``out`` are given, the Triton
        kernels write into those pre-allocated buffers (graph capture).
        When ``None``, the kernels allocate their own (eager).
        """
        # ── Prep activation ──
        if self._smoothrot_factors is not None:
            x_2d = x_2d / self._smoothrot_factors.to(
                device=x_2d.device, dtype=x_2d.dtype,
            )
        if self._rot_need:
            x_2d = rotate_activations(x_2d, self._rot_size)
        if x_2d.shape[-1] < self._w_in:
            x_2d = F.pad(x_2d, (0, self._w_in - x_2d.shape[-1]))
        if self._perm is not None:
            x_2d = x_2d[..., self._perm]

        # Old SmoothQuant: 1/s after Hadamard (non-SmoothRot layers only).
        if self._smoothrot_factors is None:
            smooth = self._smooth
            if smooth is not None:
                x_2d = x_2d / smooth.to(device=x_2d.device, dtype=x_2d.dtype)

        # ── Quantize ──
        x_int8, s_a = _tr.dynamic_quantize_activation(
            x_2d, s_a_out=s_a_out, x_int8_out=x_int8_out,
        )

        # ── GEMM + dequant ──
        return _tr.fused_int8_gemm_dequant(
            x_int8,
            self._w_int8,
            self._s_w,
            s_a,
            bias=bias,
            out_dtype=compute_dtype,
            out=out,
        )

    def _run_eager(
        self,
        x_2d: torch.Tensor,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run the same ops the graph would, without pre-allocated buffers.

        Also warms the Triton autotuner so the right kernel config is
        selected before capture.
        """
        return self._prep_and_quantize(x_2d, bias, compute_dtype)

    def _capture(
        self,
        key: tuple,
        x_2d: torch.Tensor,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
    ) -> None:
        """Record a CUDA graph for the given shape and cache it."""
        M, K = x_2d.shape
        N = self._w_int8.shape[0]

        # ── Static buffers (graph reads/writes these on replay) ──
        static_x = torch.empty(M, K, dtype=x_2d.dtype, device=x_2d.device)
        static_x_int8 = torch.empty(M, self._w_in, dtype=torch.int8, device=x_2d.device)
        static_s_a = torch.empty(M, dtype=torch.float32, device=x_2d.device)
        static_out = torch.empty(M, N, dtype=compute_dtype, device=x_2d.device)

        # Seed with valid data so the captured graph operates on it.
        static_x.copy_(x_2d)

        # Ensure autotuner warmup from _run_eager is complete.
        torch.cuda.synchronize()

        # ── Record graph ──
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._prep_and_quantize(
                static_x,
                bias,
                compute_dtype,
                s_a_out=static_s_a,
                x_int8_out=static_x_int8,
                out=static_out,
            )

        # ── Cache entry ──
        # All buffers must stay referenced or the allocator reuses their
        # memory → silent corruption on replay.
        self._graphs[key] = _GraphEntry(
            graph, static_x, static_x_int8, static_s_a, static_out, bias,
            no_clone=self._safe_no_clone,
        )
        log.debug(
            "[INT-Crush] Captured CUDA graph: M=%d N=%d K=%d dtype=%s has_bias=%s",
            M, N, K, compute_dtype, bias is not None,
        )
