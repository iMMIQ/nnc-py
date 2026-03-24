"""Compile-time weight prepacking and lowering hints."""

from __future__ import annotations

from typing import Any

import numpy as np

from nnc_py.ir.context import CompileContext
from nnc_py.ir.node import Node, OpType
from nnc_py.passes.base import PassBase


class PrepackLoweringPass(PassBase):
    """Attach lowering metadata and prepack eligible constant weights."""

    _CONV_OPS = {
        OpType.CONV2D,
        OpType.FUSED_CONV_RELU,
        OpType.FUSED_CONV_BIAS_RELU,
        OpType.FUSED_CONV_SIGMOID,
    }

    @property
    def name(self) -> str:
        return "PrepackLowering"

    def _execute(self, ctx: CompileContext) -> None:
        lowered = 0
        prepacked = 0

        for node in ctx.graph.topological_sort():
            if node.op_type in self._CONV_OPS:
                self._annotate_conv_lowering(ctx, node)
                lowered += 1
                continue

            if node.op_type == OpType.GEMM:
                if self._annotate_and_prepack_gemm(ctx, node):
                    prepacked += 1
                lowered += 1

        ctx.metadata["prepack_lowering_summary"] = {
            "lowered_nodes": lowered,
            "prepacked_weights": prepacked,
        }

    def _annotate_conv_lowering(self, ctx: CompileContext, node: Node) -> None:
        kernel_shape = _normalize_hw(node.attrs.get("kernel_shape", [1, 1]), fill=1)
        strides = _normalize_hw(node.attrs.get("strides", [1, 1]), fill=1)
        pads = _normalize_pads(node.attrs.get("pads", [0, 0, 0, 0]))
        group = int(node.attrs.get("group", 1))
        weight_name = node.inputs[1] if len(node.inputs) >= 2 else None
        weight_is_constant = weight_name in ctx.graph.constants if weight_name else False

        kernel_kind = "generic"
        kh, kw = kernel_shape
        sh, sw = strides
        pad_h, pad_w = pads[0], pads[1]
        if kh == 1 and kw == 1:
            kernel_kind = "pointwise_1x1"
        elif kh == 3 and kw == 3 and sh == 1 and sw == 1 and group == 1 and pad_h == 1 and pad_w == 1:
            kernel_kind = "spatial_3x3"
        elif kh == 7 and kw == 7 and sh == 2 and sw == 2 and pad_h == 3 and pad_w == 3:
            kernel_kind = "stem_7x7_s2"
        elif group > 1:
            kernel_kind = "grouped"

        lowering = self._get_or_create_lowering(node)
        lowering["kernel_family"] = "conv2d"
        lowering["kernel_kind"] = kernel_kind
        lowering["kernel_shape"] = kernel_shape
        lowering["strides"] = strides
        lowering["pads"] = pads
        lowering["group"] = group
        lowering["weight_is_constant"] = weight_is_constant
        lowering["weight_pack"] = (
            "oihw_constant" if weight_is_constant else "oihw_runtime"
        )

    def _annotate_and_prepack_gemm(self, ctx: CompileContext, node: Node) -> bool:
        lowering = self._get_or_create_lowering(node)
        lowering["kernel_family"] = "gemm"

        if len(node.inputs) < 2:
            lowering["weight_pack"] = "rhs_runtime"
            return False

        weight_name = node.inputs[1]
        trans_b = int(node.attrs.get("transB", 0))
        if trans_b == 1:
            lowering["weight_pack"] = (
                "rhs_transposed_constant"
                if weight_name in ctx.graph.constants
                else "rhs_runtime"
            )
            return False

        if weight_name not in ctx.graph.constants:
            lowering["weight_pack"] = "rhs_runtime"
            return False

        weight_tensor = ctx.graph.tensors.get(weight_name)
        weight_value = ctx.graph.constants.get(weight_name)
        if weight_tensor is None or weight_value is None or weight_value.ndim != 2:
            lowering["weight_pack"] = "rhs_runtime"
            return False

        if len(ctx.graph.get_consumers(weight_name)) != 1:
            lowering["weight_pack"] = "rhs_shared_constant"
            return False

        packed_weight = np.ascontiguousarray(weight_value.T)
        ctx.graph.constants[weight_name] = packed_weight
        weight_tensor.shape.dims = [int(dim) for dim in packed_weight.shape]
        node.attrs["transB"] = 1
        lowering["weight_pack"] = "rhs_transposed_constant"
        return True

    def _get_or_create_lowering(self, node: Node) -> dict[str, Any]:
        lowering = node.metadata.get("lowering")
        if isinstance(lowering, dict):
            return lowering

        lowering = {}
        node.metadata["lowering"] = lowering
        return lowering


def _normalize_hw(value: object, *, fill: int) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if isinstance(value, (list, tuple)) and len(value) == 1:
        dim = int(value[0])
        return dim, dim
    return fill, fill


def _normalize_pads(value: object) -> tuple[int, int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return tuple(int(dim) for dim in value[:4])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        pad_h = int(value[0])
        pad_w = int(value[1])
        return pad_h, pad_w, pad_h, pad_w
    if isinstance(value, (list, tuple)) and len(value) == 1:
        pad = int(value[0])
        return pad, pad, pad, pad
    return 0, 0, 0, 0
