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
        kernel_shape = list(node.attrs.get("kernel_shape", [1, 1]))
        strides = list(node.attrs.get("strides", [1, 1]))
        pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
        group = int(node.attrs.get("group", 1))
        weight_name = node.inputs[1] if len(node.inputs) >= 2 else None

        kernel_kind = "generic"
        if len(kernel_shape) >= 2:
            kh, kw = int(kernel_shape[0]), int(kernel_shape[1])
            sh = int(strides[0]) if strides else 1
            sw = int(strides[1]) if len(strides) > 1 else sh
            if len(pads) >= 4:
                pad_h = int(pads[0])
                pad_w = int(pads[1])
            elif len(pads) >= 2:
                pad_h = int(pads[0])
                pad_w = int(pads[1])
            else:
                pad_h = 0
                pad_w = 0
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
        lowering["weight_pack"] = (
            "oihw_constant" if weight_name in ctx.graph.constants else "oihw_runtime"
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
