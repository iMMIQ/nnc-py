"""Schedule-candidate analysis for tiled lowering."""

from __future__ import annotations

from dataclasses import dataclass

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.passes.base import PassBase

FAST_MEMORY_BUDGET_BYTES = 1 * 1024 * 1024


@dataclass(frozen=True)
class ScheduleCandidate:
    """Phase-1 scheduling signal for a single node."""

    node_name: str
    op_family: str
    estimated_working_set_bytes: int | None
    must_tile: bool
    reason: str


class ScheduleAnalysisPass(PassBase):
    """Mark obvious tiled-execution candidates from current tensor sizes."""

    _SUPPORTED_OP_FAMILIES = {
        OpType.CONV2D: "conv2d",
        OpType.FUSED_CONV_RELU: "conv2d",
        OpType.FUSED_CONV_BIAS_RELU: "conv2d",
        OpType.FUSED_CONV_SIGMOID: "conv2d",
        OpType.MAXPOOL: "maxpool",
        OpType.AVGPOOL: "average_pool",
        OpType.GLOBAL_AVGPOOL: "global_average_pool",
        OpType.GEMM: "gemm",
        OpType.MATMUL: "matmul",
    }

    @property
    def name(self) -> str:
        return "ScheduleAnalysis"

    def _execute(self, ctx: CompileContext) -> None:
        ctx.metadata["schedule_candidates"] = analyze_nodes(ctx.graph)


def analyze_nodes(graph: Graph) -> dict[str, ScheduleCandidate]:
    """Return schedule candidates for supported phase-1 operators."""

    candidates: dict[str, ScheduleCandidate] = {}
    for node in graph.topological_sort():
        op_family = _op_family_for(node)
        if op_family is None:
            continue

        working_set_bytes = _estimate_working_set_bytes(graph, node)
        if working_set_bytes is None:
            candidates[node.name] = ScheduleCandidate(
                node_name=node.name,
                op_family=op_family,
                estimated_working_set_bytes=None,
                must_tile=False,
                reason="unknown_working_set",
            )
            continue

        must_tile = working_set_bytes > FAST_MEMORY_BUDGET_BYTES
        candidates[node.name] = ScheduleCandidate(
            node_name=node.name,
            op_family=op_family,
            estimated_working_set_bytes=working_set_bytes,
            must_tile=must_tile,
            reason="peak_working_set" if must_tile else "fits_working_set",
        )

    return candidates


def _op_family_for(node: Node) -> str | None:
    return ScheduleAnalysisPass._SUPPORTED_OP_FAMILIES.get(node.op_type)


def _estimate_working_set_bytes(graph: Graph, node: Node) -> int | None:
    total = 0
    for tensor_name in [*node.inputs, *node.outputs]:
        tensor = graph.tensors.get(tensor_name)
        if tensor is None:
            continue
        tensor_size = tensor.byte_size()
        if tensor_size < 0:
            return None
        total += tensor_size
    return total
