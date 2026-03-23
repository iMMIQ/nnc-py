"""Tests for intra-fast-memory compaction."""

import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType
from nnc_py.passes.liveness import LivenessAnalysisPass
from nnc_py.passes.memory_strategy import (
    MovePoint,
    MemoryAllocationPlan,
)
from nnc_py.passes.strategies.cost_aware_allocator import CostAwareAllocator


def test_move_point_fields():
    mp = MovePoint(
        tensor_name="a",
        at_node_idx=2,
        from_offset=80,
        to_offset=0,
        size=64,
    )
    assert mp.tensor_name == "a"
    assert mp.at_node_idx == 2
    assert mp.from_offset == 80
    assert mp.to_offset == 0
    assert mp.size == 64


def test_plan_move_fields_default_empty():
    plan = MemoryAllocationPlan(strategy_name="test", total_fast_memory=256)
    assert plan.move_points == []
    assert plan.move_bytes == 0


def test_plan_get_move_points_at():
    mp0 = MovePoint("a", at_node_idx=2, from_offset=80, to_offset=0, size=64)
    mp1 = MovePoint("b", at_node_idx=2, from_offset=160, to_offset=64, size=64)
    mp2 = MovePoint("c", at_node_idx=5, from_offset=32, to_offset=0, size=32)
    plan = MemoryAllocationPlan(
        strategy_name="test",
        total_fast_memory=256,
        move_points=[mp0, mp1, mp2],
        move_bytes=160,
    )
    assert plan.get_move_points_at(2) == [mp0, mp1]
    assert plan.get_move_points_at(5) == [mp2]
    assert plan.get_move_points_at(0) == []


def _tensor(name: str, elements: int) -> TensorType:
    return TensorType(
        name=name,
        dtype=DataType.FLOAT32,
        shape=TensorShape([1, elements]),
    )


def _make_ctx(
    *,
    tensor_elements: dict[str, int],
    inputs: list[str],
    outputs: list[str],
    nodes: list[Node],
) -> tuple[CompileContext, dict]:
    graph = Graph("compaction-test")
    graph.inputs.extend(inputs)
    graph.outputs.extend(outputs)
    for name, elements in tensor_elements.items():
        graph.add_tensor(_tensor(name, elements))
    for node in nodes:
        graph.add_node(node)
    ctx = CompileContext(graph=graph, target="x86", optimization_level=1)
    LivenessAnalysisPass().run(ctx)
    return ctx, ctx.metadata["tensor_liveness"]


def test_demand_precheck_rejects_insufficient_memory():
    """max_memory < max_node_demand -> clear ValueError upfront."""
    ctx, liveness = _make_ctx(
        tensor_elements={"x": 16, "a": 16},
        inputs=["x"],
        outputs=["a"],
        nodes=[Node(OpType.RELU, "n0", ["x"], ["a"])],
    )
    allocator = CostAwareAllocator()
    with pytest.raises(ValueError, match="peak node demand"):
        allocator.allocate(ctx, liveness, max_memory=64)


def test_demand_precheck_accepts_sufficient_memory():
    """max_memory == max_node_demand -> no error."""
    ctx, liveness = _make_ctx(
        tensor_elements={"x": 16, "y": 16, "a": 16},
        inputs=["x", "y"],
        outputs=["a"],
        nodes=[Node(OpType.ADD, "n0", ["x", "y"], ["a"])],
    )
    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=192)
    assert plan.total_fast_memory <= 192


def test_demand_precheck_dedupes_duplicate_inputs_per_node():
    """Repeated inputs on one node should count once for peak demand."""
    ctx, liveness = _make_ctx(
        tensor_elements={"x": 16, "y": 16},
        inputs=["x"],
        outputs=["y"],
        nodes=[Node(OpType.ADD, "n0", ["x", "x"], ["y"])],
    )
    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=128)
    assert plan.total_fast_memory <= 128
