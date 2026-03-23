"""Tests for intra-fast-memory compaction."""

from types import MethodType

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


def test_compaction_triggered_on_fragmentation():
    """When alignment fragmentation blocks allocation, compaction fires and succeeds."""
    ctx, liveness = _make_ctx(
        tensor_elements={
            "x": 3,
            "y": 3,
            "a": 5,
            "b": 5,
            "out": 5,
        },
        inputs=["x", "y"],
        outputs=["out"],
        nodes=[
            Node(OpType.MATMUL, "n0", ["x"], ["a"]),
            Node(OpType.MATMUL, "n1", ["y"], ["b"]),
            Node(OpType.MATMUL, "n2", ["a", "b"], ["out"]),
        ],
    )
    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=96)
    assert plan.total_fast_memory <= 96
    assert plan.move_points
    assert plan.move_bytes == sum(mp.size for mp in plan.move_points)


def test_compaction_not_triggered_when_unnecessary():
    """Normal allocation without fragmentation produces no MovePoints."""
    ctx, liveness = _make_ctx(
        tensor_elements={"x": 16, "a": 16},
        inputs=["x"],
        outputs=["a"],
        nodes=[Node(OpType.RELU, "n0", ["x"], ["a"])],
    )
    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=256)
    assert plan.move_points == []
    assert plan.move_bytes == 0


def test_compaction_move_bytes_accurate():
    """move_bytes equals sum of moved tensor sizes."""
    ctx, liveness = _make_ctx(
        tensor_elements={
            "x": 5,
            "y": 5,
            "a": 5,
            "b": 5,
            "c": 5,
            "out": 5,
        },
        inputs=["x", "y"],
        outputs=["out"],
        nodes=[
            Node(OpType.MATMUL, "n0", ["x"], ["a"]),
            Node(OpType.MATMUL, "n1", ["y"], ["b"]),
            Node(OpType.MATMUL, "n2", ["a"], ["c"]),
            Node(OpType.MATMUL, "n3", ["b", "c"], ["out"]),
        ],
    )
    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=96)
    assert plan.move_bytes == sum(mp.size for mp in plan.move_points)


def test_compaction_post_compaction_offsets_contiguous():
    """After compaction, protected tensors are packed contiguously from offset 0."""
    ctx, liveness = _make_ctx(
        tensor_elements={
            "x": 3,
            "y": 3,
            "a": 5,
            "b": 5,
            "out": 5,
        },
        inputs=["x", "y"],
        outputs=["out"],
        nodes=[
            Node(OpType.MATMUL, "n0", ["x"], ["a"]),
            Node(OpType.MATMUL, "n1", ["y"], ["b"]),
            Node(OpType.MATMUL, "n2", ["a", "b"], ["out"]),
        ],
    )
    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=96)

    assert plan.move_points

    sorted_moves = sorted(plan.move_points, key=lambda mp: mp.to_offset)
    assert sorted_moves[0].to_offset == 0
    for i in range(1, len(sorted_moves)):
        prev = sorted_moves[i - 1]
        expected_offset = allocator._align(
            prev.to_offset + prev.size,
            allocator.DEFAULT_ALIGNMENT,
        )
        assert sorted_moves[i].to_offset == expected_offset


def test_compaction_refuses_when_non_protected_residents_remain():
    """Compaction must refuse mixed resident sets instead of creating overlaps."""
    ctx, liveness = _make_ctx(
        tensor_elements={
            "x": 1,
            "y": 1,
            "z": 1,
            "a": 4,
            "b": 4,
            "c": 5,
            "d": 5,
            "out": 6,
        },
        inputs=["x", "y", "z"],
        outputs=["out"],
        nodes=[
            Node(OpType.MATMUL, "n0", ["x"], ["a"]),
            Node(OpType.MATMUL, "n1", ["y"], ["b"]),
            Node(OpType.MATMUL, "n2", ["z"], ["c"]),
            Node(OpType.ADD, "n3", ["a", "b"], ["d"]),
            Node(OpType.ADD, "n4", ["c", "d"], ["out"]),
        ],
    )
    allocator = CostAwareAllocator()

    def force_no_eviction_plan(self, **kwargs):
        return None

    allocator._select_eviction_candidates = MethodType(force_no_eviction_plan, allocator)

    with pytest.raises(ValueError, match="Cannot allocate"):
        allocator.allocate(ctx, liveness, max_memory=96)


def test_compaction_bytes_do_not_leak_into_slow_transfer_totals():
    """Compaction bytes stay in move_bytes; slow-transfer totals track only spills/reloads."""
    ctx, liveness = _make_ctx(
        tensor_elements={
            "x": 3,
            "y": 3,
            "a": 5,
            "b": 5,
            "out": 5,
        },
        inputs=["x", "y"],
        outputs=["out"],
        nodes=[
            Node(OpType.MATMUL, "n0", ["x"], ["a"]),
            Node(OpType.MATMUL, "n1", ["y"], ["b"]),
            Node(OpType.MATMUL, "n2", ["a", "b"], ["out"]),
        ],
    )
    allocator = CostAwareAllocator()
    plan_tight = allocator.allocate(ctx, liveness, max_memory=96)
    plan_generous = CostAwareAllocator().allocate(ctx, liveness, max_memory=1024)
    assert plan_tight.move_points
    assert plan_tight.move_bytes == sum(mp.size for mp in plan_tight.move_points)
    assert plan_tight.total_transfer_bytes == (
        plan_tight.spill_bytes + plan_tight.reload_bytes
    ) == 0
    assert plan_generous.move_points == []
    assert plan_generous.move_bytes == 0
    assert plan_generous.total_transfer_bytes == (
        plan_generous.spill_bytes + plan_generous.reload_bytes
    ) == 0


def test_compaction_extreme_max_memory_equals_node_demand():
    """Allocation succeeds when max_memory == peak node demand exactly."""
    ctx, liveness = _make_ctx(
        tensor_elements={
            "x0": 4,
            "x1": 4,
            "x2": 4,
            "a": 4,
            "b": 4,
            "out": 4,
        },
        inputs=["x0", "x1", "x2"],
        outputs=["out"],
        nodes=[
            Node(OpType.RELU, "n0", ["x0"], ["a"]),
            Node(OpType.RELU, "n1", ["x1"], ["b"]),
            Node(OpType.ADD, "n2", ["a", "b"], ["out"]),
        ],
    )
    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=48)
    assert plan.total_fast_memory <= 48
