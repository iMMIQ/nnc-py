import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType
from nnc_py.passes.liveness import LivenessAnalysisPass
from nnc_py.passes.memory_planning import MemoryPlanningPassV2
from nnc_py.passes.memory_strategy import AllocationStrategy
from nnc_py.passes.strategies.cost_aware_allocator import CostAwareAllocator


def _tensor(name: str, elements: int) -> TensorType:
    return TensorType(
        name=name,
        dtype=DataType.FLOAT32,
        shape=TensorShape([1, elements]),
    )


def _make_context(
    *,
    tensor_elements: dict[str, int],
    inputs: list[str],
    outputs: list[str],
    nodes: list[Node],
) -> tuple[CompileContext, dict]:
    graph = Graph("cost-aware-test")
    graph.inputs.extend(inputs)
    graph.outputs.extend(outputs)

    for tensor_name, elements in tensor_elements.items():
        graph.add_tensor(_tensor(tensor_name, elements))

    for node in nodes:
        graph.add_node(node)

    ctx = CompileContext(graph=graph, target="x86", optimization_level=1)
    LivenessAnalysisPass().run(ctx)
    return ctx, ctx.metadata["tensor_liveness"]


def _make_eviction_setup():
    ctx, liveness = _make_context(
        tensor_elements={
            "x_big": 16,
            "y_small": 16,
            "z_small": 16,
            "near_big": 64,
            "far_small": 16,
            "newcomer": 16,
            "near_done": 16,
            "far_done": 16,
            "out": 16,
        },
        inputs=["x_big", "y_small", "z_small"],
        outputs=["near_done", "out"],
        nodes=[
            Node(OpType.RELU, "n0", ["x_big"], ["near_big"]),
            Node(OpType.RELU, "n1", ["y_small"], ["far_small"]),
            Node(OpType.RELU, "n2", ["z_small"], ["newcomer"]),
            Node(OpType.RELU, "n3", ["near_big"], ["near_done"]),
            Node(OpType.RELU, "n4", ["far_small"], ["far_done"]),
            Node(OpType.ADD, "n5", ["newcomer", "far_done"], ["out"]),
        ],
    )
    return ctx, liveness


def _make_eviction_plan(max_memory: int = 384):
    ctx, liveness = _make_eviction_setup()

    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=max_memory)
    return ctx, allocator, plan


def _make_fragmentation_plan():
    ctx, liveness = _make_context(
        tensor_elements={
            "x": 5,
            "y": 5,
            "a": 5,
            "b": 5,
            "c": 5,
            "d": 5,
            "out": 5,
        },
        inputs=["x", "y"],
        outputs=["out"],
        nodes=[
            Node(OpType.RELU, "n0", ["x"], ["a"]),
            Node(OpType.RELU, "n1", ["y"], ["b"]),
            Node(OpType.RELU, "n2", ["a"], ["c"]),
            Node(OpType.RELU, "n3", ["c"], ["d"]),
            Node(OpType.ADD, "n4", ["b", "d"], ["out"]),
        ],
    )

    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=128)
    return ctx, allocator, plan


def _make_multi_reload_plan():
    ctx, liveness = _make_context(
        tensor_elements={
            "x1": 16,
            "x2": 16,
            "x3": 16,
            "x4": 16,
            "a": 16,
            "b": 16,
            "c": 16,
            "e": 16,
            "bridge": 1,
            "out": 1,
        },
        inputs=["x1", "x2", "x3", "x4"],
        outputs=["out"],
        nodes=[
            Node(OpType.RELU, "n0", ["x1"], ["a"]),
            Node(OpType.RELU, "n1", ["x2"], ["b"]),
            Node(OpType.RELU, "n2", ["x3"], ["c"]),
            Node(OpType.RELU, "n3", ["x4"], ["e"]),
            Node(OpType.ADD, "n4", ["c", "e"], ["bridge"]),
            Node(OpType.ADD, "n5", ["a", "b"], ["out"]),
        ],
    )

    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=192)
    return ctx, allocator, plan


def _make_single_chain_plan(*, elements: int, max_memory: int):
    ctx, liveness = _make_context(
        tensor_elements={
            "x": elements,
            "out": elements,
        },
        inputs=["x"],
        outputs=["out"],
        nodes=[
            Node(OpType.RELU, "n0", ["x"], ["out"]),
        ],
    )

    allocator = CostAwareAllocator()
    return allocator, ctx, liveness, max_memory


def _make_output_preservation_plan():
    ctx, liveness = _make_context(
        tensor_elements={
            "x0": 16,
            "x1": 16,
            "x2": 16,
            "early_out": 16,
            "mid": 16,
            "final_out": 16,
        },
        inputs=["x0", "x1", "x2"],
        outputs=["early_out", "final_out"],
        nodes=[
            Node(OpType.RELU, "n0", ["x0"], ["early_out"]),
            Node(OpType.RELU, "n1", ["x1"], ["mid"]),
            Node(OpType.ADD, "n2", ["mid", "x2"], ["final_out"]),
        ],
    )

    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=192)
    return ctx, allocator, plan


def _make_delayed_reload_plan():
    ctx, liveness = _make_context(
        tensor_elements={
            "x0": 1,
            "x1": 1,
            "x2": 1,
            "x3": 1,
            "delayed": 16,
            "noop": 1,
            "big_temp": 16,
            "out": 1,
        },
        inputs=["x0", "x1", "x2", "x3"],
        outputs=["out"],
        nodes=[
            Node(OpType.RELU, "n0", ["x0"], ["delayed"]),
            Node(OpType.RELU, "n1", ["x1"], ["noop"]),
            Node(OpType.RELU, "n2", ["x2"], ["big_temp"]),
            Node(OpType.ADD, "n3", ["delayed", "x3"], ["out"]),
        ],
    )

    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=96)
    return ctx, allocator, plan


def _make_output_reload_preservation_plan():
    ctx, liveness = _make_context(
        tensor_elements={
            "x0": 1,
            "x1": 1,
            "x2": 1,
            "x3": 1,
            "early": 16,
            "filler": 16,
            "mid": 1,
            "final_out": 16,
        },
        inputs=["x0", "x1", "x2", "x3"],
        outputs=["early", "final_out"],
        nodes=[
            Node(OpType.RELU, "n0", ["x0"], ["early"]),
            Node(OpType.RELU, "n1", ["x1"], ["filler"]),
            Node(OpType.ADD, "n2", ["early", "x2"], ["mid"]),
            Node(OpType.ADD, "n3", ["mid", "x3"], ["final_out"]),
        ],
    )

    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=96)
    return ctx, allocator, plan


def _make_transfer_minimization_plan():
    ctx, liveness = _make_context(
        tensor_elements={
            "x0": 1,
            "x1": 1,
            "x2": 1,
            "x3": 1,
            "kept_output": 24,
            "reused_later": 16,
            "big_temp": 24,
            "bridge": 1,
            "final_out": 1,
        },
        inputs=["x0", "x1", "x2", "x3"],
        outputs=["kept_output", "final_out"],
        nodes=[
            Node(OpType.RELU, "n0", ["x0"], ["kept_output"]),
            Node(OpType.RELU, "n1", ["x1"], ["reused_later"]),
            Node(OpType.RELU, "n2", ["x2"], ["big_temp"]),
            Node(OpType.RELU, "n3", ["x3"], ["bridge"]),
            Node(OpType.ADD, "n4", ["reused_later", "bridge"], ["final_out"]),
        ],
    )

    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=192)
    return ctx, allocator, plan


def _make_combined_eviction_plan():
    ctx, liveness = _make_context(
        tensor_elements={
            "x0": 1,
            "x1": 1,
            "x2": 1,
            "x3": 1,
            "kept_output": 24,
            "future_a": 8,
            "future_b": 8,
            "big_temp": 24,
            "final_out": 1,
        },
        inputs=["x0", "x1", "x2", "x3"],
        outputs=["kept_output", "final_out"],
        nodes=[
            Node(OpType.RELU, "n0", ["x0"], ["kept_output"]),
            Node(OpType.RELU, "n1", ["x1"], ["future_a"]),
            Node(OpType.RELU, "n2", ["x2"], ["future_b"]),
            Node(OpType.RELU, "n3", ["x3"], ["big_temp"]),
            Node(OpType.ADD, "n4", ["future_a", "future_b"], ["final_out"]),
        ],
    )

    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=256)
    return ctx, allocator, plan


def test_cost_aware_allocator_reuses_aligned_free_regions():
    allocator = CostAwareAllocator()
    free_regions = [(0, 128)]

    offset = allocator._allocate_from_free_list(free_regions, 64, 16)

    assert offset == 0
    assert free_regions == [(64, 64)]

    allocator._free_fast_region(free_regions, 0, 64)

    assert free_regions == [(0, 128)]


def test_cost_aware_allocator_evicts_farther_smaller_tensor_first():
    _, allocator, plan = _make_eviction_plan()

    assert allocator.name == "cost_aware"
    assert allocator.strategy_type == AllocationStrategy.COST_AWARE
    assert plan.strategy_name == "cost_aware"
    assert plan.spill_points
    assert plan.spill_points[0].tensor_name == "far_small"
    assert any(
        reload.tensor_name == "far_small" and reload.before_node == "n4"
        for reload in plan.reload_points
    )


def test_cost_aware_allocator_prefers_zero_transfer_frees_over_spills():
    allocator = CostAwareAllocator()
    candidates = [
        {
            "tensor_name": "kept_output",
            "size": 64,
            "next_use_distance": None,
            "transfer_cost": 64,
        },
        {
            "tensor_name": "dead_temp",
            "size": 64,
            "next_use_distance": None,
            "transfer_cost": 0,
        },
    ]

    ordered = allocator._sort_eviction_candidates(candidates)

    assert ordered[0]["tensor_name"] == "dead_temp"


def test_cost_aware_allocator_tracks_transfer_bytes_on_spill_and_reload():
    _, _, plan = _make_eviction_plan()

    assert plan.spill_bytes == sum(point.size for point in plan.spill_points)
    assert plan.reload_bytes == sum(point.size for point in plan.reload_points)
    assert plan.total_transfer_bytes == plan.spill_bytes + plan.reload_bytes
    assert plan.spill_bytes > 0
    assert plan.reload_bytes > 0


def test_cost_aware_allocator_prefers_single_spill_over_spill_and_reload():
    _, _, plan = _make_transfer_minimization_plan()

    assert [point.tensor_name for point in plan.spill_points] == ["kept_output"]
    assert not plan.reload_points
    assert plan.spill_bytes == 96
    assert plan.reload_bytes == 0
    assert not plan.get_allocation("reused_later").is_spilled


def test_cost_aware_allocator_chooses_min_transfer_eviction_set():
    _, _, plan = _make_combined_eviction_plan()

    assert [point.tensor_name for point in plan.spill_points] == ["kept_output"]
    assert not plan.reload_points
    assert plan.spill_bytes == 96
    assert plan.reload_bytes == 0
    assert not plan.get_allocation("future_a").is_spilled
    assert not plan.get_allocation("future_b").is_spilled


def test_cost_aware_allocator_assigns_unique_reload_slots_per_node():
    ctx, _, plan = _make_multi_reload_plan()
    node_idx = next(
        idx for idx, node in enumerate(ctx.graph.topological_sort()) if node.name == "n4"
    )

    reloads = plan.get_reload_points_before(node_idx)

    assert {reload.tensor_name for reload in reloads} == {"c", "e"}
    assert len(reloads) == 2
    assert {reload.reload_slot_id for reload in reloads} == {0, 1}


def test_cost_aware_allocator_spills_before_reusing_output_slot():
    _, _, plan = _make_eviction_plan()

    spill = next(point for point in plan.spill_points if point.tensor_name == "far_small")
    newcomer = plan.get_allocation("newcomer")

    assert newcomer is not None
    assert spill.after_node == "n1"
    assert spill.after_node_idx == 1
    assert newcomer.offset == spill.from_fast_offset


def test_cost_aware_allocator_spills_early_graph_outputs_before_reuse():
    _, _, plan = _make_output_preservation_plan()

    early_out = plan.get_allocation("early_out")
    final_out = plan.get_allocation("final_out")
    spill = next(point for point in plan.spill_points if point.tensor_name == "early_out")

    assert early_out is not None
    assert final_out is not None
    assert early_out.is_spilled
    assert early_out.buffer_id == -1
    assert early_out.reload_before is None
    assert spill.after_node == "n1"
    assert spill.after_node_idx == 1
    assert final_out.offset == spill.from_fast_offset


def test_cost_aware_allocator_keeps_delayed_spills_fast_backed_until_spill():
    _, _, plan = _make_delayed_reload_plan()

    delayed = plan.get_allocation("delayed")
    spill = next(point for point in plan.spill_points if point.tensor_name == "delayed")
    reload = next(point for point in plan.reload_points if point.tensor_name == "delayed")

    assert delayed is not None
    assert not delayed.is_spilled
    assert delayed.buffer_id == 0
    assert delayed.spill_after == 1
    assert delayed.reload_before == [3]
    assert delayed.offset == spill.from_fast_offset
    assert spill.after_node_idx == 1
    assert reload.before_node_idx == 3


def test_cost_aware_allocator_keeps_reloadable_immediate_spills_fast_backed():
    _, _, plan = _make_eviction_plan()

    far_small = plan.get_allocation("far_small")
    spill = next(point for point in plan.spill_points if point.tensor_name == "far_small")

    assert far_small is not None
    assert not far_small.is_spilled
    assert far_small.buffer_id == 0
    assert far_small.spill_after == 1
    assert far_small.reload_before == [4]
    assert far_small.offset == spill.from_fast_offset


def test_cost_aware_allocator_preserves_outputs_after_internal_reload():
    _, _, plan = _make_output_reload_preservation_plan()

    early = plan.get_allocation("early")
    early_spills = [point for point in plan.spill_points if point.tensor_name == "early"]

    assert early is not None
    assert early.is_spilled
    assert early.buffer_id == -1
    assert plan.get_buffer_for_tensor("early") == -1
    assert "early" not in plan.buffers[0].tensors
    assert early.spill_after == 2
    assert early.reload_before == [2]
    assert len(early_spills) == 2
    assert early.offset == early_spills[-1].to_slow_offset


def test_cost_aware_allocator_reports_fragmented_fast_memory_footprint():
    _, _, plan = _make_fragmentation_plan()

    a_alloc = plan.get_allocation("a")
    b_alloc = plan.get_allocation("b")

    assert a_alloc is not None
    assert b_alloc is not None
    assert plan.node_memory_usage[0] == a_alloc.offset + a_alloc.size
    assert plan.node_memory_usage[1] == b_alloc.offset + b_alloc.size
    assert plan.peak_memory == max(plan.node_memory_usage)
    assert plan.total_fast_memory >= plan.peak_memory


def test_cost_aware_allocator_rejects_budget_that_only_fails_after_alignment():
    allocator, ctx, liveness, max_memory = _make_single_chain_plan(elements=1, max_memory=8)

    with pytest.raises(ValueError, match="aligned fast-memory pool"):
        allocator.allocate(ctx, liveness, max_memory=max_memory)


def test_cost_aware_allocator_reclaims_bump_alignment_padding_for_reuse():
    ctx, liveness = _make_context(
        tensor_elements={
            "x": 1,
            "y": 1,
            "a": 1,
            "b": 1,
            "out": 2,
        },
        inputs=["x", "y"],
        outputs=["out"],
        nodes=[
            Node(OpType.RELU, "n0", ["x"], ["a"]),
            Node(OpType.RELU, "n1", ["y"], ["b"]),
            Node(OpType.ADD, "n2", ["a", "b"], ["out"]),
        ],
    )

    plan = CostAwareAllocator().allocate(ctx, liveness, max_memory=48)

    out = plan.get_allocation("out")

    assert out is not None
    assert out.offset == 0
    assert not plan.spill_points
    assert not plan.reload_points


def test_cost_aware_allocator_explicitly_rejects_pre_graph_spills():
    allocator = CostAwareAllocator()

    with pytest.raises(ValueError, match="early enough before node n0"):
        allocator._validate_spill_after_idx("x", "n0", -1)


def test_cost_aware_allocator_limits_large_eviction_search():
    allocator = CostAwareAllocator()
    checked_states = 0
    original = allocator._can_allocate_with_state

    def counted_can_allocate_with_state(**kwargs):
        nonlocal checked_states
        checked_states += 1
        return original(**kwargs)

    allocator._can_allocate_with_state = counted_can_allocate_with_state  # type: ignore[method-assign]

    candidates = [
        {
            "tensor_name": f"c{i}",
            "offset": i * 16,
            "size": 16,
            "next_use_distance": 1,
            "transfer_cost": 32,
        }
        for i in range(14)
    ]

    selected = allocator._select_eviction_candidates(
        size=64,
        candidates=candidates,
        free_regions=[],
        next_fast_offset=224,
        max_memory=224,
    )

    assert selected is not None
    assert [candidate["tensor_name"] for candidate in selected] == ["c0", "c1", "c2", "c3"]
    assert checked_states < 600


def test_cost_aware_allocator_large_search_keeps_non_prefix_min_transfer_choice():
    allocator = CostAwareAllocator()
    candidates = [
        {
            "tensor_name": f"small{i}",
            "offset": i * 16,
            "size": 16,
            "next_use_distance": 1,
            "transfer_cost": 40,
        }
        for i in range(10)
    ] + [
        {
            "tensor_name": f"big{i}",
            "offset": 160 + (i * 32),
            "size": 32,
            "next_use_distance": 1,
            "transfer_cost": 60,
        }
        for i in range(3)
    ]

    selected = allocator._select_eviction_candidates(
        size=96,
        candidates=candidates,
        free_regions=[],
        next_fast_offset=256,
        max_memory=256,
    )

    assert selected is not None
    assert [candidate["tensor_name"] for candidate in selected] == ["big0", "big1", "big2"]


def test_cost_aware_allocator_marks_spilled_tensors_as_slow_backed_in_plan():
    _, _, plan = _make_output_preservation_plan()

    spilled = plan.get_allocation("early_out")

    assert spilled is not None
    assert spilled.is_spilled
    assert spilled.buffer_id == -1
    assert plan.get_buffer_for_tensor("early_out") == -1
    assert "early_out" not in plan.buffers[0].tensors


def test_cost_aware_allocator_pass_compatibility_skips_spilled_tensors_in_legacy_plan():
    ctx, _ = _make_context(
        tensor_elements={
            "x0": 16,
            "x1": 16,
            "x2": 16,
            "early_out": 16,
            "mid": 16,
            "final_out": 16,
        },
        inputs=["x0", "x1", "x2"],
        outputs=["early_out", "final_out"],
        nodes=[
            Node(OpType.RELU, "n0", ["x0"], ["early_out"]),
            Node(OpType.RELU, "n1", ["x1"], ["mid"]),
            Node(OpType.ADD, "n2", ["mid", "x2"], ["final_out"]),
        ],
    )
    ctx.metadata["memory_strategy"] = "cost_aware"
    ctx.metadata["max_memory"] = 192

    MemoryPlanningPassV2().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    legacy_plan = ctx.metadata["memory_plan"]

    assert "early_out" in plan.spilled_tensors
    assert plan.get_allocation("early_out").buffer_id == -1
    assert "early_out" not in legacy_plan.tensor_info
