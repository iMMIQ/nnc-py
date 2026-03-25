"""Tests for time-aware SRAM memory planning in MemoryPlanningPassV4."""

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    PipelineScheduleProblem,
    PipelineScheduleResult,
    ScheduledStep,
    ScheduleStep,
    SramAllocationInterval,
    SramValue,
)
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType
from nnc_py.passes.memory_planning_v4 import MemoryPlanningPassV4


def make_context(
    *,
    problem: PipelineScheduleProblem | None = None,
    result: PipelineScheduleResult | None = None,
) -> CompileContext:
    ctx = CompileContext(graph=Graph("memory_v4"), target="x86", optimization_level=3)
    if problem is not None:
        ctx.metadata["pipeline_schedule_problem"] = problem
    if result is not None:
        ctx.metadata["pipeline_schedule_result"] = result
    return ctx


def make_problem(*values: SramValue, steps: tuple[ScheduleStep, ...] = ()) -> PipelineScheduleProblem:
    return PipelineScheduleProblem(
        steps=steps,
        sram_values=values,
        resources=(PipelineResourceKind.MATMUL, PipelineResourceKind.OTHER),
        sram_capacity_bytes=1024,
    )


def make_result(
    *scheduled_steps: ScheduledStep,
    feasible: bool = True,
    makespan: int = 0,
    sram_intervals: tuple[SramAllocationInterval, ...] = (),
) -> PipelineScheduleResult:
    return PipelineScheduleResult(
        scheduled_steps=scheduled_steps,
        sram_intervals=sram_intervals,
        feasible=feasible,
        makespan=makespan,
        solver_name="list",
    )


def test_memory_planning_v4_derives_non_overlapping_intervals_from_scheduled_timing():
    problem = make_problem(
        SramValue(
            name="value_a",
            size_bytes=64,
            producer_step_id="producer_a",
            consumer_step_ids=("consumer_a",),
        ),
        SramValue(
            name="value_b",
            size_bytes=32,
            producer_step_id="producer_b",
            consumer_step_ids=("consumer_b",),
        ),
    )
    result = make_result(
        ScheduledStep(
            step_id="producer_a",
            resource_kind=PipelineResourceKind.MATMUL,
            start_time=0,
            end_time=2,
        ),
        ScheduledStep(
            step_id="consumer_a",
            resource_kind=PipelineResourceKind.OTHER,
            start_time=2,
            end_time=4,
        ),
        ScheduledStep(
            step_id="producer_b",
            resource_kind=PipelineResourceKind.MATMUL,
            start_time=4,
            end_time=6,
        ),
        ScheduledStep(
            step_id="consumer_b",
            resource_kind=PipelineResourceKind.OTHER,
            start_time=6,
            end_time=9,
        ),
        makespan=9,
        sram_intervals=(
            SramAllocationInterval(
                value_name="value_a",
                buffer_id="precomputed0",
                start_time=0,
                end_time=99,
                size_bytes=64,
            ),
            SramAllocationInterval(
                value_name="value_b",
                buffer_id="precomputed1",
                start_time=0,
                end_time=99,
                size_bytes=32,
            ),
        ),
    )
    ctx = make_context(problem=problem, result=result)

    MemoryPlanningPassV4().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.num_buffers == 1
    assert plan.get_buffer_for_tensor("value_a") == 0
    assert plan.get_buffer_for_tensor("value_b") == 0
    assert plan.buffers[0].tensors == ["value_a", "value_b"]
    assert plan.total_fast_memory == 64


def test_memory_planning_v4_emits_expected_strategy_and_logical_allocations():
    problem = make_problem(
        SramValue(
            name="value_out",
            size_bytes=48,
            producer_step_id="compute0",
            consumer_step_ids=("store0",),
        )
    )
    result = make_result(
        ScheduledStep(
            step_id="compute0",
            resource_kind=PipelineResourceKind.MATMUL,
            start_time=1,
            end_time=3,
        ),
        ScheduledStep(
            step_id="store0",
            resource_kind=PipelineResourceKind.OTHER,
            start_time=3,
            end_time=12,
        ),
        makespan=12,
    )
    ctx = make_context(problem=problem, result=result)

    MemoryPlanningPassV4().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.strategy_name == "schedule_time_v4"
    assert plan.buffers
    assert plan.tensor_allocations["value_out"].size == 48
    assert plan.tensor_allocations["value_out"].offset == plan.buffers[0].offset
    assert plan.logical_regions["value_out"].size_bytes == 48
    assert plan.logical_regions["value_out"].offset == plan.buffers[0].offset


def test_memory_planning_v4_falls_back_to_empty_plan_when_schedule_metadata_is_missing_or_unusable():
    missing_ctx = make_context()
    missing_problem_ctx = make_context(
        result=make_result(
            ScheduledStep(
                step_id="producer0",
                resource_kind=PipelineResourceKind.MATMUL,
                start_time=0,
                end_time=1,
            ),
            feasible=True,
            makespan=1,
        )
    )
    invalid_reference_ctx = make_context(
        problem=make_problem(
            SramValue(
                name="broken",
                size_bytes=32,
                producer_step_id="missing_producer",
                consumer_step_ids=("consumer0",),
            )
        ),
        result=make_result(
            ScheduledStep(
                step_id="consumer0",
                resource_kind=PipelineResourceKind.OTHER,
                start_time=1,
                end_time=4,
            ),
            feasible=True,
            makespan=4,
        ),
    )

    for ctx in (missing_ctx, missing_problem_ctx, invalid_reference_ctx):
        MemoryPlanningPassV4().run(ctx)

        plan = ctx.metadata["memory_allocation_plan"]
        assert plan.strategy_name == "schedule_time_v4"
        assert plan.buffers == []
        assert plan.tensor_allocations == {}
        assert plan.logical_regions == {}
        assert plan.total_fast_memory == 0
        assert plan.num_buffers == 0


def test_memory_planning_v4_uses_graph_tensor_size_for_external_zero_sized_values():
    graph = Graph("memory_v4_external")
    graph.add_tensor(
        TensorType(
            name="external_in",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 8]),
        )
    )
    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    ctx.metadata["pipeline_schedule_problem"] = make_problem(
        SramValue(
            name="external_in",
            size_bytes=0,
            producer_step_id=None,
            consumer_step_ids=("consumer0",),
        ),
        steps=(
            ScheduleStep(
                id="consumer0",
                node_name="consumer0",
                resource_kind=PipelineResourceKind.OTHER,
                duration=1,
                sram_input_names=("external_in",),
            ),
        ),
    )
    ctx.metadata["pipeline_schedule_result"] = make_result(
        ScheduledStep(
            step_id="consumer0",
            resource_kind=PipelineResourceKind.OTHER,
            start_time=0,
            end_time=2,
        ),
        feasible=True,
        makespan=2,
    )

    MemoryPlanningPassV4().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.total_fast_memory == 32
    assert plan.tensor_allocations["external_in"].size == 32


def test_memory_planning_v4_changes_buffer_count_when_derived_interval_timing_changes():
    problem = make_problem(
        SramValue(
            name="value_a",
            size_bytes=32,
            producer_step_id="producer_a",
            consumer_step_ids=("consumer_a",),
        ),
        SramValue(
            name="value_b",
            size_bytes=32,
            producer_step_id="producer_b",
            consumer_step_ids=("consumer_b",),
        ),
    )
    reusable_ctx = make_context(
        problem=problem,
        result=make_result(
            ScheduledStep(
                step_id="producer_a",
                resource_kind=PipelineResourceKind.MATMUL,
                start_time=0,
                end_time=2,
            ),
            ScheduledStep(
                step_id="consumer_a",
                resource_kind=PipelineResourceKind.OTHER,
                start_time=2,
                end_time=4,
            ),
            ScheduledStep(
                step_id="producer_b",
                resource_kind=PipelineResourceKind.MATMUL,
                start_time=4,
                end_time=5,
            ),
            ScheduledStep(
                step_id="consumer_b",
                resource_kind=PipelineResourceKind.OTHER,
                start_time=5,
                end_time=8,
            ),
            feasible=True,
            makespan=8,
        ),
    )
    overlapping_ctx = make_context(
        problem=problem,
        result=make_result(
            ScheduledStep(
                step_id="producer_a",
                resource_kind=PipelineResourceKind.MATMUL,
                start_time=0,
                end_time=2,
            ),
            ScheduledStep(
                step_id="consumer_a",
                resource_kind=PipelineResourceKind.OTHER,
                start_time=2,
                end_time=6,
            ),
            ScheduledStep(
                step_id="producer_b",
                resource_kind=PipelineResourceKind.MATMUL,
                start_time=4,
                end_time=5,
            ),
            ScheduledStep(
                step_id="consumer_b",
                resource_kind=PipelineResourceKind.OTHER,
                start_time=5,
                end_time=8,
            ),
            feasible=True,
            makespan=8,
        ),
    )

    MemoryPlanningPassV4().run(reusable_ctx)
    MemoryPlanningPassV4().run(overlapping_ctx)

    reusable_plan = reusable_ctx.metadata["memory_allocation_plan"]
    overlapping_plan = overlapping_ctx.metadata["memory_allocation_plan"]
    assert reusable_plan.num_buffers == 1
    assert overlapping_plan.num_buffers == 2
    assert overlapping_plan.tensor_allocations["value_a"].offset == overlapping_plan.buffers[0].offset
    assert overlapping_plan.tensor_allocations["value_b"].offset == overlapping_plan.buffers[1].offset
    assert overlapping_plan.tensor_allocations["value_b"].offset != overlapping_plan.tensor_allocations["value_a"].offset
    assert (
        overlapping_plan.logical_regions["value_b"].offset
        == overlapping_plan.tensor_allocations["value_b"].offset
    )


def test_memory_planning_v4_extends_must_reside_value_to_makespan():
    problem = make_problem(
        SramValue(
            name="resident_value",
            size_bytes=24,
            producer_step_id="producer0",
            must_reside_in_sram=True,
        )
    )
    result = make_result(
        ScheduledStep(
            step_id="producer0",
            resource_kind=PipelineResourceKind.MATMUL,
            start_time=1,
            end_time=4,
        ),
        feasible=True,
        makespan=11,
    )
    ctx = make_context(problem=problem, result=result)

    MemoryPlanningPassV4().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.num_buffers == 1
    assert plan.total_fast_memory == 32
    assert plan.tensor_allocations["resident_value"].offset == plan.buffers[0].offset
    assert plan.logical_regions["resident_value"].offset == plan.buffers[0].offset
