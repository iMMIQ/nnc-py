"""Tests for scheduled-native SRAM and slow-memory planning."""

from __future__ import annotations

import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    PipelineScheduleProblem,
    PipelineScheduleResult,
    ResidencyWindow,
    ScheduleStep,
    ScheduledStep,
    ScheduledValue,
    ScheduledValueHomeTier,
    set_pipeline_schedule_problem,
)
from nnc_py.ir.pipeline_schedule import set_pipeline_schedule_result
from nnc_py.passes.scheduled_memory_expansion import ScheduledMemoryExpansionPass
from nnc_py.scheduler import ListPipelineScheduler
from tests.test_scheduled_memory_expansion import make_scheduled_spill_context

from nnc_py.passes.scheduled_memory_planning import ScheduledMemoryPlanningPass


def make_spilled_schedule_context():
    ctx = make_scheduled_spill_context(max_memory=64)
    ScheduledMemoryExpansionPass().run(ctx)
    result = ListPipelineScheduler().solve(ctx.pipeline_schedule_problem)
    set_pipeline_schedule_result(ctx, result)
    return ctx


def make_context_with_explicit_residencies(
    *,
    residency_windows: tuple[ResidencyWindow, ...],
) -> CompileContext:
    ctx = CompileContext(
        graph=Graph("scheduled_memory_planning"),
        target="x86",
        optimization_level=3,
    )
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="open0",
                node_name="open0",
                resource_kind=PipelineResourceKind.MATMUL,
                duration=1,
            ),
            ScheduleStep(
                id="close0",
                node_name="close0",
                resource_kind=PipelineResourceKind.OTHER,
                duration=1,
            ),
            ScheduleStep(
                id="open1",
                node_name="open1",
                resource_kind=PipelineResourceKind.MATMUL,
                duration=1,
            ),
            ScheduleStep(
                id="close1",
                node_name="close1",
                resource_kind=PipelineResourceKind.OTHER,
                duration=1,
            ),
        ),
        scheduled_values=(
            ScheduledValue(
                name="value0",
                graph_tensor_name="value0",
                size_bytes=32,
                producer_step_id=None,
                consumer_step_ids=(),
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
        ),
        residency_windows=residency_windows,
        resources=(
            PipelineResourceKind.MATMUL,
            PipelineResourceKind.OTHER,
        ),
        sram_capacity_bytes=256,
    )
    result = PipelineScheduleResult(
        scheduled_steps=(
            ScheduledStep(
                step_id="open0",
                resource_kind=PipelineResourceKind.MATMUL,
                start_time=0,
                end_time=2,
            ),
            ScheduledStep(
                step_id="close0",
                resource_kind=PipelineResourceKind.OTHER,
                start_time=2,
                end_time=4,
            ),
            ScheduledStep(
                step_id="open1",
                resource_kind=PipelineResourceKind.MATMUL,
                start_time=4,
                end_time=6,
            ),
            ScheduledStep(
                step_id="close1",
                resource_kind=PipelineResourceKind.OTHER,
                start_time=6,
                end_time=8,
            ),
        ),
        scheduled_values=problem.scheduled_values,
        residency_windows=residency_windows,
        feasible=True,
        makespan=8,
        solver_name="list",
    )
    set_pipeline_schedule_problem(ctx, problem)
    set_pipeline_schedule_result(ctx, result)
    return ctx


def test_scheduled_memory_planning_assigns_slow_offsets_to_spilled_values():
    ctx = make_spilled_schedule_context()

    ScheduledMemoryPlanningPass().run(ctx)

    plan = ctx.metadata["scheduled_memory_plan"]
    assert plan.slow_allocations["value0"].offset >= 0
    assert plan.transfer_points


def test_scheduled_memory_planning_does_not_write_legacy_memory_plan():
    ctx = make_spilled_schedule_context()

    ScheduledMemoryPlanningPass().run(ctx)

    assert "memory_plan" not in ctx.metadata
    assert "spill_plan" not in ctx.metadata


def test_scheduled_memory_planning_preserves_distinct_residency_allocations():
    ctx = make_context_with_explicit_residencies(
        residency_windows=(
            ResidencyWindow(
                value_name="value0",
                residency_id="value0@0",
                opened_by_step_id="open0",
                closed_by_step_id="close0",
            ),
            ResidencyWindow(
                value_name="value0",
                residency_id="value0@1",
                opened_by_step_id="open1",
                closed_by_step_id="close1",
            ),
        ),
    )

    ScheduledMemoryPlanningPass().run(ctx)

    plan = ctx.metadata["scheduled_memory_plan"]
    assert set(plan.fast_allocations) == {"value0@0", "value0@1"}
    assert plan.fast_allocations["value0@0"].residency_id == "value0@0"
    assert plan.fast_allocations["value0@1"].residency_id == "value0@1"
    assert plan.fast_allocations["value0@0"].value_name == "value0"
    assert plan.fast_allocations["value0@1"].value_name == "value0"


def test_scheduled_memory_planning_raises_on_malformed_feasible_residency_metadata():
    ctx = make_context_with_explicit_residencies(
        residency_windows=(
            ResidencyWindow(
                value_name="value0",
                residency_id="value0@broken",
                opened_by_step_id="missing_open",
                closed_by_step_id="close0",
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="residency"):
        ScheduledMemoryPlanningPass().run(ctx)
