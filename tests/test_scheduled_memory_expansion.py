"""Tests for scheduled SRAM expansion with explicit DMA spill/reload steps."""

from __future__ import annotations

import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    PipelineScheduleProblem,
    ResidencyWindow,
    ScheduleDependencyKind,
    ScheduleEdge,
    ScheduleStep,
    ScheduleStepKind,
    ScheduledValue,
    ScheduledValueHomeTier,
    TransferStep,
    TransferStepKind,
)
from nnc_py.passes.scheduled_memory_expansion import ScheduledMemoryExpansionPass
from nnc_py.scheduler import ListPipelineScheduler


def make_scheduled_spill_context(*, max_memory: int) -> CompileContext:
    ctx = CompileContext(
        graph=Graph("scheduled_spill"),
        target="x86",
        optimization_level=3,
    )
    ctx.metadata["max_memory"] = max_memory
    ctx.metadata["pipeline_schedule_problem"] = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="produce0",
                node_name="produce0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.MATMUL,
                duration=2,
                sram_output_names=("value0",),
            ),
            ScheduleStep(
                id="produce1",
                node_name="produce1",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=2,
                sram_output_names=("value1",),
            ),
            ScheduleStep(
                id="consume1",
                node_name="consume1",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=2,
                sram_input_names=("value1",),
            ),
            ScheduleStep(
                id="consume0",
                node_name="consume0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.MATMUL,
                duration=2,
                sram_input_names=("value0",),
            ),
        ),
        edges=(
            ScheduleEdge("produce0", "produce1", ScheduleDependencyKind.ORDER),
            ScheduleEdge("produce1", "consume1", ScheduleDependencyKind.DATA),
            ScheduleEdge("produce0", "consume0", ScheduleDependencyKind.DATA),
            ScheduleEdge("consume1", "consume0", ScheduleDependencyKind.ORDER),
        ),
        scheduled_values=(
            ScheduledValue(
                name="value0",
                graph_tensor_name="value0",
                size_bytes=40,
                producer_step_id="produce0",
                consumer_step_ids=("consume0",),
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
            ScheduledValue(
                name="value1",
                graph_tensor_name="value1",
                size_bytes=40,
                producer_step_id="produce1",
                consumer_step_ids=("consume1",),
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
        ),
        resources=(
            PipelineResourceKind.DMA,
            PipelineResourceKind.MATMUL,
            PipelineResourceKind.OTHER,
        ),
        sram_capacity_bytes=256,
    )
    return ctx


def make_must_reside_context(*, max_memory: int) -> CompileContext:
    ctx = CompileContext(
        graph=Graph("must_reside"),
        target="x86",
        optimization_level=3,
    )
    ctx.metadata["max_memory"] = max_memory
    ctx.metadata["pipeline_schedule_problem"] = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="produce0",
                node_name="produce0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.MATMUL,
                duration=1,
                sram_output_names=("value0",),
            ),
            ScheduleStep(
                id="consume0",
                node_name="consume0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=1,
                sram_input_names=("value0",),
            ),
        ),
        edges=(
            ScheduleEdge("produce0", "consume0", ScheduleDependencyKind.DATA),
        ),
        scheduled_values=(
            ScheduledValue(
                name="value0",
                graph_tensor_name="value0",
                size_bytes=40,
                producer_step_id="produce0",
                consumer_step_ids=("consume0",),
                must_reside_in_sram=True,
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
        ),
        resources=(
            PipelineResourceKind.DMA,
            PipelineResourceKind.MATMUL,
            PipelineResourceKind.OTHER,
        ),
        sram_capacity_bytes=256,
    )
    return ctx


def make_repeated_consumer_context(*, max_memory: int) -> CompileContext:
    ctx = CompileContext(
        graph=Graph("repeated_consumer"),
        target="x86",
        optimization_level=3,
    )
    ctx.metadata["max_memory"] = max_memory
    ctx.metadata["pipeline_schedule_problem"] = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="produce0",
                node_name="produce0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.MATMUL,
                duration=2,
                sram_output_names=("value0",),
            ),
            ScheduleStep(
                id="produce1",
                node_name="produce1",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=2,
                sram_output_names=("value1",),
            ),
            ScheduleStep(
                id="consume1",
                node_name="consume1",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=2,
                sram_input_names=("value1",),
            ),
            ScheduleStep(
                id="consume0",
                node_name="consume0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.MATMUL,
                duration=2,
                sram_input_names=("value0", "value0"),
            ),
        ),
        edges=(
            ScheduleEdge("produce0", "produce1", ScheduleDependencyKind.ORDER),
            ScheduleEdge("produce1", "consume1", ScheduleDependencyKind.DATA),
            ScheduleEdge("produce0", "consume0", ScheduleDependencyKind.DATA),
            ScheduleEdge("consume1", "consume0", ScheduleDependencyKind.ORDER),
        ),
        scheduled_values=(
            ScheduledValue(
                name="value0",
                graph_tensor_name="value0",
                size_bytes=40,
                producer_step_id="produce0",
                consumer_step_ids=("consume0", "consume0"),
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
            ScheduledValue(
                name="value1",
                graph_tensor_name="value1",
                size_bytes=40,
                producer_step_id="produce1",
                consumer_step_ids=("consume1",),
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
        ),
        resources=(
            PipelineResourceKind.DMA,
            PipelineResourceKind.MATMUL,
            PipelineResourceKind.OTHER,
        ),
        sram_capacity_bytes=256,
    )
    return ctx


def test_expansion_adds_spill_and_reload_dma_steps_when_budget_is_tight():
    ctx = make_scheduled_spill_context(max_memory=64)

    ScheduledMemoryExpansionPass().run(ctx)

    problem = ctx.pipeline_schedule_problem
    assert problem is not None
    steps_by_id = {step.id: step for step in problem.steps}

    assert "value0.spill0" in steps_by_id
    assert "value0.reload0" in steps_by_id
    assert isinstance(steps_by_id["value0.spill0"], TransferStep)
    assert isinstance(steps_by_id["value0.reload0"], TransferStep)
    assert steps_by_id["value0.spill0"].transfer_kind is TransferStepKind.SPILL_DMA
    assert steps_by_id["value0.reload0"].transfer_kind is TransferStepKind.RELOAD_DMA
    assert ScheduleEdge("value0.reload0", "consume0", ScheduleDependencyKind.DATA) in problem.edges
    assert problem.sram_capacity_bytes == 64
    assert any(
        window == ResidencyWindow(
            value_name="value0",
            residency_id="value0@spill0",
            opened_by_step_id="produce0",
            closed_by_step_id="value0.spill0",
        )
        for window in problem.residency_windows
    )
    assert any(
        window.value_name == "value0.reload0.resident"
        and window.opened_by_step_id == "value0.reload0"
        and window.closed_by_step_id == "consume0"
        for window in problem.residency_windows
    )

    result = ListPipelineScheduler().solve(problem)
    assert result.feasible is True


def test_expansion_rejects_must_reside_value_that_cannot_fit():
    ctx = make_must_reside_context(max_memory=32)

    with pytest.raises(RuntimeError, match="must reside in SRAM"):
        ScheduledMemoryExpansionPass().run(ctx)


def test_expansion_reuses_one_reload_per_consumer_step_when_value_is_read_twice():
    ctx = make_repeated_consumer_context(max_memory=64)

    ScheduledMemoryExpansionPass().run(ctx)

    problem = ctx.pipeline_schedule_problem
    assert problem is not None

    reload_steps = [
        step
        for step in problem.steps
        if step.id.startswith("value0.reload")
    ]
    assert [step.id for step in reload_steps] == ["value0.reload0"]

    consume0 = next(step for step in problem.steps if step.id == "consume0")
    assert consume0.sram_input_names == (
        "value0.reload0.resident",
        "value0.reload0.resident",
    )

    reload_resident_values = [
        value.name
        for value in problem.scheduled_values
        if value.name.startswith("value0.reload")
    ]
    assert reload_resident_values == ["value0.reload0.resident"]
