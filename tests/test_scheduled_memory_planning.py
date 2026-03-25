"""Tests for scheduled-native SRAM and slow-memory planning."""

from __future__ import annotations

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
