"""Tests for importing joint-solver SRAM placement into scheduled metadata."""

from __future__ import annotations

from dataclasses import replace

import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    PipelineScheduleProblem,
    PipelineScheduleResult,
    ScheduleStep,
    ScheduleStepKind,
    ScheduledStep,
    ScheduledValue,
    ScheduledValueHomeTier,
    SramAllocationInterval,
    TransferStep,
    TransferStepKind,
    set_pipeline_schedule_problem,
    set_pipeline_schedule_result,
)
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType
from nnc_py.passes.joint_schedule_memory_import import JointScheduleMemoryImportPass


def _make_import_context(
    *,
    sram_intervals: tuple[SramAllocationInterval, ...],
) -> CompileContext:
    graph = Graph("joint_schedule_import")
    graph.inputs = ["input"]
    graph.outputs = ["mid"]
    graph.add_tensor(
        TensorType(
            name="input",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 4]),
        )
    )
    graph.add_tensor(
        TensorType(
            name="mid",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 4]),
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.RELU,
            name="relu0",
            inputs=["input"],
            outputs=["mid"],
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="relu0.compute",
                node_name="relu0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=2,
                sram_input_names=("input",),
                sram_output_names=("mid",),
            ),
            TransferStep(
                id="mid.spill0",
                node_name="spill:mid",
                transfer_kind=TransferStepKind.SPILL_DMA,
                moved_value_name="mid",
                bytes=16,
                duration=1,
                sram_input_names=("mid",),
            ),
        ),
        scheduled_values=(
            ScheduledValue(
                name="mid",
                graph_tensor_name="mid",
                size_bytes=16,
                producer_step_id="relu0.compute",
                consumer_step_ids=("mid.spill0",),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
        ),
        resources=(PipelineResourceKind.DMA, PipelineResourceKind.OTHER),
        sram_capacity_bytes=128,
        metadata={"origin": "joint_tiling_schedule_materialize"},
    )
    result = PipelineScheduleResult(
        scheduled_steps=(
            ScheduledStep(
                step_id="relu0.compute",
                resource_kind=PipelineResourceKind.OTHER,
                resource_slot=0,
                start_time=0,
                end_time=2,
            ),
            ScheduledStep(
                step_id="mid.spill0",
                resource_kind=PipelineResourceKind.DMA,
                resource_slot=0,
                start_time=2,
                end_time=3,
            ),
        ),
        sram_intervals=sram_intervals,
        scheduled_values=problem.scheduled_values,
        feasible=True,
        makespan=3,
        solver_name="joint_materialized",
        diagnostics={"strategy": "joint_contract"},
    )
    set_pipeline_schedule_problem(ctx, problem)
    set_pipeline_schedule_result(ctx, result)
    return ctx


def _make_no_fast_item_import_context() -> CompileContext:
    graph = Graph("joint_schedule_import_no_fast_item")
    graph.inputs = ["input"]
    graph.outputs = ["mid"]
    graph.add_tensor(
        TensorType(
            name="input",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 4]),
        )
    )
    graph.add_tensor(
        TensorType(
            name="mid",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 4]),
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.RELU,
            name="relu0",
            inputs=["input"],
            outputs=["mid"],
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="relu0.compute",
                node_name="relu0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=2,
                sram_input_names=(),
                sram_output_names=(),
            ),
        ),
        scheduled_values=(
            ScheduledValue(
                name="mid",
                graph_tensor_name="mid",
                size_bytes=16,
                producer_step_id="relu0.compute",
                consumer_step_ids=(),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.SLOW,
            ),
        ),
        resources=(PipelineResourceKind.OTHER,),
        sram_capacity_bytes=128,
        metadata={"origin": "joint_tiling_schedule_materialize"},
    )
    result = PipelineScheduleResult(
        scheduled_steps=(
            ScheduledStep(
                step_id="relu0.compute",
                resource_kind=PipelineResourceKind.OTHER,
                resource_slot=0,
                start_time=0,
                end_time=2,
            ),
        ),
        sram_intervals=(),
        scheduled_values=problem.scheduled_values,
        feasible=True,
        makespan=2,
        solver_name="joint_materialized",
        diagnostics={"strategy": "joint_contract"},
    )
    set_pipeline_schedule_problem(ctx, problem)
    set_pipeline_schedule_result(ctx, result)
    return ctx


def test_joint_schedule_memory_import_builds_compatibility_metadata_from_imported_offsets():
    ctx = _make_import_context(
        sram_intervals=(
            SramAllocationInterval(
                value_name="mid",
                item_id="mid@2.item",
                item_kind="resident_window",
                buffer_id="mid@2.item",
                offset=64,
                start_time=2,
                end_time=3,
                size_bytes=16,
            ),
        ),
    )

    JointScheduleMemoryImportPass().run(ctx)

    scheduled_plan = ctx.metadata["scheduled_memory_plan"]
    compat_plan = ctx.metadata["memory_allocation_plan"]

    assert compat_plan.strategy_name == "joint_solver_import"
    assert scheduled_plan.total_fast_memory == 80
    assert scheduled_plan.fast_allocations["mid"].offset == 64
    assert scheduled_plan.fast_allocations["mid"].opened_by_step_id == "relu0.compute"
    assert scheduled_plan.fast_allocations["mid"].closed_by_step_id == "mid.spill0"
    assert compat_plan.tensor_allocations["mid"].offset == 64
    assert compat_plan.logical_regions["mid"].offset == 64
    assert compat_plan.total_fast_memory == 80
    assert compat_plan.total_slow_memory == 16
    assert compat_plan.spill_points[0].from_fast_offset == 64
    assert compat_plan.spill_points[0].to_slow_offset == 0


def test_joint_schedule_memory_import_allows_empty_imports_when_no_sram_values_are_expected():
    ctx = _make_no_fast_item_import_context()

    JointScheduleMemoryImportPass().run(ctx)

    scheduled_plan = ctx.metadata["scheduled_memory_plan"]
    compat_plan = ctx.metadata["memory_allocation_plan"]

    assert compat_plan.strategy_name == "joint_solver_import"
    assert scheduled_plan.total_fast_memory == 0
    assert scheduled_plan.fast_allocations == {}
    assert scheduled_plan.transfer_points == ()
    assert compat_plan.total_fast_memory == 0
    assert compat_plan.buffers == []
    assert compat_plan.tensor_allocations == {}
    assert compat_plan.logical_regions == {}


def test_joint_schedule_memory_import_requires_imported_sram_intervals_for_sram_values():
    ctx = _make_import_context(sram_intervals=())

    with pytest.raises(RuntimeError, match="sram_intervals"):
        JointScheduleMemoryImportPass().run(ctx)


def test_joint_schedule_memory_import_requires_complete_imported_residency_metadata():
    ctx = _make_import_context(
        sram_intervals=(
            SramAllocationInterval(
                value_name="relu0.compute.temp",
                item_id="relu0.compute.temp",
                item_kind="temp_interval",
                buffer_id="relu0.compute.temp",
                offset=0,
                start_time=0,
                end_time=2,
                size_bytes=16,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="mid"):
        JointScheduleMemoryImportPass().run(ctx)


def test_joint_schedule_memory_import_rejects_non_imported_interval_shape():
    ctx = _make_import_context(
        sram_intervals=(
            SramAllocationInterval(
                value_name="mid",
                buffer_id="buf0",
                offset=64,
                start_time=2,
                end_time=3,
                size_bytes=16,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="item_id"):
        JointScheduleMemoryImportPass().run(ctx)


def test_joint_schedule_memory_import_requires_explicit_imported_offset():
    ctx = _make_import_context(
        sram_intervals=(
            SramAllocationInterval(
                value_name="mid",
                item_id="mid@2.item",
                item_kind="resident_window",
                buffer_id="mid@2.item",
                offset=None,
                start_time=2,
                end_time=3,
                size_bytes=16,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="offset"):
        JointScheduleMemoryImportPass().run(ctx)


def test_joint_schedule_memory_import_requires_interval_size_to_match_scheduled_value():
    ctx = _make_import_context(
        sram_intervals=(
            SramAllocationInterval(
                value_name="mid",
                item_id="mid@2.item",
                item_kind="resident_window",
                buffer_id="mid@2.item",
                offset=64,
                start_time=2,
                end_time=3,
                size_bytes=32,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="size_bytes"):
        JointScheduleMemoryImportPass().run(ctx)


def test_joint_schedule_memory_import_requires_spill_producer_node_from_problem_steps():
    ctx = _make_import_context(
        sram_intervals=(
            SramAllocationInterval(
                value_name="mid",
                item_id="mid@2.item",
                item_kind="resident_window",
                buffer_id="mid@2.item",
                offset=64,
                start_time=2,
                end_time=3,
                size_bytes=16,
            ),
        ),
    )
    problem = ctx.pipeline_schedule_problem
    assert problem is not None
    set_pipeline_schedule_problem(
        ctx,
        replace(
            problem,
            steps=(problem.steps[1],),
        ),
    )

    with pytest.raises(RuntimeError, match="producer node"):
        JointScheduleMemoryImportPass().run(ctx)
