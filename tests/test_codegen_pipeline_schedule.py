"""Tests for schedule-aware x86 simulation code generation."""

from __future__ import annotations

from dataclasses import replace
import subprocess
import tempfile
from pathlib import Path

from nnc_py.codegen.x86_backend import X86Backend
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    ScheduleDependencyKind,
    ScheduleEdge,
    PipelineScheduleProblem,
    PipelineScheduleResult,
    SramAllocationInterval,
    ScheduledValue,
    ScheduledValueHomeTier,
    ScheduledStep,
    ScheduleStep,
    ScheduleStepKind,
    TransferStep,
    TransferStepKind,
)
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType
from nnc_py.passes.liveness import LivenessAnalysisPass
from nnc_py.passes.memory_planning_v4 import MemoryPlanningPassV4
from nnc_py.passes.memory_strategy import (
    MemoryAllocationPlan,
    ReloadPoint,
    SpillPoint,
    TensorAllocation,
)
from nnc_py.passes.scheduled_memory_planning import (
    ScheduledFastAllocation,
    ScheduledMemoryPlan,
    ScheduledSlowAllocation,
    ScheduledTransferPoint,
)


def _artifact_text(artifacts, filename: str) -> str:
    return next(file.content for file in artifacts.files if file.filename == filename)


def _staged_value_name(node_name: str, tensor_name: str) -> str:
    return (
        f"sram|node|{len(node_name)}:{node_name}"
        f"|tensor|{len(tensor_name)}:{tensor_name}"
    )


def _make_relu_context() -> CompileContext:
    graph = Graph("schedule_codegen")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor(
        TensorType(
            name="input",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 4]),
        )
    )
    graph.add_tensor(
        TensorType(
            name="output",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 4]),
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.RELU,
            name="relu0",
            inputs=["input"],
            outputs=["output"],
        )
    )
    return CompileContext(graph=graph, target="x86", optimization_level=3)


def make_codegen_context_with_native_spill() -> CompileContext:
    graph = Graph("schedule_native_spill_codegen")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
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
    graph.add_tensor(
        TensorType(
            name="output",
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
    graph.add_node(
        Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["mid"],
            outputs=["output"],
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    ctx.metadata["pipeline_scheduler_enabled"] = True
    ctx.metadata["pipeline_schedule_problem"] = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="relu0.compute",
                node_name="relu0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=2,
                sram_input_names=("input",),
                sram_output_names=("mid",),
                attrs={"cost_model": "unit_test_cost_model"},
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
            TransferStep(
                id="mid.reload0",
                node_name="reload:mid",
                transfer_kind=TransferStepKind.RELOAD_DMA,
                moved_value_name="mid",
                bytes=16,
                duration=1,
                sram_output_names=("mid.reload0.resident",),
            ),
            ScheduleStep(
                id="relu1.compute",
                node_name="relu1",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=2,
                sram_input_names=("mid.reload0.resident",),
                sram_output_names=("output",),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
        ),
        edges=(
            ScheduleEdge("relu0.compute", "mid.spill0", ScheduleDependencyKind.DATA),
            ScheduleEdge("mid.spill0", "mid.reload0", ScheduleDependencyKind.ORDER),
            ScheduleEdge("mid.reload0", "relu1.compute", ScheduleDependencyKind.DATA),
        ),
        scheduled_values=(
            ScheduledValue(
                name="input",
                graph_tensor_name="input",
                size_bytes=16,
                producer_step_id=None,
                consumer_step_ids=("relu0.compute",),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.INPUT,
            ),
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
            ScheduledValue(
                name="mid.reload0.resident",
                graph_tensor_name="mid",
                size_bytes=16,
                producer_step_id="mid.reload0",
                consumer_step_ids=("relu1.compute",),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
            ScheduledValue(
                name="output",
                graph_tensor_name="output",
                size_bytes=16,
                producer_step_id="relu1.compute",
                consumer_step_ids=(),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
        ),
        resources=(PipelineResourceKind.DMA, PipelineResourceKind.OTHER),
        sram_capacity_bytes=16,
        metadata={"origin": "test"},
    )
    ctx.metadata["pipeline_schedule_result"] = PipelineScheduleResult(
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
            ScheduledStep(
                step_id="mid.reload0",
                resource_kind=PipelineResourceKind.DMA,
                resource_slot=0,
                start_time=3,
                end_time=4,
            ),
            ScheduledStep(
                step_id="relu1.compute",
                resource_kind=PipelineResourceKind.OTHER,
                resource_slot=0,
                start_time=4,
                end_time=6,
            ),
        ),
        feasible=True,
        makespan=6,
        solver_name="list",
        diagnostics={"strategy": "scheduled"},
    )
    ctx.metadata["scheduled_memory_plan"] = ScheduledMemoryPlan(
        total_fast_memory=32,
        total_slow_memory=16,
        fast_allocations={
            "mid@spill0": ScheduledFastAllocation(
                residency_id="mid@spill0",
                value_name="mid",
                buffer_id=0,
                offset=0,
                size_bytes=16,
                start_time=2,
                end_time=3,
                opened_by_step_id="relu0.compute",
                closed_by_step_id="mid.spill0",
            ),
            "mid.reload0.resident@0": ScheduledFastAllocation(
                residency_id="mid.reload0.resident@0",
                value_name="mid.reload0.resident",
                buffer_id=1,
                offset=16,
                size_bytes=16,
                start_time=4,
                end_time=6,
                opened_by_step_id="mid.reload0",
                closed_by_step_id="relu1.compute",
            ),
        },
        slow_allocations={
            "mid": ScheduledSlowAllocation(
                value_name="mid",
                offset=0,
                size_bytes=16,
            )
        },
        transfer_points=(
            ScheduledTransferPoint(
                step_id="mid.spill0",
                transfer_kind=TransferStepKind.SPILL_DMA,
                value_name="mid",
                size_bytes=16,
                start_time=2,
                end_time=3,
                fast_offset=0,
                slow_offset=0,
                after_node_name="relu0",
            ),
            ScheduledTransferPoint(
                step_id="mid.reload0",
                transfer_kind=TransferStepKind.RELOAD_DMA,
                value_name="mid",
                size_bytes=16,
                start_time=3,
                end_time=4,
                fast_offset=16,
                slow_offset=0,
                resident_value_name="mid.reload0.resident",
                before_node_name="relu1",
            ),
        ),
    )
    return ctx


def make_codegen_context_with_missing_reload_graph_tensor_name() -> CompileContext:
    ctx = make_codegen_context_with_native_spill()
    problem = ctx.metadata["pipeline_schedule_problem"]
    ctx.metadata["pipeline_schedule_problem"] = replace(
        problem,
        scheduled_values=tuple(
            replace(value, graph_tensor_name="")
            if value.name == "mid.reload0.resident"
            else value
            for value in problem.scheduled_values
        ),
    )
    return ctx


def make_codegen_context_with_scheduled_native_fast_bindings() -> CompileContext:
    graph = Graph("schedule_native_fast_bindings")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor(
        TensorType(
            name="input",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 4]),
        )
    )
    graph.add_tensor(
        TensorType(
            name="output",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 4]),
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.RELU,
            name="relu0",
            inputs=["input"],
            outputs=["output"],
        )
    )

    staged_input = _staged_value_name("relu0", "input")
    staged_output = _staged_value_name("relu0", "output")

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    ctx.metadata["pipeline_scheduler_enabled"] = True
    ctx.metadata["pipeline_schedule_problem"] = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="relu0.dma_in",
                node_name="relu0",
                step_kind=ScheduleStepKind.DMA_IN,
                resource_kind=PipelineResourceKind.DMA,
                duration=2,
                sram_input_names=("input",),
                sram_output_names=(staged_input,),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
            ScheduleStep(
                id="relu0.compute",
                node_name="relu0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=3,
                sram_input_names=(staged_input,),
                sram_output_names=(staged_output,),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
            ScheduleStep(
                id="relu0.dma_out",
                node_name="relu0",
                step_kind=ScheduleStepKind.DMA_OUT,
                resource_kind=PipelineResourceKind.DMA,
                duration=2,
                sram_input_names=(staged_output,),
                sram_output_names=("output",),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
        ),
        edges=(
            ScheduleEdge("relu0.dma_in", "relu0.compute", ScheduleDependencyKind.DATA),
            ScheduleEdge("relu0.compute", "relu0.dma_out", ScheduleDependencyKind.DATA),
        ),
        scheduled_values=(
            ScheduledValue(
                name="input",
                graph_tensor_name="input",
                size_bytes=16,
                producer_step_id=None,
                consumer_step_ids=("relu0.dma_in",),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.INPUT,
            ),
            ScheduledValue(
                name=staged_input,
                graph_tensor_name="input",
                size_bytes=16,
                producer_step_id="relu0.dma_in",
                consumer_step_ids=("relu0.compute",),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.INPUT,
            ),
            ScheduledValue(
                name=staged_output,
                graph_tensor_name="output",
                size_bytes=16,
                producer_step_id="relu0.compute",
                consumer_step_ids=("relu0.dma_out",),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
            ScheduledValue(
                name="output",
                graph_tensor_name="output",
                size_bytes=16,
                producer_step_id="relu0.dma_out",
                consumer_step_ids=(),
                must_reside_in_sram=True,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
        ),
        resources=(PipelineResourceKind.DMA, PipelineResourceKind.OTHER),
        sram_capacity_bytes=64,
        metadata={"origin": "test"},
    )
    ctx.metadata["pipeline_schedule_result"] = PipelineScheduleResult(
        scheduled_steps=(
            ScheduledStep(
                step_id="relu0.dma_in",
                resource_kind=PipelineResourceKind.DMA,
                resource_slot=0,
                start_time=0,
                end_time=2,
            ),
            ScheduledStep(
                step_id="relu0.compute",
                resource_kind=PipelineResourceKind.OTHER,
                resource_slot=0,
                start_time=2,
                end_time=5,
            ),
            ScheduledStep(
                step_id="relu0.dma_out",
                resource_kind=PipelineResourceKind.DMA,
                resource_slot=0,
                start_time=5,
                end_time=7,
            ),
        ),
        feasible=True,
        makespan=7,
        solver_name="list",
        diagnostics={"strategy": "scheduled_native"},
    )
    ctx.metadata["scheduled_memory_plan"] = ScheduledMemoryPlan(
        total_fast_memory=64,
        total_slow_memory=0,
        fast_allocations={
            f"{staged_input}@0": ScheduledFastAllocation(
                residency_id=f"{staged_input}@0",
                value_name=staged_input,
                buffer_id=0,
                offset=0,
                size_bytes=16,
                start_time=2,
                end_time=5,
                opened_by_step_id="relu0.dma_in",
                closed_by_step_id="relu0.compute",
            ),
            f"{staged_output}@0": ScheduledFastAllocation(
                residency_id=f"{staged_output}@0",
                value_name=staged_output,
                buffer_id=1,
                offset=16,
                size_bytes=16,
                start_time=5,
                end_time=7,
                opened_by_step_id="relu0.compute",
                closed_by_step_id="relu0.dma_out",
            ),
            "output@0": ScheduledFastAllocation(
                residency_id="output@0",
                value_name="output",
                buffer_id=2,
                offset=32,
                size_bytes=16,
                start_time=7,
                end_time=7,
                opened_by_step_id="relu0.dma_out",
                closed_by_step_id=None,
            ),
        },
        slow_allocations={},
        transfer_points=(),
    )
    return ctx


def _attach_schedule_metadata(
    ctx: CompileContext,
    *,
    sram_intervals: tuple[SramAllocationInterval, ...] = (),
    run_memory_planning: bool = True,
) -> None:
    ctx.metadata["pipeline_schedule_problem"] = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="relu0.compute",
                node_name="relu0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=3,
                sram_input_names=("input",),
                sram_output_names=("output",),
                attrs={
                    "cost_model": "unit_test_cost_model",
                    "tile": "tile0",
                },
            ),
        ),
        scheduled_values=(
            ScheduledValue(
                name="output",
                graph_tensor_name="output",
                size_bytes=16,
                producer_step_id="relu0.compute",
                consumer_step_ids=(),
                must_reside_in_sram=True,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
        ),
        resources=(PipelineResourceKind.OTHER,),
        sram_capacity_bytes=128,
        metadata={"origin": "test"},
    )
    ctx.metadata["pipeline_schedule_result"] = PipelineScheduleResult(
        scheduled_steps=(
            ScheduledStep(
                step_id="relu0.compute",
                resource_kind=PipelineResourceKind.OTHER,
                resource_slot=0,
                start_time=2,
                end_time=5,
            ),
        ),
        sram_intervals=sram_intervals,
        feasible=True,
        makespan=5,
        solver_name="list",
        diagnostics={"strategy": "scheduled"},
    )
    if run_memory_planning:
        MemoryPlanningPassV4().run(ctx)


def _write_artifacts(tmpdir: Path, artifacts) -> None:
    for artifact in artifacts.files:
        output_path = tmpdir / artifact.filename
        if artifact.file_type == "binary":
            output_path.write_bytes(artifact.content)
            continue
        output_path.write_text(artifact.content)


def _compile_generated_sources(tmpdir: Path) -> None:
    runtime_dir = Path(__file__).resolve().parents[1] / "runtime"
    runtime_include = runtime_dir / "include"
    runtime_ops = runtime_dir / "x86" / "ops.c"
    cflags = ["-D_GNU_SOURCE", "-std=c11", "-Wall", "-Wextra", "-fno-common", "-fPIC", "-pthread"]
    object_files: list[Path] = []

    for filename in ("model.c", "tensors.c", "test_runner.c"):
        source_path = tmpdir / filename
        if not source_path.exists():
            continue
        object_path = source_path.with_suffix(".o")
        result = subprocess.run(
            ["gcc", *cflags, f"-I{runtime_include}", "-c", str(source_path), "-o", str(object_path)],
            capture_output=True,
            text=True,
            cwd=tmpdir,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        object_files.append(object_path)

    ops_object = tmpdir / "ops.o"
    result = subprocess.run(
        ["gcc", *cflags, f"-I{runtime_include}", "-c", str(runtime_ops), "-o", str(ops_object)],
        capture_output=True,
        text=True,
        cwd=tmpdir,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    object_files.append(ops_object)

    result = subprocess.run(
        [
            "gcc",
            "-o",
            str(tmpdir / "model"),
            *[str(path) for path in object_files],
            "-lm",
            "-pthread",
        ],
        capture_output=True,
        text=True,
        cwd=tmpdir,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_generated_x86_code_contains_pipeline_schedule_annotations():
    ctx = _make_relu_context()
    _attach_schedule_metadata(
        ctx,
        sram_intervals=(
            SramAllocationInterval(
                value_name="output",
                buffer_id="buf_conflict",
                start_time=2,
                end_time=5,
                size_bytes=16,
            ),
        ),
    )

    artifacts = X86Backend().generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")

    assert "Pipeline schedule summary" in model_c
    assert "NNC_PIPELINE_WORKER_COUNT 4" in model_c
    assert "nnc_pipeline_run_parallel" in model_c
    assert "solver=list" in model_c
    assert "feasible=yes" in model_c
    assert "memory_plan_strategy=schedule_time_v4" in model_c
    assert "step_id=relu0.compute" in model_c
    assert "resource=other" in model_c
    assert "start=2" in model_c
    assert "end=5" in model_c
    assert "duration=3" in model_c
    assert "cost_source=unit_test_cost_model" in model_c
    assert "output@0" in model_c
    assert "region=output" in model_c
    assert "buffer=buf_conflict" not in model_c
    assert "sram_bindings=output@0[region=output]" in model_c


def test_debug_schedule_annotation_output_stays_buildable():
    ctx = _make_relu_context()
    _attach_schedule_metadata(
        ctx,
        sram_intervals=(
            SramAllocationInterval(
                value_name="output",
                buffer_id="buf_conflict",
                start_time=2,
                end_time=5,
                size_bytes=16,
            ),
        ),
    )

    artifacts = X86Backend(debug_mode=True).generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")

    assert "Pipeline schedule summary" in model_c
    assert "solver=list" in model_c
    assert "feasible=yes" in model_c

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_artifacts(output_dir, artifacts)
        _compile_generated_sources(output_dir)


def test_schedule_annotation_uses_scheduler_binding_only_without_v4_plan():
    ctx = _make_relu_context()
    _attach_schedule_metadata(
        ctx,
        sram_intervals=(
            SramAllocationInterval(
                value_name="output",
                buffer_id="buf0",
                start_time=2,
                end_time=5,
                size_bytes=16,
            ),
        ),
        run_memory_planning=False,
    )

    artifacts = X86Backend().generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")

    assert "step_id=relu0.compute" in model_c
    assert "sram_bindings=output[buffer=buf0]" in model_c
    assert "output@0" not in model_c
    assert "nnc_pipeline_run_parallel" in model_c


def test_schedule_codegen_materializes_step_level_worker_functions():
    ctx = _make_relu_context()
    ctx.metadata["pipeline_schedule_problem"] = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="relu0.dma_in",
                node_name="relu0",
                step_kind=ScheduleStepKind.DMA_IN,
                resource_kind=PipelineResourceKind.DMA,
                duration=2,
                sram_input_names=("input",),
                sram_output_names=("input",),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
            ScheduleStep(
                id="relu0.shape_prep",
                node_name="relu0",
                step_kind=ScheduleStepKind.SHAPE_PREP,
                resource_kind=PipelineResourceKind.SHAPE,
                duration=1,
                sram_input_names=("input",),
                sram_output_names=(),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
            ScheduleStep(
                id="relu0.compute",
                node_name="relu0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=3,
                sram_input_names=("input",),
                sram_output_names=("output",),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
            ScheduleStep(
                id="relu0.dma_out",
                node_name="relu0",
                step_kind=ScheduleStepKind.DMA_OUT,
                resource_kind=PipelineResourceKind.DMA,
                duration=2,
                sram_input_names=("output",),
                sram_output_names=(),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
        ),
        scheduled_values=(
            ScheduledValue(
                name="input",
                graph_tensor_name="input",
                size_bytes=16,
                producer_step_id=None,
                consumer_step_ids=("relu0.dma_in", "relu0.shape_prep", "relu0.compute"),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.INPUT,
            ),
            ScheduledValue(
                name="output",
                graph_tensor_name="output",
                size_bytes=16,
                producer_step_id="relu0.compute",
                consumer_step_ids=("relu0.dma_out",),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.SLOW,
            ),
        ),
        resources=(
            PipelineResourceKind.DMA,
            PipelineResourceKind.SHAPE,
            PipelineResourceKind.OTHER,
        ),
        sram_capacity_bytes=128,
        metadata={"origin": "test"},
    )
    ctx.metadata["pipeline_schedule_result"] = PipelineScheduleResult(
        scheduled_steps=(
            ScheduledStep(
                step_id="relu0.dma_in",
                resource_kind=PipelineResourceKind.DMA,
                resource_slot=0,
                start_time=0,
                end_time=2,
            ),
            ScheduledStep(
                step_id="relu0.shape_prep",
                resource_kind=PipelineResourceKind.SHAPE,
                resource_slot=0,
                start_time=0,
                end_time=1,
            ),
            ScheduledStep(
                step_id="relu0.compute",
                resource_kind=PipelineResourceKind.OTHER,
                resource_slot=0,
                start_time=2,
                end_time=5,
            ),
            ScheduledStep(
                step_id="relu0.dma_out",
                resource_kind=PipelineResourceKind.DMA,
                resource_slot=0,
                start_time=5,
                end_time=7,
            ),
        ),
        feasible=True,
        makespan=7,
        solver_name="list",
        diagnostics={"strategy": "scheduled"},
    )

    artifacts = X86Backend().generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")

    assert "static void nnc_pipeline_step_relu0_dma_in(void)" in model_c
    assert "static void nnc_pipeline_step_relu0_shape_prep(void)" in model_c
    assert "static void nnc_pipeline_step_relu0_compute(void)" in model_c
    assert "static void nnc_pipeline_step_relu0_dma_out(void)" in model_c
    assert "_nnc_pipeline_touch_tensor_read(&tensor_input);" in model_c
    assert "_nnc_pipeline_shape_touch_tensor(&tensor_input);" in model_c
    assert "_nnc_pipeline_touch_tensor_write(&tensor_output);" in model_c
    assert "{0, 0, 2, nnc_pipeline_step_relu0_dma_in}" in model_c
    assert "{1, 0, 1, nnc_pipeline_step_relu0_shape_prep}" in model_c
    assert "{3, 2, 5, nnc_pipeline_step_relu0_compute}" in model_c
    assert "{0, 5, 7, nnc_pipeline_step_relu0_dma_out}" in model_c

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_artifacts(output_dir, artifacts)
        _compile_generated_sources(output_dir)


def test_schedule_codegen_materializes_staged_dma_buffers_and_memcpy_flow():
    ctx = _make_relu_context()
    staged_input = _staged_value_name("relu0", "input")
    staged_output = _staged_value_name("relu0", "output")
    ctx.metadata["pipeline_schedule_problem"] = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="relu0.dma_in",
                node_name="relu0",
                step_kind=ScheduleStepKind.DMA_IN,
                resource_kind=PipelineResourceKind.DMA,
                duration=2,
                sram_input_names=("input",),
                sram_output_names=(staged_input,),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
            ScheduleStep(
                id="relu0.compute",
                node_name="relu0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=3,
                sram_input_names=(staged_input,),
                sram_output_names=(staged_output,),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
            ScheduleStep(
                id="relu0.dma_out",
                node_name="relu0",
                step_kind=ScheduleStepKind.DMA_OUT,
                resource_kind=PipelineResourceKind.DMA,
                duration=2,
                sram_input_names=(staged_output,),
                sram_output_names=(),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
        ),
        scheduled_values=(
            ScheduledValue(
                name=staged_input,
                graph_tensor_name="input",
                size_bytes=16,
                producer_step_id="relu0.dma_in",
                consumer_step_ids=("relu0.compute",),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
            ScheduledValue(
                name=staged_output,
                graph_tensor_name="output",
                size_bytes=16,
                producer_step_id="relu0.compute",
                consumer_step_ids=("relu0.dma_out",),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
        ),
        resources=(PipelineResourceKind.DMA, PipelineResourceKind.OTHER),
        sram_capacity_bytes=128,
        metadata={"origin": "test"},
    )
    ctx.metadata["pipeline_schedule_result"] = PipelineScheduleResult(
        scheduled_steps=(
            ScheduledStep(
                step_id="relu0.dma_in",
                resource_kind=PipelineResourceKind.DMA,
                resource_slot=0,
                start_time=0,
                end_time=2,
            ),
            ScheduledStep(
                step_id="relu0.compute",
                resource_kind=PipelineResourceKind.OTHER,
                resource_slot=0,
                start_time=2,
                end_time=5,
            ),
            ScheduledStep(
                step_id="relu0.dma_out",
                resource_kind=PipelineResourceKind.DMA,
                resource_slot=0,
                start_time=5,
                end_time=7,
            ),
        ),
        feasible=True,
        makespan=7,
        solver_name="list",
        diagnostics={"strategy": "scheduled"},
    )

    artifacts = X86Backend().generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")

    assert "static unsigned char _nnc_pipeline_value_sram_node_5_relu0_tensor_5_input_buffer[16];" in model_c
    assert "static unsigned char _nnc_pipeline_value_sram_node_5_relu0_tensor_6_output_buffer[16];" in model_c
    assert "memcpy(_nnc_pipeline_value_sram_node_5_relu0_tensor_5_input_buffer, tensor_input.data, 16);" in model_c
    assert "tensor_input.data = _nnc_pipeline_value_sram_node_5_relu0_tensor_5_input_buffer;" in model_c
    assert "tensor_output.data = _nnc_pipeline_value_sram_node_5_relu0_tensor_6_output_buffer;" in model_c
    assert "_nnc_pipeline_value_sram_node_5_relu0_tensor_6_output_saved_data" in model_c
    assert (
        "memcpy(_nnc_pipeline_value_sram_node_5_relu0_tensor_6_output_saved_data, "
        "_nnc_pipeline_value_sram_node_5_relu0_tensor_6_output_buffer, 16);"
    ) not in model_c

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_artifacts(output_dir, artifacts)
        _compile_generated_sources(output_dir)


def test_schedule_codegen_moves_unified_spill_transfers_onto_dma_workers():
    ctx = _make_relu_context()
    LivenessAnalysisPass().run(ctx)
    ctx.metadata["pipeline_schedule_problem"] = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="relu0.dma_in",
                node_name="relu0",
                step_kind=ScheduleStepKind.DMA_IN,
                resource_kind=PipelineResourceKind.DMA,
                duration=2,
                sram_input_names=("input",),
                sram_output_names=("input",),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
            ScheduleStep(
                id="relu0.compute",
                node_name="relu0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=3,
                sram_input_names=("input",),
                sram_output_names=("output",),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
            ScheduleStep(
                id="relu0.dma_out",
                node_name="relu0",
                step_kind=ScheduleStepKind.DMA_OUT,
                resource_kind=PipelineResourceKind.DMA,
                duration=2,
                sram_input_names=("output",),
                sram_output_names=(),
                attrs={"cost_model": "unit_test_cost_model"},
            ),
        ),
        edges=(
            ScheduleEdge(
                src_step_id="relu0.dma_in",
                dst_step_id="relu0.compute",
                kind=ScheduleDependencyKind.SAME_NODE_SEQUENCE,
            ),
            ScheduleEdge(
                src_step_id="relu0.compute",
                dst_step_id="relu0.dma_out",
                kind=ScheduleDependencyKind.SAME_NODE_SEQUENCE,
            ),
        ),
        scheduled_values=(
            ScheduledValue(
                name="input",
                graph_tensor_name="input",
                size_bytes=16,
                producer_step_id=None,
                consumer_step_ids=("relu0.dma_in", "relu0.compute"),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.INPUT,
            ),
            ScheduledValue(
                name="output",
                graph_tensor_name="output",
                size_bytes=16,
                producer_step_id="relu0.compute",
                consumer_step_ids=("relu0.dma_out",),
                must_reside_in_sram=False,
                can_alias=True,
                home_tier=ScheduledValueHomeTier.SLOW,
            ),
        ),
        resources=(PipelineResourceKind.DMA, PipelineResourceKind.OTHER),
        sram_capacity_bytes=128,
        metadata={"origin": "test"},
    )
    ctx.metadata["pipeline_schedule_result"] = PipelineScheduleResult(
        scheduled_steps=(
            ScheduledStep(
                step_id="relu0.dma_in",
                resource_kind=PipelineResourceKind.DMA,
                resource_slot=0,
                start_time=0,
                end_time=2,
            ),
            ScheduledStep(
                step_id="relu0.compute",
                resource_kind=PipelineResourceKind.OTHER,
                resource_slot=0,
                start_time=2,
                end_time=5,
            ),
            ScheduledStep(
                step_id="relu0.dma_out",
                resource_kind=PipelineResourceKind.DMA,
                resource_slot=0,
                start_time=5,
                end_time=7,
            ),
        ),
        feasible=True,
        makespan=7,
        solver_name="list",
        diagnostics={"strategy": "scheduled"},
    )
    ctx.metadata["memory_allocation_plan"] = MemoryAllocationPlan(
        strategy_name="cost_aware",
        total_fast_memory=16,
        total_slow_memory=32,
        num_buffers=1,
        tensor_allocations={
            "input": TensorAllocation(
                tensor_name="input",
                buffer_id=0,
                offset=0,
                size=16,
                is_spilled=True,
            ),
            "output": TensorAllocation(
                tensor_name="output",
                buffer_id=0,
                offset=16,
                size=16,
                is_spilled=True,
            ),
        },
        tensor_to_buffer={"input": 0, "output": 0},
        spill_points=[
            SpillPoint(
                tensor_name="output",
                after_node="relu0",
                after_node_idx=0,
                from_buffer_id=0,
                from_fast_offset=16,
                to_slow_offset=16,
                size=16,
            )
        ],
        reload_points=[
            ReloadPoint(
                tensor_name="input",
                before_node="relu0",
                before_node_idx=0,
                from_slow_offset=0,
                to_buffer_id=0,
                to_fast_offset=0,
                size=16,
                reload_slot_id=0,
            )
        ],
    )

    artifacts = X86Backend().generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")

    assert "static void node_relu0_body(void)" in model_c
    assert (
        "static void nnc_pipeline_step_relu0_compute(void) {\n"
        "    _nnc_pipeline_saved_tensor_output_data = tensor_output.data;\n"
        "    tensor_output.data = _nnc_reload_buffer_1;\n"
        "    node_relu0_body();\n"
        "    if (_nnc_pipeline_saved_tensor_input_data != NULL) {\n"
        "        tensor_input.data = _nnc_pipeline_saved_tensor_input_data;\n"
        "    }\n"
        "}"
    ) in model_c
    assert (
        "static void nnc_pipeline_step_relu0_dma_in(void) {\n"
        "    memcpy(_nnc_slow_pool + 0, _nnc_bound_input_tensor_input, 16);\n"
        "    tensor_input.data = _nnc_slow_pool + 0;\n"
    ) in model_c
    assert "memcpy(_nnc_reload_buffer_0," in model_c
    assert "tensor_input.data = _nnc_reload_buffer_0;" in model_c
    assert (
        "static void nnc_pipeline_step_relu0_dma_out(void) {\n"
        "    memcpy(_nnc_slow_pool + 16,\n"
        "           _nnc_reload_buffer_1, 16);\n"
        "    if (_nnc_pipeline_saved_tensor_output_data != NULL) {\n"
        "        tensor_output.data = _nnc_pipeline_saved_tensor_output_data;\n"
        "    }\n"
        "}"
    ) in model_c
    assert "_nnc_pipeline_dep_counts[NNC_PIPELINE_STEP_COUNT] = {0, 1, 1};" in model_c

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_artifacts(output_dir, artifacts)
        _compile_generated_sources(output_dir)


def test_codegen_emits_real_dma_spill_and_reload_worker_steps():
    ctx = make_codegen_context_with_native_spill()

    artifacts = X86Backend().generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")
    tensors_c = _artifact_text(artifacts, "tensors.c")

    assert "spill_dma" in model_c
    assert "reload_dma" in model_c
    assert "nnc_pipeline_run_parallel" in model_c
    assert "static void nnc_pipeline_step_mid_spill0(void)" in model_c
    assert "static void nnc_pipeline_step_mid_reload0(void)" in model_c
    assert "memcpy(_nnc_slow_pool + 0, _nnc_fast_pool + 0, 16);" in model_c
    assert "memcpy(_nnc_fast_pool + 16, _nnc_slow_pool + 0, 16);" in model_c
    assert "#define NNC_FAST_MEMORY_SIZE 32" in tensors_c
    assert "#define NNC_SLOW_MEMORY_SIZE 16" in tensors_c

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_artifacts(output_dir, artifacts)
        _compile_generated_sources(output_dir)


def test_codegen_debug_mode_keeps_native_spill_and_reload_buildable():
    ctx = make_codegen_context_with_native_spill()

    artifacts = X86Backend(debug_mode=True).generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")

    assert "spill_dma" in model_c
    assert "reload_dma" in model_c
    assert "nnc_pipeline_run_parallel" not in model_c
    assert "memcpy(_nnc_slow_pool + 0, _nnc_fast_pool + 0, 16);" in model_c
    assert "memcpy(_nnc_fast_pool + 16, _nnc_slow_pool + 0, 16);" in model_c

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_artifacts(output_dir, artifacts)
        _compile_generated_sources(output_dir)


def test_scheduled_native_test_runner_avoids_dynamic_tensor_allocation():
    ctx = make_codegen_context_with_native_spill()

    artifacts = X86Backend().generate(ctx)
    test_runner_c = _artifact_text(artifacts, "test_runner.c")

    assert "malloc(" not in test_runner_c
    assert "calloc(" not in test_runner_c
    assert "free(" not in test_runner_c

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_artifacts(output_dir, artifacts)
        _compile_generated_sources(output_dir)
        result = subprocess.run(
            [str(output_dir / "model")],
            capture_output=True,
            text=True,
            cwd=output_dir,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout


def test_scheduled_native_fast_bindings_use_real_fast_pool_offsets():
    ctx = make_codegen_context_with_scheduled_native_fast_bindings()

    artifacts = X86Backend().generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")

    assert "_nnc_pipeline_value_sram_node_5_relu0_tensor_5_input_buffer[16]" in model_c
    assert "_nnc_pipeline_value_sram_node_5_relu0_tensor_6_output_buffer[16]" in model_c
    assert (
        "memcpy(_nnc_pipeline_value_sram_node_5_relu0_tensor_5_input_buffer, tensor_input.data, 16);"
    ) in model_c
    assert "tensor_input.data = _nnc_pipeline_value_sram_node_5_relu0_tensor_5_input_buffer;" in model_c
    assert "tensor_output.data = _nnc_pipeline_value_sram_node_5_relu0_tensor_6_output_buffer;" in model_c
    assert (
        "memcpy(_nnc_fast_pool + 32, "
        "_nnc_pipeline_value_sram_node_5_relu0_tensor_6_output_buffer, 16);"
    ) in model_c
    assert "tensor_output.data = _nnc_fast_pool + 32;" in model_c

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_artifacts(output_dir, artifacts)
        _compile_generated_sources(output_dir)


def test_codegen_missing_reload_graph_tensor_name_stays_buildable():
    ctx = make_codegen_context_with_missing_reload_graph_tensor_name()

    artifacts = X86Backend().generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")
    tensors_c = _artifact_text(artifacts, "tensors.c")

    assert "mid.reload0.resident.data" not in model_c
    assert "mid.reload0.resident_shape" not in tensors_c

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_artifacts(output_dir, artifacts)
        _compile_generated_sources(output_dir)


def test_codegen_without_schedule_metadata_still_generates_compilable_output():
    ctx = _make_relu_context()

    artifacts = X86Backend().generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")

    assert "Pipeline schedule summary" in model_c
    assert "schedule_metadata=absent" in model_c
    assert "void nnc_run(void)" in model_c
    assert "nnc_pipeline_run_parallel" not in model_c

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_artifacts(output_dir, artifacts)
        _compile_generated_sources(output_dir)
