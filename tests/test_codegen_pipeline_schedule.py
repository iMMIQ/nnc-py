"""Tests for schedule-aware x86 simulation code generation."""

from __future__ import annotations

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
    ScheduledStep,
    ScheduleStep,
    ScheduleStepKind,
    SramValue,
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
        sram_values=(
            SramValue(
                name="output",
                size_bytes=16,
                producer_step_id="relu0.compute",
                consumer_step_ids=(),
                must_reside_in_sram=True,
                can_alias=True,
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
        sram_values=(
            SramValue(
                name="input",
                size_bytes=16,
                producer_step_id=None,
                consumer_step_ids=("relu0.dma_in", "relu0.shape_prep", "relu0.compute"),
                must_reside_in_sram=False,
                can_alias=True,
            ),
            SramValue(
                name="output",
                size_bytes=16,
                producer_step_id="relu0.compute",
                consumer_step_ids=("relu0.dma_out",),
                must_reside_in_sram=False,
                can_alias=True,
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
        sram_values=(
            SramValue(
                name=staged_input,
                size_bytes=16,
                producer_step_id="relu0.dma_in",
                consumer_step_ids=("relu0.compute",),
                must_reside_in_sram=False,
                can_alias=True,
            ),
            SramValue(
                name=staged_output,
                size_bytes=16,
                producer_step_id="relu0.compute",
                consumer_step_ids=("relu0.dma_out",),
                must_reside_in_sram=False,
                can_alias=True,
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
    assert (
        "memcpy(_nnc_pipeline_value_sram_node_5_relu0_tensor_6_output_saved_data, "
        "_nnc_pipeline_value_sram_node_5_relu0_tensor_6_output_buffer, 16);"
    ) in model_c
    assert (
        "tensor_output.data = _nnc_pipeline_value_sram_node_5_relu0_tensor_6_output_saved_data;"
    ) in model_c

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
        sram_values=(
            SramValue(
                name="input",
                size_bytes=16,
                producer_step_id=None,
                consumer_step_ids=("relu0.dma_in", "relu0.compute"),
                must_reside_in_sram=False,
                can_alias=True,
            ),
            SramValue(
                name="output",
                size_bytes=16,
                producer_step_id="relu0.compute",
                consumer_step_ids=("relu0.dma_out",),
                must_reside_in_sram=False,
                can_alias=True,
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
