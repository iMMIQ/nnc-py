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
from nnc_py.passes.memory_planning_v4 import MemoryPlanningPassV4


def _artifact_text(artifacts, filename: str) -> str:
    return next(file.content for file in artifacts.files if file.filename == filename)


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
    cflags = ["-std=c11", "-Wall", "-Wextra", "-fno-common", "-fPIC"]
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
        ["gcc", "-o", str(tmpdir / "model"), *[str(path) for path in object_files], "-lm"],
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


def test_codegen_without_schedule_metadata_still_generates_compilable_output():
    ctx = _make_relu_context()

    artifacts = X86Backend().generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")

    assert "Pipeline schedule summary" in model_c
    assert "schedule_metadata=absent" in model_c
    assert "void nnc_run(void)" in model_c

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_artifacts(output_dir, artifacts)
        _compile_generated_sources(output_dir)
