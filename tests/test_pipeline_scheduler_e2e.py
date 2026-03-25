"""End-to-end coverage for pipeline scheduling and strict O3 scheduler mode."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

from nnc_py.compiler import Compiler
from nnc_py.codegen.x86_backend import X86Backend


class _CapturingX86Backend:
    def __init__(self) -> None:
        self.delegate = X86Backend()
        self.ctx = None

    def generate(self, ctx):
        self.ctx = ctx
        return self.delegate.generate(ctx)


def _make_pipeline_ready_matmul_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [4, 3],
        np.ones((4, 3), dtype=np.float32).reshape(-1).tolist(),
    )
    matmul = helper.make_node(
        "MatMul",
        inputs=["input", "weight"],
        outputs=["output"],
        name="matmul0",
    )
    graph = helper.make_graph(
        [matmul],
        "pipeline_scheduler_e2e_matmul",
        [input_info],
        [output_info],
        [weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_pipeline_ready_gemm_model() -> onnx.ModelProto:
    lhs = helper.make_tensor(
        "lhs",
        TensorProto.FLOAT,
        [1, 4],
        np.ones((1, 4), dtype=np.float32).reshape(-1).tolist(),
    )
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [4, 3],
        np.ones((4, 3), dtype=np.float32).reshape(-1).tolist(),
    )
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
    gemm = helper.make_node(
        "Gemm",
        inputs=["lhs", "weight"],
        outputs=["output"],
        name="gemm0",
    )
    graph = helper.make_graph(
        [gemm],
        "pipeline_scheduler_e2e_gemm",
        [],
        [output_info],
        [lhs, weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _compile_model(
    tmp_path,
    *,
    enable_pipeline_scheduler: bool | None,
    cost_model_cli_command: list[str] | None = None,
    metadata: dict[str, object] | None = None,
    max_memory: str | None = None,
    model_factory=_make_pipeline_ready_matmul_model,
):
    model = model_factory()
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "build"
    onnx.save(model, model_path)

    compiler = Compiler(
        target="x86",
        opt_level=3,
        enable_constant_folding=False,
        cost_model_cli_command=cost_model_cli_command,
    )
    backend = _CapturingX86Backend()
    compiler.backend = backend
    compiler.compile(
        str(model_path),
        str(output_dir),
        enable_pipeline_scheduler=enable_pipeline_scheduler,
        metadata=metadata,
        max_memory=max_memory,
    )

    assert backend.ctx is not None
    return backend.ctx, output_dir


def _build_generated_x86_source(output_dir: Path) -> None:
    subprocess.run(
        ["make", "clean"],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    result = subprocess.run(
        ["make"],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert (output_dir / "model").exists()


def test_generated_makefile_uses_real_runtime_path_for_out_of_tree_builds(tmp_path):
    _, output_dir = _compile_model(
        tmp_path,
        enable_pipeline_scheduler=True,
    )

    makefile_text = (output_dir / "Makefile").read_text()
    runtime_dir = Path(__file__).resolve().parents[1] / "runtime"

    assert "NNC_RUNTIME ?= ../../runtime" not in makefile_text
    assert f"NNC_RUNTIME ?= {runtime_dir}" in makefile_text
    assert "NNC_RUNTIME ?=" in makefile_text


def test_explicit_scheduler_enable_emits_current_schedule_contract_and_buildable_x86_source(
    tmp_path,
):
    ctx, output_dir = _compile_model(
        tmp_path,
        enable_pipeline_scheduler=True,
    )

    assert ctx.metadata["pipeline_scheduler_enabled"] is True
    assert ctx.pipeline_schedule_problem is not None
    assert ctx.pipeline_schedule_problem.metadata["origin"] == "pipeline_step_lowering"
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.solver_name == "list"
    assert ctx.pipeline_schedule_result.feasible is True
    assert ctx.pipeline_schedule_result.makespan > 0
    assert len(ctx.pipeline_schedule_result.scheduled_steps) == 1
    assert ctx.pipeline_schedule_result.diagnostics["scheduled_order"] == (
        "matmul0.compute",
    )
    assert ctx.metadata["memory_allocation_plan"].strategy_name in {
        "schedule_time_v4",
        "tile_regions_v3",
    }

    model_c = (output_dir / "model.c").read_text()
    assert "Pipeline schedule summary" in model_c
    assert "schedule_metadata=present" in model_c
    assert "solver=list" in model_c
    assert "feasible=yes" in model_c
    assert "pipeline step:" in model_c
    assert "memory_plan_strategy=" in model_c

    _build_generated_x86_source(output_dir)


def test_default_o3_branch_uses_scheduled_contract_and_builds_generated_output(
    tmp_path,
):
    ctx, output_dir = _compile_model(
        tmp_path,
        enable_pipeline_scheduler=None,
        cost_model_cli_command=["/definitely-missing-cost-model-command-12345"],
    )

    assert ctx.metadata["pipeline_scheduler_enabled"] is True
    assert "pipeline_scheduler_fallback" not in ctx.metadata
    assert ctx.metadata["cost_model_cli_command"] == [
        "/definitely-missing-cost-model-command-12345"
    ]
    assert ctx.pipeline_schedule_problem is not None
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.solver_name == "list"
    assert ctx.pipeline_schedule_result.feasible is True
    assert ctx.metadata["memory_allocation_plan"].strategy_name == "schedule_time_v4"

    model_c = (output_dir / "model.c").read_text()
    assert "Pipeline schedule summary" in model_c
    assert "schedule_metadata=present" in model_c
    assert "solver=list" in model_c
    assert "memory_plan_strategy=" in model_c

    _build_generated_x86_source(output_dir)


def test_missing_cli_cost_model_falls_back_without_failing_compile_or_build(tmp_path):
    ctx, output_dir = _compile_model(
        tmp_path,
        enable_pipeline_scheduler=True,
        cost_model_cli_command=["/definitely-missing-cost-model-command-12345"],
    )

    assert ctx.metadata["pipeline_scheduler_enabled"] is True
    assert ctx.metadata["cost_model_cli_command"] == [
        "/definitely-missing-cost-model-command-12345"
    ]
    assert ctx.pipeline_schedule_problem is not None
    assert ctx.pipeline_schedule_result is not None
    assert all(
        step.attrs.get("cost_model") == "simple"
        for step in ctx.pipeline_schedule_problem.steps
    )
    assert (output_dir / "model.c").exists()

    _build_generated_x86_source(output_dir)


def test_strict_o3_scheduled_compile_with_impossible_max_memory_preserves_budget_diagnostics(
    tmp_path,
):
    model = _make_pipeline_ready_gemm_model()
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "build"
    onnx.save(model, model_path)

    compiler = Compiler(
        target="x86",
        opt_level=3,
        enable_constant_folding=False,
    )
    backend = _CapturingX86Backend()
    compiler.backend = backend

    with pytest.raises(RuntimeError) as exc_info:
        compiler.compile(
            str(model_path),
            str(output_dir),
            enable_pipeline_scheduler=True,
            max_memory="80",
        )

    message = str(exc_info.value)
    assert "no_feasible_schedule_under_budget" in message
    assert "step_id=gemm0.shape_prep" in message
    assert "disable-pipeline-scheduler" not in message


def test_scheduler_disable_keeps_conservative_fallback_metadata_explicit_and_buildable(
    tmp_path,
):
    ctx, output_dir = _compile_model(
        tmp_path,
        enable_pipeline_scheduler=False,
    )

    assert ctx.metadata["pipeline_scheduler_enabled"] is False
    assert ctx.metadata["pipeline_scheduler_fallback"] == "legacy_o3_disabled"
    assert ctx.pipeline_schedule_problem is None
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.solver_name == "disabled"
    assert ctx.pipeline_schedule_result.diagnostics["strategy"] == "serial"
    assert ctx.pipeline_schedule_result.diagnostics["reason"] == "pipeline_scheduler_disabled"
    assert ctx.metadata["memory_allocation_plan"].strategy_name != "schedule_time_v4"

    model_c = (output_dir / "model.c").read_text()
    assert "Pipeline schedule summary" in model_c
    assert "schedule_metadata=present" in model_c
    assert "memory_plan_strategy=" in model_c

    _build_generated_x86_source(output_dir)
