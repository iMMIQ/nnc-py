"""End-to-end coverage for scheduled and joint-contract O3 compile paths."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nnc_py.codegen.x86_backend import X86Backend
from nnc_py.compiler import Compiler
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType


class _CapturingX86Backend:
    def __init__(self) -> None:
        self.delegate = X86Backend()
        self.ctx = None

    def generate(self, ctx):
        self.ctx = ctx
        return self.delegate.generate(ctx)


def _make_pipeline_ready_matmul_graph() -> Graph:
    graph = Graph("pipeline_scheduler_e2e_matmul")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1, 4]),
            name="input",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4, 3]),
            name="weight",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1, 3]),
            name="output",
        )
    )
    graph.constants["weight"] = np.ones((4, 3), dtype=np.float32)
    graph.add_node(
        Node(
            op_type=OpType.MATMUL,
            name="matmul0",
            inputs=["input", "weight"],
            outputs=["output"],
        )
    )
    return graph


def _make_pipeline_ready_gemm_graph() -> Graph:
    graph = Graph("pipeline_scheduler_e2e_gemm")
    graph.outputs = ["output"]
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1, 4]),
            name="lhs",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4, 3]),
            name="weight",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1, 3]),
            name="output",
        )
    )
    graph.constants["lhs"] = np.ones((1, 4), dtype=np.float32)
    graph.constants["weight"] = np.ones((4, 3), dtype=np.float32)
    graph.add_node(
        Node(
            op_type=OpType.GEMM,
            name="gemm0",
            inputs=["lhs", "weight"],
            outputs=["output"],
        )
    )
    return graph


def _compile_model(
    tmp_path,
    *,
    enable_pipeline_scheduler: bool | None,
    cost_model_cli_command: list[str] | None = None,
    metadata: dict[str, object] | None = None,
    max_memory: str | None = None,
    graph_factory=_make_pipeline_ready_matmul_graph,
):
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "build"

    compiler = Compiler(
        target="x86",
        opt_level=3,
        enable_constant_folding=False,
        cost_model_cli_command=cost_model_cli_command,
    )
    compiler.frontend = SimpleNamespace(load=lambda _: graph_factory())
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


def _joint_solver_command(mode: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("fake_joint_solver.py")),
        mode,
    ]


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


def test_joint_contract_path_materializes_and_builds_generated_output(tmp_path):
    ctx, output_dir = _compile_model(
        tmp_path,
        enable_pipeline_scheduler=None,
        metadata={"enable_joint_tiling_schedule_contract": True},
    )

    assert ctx.metadata["enable_joint_tiling_schedule_contract"] is True
    assert ctx.joint_tiling_schedule_problem is not None
    assert ctx.joint_tiling_schedule_solution is not None
    assert ctx.joint_tiling_schedule_failure is None
    assert ctx.pipeline_schedule_problem is not None
    assert ctx.pipeline_schedule_problem.metadata["origin"] == "joint_tiling_schedule_materialize"
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.feasible is True
    assert ctx.pipeline_schedule_result.solver_name == "joint_materialized"

    model_c = (output_dir / "model.c").read_text()
    assert "Pipeline schedule summary" in model_c
    assert "schedule_metadata=present" in model_c
    assert "solver=joint_materialized" in model_c

    _build_generated_x86_source(output_dir)


def test_joint_contract_external_solver_solution_materializes_and_builds(tmp_path):
    ctx, output_dir = _compile_model(
        tmp_path,
        enable_pipeline_scheduler=None,
        metadata={
            "enable_joint_tiling_schedule_contract": True,
            "joint_tiling_schedule_solver_command": _joint_solver_command("solution"),
        },
    )

    assert ctx.joint_tiling_schedule_solution is not None
    assert ctx.joint_tiling_schedule_solution.diagnostics["mode"] == "solution"
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.solver_name == "joint_materialized"

    model_c = (output_dir / "model.c").read_text()
    assert "solver=joint_materialized" in model_c
    assert "feasible=yes" in model_c

    _build_generated_x86_source(output_dir)


def test_joint_contract_external_solver_failure_surfaces_standardized_category(
    tmp_path,
):
    compiler = Compiler(
        target="x86",
        opt_level=3,
        enable_constant_folding=False,
    )
    compiler.frontend = SimpleNamespace(load=lambda _: _make_pipeline_ready_matmul_graph())
    compiler.backend = _CapturingX86Backend()

    with pytest.raises(RuntimeError) as exc_info:
        compiler.compile(
            str(tmp_path / "model.onnx"),
            str(tmp_path / "build"),
            enable_pipeline_scheduler=None,
            metadata={
                "enable_joint_tiling_schedule_contract": True,
                "joint_tiling_schedule_solver_command": _joint_solver_command(
                    "infeasible"
                ),
            },
        )

    assert getattr(exc_info.value, "error_category", None) == "solver_reported_infeasible"
    assert "solver_reported_infeasible" in str(exc_info.value)


def test_strict_o3_scheduled_compile_with_impossible_max_memory_preserves_budget_diagnostics(
    tmp_path,
):
    compiler = Compiler(
        target="x86",
        opt_level=3,
        enable_constant_folding=False,
    )
    compiler.frontend = SimpleNamespace(load=lambda _: _make_pipeline_ready_gemm_graph())
    backend = _CapturingX86Backend()
    compiler.backend = backend

    with pytest.raises(RuntimeError) as exc_info:
        compiler.compile(
            str(tmp_path / "model.onnx"),
            str(tmp_path / "build"),
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
