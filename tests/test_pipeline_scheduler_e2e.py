"""End-to-end coverage for joint-tiling-schedule O3 compile path."""

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


def _require_resnet18_model_path() -> Path:
    model_path = Path(__file__).resolve().parents[1] / "models" / "resnet18.onnx"
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")
    return model_path


def _compile_model(
    tmp_path,
    *,
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
        metadata=metadata,
        max_memory=max_memory,
    )

    assert backend.ctx is not None
    return backend.ctx, output_dir


def _compile_existing_model(
    model_path: Path,
    output_dir: Path,
    *,
    metadata: dict[str, object] | None = None,
):
    compiler = Compiler(
        target="x86",
        opt_level=3,
        enable_constant_folding=False,
    )
    backend = _CapturingX86Backend()
    compiler.backend = backend
    compiler.compile(
        str(model_path),
        str(output_dir),
        metadata=metadata,
    )
    assert backend.ctx is not None
    return backend.ctx


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


def _assert_joint_imported_offsets(ctx) -> None:
    result = ctx.pipeline_schedule_result
    assert result is not None

    scheduled_plan = ctx.metadata["scheduled_memory_plan"]
    compat_plan = ctx.metadata["memory_allocation_plan"]
    imported_offsets = {
        interval.value_name: interval.offset
        for interval in result.sram_intervals
        if interval.item_kind == "resident_window"
    }

    if imported_offsets:
        assert set(imported_offsets).issubset(scheduled_plan.fast_allocations)
        assert set(imported_offsets).issubset(compat_plan.tensor_allocations)
        assert set(imported_offsets).issubset(compat_plan.logical_regions)
        for value_name, offset in imported_offsets.items():
            assert offset is not None
            assert scheduled_plan.fast_allocations[value_name].offset == offset
            assert compat_plan.tensor_allocations[value_name].offset == offset
            assert compat_plan.logical_regions[value_name].offset == offset
        return

    assert scheduled_plan.fast_allocations == {}
    assert compat_plan.tensor_allocations == {}
    assert compat_plan.logical_regions == {}
    assert compat_plan.total_fast_memory == 0


def test_generated_makefile_uses_real_runtime_path_for_out_of_tree_builds(tmp_path):
    _, output_dir = _compile_model(tmp_path)

    makefile_text = (output_dir / "Makefile").read_text()
    runtime_dir = Path(__file__).resolve().parents[1] / "runtime"

    assert "NNC_RUNTIME ?= ../../runtime" not in makefile_text
    assert f"NNC_RUNTIME ?= {runtime_dir}" in makefile_text
    assert "NNC_RUNTIME ?=" in makefile_text


def test_joint_contract_path_materializes_and_builds_generated_output(tmp_path):
    ctx, output_dir = _compile_model(tmp_path)

    assert ctx.joint_tiling_schedule_problem is not None
    assert ctx.joint_tiling_schedule_solution is not None
    assert ctx.joint_tiling_schedule_failure is None
    assert ctx.pipeline_schedule_problem is not None
    assert ctx.pipeline_schedule_problem.metadata["origin"] == "joint_tiling_schedule_materialize"
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.feasible is True
    assert ctx.pipeline_schedule_result.solver_name == "joint_materialized"
    _assert_joint_imported_offsets(ctx)

    model_c = (output_dir / "model.c").read_text()
    assert "Pipeline schedule summary" in model_c
    assert "schedule_metadata=present" in model_c
    assert "solver=joint_materialized" in model_c

    _build_generated_x86_source(output_dir)


def test_joint_contract_external_solver_solution_materializes_and_builds(tmp_path):
    ctx, output_dir = _compile_model(
        tmp_path,
        metadata={
            "joint_tiling_schedule_solver_command": _joint_solver_command("solution"),
        },
    )

    assert ctx.joint_tiling_schedule_solution is not None
    assert ctx.joint_tiling_schedule_solution.diagnostics["mode"] == "solution"
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.solver_name == "joint_materialized"
    _assert_joint_imported_offsets(ctx)

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
            metadata={
                "joint_tiling_schedule_solver_command": _joint_solver_command(
                    "infeasible"
                ),
            },
        )

    assert getattr(exc_info.value, "error_category", None) == "solver_reported_infeasible"
    assert "solver_reported_infeasible" in str(exc_info.value)


def test_joint_contract_resnet18_compiles_with_default_submodule_solver(tmp_path):
    model_path = _require_resnet18_model_path()

    ctx = _compile_existing_model(
        model_path,
        tmp_path / "resnet18_joint_build",
    )

    assert ctx.joint_tiling_schedule_problem is not None
    assert ctx.joint_tiling_schedule_solution is not None
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.feasible is True
