"""Integration tests for the O3 pipeline scheduling path and strict scheduler mode."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from nnc_py.codegen.base import CodeGenResult
from nnc_py.compiler import Compiler
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType
from nnc_py.joint_schedule.solver import CliJointScheduleSolver
from nnc_py.passes.base import PassManager
from nnc_py.passes import joint_tiling_schedule as joint_pass_module
from tests.joint_solver_helpers import joint_solver_cli_command


class _CapturingBackend:
    def __init__(self) -> None:
        self.ctx = None

    def generate(self, ctx):
        self.ctx = ctx
        return CodeGenResult()


def _make_gemm_graph() -> Graph:
    graph = Graph("pipeline_integration")
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


def _compile_graph(
    monkeypatch,
    tmp_path,
    *,
    graph_factory=_make_gemm_graph,
    metadata: dict[str, object] | None = None,
    cost_model_cli_command: list[str] | None = None,
    max_memory: str | None = None,
    enable_pipeline_scheduler: bool | None = None,
    relax_scheduled_validation: bool = False,
):
    backend = _CapturingBackend()
    compiler = Compiler(
        target="x86",
        opt_level=3,
        cost_model_cli_command=cost_model_cli_command,
    )
    compiler.frontend = SimpleNamespace(load=lambda _: graph_factory())
    compiler.backend = backend
    monkeypatch.setattr(
        compiler,
        "_write_output",
        lambda artifacts, output_dir, entry_point: None,
    )
    if relax_scheduled_validation:
        from nnc_py.ir.pipeline_schedule import set_pipeline_schedule_result
        from nnc_py.passes.pipeline_scheduling import PipelineSchedulingPass

        monkeypatch.setattr(
            compiler,
            "_validate_scheduled_o3_result",
            lambda ctx: None,
        )
        monkeypatch.setattr(
            PipelineSchedulingPass,
            "_execute",
            lambda self, ctx: set_pipeline_schedule_result(
                ctx,
                self._scheduler.solve(ctx.pipeline_schedule_problem),
            ),
        )

    compiler.compile(
        "model.onnx",
        str(tmp_path),
        metadata=metadata,
        max_memory=max_memory,
        enable_pipeline_scheduler=enable_pipeline_scheduler,
    )

    assert backend.ctx is not None
    return backend.ctx


def _fake_joint_solver_command(mode: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("fake_joint_solver.py")),
        mode,
    ]


def test_o3_default_pass_order_uses_tile_aware_v3_without_scheduler_or_legacy_spill():
    names = [pass_obj.__class__.__name__ for pass_obj in PassManager.get_default_passes(3)]

    assert "ScheduleAnalysisPass" in names
    assert "LayoutPlanningPass" in names
    assert "TiledLoweringPass" in names
    assert "MemoryPlanningPassV3" in names
    assert "PipelineStepLoweringPass" not in names
    assert "PipelineSchedulingPass" not in names
    assert "MemoryPlanningPassV4" not in names
    assert "SpillAnalysisPass" not in names
    assert names.index("PrepackLoweringPass") < names.index("DominatorFusionPass")
    assert names.index("DominatorFusionPass") < names.index("ScheduleAnalysisPass")
    assert names.index("ScheduleAnalysisPass") < names.index("LayoutPlanningPass")
    assert names.index("LayoutPlanningPass") < names.index("TiledLoweringPass")
    assert names.index("TiledLoweringPass") < names.index("LivenessAnalysisPass")
    assert names.index("LivenessAnalysisPass") < names.index("MemoryPlanningPassV3")


def test_o3_conservative_helper_excludes_legacy_spill_pass():
    names = [
        pass_obj.__class__.__name__
        for pass_obj in PassManager.get_conservative_o3_passes()
    ]

    assert "MemoryPlanningPassV3" in names
    assert "SpillAnalysisPass" not in names


def test_o3_scheduled_pass_order_requires_explicit_helper():
    names = [
        pass_obj.__class__.__name__
        for pass_obj in PassManager.get_scheduled_o3_passes()
    ]

    assert names.index("ScheduleAnalysisPass") < names.index("LayoutPlanningPass")
    assert names.index("LayoutPlanningPass") < names.index("TiledLoweringPass")
    assert names.index("TiledLoweringPass") < names.index("PipelineStepLoweringPass")
    assert names.index("PipelineStepLoweringPass") < names.index("ScheduledMemoryExpansionPass")
    assert names.index("ScheduledMemoryExpansionPass") < names.index("PipelineSchedulingPass")
    assert names.index("PipelineSchedulingPass") < names.index("LivenessAnalysisPass")
    assert names.index("LivenessAnalysisPass") < names.index("ScheduledMemoryPlanningPass")
    assert "ScheduledMemoryExpansionPass" in names
    assert "MemoryPlanningPassV4" not in names
    assert "SpillAnalysisPass" not in names


def test_o3_joint_tiling_schedule_pass_order_runs_materialization_before_liveness():
    names = [
        pass_obj.__class__.__name__
        for pass_obj in PassManager.get_joint_tiling_schedule_o3_passes()
    ]

    assert names.index("ScheduleAnalysisPass") < names.index("LayoutPlanningPass")
    assert names.index("LayoutPlanningPass") < names.index("TiledLoweringPass")
    assert names.index("TiledLoweringPass") < names.index("JointTilingScheduleProblemPass")
    assert names.index("JointTilingScheduleProblemPass") < names.index("JointTilingScheduleSolvePass")
    assert names.index("JointTilingScheduleSolvePass") < names.index("JointTilingScheduleMaterializationPass")
    assert names.index("JointTilingScheduleMaterializationPass") < names.index("LivenessAnalysisPass")
    assert names.index("LivenessAnalysisPass") < names.index("JointScheduleMemoryImportPass")
    assert "PipelineStepLoweringPass" not in names
    assert "ScheduledMemoryExpansionPass" not in names
    assert "PipelineSchedulingPass" not in names
    assert "ScheduledMemoryPlanningPass" not in names


def test_o3_compile_defaults_to_scheduled_path(monkeypatch, tmp_path):
    command = ["external-cost-model", "--stdio"]

    ctx = _compile_graph(
        monkeypatch,
        tmp_path,
        cost_model_cli_command=command,
    )

    assert ctx.metadata["cost_model_cli_command"] == command
    assert ctx.metadata["pipeline_scheduler_enabled"] is True
    assert "pipeline_scheduler_fallback" not in ctx.metadata
    assert ctx.pipeline_schedule_problem is not None
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.feasible is True
    assert ctx.pipeline_schedule_result.solver_name == "list"
    assert "gemm0" in ctx.metadata.get("node_execution_plans", {})
    assert "scheduled_memory_plan" in ctx.metadata
    assert "memory_plan" not in ctx.metadata
    assert "spill_plan" not in ctx.metadata
    assert ctx.metadata["memory_allocation_plan"].strategy_name == "schedule_time_v4"


def test_explicitly_enabling_scheduler_path_populates_schedule_metadata(monkeypatch, tmp_path):
    command = ["external-cost-model", "--stdio"]

    ctx = _compile_graph(
        monkeypatch,
        tmp_path,
        cost_model_cli_command=command,
        enable_pipeline_scheduler=True,
    )

    assert ctx.metadata["pipeline_scheduler_enabled"] is True
    assert "pipeline_scheduler_fallback" not in ctx.metadata
    assert ctx.metadata["cost_model_cli_command"] == command
    assert ctx.pipeline_schedule_problem is not None
    assert ctx.pipeline_schedule_problem.metadata["origin"] == "pipeline_step_lowering"
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.solver_name == "list"
    assert "scheduled_memory_plan" in ctx.metadata
    assert "memory_plan" not in ctx.metadata
    assert "spill_plan" not in ctx.metadata
    assert ctx.metadata["memory_allocation_plan"].strategy_name == "schedule_time_v4"


def test_scheduled_compile_with_max_memory_records_expansion_output(monkeypatch, tmp_path):
    ctx = _compile_graph(
        monkeypatch,
        tmp_path,
        max_memory="80",
        relax_scheduled_validation=True,
    )

    problem = ctx.pipeline_schedule_problem
    assert problem is not None
    assert problem.metadata["scheduled_memory_expansion"]["max_memory"] == 80
    assert problem.metadata["scheduled_memory_expansion"]["spilled_values"]
    assert any(step.id.endswith(".spill0") for step in problem.steps)
    assert any(step.id.endswith(".reload0") for step in problem.steps)
    assert "memory_plan" not in ctx.metadata
    assert "spill_plan" not in ctx.metadata


def test_disabling_scheduler_path_keeps_fallback_state_explicit(monkeypatch, tmp_path):
    ctx = _compile_graph(
        monkeypatch,
        tmp_path,
        metadata={"disable_pipeline_scheduler": True},
    )

    assert ctx.metadata["pipeline_scheduler_enabled"] is False
    assert ctx.pipeline_schedule_problem is None
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.feasible is False
    assert ctx.pipeline_schedule_result.diagnostics["strategy"] == "serial"
    assert ctx.pipeline_schedule_result.diagnostics["reason"] == "pipeline_scheduler_disabled"
    assert ctx.metadata["pipeline_scheduler_fallback"] == "legacy_o3_disabled"
    assert ctx.metadata["memory_allocation_plan"].strategy_name == "tile_regions_v3"


def test_disabling_scheduler_with_max_memory_uses_legacy_compatible_fallback(
    monkeypatch,
    tmp_path,
):
    ctx = _compile_graph(
        monkeypatch,
        tmp_path,
        metadata={"disable_pipeline_scheduler": True},
        max_memory="64",
    )

    assert ctx.metadata["pipeline_scheduler_enabled"] is False
    assert ctx.pipeline_schedule_problem is None
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.diagnostics["strategy"] == "serial"
    assert ctx.metadata["pipeline_scheduler_fallback"] == "legacy_o3_disabled"
    assert ctx.metadata["memory_allocation_plan"].strategy_name == "tile_regions_v3"


def test_compiler_default_o3_uses_scheduled_o3_helper(
    monkeypatch,
    tmp_path,
):
    backend = _CapturingBackend()
    compiler = Compiler(target="x86", opt_level=3)
    compiler.frontend = SimpleNamespace(load=lambda _: _make_gemm_graph())
    compiler.backend = backend
    monkeypatch.setattr(
        compiler,
        "_write_output",
        lambda artifacts, output_dir, entry_point: None,
    )
    monkeypatch.setattr(
        PassManager, "get_default_passes", classmethod(lambda cls, opt_level: [])
    )
    monkeypatch.setattr(
        PassManager,
        "get_scheduled_o3_passes",
        classmethod(lambda cls: []),
    )

    compiler.compile(
        "model.onnx",
        str(tmp_path),
    )

    assert backend.ctx is not None


def test_compiler_enables_joint_tiling_schedule_contract_via_metadata(monkeypatch, tmp_path):
    ctx = _compile_graph(
        monkeypatch,
        tmp_path,
        metadata={"enable_joint_tiling_schedule_contract": True},
    )

    assert ctx.metadata["enable_joint_tiling_schedule_contract"] is True
    assert ctx.metadata["pipeline_scheduler_enabled"] is True
    assert ctx.joint_tiling_schedule_problem is not None
    assert ctx.joint_tiling_schedule_solution is not None
    assert ctx.joint_tiling_schedule_failure is None
    assert ctx.pipeline_schedule_problem is not None
    assert ctx.pipeline_schedule_problem.metadata["origin"] == "joint_tiling_schedule_materialize"
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.feasible is True
    assert ctx.pipeline_schedule_result.solver_name == "joint_materialized"
    assert "scheduled_memory_plan" in ctx.metadata
    assert ctx.metadata["memory_allocation_plan"].strategy_name == "joint_solver_import"


def test_joint_solver_defaults_to_submodule_cli_command():
    solver = joint_pass_module._build_solver(SimpleNamespace(metadata={}))

    assert isinstance(solver, CliJointScheduleSolver)
    assert list(solver.command) == joint_solver_cli_command()


def test_joint_solver_missing_submodule_cli_surfaces_bootstrap_message(
    monkeypatch,
    tmp_path,
):
    compiler = Compiler(target="x86", opt_level=3)
    compiler.frontend = SimpleNamespace(load=lambda _: _make_gemm_graph())
    compiler.backend = _CapturingBackend()
    monkeypatch.setattr(
        compiler,
        "_write_output",
        lambda artifacts, output_dir, entry_point: None,
    )
    monkeypatch.setattr(
        joint_pass_module,
        "_default_joint_solver_command",
        lambda: (_ for _ in ()).throw(
            RuntimeError(
                "Joint tiling schedule solver requires the checked-out 'joint_solver' "
                "submodule CLI. Run 'git submodule update --init --recursive'."
            )
        ),
    )

    with pytest.raises(RuntimeError, match="git submodule update --init --recursive"):
        compiler.compile(
            "model.onnx",
            str(tmp_path),
            metadata={"enable_joint_tiling_schedule_contract": True},
        )


def test_joint_tiling_schedule_infeasible_failure_surfaces_standardized_category(
    monkeypatch,
    tmp_path,
):
    compiler = Compiler(target="x86", opt_level=3)
    compiler.frontend = SimpleNamespace(load=lambda _: _make_gemm_graph())
    compiler.backend = _CapturingBackend()
    monkeypatch.setattr(
        compiler,
        "_write_output",
        lambda artifacts, output_dir, entry_point: None,
    )

    with pytest.raises(RuntimeError) as exc_info:
        compiler.compile(
            "model.onnx",
            str(tmp_path),
            metadata={
                "enable_joint_tiling_schedule_contract": True,
                "joint_tiling_schedule_solver_command": _fake_joint_solver_command(
                    "infeasible"
                ),
            },
        )

    assert getattr(exc_info.value, "error_category", None) == "solver_reported_infeasible"
    assert "solver_reported_infeasible" in str(exc_info.value)


def test_joint_tiling_schedule_problem_failure_preserves_structured_category(
    monkeypatch,
    tmp_path,
):
    from nnc_py.ir.joint_tiling_schedule import (
        JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
        JointFailure,
        JointFailureCategory,
        JointFailureStatus,
    )
    from nnc_py.passes import joint_tiling_schedule as joint_pass_module

    compiler = Compiler(target="x86", opt_level=3)
    compiler.frontend = SimpleNamespace(load=lambda _: _make_gemm_graph())
    compiler.backend = _CapturingBackend()
    monkeypatch.setattr(
        compiler,
        "_write_output",
        lambda artifacts, output_dir, entry_point: None,
    )
    monkeypatch.setattr(
        joint_pass_module,
        "validate_joint_problem",
        lambda problem: JointFailure(
            schema_version=JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
            status=JointFailureStatus.INVALID_PROBLEM,
            error_category=JointFailureCategory.INVALID_SOLUTION,
            diagnostics={"reason": "synthetic_invalid_problem"},
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        compiler.compile(
            "model.onnx",
            str(tmp_path),
            metadata={"enable_joint_tiling_schedule_contract": True},
        )

    assert getattr(exc_info.value, "error_category", None) == "invalid_solution"
    assert getattr(exc_info.value, "failure_status", None) == "invalid_problem"
    assert "synthetic_invalid_problem" in str(exc_info.value)


def test_o3_default_scheduler_failure_does_not_fallback(monkeypatch, tmp_path):
    backend = _CapturingBackend()
    compiler = Compiler(target="x86", opt_level=3)
    compiler.frontend = SimpleNamespace(load=lambda _: _make_gemm_graph())
    compiler.backend = backend
    monkeypatch.setattr(
        compiler,
        "_write_output",
        lambda artifacts, output_dir, entry_point: None,
    )

    class _FailingScheduledPass:
        @property
        def name(self) -> str:
            return "FailingScheduledPass"

        def run(self, ctx):
            from nnc_py.ir.pipeline_schedule import PipelineScheduleResult, set_pipeline_schedule_result

            set_pipeline_schedule_result(
                ctx,
                PipelineScheduleResult(
                    feasible=False,
                    solver_name="test",
                    diagnostics={"strategy": "serial", "reason": "synthetic_failure"},
                ),
            )

    monkeypatch.setattr(
        PassManager,
        "get_scheduled_o3_passes",
        classmethod(lambda cls: [_FailingScheduledPass()]),
    )

    with pytest.raises(RuntimeError, match="scheduled pipeline path") as exc_info:
        compiler.compile("model.onnx", str(tmp_path))

    assert "disable-pipeline-scheduler" not in str(exc_info.value)
