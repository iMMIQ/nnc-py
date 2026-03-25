"""Integration tests for the O3 pipeline scheduling path and fallbacks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nnc_py.codegen.base import CodeGenResult
from nnc_py.compiler import Compiler
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType
from nnc_py.passes.base import PassManager


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
    metadata: dict[str, object] | None = None,
    cost_model_cli_command: list[str] | None = None,
    max_memory: str | None = None,
    enable_pipeline_scheduler: bool | None = None,
):
    backend = _CapturingBackend()
    compiler = Compiler(
        target="x86",
        opt_level=3,
        cost_model_cli_command=cost_model_cli_command,
    )
    compiler.frontend = SimpleNamespace(load=lambda _: _make_gemm_graph())
    compiler.backend = backend
    monkeypatch.setattr(
        compiler,
        "_write_output",
        lambda artifacts, output_dir, entry_point: None,
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


def test_o3_default_pass_order_uses_tile_aware_v3_without_scheduler_and_v4():
    names = [pass_obj.__class__.__name__ for pass_obj in PassManager.get_default_passes(3)]

    assert "ScheduleAnalysisPass" in names
    assert "LayoutPlanningPass" in names
    assert "TiledLoweringPass" in names
    assert "MemoryPlanningPassV3" in names
    assert "PipelineStepLoweringPass" not in names
    assert "PipelineSchedulingPass" not in names
    assert "MemoryPlanningPassV4" not in names
    assert names.index("PrepackLoweringPass") < names.index("DominatorFusionPass")
    assert names.index("DominatorFusionPass") < names.index("ScheduleAnalysisPass")
    assert names.index("ScheduleAnalysisPass") < names.index("LayoutPlanningPass")
    assert names.index("LayoutPlanningPass") < names.index("TiledLoweringPass")
    assert names.index("TiledLoweringPass") < names.index("LivenessAnalysisPass")
    assert names.index("LivenessAnalysisPass") < names.index("MemoryPlanningPassV3")
    assert names.index("MemoryPlanningPassV3") < names.index("SpillAnalysisPass")


def test_o3_scheduled_pass_order_requires_explicit_helper():
    names = [
        pass_obj.__class__.__name__
        for pass_obj in PassManager.get_scheduled_o3_passes()
    ]

    assert names.index("ScheduleAnalysisPass") < names.index("LayoutPlanningPass")
    assert names.index("LayoutPlanningPass") < names.index("TiledLoweringPass")
    assert names.index("TiledLoweringPass") < names.index("PipelineStepLoweringPass")
    assert names.index("PipelineStepLoweringPass") < names.index("PipelineSchedulingPass")
    assert names.index("PipelineSchedulingPass") < names.index("LivenessAnalysisPass")
    assert names.index("LivenessAnalysisPass") < names.index("MemoryPlanningPassV4")
    assert names.index("MemoryPlanningPassV4") < names.index("SpillAnalysisPass")


def test_o3_compile_defaults_to_conservative_legacy_compatible_path(monkeypatch, tmp_path):
    command = ["external-cost-model", "--stdio"]

    ctx = _compile_graph(
        monkeypatch,
        tmp_path,
        cost_model_cli_command=command,
    )

    assert ctx.metadata["cost_model_cli_command"] == command
    assert ctx.metadata["pipeline_scheduler_enabled"] is False
    assert ctx.metadata["pipeline_scheduler_fallback"] == "legacy_o3_default"
    assert ctx.pipeline_schedule_problem is None
    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.feasible is False
    assert ctx.pipeline_schedule_result.diagnostics["strategy"] == "serial"
    assert ctx.pipeline_schedule_result.diagnostics["reason"] == "pipeline_scheduler_default_off"
    assert "gemm0" in ctx.metadata.get("node_execution_plans", {})
    assert ctx.metadata["memory_allocation_plan"].strategy_name == "tile_regions_v3"


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
    assert ctx.metadata["memory_allocation_plan"].strategy_name == "schedule_time_v4"


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


def test_compiler_explicit_enable_uses_scheduled_o3_helper(
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
        PassManager, "get_default_passes", classmethod(lambda cls, opt_level: None)
    )
    monkeypatch.setattr(
        PassManager,
        "get_scheduled_o3_passes",
        classmethod(lambda cls: []),
    )

    compiler.compile(
        "model.onnx",
        str(tmp_path),
        enable_pipeline_scheduler=True,
    )

    assert backend.ctx is not None
