"""Tests for memory strategy default selection and overrides."""

from types import SimpleNamespace

import onnx
import pytest
from onnx import TensorProto, helper

from nnc_py.compiler import Compiler
from nnc_py.frontend.onnx_loader import ONNXFrontend
from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassManager
from nnc_py.passes.liveness import LivenessAnalysisPass
from nnc_py.passes.memory_planning import (
    MemoryPlanningPassV2,
    MemoryPlanningPassV3,
    get_memory_allocation_plan,
)
from nnc_py.passes.memory_strategy import AllocationStrategy, StrategyRegistry, get_memory_strategy


def make_context(opt_level: int) -> CompileContext:
    graph = SimpleNamespace(
        nodes=[],
        inputs=[],
        outputs=[],
        constants=[],
    )
    return CompileContext(graph=graph, target="x86", optimization_level=opt_level)


def test_cost_aware_strategy_is_registered_in_production():
    assert StrategyRegistry.is_registered("cost_aware")


def test_o0_defaults_to_basic_strategy():
    ctx = make_context(0)

    strategy = get_memory_strategy(ctx)

    assert strategy.name == "basic"


@pytest.mark.parametrize("opt_level", [1, 2, 3])
def test_higher_optimization_levels_default_to_cost_aware(opt_level: int):
    ctx = make_context(opt_level)

    strategy = get_memory_strategy(ctx)

    assert strategy.name == "cost_aware"
    assert strategy.strategy_type == AllocationStrategy.COST_AWARE


def test_explicit_metadata_strategy_override_is_respected():
    ctx = make_context(3)
    ctx.metadata["memory_strategy"] = "basic"

    strategy = get_memory_strategy(ctx)

    assert strategy.name == "basic"


def test_memory_planning_pass_respects_explicit_strategy_config():
    strategy = MemoryPlanningPassV2()._get_strategy(AllocationStrategy.BASIC, 3)

    assert strategy.name == "basic"


def test_memory_planning_pass_defaults_to_cost_aware_at_higher_optimization_levels():
    strategy = MemoryPlanningPassV2()._get_strategy(None, 3)

    assert strategy.name == "cost_aware"
    assert strategy.strategy_type == AllocationStrategy.COST_AWARE


def test_o1_planning_path_produces_cost_aware_plan_identity(tmp_path):
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["input"], ["hidden"]),
            helper.make_node("Sigmoid", ["hidden"], ["output"]),
        ],
        "chain",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    path = tmp_path / "chain.onnx"
    onnx.save(model, path)

    frontend = ONNXFrontend()
    ir_graph = frontend.load(str(path))
    ctx = CompileContext(graph=ir_graph, target="x86", optimization_level=1)

    LivenessAnalysisPass().run(ctx)
    MemoryPlanningPassV2().run(ctx)
    plan = get_memory_allocation_plan(ctx)

    assert plan is not None
    assert plan.strategy_name == "cost_aware"


def test_memory_planning_v3_falls_back_to_cost_aware_without_tiled_metadata(tmp_path):
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["input"], ["hidden"]),
            helper.make_node("Sigmoid", ["hidden"], ["output"]),
        ],
        "chain",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    path = tmp_path / "chain_v3.onnx"
    onnx.save(model, path)

    frontend = ONNXFrontend()
    ir_graph = frontend.load(str(path))
    ctx = CompileContext(graph=ir_graph, target="x86", optimization_level=3)

    LivenessAnalysisPass().run(ctx)
    MemoryPlanningPassV3().run(ctx)
    plan = get_memory_allocation_plan(ctx)

    assert plan is not None
    assert plan.strategy_name == "cost_aware"


def test_compiler_explicit_memory_strategy_override_is_stored_in_context(monkeypatch, tmp_path):
    captured = {}
    graph = SimpleNamespace(
        nodes=[],
        inputs=[],
        outputs=[],
        constants=[],
    )

    compiler = Compiler(opt_level=3)
    compiler.frontend = SimpleNamespace(load=lambda _: graph)

    def fake_generate(ctx):
        captured["memory_strategy"] = ctx.metadata.get("memory_strategy")
        return SimpleNamespace(files=[], metadata={})

    compiler.backend = SimpleNamespace(generate=fake_generate)
    monkeypatch.setattr(PassManager, "get_default_passes", classmethod(lambda cls, opt_level: []))
    monkeypatch.setattr(compiler, "_write_output", lambda artifacts, output_dir, entry_point: None)

    compiler.compile(
        "model.onnx",
        str(tmp_path),
        memory_strategy="basic",
    )

    assert captured["memory_strategy"] == "basic"
