import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from nnc_py.codegen.x86_backend import X86Backend
from nnc_py.frontend.onnx_loader import ONNXFrontend
from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassManager


def _make_conv_add_relu_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 96, 96])
    residual_info = helper.make_tensor_value_info("residual", TensorProto.FLOAT, [1, 16, 96, 96])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 96, 96])

    conv_weight = helper.make_tensor(
        "conv_weight",
        TensorProto.FLOAT,
        [16, 16, 3, 3],
        (np.arange(16 * 16 * 3 * 3, dtype=np.float32) / 2048.0).reshape(-1).tolist(),
    )

    conv = helper.make_node(
        "Conv",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        name="conv0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    add = helper.make_node(
        "Add",
        inputs=["conv_out", "residual"],
        outputs=["add_out"],
        name="add0",
    )
    relu = helper.make_node(
        "Relu",
        inputs=["add_out"],
        outputs=["output"],
        name="relu0",
    )

    graph = helper.make_graph(
        [conv, add, relu],
        "tiled_execution_group_conv_add_relu",
        [input_info, residual_info],
        [output_info],
        [conv_weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_conv_add_relu_with_escape_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 96, 96])
    residual_info = helper.make_tensor_value_info("residual", TensorProto.FLOAT, [1, 16, 96, 96])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 96, 96])
    side_info = helper.make_tensor_value_info("side", TensorProto.FLOAT, [1, 16, 96, 96])

    conv_weight = helper.make_tensor(
        "conv_weight",
        TensorProto.FLOAT,
        [16, 16, 3, 3],
        (np.arange(16 * 16 * 3 * 3, dtype=np.float32) / 2048.0).reshape(-1).tolist(),
    )

    conv = helper.make_node(
        "Conv",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        name="conv0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    add = helper.make_node(
        "Add",
        inputs=["conv_out", "residual"],
        outputs=["add_out"],
        name="add0",
    )
    relu = helper.make_node(
        "Relu",
        inputs=["add_out"],
        outputs=["output"],
        name="relu0",
    )
    sigmoid = helper.make_node(
        "Sigmoid",
        inputs=["conv_out"],
        outputs=["side"],
        name="sigmoid0",
    )

    graph = helper.make_graph(
        [conv, add, relu, sigmoid],
        "tiled_execution_group_escape",
        [input_info, residual_info],
        [output_info, side_info],
        [conv_weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _compile_ctx(model: onnx.ModelProto) -> CompileContext:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.onnx"
        onnx.save(model, model_path)

        graph = ONNXFrontend().load(str(model_path))
        ctx = CompileContext(graph=graph, target="x86", optimization_level=3)

        pass_manager = PassManager()
        for pass_obj in PassManager.get_default_passes(3):
            pass_manager.register(pass_obj)
        pass_manager.run(ctx)
        return ctx


def test_tile_execution_group_allows_conv_add_relu_coverage_for_memory_planning_v3():
    ctx = _compile_ctx(_make_conv_add_relu_model())

    plan = ctx.metadata.get("memory_allocation_plan")
    assert plan is not None
    assert plan.strategy_name == "tile_regions_v3"

    runtime_plan = X86Backend()._get_tile_aware_runtime_plan(ctx, plan)
    assert set(runtime_plan["wrapper_nodes"]) == {"conv0", "fused_add_relu_1"}
    assert runtime_plan["tensor_bindings"]["conv_out"]["kind"] == "fast_pool"
    assert runtime_plan["tensor_bindings"]["output"]["kind"] == "fast_pool"


def test_tile_execution_group_rejects_conv_add_relu_when_intermediate_tensor_escapes():
    ctx = _compile_ctx(_make_conv_add_relu_with_escape_model())

    plan = ctx.metadata.get("memory_allocation_plan")
    assert plan is not None
    assert plan.strategy_name == "cost_aware"

    runtime_plan = X86Backend()._get_tile_aware_runtime_plan(ctx, plan)
    assert runtime_plan == {}
