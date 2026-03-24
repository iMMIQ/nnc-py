from dataclasses import replace
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from nnc_py.codegen.x86_backend import X86Backend
from nnc_py.ir.context import CompileContext
from nnc_py.ir.execution_plan import LayoutClass
from nnc_py.passes.base import PassManager
from nnc_py.frontend.onnx_loader import ONNXFrontend


def _artifact_text(artifacts, filename: str) -> str:
    return next(file.content for file in artifacts.files if file.filename == filename)


def _make_phase1_conv_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 96, 96])
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
        outputs=["output"],
        name="conv0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )

    graph = helper.make_graph(
        [conv],
        "phase1_tiled_conv_codegen",
        [input_info],
        [output_info],
        [conv_weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_phase1_conv_maxpool_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 96, 96])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 48, 48])

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
    maxpool = helper.make_node(
        "MaxPool",
        inputs=["conv_out"],
        outputs=["output"],
        name="pool0",
        kernel_shape=[2, 2],
        strides=[2, 2],
    )

    graph = helper.make_graph(
        [conv, maxpool],
        "phase1_tiled_conv_maxpool_codegen",
        [input_info],
        [output_info],
        [conv_weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_phase1_conv_add_relu_model() -> onnx.ModelProto:
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
        "phase1_tiled_conv_add_relu_codegen",
        [input_info, residual_info],
        [output_info],
        [conv_weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_phase1_conv_add_relu_escape_model() -> onnx.ModelProto:
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
        "phase1_tiled_conv_add_relu_escape_codegen",
        [input_info, residual_info],
        [output_info, side_info],
        [conv_weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_phase1_gemm_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 512])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1000])

    weight = helper.make_tensor(
        "fc_weight",
        TensorProto.FLOAT,
        [512, 1000],
        (np.arange(512 * 1000, dtype=np.float32) / 65536.0).tolist(),
    )
    bias = helper.make_tensor(
        "fc_bias",
        TensorProto.FLOAT,
        [1000],
        np.zeros(1000, dtype=np.float32).tolist(),
    )

    gemm = helper.make_node(
        "Gemm",
        inputs=["input", "fc_weight", "fc_bias"],
        outputs=["output"],
        name="fc",
        transB=0,
    )

    graph = helper.make_graph(
        [gemm],
        "phase1_tiled_gemm_codegen",
        [input_info],
        [output_info],
        [weight, bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _compile_model_with_real_tiled_plan(
    model: onnx.ModelProto,
    *,
    expected_strategy: str = "tile_regions_v3",
    target_physical_layout: str | None = None,
) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.onnx"
        onnx.save(model, model_path)

        graph = ONNXFrontend().load(str(model_path))
        ctx = CompileContext(graph=graph, target="x86", optimization_level=3)

        pass_manager = PassManager()
        for pass_obj in PassManager.get_default_passes(3):
            pass_manager.register(pass_obj)
        pass_manager.run(ctx)

        alloc_plan = ctx.metadata.get("memory_allocation_plan")
        assert alloc_plan is not None
        assert alloc_plan.strategy_name == expected_strategy
        if expected_strategy == "tile_regions_v3":
            execution_plans = ctx.metadata.get("node_execution_plans", {})
            region_sizes = ctx.metadata.get("node_execution_plan_region_sizes", {})
            assert execution_plans
            assert set(execution_plans).issubset(set(region_sizes))
            if target_physical_layout is not None:
                conv0_plan = execution_plans["conv0"]
                execution_plans["conv0"] = replace(
                    conv0_plan,
                    target_physical_layout=target_physical_layout,
                )

        artifacts = X86Backend().generate(ctx)
        return {artifact.filename: artifact.content for artifact in artifacts.files}


def test_x86_backend_emits_tiled_fast_memory_regions_for_safe_real_plan():
    code = _compile_model_with_real_tiled_plan(_make_phase1_conv_model())

    tensors_c = code["tensors.c"]

    assert "#define NNC_FAST_MEMORY_SIZE" in tensors_c
    assert "#define NNC_TILE_MEMORY_SIZE" in tensors_c
    assert "_nnc_fast_pool +" in tensors_c
    assert "_nnc_memory_pool" not in tensors_c


def test_codegen_emits_supported_tile_wrapper_for_phase1_conv():
    code = _compile_model_with_real_tiled_plan(_make_phase1_conv_model())

    assert "tile-aware wrapper" in code["model.c"]
    assert "node_conv0_body();" in code["model.c"]
    assert ".data = NULL" not in code["tensors.c"]
    assert "_nnc_input_buffer_tensor_input" in code["tensors.c"]
    assert "_nnc_fast_pool +" in code["tensors.c"]


def test_codegen_rejected_real_tiled_graph_falls_back_to_self_consistent_storage():
    code = _compile_model_with_real_tiled_plan(
        _make_phase1_conv_add_relu_escape_model(),
        expected_strategy="cost_aware",
    )

    assert "tile-aware wrapper" not in code["model.c"]
    assert ".data = NULL" not in code["tensors.c"]
    assert "_nnc_memory_pool" in code["tensors.c"]
    assert "_nnc_fast_pool" not in code["tensors.c"]


def test_codegen_emits_supported_tile_wrappers_for_phase1_conv_add_relu_group():
    code = _compile_model_with_real_tiled_plan(_make_phase1_conv_add_relu_model())

    assert code["model.c"].count("tile-aware wrapper") >= 2
    assert "node_conv0_body();" in code["model.c"]
    assert "node_fused_add_relu_1_body();" in code["model.c"]
    assert ".data = NULL" not in code["tensors.c"]
    assert "_nnc_input_buffer_tensor_input" in code["tensors.c"]
    assert "_nnc_input_buffer_tensor_residual" in code["tensors.c"]
    assert "_nnc_fast_pool +" in code["tensors.c"]


def test_codegen_keeps_single_gemm_storage_self_consistent_under_tile_regions_v3():
    code = _compile_model_with_real_tiled_plan(_make_phase1_gemm_model())

    assert "tile-aware wrapper" not in code["model.c"]
    assert "node_fc();" in code["model.c"]
    assert "nnc_gemm" in code["model.c"]
    assert ".data = NULL" not in code["tensors.c"]
    assert "_nnc_memory_pool" in code["tensors.c"]
    assert "_nnc_fast_pool" not in code["tensors.c"]


def test_execution_plan_keeps_generic_blocked_layout_before_physical_mapping():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.onnx"
        onnx.save(_make_phase1_conv_model(), model_path)

        graph = ONNXFrontend().load(str(model_path))
        ctx = CompileContext(graph=graph, target="x86", optimization_level=3)

        pass_manager = PassManager()
        for pass_obj in PassManager.get_default_passes(3):
            pass_manager.register(pass_obj)
        pass_manager.run(ctx)

        plan = ctx.metadata["node_execution_plans"]["conv0"]
        assert plan.layout_class is LayoutClass.BLOCKED_ACTIVATION
        assert plan.layout_class.value == "blocked_activation"
        assert plan.target_physical_layout is None


def test_backend_records_target_physical_layout_mapping_comment():
    code = _compile_model_with_real_tiled_plan(
        _make_phase1_conv_model(),
        target_physical_layout="nchwc16",
    )

    assert "target_physical_layout=nchwc16" in code["model.c"]
    assert "generic_blocked_layout=blocked_activation" in code["model.c"]
