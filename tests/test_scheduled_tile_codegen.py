from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from nnc_py.codegen.x86_backend import X86Backend
from nnc_py.frontend.onnx_loader import ONNXFrontend
from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassManager


def _artifact_text(artifacts, filename: str) -> str:
    return next(file.content for file in artifacts.files if file.filename == filename)


def _make_scheduled_conv_maxpool_model() -> onnx.ModelProto:
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
        "scheduled_conv_maxpool_tile_codegen",
        [input_info],
        [output_info],
        [conv_weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_scheduled_conv_relu_model() -> onnx.ModelProto:
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
        outputs=["mid"],
        name="conv0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    relu = helper.make_node(
        "Relu",
        inputs=["mid"],
        outputs=["output"],
        name="relu0",
    )

    graph = helper.make_graph(
        [conv, relu],
        "scheduled_conv_relu_tile_codegen",
        [input_info],
        [output_info],
        [conv_weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _build_scheduled_codegen_context(*, max_memory: int) -> CompileContext:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.onnx"
        onnx.save(_make_scheduled_conv_maxpool_model(), model_path)

        graph = ONNXFrontend().load(str(model_path))
        ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
        ctx.metadata["pipeline_scheduler_enabled"] = True
        ctx.metadata["max_memory"] = max_memory

        pass_manager = PassManager()
        for pass_obj in PassManager.get_scheduled_o3_passes():
            pass_manager.register(pass_obj)
        pass_manager.run(ctx)
        return ctx


def _build_scheduled_conv_relu_codegen_context(*, max_memory: int) -> CompileContext:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.onnx"
        onnx.save(_make_scheduled_conv_relu_model(), model_path)

        graph = ONNXFrontend().load(str(model_path))
        ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
        ctx.metadata["pipeline_scheduler_enabled"] = True
        ctx.metadata["max_memory"] = max_memory

        pass_manager = PassManager()
        for pass_obj in PassManager.get_scheduled_o3_passes():
            pass_manager.register(pass_obj)
        pass_manager.run(ctx)
        return ctx


def _buffer_sizes_by_symbol(model_c: str) -> dict[str, int]:
    matches = re.findall(
        r"static unsigned char ([A-Za-z0-9_]+)_buffer\[(\d+)\];",
        model_c,
    )
    return {symbol: int(size) for symbol, size in matches}


def test_scheduled_parallel_codegen_keeps_tiled_buffers_within_scheduled_value_sizes():
    ctx = _build_scheduled_codegen_context(max_memory=1024 * 1024)
    artifacts = X86Backend().generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")

    scheduled_values = {value.name: value for value in ctx.pipeline_schedule_problem.scheduled_values}
    backend = X86Backend()
    buffer_sizes = _buffer_sizes_by_symbol(model_c)

    expected_upper_bounds = {
        backend._parallel_value_storage_name("sram|node|5:conv0|tensor|5:input"): scheduled_values[
            "sram|node|5:conv0|tensor|5:input"
        ].size_bytes,
        backend._parallel_value_storage_name("sram|node|5:conv0|tensor|8:conv_out"): scheduled_values[
            "sram|node|5:conv0|tensor|8:conv_out"
        ].size_bytes,
        backend._parallel_value_storage_name("sram|node|5:pool0|tensor|6:output"): scheduled_values[
            "sram|node|5:pool0|tensor|6:output"
        ].size_bytes,
    }

    for symbol, upper_bound in expected_upper_bounds.items():
        if symbol in buffer_sizes:
            assert buffer_sizes[symbol] <= upper_bound


def test_scheduled_parallel_codegen_streams_fused_conv_relu_without_home_fallback():
    ctx = _build_scheduled_conv_relu_codegen_context(max_memory=1024 * 1024)
    artifacts = X86Backend().generate(ctx)

    model_c = _artifact_text(artifacts, "model.c")

    assert "scheduled home execution" not in model_c
    assert "nnc_conv_relu(" in model_c


def test_scheduled_parallel_codegen_streams_fused_conv_relu_with_explicit_tile_helper():
    ctx = _build_scheduled_conv_relu_codegen_context(max_memory=1024 * 1024)
    artifacts = X86Backend().generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")

    assert "nnc_pipeline_tile_stream_" in model_c
    assert "nnc_pipeline_step_fused_conv_relu_1_compute" in model_c
