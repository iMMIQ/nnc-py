from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from nnc_py import Compiler
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


def _make_scheduled_conv_add_relu_model() -> onnx.ModelProto:
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
        "scheduled_conv_add_relu_tile_codegen",
        [input_info, residual_info],
        [output_info],
        [conv_weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_scheduled_conv_downsample_add_relu_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 96, 96])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 48, 48])

    conv_weight = helper.make_tensor(
        "conv_weight",
        TensorProto.FLOAT,
        [32, 16, 3, 3],
        (np.arange(32 * 16 * 3 * 3, dtype=np.float32) / 4096.0).reshape(-1).tolist(),
    )
    skip_weight = helper.make_tensor(
        "skip_weight",
        TensorProto.FLOAT,
        [32, 16, 1, 1],
        (np.arange(32 * 16, dtype=np.float32) / 1024.0).reshape(-1).tolist(),
    )

    main_conv = helper.make_node(
        "Conv",
        inputs=["input", "conv_weight"],
        outputs=["main_out"],
        name="main_conv",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
    )
    skip_conv = helper.make_node(
        "Conv",
        inputs=["input", "skip_weight"],
        outputs=["skip_out"],
        name="skip_conv",
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[2, 2],
    )
    add = helper.make_node(
        "Add",
        inputs=["main_out", "skip_out"],
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
        [main_conv, skip_conv, add, relu],
        "scheduled_conv_downsample_add_relu_tile_codegen",
        [input_info],
        [output_info],
        [conv_weight, skip_weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_scheduled_global_avgpool_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 14, 14])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 1, 1])

    avgpool = helper.make_node(
        "GlobalAveragePool",
        inputs=["input"],
        outputs=["output"],
        name="avgpool0",
    )

    graph = helper.make_graph(
        [avgpool],
        "scheduled_global_avgpool_tile_codegen",
        [input_info],
        [output_info],
        [],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_scheduled_padded_maxpool_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 15, 15])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 8, 8])

    maxpool = helper.make_node(
        "MaxPool",
        inputs=["input"],
        outputs=["output"],
        name="pool0",
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
    )

    graph = helper.make_graph(
        [maxpool],
        "scheduled_padded_maxpool_tile_codegen",
        [input_info],
        [output_info],
        [],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_scheduled_gemm_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 128])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])

    weight = helper.make_tensor(
        "fc_weight",
        TensorProto.FLOAT,
        [64, 128],
        (np.arange(64 * 128, dtype=np.float32) / 4096.0).reshape(-1).tolist(),
    )
    bias = helper.make_tensor(
        "fc_bias",
        TensorProto.FLOAT,
        [64],
        (np.arange(64, dtype=np.float32) / 1024.0).reshape(-1).tolist(),
    )

    gemm = helper.make_node(
        "Gemm",
        inputs=["input", "fc_weight", "fc_bias"],
        outputs=["output"],
        name="fc0",
        transB=1,
    )

    graph = helper.make_graph(
        [gemm],
        "scheduled_gemm_tile_codegen",
        [input_info],
        [output_info],
        [weight, bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _build_scheduled_model_codegen_context(model: onnx.ModelProto, *, max_memory: int) -> CompileContext:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.onnx"
        onnx.save(model, model_path)

        graph = ONNXFrontend().load(str(model_path))
        ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
        ctx.metadata["pipeline_scheduler_enabled"] = True
        ctx.metadata["max_memory"] = max_memory

        pass_manager = PassManager()
        for pass_obj in PassManager.get_scheduled_o3_passes():
            pass_manager.register(pass_obj)
        pass_manager.run(ctx)
        return ctx


def _build_scheduled_codegen_context(*, max_memory: int) -> CompileContext:
    return _build_scheduled_model_codegen_context(
        _make_scheduled_conv_maxpool_model(),
        max_memory=max_memory,
    )


def _build_scheduled_conv_relu_codegen_context(*, max_memory: int) -> CompileContext:
    return _build_scheduled_model_codegen_context(
        _make_scheduled_conv_relu_model(),
        max_memory=max_memory,
    )


def _build_scheduled_conv_add_relu_codegen_context(*, max_memory: int) -> CompileContext:
    return _build_scheduled_model_codegen_context(
        _make_scheduled_conv_add_relu_model(),
        max_memory=max_memory,
    )


def _build_scheduled_conv_downsample_add_relu_codegen_context(*, max_memory: int) -> CompileContext:
    return _build_scheduled_model_codegen_context(
        _make_scheduled_conv_downsample_add_relu_model(),
        max_memory=max_memory,
    )


def _build_scheduled_global_avgpool_codegen_context(*, max_memory: int) -> CompileContext:
    return _build_scheduled_model_codegen_context(
        _make_scheduled_global_avgpool_model(),
        max_memory=max_memory,
    )


def _build_scheduled_padded_maxpool_codegen_context(*, max_memory: int) -> CompileContext:
    return _build_scheduled_model_codegen_context(
        _make_scheduled_padded_maxpool_model(),
        max_memory=max_memory,
    )


def _build_scheduled_gemm_codegen_context(*, max_memory: int) -> CompileContext:
    return _build_scheduled_model_codegen_context(
        _make_scheduled_gemm_model(),
        max_memory=max_memory,
    )


def _buffer_sizes_by_symbol(model_c: str) -> dict[str, int]:
    matches = re.findall(
        r"static unsigned char ([A-Za-z0-9_]+)_buffer\[(\d+)\];",
        model_c,
    )
    return {symbol: int(size) for symbol, size in matches}


def _declared_symbol_names(model_c: str, pattern: str) -> list[str]:
    return re.findall(pattern, model_c)


def _assert_declared_symbols_are_referenced(model_c: str, symbols: list[str]) -> None:
    unused = [symbol for symbol in symbols if model_c.count(symbol) <= 1]
    assert unused == []


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


def test_scheduled_parallel_codegen_streams_conv_add_relu_without_home_fallback():
    ctx = _build_scheduled_conv_add_relu_codegen_context(max_memory=1024 * 1024)
    artifacts = X86Backend().generate(ctx)

    model_c = _artifact_text(artifacts, "model.c")
    tensors_c = _artifact_text(artifacts, "tensors.c")

    assert "nnc_pipeline_tile_stream_" in model_c
    assert "nnc_pipeline_step_fused_add_relu_1_compute" in model_c
    assert "node_fused_add_relu_1();" not in model_c
    assert "scheduled home execution" not in model_c
    assert "Detached tensor buffer: conv_out" not in tensors_c
    assert "Detached tensor buffer: add_out" not in tensors_c


def test_scheduled_parallel_codegen_streams_downsample_residual_add_group():
    ctx = _build_scheduled_conv_downsample_add_relu_codegen_context(max_memory=1024 * 1024)
    artifacts = X86Backend().generate(ctx)

    model_c = _artifact_text(artifacts, "model.c")

    assert "nnc_pipeline_tile_stream_" in model_c
    assert "nnc_pipeline_step_fused_add_relu_1_compute" in model_c
    assert "node_fused_add_relu_1();" not in model_c
    assert "_nnc_pipeline_stage_nchw_tile(&tensor_skip_out" in model_c


def test_scheduled_parallel_codegen_omits_unused_tile_stream_storage_declarations():
    ctx = _build_scheduled_conv_add_relu_codegen_context(max_memory=1024 * 1024)
    artifacts = X86Backend().generate(ctx)

    model_c = _artifact_text(artifacts, "model.c")

    sram_buffers = _declared_symbol_names(
        model_c,
        r"static unsigned char ([A-Za-z0-9_]+_buffer)\[\d+\];",
    )
    saved_data = _declared_symbol_names(
        model_c,
        r"static void\* ([A-Za-z0-9_]+_saved_data) = NULL;",
    )

    _assert_declared_symbols_are_referenced(model_c, sram_buffers)
    _assert_declared_symbols_are_referenced(model_c, saved_data)


def test_scheduled_parallel_codegen_streams_maxpool_without_home_fallback():
    ctx = _build_scheduled_codegen_context(max_memory=1024 * 1024)
    artifacts = X86Backend().generate(ctx)

    model_c = _artifact_text(artifacts, "model.c")

    assert "scheduled home execution" not in model_c
    assert "nnc_pipeline_tile_stream_pool0" in model_c
    assert "node_pool0();" not in model_c


def test_scheduled_parallel_codegen_streams_padded_maxpool_without_home_fallback():
    ctx = _build_scheduled_padded_maxpool_codegen_context(max_memory=1024 * 1024)
    artifacts = X86Backend().generate(ctx)

    model_c = _artifact_text(artifacts, "model.c")

    assert "scheduled home execution" not in model_c
    assert "nnc_pipeline_tile_stream_pool0" in model_c
    assert "node_pool0();" not in model_c


def test_scheduled_parallel_codegen_streams_global_avgpool_without_home_fallback():
    ctx = _build_scheduled_global_avgpool_codegen_context(max_memory=1024 * 1024)
    artifacts = X86Backend().generate(ctx)

    model_c = _artifact_text(artifacts, "model.c")

    assert "scheduled home execution" not in model_c
    assert "nnc_pipeline_tile_stream_avgpool0" in model_c
    assert "node_avgpool0();" not in model_c


def test_scheduled_parallel_codegen_uses_default_parallel_steps_for_gemm():
    ctx = _build_scheduled_gemm_codegen_context(max_memory=1024 * 1024)
    artifacts = X86Backend().generate(ctx)

    model_c = _artifact_text(artifacts, "model.c")

    assert "scheduled home execution" not in model_c
    assert "nnc_pipeline_step_fc0_dma_in" in model_c
    assert "nnc_pipeline_step_fc0_compute" in model_c
    assert "node_fc0();" in model_c


def _compile_generated_sources_with_runner(tmpdir: Path, runner_name: str) -> Path:
    runtime_dir = Path(__file__).resolve().parents[1] / "runtime"
    runtime_include = runtime_dir / "include"
    runtime_ops = runtime_dir / "x86" / "ops.c"
    cflags = ["-D_GNU_SOURCE", "-std=c11", "-Wall", "-Wextra", "-fno-common", "-fPIC", "-pthread"]
    object_files: list[Path] = []

    for filename in ("model.c", "tensors.c", "constants_loader.c", f"{runner_name}.c"):
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

    exe_path = tmpdir / runner_name
    result = subprocess.run(
        ["gcc", "-o", str(exe_path), *[str(path) for path in object_files], "-lm", "-pthread"],
        capture_output=True,
        text=True,
        cwd=tmpdir,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return exe_path


def _write_minimal_runner(
    tmpdir: Path,
    *,
    runner_name: str,
    input_tensor_symbols: dict[str, str],
    output_tensor_symbol: str,
    load_constants: bool,
) -> None:
    extern_inputs = "\n".join(f"extern Tensor {symbol};" for symbol in input_tensor_symbols.values())
    load_calls = "\n".join(
        f'    if (load_tensor("{input_name}.bin", &{tensor_symbol}) != 0) {{ return 3; }}'
        for input_name, tensor_symbol in input_tensor_symbols.items()
    )
    constants_block = ""
    if load_constants:
        constants_block = """    if (nnc_load_constants("constants.bin") != 0) {
        return 2;
    }
"""
    runner_source = f"""#include <stdio.h>
#include <stdint.h>
#include "model.h"

{extern_inputs}
extern Tensor {output_tensor_symbol};

static int load_tensor(const char *path, Tensor *tensor) {{
    FILE *f = fopen(path, "rb");
    if (f == NULL) {{
        return -1;
    }}
    size_t got = fread(tensor->data, 1, (size_t)tensor->nbytes, f);
    fclose(f);
    return got == (size_t)tensor->nbytes ? 0 : -2;
}}

int main(void) {{
{constants_block}\
{load_calls}
    nnc_run();
    FILE *out = fopen("out.bin", "wb");
    if (out == NULL) {{
        return 4;
    }}
    size_t wrote = fwrite({output_tensor_symbol}.data, 1, (size_t){output_tensor_symbol}.nbytes, out);
    fclose(out);
    return wrote == (size_t){output_tensor_symbol}.nbytes ? 0 : 5;
}}
"""
    (tmpdir / f"{runner_name}.c").write_text(runner_source)


def _run_scheduled_runtime_case(
    model: onnx.ModelProto,
    *,
    input_arrays: dict[str, np.ndarray],
    max_memory: str,
) -> tuple[str, np.ndarray, np.ndarray]:
    ort = pytest.importorskip("onnxruntime")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        model_path = tmpdir_path / "model.onnx"
        onnx.save(model, model_path)

        compiler = Compiler(target="x86", opt_level=3)
        compiler.compile(str(model_path), str(tmpdir_path), max_memory=max_memory)

        model_c = (tmpdir_path / "model.c").read_text()
        _write_minimal_runner(
            tmpdir_path,
            runner_name="min_runner",
            input_tensor_symbols={name: f"tensor_{name}" for name in input_arrays},
            output_tensor_symbol="tensor_output",
            load_constants=(tmpdir_path / "constants_loader.c").exists(),
        )
        exe_path = _compile_generated_sources_with_runner(tmpdir_path, "min_runner")

        for input_name, array in input_arrays.items():
            (tmpdir_path / f"{input_name}.bin").write_bytes(array.astype(np.float32).tobytes())

        result = subprocess.run(
            [str(exe_path)],
            capture_output=True,
            text=True,
            cwd=tmpdir_path,
            timeout=90,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout

        output_info = model.graph.output[0]
        output_shape = tuple(dim.dim_value for dim in output_info.type.tensor_type.shape.dim)
        actual = np.fromfile(tmpdir_path / "out.bin", dtype=np.float32).reshape(output_shape)

        session = ort.InferenceSession(model.SerializeToString())
        expected = session.run(None, input_arrays)[0]
        return model_c, actual, expected


def test_scheduled_conv_add_relu_runtime_matches_onnxruntime_without_home_fallback():
    rng = np.random.default_rng(7)
    input_arrays = {
        "input": rng.standard_normal((1, 16, 96, 96), dtype=np.float32),
        "residual": rng.standard_normal((1, 16, 96, 96), dtype=np.float32),
    }

    model_c, actual, expected = _run_scheduled_runtime_case(
        _make_scheduled_conv_add_relu_model(),
        input_arrays=input_arrays,
        max_memory="1M",
    )

    assert "scheduled home execution" not in model_c
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


def test_scheduled_downsample_residual_runtime_matches_onnxruntime_for_add_group():
    rng = np.random.default_rng(11)
    input_arrays = {
        "input": rng.standard_normal((1, 16, 96, 96), dtype=np.float32),
    }

    model_c, actual, expected = _run_scheduled_runtime_case(
        _make_scheduled_conv_downsample_add_relu_model(),
        input_arrays=input_arrays,
        max_memory="1M",
    )

    assert "node_fused_add_relu_1();" not in model_c
    assert "_nnc_pipeline_stage_nchw_tile(&tensor_skip_out" in model_c
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


def test_scheduled_conv_maxpool_runtime_matches_onnxruntime_without_home_fallback():
    rng = np.random.default_rng(17)
    input_arrays = {
        "input": rng.standard_normal((1, 16, 96, 96), dtype=np.float32),
    }

    model_c, actual, expected = _run_scheduled_runtime_case(
        _make_scheduled_conv_maxpool_model(),
        input_arrays=input_arrays,
        max_memory="1M",
    )

    assert "scheduled home execution" not in model_c
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


def test_scheduled_padded_maxpool_runtime_matches_onnxruntime_without_home_fallback():
    rng = np.random.default_rng(19)
    input_arrays = {
        "input": rng.standard_normal((1, 8, 15, 15), dtype=np.float32),
    }

    model_c, actual, expected = _run_scheduled_runtime_case(
        _make_scheduled_padded_maxpool_model(),
        input_arrays=input_arrays,
        max_memory="1M",
    )

    assert "scheduled home execution" not in model_c
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_scheduled_global_avgpool_runtime_matches_onnxruntime_without_home_fallback():
    rng = np.random.default_rng(23)
    input_arrays = {
        "input": rng.standard_normal((1, 16, 14, 14), dtype=np.float32),
    }

    model_c, actual, expected = _run_scheduled_runtime_case(
        _make_scheduled_global_avgpool_model(),
        input_arrays=input_arrays,
        max_memory="1M",
    )

    assert "scheduled home execution" not in model_c
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_scheduled_gemm_runtime_matches_onnxruntime_without_home_fallback():
    rng = np.random.default_rng(29)
    input_arrays = {
        "input": rng.standard_normal((1, 128), dtype=np.float32),
    }

    model_c, actual, expected = _run_scheduled_runtime_case(
        _make_scheduled_gemm_model(),
        input_arrays=input_arrays,
        max_memory="1M",
    )

    assert "scheduled home execution" not in model_c
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
