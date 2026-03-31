from __future__ import annotations

import subprocess
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from nnc_py.compiler import Compiler


def _make_two_stage_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["X"], ["relu_out"], name="relu0"),
            helper.make_node("Add", ["relu_out", "const"], ["Y"], name="add0"),
        ],
        "costmodel_runtime_model",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4]),
        ],
        [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4]),
        ],
        [
            helper.make_tensor(
                "const",
                TensorProto.FLOAT,
                [1, 4],
                [1.0, 1.0, 1.0, 1.0],
            ),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def test_generated_makefile_supports_costmodel_runtime(tmp_path):
    model = _make_two_stage_model()
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "build"
    onnx.save(model, model_path)

    compiler = Compiler(target="x86", opt_level=0, enable_constant_folding=False)
    compiler.compile(str(model_path), str(output_dir))

    makefile_text = (output_dir / "Makefile").read_text()

    assert "NNC_RUNTIME_IMPL ?= ops" in makefile_text
    assert "$(NNC_RUNTIME)/x86/$(NNC_RUNTIME_IMPL).c" in makefile_text
    assert "run-costmodel" in makefile_text


def test_costmodel_runtime_prints_execution_order_and_timing(tmp_path):
    model = _make_two_stage_model()
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "build"
    onnx.save(model, model_path)

    compiler = Compiler(target="x86", opt_level=0, enable_constant_folding=False)
    compiler.compile(str(model_path), str(output_dir))

    build = subprocess.run(
        ["make", "NNC_RUNTIME_IMPL=costmodel"],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert build.returncode == 0, build.stderr or build.stdout

    run = subprocess.run(
        [str(output_dir / "model")],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert run.returncode == 0, run.stderr or run.stdout
    assert "[costmodel] #1 op=nnc_relu" in run.stdout
    assert "[costmodel] #2 op=nnc_add" in run.stdout
    assert "[costmodel] summary ops=2" in run.stdout
    assert "total_cycles=" in run.stdout

    runtime_source = Path(__file__).resolve().parents[1] / "runtime" / "x86" / "costmodel.c"
    assert runtime_source.exists()
