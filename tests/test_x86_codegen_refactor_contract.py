"""Characterization tests for the x86 codegen refactor seam."""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

from nnc_py.compiler import Compiler
from test_pipeline_scheduler_e2e import _compile_model


def _make_simple_add_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["X", "const"], ["Y"], name="add0"),
        ],
        "serial_contract_model",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2]),
        ],
        [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2]),
        ],
        [
            helper.make_tensor(
                "const",
                TensorProto.FLOAT,
                [2, 2],
                [1.0, 1.0, 1.0, 1.0],
            ),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _read(path: Path) -> str:
    return path.read_text()


def test_x86_backend_serial_contract(tmp_path):
    model = _make_simple_add_model()
    model_path = tmp_path / "serial.onnx"
    output_dir = tmp_path / "build"
    onnx.save(model, model_path)

    compiler = Compiler(
        target="x86",
        opt_level=0,
        enable_constant_folding=False,
    )
    compiler.compile(
        str(model_path),
        str(output_dir),
        entry_point="my_infer",
    )

    expected_files = [
        output_dir / "model.c",
        output_dir / "model.h",
        output_dir / "tensors.c",
        output_dir / "test_runner.c",
    ]
    for path in expected_files:
        assert path.exists(), f"missing expected artifact: {path.name}"

    model_header = _read(output_dir / "model.h")
    model_source = _read(output_dir / "model.c")

    assert "void nnc_run(void);" in model_header
    assert "void my_infer(void);" in model_header
    assert "void nnc_run(void)" in model_source
    assert "void my_infer(void) {" in model_source
    assert "nnc_run();" in model_source


def test_x86_backend_scheduled_contract(tmp_path):
    _, output_dir = _compile_model(
        tmp_path,
        enable_pipeline_scheduler=True,
    )

    model_c = _read(output_dir / "model.c")

    assert (output_dir / "model.c").exists()
    assert "Pipeline schedule summary" in model_c
    assert "schedule_metadata=present" in model_c
    assert (
        "parallel_runtime=enabled" in model_c
        or "parallel_runtime=disabled" in model_c
    )


def test_x86_codegen_package_defaults():
    from nnc_py.codegen.x86_ir import X86CodegenPackage

    package = X86CodegenPackage(mode="serial", entry_point="nnc_run")

    assert package.mode == "serial"
    assert package.entry_point == "nnc_run"
    assert package.files == {}
    assert package.pipeline_summary_lines == []
