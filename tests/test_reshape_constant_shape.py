"""Test for Reshape with Constant node providing shape."""

import tempfile
import subprocess
from pathlib import Path

import onnx
from onnx import helper

from nnc_py import Compiler


def test_reshape_with_constant_shape():
    """Test Reshape where shape comes from a Constant node output.

    This reproduces the issue where nnc_reshape uses tensor_Constant_output_0_shape
    but the actual variable is tensor_Constant_output_0_data.
    """
    # Create a Constant node that provides the shape
    # Shape: [batch, height, width, channels] -> [batch, -1] for flattening
    shape_const = helper.make_tensor(
        "shape_const",
        onnx.TensorProto.INT64,
        [2],
        [1, -1]  # Flatten to (batch, -1)
    )

    # Constant node with output name "Constant_output_0"
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["Constant_output_0"],
        value=shape_const
    )

    # Input: [1, 2, 3, 4] -> Output: [1, 24]
    input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2, 3, 4])
    output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 24])

    # Reshape using the Constant node's output as shape
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["input", "Constant_output_0"],
        outputs=["output"]
    )

    graph = helper.make_graph(
        [const_node, reshape_node],
        "reshape_with_constant_shape",
        [input_val],
        [output_val]
    )

    model = helper.make_model(graph)

    # Compile
    tmpdir = tempfile.mkdtemp()
    onnx_path = f"{tmpdir}/model.onnx"
    onnx.save(model, onnx_path)

    output_dir = f"{tmpdir}/build"
    compiler = Compiler(target="x86", opt_level=0)
    compiler.compile(onnx_path, output_dir)

    # Check generated model.c
    model_c = Path(output_dir) / "model.c"
    model_c_content = model_c.read_text()

    print("=== Generated model.c ===")
    print(model_c_content)
    print()

    # Check constants.c
    constants_c = Path(output_dir) / "constants.c"
    if constants_c.exists():
        constants_c_content = constants_c.read_text()
        print("=== Generated constants.c ===")
        print(constants_c_content)
        print()

    # Check model.h
    model_h = Path(output_dir) / "model.h"
    model_h_content = model_h.read_text()
    print("=== Generated model.h ===")
    print(model_h_content)
    print()

    # Try to build
    runtime_dir = Path(__file__).parent.parent / "runtime"
    makefile = Path(output_dir) / "Makefile"
    makefile_content = makefile.read_text()
    makefile_content = makefile_content.replace(
        "NNC_RUNTIME ?= ../../runtime",
        f"NNC_RUNTIME = {runtime_dir}"
    )
    makefile.write_text(makefile_content)

    result = subprocess.run(
        ["make", "clean"],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=30
    )
    result = subprocess.run(
        ["make"],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0:
        print("=== Build Failed ===")
        print("STDERR:", result.stderr)
        print()
        # Check if the issue is with tensor_Constant_output_0_shape
        if "tensor_Constant_output_0_shape" in result.stderr or "tensor_Constant_output_0_shape" in model_c_content:
            print("BUG CONFIRMED: Using tensor_Constant_output_0_shape")
            print("Expected: tensor_Constant_output_0_data (the actual INT64 array)")
            return False
        return False

    print("=== Build Succeeded ===")
    return True


if __name__ == "__main__":
    test_reshape_with_constant_shape()
