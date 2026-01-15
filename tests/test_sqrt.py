"""Tests for Sqrt operator support."""

import tempfile
import subprocess
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

from nnc_py.frontend.onnx_loader import ONNXFrontend
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType

from nnc_py import Compiler


class TestSqrtOpSupport:
    """Test Sqrt operator integration."""

    def test_op_type_exists(self):
        """Test that SQRT OpType is defined."""
        assert OpType.SQRT is not None
        assert OpType.SQRT.value == "Sqrt"

    def test_create_sqrt_node(self):
        """Test creating a Sqrt node."""
        node = Node(
            op_type=OpType.SQRT,
            name="sqrt_1",
            inputs=["input"],
            outputs=["output"],
        )

        assert node.op_type == OpType.SQRT
        assert node.name == "sqrt_1"
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1

    def test_sqrt_is_computational(self):
        """Test that Sqrt is considered a computational op."""
        node = Node(
            op_type=OpType.SQRT,
            name="sqrt_test",
            inputs=["input"],
            outputs=["output"],
        )
        assert node.is_computational()

    def test_load_onnx_with_sqrt(self):
        """Test loading ONNX model with Sqrt node."""
        # Create a simple ONNX model with a Sqrt node
        sqrt_node = helper.make_node(
            "Sqrt",
            inputs=["input"],
            outputs=["output"],
            name="sqrt_node",
        )

        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 3])

        graph = helper.make_graph(
            [sqrt_node],
            "sqrt_model",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

        # Save to temp file and load with frontend
        tmpdir = tempfile.mkdtemp()
        onnx_path = f"{tmpdir}/sqrt_model.onnx"
        onnx.save(model, onnx_path)

        try:
            frontend = ONNXFrontend()
            ir_graph = frontend.load(onnx_path)

            # Verify graph structure
            assert "input" in ir_graph.tensors
            assert "output" in ir_graph.tensors
            assert "sqrt_node" in ir_graph.nodes

            # Check tensor shapes
            input_tensor = ir_graph.tensors["input"]
            output_tensor = ir_graph.tensors["output"]
            assert input_tensor.shape.dims == [2, 3]
            assert output_tensor.shape.dims == [2, 3]
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_sqrt_3d_tensor(self):
        """Test Sqrt with 3D tensor."""
        sqrt_node = helper.make_node(
            "Sqrt",
            inputs=["input"],
            outputs=["output"],
        )

        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3, 4])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 3, 4])

        graph = helper.make_graph(
            [sqrt_node],
            "sqrt_3d_model",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

        # Save to temp file and load
        tmpdir = tempfile.mkdtemp()
        onnx_path = f"{tmpdir}/sqrt_3d.onnx"
        onnx.save(model, onnx_path)

        try:
            frontend = ONNXFrontend()
            ir_graph = frontend.load(onnx_path)

            output_tensor = ir_graph.tensors["output"]
            assert output_tensor.shape.dims == [2, 3, 4]
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_sqrt_code_emission(self):
        """Test that Sqrt node generates correct C code."""
        sqrt_node = helper.make_node(
            "Sqrt",
            inputs=["input"],
            outputs=["output"],
            name="my_sqrt",
        )

        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

        graph = helper.make_graph(
            [sqrt_node],
            "sqrt_code_test",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

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

        # Should contain nnc_sqrt call
        assert "nnc_sqrt" in model_c_content

        # Clean up
        import shutil
        shutil.rmtree(tmpdir)


def test_sqrt_compilation_and_execution():
    """Test Sqrt compilation and execution."""
    print("\n=== Testing Sqrt Compilation and Execution ===")

    # Create model: sqrt of [2, 2] tensor
    sqrt_node = helper.make_node(
        "Sqrt",
        inputs=["input"],
        outputs=["output"],
    )

    input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
    output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

    graph = helper.make_graph(
        [sqrt_node],
        "sqrt_exec_test",
        [input_val],
        [output_val]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

    # Compile
    tmpdir = tempfile.mkdtemp()
    onnx_path = f"{tmpdir}/model.onnx"
    onnx.save(model, onnx_path)

    output_dir = f"{tmpdir}/build"
    compiler = Compiler(target="x86", opt_level=0)
    compiler.compile(onnx_path, output_dir)

    print("Compilation successful!")

    # Check generated files
    model_c = Path(output_dir) / "model.c"
    assert model_c.exists()

    model_c_content = model_c.read_text()
    print("model.c contains nnc_sqrt:", "nnc_sqrt" in model_c_content)

    # Try to build
    runtime_dir = Path(__file__).parent.parent / "runtime"
    makefile = Path(output_dir) / "Makefile"
    makefile_content = makefile.read_text()
    makefile_content = makefile_content.replace(
        "NNC_RUNTIME ?= ../../runtime",
        f"NNC_RUNTIME = {runtime_dir}"
    )
    # Add address sanitizer
    makefile_content = makefile_content.replace(
        "CFLAGS = -std=c11 -O2",
        "CFLAGS = -std=c11 -O2 -g -fsanitize=address"
    )
    makefile.write_text(makefile_content)

    subprocess.run(["make", "clean"], cwd=output_dir, capture_output=True)
    result = subprocess.run(["make"], cwd=output_dir, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print("Build failed!")
        print("STDERR:", result.stderr)
        return False

    print("Build successful!")

    # Run the executable
    exe_path = Path(output_dir) / "model"
    result = subprocess.run([str(exe_path)], cwd=output_dir, capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        print(f"Execution failed with exit code {result.returncode}!")
        if result.stderr:
            print("STDERR:", result.stderr)
        return False

    print("Execution successful!")

    # Clean up
    import shutil
    shutil.rmtree(tmpdir)

    return True


if __name__ == "__main__":
    test_sqrt_compilation_and_execution()
