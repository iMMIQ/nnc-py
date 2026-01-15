"""Tests for Equal operator support."""

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


class TestEqualOpSupport:
    """Test Equal operator integration."""

    def test_op_type_exists(self):
        """Test that EQUAL OpType is defined."""
        assert OpType.EQUAL is not None
        assert OpType.EQUAL.value == "Equal"

    def test_create_equal_node(self):
        """Test creating an Equal node."""
        node = Node(
            op_type=OpType.EQUAL,
            name="equal_1",
            inputs=["input_a", "input_b"],
            outputs=["output"],
        )

        assert node.op_type == OpType.EQUAL
        assert node.name == "equal_1"
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1

    def test_equal_is_computational(self):
        """Test that Equal is considered a computational op."""
        node = Node(
            op_type=OpType.EQUAL,
            name="equal_test",
            inputs=["input_a", "input_b"],
            outputs=["output"],
        )
        assert node.is_computational()

    def test_load_onnx_with_equal(self):
        """Test loading ONNX model with Equal node."""
        # Create a simple ONNX model with an Equal node
        # Compare two [2, 2] tensors
        equal_node = helper.make_node(
            "Equal",
            inputs=["input_a", "input_b"],
            outputs=["output"],
            name="equal_node",
        )

        input_a_val = helper.make_tensor_value_info("input_a", onnx.TensorProto.FLOAT, [2, 2])
        input_b_val = helper.make_tensor_value_info("input_b", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, [2, 2])

        graph = helper.make_graph(
            [equal_node],
            "equal_model",
            [input_a_val, input_b_val],
            [output_val]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

        # Save to temp file and load with frontend
        tmpdir = tempfile.mkdtemp()
        onnx_path = f"{tmpdir}/equal_model.onnx"
        onnx.save(model, onnx_path)

        try:
            frontend = ONNXFrontend()
            ir_graph = frontend.load(onnx_path)

            # Verify graph structure
            assert "input_a" in ir_graph.tensors
            assert "input_b" in ir_graph.tensors
            assert "output" in ir_graph.tensors
            assert "equal_node" in ir_graph.nodes

            # Check tensor shapes
            input_a_tensor = ir_graph.tensors["input_a"]
            input_b_tensor = ir_graph.tensors["input_b"]
            output_tensor = ir_graph.tensors["output"]
            assert input_a_tensor.shape.dims == [2, 2]
            assert input_b_tensor.shape.dims == [2, 2]
            assert output_tensor.shape.dims == [2, 2]
            # Output should be boolean
            assert output_tensor.dtype == DataType.BOOL
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_equal_with_broadcast(self):
        """Test Equal with broadcasting."""
        # Compare [2, 3] with [3] (broadcast second input)
        equal_node = helper.make_node(
            "Equal",
            inputs=["input_a", "input_b"],
            outputs=["output"],
        )

        input_a_val = helper.make_tensor_value_info("input_a", onnx.TensorProto.FLOAT, [2, 3])
        input_b_val = helper.make_tensor_value_info("input_b", onnx.TensorProto.FLOAT, [3])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, [2, 3])

        graph = helper.make_graph(
            [equal_node],
            "equal_broadcast_model",
            [input_a_val, input_b_val],
            [output_val]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

        # Save to temp file and load
        tmpdir = tempfile.mkdtemp()
        onnx_path = f"{tmpdir}/equal_broadcast.onnx"
        onnx.save(model, onnx_path)

        try:
            frontend = ONNXFrontend()
            ir_graph = frontend.load(onnx_path)

            output_tensor = ir_graph.tensors["output"]
            assert output_tensor.shape.dims == [2, 3]
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_equal_code_emission(self):
        """Test that Equal node generates correct C code."""
        equal_node = helper.make_node(
            "Equal",
            inputs=["input_a", "input_b"],
            outputs=["output"],
            name="my_equal",
        )

        input_a_val = helper.make_tensor_value_info("input_a", onnx.TensorProto.FLOAT, [2, 2])
        input_b_val = helper.make_tensor_value_info("input_b", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, [2, 2])

        graph = helper.make_graph(
            [equal_node],
            "equal_code_test",
            [input_a_val, input_b_val],
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

        # Should contain nnc_equal call
        assert "nnc_equal" in model_c_content

        # Clean up
        import shutil
        shutil.rmtree(tmpdir)


def test_equal_compilation_and_execution():
    """Test Equal compilation and execution."""
    print("\n=== Testing Equal Compilation and Execution ===")

    # Create model: compare two [2, 2] tensors
    equal_node = helper.make_node(
        "Equal",
        inputs=["input_a", "input_b"],
        outputs=["output"],
    )

    input_a_val = helper.make_tensor_value_info("input_a", onnx.TensorProto.FLOAT, [2, 2])
    input_b_val = helper.make_tensor_value_info("input_b", onnx.TensorProto.FLOAT, [2, 2])
    output_val = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, [2, 2])

    graph = helper.make_graph(
        [equal_node],
        "equal_exec_test",
        [input_a_val, input_b_val],
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
    print("model.c contains nnc_equal:", "nnc_equal" in model_c_content)

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
    test_equal_compilation_and_execution()
