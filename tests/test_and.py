"""Tests for And operator support."""

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


class TestAndOpSupport:
    """Test And operator integration."""

    def test_op_type_exists(self):
        """Test that AND OpType is defined."""
        assert OpType.AND is not None
        assert OpType.AND.value == "And"

    def test_create_and_node(self):
        """Test creating an And node."""
        node = Node(
            op_type=OpType.AND,
            name="and_1",
            inputs=["input_a", "input_b"],
            outputs=["output"],
        )

        assert node.op_type == OpType.AND
        assert node.name == "and_1"
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1

    def test_and_is_computational(self):
        """Test that And is considered a computational op."""
        node = Node(
            op_type=OpType.AND,
            name="and_test",
            inputs=["input_a", "input_b"],
            outputs=["output"],
        )
        assert node.is_computational()

    def test_load_onnx_with_and(self):
        """Test loading ONNX model with And node."""
        # Create a simple ONNX model with an And node
        # Compare two [2, 2] boolean tensors
        and_node = helper.make_node(
            "And",
            inputs=["input_a", "input_b"],
            outputs=["output"],
            name="and_node",
        )

        input_a_val = helper.make_tensor_value_info("input_a", onnx.TensorProto.BOOL, [2, 2])
        input_b_val = helper.make_tensor_value_info("input_b", onnx.TensorProto.BOOL, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, [2, 2])

        graph = helper.make_graph(
            [and_node],
            "and_model",
            [input_a_val, input_b_val],
            [output_val]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

        # Save to temp file and load with frontend
        tmpdir = tempfile.mkdtemp()
        onnx_path = f"{tmpdir}/and_model.onnx"
        onnx.save(model, onnx_path)

        try:
            frontend = ONNXFrontend()
            ir_graph = frontend.load(onnx_path)

            # Verify graph structure
            assert "input_a" in ir_graph.tensors
            assert "input_b" in ir_graph.tensors
            assert "output" in ir_graph.tensors
            assert "and_node" in ir_graph.nodes

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

    def test_and_with_broadcast(self):
        """Test And with broadcasting."""
        # AND [2, 3] with [3] (broadcast second input)
        and_node = helper.make_node(
            "And",
            inputs=["input_a", "input_b"],
            outputs=["output"],
        )

        input_a_val = helper.make_tensor_value_info("input_a", onnx.TensorProto.BOOL, [2, 3])
        input_b_val = helper.make_tensor_value_info("input_b", onnx.TensorProto.BOOL, [3])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, [2, 3])

        graph = helper.make_graph(
            [and_node],
            "and_broadcast_model",
            [input_a_val, input_b_val],
            [output_val]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

        # Save to temp file and load
        tmpdir = tempfile.mkdtemp()
        onnx_path = f"{tmpdir}/and_broadcast.onnx"
        onnx.save(model, onnx_path)

        try:
            frontend = ONNXFrontend()
            ir_graph = frontend.load(onnx_path)

            output_tensor = ir_graph.tensors["output"]
            assert output_tensor.shape.dims == [2, 3]
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_and_code_emission(self):
        """Test that And node generates correct C code."""
        and_node = helper.make_node(
            "And",
            inputs=["input_a", "input_b"],
            outputs=["output"],
            name="my_and",
        )

        input_a_val = helper.make_tensor_value_info("input_a", onnx.TensorProto.BOOL, [2, 2])
        input_b_val = helper.make_tensor_value_info("input_b", onnx.TensorProto.BOOL, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, [2, 2])

        graph = helper.make_graph(
            [and_node],
            "and_code_test",
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

        # Should contain nnc_and call
        assert "nnc_and" in model_c_content

        # Clean up
        import shutil
        shutil.rmtree(tmpdir)


def test_and_compilation_and_execution():
    """Test And compilation and execution."""
    print("\n=== Testing And Compilation and Execution ===")

    # Create model: logical AND of two [2, 2] tensors
    and_node = helper.make_node(
        "And",
        inputs=["input_a", "input_b"],
        outputs=["output"],
    )

    input_a_val = helper.make_tensor_value_info("input_a", onnx.TensorProto.BOOL, [2, 2])
    input_b_val = helper.make_tensor_value_info("input_b", onnx.TensorProto.BOOL, [2, 2])
    output_val = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, [2, 2])

    graph = helper.make_graph(
        [and_node],
        "and_exec_test",
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
    print("model.c contains nnc_and:", "nnc_and" in model_c_content)

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
    test_and_compilation_and_execution()
