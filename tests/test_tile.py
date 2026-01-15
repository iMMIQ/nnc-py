"""Tests for Tile operator support."""

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


class TestTileOpSupport:
    """Test Tile operator integration."""

    def test_op_type_exists(self):
        """Test that TILE OpType is defined."""
        assert OpType.TILE is not None
        assert OpType.TILE.value == "Tile"

    def test_create_tile_node(self):
        """Test creating a Tile node."""
        node = Node(
            op_type=OpType.TILE,
            name="tile_1",
            inputs=["input"],
            outputs=["output"],
        )

        assert node.op_type == OpType.TILE
        assert node.name == "tile_1"
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1

    def test_tile_is_not_computational(self):
        """Test that Tile is not considered a computational op."""
        node = Node(
            op_type=OpType.TILE,
            name="tile_test",
            inputs=["input"],
            outputs=["output"],
        )
        assert not node.is_computational()

    def test_load_onnx_with_tile(self):
        """Test loading ONNX model with Tile node."""
        # Create a simple ONNX model with a Tile node
        # Tile a [2, 3] tensor to [4, 6] (repeat 2x along each axis)
        repeats = [2, 2]

        # Constant node for repeats
        repeats_const = helper.make_tensor(
            "repeats_const",
            onnx.TensorProto.INT64,
            [len(repeats)],
            repeats
        )
        repeats_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["repeats"],
            value=repeats_const
        )

        # Tile node
        tile_node = helper.make_node(
            "Tile",
            inputs=["input", "repeats"],
            outputs=["output"],
            name="tile_node",
        )

        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4, 6])

        graph = helper.make_graph(
            [repeats_node, tile_node],
            "tile_model",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

        # Save to temp file and load with frontend
        import tempfile
        import os
        tmpdir = tempfile.mkdtemp()
        onnx_path = os.path.join(tmpdir, "tile_model.onnx")
        onnx.save(model, onnx_path)

        try:
            frontend = ONNXFrontend()
            ir_graph = frontend.load(onnx_path)

            # Verify graph structure
            assert "input" in ir_graph.tensors
            assert "output" in ir_graph.tensors
            assert "tile_node" in ir_graph.nodes

            # Check tensor shapes
            input_tensor = ir_graph.tensors["input"]
            output_tensor = ir_graph.tensors["output"]
            assert input_tensor.shape.dims == [2, 3]
            assert output_tensor.shape.dims == [4, 6]
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_tile_with_different_repeats(self):
        """Test Tile with different repeat values."""
        # Create model: [2, 2] -> [6, 4] (repeat 3x on axis 0, 2x on axis 1)
        repeats = [3, 2]

        repeats_const = helper.make_tensor(
            "repeats_const",
            onnx.TensorProto.INT64,
            [len(repeats)],
            repeats
        )
        repeats_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["repeats"],
            value=repeats_const
        )

        tile_node = helper.make_node(
            "Tile",
            inputs=["input", "repeats"],
            outputs=["output"],
        )

        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [6, 4])

        graph = helper.make_graph(
            [repeats_node, tile_node],
            "tile_model_2",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

        # Save to temp file and load
        import tempfile
        import os
        tmpdir = tempfile.mkdtemp()
        onnx_path = os.path.join(tmpdir, "tile_model_2.onnx")
        onnx.save(model, onnx_path)

        try:
            frontend = ONNXFrontend()
            ir_graph = frontend.load(onnx_path)

            output_tensor = ir_graph.tensors["output"]
            assert output_tensor.shape.dims == [6, 4]
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_tile_with_1d_repeat(self):
        """Test Tile with 1D repeat array (broadcast to match input rank)."""
        # Create model: [2, 3] -> [4, 3] (repeat 2x on axis 0 only)
        # In ONNX, a 1D repeat array [2] means repeat only the last axis
        # But we need to test broadcasting behavior

        # Use 1D repeat [2] - this should broadcast to [1, 2] for a 2D input
        repeats = [2]

        repeats_const = helper.make_tensor(
            "repeats_const",
            onnx.TensorProto.INT64,
            [len(repeats)],
            repeats
        )
        repeats_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["repeats"],
            value=repeats_const
        )

        tile_node = helper.make_node(
            "Tile",
            inputs=["input", "repeats"],
            outputs=["output"],
        )

        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 6])

        graph = helper.make_graph(
            [repeats_node, tile_node],
            "tile_model_1d",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

        # Save to temp file and load
        import tempfile
        import os
        tmpdir = tempfile.mkdtemp()
        onnx_path = os.path.join(tmpdir, "tile_model_1d.onnx")
        onnx.save(model, onnx_path)

        try:
            frontend = ONNXFrontend()
            ir_graph = frontend.load(onnx_path)

            output_tensor = ir_graph.tensors["output"]
            assert output_tensor.shape.dims == [2, 6]
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_tile_code_emission(self):
        """Test that Tile node generates correct C code."""
        repeats = [2, 3]

        repeats_const = helper.make_tensor(
            "repeats_const",
            onnx.TensorProto.INT64,
            [len(repeats)],
            repeats
        )
        repeats_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["repeats"],
            value=repeats_const
        )

        tile_node = helper.make_node(
            "Tile",
            inputs=["input", "repeats"],
            outputs=["output"],
            name="my_tile",
        )

        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4, 6])

        graph = helper.make_graph(
            [repeats_node, tile_node],
            "tile_code_test",
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

        # Should contain nnc_tile call
        assert "nnc_tile" in model_c_content

        # Clean up
        import shutil
        shutil.rmtree(tmpdir)


def test_tile_compilation_and_execution():
    """Test Tile compilation and execution."""
    print("\n=== Testing Tile Compilation and Execution ===")

    # Create model: [2, 2] -> [4, 6] (repeat 2x on axis 0, 3x on axis 1)
    repeats = [2, 3]

    repeats_const = helper.make_tensor(
        "repeats_const",
        onnx.TensorProto.INT64,
        [len(repeats)],
        repeats
    )
    repeats_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["repeats"],
        value=repeats_const
    )

    tile_node = helper.make_node(
        "Tile",
        inputs=["input", "repeats"],
        outputs=["output"],
    )

    input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
    output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4, 6])

    graph = helper.make_graph(
        [repeats_node, tile_node],
        "tile_exec_test",
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
    print("model.c contains nnc_tile:", "nnc_tile" in model_c_content)

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
    test_tile_compilation_and_execution()
