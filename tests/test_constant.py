"""Tests for Constant operator support."""

import numpy as np
import pytest
import onnx
from onnx import helper, numpy_helper

from nnc_py.frontend.onnx_loader import ONNXFrontend
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType


class TestConstantOpSupport:
    """Test Constant operator integration."""

    def test_op_type_exists(self):
        """Test that CONSTANT OpType is defined."""
        assert OpType.CONSTANT is not None
        assert OpType.CONSTANT.value == "Constant"

    def test_create_constant_node(self):
        """Test creating a Constant node."""
        node = Node(
            op_type=OpType.CONSTANT,
            name="const_1",
            inputs=[],
            outputs=["output"],
            attrs={"value": 42}
        )

        assert node.op_type == OpType.CONSTANT
        assert node.name == "const_1"
        assert len(node.inputs) == 0
        assert len(node.outputs) == 1

    def test_constant_has_no_inputs(self):
        """Test that Constant has no inputs."""
        node = Node(
            op_type=OpType.CONSTANT,
            name="const_no_input",
            inputs=[],
            outputs=["output"],
            attrs={}
        )

        assert len(node.inputs) == 0

    def test_load_onnx_with_constant(self):
        """Test loading ONNX model with Constant node."""
        # Create a simple ONNX model with a Constant node
        const_tensor = helper.make_tensor(
            name="const_value",
            data_type=onnx.TensorProto.FLOAT,
            dims=[2, 2],
            vals=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).flatten()
        )

        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_out"],
            name="const_node",
            value=const_tensor
        )

        # Create graph
        graph = helper.make_graph(
            [const_node],
            "test_const",
            [],
            [helper.make_tensor_value_info("const_out", onnx.TensorProto.FLOAT, [2, 2])]
        )

        model = helper.make_model(graph)

        # Save and load with our frontend
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            frontend = ONNXFrontend()
            ir_graph = frontend.load(f.name)

        # Verify constant was extracted
        assert "const_out" in ir_graph.constants
        np.testing.assert_array_equal(
            ir_graph.constants["const_out"],
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )

    def test_constant_scalar(self):
        """Test Constant with scalar value."""
        # Create a scalar constant
        const_tensor = helper.make_tensor(
            name="scalar_value",
            data_type=onnx.TensorProto.FLOAT,
            dims=[],
            vals=[5.0]
        )

        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["scalar_out"],
            name="scalar_const",
            value=const_tensor
        )

        graph = helper.make_graph(
            [const_node],
            "test_scalar_const",
            [],
            [helper.make_tensor_value_info("scalar_out", onnx.TensorProto.FLOAT, [])]
        )

        model = helper.make_model(graph)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            frontend = ONNXFrontend()
            ir_graph = frontend.load(f.name)

        assert "scalar_out" in ir_graph.constants
        assert ir_graph.constants["scalar_out"] == 5.0

    def test_constant_int32(self):
        """Test Constant with INT32 value."""
        const_tensor = helper.make_tensor(
            name="int_value",
            data_type=onnx.TensorProto.INT32,
            dims=[3],
            vals=[10, 20, 30]
        )

        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["int_out"],
            name="int_const",
            value=const_tensor
        )

        graph = helper.make_graph(
            [const_node],
            "test_int_const",
            [],
            [helper.make_tensor_value_info("int_out", onnx.TensorProto.INT32, [3])]
        )

        model = helper.make_model(graph)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            frontend = ONNXFrontend()
            ir_graph = frontend.load(f.name)

        assert "int_out" in ir_graph.constants
        np.testing.assert_array_equal(
            ir_graph.constants["int_out"],
            np.array([10, 20, 30], dtype=np.int32)
        )

    def test_constant_in_graph_with_other_ops(self):
        """Test Constant node used as input to other operations."""
        # Create a constant and use it in an Add operation
        const_tensor = helper.make_tensor(
            name="const_value",
            data_type=onnx.TensorProto.FLOAT,
            dims=[2, 2],
            vals=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32).flatten()
        )

        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_out"],
            name="const_node",
            value=const_tensor
        )

        # Input value info
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])

        # Add node using constant
        add_node = helper.make_node(
            "Add",
            inputs=["input", "const_out"],
            outputs=["output"],
            name="add_node"
        )

        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

        graph = helper.make_graph(
            [const_node, add_node],
            "test_const_with_ops",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            # Disable onnxsim to keep Constant node in graph
            frontend = ONNXFrontend(enable_simplify=False)
            ir_graph = frontend.load(f.name)

        # Verify constant was extracted
        assert "const_out" in ir_graph.constants
        # Verify both nodes are in graph
        assert "const_node" in ir_graph.nodes
        assert "add_node" in ir_graph.nodes

    def test_constant_1d_tensor(self):
        """Test Constant with 1D tensor."""
        const_tensor = helper.make_tensor(
            name="vec_value",
            data_type=onnx.TensorProto.FLOAT,
            dims=[4],
            vals=[0.25, 0.5, 0.75, 1.0]
        )

        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["vec_out"],
            name="vec_const",
            value=const_tensor
        )

        graph = helper.make_graph(
            [const_node],
            "test_1d_const",
            [],
            [helper.make_tensor_value_info("vec_out", onnx.TensorProto.FLOAT, [4])]
        )

        model = helper.make_model(graph)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            frontend = ONNXFrontend()
            ir_graph = frontend.load(f.name)

        assert "vec_out" in ir_graph.constants
        np.testing.assert_array_equal(
            ir_graph.constants["vec_out"],
            np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        )
