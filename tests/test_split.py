"""Tests for Split operator support."""

import numpy as np
import pytest
import onnx
from onnx import helper

from nnc_py.frontend.onnx_loader import ONNXFrontend
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType


class TestSplitOpSupport:
    """Test Split operator integration."""

    def test_op_type_exists(self):
        """Test that SPLIT OpType is defined."""
        assert OpType.SPLIT is not None
        assert OpType.SPLIT.value == "Split"

    def test_create_split_node(self):
        """Test creating a Split node."""
        node = Node(
            op_type=OpType.SPLIT,
            name="split_1",
            inputs=["input"],
            outputs=["output1", "output2"],
            attrs={"axis": 1}
        )

        assert node.op_type == OpType.SPLIT
        assert node.name == "split_1"
        assert len(node.inputs) == 1
        assert len(node.outputs) == 2
        assert node.attrs["axis"] == 1

    def test_split_with_axis(self):
        """Test Split with specified axis."""
        node = Node(
            op_type=OpType.SPLIT,
            name="split_axis",
            inputs=["input"],
            outputs=["out1", "out2", "out3"],
            attrs={"axis": 0}
        )

        assert node.attrs["axis"] == 0
        assert len(node.outputs) == 3

    def test_split_with_split_sizes(self):
        """Test Split with explicit split sizes."""
        node = Node(
            op_type=OpType.SPLIT,
            name="split_sizes",
            inputs=["input"],
            outputs=["out1", "out2"],
            attrs={"axis": 1, "split": [32, 64]}
        )

        assert node.attrs["split"] == [32, 64]

    def test_load_onnx_with_split(self):
        """Test loading ONNX model with Split node."""
        # Create a simple ONNX model with a Split node
        # Split a [1, 4, 2, 2] tensor into 2 [1, 2, 2, 2] tensors along axis 1
        split_node = helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["output1", "output2"],
            name="split_node",
            axis=1
        )

        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4, 2, 2])
        output1_val = helper.make_tensor_value_info("output1", onnx.TensorProto.FLOAT, [1, 2, 2, 2])
        output2_val = helper.make_tensor_value_info("output2", onnx.TensorProto.FLOAT, [1, 2, 2, 2])

        graph = helper.make_graph(
            [split_node],
            "test_split",
            [input_val],
            [output1_val, output2_val]
        )

        model = helper.make_model(graph)

        # Save and load with our frontend
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            frontend = ONNXFrontend()
            ir_graph = frontend.load(f.name)

        # Verify nodes and outputs
        assert "split_node" in ir_graph.nodes
        assert len(ir_graph.nodes["split_node"].outputs) == 2
        assert "output1" in ir_graph.tensors
        assert "output2" in ir_graph.tensors

        # Verify output shapes
        output1_tensor = ir_graph.tensors["output1"]
        output2_tensor = ir_graph.tensors["output2"]
        assert output1_tensor.shape.dims == [1, 2, 2, 2]
        assert output2_tensor.shape.dims == [1, 2, 2, 2]

    def test_split_axis_0(self):
        """Test Split along axis 0."""
        split_node = helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["out1", "out2"],
            name="split_axis0",
            axis=0
        )

        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [4, 3])
        out1_val = helper.make_tensor_value_info("out1", onnx.TensorProto.FLOAT, [2, 3])
        out2_val = helper.make_tensor_value_info("out2", onnx.TensorProto.FLOAT, [2, 3])

        graph = helper.make_graph(
            [split_node],
            "test_split_axis0",
            [input_val],
            [out1_val, out2_val]
        )

        model = helper.make_model(graph)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            frontend = ONNXFrontend()
            ir_graph = frontend.load(f.name)

        out1_tensor = ir_graph.tensors["out1"]
        out2_tensor = ir_graph.tensors["out2"]
        assert out1_tensor.shape.dims == [2, 3]
        assert out2_tensor.shape.dims == [2, 3]

    def test_split_with_negative_axis(self):
        """Test Split with negative axis."""
        split_node = helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["out1", "out2"],
            name="split_neg_axis",
            axis=-1
        )

        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 4])
        out1_val = helper.make_tensor_value_info("out1", onnx.TensorProto.FLOAT, [2, 2])
        out2_val = helper.make_tensor_value_info("out2", onnx.TensorProto.FLOAT, [2, 2])

        graph = helper.make_graph(
            [split_node],
            "test_split_neg_axis",
            [input_val],
            [out1_val, out2_val]
        )

        model = helper.make_model(graph)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            frontend = ONNXFrontend()
            ir_graph = frontend.load(f.name)

        assert "split_neg_axis" in ir_graph.nodes
        assert ir_graph.nodes["split_neg_axis"].attrs["axis"] == -1

    def test_split_three_outputs(self):
        """Test Split into 3 outputs."""
        split_node = helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["out1", "out2", "out3"],
            name="split_three",
            axis=1
        )

        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 6, 2])
        out_val = [helper.make_tensor_value_info(f"out{i}", onnx.TensorProto.FLOAT, [1, 2, 2])
                   for i in range(1, 4)]

        graph = helper.make_graph(
            [split_node],
            "test_split_three",
            [input_val],
            out_val
        )

        model = helper.make_model(graph)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            frontend = ONNXFrontend()
            ir_graph = frontend.load(f.name)

        assert len(ir_graph.nodes["split_three"].outputs) == 3
        for i in range(1, 4):
            tensor = ir_graph.tensors[f"out{i}"]
            assert tensor.shape.dims == [1, 2, 2]

    def test_split_code_emission(self):
        """Test that Split generates correct C code call."""
        from nnc_py.codegen.c_emitter import CEmitter

        graph = Graph(name="test_split")
        ctx = CompileContext(graph=graph, target="x86")

        # Add tensors
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[2, 4]),
            name="input"
        ))
        for i in range(2):
            graph.add_tensor(TensorType(
                dtype=DataType.FLOAT32,
                shape=TensorShape(dims=[2, 2]),
                name=f"output{i+1}"
            ))

        # Add node
        split_node = Node(
            op_type=OpType.SPLIT,
            name="split_1",
            inputs=["input"],
            outputs=["output1", "output2"],
            attrs={"axis": 1}
        )
        graph.add_node(split_node)

        # Generate code
        emitter = CEmitter()
        code = emitter.emit(ctx)

        # Check for function call
        assert "nnc_split" in code
        assert "&input" in code

    def test_split_in_graph_with_other_ops(self):
        """Test Split used with other operations."""
        # Split -> Add both outputs
        split_node = helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["split1", "split2"],
            name="split_node",
            axis=1
        )

        # Add split outputs together
        add_node = helper.make_node(
            "Add",
            inputs=["split1", "split2"],
            outputs=["output"],
            name="add_node"
        )

        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4, 2, 2])
        split1_val = helper.make_tensor_value_info("split1", onnx.TensorProto.FLOAT, [1, 2, 2, 2])
        split2_val = helper.make_tensor_value_info("split2", onnx.TensorProto.FLOAT, [1, 2, 2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2, 2, 2])

        graph = helper.make_graph(
            [split_node, add_node],
            "test_split_with_ops",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            frontend = ONNXFrontend()
            ir_graph = frontend.load(f.name)

        # Verify both nodes are in graph
        assert "split_node" in ir_graph.nodes
        assert "add_node" in ir_graph.nodes
        # Verify Add node has correct inputs
        assert ir_graph.nodes["add_node"].inputs == ["split1", "split2"]
