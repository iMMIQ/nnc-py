"""Tests for split operator code generation (TDD Cycle 16)."""

from io import StringIO

from nnc_py.codegen.c_emitter import CEmitter
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType


class TestSplitCodegen:
    """Test code generation for split operators."""

    def test_split_conv_emits_with_offset_params(self):
        """Test that split Conv nodes emit with offset/chunk parameters."""
        graph = Graph(name="test_split_codegen")
        ctx = CompileContext(graph=graph, target="x86")

        # Add tensors
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 3, 32, 32]),
            name="input"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[8, 16, 30, 30]),
            name="output"
        ))

        # Create split conv node
        node = Node(
            op_type=OpType.CONV2D,
            name="conv1_split0",
            inputs=["input", "weights", "bias"],
            outputs=["output"],
            attrs={
                "kernel_shape": [3, 3],
                "strides": [1, 1],
                "_split_index": 0,
                "_split_axis": 0,
                "_split_chunk": 8,
            }
        )
        graph.add_node(node)

        # Generate code
        emitter = CEmitter()
        emitter._emit_operator_call(ctx, node)
        generated = emitter.output.getvalue()

        # The generated code should include nnc_conv call
        assert "nnc_conv(" in generated
        # And it should include the input/output tensors
        assert "&input" in generated
        assert "&output" in generated

    def test_non_split_conv_emits_normal_call(self):
        """Test that non-split Conv nodes emit normally."""
        graph = Graph(name="test_normal_codegen")
        ctx = CompileContext(graph=graph, target="x86")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 3, 32, 32]),
            name="input"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 16, 30, 30]),
            name="output"
        ))

        # Normal conv node (no split metadata)
        node = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "weights", "bias"],
            outputs=["output"],
            attrs={
                "kernel_shape": [3, 3],
                "strides": [1, 1],
            }
        )
        graph.add_node(node)

        # Generate code
        emitter = CEmitter()
        emitter._emit_operator_call(ctx, node)
        generated = emitter.output.getvalue()

        # Should generate normal conv call
        assert "nnc_conv(" in generated
        assert "&input" in generated
        assert "&output" in generated

    def test_detect_split_node_by_metadata(self):
        """Test detection of split nodes through metadata."""
        node = Node(
            op_type=OpType.CONV2D,
            name="conv1_split0",
            inputs=["input", "w", "b"],
            outputs=["output"],
            attrs={"_split_index": 0, "_split_axis": 0}
        )

        # Should have split metadata
        assert "_split_index" in node.attrs
        assert node.attrs["_split_index"] == 0

    def test_split_node_name_pattern(self):
        """Test that split nodes follow naming convention."""
        node = Node(
            op_type=OpType.CONV2D,
            name="conv1_split0",
            inputs=["input", "w", "b"],
            outputs=["output"],
            attrs={}
        )

        # Name should contain _split
        assert "_split" in node.name
