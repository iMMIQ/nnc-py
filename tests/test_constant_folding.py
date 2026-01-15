"""Tests for ConstantFoldingPass."""

import numpy as np
import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.passes.constant_folding import ConstantFoldingPass


class TestConstantFoldingPass:
    """Test suite for ConstantFoldingPass."""

    def test_fold_add_two_constants(self):
        """Test folding Add node with two constant inputs."""
        # Create graph
        graph = Graph(name="test_add")
        ctx = CompileContext(graph=graph, target="x86")

        # Add constant tensors
        const_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        const_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        graph.constants["const_a"] = const_a
        graph.constants["const_b"] = const_b

        # Add tensor types
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="const_a"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="const_b"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="output"
        ))

        # Create Add node
        add_node = Node(
            op_type=OpType.ADD,
            name="add_const",
            inputs=["const_a", "const_b"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(add_node)

        # Run pass
        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        # Verify folding
        assert "output" in graph.constants
        result = graph.constants["output"]
        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        assert add_node.metadata.get("folded") is True

    def test_fold_mul_two_constants(self):
        """Test folding Mul node with two constant inputs."""
        graph = Graph(name="test_mul")
        ctx = CompileContext(graph=graph, target="x86")

        const_a = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        const_b = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        graph.constants["const_a"] = const_a
        graph.constants["const_b"] = const_b

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="const_a"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="const_b"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="output"
        ))

        mul_node = Node(
            op_type=OpType.MUL,
            name="mul_const",
            inputs=["const_a", "const_b"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(mul_node)

        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        assert "output" in graph.constants
        result = graph.constants["output"]
        expected = np.array([10.0, 18.0, 28.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_fold_relu(self):
        """Test folding ReLU with constant input."""
        graph = Graph(name="test_relu")
        ctx = CompileContext(graph=graph, target="x86")

        const_input = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        graph.constants["const_input"] = const_input

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4]),
            name="const_input"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4]),
            name="output"
        ))

        relu_node = Node(
            op_type=OpType.RELU,
            name="relu_const",
            inputs=["const_input"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(relu_node)

        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        assert "output" in graph.constants
        result = graph.constants["output"]
        expected = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_no_fold_with_non_constant_input(self):
        """Test that nodes with non-constant inputs are not folded."""
        graph = Graph(name="test_no_fold")
        ctx = CompileContext(graph=graph, target="x86")

        # Only one input is constant
        const_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        graph.constants["const_a"] = const_a

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="const_a"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="input_var"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="output"
        ))

        add_node = Node(
            op_type=OpType.ADD,
            name="add_mixed",
            inputs=["const_a", "input_var"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(add_node)

        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        # Output should not be in constants
        assert "output" not in graph.constants
        assert add_node.metadata.get("folded") is not True

    def test_fold_sub(self):
        """Test folding Sub node."""
        graph = Graph(name="test_sub")
        ctx = CompileContext(graph=graph, target="x86")

        const_a = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        const_b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        graph.constants["const_a"] = const_a
        graph.constants["const_b"] = const_b

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="const_a"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="const_b"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="output"
        ))

        sub_node = Node(
            op_type=OpType.SUB,
            name="sub_const",
            inputs=["const_a", "const_b"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(sub_node)

        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        result = graph.constants["output"]
        expected = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_fold_div(self):
        """Test folding Div node."""
        graph = Graph(name="test_div")
        ctx = CompileContext(graph=graph, target="x86")

        const_a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        const_b = np.array([2.0, 4.0, 5.0], dtype=np.float32)
        graph.constants["const_a"] = const_a
        graph.constants["const_b"] = const_b

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="const_a"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="const_b"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="output"
        ))

        div_node = Node(
            op_type=OpType.DIV,
            name="div_const",
            inputs=["const_a", "const_b"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(div_node)

        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        result = graph.constants["output"]
        expected = np.array([5.0, 5.0, 6.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_fold_transpose(self):
        """Test folding Transpose node."""
        graph = Graph(name="test_transpose")
        ctx = CompileContext(graph=graph, target="x86")

        const_input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        graph.constants["const_input"] = const_input

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[2, 2]),
            name="const_input"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[2, 2]),
            name="output"
        ))

        transpose_node = Node(
            op_type=OpType.TRANSPOSE,
            name="transpose_const",
            inputs=["const_input"],
            outputs=["output"],
            attrs={"perm": [1, 0]}
        )
        graph.add_node(transpose_node)

        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        result = graph.constants["output"]
        expected = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_fold_reshape(self):
        """Test folding Reshape node."""
        graph = Graph(name="test_reshape")
        ctx = CompileContext(graph=graph, target="x86")

        const_input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        graph.constants["const_input"] = const_input

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[2, 2]),
            name="const_input"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4]),
            name="output"
        ))

        reshape_node = Node(
            op_type=OpType.RESHAPE,
            name="reshape_const",
            inputs=["const_input"],
            outputs=["output"],
            attrs={"shape": [4]}
        )
        graph.add_node(reshape_node)

        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        result = graph.constants["output"]
        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_fold_clip(self):
        """Test folding Clip node."""
        graph = Graph(name="test_clip")
        ctx = CompileContext(graph=graph, target="x86")

        const_input = np.array([-5.0, 0.0, 5.0, 10.0], dtype=np.float32)
        graph.constants["const_input"] = const_input

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4]),
            name="const_input"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4]),
            name="output"
        ))

        clip_node = Node(
            op_type=OpType.CLIP,
            name="clip_const",
            inputs=["const_input"],
            outputs=["output"],
            attrs={"min": 0.0, "max": 5.0}
        )
        graph.add_node(clip_node)

        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        result = graph.constants["output"]
        expected = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_fold_chain_of_constant_ops(self):
        """Test folding a chain of constant operations."""
        graph = Graph(name="test_chain")
        ctx = CompileContext(graph=graph, target="x86")

        # Input constant
        const_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        graph.constants["input"] = const_input

        # Add tensors
        for name in ["input", "mid1", "mid2", "output"]:
            graph.add_tensor(TensorType(
                dtype=DataType.FLOAT32,
                shape=TensorShape(dims=[3]),
                name=name
            ))

        # Chain: input -> add -> relu -> output
        add_node = Node(
            op_type=OpType.ADD,
            name="add1",
            inputs=["input", "input"],  # x + x
            outputs=["mid1"],
            attrs={}
        )
        graph.add_node(add_node)

        relu_node = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["mid1"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(relu_node)

        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        # After pass, mid1 and output should be constants
        assert "mid1" in graph.constants
        assert "output" in graph.constants

        # output = relu(input + input) = relu([2, 4, 6]) = [2, 4, 6]
        result = graph.constants["output"]
        expected = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_fold_sigmoid(self):
        """Test folding Sigmoid node."""
        graph = Graph(name="test_sigmoid")
        ctx = CompileContext(graph=graph, target="x86")

        const_input = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        graph.constants["const_input"] = const_input

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="const_input"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="output"
        ))

        sigmoid_node = Node(
            op_type=OpType.SIGMOID,
            name="sigmoid_const",
            inputs=["const_input"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(sigmoid_node)

        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        result = graph.constants["output"]
        # sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269
        expected = 1 / (1 + np.exp(-const_input))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_fold_tanh(self):
        """Test folding Tanh node."""
        graph = Graph(name="test_tanh")
        ctx = CompileContext(graph=graph, target="x86")

        const_input = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        graph.constants["const_input"] = const_input

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="const_input"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[3]),
            name="output"
        ))

        tanh_node = Node(
            op_type=OpType.TANH,
            name="tanh_const",
            inputs=["const_input"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(tanh_node)

        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        result = graph.constants["output"]
        expected = np.tanh(const_input)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_fold_concat(self):
        """Test folding Concat node."""
        graph = Graph(name="test_concat")
        ctx = CompileContext(graph=graph, target="x86")

        const_a = np.array([[1.0, 2.0]], dtype=np.float32)
        const_b = np.array([[3.0, 4.0]], dtype=np.float32)
        graph.constants["const_a"] = const_a
        graph.constants["const_b"] = const_b

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1, 2]),
            name="const_a"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1, 2]),
            name="const_b"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[2, 2]),
            name="output"
        ))

        concat_node = Node(
            op_type=OpType.CONCAT,
            name="concat_const",
            inputs=["const_a", "const_b"],
            outputs=["output"],
            attrs={"axis": 0}
        )
        graph.add_node(concat_node)

        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        result = graph.constants["output"]
        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_fold_flatten(self):
        """Test folding Flatten node."""
        graph = Graph(name="test_flatten")
        ctx = CompileContext(graph=graph, target="x86")

        const_input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        graph.constants["const_input"] = const_input

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[2, 2]),
            name="const_input"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4]),
            name="output"
        ))

        flatten_node = Node(
            op_type=OpType.FLATTEN,
            name="flatten_const",
            inputs=["const_input"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(flatten_node)

        pass_obj = ConstantFoldingPass()
        pass_obj.run(ctx)

        result = graph.constants["output"]
        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_pass_name_property(self):
        """Test that pass returns correct name."""
        pass_obj = ConstantFoldingPass()
        assert pass_obj.name == "constant_folding"
