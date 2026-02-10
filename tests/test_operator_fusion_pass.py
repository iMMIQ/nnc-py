"""Tests for OperatorFusionPass."""

import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.passes.operator_fusion import OperatorFusionPass


def test_operator_fusion_pass_exists():
    """Test that OperatorFusionPass can be imported and instantiated."""
    pass_obj = OperatorFusionPass()
    assert pass_obj.name == "OperatorFusion"


def test_conv_relu_fusion_basic():
    """Test fusing Conv followed by ReLU."""
    graph = Graph(name="test_conv_relu")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 3, 32, 32]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[16, 3, 3, 3]),
        name="conv_weight"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="conv_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="relu_out"
    ))

    # Conv node
    conv_node = Node(
        op_type=OpType.CONV2D,
        name="conv_1",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [0, 0, 0, 0]}
    )
    graph.add_node(conv_node)

    # ReLU node (only consumer of conv_out)
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["conv_out"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    graph.outputs = ["relu_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # Conv and ReLU nodes should be replaced with a fused node
    assert "conv_1" not in ctx.graph.nodes, "Original conv node should be removed"
    assert "relu_1" not in ctx.graph.nodes, "Original relu node should be removed"

    # Check for fused node
    assert "fused_conv_relu_1" in ctx.graph.nodes, "Fused node should be created"
    fused_node = ctx.graph.nodes["fused_conv_relu_1"]
    assert fused_node.op_type == OpType.FUSED_CONV_RELU
    assert fused_node.inputs == ["input", "conv_weight"]
    assert fused_node.outputs == ["relu_out"]


def test_conv_relu_not_fused_when_multiple_consumers():
    """Test that Conv+ReLU is NOT fused when conv output has multiple consumers."""
    graph = Graph(name="test_conv_relu_multi_consumer")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 3, 32, 32]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[16, 3, 3, 3]),
        name="conv_weight"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="conv_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="relu_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="other_out"
    ))

    # Conv node
    conv_node = Node(
        op_type=OpType.CONV2D,
        name="conv_1",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [0, 0, 0, 0]}
    )
    graph.add_node(conv_node)

    # ReLU node
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["conv_out"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    # Another consumer of conv_out
    other_node = Node(
        op_type=OpType.ADD,
        name="add_1",
        inputs=["conv_out", "conv_out"],
        outputs=["other_out"],
        attrs={}
    )
    graph.add_node(other_node)

    graph.outputs = ["relu_out", "other_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # Nodes should NOT be fused (conv_out has multiple consumers)
    assert "conv_1" in ctx.graph.nodes, "Conv node should remain when output has multiple consumers"
    assert "relu_1" in ctx.graph.nodes, "ReLU node should remain when conv output has multiple consumers"


def test_add_relu_fusion():
    """Test fusing Add followed by ReLU."""
    graph = Graph(name="test_add_relu")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="input1"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="input2"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="add_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="relu_out"
    ))

    # Add node
    add_node = Node(
        op_type=OpType.ADD,
        name="add_1",
        inputs=["input1", "input2"],
        outputs=["add_out"],
        attrs={}
    )
    graph.add_node(add_node)

    # ReLU node (only consumer of add_out)
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["add_out"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    graph.outputs = ["relu_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # Add and ReLU nodes should be replaced with a fused node
    assert "add_1" not in ctx.graph.nodes
    assert "relu_1" not in ctx.graph.nodes

    # Check for fused node
    assert "fused_add_relu_1" in ctx.graph.nodes
    fused_node = ctx.graph.nodes["fused_add_relu_1"]
    assert fused_node.op_type == OpType.FUSED_ADD_RELU
    assert fused_node.inputs == ["input1", "input2"]
    assert fused_node.outputs == ["relu_out"]


def test_idempotent():
    """Test that running the pass twice produces the same result."""
    graph = Graph(name="test_fusion_idempotent")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 3, 32, 32]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[16, 3, 3, 3]),
        name="conv_weight"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="conv_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="relu_out"
    ))

    conv_node = Node(
        op_type=OpType.CONV2D,
        name="conv_1",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [0, 0, 0, 0]}
    )
    graph.add_node(conv_node)

    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["conv_out"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    graph.outputs = ["relu_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()

    # Run pass once
    pass_obj.run(ctx)
    node_count_after_first = len(ctx.graph.nodes)
    fused_node_name = list(ctx.graph.nodes.keys())[0]

    # Run pass again
    pass_obj.run(ctx)
    node_count_after_second = len(ctx.graph.nodes)

    # Should have the same number of nodes
    assert node_count_after_first == node_count_after_second
    assert fused_node_name in ctx.graph.nodes
