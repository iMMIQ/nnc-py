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


def test_conv_sigmoid_fusion():
    """Test fusing Conv followed by Sigmoid."""
    graph = Graph(name="test_conv_sigmoid")

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
        name="sigmoid_out"
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

    # Sigmoid node
    sigmoid_node = Node(
        op_type=OpType.SIGMOID,
        name="sigmoid_1",
        inputs=["conv_out"],
        outputs=["sigmoid_out"],
        attrs={}
    )
    graph.add_node(sigmoid_node)

    graph.outputs = ["sigmoid_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # Check for fused node
    assert "fused_conv_sigmoid_1" in ctx.graph.nodes
    fused_node = ctx.graph.nodes["fused_conv_sigmoid_1"]
    assert fused_node.op_type == OpType.FUSED_CONV_SIGMOID
    assert "conv_1" not in ctx.graph.nodes, "Original conv node should be removed"
    assert "sigmoid_1" not in ctx.graph.nodes, "Original sigmoid node should be removed"


def test_add_sigmoid_fusion():
    """Test fusing Add followed by Sigmoid."""
    graph = Graph(name="test_add_sigmoid")

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
        name="sigmoid_out"
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

    # Sigmoid node
    sigmoid_node = Node(
        op_type=OpType.SIGMOID,
        name="sigmoid_1",
        inputs=["add_out"],
        outputs=["sigmoid_out"],
        attrs={}
    )
    graph.add_node(sigmoid_node)

    graph.outputs = ["sigmoid_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # Check for fused node
    assert "fused_add_sigmoid_1" in ctx.graph.nodes
    fused_node = ctx.graph.nodes["fused_add_sigmoid_1"]
    assert fused_node.op_type == OpType.FUSED_ADD_SIGMOID
    assert "add_1" not in ctx.graph.nodes, "Original add node should be removed"
    assert "sigmoid_1" not in ctx.graph.nodes, "Original sigmoid node should be removed"


def test_multiple_fusions_in_graph():
    """Test fusing multiple patterns in the same graph."""
    graph = Graph(name="test_multiple_fusions")

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
        name="add_in"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="add_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="relu2_out"
    ))

    # Conv + ReLU
    conv_node = Node(
        op_type=OpType.CONV2D,
        name="conv_1",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        attrs={}
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

    # Add + ReLU
    add_node = Node(
        op_type=OpType.ADD,
        name="add_1",
        inputs=["relu_out", "add_in"],
        outputs=["add_out"],
        attrs={}
    )
    graph.add_node(add_node)

    relu2_node = Node(
        op_type=OpType.RELU,
        name="relu_2",
        inputs=["add_out"],
        outputs=["relu2_out"],
        attrs={}
    )
    graph.add_node(relu2_node)

    graph.outputs = ["relu2_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # Should have 2 fused nodes
    fused_conv_nodes = [n for n in ctx.graph.nodes.values() if n.op_type == OpType.FUSED_CONV_RELU]
    fused_add_nodes = [n for n in ctx.graph.nodes.values() if n.op_type == OpType.FUSED_ADD_RELU]

    assert len(fused_conv_nodes) == 1, "Should have 1 fused Conv+ReLU node"
    assert len(fused_add_nodes) == 1, "Should have 1 fused Add+ReLU node"


def test_does_not_fuse_graph_output_as_intermediate():
    """Test that fusion doesn't break when producer output is a graph output."""
    graph = Graph(name="test_output_preservation")

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
        attrs={}
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

    # Set both as outputs (edge case)
    graph.outputs = ["conv_out", "relu_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # In this case, fusion should still happen since only relu_out is used
    # conv_out is an output but relu also uses it
    # After fusion, conv_out gets replaced with relu_out in the outputs list
    assert "fused_conv_relu_1" in ctx.graph.nodes
    assert "conv_out" not in ctx.graph.outputs, "conv_out should be replaced in outputs"
    assert ctx.graph.outputs == ["relu_out", "relu_out"], "conv_out replaced with relu_out"


def test_preserves_conv_attributes():
    """Test that fused node preserves Conv attributes."""
    graph = Graph(name="test_conv_attrs")

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

    # Conv node with specific attributes
    conv_attrs = {
        "kernel_shape": [5, 5],
        "strides": [2, 2],
        "pads": [1, 1, 1, 1],
        "dilations": [1, 1],
        "group": 1,
    }
    conv_node = Node(
        op_type=OpType.CONV2D,
        name="conv_1",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        attrs=conv_attrs
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
    pass_obj.run(ctx)

    # Check that attributes are preserved
    fused_node = ctx.graph.nodes["fused_conv_relu_1"]
    assert fused_node.attrs["kernel_shape"] == [5, 5]
    assert fused_node.attrs["strides"] == [2, 2]
    assert fused_node.attrs["pads"] == [1, 1, 1, 1]
    assert fused_node.attrs["dilations"] == [1, 1]
    assert fused_node.attrs["group"] == 1
