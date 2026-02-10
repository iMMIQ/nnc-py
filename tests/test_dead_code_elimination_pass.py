"""Tests for DeadCodeEliminationPass."""

import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.passes.dead_code_elimination import DeadCodeEliminationPass


def test_dead_code_elimination_pass_exists():
    """Test that DeadCodeEliminationPass can be imported and instantiated."""
    pass_obj = DeadCodeEliminationPass()
    assert pass_obj.name == "DeadCodeElimination"


def test_removes_unused_node():
    """Test that a node with no consumers is removed."""
    # Create graph: input -> Relu -> (unused) -> Add -> output
    # The Relu output should be removed since Add uses a different input
    graph = Graph(name="test_dead_code")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="unused_relu_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="const"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="output"
    ))

    # Add nodes - Relu produces unused output
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_unused",
        inputs=["input"],
        outputs=["unused_relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    # Add uses a constant, not the relu output
    add_node = Node(
        op_type=OpType.ADD,
        name="add_final",
        inputs=["input", "const"],  # Uses input directly, not relu output
        outputs=["output"],
        attrs={}
    )
    graph.add_node(add_node)

    # Set graph outputs
    graph.outputs = ["output"]

    # Create context and run pass
    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = DeadCodeEliminationPass()
    pass_obj.run(ctx)

    # Relu node should be removed (its output is not used)
    assert "relu_unused" not in ctx.graph.nodes
    # Add node should remain
    assert "add_final" in ctx.graph.nodes


def test_keeps_used_node():
    """Test that a node whose output is used is kept."""
    graph = Graph(name="test_keep_used")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="relu_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="output"
    ))

    # Add nodes - Relu output is used by Add
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_used",
        inputs=["input"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    add_node = Node(
        op_type=OpType.ADD,
        name="add_final",
        inputs=["relu_out", "relu_out"],  # Uses relu output
        outputs=["output"],
        attrs={}
    )
    graph.add_node(add_node)

    graph.outputs = ["output"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = DeadCodeEliminationPass()
    pass_obj.run(ctx)

    # Both nodes should remain
    assert "relu_used" in ctx.graph.nodes
    assert "add_final" in ctx.graph.nodes


def test_keeps_output_nodes():
    """Test that nodes producing graph outputs are kept."""
    graph = Graph(name="test_outputs")

    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="output1"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="output2"
    ))

    relu_node = Node(
        op_type=OpType.RELU,
        name="relu1",
        inputs=["input"],
        outputs=["output1"],
        attrs={}
    )
    graph.add_node(relu_node)

    sigmoid_node = Node(
        op_type=OpType.SIGMOID,
        name="sigmoid1",
        inputs=["input"],
        outputs=["output2"],
        attrs={}
    )
    graph.add_node(sigmoid_node)

    graph.outputs = ["output1", "output2"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = DeadCodeEliminationPass()
    pass_obj.run(ctx)

    # Both nodes should be kept (they produce outputs)
    assert "relu1" in ctx.graph.nodes
    assert "sigmoid1" in ctx.graph.nodes


def test_keeps_input_producers():
    """Test that nodes producing inputs to kept nodes are kept."""
    graph = Graph(name="test_input_producers")

    # Chain: input -> relu -> sigmoid -> output
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="relu_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="sigmoid_out"
    ))

    relu_node = Node(
        op_type=OpType.RELU,
        name="relu1",
        inputs=["input"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    sigmoid_node = Node(
        op_type=OpType.SIGMOID,
        name="sigmoid1",
        inputs=["relu_out"],
        outputs=["sigmoid_out"],
        attrs={}
    )
    graph.add_node(sigmoid_node)

    graph.outputs = ["sigmoid_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = DeadCodeEliminationPass()
    pass_obj.run(ctx)

    # Both nodes should be kept (sigmoid is output, relu produces sigmoid's input)
    assert "relu1" in ctx.graph.nodes
    assert "sigmoid1" in ctx.graph.nodes


def test_idempotent():
    """Test that running the pass twice produces the same result."""
    graph = Graph(name="test_idempotent")

    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="unused"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="output"
    ))

    unused_node = Node(
        op_type=OpType.RELU,
        name="unused_relu",
        inputs=["input"],
        outputs=["unused"],
        attrs={}
    )
    graph.add_node(unused_node)

    add_node = Node(
        op_type=OpType.ADD,
        name="add_final",
        inputs=["input", "input"],
        outputs=["output"],
        attrs={}
    )
    graph.add_node(add_node)

    graph.outputs = ["output"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = DeadCodeEliminationPass()

    # Run pass once
    pass_obj.run(ctx)
    node_count_after_first = len(ctx.graph.nodes)

    # Run pass again
    pass_obj.run(ctx)
    node_count_after_second = len(ctx.graph.nodes)

    # Should have the same number of nodes
    assert node_count_after_first == node_count_after_second
