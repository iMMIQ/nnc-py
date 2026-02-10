"""Tests for IdentityEliminationPass."""

import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.passes.identity_elimination import IdentityEliminationPass


def test_identity_elimination_pass_exists():
    """Test that IdentityEliminationPass can be imported and instantiated."""
    pass_obj = IdentityEliminationPass()
    assert pass_obj.name == "IdentityElimination"


def test_removes_single_identity():
    """Test removing a single Identity node."""
    graph = Graph(name="test_single_identity")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="identity_out"
    ))

    # Add Identity node
    identity_node = Node(
        op_type=OpType.IDENTITY,
        name="identity_1",
        inputs=["input"],
        outputs=["identity_out"],
        attrs={}
    )
    graph.add_node(identity_node)

    # Set output to identity output
    graph.outputs = ["identity_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = IdentityEliminationPass()
    pass_obj.run(ctx)

    # Identity node should be removed
    assert "identity_1" not in ctx.graph.nodes
    # Output should now be the input
    assert ctx.graph.outputs == ["input"]


def test_identity_chain():
    """Test removing a chain of Identity nodes."""
    graph = Graph(name="test_identity_chain")

    # Add tensors
    for i in range(4):
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[2, 2]),
            name=f"tensor_{i}"
        ))

    # Chain: tensor_0 -> Identity -> tensor_1 -> Identity -> tensor_2 -> Identity -> tensor_3
    for i in range(3):
        node = Node(
            op_type=OpType.IDENTITY,
            name=f"identity_{i}",
            inputs=[f"tensor_{i}"],
            outputs=[f"tensor_{i+1}"],
            attrs={}
        )
        graph.add_node(node)

    graph.outputs = ["tensor_3"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = IdentityEliminationPass()
    pass_obj.run(ctx)

    # All identity nodes should be removed
    for i in range(3):
        assert f"identity_{i}" not in ctx.graph.nodes
    # Output should be the original input
    assert ctx.graph.outputs == ["tensor_0"]


def test_identity_with_consumers():
    """Test that consumer nodes are updated correctly."""
    graph = Graph(name="test_identity_consumers")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="identity_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="output"
    ))

    # Identity node
    identity_node = Node(
        op_type=OpType.IDENTITY,
        name="identity_1",
        inputs=["input"],
        outputs=["identity_out"],
        attrs={}
    )
    graph.add_node(identity_node)

    # Add node that uses identity output
    add_node = Node(
        op_type=OpType.ADD,
        name="add_1",
        inputs=["identity_out", "identity_out"],
        outputs=["output"],
        attrs={}
    )
    graph.add_node(add_node)

    graph.outputs = ["output"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = IdentityEliminationPass()
    pass_obj.run(ctx)

    # Identity node should be removed
    assert "identity_1" not in ctx.graph.nodes
    # Add node should now use input directly
    add_node_after = ctx.graph.nodes["add_1"]
    assert add_node_after.inputs == ["input", "input"]


def test_keeps_non_identity_nodes():
    """Test that non-Identity nodes are not affected."""
    graph = Graph(name="test_keep_non_identity")

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

    # Relu node (not identity)
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["input"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    graph.outputs = ["relu_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = IdentityEliminationPass()
    pass_obj.run(ctx)

    # Relu node should remain
    assert "relu_1" in ctx.graph.nodes
    assert ctx.graph.nodes["relu_1"].op_type == OpType.RELU


def test_idempotent():
    """Test that running the pass twice produces the same result."""
    graph = Graph(name="test_identity_idempotent")

    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="identity_out"
    ))

    identity_node = Node(
        op_type=OpType.IDENTITY,
        name="identity_1",
        inputs=["input"],
        outputs=["identity_out"],
        attrs={}
    )
    graph.add_node(identity_node)

    graph.outputs = ["identity_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = IdentityEliminationPass()

    # Run pass once
    pass_obj.run(ctx)
    node_count_after_first = len(ctx.graph.nodes)

    # Run pass again
    pass_obj.run(ctx)
    node_count_after_second = len(ctx.graph.nodes)

    # Should have the same number of nodes
    assert node_count_after_first == node_count_after_second
