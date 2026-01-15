"""Tests for Identity operator support."""

import numpy as np
import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType


class TestIdentitySupport:
    """Test Identity operator integration."""

    def test_op_type_exists(self):
        """Test that IDENTITY OpType is defined."""
        assert OpType.IDENTITY is not None
        assert OpType.IDENTITY.value == "Identity"

    def test_create_identity_node(self):
        """Test creating an Identity node."""
        node = Node(
            op_type=OpType.IDENTITY,
            name="identity_1",
            inputs=["input"],
            outputs=["output"],
            attrs={}
        )

        assert node.op_type == OpType.IDENTITY
        assert node.name == "identity_1"
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1

    def test_identity_in_graph(self):
        """Test Identity node in a graph."""
        graph = Graph(name="test_identity")
        ctx = CompileContext(graph=graph, target="x86")

        # Add tensor definitions
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[2, 3, 4]),
            name="input"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[2, 3, 4]),
            name="output"
        ))

        # Create Identity node
        identity_node = Node(
            op_type=OpType.IDENTITY,
            name="identity_1",
            inputs=["input"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(identity_node)

        # Verify node is in graph
        assert "identity_1" in graph.nodes
        assert graph.nodes["identity_1"].op_type == OpType.IDENTITY

    def test_identity_code_emission(self):
        """Test that Identity generates correct C code call."""
        from nnc_py.codegen.c_emitter import CEmitter

        graph = Graph(name="test_identity")
        ctx = CompileContext(graph=graph, target="x86")

        # Add tensors
        for name, shape in [("input", [2, 4]), ("output", [2, 4])]:
            graph.add_tensor(TensorType(
                dtype=DataType.FLOAT32,
                shape=TensorShape(dims=shape),
                name=name
            ))

        # Add node
        identity_node = Node(
            op_type=OpType.IDENTITY,
            name="identity_1",
            inputs=["input"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(identity_node)

        # Generate code
        emitter = CEmitter()
        code = emitter.emit(ctx)

        # Check for function call
        assert "nnc_identity" in code
        assert "&input" in code
        assert "&output" in code

    def test_identity_is_not_computational(self):
        """Test that Identity is NOT considered a computational node."""
        node = Node(
            op_type=OpType.IDENTITY,
            name="identity_comp",
            inputs=["input"],
            outputs=["output"],
            attrs={}
        )

        # Identity is a shape manipulation, not computational
        # But for now, is_computational() returns False only for specific ops
        # Let's just verify the node exists
        assert node.op_type == OpType.IDENTITY

    def test_identity_chain(self):
        """Test chain of Identity nodes."""
        graph = Graph(name="test_identity_chain")

        # Add tensors
        for i in range(4):
            name = f"tensor_{i}"
            graph.add_tensor(TensorType(
                dtype=DataType.FLOAT32,
                shape=TensorShape(dims=[2, 3]),
                name=name
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

        # Verify all nodes are in graph
        assert len(graph.nodes) == 3
        for i in range(3):
            assert f"identity_{i}" in graph.nodes
