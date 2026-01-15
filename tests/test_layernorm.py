"""Tests for LayerNormalization operator support."""

import numpy as np
import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType


class TestLayerNormSupport:
    """Test LayerNormalization operator integration."""

    def test_op_type_exists(self):
        """Test that LAYER_NORM OpType is defined."""
        assert OpType.LAYER_NORM is not None
        assert OpType.LAYER_NORM.value == "LayerNormalization"

    def test_create_layernorm_node(self):
        """Test creating a LayerNorm node."""
        node = Node(
            op_type=OpType.LAYER_NORM,
            name="layernorm_1",
            inputs=["input", "scale", "bias"],
            outputs=["output"],
            attrs={"axis": -1, "epsilon": 1e-5}
        )

        assert node.op_type == OpType.LAYER_NORM
        assert node.name == "layernorm_1"
        assert len(node.inputs) == 3
        assert len(node.outputs) == 1
        assert node.attrs["axis"] == -1
        assert node.attrs["epsilon"] == 1e-5

    def test_layernorm_in_graph(self):
        """Test LayerNorm node in a graph."""
        graph = Graph(name="test_layernorm")
        ctx = CompileContext(graph=graph, target="x86")

        # Add tensor definitions
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[2, 3, 4]),
            name="input"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4]),
            name="scale"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4]),
            name="bias"
        ))
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[2, 3, 4]),
            name="output"
        ))

        # Create LayerNorm node
        layernorm_node = Node(
            op_type=OpType.LAYER_NORM,
            name="layernorm_1",
            inputs=["input", "scale", "bias"],
            outputs=["output"],
            attrs={"axis": -1, "epsilon": 1e-5}
        )
        graph.add_node(layernorm_node)

        # Verify node is in graph
        assert "layernorm_1" in graph.nodes
        assert graph.nodes["layernorm_1"].op_type == OpType.LAYER_NORM

    def test_layernorm_without_scale_bias(self):
        """Test LayerNorm node without scale/bias."""
        node = Node(
            op_type=OpType.LAYER_NORM,
            name="layernom_noscale",
            inputs=["input"],
            outputs=["output"],
            attrs={"axis": -1, "epsilon": 1e-5}
        )

        assert node.op_type == OpType.LAYER_NORM
        assert len(node.inputs) == 1

    def test_layernorm_code_emission(self):
        """Test that LayerNorm generates correct C code call."""
        from nnc_py.codegen.c_emitter import CEmitter

        graph = Graph(name="test_layernorm")
        ctx = CompileContext(graph=graph, target="x86")

        # Add tensors
        for name, shape in [("input", [2, 4]), ("scale", [4]), ("bias", [4]), ("output", [2, 4])]:
            graph.add_tensor(TensorType(
                dtype=DataType.FLOAT32,
                shape=TensorShape(dims=shape),
                name=name
            ))

        # Add node
        layernorm_node = Node(
            op_type=OpType.LAYER_NORM,
            name="layernorm_1",
            inputs=["input", "scale", "bias"],
            outputs=["output"],
            attrs={"axis": -1, "epsilon": 1e-5}
        )
        graph.add_node(layernorm_node)

        # Generate code
        emitter = CEmitter()
        code = emitter.emit(ctx)

        # Check for function call
        assert "nnc_layernorm" in code
        assert "&input" in code
        assert "&scale" in code
        assert "&bias" in code
        assert "&output" in code

    def test_layernorm_default_attributes(self):
        """Test LayerNorm with default attributes."""
        node = Node(
            op_type=OpType.LAYER_NORM,
            name="layernorm_default",
            inputs=["input"],
            outputs=["output"],
            attrs={}
        )

        # Default values should be handled by code generation
        assert node.op_type == OpType.LAYER_NORM

    def test_layernorm_is_computational(self):
        """Test that LayerNorm is considered a computational node."""
        node = Node(
            op_type=OpType.LAYER_NORM,
            name="layernorm_comp",
            inputs=["input"],
            outputs=["output"],
            attrs={}
        )

        assert node.is_computational() is True
