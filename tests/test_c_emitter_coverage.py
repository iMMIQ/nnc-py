"""Test coverage for CEmitter operator emission methods.

This file targets increasing coverage of src/nnc_py/codegen/c_emitter.py
from 37% to 70%+ by testing individual operator emit methods.
"""

import pytest
from nnc_py.codegen.c_emitter import CEmitter
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType


def create_basic_graph(name: str = "test_graph") -> Graph:
    """Create a basic graph for testing."""
    graph = Graph(name=name)
    return graph


def create_compile_context(graph: Graph, target: str = "x86") -> CompileContext:
    """Create a compile context for testing."""
    return CompileContext(graph=graph, target=target)


def add_tensor_to_graph(graph: Graph, name: str, shape: list, dtype: DataType = DataType.FLOAT32) -> str:
    """Add a tensor to the graph and return its name."""
    tensor_shape = TensorShape(dims=shape)
    tensor = TensorType(dtype=dtype, shape=tensor_shape, name=name)
    graph.add_tensor(tensor)
    return name


def test_helper_functions_exist():
    """Test that helper functions work correctly."""
    graph = create_basic_graph()
    assert graph.name == "test_graph"

    ctx = create_compile_context(graph)
    assert ctx.target == "x86"

    add_tensor_to_graph(graph, "input", [1, 3, 224, 224])
    assert "input" in graph.tensors
