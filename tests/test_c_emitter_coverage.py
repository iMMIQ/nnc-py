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


def test_emit_concat_call():
    """Test _emit_concat_call generates correct C code."""
    graph = create_basic_graph("concat_test")

    # Add input tensors
    add_tensor_to_graph(graph, "input1", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "input2", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "output", [1, 6, 224, 224])

    # Create concat node
    node = Node(
        op_type=OpType.CONCAT,
        name="concat1",
        inputs=["input1", "input2"],
        outputs=["output"],
        attrs={"axis": 1},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_concat_call(ctx, node)
    output = emitter.output.getvalue()

    # Verify generated code contains expected patterns
    assert "nnc_concat" in output
    assert "concat1_inputs" in output
    assert ("&input1" in output or "&tensor_input1" in output)
    assert ("&input2" in output or "&tensor_input2" in output)
    assert ("&output" in output or "&tensor_output" in output)
    assert "axis" in output or "1" in output
