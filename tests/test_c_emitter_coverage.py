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


def test_emit_clip_call_with_attributes():
    """Test _emit_clip_call with min/max as attributes."""
    graph = create_basic_graph("clip_test")

    add_tensor_to_graph(graph, "input", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "output", [1, 3, 224, 224])

    node = Node(
        op_type=OpType.CLIP,
        name="clip1",
        inputs=["input"],
        outputs=["output"],
        attrs={"min": 0.0, "max": 6.0},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_clip_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_clip" in output
    assert ("0.0" in output or "0" in output)
    assert ("6.0" in output or "6" in output)


def test_emit_clip_call_with_inputs():
    """Test _emit_clip_call with min/max as constant inputs."""
    graph = create_basic_graph("clip_test2")

    add_tensor_to_graph(graph, "input", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "min_val", [1])
    add_tensor_to_graph(graph, "max_val", [1])
    add_tensor_to_graph(graph, "output", [1, 3, 224, 224])

    # Add constants
    graph.constants["min_val"] = [0.0]
    graph.constants["max_val"] = [6.0]

    node = Node(
        op_type=OpType.CLIP,
        name="clip2",
        inputs=["input", "min_val", "max_val"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_clip_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_clip" in output


def test_emit_shape_call():
    """Test _emit_shape_call generates correct C code."""
    graph = create_basic_graph("shape_test")

    add_tensor_to_graph(graph, "input", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "output", [4], dtype=DataType.INT64)

    node = Node(
        op_type=OpType.SHAPE,
        name="shape1",
        inputs=["input"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_shape_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_shape" in output
    assert ("1, 3, 224, 224" in output or "[1, 3, 224, 224]" in output)


def test_emit_constantofshape_call():
    """Test _emit_constantofshape_call generates correct C code."""
    graph = create_basic_graph("constantofshape_test")

    add_tensor_to_graph(graph, "shape_input", [3], dtype=DataType.INT64)
    add_tensor_to_graph(graph, "output", [2, 3, 4])

    # Add shape constant
    graph.constants["shape_input"] = [2, 3, 4]

    node = Node(
        op_type=OpType.CONSTANT_OF_SHAPE,
        name="constantofshape1",
        inputs=["shape_input"],
        outputs=["output"],
        attrs={"value": 1.0},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_constantofshape_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_constantofshape" in output
    assert ("1.0" in output or "1f" in output)


def test_emit_expand_call():
    """Test _emit_expand_call generates correct C code."""
    graph = create_basic_graph("expand_test")

    add_tensor_to_graph(graph, "input", [1, 3])
    add_tensor_to_graph(graph, "shape_input", [3], dtype=DataType.INT64)
    add_tensor_to_graph(graph, "output", [2, 3])

    # Add shape constant
    graph.constants["shape_input"] = [2, 3]

    node = Node(
        op_type=OpType.EXPAND,
        name="expand1",
        inputs=["input", "shape_input"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_expand_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_expand" in output


def test_emit_greater_call():
    """Test _emit_greater_call generates correct C code."""
    graph = create_basic_graph("greater_test")

    add_tensor_to_graph(graph, "input1", [1, 3])
    add_tensor_to_graph(graph, "input2", [1, 3])
    add_tensor_to_graph(graph, "output", [1, 3], dtype=DataType.BOOL)

    node = Node(
        op_type=OpType.GREATER,
        name="greater1",
        inputs=["input1", "input2"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_greater_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_greater" in output


def test_emit_or_call():
    """Test _emit_or_call generates correct C code."""
    graph = create_basic_graph("or_test")

    add_tensor_to_graph(graph, "input1", [1, 3], dtype=DataType.BOOL)
    add_tensor_to_graph(graph, "input2", [1, 3], dtype=DataType.BOOL)
    add_tensor_to_graph(graph, "output", [1, 3], dtype=DataType.BOOL)

    node = Node(
        op_type=OpType.OR,
        name="or1",
        inputs=["input1", "input2"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_or_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_or" in output


def test_emit_not_call():
    """Test _emit_not_call generates correct C code."""
    graph = create_basic_graph("not_test")

    add_tensor_to_graph(graph, "input", [1, 3], dtype=DataType.BOOL)
    add_tensor_to_graph(graph, "output", [1, 3], dtype=DataType.BOOL)

    node = Node(
        op_type=OpType.NOT,
        name="not1",
        inputs=["input"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_not_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_not" in output
