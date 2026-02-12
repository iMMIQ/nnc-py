# tests/test_pattern_concrete.py
import pytest
from nnc_py.pattern.patterns import WildcardPattern, OpPattern, OrPattern, AndPattern
from nnc_py.pattern.patterns import UsePattern, ExclusiveUsePattern
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.graph import Graph
from nnc_py.pattern.base import MatchContext


def test_wildcard_matches_anything():
    """Test that WildcardPattern matches any node."""
    pattern = WildcardPattern("wc")

    conv_node = Node(op_type=OpType.CONV2D, name="conv1", inputs=[], outputs=["out1"])
    relu_node = Node(op_type=OpType.RELU, name="relu1", inputs=["out1"], outputs=[])

    graph = Graph()
    graph.add_node(conv_node)
    graph.add_node(relu_node)

    ctx = MatchContext()

    result1 = pattern.match(conv_node, graph, ctx)
    result2 = pattern.match(relu_node, graph, ctx)

    assert result1 is not None
    assert result2 is not None
    assert result1.bindings["wc"] == conv_node
    assert result2.bindings["wc"] == relu_node


def test_op_pattern_matches_specific_type():
    """Test OpPattern matches only its operator type."""
    conv_pattern = OpPattern(OpType.CONV2D, "conv")
    relu_pattern = OpPattern(OpType.RELU, "relu")

    conv_node = Node(op_type=OpType.CONV2D, name="conv1", inputs=[], outputs=["out1"])

    graph = Graph()
    graph.add_node(conv_node)

    ctx = MatchContext()

    assert conv_pattern.match(conv_node, graph, ctx) is not None
    assert relu_pattern.match(conv_node, graph, ctx) is None


def test_or_pattern():
    """Test OrPattern matches either pattern."""
    pattern = OpPattern(OpType.RELU, "act") | OpPattern(OpType.SIGMOID, "act")

    relu_node = Node(op_type=OpType.RELU, name="relu1", inputs=[], outputs=[])
    sigmoid_node = Node(op_type=OpType.SIGMOID, name="sig1", inputs=[], outputs=[])
    tanh_node = Node(op_type=OpType.TANH, name="tanh1", inputs=[], outputs=[])

    graph = Graph()
    for n in [relu_node, sigmoid_node, tanh_node]:
        graph.add_node(n)

    ctx = MatchContext()

    assert pattern.match(relu_node, graph, ctx) is not None
    assert pattern.match(sigmoid_node, graph, ctx) is not None
    assert pattern.match(tanh_node, graph, ctx) is None


def test_exclusive_use_pattern():
    """Test that only_used_by requires single consumer."""

    conv = OpPattern(OpType.CONV2D, "conv")
    relu = OpPattern(OpType.RELU, "relu")
    pattern = conv.only_used_by(relu)

    # Case 1: Single consumer - should match
    conv_node = Node(op_type=OpType.CONV2D, name="conv1", inputs=[], outputs=["conv_out"])
    relu_node = Node(op_type=OpType.RELU, name="relu1", inputs=["conv_out"], outputs=["relu_out"])

    graph1 = Graph()
    graph1.add_node(conv_node)
    graph1.add_node(relu_node)

    ctx1 = MatchContext()
    result1 = pattern.match(conv_node, graph1, ctx1)
    assert result1 is not None
    assert "conv" in result1.bindings
    assert "relu" in result1.bindings

    # Case 2: Multiple consumers - should NOT match
    conv_node2 = Node(op_type=OpType.CONV2D, name="conv2", inputs=[], outputs=["conv_out2"])
    relu_node2 = Node(op_type=OpType.RELU, name="relu2", inputs=["conv_out2"], outputs=["relu_out2"])
    other_node = Node(op_type=OpType.SIGMOID, name="other", inputs=["conv_out2"], outputs=[])

    graph2 = Graph()
    graph2.add_node(conv_node2)
    graph2.add_node(relu_node2)
    graph2.add_node(other_node)

    ctx2 = MatchContext()
    result2 = pattern.match(conv_node2, graph2, ctx2)
    assert result2 is None  # Should fail - multiple consumers
