# tests/test_pattern_matcher.py
import pytest
from nnc_py.pattern.matcher import PatternMatcher
from nnc_py.pattern.patterns import OpPattern, WildcardPattern
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.graph import Graph


def test_matcher_finds_all_matches():
    """Test that PatternMatcher finds all pattern matches."""
    graph = Graph()

    # Create a simple chain: conv -> relu
    conv = Node(op_type=OpType.CONV2D, name="conv1", inputs=[], outputs=["c_out"])
    relu = Node(op_type=OpType.RELU, name="relu1", inputs=["c_out"], outputs=["r_out"])
    relu2 = Node(op_type=OpType.RELU, name="relu2", inputs=[], outputs=["r_out2"])

    graph.add_node(conv)
    graph.add_node(relu)
    graph.add_node(relu2)

    pattern = OpPattern(OpType.RELU, "r")
    matcher = PatternMatcher(graph)

    matches = matcher.match_pattern(pattern)

    # Should find both relu nodes
    assert len(matches) == 2


def test_matcher_filters_overlapping():
    """Test that matcher returns non-overlapping matches."""
    graph = Graph()

    # Create overlapping potential matches
    conv1 = Node(op_type=OpType.CONV2D, name="conv1", inputs=[], outputs=["c1_out"])
    conv2 = Node(op_type=OpType.CONV2D, name="conv2", inputs=[], outputs=["c2_out"])

    # Both convs are valid matches - should get both since they don't overlap
    graph.add_node(conv1)
    graph.add_node(conv2)

    pattern = OpPattern(OpType.CONV2D, "c")
    matcher = PatternMatcher(graph)

    matches = matcher.match_pattern(pattern)

    assert len(matches) == 2
