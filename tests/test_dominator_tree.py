"""Tests for post-dominator tree implementation."""

import pytest

from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.passes.indexed_forward_graph import IndexedForwardGraph
from nnc_py.passes.dominator_tree import DominatorTree


def test_simple_chain_dominator():
    """Test dominator on simple chain: conv -> relu."""
    graph = Graph("test")
    conv = Node(
        op_type=OpType.CONV2D,
        name="conv1",
        inputs=["input"],
        outputs=["conv_out"],
    )
    relu = Node(
        op_type=OpType.RELU,
        name="relu1",
        inputs=["conv_out"],
        outputs=["output"]
    )
    graph.add_node(conv)
    graph.add_node(relu)
    graph.outputs = ["output"]

    ifg = IndexedForwardGraph(graph)
    dom_tree = DominatorTree(ifg)

    # In post-domination: relu post-dominates conv
    # (all paths from conv to exit go through relu)
    assert dom_tree.get_immediate_dominator("conv1") == "relu1"
    assert dom_tree.get_immediate_dominator("relu1") is None  # Exit node


def test_diamond_pattern_dominator():
    """Test dominator on diamond pattern."""
    graph = Graph("test")
    # Diamond: conv -> [add1, add2] -> add3
    conv = Node(
        op_type=OpType.CONV2D,
        name="conv1",
        inputs=["input"],
        outputs=["conv_out"],
    )
    add1 = Node(
        op_type=OpType.ADD,
        name="add1",
        inputs=["conv_out", "bias1"],
        outputs=["add1_out"]
    )
    add2 = Node(
        op_type=OpType.ADD,
        name="add2",
        inputs=["conv_out", "bias2"],
        outputs=["add2_out"]
    )
    add3 = Node(
        op_type=OpType.ADD,
        name="add3",
        inputs=["add1_out", "add2_out"],
        outputs=["output"]
    )
    for node in [conv, add1, add2, add3]:
        graph.add_node(node)
    graph.outputs = ["output"]

    ifg = IndexedForwardGraph(graph)
    dom_tree = DominatorTree(ifg)

    # add3 should post-dominate all other nodes
    assert dom_tree.get_immediate_dominator("conv1") == "add3"
    assert dom_tree.get_immediate_dominator("add1") == "add3"
    assert dom_tree.get_immediate_dominator("add2") == "add3"
    assert dom_tree.get_immediate_dominator("add3") is None
