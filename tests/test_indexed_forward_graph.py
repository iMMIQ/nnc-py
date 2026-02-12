import pytest
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.passes.indexed_forward_graph import IndexedForwardGraph


def test_simple_chain_indexing():
    """Test indexing a simple chain: conv -> relu."""
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

    # Check post_dfs_order
    assert len(ifg.post_dfs_order) == 2
    assert ifg.post_dfs_order[0].node.name == "relu1"  # Leaves first
    assert ifg.post_dfs_order[1].node.name == "conv1"

    # Check node_map
    assert "conv1" in ifg.node_map
    assert "relu1" in ifg.node_map


def test_diamond_pattern_indexing():
    """Test indexing a diamond pattern."""
    graph = Graph("test")
    # Create diamond: conv -> [add1, add2] -> add3
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

    # conv should have 2 outputs (add1, add2)
    conv_entry = ifg.node_map["conv1"]
    assert len(conv_entry.outputs) == 2

    # Check topological order
    assert len(ifg.post_dfs_order) == 4
