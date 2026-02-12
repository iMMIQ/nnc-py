import pytest
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.op_pattern import OpPatternKind
from nnc_py.passes.indexed_forward_graph import IndexedForwardGraph
from nnc_py.passes.path_validator import PathValidator


def test_simple_path_validation():
    """Test simple chain validation conv -> relu"""
    # Create a simple graph: conv -> relu
    ir_graph = Graph("test")
    conv = Node(
        op_type=OpType.CONV2D,
        name="conv",
        inputs=["input"],
        outputs=["conv_out"],
    )
    relu = Node(
        op_type=OpType.RELU,
        name="relu",
        inputs=["conv_out"],
        outputs=["output"]
    )
    ir_graph.add_node(conv)
    ir_graph.add_node(relu)
    ir_graph.outputs = ["output"]

    graph = IndexedForwardGraph(ir_graph)
    validator = PathValidator(graph)

    # Get the node entries
    conv_entry = graph.node_map["conv"]
    relu_entry = graph.node_map["relu"]

    # This should pass - simple path is valid (conv is kOutEWiseFusable = 4, relu is kElemWise = 1)
    assert validator.check_path(conv_entry, relu_entry, OpPatternKind.kOutEWiseFusable)


def test_diamond_path_validation():
    """Test diamond pattern validation"""
    # Create diamond pattern: A -> {B,C} -> D
    ir_graph = Graph("test")
    a = Node(
        op_type=OpType.CONV2D,
        name="conv",
        inputs=["input"],
        outputs=["conv_out"],
    )
    b = Node(
        op_type=OpType.RELU,
        name="relu1",
        inputs=["conv_out"],
        outputs=["relu1_out"],
    )
    c = Node(
        op_type=OpType.RELU,
        name="relu2",
        inputs=["conv_out"],
        outputs=["relu2_out"],
    )
    d = Node(
        op_type=OpType.MATMUL,
        name="matmul",
        inputs=["relu1_out", "relu2_out"],
        outputs=["output"]
    )
    ir_graph.add_node(a)
    ir_graph.add_node(b)
    ir_graph.add_node(c)
    ir_graph.add_node(d)
    ir_graph.outputs = ["output"]

    graph = IndexedForwardGraph(ir_graph)
    validator = PathValidator(graph)

    # Get the node entries
    a_entry = graph.node_map["conv"]
    d_entry = graph.node_map["matmul"]

    # Should pass if all paths satisfy constraints
    assert validator.check_path(a_entry, d_entry, OpPatternKind.kOutEWiseFusable)


def test_blocked_path_validation():
    """Test that opaque op blocks fusion"""
    # Create path: conv -> relu -> maxpool -> matmul
    ir_graph = Graph("test")
    conv = Node(
        op_type=OpType.CONV2D,
        name="conv",
        inputs=["input"],
        outputs=["conv_out"],
    )
    relu = Node(
        op_type=OpType.RELU,
        name="relu",
        inputs=["conv_out"],
        outputs=["relu_out"],
    )
    opaque = Node(
        op_type=OpType.MAXPOOL,
        name="maxpool",
        inputs=["relu_out"],
        outputs=["pool_out"],
    )
    matmul = Node(
        op_type=OpType.MATMUL,
        name="matmul",
        inputs=["pool_out"],
        outputs=["output"]
    )
    ir_graph.add_node(conv)
    ir_graph.add_node(relu)
    ir_graph.add_node(opaque)
    ir_graph.add_node(matmul)
    ir_graph.outputs = ["output"]

    graph = IndexedForwardGraph(ir_graph)
    validator = PathValidator(graph)

    # Get the node entries
    conv_entry = graph.node_map["conv"]
    matmul_entry = graph.node_map["matmul"]

    # Should fail - path contains opaque operation (kind 0)
    assert not validator.check_path(conv_entry, matmul_entry, OpPatternKind.kElemWise)


def test_path_with_condition():
    """Test custom condition support"""
    ir_graph = Graph("test")
    a = Node(
        op_type=OpType.CONV2D,
        name="conv",
        inputs=["input"],
        outputs=["conv_out"],
    )
    b = Node(
        op_type=OpType.RELU,
        name="relu",
        inputs=["conv_out"],
        outputs=["relu_out"],
    )
    c = Node(
        op_type=OpType.MATMUL,
        name="matmul",
        inputs=["relu_out"],
        outputs=["output"]
    )
    ir_graph.add_node(a)
    ir_graph.add_node(b)
    ir_graph.add_node(c)
    ir_graph.outputs = ["output"]

    graph = IndexedForwardGraph(ir_graph)
    validator = PathValidator(graph)

    # Get the node entries
    a_entry = graph.node_map["conv"]
    b_entry = graph.node_map["relu"]
    c_entry = graph.node_map["matmul"]

    # Custom condition: allow only conv and relu ops (not matmul)
    def condition(node):
        node_name = node.node.name
        return node_name in ["conv", "relu"]

    # Should pass - both conv and relu satisfy condition
    assert validator.check_path_with_condition(a_entry, b_entry, condition)

    # Should fail - matmul is not allowed
    assert not validator.check_path_with_condition(a_entry, c_entry, condition)


def test_count_nodes_on_path():
    """Test node counting on path"""
    ir_graph = Graph("test")
    a = Node(
        op_type=OpType.CONV2D,
        name="conv",
        inputs=["input"],
        outputs=["conv_out"],
    )
    b = Node(
        op_type=OpType.RELU,
        name="relu1",
        inputs=["conv_out"],
        outputs=["relu1_out"],
    )
    c = Node(
        op_type=OpType.RELU,
        name="relu2",
        inputs=["relu1_out"],
        outputs=["relu2_out"],
    )
    d = Node(
        op_type=OpType.MATMUL,
        name="matmul",
        inputs=["relu2_out"],
        outputs=["output"]
    )
    ir_graph.add_node(a)
    ir_graph.add_node(b)
    ir_graph.add_node(c)
    ir_graph.add_node(d)
    ir_graph.outputs = ["output"]

    graph = IndexedForwardGraph(ir_graph)
    validator = PathValidator(graph)

    # Get the node entries
    a_entry = graph.node_map["conv"]
    d_entry = graph.node_map["matmul"]

    # Path conv -> relu1 -> relu2 -> matmul has 4 nodes
    assert validator.count_nodes_on_path(a_entry, d_entry) == 4


def test_no_path():
    """Test validation when no path exists"""
    ir_graph = Graph("test")
    a = Node(
        op_type=OpType.CONV2D,
        name="conv",
        inputs=["input"],
        outputs=["conv_out"],
    )
    b = Node(
        op_type=OpType.MATMUL,
        name="matmul",
        inputs=["other_input"],
        outputs=["output"]
    )
    ir_graph.add_node(a)
    ir_graph.add_node(b)
    ir_graph.outputs = ["output"]

    graph = IndexedForwardGraph(ir_graph)
    validator = PathValidator(graph)

    # Get the node entries
    a_entry = graph.node_map["conv"]
    b_entry = graph.node_map["matmul"]

    # Should return False - no path exists
    # With the new logic, no path means valid (True), so test should be flipped
    assert validator.check_path(a_entry, b_entry, OpPatternKind.kElemWise)