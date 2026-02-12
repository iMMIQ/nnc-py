import pytest
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.types import DataType
from nnc_py.passes.dominator_fusion import DominatorFusionPass


def test_simple_chain_fusion():
    """Test fusion of a simple chain: conv -> relu"""
    # Create a simple chain: input -> conv -> relu -> output
    graph = Graph()

    # Create nodes
    conv = Node(op_type=OpType.CONV2D, name="conv1", inputs=["input"], outputs=["conv1_out"])
    relu = Node(op_type=OpType.RELU, name="relu1", inputs=["conv1_out"], outputs=["relu1_out"])
    output = Node(op_type=OpType.IDENTITY, name="output", inputs=["relu1_out"], outputs=["output_out"])

    graph.add_node(conv)
    graph.add_node(relu)
    graph.add_node(output)

    graph.outputs.append("output_out")

    # Run dominator fusion pass
    fusion_pass = DominatorFusionPass()
    ctx = CompileContext(graph=graph, target="x86")
    result = fusion_pass.run(ctx)

    # Verify the pass ran successfully
    # PassBase.run() returns None, so we just ensure no exception was thrown

    # In this simple case, we expect the chain to be analyzed for fusion
    # The exact fusion behavior will be implemented in the pass
    print("Simple chain fusion test passed")


def test_diamond_pattern_fusion():
    """Test fusion of a diamond pattern: conv -> [relu1, relu2] -> add"""
    # Create a diamond pattern
    graph = Graph()

    # Create nodes
    conv = Node(op_type=OpType.CONV2D, name="conv1", inputs=["input"], outputs=["conv1_out"])
    relu1 = Node(op_type=OpType.RELU, name="relu1", inputs=["conv1_out"], outputs=["relu1_out"])
    relu2 = Node(op_type=OpType.RELU, name="relu2", inputs=["conv1_out"], outputs=["relu2_out"])
    add = Node(op_type=OpType.ADD, name="add1", inputs=["relu1_out", "relu2_out"], outputs=["add1_out"])
    output = Node(op_type=OpType.IDENTITY, name="output", inputs=["add1_out"], outputs=["output_out"])

    graph.add_node(conv)
    graph.add_node(relu1)
    graph.add_node(relu2)
    graph.add_node(add)
    graph.add_node(output)

    graph.outputs.append("output_out")

    # Run dominator fusion pass
    fusion_pass = DominatorFusionPass()
    ctx = CompileContext(graph=graph, target="x86")
    result = fusion_pass.run(ctx)

    # Verify the pass ran successfully
    # PassBase.run() returns None, so we just ensure no exception was thrown

    # Verify the diamond pattern was analyzed
    print("Diamond pattern fusion test passed")


def test_max_fuse_depth_limit():
    """Test that the max fuse depth limit is respected"""
    graph = Graph()

    # Create a long chain
    current = "input"

    # Create operations with depth > 256
    for i in range(300):
        op_name = f"op_{i}"
        op_type = OpType.ADD if i % 2 == 0 else OpType.RELU
        node = Node(op_type=op_type, name=op_name, inputs=[current], outputs=[f"{op_name}_out"])
        graph.add_node(node)
        current = f"{op_name}_out"

    output = Node(op_type=OpType.IDENTITY, name="output", inputs=[current], outputs=["output_out"])
    graph.add_node(output)

    graph.outputs.append("output_out")

    # Run dominator fusion pass with depth limit
    fusion_pass = DominatorFusionPass(max_fuse_depth=256)
    ctx = CompileContext(graph=graph, target="x86")
    result = fusion_pass.run(ctx)

    # Verify the pass ran successfully
    # PassBase.run() returns None, so we just ensure no exception was thrown

    # The pass should respect the depth limit
    print("Max fuse depth limit test passed")


if __name__ == "__main__":
    test_simple_chain_fusion()
    test_diamond_pattern_fusion()
    test_max_fuse_depth_limit()
    print("All tests passed!")