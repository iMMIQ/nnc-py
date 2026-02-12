# tests/test_pattern_fusion_pass.py
import pytest
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.passes.pattern_fusion import PatternFusionPass
from nnc_py.pattern.registry import PatternRegistry, register_pattern
from nnc_py.pattern.patterns import OpPattern


def test_pattern_fusion_pass():
    """Test PatternFusionPass with a simple pattern."""
    # Clear and register a test pattern
    PatternRegistry.clear()

    register_pattern(
        name="test_conv_relu",
        pattern=OpPattern(OpType.CONV2D, "conv").only_used_by(
            OpPattern(OpType.RELU, "relu")
        ),
        priority=100,
        fused_op_type=OpType.FUSED_CONV_RELU,
    )

    # Create test graph: conv -> relu
    graph = Graph("test")
    conv = Node(
        op_type=OpType.CONV2D,
        name="conv1",
        inputs=["input"],
        outputs=["conv_out"],
        attrs={"kernel_shape": [3, 3, 1, 1]}
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

    # Run pass
    ctx = CompileContext(graph=graph, target="x86")
    ctx.debug = False

    pass_obj = PatternFusionPass()
    pass_obj.run(ctx)

    # Verify fusion occurred
    assert "fused_test_conv_relu_1" in graph.nodes
    fused_node = graph.nodes["fused_test_conv_relu_1"]
    assert fused_node.op_type == OpType.FUSED_CONV_RELU
    assert fused_node.inputs == ["input"]
    assert fused_node.outputs == ["output"]

    # Verify original nodes removed
    assert "conv1" not in graph.nodes
    assert "relu1" not in graph.nodes
