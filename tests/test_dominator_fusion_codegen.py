"""Tests for dominator fusion code generation support."""

import pytest
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.types import DataType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.passes.dominator_fusion import DominatorFusionPass


def test_dominator_fusion_codegen():
    """Test dominator fusion with code generation.

    Creates a conv->relu graph, runs DominatorFusionPass, and verifies
    the pass doesn't crash. This is a placeholder test - full code
    generation support for dominator-fused groups would be implemented
    separately.
    """
    # Create a simple chain: input -> conv -> relu -> output
    graph = Graph(name="conv_relu_fusion_graph")

    # Add input tensor
    input_shape = TensorShape(dims=[1, 3, 224, 224])
    input_tensor = TensorType(
        dtype=DataType.FLOAT32,
        shape=input_shape,
        name="input",
    )
    graph.add_tensor(input_tensor)
    graph.inputs.append("input")

    # Add output tensor
    output_shape = TensorShape(dims=[1, 64, 112, 112])
    output_tensor = TensorType(
        dtype=DataType.FLOAT32,
        shape=output_shape,
        name="output",
    )
    graph.add_tensor(output_tensor)
    graph.outputs.append("output")

    # Create nodes
    conv = Node(
        op_type=OpType.CONV2D,
        name="conv1",
        inputs=["input"],
        outputs=["conv1_out"],
        attrs={
            "kernel_shape": [3, 3],
            "strides": [1, 1],
            "pads": [1, 1, 1, 1],
            "group": 1
        }
    )
    relu = Node(
        op_type=OpType.RELU,
        name="relu1",
        inputs=["conv1_out"],
        outputs=["output"],
        attrs={}
    )

    graph.add_node(conv)
    graph.add_node(relu)

    # Run dominator fusion pass
    fusion_pass = DominatorFusionPass()
    ctx = CompileContext(graph=graph, target="x86")

    # The pass should run without crashing
    # Note: PassBase.run() returns None, so we just ensure no exception was thrown
    result = fusion_pass.run(ctx)

    # Verify the pass completed successfully
    assert result is None

    # This is a placeholder test to verify the pass doesn't crash
    # Full code generation support for dominator-fused groups
    # would be implemented separately
    print("Dominator fusion codegen placeholder test passed")


def test_dominator_fusion_codegen_with_multiple_operations():
    """Test dominator fusion with code generation on a longer chain.

    Creates a conv->relu->add->relu graph, runs DominatorFusionPass,
    and verifies the pass doesn't crash.
    """
    # Create a longer chain: input -> conv -> relu -> add -> relu -> output
    graph = Graph(name="longer_chain_fusion_graph")

    # Add input tensor
    input_shape = TensorShape(dims=[1, 3, 224, 224])
    input_tensor = TensorType(
        dtype=DataType.FLOAT32,
        shape=input_shape,
        name="input",
    )
    graph.add_tensor(input_tensor)
    graph.inputs.append("input")

    # Add output tensor
    output_shape = TensorShape(dims=[1, 64, 112, 112])
    output_tensor = TensorType(
        dtype=DataType.FLOAT32,
        shape=output_shape,
        name="output",
    )
    graph.add_tensor(output_tensor)
    graph.outputs.append("output")

    # Create nodes
    conv = Node(
        op_type=OpType.CONV2D,
        name="conv1",
        inputs=["input"],
        outputs=["conv1_out"],
        attrs={
            "kernel_shape": [3, 3],
            "strides": [1, 1],
            "pads": [1, 1, 1, 1],
            "group": 1
        }
    )
    relu1 = Node(
        op_type=OpType.RELU,
        name="relu1",
        inputs=["conv1_out"],
        outputs=["relu1_out"],
        attrs={}
    )
    add = Node(
        op_type=OpType.ADD,
        name="add1",
        inputs=["relu1_out", "input"],  # Add shortcut connection
        outputs=["add1_out"],
        attrs={}
    )
    relu2 = Node(
        op_type=OpType.RELU,
        name="relu2",
        inputs=["add1_out"],
        outputs=["output"],
        attrs={}
    )

    graph.add_node(conv)
    graph.add_node(relu1)
    graph.add_node(add)
    graph.add_node(relu2)

    # Run dominator fusion pass
    fusion_pass = DominatorFusionPass()
    ctx = CompileContext(graph=graph, target="x86")

    # The pass should run without crashing
    result = fusion_pass.run(ctx)

    # Verify the pass completed successfully
    assert result is None

    # This is a placeholder test to verify the pass doesn't crash
    # Full code generation support for dominator-fused groups
    # would be implemented separately
    print("Dominator fusion codegen with longer chain placeholder test passed")


if __name__ == "__main__":
    test_dominator_fusion_codegen()
    test_dominator_fusion_codegen_with_multiple_operations()
    print("All dominator fusion codegen tests passed!")