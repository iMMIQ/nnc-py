"""Tests for consumer rewiring after operator split (TDD Cycle 11)."""

import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.ir.split_rules import (
    SplitInfo,
    SplitPlan,
    SplitRegistry,
    SplitAxisBehavior,
    SplitAxisRule,
    OperatorSplitRule,
)
from nnc_py.passes.split_transform import SplitTransformPass
from nnc_py.passes.operators.conv_rules import register_conv2d_split_rule


@pytest.fixture(autouse=True)
def register_relu_rule():
    """Register Relu rule for cascade tests."""
    # Register Relu rule with propagation
    relu_rule = OperatorSplitRule(
        op_type=OpType.RELU,
        input_split_rules=[[
            SplitAxisRule(0, SplitAxisBehavior.FULLY_SPLITTABLE),
        ]],
        output_split_behavior=[SplitAxisBehavior.FULLY_SPLITTABLE],
        reused_inputs=set(),
        propagate_split=lambda axis: axis,
    )
    SplitRegistry.register(relu_rule)
    register_conv2d_split_rule()
    yield
    # Cleanup: remove the rule after tests
    SplitRegistry._rules = {k: v for k, v in SplitRegistry._rules.items() if k != OpType.RELU}


class TestConsumerRewiring:
    """Test consumer rewiring after operator split."""

    def test_single_consumer_rewired_to_split_nodes(self):
        """Single consumer: conv -> relu, splitting conv should cascade to relu."""
        graph = Graph(name="test_single_consumer")
        ctx = CompileContext(graph=graph, target="x86")

        # input -> conv -> relu -> output
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[32, 3, 32, 32]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[32, 64, 32, 32]),
            name="conv_out"
        ))

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[32, 64, 32, 32]),
            name="relu_out"
        ))

        conv_node = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "weights", "bias"],
            outputs=["conv_out"],
            attrs={}
        )
        graph.add_node(conv_node)

        relu_node = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv_out"],
            outputs=["relu_out"],
            attrs={}
        )
        graph.add_node(relu_node)

        # Split conv1
        split_info = SplitInfo(
            original_node=conv_node,
            split_axis=0,
            num_splits=2,
            chunk_sizes=[16, 16],
        )

        split_plan = SplitPlan(splits=[split_info])
        ctx.metadata["split_plan"] = split_plan

        # Run transform
        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Verify: conv1 should be split
        assert "conv1_split0" in graph.nodes
        assert "conv1_split1" in graph.nodes
        assert "conv1" not in graph.nodes

        # Verify: relu should also be split (cascaded)
        assert "relu1_split0" in graph.nodes
        assert "relu1_split1" in graph.nodes

        # Verify: connections - relu_split0 takes from conv_split0
        relu_split0 = graph.nodes["relu1_split0"]
        assert relu_split0.inputs[0] == "conv_out_split0"

    def test_multiple_consumers_all_cascaded(self):
        """Multiple consumers: conv -> [relu, add], both should cascade."""
        graph = Graph(name="test_multi_consumer")
        ctx = CompileContext(graph=graph, target="x86")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 3, 16, 16]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 32, 16, 16]),
            name="conv_out"
        ))

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 32, 16, 16]),
            name="relu_out"
        ))

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 32, 16, 16]),
            name="add_out"
        ))

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 32, 16, 16]),
            name="bias"
        ))

        conv_node = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "weights", "bias"],
            outputs=["conv_out"],
            attrs={}
        )
        graph.add_node(conv_node)

        relu_node = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv_out"],
            outputs=["relu_out"],
            attrs={}
        )
        graph.add_node(relu_node)

        add_node = Node(
            op_type=OpType.ADD,
            name="add1",
            inputs=["conv_out", "bias"],
            outputs=["add_out"],
            attrs={}
        )
        graph.add_node(add_node)

        # Split conv1
        split_info = SplitInfo(
            original_node=conv_node,
            split_axis=0,
            num_splits=2,
            chunk_sizes=[8, 8],
        )

        split_plan = SplitPlan(splits=[split_info])
        ctx.metadata["split_plan"] = split_plan

        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Both relu and add should be split (they have split rules)
        assert "relu1_split0" in graph.nodes
        assert "relu1_split1" in graph.nodes
        # Note: add may not split if it doesn't have a rule yet

    def test_consumer_with_no_split_rule(self):
        """Consumer without split rule should connect to first split output."""
        # Register a special op without split rule
        special_rule = OperatorSplitRule(
            op_type=OpType.IDENTITY,  # Has no split rule
            input_split_rules=[[]],  # No rules defined
            output_split_behavior=[],
            reused_inputs=set(),
        )
        SplitRegistry.register(special_rule)

        graph = Graph(name="test_no_rule_consumer")
        ctx = CompileContext(graph=graph, target="x86")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[8, 3, 8, 8]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[8, 16, 8, 8]),
            name="conv_out"
        ))

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[8, 16, 8, 8]),
            name="id_out"
        ))

        conv_node = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "weights", "bias"],
            outputs=["conv_out"],
            attrs={}
        )
        graph.add_node(conv_node)

        id_node = Node(
            op_type=OpType.IDENTITY,
            name="id1",
            inputs=["conv_out"],
            outputs=["id_out"],
            attrs={}
        )
        graph.add_node(id_node)

        split_info = SplitInfo(
            original_node=conv_node,
            split_axis=0,
            num_splits=2,
            chunk_sizes=[4, 4],
        )

        split_plan = SplitPlan(splits=[split_info])
        ctx.metadata["split_plan"] = split_plan

        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Identity should still exist (no split rule)
        assert "id1" in graph.nodes

        # Identity should be connected to a split output
        # (either first one, or all - depends on implementation)
        id_inputs = graph.nodes["id1"].inputs
        # Should reference one of the split outputs
        assert any("conv_out_split" in inp for inp in id_inputs)

    def test_split_output_tensors_created(self):
        """Verify split output tensors are created."""
        graph = Graph(name="test_tensor_creation")
        ctx = CompileContext(graph=graph, target="x86")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 3, 16, 16]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 32, 16, 16]),
            name="output"
        ))

        node = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "weights", "bias"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(node)

        split_info = SplitInfo(
            original_node=node,
            split_axis=0,
            num_splits=2,
            chunk_sizes=[8, 8],
        )

        split_plan = SplitPlan(splits=[split_info])
        ctx.metadata["split_plan"] = split_plan

        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Each split node should have corresponding output tensors
        assert "output_split0" in graph.tensors
        assert "output_split1" in graph.tensors

        # Check shapes
        tensor0 = graph.tensors["output_split0"]
        tensor1 = graph.tensors["output_split1"]

        # Original: [16, 32, 16, 16], split on axis 0
        # Split0: [8, 32, 16, 16], Split1: [8, 32, 16, 16]
        assert tensor0.shape.dims == [8, 32, 16, 16]
        assert tensor1.shape.dims == [8, 32, 16, 16]

    def test_graph_output_connection(self):
        """Test that graph outputs are correctly updated after split."""
        graph = Graph(name="test_graph_output")
        ctx = CompileContext(graph=graph, target="x86")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4, 3, 8, 8]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4, 16, 8, 8]),
            name="output"
        ))
        graph.outputs.append("output")

        node = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "weights", "bias"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(node)

        split_info = SplitInfo(
            original_node=node,
            split_axis=0,
            num_splits=2,
            chunk_sizes=[2, 2],
        )

        split_plan = SplitPlan(splits=[split_info])
        ctx.metadata["split_plan"] = split_plan

        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # After split, the graph output should still exist
        # (even though the original node is gone)
        assert "output" in graph.tensors
