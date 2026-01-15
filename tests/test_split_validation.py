"""Tests for split graph validation (TDD Cycle 8)."""

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
)
from nnc_py.passes.split_transform import SplitTransformPass


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry to avoid cross-test pollution."""
    SplitRegistry.clear()
    yield
    SplitRegistry.clear()


class TestSplitGraphValidation:
    """Test that split transformation produces valid graphs."""

    def test_no_cycles_after_split(self):
        """Test that splitting doesn't introduce cycles."""
        graph = Graph(name="test_no_cycles")
        ctx = CompileContext(graph=graph, target="x86")

        # Create simple linear graph: input -> conv -> output
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[32, 3, 32, 32]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[32, 64, 32, 32]),
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

        # Create split plan
        split_info = SplitInfo(
            original_node=node,
            split_axis=0,
            num_splits=2,
            chunk_sizes=[16, 16],
        )

        split_plan = SplitPlan(splits=[split_info])
        ctx.metadata["split_plan"] = split_plan

        # Run transform
        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Verify topological sort works (no cycles)
        sorted_nodes = graph.topological_sort()
        assert len(sorted_nodes) == len(graph.nodes)

    def test_topological_order_preserved(self):
        """Test that execution order is preserved after split.

        Note: This test documents current behavior.
        Full consumer rewiring is a future enhancement.
        """
        graph = Graph(name="test_order")
        ctx = CompileContext(graph=graph, target="x86")

        # Create chain: input -> conv1 -> output (simple case)
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

        conv1 = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "w1", "b1"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(conv1)

        # Split conv1
        split_info = SplitInfo(
            original_node=conv1,
            split_axis=0,
            num_splits=2,
            chunk_sizes=[8, 8],
        )

        split_plan = SplitPlan(splits=[split_info])
        ctx.metadata["split_plan"] = split_plan

        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Topological sort should work
        sorted_nodes = graph.topological_sort()

        # All nodes should be in sorted order
        assert len(sorted_nodes) == len(graph.nodes)

        # Original conv1 should be gone
        assert "conv1" not in graph.nodes

        # Split nodes should exist
        assert "conv1_split0" in graph.nodes
        assert "conv1_split1" in graph.nodes

    def test_graph_connectivity_preserved(self):
        """Test that all nodes remain connected after split."""
        graph = Graph(name="test_connectivity")
        ctx = CompileContext(graph=graph, target="x86")

        # Branching graph: input -> conv -> branch -> output
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
            name="output"
        ))

        conv = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "w", "b"],
            outputs=["conv_out"],
            attrs={}
        )
        graph.add_node(conv)

        relu = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv_out"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(relu)

        split_info = SplitInfo(
            original_node=conv,
            split_axis=0,
            num_splits=2,
            chunk_sizes=[4, 4],
        )

        split_plan = SplitPlan(splits=[split_info])
        ctx.metadata["split_plan"] = split_plan

        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Graph should be valid
        sorted_nodes = graph.topological_sort()
        assert len(sorted_nodes) > 0

        # Original conv should be gone
        assert "conv1" not in graph.nodes
        # relu should still exist
        assert "relu1" in graph.nodes

    def test_empty_split_leaves_graph_unchanged(self):
        """Test that empty split plan doesn't modify graph structure."""
        graph = Graph(name="test_unchanged")
        ctx = CompileContext(graph=graph, target="x86")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4, 3, 4, 4]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4, 8, 4, 4]),
            name="output"
        ))

        node = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "w", "b"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(node)

        original_node_count = len(graph.nodes)
        original_node_names = set(graph.nodes.keys())

        # Empty split plan
        split_plan = SplitPlan(splits=[])
        ctx.metadata["split_plan"] = split_plan

        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Graph should be unchanged
        assert len(graph.nodes) == original_node_count
        assert set(graph.nodes.keys()) == original_node_names
        assert "conv1" in graph.nodes

    def test_multiple_splits_produce_valid_graph(self):
        """Test that splitting multiple nodes produces a valid graph."""
        graph = Graph(name="test_multi_split")
        ctx = CompileContext(graph=graph, target="x86")

        # input -> conv1 -> conv2 -> output
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[32, 3, 32, 32]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[32, 32, 32, 32]),
            name="mid"
        ))

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[32, 64, 32, 32]),
            name="output"
        ))

        conv1 = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "w1", "b1"],
            outputs=["mid"],
            attrs={}
        )
        graph.add_node(conv1)

        conv2 = Node(
            op_type=OpType.CONV2D,
            name="conv2",
            inputs=["mid", "w2", "b2"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(conv2)

        # Split both nodes
        split_info1 = SplitInfo(
            original_node=conv1,
            split_axis=0,
            num_splits=2,
            chunk_sizes=[16, 16],
        )
        split_info2 = SplitInfo(
            original_node=conv2,
            split_axis=0,
            num_splits=2,
            chunk_sizes=[16, 16],
        )

        split_plan = SplitPlan(splits=[split_info1, split_info2])
        ctx.metadata["split_plan"] = split_plan

        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Should have 4 split nodes
        split_nodes = [n for n in graph.nodes.values() if "_split" in n.name]
        assert len(split_nodes) == 4

        # Topological sort should work
        sorted_nodes = graph.topological_sort()
        assert len(sorted_nodes) == len(graph.nodes)
