"""Tests for SplitTransformPass (TDD Cycle 6)."""

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.ir.split_rules import (
    SplitRegistry,
    SplitInfo,
    SplitPlan,
    SplitAxisBehavior,
    SplitAxisRule,
    OperatorSplitRule,
)
from nnc_py.passes.split_transform import SplitTransformPass


class TestSplitTransformPass:
    """Test SplitTransformPass graph transformation functionality."""

    def test_pass_exists(self):
        """Test that SplitTransformPass can be instantiated."""
        pass_obj = SplitTransformPass()
        assert pass_obj is not None

    def test_pass_name_property(self):
        """Test that pass returns correct name."""
        pass_obj = SplitTransformPass()
        assert pass_obj.name == "split_transform"

    def test_create_split_nodes(self):
        """Test creating split nodes from a split plan."""
        graph = Graph(name="test_split_nodes")
        ctx = CompileContext(graph=graph, target="x86")

        # Add input and output tensors
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[256, 3, 224, 224]),
            name="input"
        ))
        graph.inputs.append("input")

        # Original output tensor
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[256, 128, 112, 112]),
            name="output"
        ))

        # Create original node
        node = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "weights", "bias"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(node)

        # Create split plan: split into 2 nodes along axis 0
        split_info = SplitInfo(
            original_node=node,
            split_axis=0,
            num_splits=2,
            chunk_sizes=[128, 128],
        )

        split_plan = SplitPlan(splits=[split_info])
        ctx.metadata["split_plan"] = split_plan

        # Run transform pass
        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Check that split nodes were created
        split_nodes = [n for n in graph.nodes.values() if "_split" in n.name]
        assert len(split_nodes) == 2

    def test_split_nodes_have_correct_names(self):
        """Test that split nodes have correct naming convention."""
        graph = Graph(name="test_naming")
        ctx = CompileContext(graph=graph, target="x86")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[100, 64, 56, 56]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[100, 128, 56, 56]),
            name="output"
        ))

        node = Node(
            op_type=OpType.CONV2D,
            name="my_conv",
            inputs=["input", "weights", "bias"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(node)

        split_info = SplitInfo(
            original_node=node,
            split_axis=0,
            num_splits=3,
            chunk_sizes=[34, 33, 33],
        )

        split_plan = SplitPlan(splits=[split_info])
        ctx.metadata["split_plan"] = split_plan

        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Check naming: my_conv_split0, my_conv_split1, my_conv_split2
        assert "my_conv_split0" in graph.nodes
        assert "my_conv_split1" in graph.nodes
        assert "my_conv_split2" in graph.nodes

    def test_original_node_removed(self):
        """Test that original node is removed after transformation."""
        graph = Graph(name="test_remove")
        ctx = CompileContext(graph=graph, target="x86")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[64, 3, 32, 32]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[64, 64, 32, 32]),
            name="output"
        ))

        node = Node(
            op_type=OpType.CONV2D,
            name="conv_to_remove",
            inputs=["input", "weights", "bias"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(node)

        split_info = SplitInfo(
            original_node=node,
            split_axis=0,
            num_splits=2,
            chunk_sizes=[32, 32],
        )

        split_plan = SplitPlan(splits=[split_info])
        ctx.metadata["split_plan"] = split_plan

        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Original node should be removed
        assert "conv_to_remove" not in graph.nodes

    def test_split_nodes_preserve_op_type(self):
        """Test that split nodes have the same op_type as original."""
        graph = Graph(name="test_op_type")
        ctx = CompileContext(graph=graph, target="x86")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[32, 16, 28, 28]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[32, 32, 28, 28]),
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
            chunk_sizes=[16, 16],
        )

        split_plan = SplitPlan(splits=[split_info])
        ctx.metadata["split_plan"] = split_plan

        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # All split nodes should have the same op_type
        for node in graph.nodes.values():
            if "_split" in node.name:
                assert node.op_type == OpType.CONV2D

    def test_split_nodes_have_split_metadata(self):
        """Test that split nodes have metadata indicating they are splits."""
        graph = Graph(name="test_metadata")
        ctx = CompileContext(graph=graph, target="x86")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 8, 14, 14]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 16, 14, 14]),
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
            split_axis=1,  # Split on channel axis
            num_splits=2,
            chunk_sizes=[8, 8],
        )

        split_plan = SplitPlan(splits=[split_info])
        ctx.metadata["split_plan"] = split_plan

        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Check metadata on split nodes
        split_node = graph.nodes["conv1_split0"]
        assert "is_split" in split_node.metadata
        assert split_node.metadata["is_split"] is True
        assert "original_node" in split_node.metadata
        assert split_node.metadata["original_node"] == "conv1"

    def test_empty_split_plan_does_not_modify_graph(self):
        """Test that an empty split plan leaves the graph unchanged."""
        graph = Graph(name="test_empty")
        ctx = CompileContext(graph=graph, target="x86")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[8, 3, 16, 16]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[8, 16, 16, 16]),
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

        # Empty split plan
        split_plan = SplitPlan(splits=[])
        ctx.metadata["split_plan"] = split_plan

        original_node_count = len(graph.nodes)

        pass_obj = SplitTransformPass()
        pass_obj.run(ctx)

        # Graph should be unchanged
        assert len(graph.nodes) == original_node_count
        assert "conv1" in graph.nodes
