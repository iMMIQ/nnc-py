"""Tests for SplitAnalysisPass (TDD Cycle 3)."""

import numpy as np

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.ir.split_rules import SplitInfo, SplitPlan
from nnc_py.passes.split_analysis import SplitAnalysisPass
from nnc_py.passes.operators.conv_rules import register_conv2d_split_rule


class TestSplitAnalysisPass:
    """Test SplitAnalysisPass functionality."""

    def test_pass_exists(self):
        """Test that SplitAnalysisPass can be instantiated."""
        pass_obj = SplitAnalysisPass()
        assert pass_obj is not None

    def test_pass_name_property(self):
        """Test that pass returns correct name."""
        pass_obj = SplitAnalysisPass()
        assert pass_obj.name == "split_analysis"

    def test_detect_large_tensor(self):
        """Test detecting a tensor that exceeds memory limit."""
        register_conv2d_split_rule()

        # Create graph with a large Conv2D output
        # [1024, 512, 224, 224] * 4 bytes = ~1GB
        graph = Graph(name="test_large_conv")
        ctx = CompileContext(graph=graph, target="x86")
        ctx.metadata["max_memory"] = 10 * 1024 * 1024  # 10MB limit

        # Input tensor
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1024, 3, 224, 224]),
            name="input"
        ))
        graph.inputs.append("input")

        # Large output tensor
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1024, 512, 224, 224]),
            name="conv_output"
        ))

        # Create Conv2D node
        conv_node = Node(
            op_type=OpType.CONV2D,
            name="large_conv",
            inputs=["input", "weights", "bias"],
            outputs=["conv_output"],
            attrs={"kernel_shape": [7, 7], "strides": [2, 2], "pads": [3, 3, 3, 3]}
        )
        graph.add_node(conv_node)

        # Run pass
        pass_obj = SplitAnalysisPass()
        pass_obj.run(ctx)

        # Check that split plan was created
        assert "split_plan" in ctx.metadata
        split_plan = ctx.metadata["split_plan"]
        assert split_plan is not None
        assert isinstance(split_plan, SplitPlan)

    def test_split_plan_has_split_info(self):
        """Test that split plan contains SplitInfo for large tensor."""
        register_conv2d_split_rule()

        graph = Graph(name="test_split_info")
        ctx = CompileContext(graph=graph, target="x86")
        ctx.metadata["max_memory"] = 5 * 1024 * 1024  # 5MB limit

        # [256, 256, 112, 112] * 4 bytes = ~325MB
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[256, 3, 224, 224]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[256, 256, 112, 112]),
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

        pass_obj = SplitAnalysisPass()
        pass_obj.run(ctx)

        split_plan = ctx.metadata["split_plan"]
        assert len(split_plan.splits) > 0

        # Check first split info
        split_info = split_plan.splits[0]
        assert isinstance(split_info, SplitInfo)
        assert split_info.original_node == node
        assert split_info.split_axis is not None
        assert split_info.num_splits >= 2

    def test_no_split_for_small_tensor(self):
        """Test that small tensors don't trigger splitting."""
        register_conv2d_split_rule()

        graph = Graph(name="test_small")
        ctx = CompileContext(graph=graph, target="x86")
        ctx.metadata["max_memory"] = 100 * 1024 * 1024  # 100MB limit

        # Small tensor: [4, 3, 32, 32] * 4 bytes = ~48KB
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4, 3, 32, 32]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4, 16, 32, 32]),
            name="output"
        ))

        node = Node(
            op_type=OpType.CONV2D,
            name="small_conv",
            inputs=["input", "weights", "bias"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(node)

        pass_obj = SplitAnalysisPass()
        pass_obj.run(ctx)

        # Should still create plan, but with no splits
        split_plan = ctx.metadata.get("split_plan")
        if split_plan is not None:
            assert len(split_plan.splits) == 0

    def test_no_split_without_max_memory(self):
        """Test that pass does nothing when max_memory is not set."""
        register_conv2d_split_rule()

        graph = Graph(name="test_no_limit")
        ctx = CompileContext(graph=graph, target="x86")
        # No max_memory set

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1024, 512, 224, 224]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1024, 512, 224, 224]),
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

        pass_obj = SplitAnalysisPass()
        pass_obj.run(ctx)

        # Should not create split plan
        split_plan = ctx.metadata.get("split_plan")
        assert split_plan is None or len(split_plan.splits) == 0
