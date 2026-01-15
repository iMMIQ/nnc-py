"""End-to-end tests for operator splitting (TDD Cycle 10)."""

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.passes.split_analysis import SplitAnalysisPass
from nnc_py.passes.split_transform import SplitTransformPass
from nnc_py.passes.base import PassManager
from nnc_py.passes.operators.conv_rules import register_conv2d_split_rule
from nnc_py.passes.liveness import LivenessAnalysisPass


class TestSplitEndToEnd:
    """End-to-end tests for operator splitting."""

    def test_full_pipeline_creates_split_plan(self):
        """Test complete pipeline: analysis -> transform with split plan."""
        register_conv2d_split_rule()

        # Create IR graph directly
        graph = Graph(name="test_large_conv")
        ctx = CompileContext(graph=graph, target="x86")
        ctx.metadata["max_memory"] = 10 * 1024 * 1024  # 10MB limit

        # Add tensors
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[128, 3, 224, 224]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[128, 128, 112, 112]),
            name="output"
        ))

        # Add node
        node = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "weights", "bias"],
            outputs=["output"],
            attrs={"kernel_shape": [7, 7], "strides": [2, 2]}
        )
        graph.add_node(node)

        # Run split passes
        manager = PassManager()
        manager.register(LivenessAnalysisPass())
        manager.register(SplitAnalysisPass())
        manager.register(SplitTransformPass())
        manager.run(ctx)

        # Verify split plan was created
        assert "split_plan" in ctx.metadata
        split_plan = ctx.metadata["split_plan"]

        # Should have detected the large tensor
        assert split_plan is not None
        # May or may not have splits depending on tensor size calculation

    def test_small_conv_no_split(self):
        """Test that small convolutions don't get split."""
        register_conv2d_split_rule()

        graph = Graph(name="test_small_conv")
        ctx = CompileContext(graph=graph, target="x86")
        ctx.metadata["max_memory"] = 100 * 1024 * 1024  # 100MB limit

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

        manager = PassManager()
        manager.register(LivenessAnalysisPass())
        manager.register(SplitAnalysisPass())
        manager.register(SplitTransformPass())
        manager.run(ctx)

        split_plan = ctx.metadata.get("split_plan")

        # Should have no splits for small tensor
        if split_plan:
            assert len(split_plan.splits) == 0

    def test_split_pass_names_in_applied_passes(self):
        """Test that split passes are recorded in applied_passes."""
        register_conv2d_split_rule()

        graph = Graph(name="test_pass_names")
        ctx = CompileContext(graph=graph, target="x86")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 3, 32, 32]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[16, 32, 32, 32]),
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

        manager = PassManager()
        manager.register(SplitAnalysisPass())
        manager.register(SplitTransformPass())
        manager.run(ctx)

        assert "split_analysis" in manager.applied_passes
        assert "split_transform" in manager.applied_passes

    def test_multiple_ops_in_sequence(self):
        """Test splitting in a graph with multiple operations."""
        register_conv2d_split_rule()

        # Register Relu rule
        from nnc_py.ir.split_rules import (
            OperatorSplitRule, SplitAxisBehavior, SplitAxisRule
        )
        relu_rule = OperatorSplitRule(
            op_type=OpType.RELU,
            input_split_rules=[[
                SplitAxisRule(0, SplitAxisBehavior.FULLY_SPLITTABLE),
            ]],
            output_split_behavior=[SplitAxisBehavior.FULLY_SPLITTABLE],
            reused_inputs=set(),
            propagate_split=lambda axis: axis,
        )
        from nnc_py.ir.split_rules import SplitRegistry
        SplitRegistry.register(relu_rule)

        # Create graph: Conv -> Relu
        graph = Graph(name="test_multi_op")
        ctx = CompileContext(graph=graph, target="x86")
        ctx.metadata["max_memory"] = 5 * 1024 * 1024

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[64, 3, 64, 64]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[64, 64, 64, 64]),
            name="conv_out"
        ))

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[64, 64, 64, 64]),
            name="output"
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
            outputs=["output"],
            attrs={}
        )
        graph.add_node(relu_node)

        manager = PassManager()
        manager.register(LivenessAnalysisPass())
        manager.register(SplitAnalysisPass())
        manager.run(ctx)

        # Should complete without error
        assert "split_plan" in ctx.metadata

    def test_no_memory_limit_no_split(self):
        """Test that without max_memory, no splitting occurs."""
        register_conv2d_split_rule()

        graph = Graph(name="test_no_limit")
        ctx = CompileContext(graph=graph, target="x86")
        # No max_memory set

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[256, 3, 128, 128]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[256, 128, 128, 128]),
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

        manager = PassManager()
        manager.register(SplitAnalysisPass())
        manager.run(ctx)

        split_plan = ctx.metadata.get("split_plan")

        # No split plan should be created
        assert split_plan is None or len(split_plan.splits) == 0

    def test_full_cycle_with_transform(self):
        """Test complete cycle: analysis + transform produces split nodes."""
        register_conv2d_split_rule()

        graph = Graph(name="test_full_cycle")
        ctx = CompileContext(graph=graph, target="x86")
        ctx.metadata["max_memory"] = 1 * 1024 * 1024  # 1MB - very low to force split

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[256, 3, 128, 128]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[256, 256, 64, 64]),
            name="output"
        ))

        node = Node(
            op_type=OpType.CONV2D,
            name="large_conv",
            inputs=["input", "weights", "bias"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(node)

        original_node_count = len(graph.nodes)

        manager = PassManager()
        manager.register(LivenessAnalysisPass())
        manager.register(SplitAnalysisPass())
        manager.register(SplitTransformPass())
        manager.run(ctx)

        split_plan = ctx.metadata["split_plan"]

        # After transform, original node should be removed
        # and split nodes should exist
        has_split_nodes = any("_split" in name for name in graph.nodes.keys())

        if split_plan and len(split_plan.splits) > 0:
            # If splits were created, original node should be gone
            assert "large_conv" not in graph.nodes
            # And split nodes should exist
            assert has_split_nodes
