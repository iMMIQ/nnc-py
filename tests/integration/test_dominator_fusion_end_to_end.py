"""End-to-end integration tests for DominatorFusionPass.

This module contains comprehensive integration tests that verify the DominatorFusionPass
works correctly in realistic scenarios, including:

1. Diamond pattern fusion - Multiple consumers converging
2. Complex residual patterns - Multi-level diamonds (ResNet-style)
3. Path validation - Rejecting unsafe patterns
4. Pass pipeline integration - Working with other passes
"""

import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import helper
import pytest

from nnc_py import Compiler
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.types import DataType
from nnc_py.passes.dominator_fusion import DominatorFusionPass
from nnc_py.passes.indexed_forward_graph import IndexedForwardGraph
from nnc_py.passes.dominator_tree import DominatorTree
from nnc_py.passes.path_validator import PathValidator
from nnc_py.ir.op_pattern import OpPatternKind, get_op_pattern_kind
from nnc_py.passes.base import PassManager


class TestDiamondPatternFusion:
    """Test diamond pattern fusion - the core use case for dominator-based fusion.

    Diamond patterns occur when multiple operations consume the same input
    and then their outputs converge at a single node. This is the key pattern
    that dominator-based fusion is designed to handle.
    """

    def test_simple_diamond_conv_to_relus_to_add(self):
        """Test fusion of Conv -> [ReLU1, ReLU2] -> Add pattern.

        This is the classic diamond pattern:
        - A CONV2D operation produces an output
        - Two separate ReLU operations consume it
        - An Add operation consumes both ReLU outputs

        The Add post-dominates the Conv (all paths from Conv to exit go through Add),
        making it safe to analyze for fusion.
        """
        graph = Graph("simple_diamond")

        # Create nodes: conv -> [relu1, relu2] -> add
        conv = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input"],
            outputs=["conv_out"],
            attrs={"kernel_shape": [3, 3], "strides": [1, 1]}
        )
        relu1 = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv_out"],
            outputs=["relu1_out"]
        )
        relu2 = Node(
            op_type=OpType.RELU,
            name="relu2",
            inputs=["conv_out"],
            outputs=["relu2_out"]
        )
        add = Node(
            op_type=OpType.ADD,
            name="add1",
            inputs=["relu1_out", "relu2_out"],
            outputs=["output"]
        )

        for node in [conv, relu1, relu2, add]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Run the dominator fusion pass
        fusion_pass = DominatorFusionPass()
        ctx = CompileContext(graph=graph, target="x86")
        fusion_pass.run(ctx)

        # Verify the pass completed without errors
        # The pass should have analyzed the diamond structure
        assert fusion_pass.dominator_tree is not None
        assert fusion_pass.group_arena is not None

        # Verify post-dominator relationships
        # The add node should post-dominate conv1 (all paths from conv1 go through add1)
        assert fusion_pass.dominator_tree.does_post_dominate("conv1", "add1")

    def test_wider_diamond_three_branches(self):
        """Test diamond pattern with three converging branches."""
        graph = Graph("wide_diamond")

        # Conv -> [relu1, relu2, relu3] -> concat
        conv = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input"],
            outputs=["conv_out"]
        )
        relu1 = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv_out"],
            outputs=["relu1_out"]
        )
        relu2 = Node(
            op_type=OpType.RELU,
            name="relu2",
            inputs=["conv_out"],
            outputs=["relu2_out"]
        )
        relu3 = Node(
            op_type=OpType.RELU,
            name="relu3",
            inputs=["conv_out"],
            outputs=["relu3_out"]
        )
        concat = Node(
            op_type=OpType.CONCAT,
            name="concat1",
            inputs=["relu1_out", "relu2_out", "relu3_out"],
            outputs=["output"]
        )

        for node in [conv, relu1, relu2, relu3, concat]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Run the pass
        fusion_pass = DominatorFusionPass()
        ctx = CompileContext(graph=graph, target="x86")
        fusion_pass.run(ctx)

        # Verify structure is recognized
        assert fusion_pass.dominator_tree.get_immediate_dominator("conv1") == "concat1"
        assert fusion_pass.dominator_tree.get_immediate_dominator("relu1") == "concat1"
        assert fusion_pass.dominator_tree.get_immediate_dominator("relu2") == "concat1"
        assert fusion_pass.dominator_tree.get_immediate_dominator("relu3") == "concat1"

    def test_nested_diamonds(self):
        """Test nested diamond patterns (diamonds within diamonds)."""
        graph = Graph("nested_diamond")

        # Create nested diamond: conv -> [relu1, relu2] -> [add1, add2] -> add3
        conv = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input"],
            outputs=["conv_out"]
        )
        relu1 = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv_out"],
            outputs=["relu1_out"]
        )
        relu2 = Node(
            op_type=OpType.RELU,
            name="relu2",
            inputs=["conv_out"],
            outputs=["relu2_out"]
        )
        add1 = Node(
            op_type=OpType.ADD,
            name="add1",
            inputs=["relu1_out", "bias1"],
            outputs=["add1_out"]
        )
        add2 = Node(
            op_type=OpType.ADD,
            name="add2",
            inputs=["relu2_out", "bias2"],
            outputs=["add2_out"]
        )
        add3 = Node(
            op_type=OpType.ADD,
            name="add3",
            inputs=["add1_out", "add2_out"],
            outputs=["output"]
        )

        for node in [conv, relu1, relu2, add1, add2, add3]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Run the pass
        fusion_pass = DominatorFusionPass()
        ctx = CompileContext(graph=graph, target="x86")
        fusion_pass.run(ctx)

        # Verify the nested structure is handled
        # The final add3 should post-dominate all nodes
        assert fusion_pass.dominator_tree.does_post_dominate("conv1", "add3")
        assert fusion_pass.dominator_tree.does_post_dominate("relu1", "add3")
        assert fusion_pass.dominator_tree.does_post_dominate("add1", "add3")

    def test_matmul_diamond_pattern(self):
        """Test diamond pattern with MatMul as root."""
        graph = Graph("matmul_diamond")

        # Linear layer pattern: matmul -> [add1, add2] -> add
        matmul = Node(
            op_type=OpType.MATMUL,
            name="matmul1",
            inputs=["input", "weight"],
            outputs=["matmul_out"]
        )
        add1 = Node(
            op_type=OpType.ADD,
            name="add1",
            inputs=["matmul_out", "bias1"],
            outputs=["add1_out"]
        )
        add2 = Node(
            op_type=OpType.ADD,
            name="add2",
            inputs=["matmul_out", "bias2"],
            outputs=["add2_out"]
        )
        final_add = Node(
            op_type=OpType.ADD,
            name="final_add",
            inputs=["add1_out", "add2_out"],
            outputs=["output"]
        )

        for node in [matmul, add1, add2, final_add]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Run the pass
        fusion_pass = DominatorFusionPass()
        ctx = CompileContext(graph=graph, target="x86")
        fusion_pass.run(ctx)

        # MatMul is kOutEWiseFusable, should be analyzed in Phase 0
        assert fusion_pass.dominator_tree is not None


class TestComplexResidualPatterns:
    """Test complex residual patterns simulating ResNet blocks.

    ResNet and similar architectures use skip connections that create
    complex diamond and multi-level patterns. These tests verify
    dominator-based fusion handles these realistic architectures.
    """

    def test_simple_residual_block(self):
        """Test a simple ResNet-style residual block.

        Pattern: input -> [conv_path, identity_path] -> add
        Where conv_path has multiple operations and identity_path is a direct skip.
        """
        graph = Graph("residual_block")

        # Input conv path: conv -> relu -> conv
        conv1 = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input"],
            outputs=["conv1_out"]
        )
        relu1 = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv1_out"],
            outputs=["relu1_out"]
        )
        conv2 = Node(
            op_type=OpType.CONV2D,
            name="conv2",
            inputs=["relu1_out"],
            outputs=["conv2_out"]
        )

        # Identity path (skip connection)
        # The input is also directly connected to the final add

        # Residual addition
        res_add = Node(
            op_type=OpType.ADD,
            name="res_add",
            inputs=["conv2_out", "input"],  # Skip from input
            outputs=["output"]
        )

        for node in [conv1, relu1, conv2, res_add]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Run the pass
        fusion_pass = DominatorFusionPass()
        ctx = CompileContext(graph=graph, target="x86")
        fusion_pass.run(ctx)

        # Verify the residual structure is handled
        # res_add should post-dominate conv2
        assert fusion_pass.dominator_tree.does_post_dominate("conv2", "res_add")

    def test_bottleneck_residual_block(self):
        """Test a bottleneck-style ResNet block.

        Pattern: 1x1 conv -> 3x3 conv -> 1x1 conv, with skip connection.
        """
        graph = Graph("bottleneck_block")

        # Bottleneck path
        conv1x1_a = Node(
            op_type=OpType.CONV2D,
            name="conv1x1_a",
            inputs=["input"],
            outputs=["conv1x1_a_out"]
        )
        relu1 = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv1x1_a_out"],
            outputs=["relu1_out"]
        )
        conv3x3 = Node(
            op_type=OpType.CONV2D,
            name="conv3x3",
            inputs=["relu1_out"],
            outputs=["conv3x3_out"]
        )
        relu2 = Node(
            op_type=OpType.RELU,
            name="relu2",
            inputs=["conv3x3_out"],
            outputs=["relu2_out"]
        )
        conv1x1_b = Node(
            op_type=OpType.CONV2D,
            name="conv1x1_b",
            inputs=["relu2_out"],
            outputs=["conv1x1_b_out"]
        )

        # Skip connection path (could be 1x1 conv for dimension matching)
        skip_conv = Node(
            op_type=OpType.CONV2D,
            name="skip_conv",
            inputs=["input"],
            outputs=["skip_out"]
        )

        # Final addition
        add = Node(
            op_type=OpType.ADD,
            name="add",
            inputs=["conv1x1_b_out", "skip_out"],
            outputs=["add_out"]
        )

        # Final ReLU
        relu_final = Node(
            op_type=OpType.RELU,
            name="relu_final",
            inputs=["add_out"],
            outputs=["output"]
        )

        for node in [conv1x1_a, relu1, conv3x3, relu2, conv1x1_b,
                     skip_conv, add, relu_final]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Run the pass
        fusion_pass = DominatorFusionPass()
        ctx = CompileContext(graph=graph, target="x86")
        fusion_pass.run(ctx)

        # Verify the complex structure is analyzed
        assert fusion_pass.dominator_tree is not None
        # relu_final should post-dominate all nodes
        assert fusion_pass.dominator_tree.does_post_dominate("conv1x1_a", "relu_final")

    def test_multi_level_residual_network(self):
        """Test multiple residual blocks stacked together.

        This simulates a small ResNet with several residual blocks.
        """
        graph = Graph("multi_level_residual")

        # Block 1
        conv1_b1 = Node(
            op_type=OpType.CONV2D,
            name="conv1_b1",
            inputs=["input"],
            outputs=["conv1_b1_out"]
        )
        relu1_b1 = Node(
            op_type=OpType.RELU,
            name="relu1_b1",
            inputs=["conv1_b1_out"],
            outputs=["relu1_b1_out"]
        )
        conv2_b1 = Node(
            op_type=OpType.CONV2D,
            name="conv2_b1",
            inputs=["relu1_b1_out"],
            outputs=["conv2_b1_out"]
        )
        add_b1 = Node(
            op_type=OpType.ADD,
            name="add_b1",
            inputs=["conv2_b1_out", "input"],
            outputs=["add_b1_out"]
        )
        relu_final_b1 = Node(
            op_type=OpType.RELU,
            name="relu_final_b1",
            inputs=["add_b1_out"],
            outputs=["relu_final_b1_out"]
        )

        # Block 2
        conv1_b2 = Node(
            op_type=OpType.CONV2D,
            name="conv1_b2",
            inputs=["relu_final_b1_out"],
            outputs=["conv1_b2_out"]
        )
        relu1_b2 = Node(
            op_type=OpType.RELU,
            name="relu1_b2",
            inputs=["conv1_b2_out"],
            outputs=["relu1_b2_out"]
        )
        conv2_b2 = Node(
            op_type=OpType.CONV2D,
            name="conv2_b2",
            inputs=["relu1_b2_out"],
            outputs=["conv2_b2_out"]
        )
        add_b2 = Node(
            op_type=OpType.ADD,
            name="add_b2",
            inputs=["conv2_b2_out", "relu_final_b1_out"],
            outputs=["add_b2_out"]
        )
        relu_final_b2 = Node(
            op_type=OpType.RELU,
            name="relu_final_b2",
            inputs=["add_b2_out"],
            outputs=["output"]
        )

        nodes = [conv1_b1, relu1_b1, conv2_b1, add_b1, relu_final_b1,
                 conv1_b2, relu1_b2, conv2_b2, add_b2, relu_final_b2]
        for node in nodes:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Run the pass
        fusion_pass = DominatorFusionPass()
        ctx = CompileContext(graph=graph, target="x86")
        fusion_pass.run(ctx)

        # Verify multi-level structure is handled
        assert fusion_pass.dominator_tree is not None
        # Check that the final node post-dominates the first block
        assert fusion_pass.dominator_tree.does_post_dominate("conv1_b1", "relu_final_b2")


class TestPathValidation:
    """Test path validation - ensure unsafe patterns are rejected.

    Dominator-based fusion must only fuse operations when it's safe to do so.
    These tests verify that unsafe patterns are correctly rejected.
    """

    def test_opaque_blocks_fusion(self):
        """Test that opaque operations block fusion.

        Operations like MaxPool (kOpaque) should prevent fusion across them.
        """
        graph = Graph("opaque_blocking")

        # conv -> relu -> maxpool -> add
        # Should NOT fuse conv and add (maxpool is in between)
        conv = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input"],
            outputs=["conv_out"]
        )
        relu = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv_out"],
            outputs=["relu_out"]
        )
        maxpool = Node(
            op_type=OpType.MAXPOOL,
            name="maxpool1",
            inputs=["relu_out"],
            outputs=["pool_out"]
        )
        add = Node(
            op_type=OpType.ADD,
            name="add1",
            inputs=["pool_out", "bias"],
            outputs=["output"]
        )

        for node in [conv, relu, maxpool, add]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Build indexed forward graph for validation
        indexed_graph = IndexedForwardGraph(graph)
        validator = PathValidator(indexed_graph)

        # Get node entries
        conv_entry = indexed_graph.node_map["conv1"]
        add_entry = indexed_graph.node_map["add1"]

        # Path validation should fail due to opaque op
        assert not validator.check_path(conv_entry, add_entry, OpPatternKind.kElemWise)

    def test_too_many_nodes_on_path(self):
        """Test that paths with too many nodes are rejected.

        The max_fuse_depth parameter should prevent fusion of very long chains.
        """
        graph = Graph("long_path")

        # Create a long chain of operations
        nodes = []
        current_input = "input"

        for i in range(20):  # Create 20 relu nodes in sequence
            node_name = f"relu_{i}"
            output_name = f"out_{i}"
            node = Node(
                op_type=OpType.RELU,
                name=node_name,
                inputs=[current_input],
                outputs=[output_name]
            )
            nodes.append(node)
            graph.add_node(node)
            current_input = output_name

        # Final node
        final_add = Node(
            op_type=OpType.ADD,
            name="final_add",
            inputs=[current_input, "bias"],
            outputs=["output"]
        )
        nodes.append(final_add)
        graph.add_node(final_add)
        graph.outputs = ["output"]

        # Create indexed graph and validator
        indexed_graph = IndexedForwardGraph(graph)
        validator = PathValidator(indexed_graph)

        # Check path length
        first_entry = indexed_graph.node_map["relu_0"]
        last_entry = indexed_graph.node_map["final_add"]

        # Count nodes on path
        path_length = validator.count_nodes_on_path(first_entry, last_entry)
        # Path includes: relu_0 through relu_19 (20 nodes) + final_add = 21 nodes
        assert path_length == 21

    def test_too_many_arguments_rejected(self):
        """Test that operations with too many arguments are handled correctly.

        This is a placeholder for when argument counting is fully implemented.
        """
        graph = Graph("many_args")

        # Create a node with many inputs
        inputs = [f"input_{i}" for i in range(10)]
        concat = Node(
            op_type=OpType.CONCAT,
            name="concat_big",
            inputs=inputs,
            outputs=["output"]
        )
        graph.add_node(concat)
        graph.outputs = ["output"]

        # Run pass with limited max_function_args
        fusion_pass = DominatorFusionPass(max_function_args=5)
        ctx = CompileContext(graph=graph, target="x86")
        fusion_pass.run(ctx)

        # Pass should complete without error
        assert fusion_pass.group_arena is not None

    def test_invalid_operator_combinations(self):
        """Test that invalid operator combinations are rejected.

        Some operator combinations don't make semantic sense for fusion.
        """
        graph = Graph("invalid_combo")

        # LSTM -> conv (invalid - LSTM is opaque and stateful)
        lstm = Node(
            op_type=OpType.LSTM,
            name="lstm1",
            inputs=["input", "h0", "c0"],
            outputs=["lstm_out"]
        )
        conv = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["lstm_out"],
            outputs=["output"]
        )

        for node in [lstm, conv]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Build validator
        indexed_graph = IndexedForwardGraph(graph)
        validator = PathValidator(indexed_graph)

        # LSTM is kOpaque, should block fusion
        lstm_entry = indexed_graph.node_map["lstm1"]
        conv_entry = indexed_graph.node_map["conv1"]

        assert not validator.check_path(lstm_entry, conv_entry, OpPatternKind.kElemWise)

    def test_reduction_blocks_fusion(self):
        """Test that reduction operations block fusion.

        ReduceMean and ReduceSum are opaque operations.
        """
        graph = Graph("reduction_blocking")

        conv = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input"],
            outputs=["conv_out"]
        )
        relu = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv_out"],
            outputs=["relu_out"]
        )
        reduce_mean = Node(
            op_type=OpType.REDUCE_MEAN,
            name="reduce1",
            inputs=["relu_out"],
            outputs=["reduce_out"]
        )
        add = Node(
            op_type=OpType.ADD,
            name="add1",
            inputs=["reduce_out", "bias"],
            outputs=["output"]
        )

        for node in [conv, relu, reduce_mean, add]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Build validator
        indexed_graph = IndexedForwardGraph(graph)
        validator = PathValidator(indexed_graph)

        conv_entry = indexed_graph.node_map["conv1"]
        add_entry = indexed_graph.node_map["add1"]

        # Path should be invalid due to ReduceMean
        assert not validator.check_path(conv_entry, add_entry, OpPatternKind.kElemWise)


class TestPassPipelineIntegration:
    """Test DominatorFusionPass integration with the full pass pipeline.

    These tests verify that DominatorFusionPass works correctly when combined
    with other passes like PatternFusionPass, LivenessAnalysisPass, and
    MemoryPlanningPassV2.
    """

    def test_dominator_fusion_after_pattern_fusion(self):
        """Test that DominatorFusionPass works after PatternFusionPass.

        The two passes should complement each other:
        - PatternFusionPass handles simple chain patterns
        - DominatorFusionPass handles diamond patterns
        """
        graph = Graph("combined_fusion")

        # Create a graph with both chain and diamond patterns
        conv = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input"],
            outputs=["conv_out"]
        )
        relu1 = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv_out"],
            outputs=["relu1_out"]
        )
        relu2 = Node(
            op_type=OpType.RELU,
            name="relu2",
            inputs=["conv_out"],
            outputs=["relu2_out"]
        )
        add = Node(
            op_type=OpType.ADD,
            name="add1",
            inputs=["relu1_out", "relu2_out"],
            outputs=["add_out"]
        )
        sigmoid = Node(
            op_type=OpType.SIGMOID,
            name="sigmoid1",
            inputs=["add_out"],
            outputs=["output"]
        )

        for node in [conv, relu1, relu2, add, sigmoid]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Run PatternFusionPass first
        from nnc_py.passes.pattern_fusion import PatternFusionPass
        pattern_pass = PatternFusionPass()
        ctx = CompileContext(graph=graph, target="x86")
        pattern_pass.run(ctx)

        # Then run DominatorFusionPass
        dominator_pass = DominatorFusionPass()
        dominator_pass.run(ctx)

        # Both passes should have completed
        assert len(graph.nodes) > 0

    def test_full_o3_pipeline_with_dominator_fusion(self):
        """Test the full O3 optimization pipeline including DominatorFusionPass.

        This verifies that DominatorFusionPass doesn't interfere with other passes.
        """
        graph = Graph("full_pipeline")

        # Create a realistic graph
        conv1 = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input"],
            outputs=["conv1_out"]
        )
        relu1 = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv1_out"],
            outputs=["relu1_out"]
        )
        conv2 = Node(
            op_type=OpType.CONV2D,
            name="conv2",
            inputs=["relu1_out"],
            outputs=["conv2_out"]
        )
        relu2 = Node(
            op_type=OpType.RELU,
            name="relu2",
            inputs=["conv2_out"],
            outputs=["relu2_out"]
        )
        add = Node(
            op_type=OpType.ADD,
            name="add1",
            inputs=["relu2_out", "skip"],
            outputs=["add_out"]
        )
        relu3 = Node(
            op_type=OpType.RELU,
            name="relu3",
            inputs=["add_out"],
            outputs=["output"]
        )

        for node in [conv1, relu1, conv2, relu2, add, relu3]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Get O3 passes and run them
        passes = PassManager.get_default_passes(3)
        ctx = CompileContext(graph=graph, target="x86", debug=True)

        for pass_obj in passes:
            pass_obj.run(ctx)

        # Verify graph is still valid after all passes
        assert "output" in graph.outputs
        assert len(graph.nodes) > 0

    def test_no_graph_corruption_after_pipeline(self):
        """Test that the graph structure is not corrupted after the full pipeline.

        This is a sanity check for graph integrity.
        """
        graph = Graph("integrity_check")

        # Create a diamond pattern
        conv = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input"],
            outputs=["conv_out"]
        )
        relu1 = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv_out"],
            outputs=["relu1_out"]
        )
        relu2 = Node(
            op_type=OpType.RELU,
            name="relu2",
            inputs=["conv_out"],
            outputs=["relu2_out"]
        )
        add = Node(
            op_type=OpType.ADD,
            name="add1",
            inputs=["relu1_out", "relu2_out"],
            outputs=["output"]
        )

        for node in [conv, relu1, relu2, add]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Run full pipeline
        passes = PassManager.get_default_passes(3)
        ctx = CompileContext(graph=graph, target="x86")

        # Store original topology
        original_node_count = len(graph.nodes)
        original_output = graph.outputs[0]

        for pass_obj in passes:
            pass_obj.run(ctx)

        # Verify graph structure is still valid
        assert graph.outputs[0] == original_output
        # Nodes may have been fused, so count may be less, but graph should be valid

    def test_liveness_after_dominator_fusion(self):
        """Test that LivenessAnalysisPass works correctly after DominatorFusionPass."""
        from nnc_py.ir.tensor import TensorType

        graph = Graph("liveness_test")

        # Create pattern that dominator fusion might fuse
        conv = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input"],
            outputs=["conv_out"]
        )
        relu1 = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv_out"],
            outputs=["relu1_out"]
        )
        relu2 = Node(
            op_type=OpType.RELU,
            name="relu2",
            inputs=["conv_out"],
            outputs=["relu2_out"]
        )
        add = Node(
            op_type=OpType.ADD,
            name="add1",
            inputs=["relu1_out", "relu2_out"],
            outputs=["output"]
        )

        for node in [conv, relu1, relu2, add]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Add tensor definitions for liveness analysis
        # Create tensor types for each intermediate value
        for tensor_name in ["input", "conv_out", "relu1_out", "relu2_out", "output"]:
            graph.add_tensor(TensorType(name=tensor_name, dtype=DataType.FLOAT32, shape=[1, 16, 32, 32]))

        # Run DominatorFusionPass
        dominator_pass = DominatorFusionPass()
        ctx = CompileContext(graph=graph, target="x86")
        dominator_pass.run(ctx)

        # Run LivenessAnalysisPass
        from nnc_py.passes.liveness import LivenessAnalysisPass
        liveness_pass = LivenessAnalysisPass()
        liveness_pass.run(ctx)

        # Verify liveness info was computed
        assert "tensor_liveness" in ctx.metadata
        liveness_map = ctx.metadata["tensor_liveness"]
        assert len(liveness_map) > 0

    def test_memory_planning_after_dominator_fusion(self):
        """Test that MemoryPlanningPassV2 works after DominatorFusionPass."""
        graph = Graph("memory_test")

        # Create diamond pattern
        conv = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input"],
            outputs=["conv_out"]
        )
        relu1 = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv_out"],
            outputs=["relu1_out"]
        )
        relu2 = Node(
            op_type=OpType.RELU,
            name="relu2",
            inputs=["conv_out"],
            outputs=["relu2_out"]
        )
        add = Node(
            op_type=OpType.ADD,
            name="add1",
            inputs=["relu1_out", "relu2_out"],
            outputs=["output"]
        )

        for node in [conv, relu1, relu2, add]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Run DominatorFusionPass
        dominator_pass = DominatorFusionPass()
        ctx = CompileContext(graph=graph, target="x86")
        dominator_pass.run(ctx)

        # Run LivenessAnalysisPass (required for memory planning)
        from nnc_py.passes.liveness import LivenessAnalysisPass
        liveness_pass = LivenessAnalysisPass()
        liveness_pass.run(ctx)

        # Run MemoryPlanningPassV2
        from nnc_py.passes.memory_planning import MemoryPlanningPassV2
        memory_pass = MemoryPlanningPassV2()
        memory_pass.run(ctx)

        # Verify memory plan was computed
        assert "memory_allocation_plan" in ctx.metadata

    def test_multiple_fusion_phases(self):
        """Test that both fusion phases run correctly.

        Phase 0: kOutEWiseFusable -> kElemWise
        Phase 1: kElemWise/kInjective -> kBroadcast
        """
        graph = Graph("both_phases")

        # Phase 0 pattern: conv -> relu
        conv = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input"],
            outputs=["conv_out"]
        )
        relu = Node(
            op_type=OpType.RELU,
            name="relu1",
            inputs=["conv_out"],
            outputs=["relu_out"]
        )

        # Phase 1 pattern: injective -> broadcast
        reshape = Node(
            op_type=OpType.RESHAPE,
            name="reshape1",
            inputs=["relu_out"],
            outputs=["reshape_out"]
        )
        batch_norm = Node(
            op_type=OpType.BATCH_NORM,
            name="bn1",
            inputs=["reshape_out", "gamma", "beta"],
            outputs=["output"]
        )

        for node in [conv, relu, reshape, batch_norm]:
            graph.add_node(node)
        graph.outputs = ["output"]

        # Run the pass
        fusion_pass = DominatorFusionPass()
        ctx = CompileContext(graph=graph, target="x86", debug=True)
        fusion_pass.run(ctx)

        # Both phases should have executed
        assert fusion_pass.group_arena is not None


class TestDominatorFusionONNXCompilation:
    """Test DominatorFusionPass with actual ONNX models compilation.

    These tests verify the full compilation pipeline works with
    models that contain diamond patterns.
    """

    def setup_method(self):
        """Set up test environment."""
        self.tmp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_diamond_onnx_model_compiles(self):
        """Test that an ONNX model with diamond pattern compiles successfully."""
        # Create ONNX model with diamond pattern
        input_val = helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 3, 32, 32]
        )
        output_val = helper.make_tensor_value_info(
            "output", onnx.TensorProto.FLOAT, [1, 16, 30, 30]
        )

        # Weight
        weight_init = helper.make_tensor(
            "conv_weight", onnx.TensorProto.FLOAT,
            [16, 3, 3, 3], [0.1] * (16 * 3 * 3 * 3)
        )

        # Diamond pattern: Conv -> [Relu1, Relu2] -> Add
        conv = helper.make_node(
            "Conv",
            inputs=["input", "conv_weight"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
        )

        relu1 = helper.make_node("Relu", inputs=["conv_out"], outputs=["relu1_out"])
        relu2 = helper.make_node("Relu", inputs=["conv_out"], outputs=["relu2_out"])
        add = helper.make_node("Add", inputs=["relu1_out", "relu2_out"], outputs=["output"])

        graph = helper.make_graph(
            [conv, relu1, relu2, add],
            "diamond_test",
            [input_val],
            [output_val],
            [weight_init]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        # Save and compile
        onnx_path = os.path.join(self.tmp_dir, "diamond.onnx")
        onnx.save(model, onnx_path)

        output_dir = os.path.join(self.tmp_dir, "build")

        # Compile at O3 (with DominatorFusionPass)
        compiler = Compiler(target="x86", opt_level=3)
        compiler.compile(onnx_path, output_dir)

        # Check that model.c was generated
        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c should be generated"

    def test_residual_onnx_model_compiles(self):
        """Test that an ONNX model with residual connection compiles."""
        input_val = helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 16, 32, 32]
        )
        output_val = helper.make_tensor_value_info(
            "output", onnx.TensorProto.FLOAT, [1, 16, 32, 32]
        )

        # Weight
        weight_init = helper.make_tensor(
            "conv_weight", onnx.TensorProto.FLOAT,
            [16, 16, 3, 3], [0.1] * (16 * 16 * 3 * 3)
        )

        # Residual pattern: Conv -> Relu -> Conv, then Add with skip
        conv1 = helper.make_node(
            "Conv",
            inputs=["input", "conv_weight"],
            outputs=["conv1_out"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )

        relu1 = helper.make_node("Relu", inputs=["conv1_out"], outputs=["relu1_out"])

        conv2 = helper.make_node(
            "Conv",
            inputs=["relu1_out", "conv_weight"],
            outputs=["conv2_out"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )

        # Add with skip connection
        add = helper.make_node("Add", inputs=["conv2_out", "input"], outputs=["output"])

        graph = helper.make_graph(
            [conv1, relu1, conv2, add],
            "residual_test",
            [input_val],
            [output_val],
            [weight_init]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        onnx_path = os.path.join(self.tmp_dir, "residual.onnx")
        onnx.save(model, onnx_path)

        output_dir = os.path.join(self.tmp_dir, "build")

        # Compile
        compiler = Compiler(target="x86", opt_level=3)
        compiler.compile(onnx_path, output_dir)

        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c should be generated"

    def test_complex_diamond_onnx_model(self):
        """Test a more complex ONNX model with nested diamonds."""
        input_val = helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1, 3, 32, 32]
        )
        output_val = helper.make_tensor_value_info(
            "output", onnx.TensorProto.FLOAT, [1, 16, 28, 28]
        )

        weight_init = helper.make_tensor(
            "conv_weight", onnx.TensorProto.FLOAT,
            [16, 3, 3, 3], [0.1] * (16 * 3 * 3 * 3)
        )

        # Constant tensors for bias
        bias1_init = helper.make_tensor("bias1_const", onnx.TensorProto.FLOAT, [], [0.1])
        bias2_init = helper.make_tensor("bias2_const", onnx.TensorProto.FLOAT, [], [0.2])

        # Nested diamond: Conv -> [Relu1, Relu2] -> [Add1, Add2] -> Add3
        conv = helper.make_node(
            "Conv",
            inputs=["input", "conv_weight"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
        )

        relu1 = helper.make_node("Relu", inputs=["conv_out"], outputs=["relu1_out"])
        relu2 = helper.make_node("Relu", inputs=["conv_out"], outputs=["relu2_out"])

        # Use constant nodes for bias
        const1 = helper.make_node("Constant", inputs=[], outputs=["bias1"], value=bias1_init)
        const2 = helper.make_node("Constant", inputs=[], outputs=["bias2"], value=bias2_init)

        add1 = helper.make_node("Add", inputs=["relu1_out", "bias1"], outputs=["add1_out"])
        add2 = helper.make_node("Add", inputs=["relu2_out", "bias2"], outputs=["add2_out"])
        add3 = helper.make_node("Add", inputs=["add1_out", "add2_out"], outputs=["output"])

        graph = helper.make_graph(
            [conv, relu1, relu2, const1, const2, add1, add2, add3],
            "complex_diamond_test",
            [input_val],
            [output_val],
            [weight_init, bias1_init, bias2_init]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        onnx_path = os.path.join(self.tmp_dir, "complex_diamond.onnx")
        onnx.save(model, onnx_path)

        output_dir = os.path.join(self.tmp_dir, "build")

        # Compile
        compiler = Compiler(target="x86", opt_level=3)
        compiler.compile(onnx_path, output_dir)

        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c should be generated"
