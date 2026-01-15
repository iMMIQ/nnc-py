"""Operator splitting transform pass.

This pass transforms the computation graph by splitting operators
into multiple smaller operators based on the split plan.
"""

from typing import Dict, List

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.split_rules import (
    SplitInfo,
    SplitPlan,
    SplitRegistry,
)
from nnc_py.passes.base import PassBase


class SplitTransformPass(PassBase):
    """Transforms the graph to split operators."""

    @property
    def name(self) -> str:
        return "split_transform"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute graph transformation."""
        split_plan = ctx.metadata.get("split_plan")
        if not split_plan or not split_plan.splits:
            return  # Nothing to split

        graph = ctx.graph

        # Process each split in the plan
        for split_info in split_plan.splits:
            self._split_node(ctx, split_info)

        # Validate the transformed graph
        self._validate_graph(ctx)

        if ctx.debug:
            self._log_summary(ctx, split_plan)

    def _split_node(self, ctx: CompileContext, split_info: SplitInfo) -> None:
        """Split a single node into multiple nodes.

        Args:
            ctx: Compilation context
            split_info: Information about how to split the node
        """
        graph = ctx.graph
        original = split_info.original_node

        # Create split nodes
        split_nodes = []
        for i in range(split_info.num_splits):
            split_node = self._create_split_node(
                original, i, split_info
            )
            graph.add_node(split_node)
            split_nodes.append(split_node)

        # Store split nodes in split_info for reference
        split_info.split_nodes = split_nodes

        # Create split output tensors
        self._create_split_output_tensors(ctx, original, split_info)

        # Handle input slicing if needed
        self._handle_input_slices(ctx, original, split_info)

        # Handle consumer rewiring
        self._rewire_consumers(ctx, original, split_info)

        # Remove the original node
        self._remove_original_node(graph, original)

    def _create_split_node(
        self,
        original,
        index: int,
        split_info: SplitInfo
    ):
        """Create a split instance of a node.

        Args:
            original: Original node to split
            index: Index of this split (0 to num_splits-1)
            split_info: Split information

        Returns:
            New split node
        """
        from nnc_py.ir.node import Node

        new_name = f"{original.name}_split{index}"
        new_outputs = [f"{out}_split{index}" for out in original.outputs]

        # Copy attributes, adding split metadata
        new_attrs = original.attrs.copy()
        new_attrs["_split_index"] = index
        new_attrs["_split_axis"] = split_info.split_axis
        new_attrs["_split_chunk"] = split_info.chunk_sizes[index] if index < len(split_info.chunk_sizes) else 0

        # Create metadata
        new_metadata = {
            "is_split": True,
            "original_node": original.name,
            "split_axis": split_info.split_axis,
            "split_index": index,
        }

        return Node(
            op_type=original.op_type,
            name=new_name,
            inputs=original.inputs.copy(),  # Will be remapped later
            outputs=new_outputs,
            attrs=new_attrs,
            metadata=new_metadata,
        )

    def _create_split_output_tensors(
        self,
        ctx: CompileContext,
        original,
        split_info: SplitInfo
    ) -> None:
        """Create output tensors for split nodes.

        Args:
            ctx: Compilation context
            original: Original node
            split_info: Split information
        """
        from nnc_py.ir.tensor import TensorType, TensorShape

        graph = ctx.graph

        # Get original output tensor (first output)
        original_output_name = original.outputs[0]
        original_tensor = graph.get_tensor(original_output_name)

        # Calculate new shape for split outputs
        new_shape_dims = original_tensor.shape.dims.copy()
        axis = split_info.split_axis

        for i, chunk_size in enumerate(split_info.chunk_sizes):
            # Update the split axis dimension
            new_shape_dims[axis] = chunk_size

            # Create new tensor for this split output
            split_tensor = TensorType(
                dtype=original_tensor.dtype,
                shape=TensorShape(dims=new_shape_dims.copy()),
                name=f"{original_output_name}_split{i}",
            )
            graph.add_tensor(split_tensor)

    def _handle_input_slices(
        self,
        ctx: CompileContext,
        original,
        split_info: SplitInfo
    ) -> None:
        """Handle input tensor slicing for split nodes.

        This method updates the inputs of split nodes to use the correct
        input tensors. For now, we keep the original inputs since
        actual slicing will be handled during code generation.

        Args:
            ctx: Compilation context
            original: Original node
            split_info: Split information
        """
        graph = ctx.graph

        # For each split node, update its inputs
        for i, split_node in enumerate(split_info.split_nodes):
            # Create split input names
            new_inputs = []
            for input_name in original.inputs:
                # Check if this input should be sliced
                # (inputs marked as reused should not be sliced)
                # For now, we keep original input names
                # Actual slicing logic will be in code generation
                new_inputs.append(input_name)

            split_node.inputs = new_inputs

    def _remove_original_node(self, graph: Graph, original) -> None:
        """Remove the original node from the graph.

        Args:
            graph: The computation graph
            original: Original node to remove
        """
        if original.name in graph.nodes:
            del graph.nodes[original.name]
            # Invalidate cached graph
            graph._nx_graph = None

    def _validate_graph(self, ctx: CompileContext) -> None:
        """Validate the transformed graph.

        Args:
            ctx: Compilation context
        """
        graph = ctx.graph

        # Rebuild the graph to update dependencies
        graph._nx_graph = None

        # Verify topological sort works (no cycles)
        try:
            graph.topological_sort()
        except Exception as e:
            raise RuntimeError(f"Graph transformation created a cycle: {e}")

    def _log_summary(self, ctx: CompileContext, plan: SplitPlan) -> None:
        """Log transformation summary.

        Args:
            ctx: Compilation context
            plan: Split plan that was applied
        """
        print(f"\n[SplitTransform] Transformed {len(plan.splits)} operators:")
        for split_info in plan.splits:
            print(f"  {split_info.original_node.name} -> {split_info.num_splits} nodes")
            for node in split_info.split_nodes:
                print(f"    - {node.name}")

    def _rewire_consumers(
        self,
        ctx: CompileContext,
        original,
        split_info: SplitInfo
    ) -> None:
        """Rewire consumers of the original node to split outputs.

        Args:
            ctx: Compilation context
            original: Original node that was split
            split_info: Split information
        """
        graph = ctx.graph

        for output_name in original.outputs:
            # Check if this is a graph output
            if output_name in graph.outputs:
                # Graph output handling: update to list of split outputs
                # (For now, we keep the original output name and let later passes handle it)
                continue

            # Find consumers of this output
            consumers = graph.get_consumers(output_name)
            if not consumers:
                continue

            # Process each consumer
            for consumer in consumers:
                should_cascade = self._should_cascade_split(consumer, split_info)

                if should_cascade:
                    # Consumer also needs to be split
                    self._cascade_split_to_consumer(ctx, consumer, output_name, split_info)
                else:
                    # Consumer doesn't split, connect to first split output
                    self._connect_to_split_output(ctx, consumer, output_name, split_info)

    def _should_cascade_split(
        self,
        consumer,
        split_info: SplitInfo
    ) -> bool:
        """Check if a consumer should be split along with its input.

        Args:
            consumer: Consumer node
            split_info: Split information

        Returns:
            True if consumer should be split
        """
        # Check if consumer has a split rule
        rule = SplitRegistry.get_rule(consumer.op_type)
        if rule is None or not rule.input_split_rules:
            return False

        # Check if consumer has valid input split rules
        if not rule.input_split_rules[0]:
            return False

        return True

    def _cascade_split_to_consumer(
        self,
        ctx: CompileContext,
        consumer,
        output_name: str,
        source_split_info: SplitInfo
    ) -> None:
        """Create split versions of a consumer node.

        Args:
            ctx: Compilation context
            consumer: Original consumer node
            output_name: Name of the input tensor to consumer
            source_split_info: Split info of the source node
        """
        from nnc_py.ir.node import Node
        from nnc_py.ir.tensor import TensorType, TensorShape

        graph = ctx.graph

        # Create split versions of the consumer
        num_splits = source_split_info.num_splits
        consumer_split_nodes = []

        for i in range(num_splits):
            split_output_name = f"{consumer.outputs[0]}_split{i}"
            split_input_name = f"{output_name}_split{i}"

            # Create output tensor for this split
            original_tensor = graph.get_tensor(consumer.outputs[0])
            if original_tensor:
                split_tensor = TensorType(
                    dtype=original_tensor.dtype,
                    shape=TensorShape(dims=original_tensor.shape.dims.copy()),
                    name=split_output_name,
                )
                graph.add_tensor(split_tensor)

            # Create split consumer node
            consumer_split = Node(
                op_type=consumer.op_type,
                name=f"{consumer.name}_split{i}",
                inputs=[split_input_name],
                outputs=[split_output_name],
                attrs=consumer.attrs.copy(),
                metadata={
                    "is_split": True,
                    "original_node": consumer.name,
                },
            )
            graph.add_node(consumer_split)
            consumer_split_nodes.append(consumer_split)

        # Remove original consumer
        if consumer.name in graph.nodes:
            del graph.nodes[consumer.name]
            graph._nx_graph = None

    def _connect_to_split_output(
        self,
        ctx: CompileContext,
        consumer,
        output_name: str,
        split_info: SplitInfo
    ) -> None:
        """Connect a consumer to a split output.

        For consumers without split rules, we connect to the first split output.

        Args:
            ctx: Compilation context
            consumer: Consumer node
            output_name: Original output name
            split_info: Split information
        """
        # Connect to the first split output
        split_output_name = f"{output_name}_split0"

        # Update consumer's input
        for i, inp in enumerate(consumer.inputs):
            if inp == output_name:
                consumer.inputs[i] = split_output_name
                break

