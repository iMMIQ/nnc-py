"""Operator splitting analysis pass.

This pass analyzes the computation graph to identify operators
whose output tensors exceed available memory and calculates
how they should be split.
"""

import math
from typing import List, Tuple

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.split_rules import (
    SplitInfo,
    SplitPlan,
    SplitRegistry,
    SplitAxisBehavior,
    CascadeInfo,
)
from nnc_py.passes.base import PassBase


# Threshold for triggering split: tensor size > max_memory * SPLIT_THRESHOLD
SPLIT_THRESHOLD = 0.5


class SplitAnalysisPass(PassBase):
    """Analyzes which operators need splitting based on memory constraints."""

    @property
    def name(self) -> str:
        return "split_analysis"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute split analysis."""
        # Check if max_memory is set
        max_memory = ctx.metadata.get("max_memory")
        if max_memory is None:
            return  # No memory limit, no need to split

        # Find nodes that need splitting
        candidates = self._find_split_candidates(ctx, max_memory)

        if not candidates:
            # Create empty plan
            ctx.metadata["split_plan"] = SplitPlan()
            return

        # Calculate splits for each candidate
        splits = self._calculate_splits(ctx, candidates, max_memory)

        # Create and store the split plan
        split_plan = SplitPlan(splits=splits)

        # Analyze cascading splits
        self._analyze_cascading_splits(ctx, split_plan)

        ctx.metadata["split_plan"] = split_plan

        if ctx.debug:
            self._log_summary(ctx, split_plan)

    def _find_split_candidates(
        self, ctx: CompileContext, max_memory: int
    ) -> List[Tuple]:
        """Find operators whose outputs exceed memory threshold.

        Returns:
            List of (node, output_name, tensor_size) tuples.
        """
        candidates = []
        threshold = max_memory * SPLIT_THRESHOLD

        for node in ctx.graph.topological_sort():
            for output_name in node.outputs:
                tensor = ctx.graph.get_tensor(output_name)
                size = tensor.byte_size()

                # Skip unknown sizes (dynamic shapes)
                if size < 0:
                    continue

                # Check if tensor exceeds threshold
                if size > threshold:
                    candidates.append((node, output_name, size))

        # Sort by size (largest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def _calculate_splits(
        self,
        ctx: CompileContext,
        candidates: List[Tuple],
        max_memory: int
    ) -> List[SplitInfo]:
        """Calculate split parameters for each candidate.

        Args:
            ctx: Compilation context
            candidates: List of (node, output_name, tensor_size) tuples
            max_memory: Maximum memory limit

        Returns:
            List of SplitInfo objects
        """
        splits = []

        for node, output_name, tensor_size in candidates:
            # Check if this node has a split rule
            rule = SplitRegistry.get_rule(node.op_type)
            if rule is None:
                continue  # No rule available, cannot split

            # Get output tensor
            tensor = ctx.graph.get_tensor(output_name)

            # Select best axis to split
            split_axis = self._select_split_axis(tensor, rule)
            if split_axis is None:
                continue  # No splittable axis found

            # Calculate number of splits needed
            axis_size = tensor.shape.dims[split_axis]
            num_splits = self._calculate_num_splits(
                tensor_size, max_memory, axis_size
            )

            # Calculate chunk sizes
            chunk_size = axis_size // num_splits
            remainder = axis_size % num_splits
            chunk_sizes = [
                chunk_size + 1 if i < remainder else chunk_size
                for i in range(num_splits)
            ]

            # Create SplitInfo
            split_info = SplitInfo(
                original_node=node,
                split_axis=split_axis,
                num_splits=num_splits,
                chunk_sizes=chunk_sizes,
            )
            splits.append(split_info)

        return splits

    def _select_split_axis(
        self, tensor, rule
    ) -> int:
        """Select the best axis for splitting.

        Chooses the largest fully-splittable axis to minimize
        the number of splits needed.

        Args:
            tensor: Output tensor
            rule: Operator split rule

        Returns:
            Best axis index, or None if no axis is splittable
        """
        # Get splittable axes for the first input
        splittable = [
            r.axis_index
            for r in rule.input_split_rules[0]
            if r.behavior == SplitAxisBehavior.FULLY_SPLITTABLE
        ]

        if not splittable:
            return None

        # Filter to axes that exist in the tensor
        valid_axes = [
            ax for ax in splittable
            if 0 <= ax < len(tensor.shape.dims)
        ]

        if not valid_axes:
            return None

        # Select the axis with the largest dimension
        # to minimize the number of splits
        return max(valid_axes, key=lambda a: tensor.shape.dims[a])

    def _calculate_num_splits(
        self, tensor_size: int, max_memory: int, axis_size: int
    ) -> int:
        """Calculate the minimum number of splits needed.

        Args:
            tensor_size: Total tensor size in bytes
            max_memory: Maximum memory limit
            axis_size: Size of the selected axis

        Returns:
            Number of splits (at least 2)
        """
        # Target size per split
        target_size = max_memory * SPLIT_THRESHOLD

        # Calculate splits needed
        num_splits = max(2, math.ceil(tensor_size / target_size))

        # Limit to axis size (can't split more elements than exist)
        return min(num_splits, axis_size)

    def _analyze_cascading_splits(
        self, ctx: CompileContext, split_plan: SplitPlan
    ) -> None:
        """Analyze which dependent operators also need splitting.

        When an operator is split, its consumers may also need to be split
        to handle the split outputs correctly.

        Args:
            ctx: Compilation context
            split_plan: Split plan to add cascades to
        """
        graph = ctx.graph

        for split_info in split_plan.splits:
            source_node = split_info.original_node
            source_axis = split_info.split_axis

            # Find all consumers of the split node's outputs
            for output_name in source_node.outputs:
                consumers = graph.get_consumers(output_name)

                for consumer in consumers:
                    # Check if consumer has a split rule with propagation
                    rule = SplitRegistry.get_rule(consumer.op_type)
                    if rule is None or rule.propagate_split is None:
                        continue  # No propagation

                    # Calculate the target axis for the consumer
                    target_axis = rule.propagate_split(source_axis)
                    if target_axis is None:
                        continue  # Propagation returned None

                    # Create cascade info
                    cascade = CascadeInfo(
                        source_node=source_node,
                        target_node=consumer,
                        source_axis=source_axis,
                        target_axis=target_axis,
                        required_splits=split_info.num_splits,
                    )
                    split_plan.cascades.append(cascade)

    def _log_summary(self, ctx: CompileContext, plan: SplitPlan) -> None:
        """Log split analysis summary."""
        if not plan.splits:
            print("\n[SplitAnalysis] No operators need splitting.")
            return

        print(f"\n[SplitAnalysis] Found {len(plan.splits)} operators to split:")
        for i, split_info in enumerate(plan.splits, 1):
            node = split_info.original_node
            print(f"  {i}. {node.name} ({node.op_type.value})")
            print(f"     Axis: {split_info.split_axis}, Splits: {split_info.num_splits}")
            print(f"     Chunk sizes: {split_info.chunk_sizes}")
