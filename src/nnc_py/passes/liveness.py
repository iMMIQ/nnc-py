"""Liveness analysis for memory planning.

This pass analyzes the computation graph to determine the lifetime range
of each tensor - when it becomes live (produced) and when it dies (last use).

This information is used for memory reuse in the MemoryPlanningPass.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.passes.base import PassBase


@dataclass
class TensorLiveness:
    """Liveness information for a single tensor."""

    tensor_name: str
    live_start: int  # Index of node that produces this tensor
    live_end: int    # Index of last node that uses this tensor
    is_input: bool = False
    is_output: bool = False
    is_constant: bool = False

    @property
    def lifetime_range(self) -> int:
        """Return the length of this tensor's lifetime."""
        return self.live_end - self.live_start + 1


class LivenessAnalysisPass(PassBase):
    """Analyze tensor lifetimes for memory planning.

    This pass computes for each tensor:
    - live_start: The index of the node that produces the tensor
    - live_end: The index of the last node that consumes the tensor

    Tensors with non-overlapping lifetimes can share the same memory.
    """

    @property
    def name(self) -> str:
        return "LivenessAnalysis"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute liveness analysis on the computation graph."""
        graph = ctx.graph
        nodes = graph.topological_sort()

        # Build node index mapping
        node_index = {node.name: i for i, node in enumerate(nodes)}

        # Compute liveness for all tensors
        liveness_map: Dict[str, TensorLiveness] = {}

        # Process all tensors in the graph
        for tensor_name in graph.tensors:
            liveness = self._analyze_tensor(graph, tensor_name, node_index, nodes)
            liveness_map[tensor_name] = liveness

        # Store results in context metadata
        ctx.metadata["tensor_liveness"] = liveness_map

        # Log summary if debug mode is on
        if ctx.debug:
            self._log_summary(liveness_map, nodes)

    def _analyze_tensor(
        self,
        graph: Graph,
        tensor_name: str,
        node_index: Dict[str, int],
        nodes: List,
    ) -> TensorLiveness:
        """Analyze liveness for a single tensor."""
        # Find producers and consumers
        producers = graph.get_producers(tensor_name)
        consumers = graph.get_consumers(tensor_name)

        # Determine if this is an input, output, or constant tensor
        is_input = tensor_name in graph.inputs
        is_output = tensor_name in graph.outputs
        is_constant = tensor_name in graph.constants

        # Compute live_start
        if producers:
            live_start = node_index[producers[0].name]
        elif is_input:
            live_start = 0  # Inputs are live from the start
        else:
            # Constant or other - assume live from start
            live_start = 0

        # Compute live_end
        if consumers:
            live_end = max(node_index[c.name] for c in consumers)
        else:
            # No consumers - must be an output
            live_end = len(nodes) - 1

        return TensorLiveness(
            tensor_name=tensor_name,
            live_start=live_start,
            live_end=live_end,
            is_input=is_input,
            is_output=is_output,
            is_constant=is_constant,
        )

    def _log_summary(self, liveness_map: Dict[str, TensorLiveness], nodes: List) -> None:
        """Log a summary of liveness analysis."""
        print(f"\n{'='*60}")
        print(f"Liveness Analysis Summary")
        print(f"{'='*60}")
        print(f"Total nodes: {len(nodes)}")
        print(f"Total tensors: {len(liveness_map)}")
        print()

        # Group by type
        inputs = [t for t in liveness_map.values() if t.is_input]
        outputs = [t for t in liveness_map.values() if t.is_output]
        constants = [t for t in liveness_map.values() if t.is_constant]
        intermediates = [
            t for t in liveness_map.values()
            if not t.is_input and not t.is_output and not t.is_constant
        ]

        print(f"Inputs: {len(inputs)}")
        print(f"Outputs: {len(outputs)}")
        print(f"Constants: {len(constants)}")
        print(f"Intermediates: {len(intermediates)}")
        print()

        # Show tensors with their lifetimes
        print(f"{'Tensor':<20} {'Start':>6} {'End':>6} {'Lifetime':>8} {'Type':>12}")
        print(f"{'-'*60}")

        for liveness in sorted(liveness_map.values(), key=lambda x: x.live_start):
            t_type = []
            if liveness.is_input:
                t_type.append("input")
            if liveness.is_output:
                t_type.append("output")
            if liveness.is_constant:
                t_type.append("const")
            if not t_type:
                t_type.append("intermediate")

            print(
                f"{liveness.tensor_name:<20} "
                f"{liveness.live_start:>6} "
                f"{liveness.live_end:>6} "
                f"{liveness.lifetime_range:>8} "
                f"{','.join(t_type):>12}"
            )

        # Find peak liveness (maximum number of simultaneously live tensors)
        self._log_peak_liveness(liveness_map, len(nodes))

    def _log_peak_liveness(self, liveness_map: Dict[str, TensorLiveness], num_nodes: int) -> None:
        """Calculate and log the peak number of live tensors."""
        peak_live = 0
        peak_at = 0

        for i in range(num_nodes):
            live_count = sum(
                1 for l in liveness_map.values()
                if l.live_start <= i <= l.live_end
            )
            if live_count > peak_live:
                peak_live = live_count
                peak_at = i

        print()
        print(f"Peak liveness: {peak_live} tensors live at node {peak_at}")
        print(f"{'='*60}")


def get_liveness(ctx: CompileContext, tensor_name: str) -> TensorLiveness:
    """Get liveness information for a tensor from the context.

    Args:
        ctx: Compilation context
        tensor_name: Name of the tensor

    Returns:
        TensorLiveness object for the tensor

    Raises:
        KeyError: If liveness analysis hasn't been run or tensor not found
    """
    liveness_map = ctx.metadata.get("tensor_liveness")
    if liveness_map is None:
        raise RuntimeError("LivenessAnalysisPass must be run before calling get_liveness")
    return liveness_map[tensor_name]
