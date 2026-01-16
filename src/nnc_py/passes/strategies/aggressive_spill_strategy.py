"""Aggressive spill memory allocation strategy.

This module implements a radical approach to memory allocation:
- Only the current operator's inputs and outputs are in fast memory
- Everything else can be spilled to slow memory
- No liveness analysis needed - just check per-operator memory requirements
- Minimizes memory fragmentation and ensures predictability
"""

from typing import Dict, List, Optional, Set

from nnc_py.ir.context import CompileContext
from nnc_py.passes.liveness import TensorLiveness
from nnc_py.passes.memory_plan import MemoryBuffer
from nnc_py.passes.memory_strategy import (
    AllocationStrategy,
    MemoryAllocationPlan,
    MemoryAllocationStrategy,
    TensorAllocation,
    SpillPoint,
    ReloadPoint,
)


class AggressiveSpillStrategy(MemoryAllocationStrategy):
    """Aggressive spill strategy for memory allocation.

    Core principle: Since we have spill/reload capability, we don't need
    complex liveness analysis. We can be very aggressive about spilling:
    - For each operator, only its inputs and outputs need to be in fast memory
    - All other tensors can be in slow memory
    - This guarantees that as long as single-operator inputs+outputs fit,
      the entire model can be compiled

    Benefits:
    1. Simple - no complex interval overlap analysis
    2. Predictable - peak memory = max(operator inputs + outputs)
    3. No fragmentation - memory is freed immediately after use
    4. Always works - if any operator can run, the whole model can run
    """

    DEFAULT_ALIGNMENT = 16

    @property
    def name(self) -> str:
        return "aggressive_spill"

    @property
    def strategy_type(self) -> AllocationStrategy:
        return AllocationStrategy.AGGRESSIVE_SPILL

    def allocate(
        self,
        ctx: CompileContext,
        liveness_map: Dict[str, TensorLiveness],
        max_memory: Optional[int] = None,
    ) -> MemoryAllocationPlan:
        """Execute aggressive spill allocation.

        The key insight: we don't use liveness_map for allocation decisions.
        Instead, we calculate per-operator memory requirements.

        Memory for an operator = max(largest_input, largest_output)
        because we can reuse input memory for output.
        """
        if max_memory is None:
            max_memory = float("inf")

        graph = ctx.graph
        nodes = graph.topological_sort()

        # Step 1: Calculate per-operator memory requirements
        node_memory = []
        for i, node in enumerate(nodes):
            input_tensors = self._get_non_constant_inputs(node, graph)
            output_tensors = node.outputs

            # Find the largest single tensor in inputs and outputs
            # We can reuse memory, so peak = max(largest_input, largest_output)
            input_sizes = [
                self._get_tensor_size(t, graph)
                for t in input_tensors
            ]
            output_sizes = [
                self._get_tensor_size(t, graph)
                for t in output_tensors
            ]

            largest_input = max(input_sizes) if input_sizes else 0
            largest_output = max(output_sizes) if output_sizes else 0

            # Peak memory for this node is the larger of the two
            # (we can reuse input memory for output)
            total = max(largest_input, largest_output)

            # However, if we have multiple inputs that need to be alive simultaneously
            # (e.g., Add node with two inputs), we need to account for that
            # Sum of all inputs (they all need to be present)
            total_inputs = sum(input_sizes)

            # For binary ops, we need all inputs present
            # The memory needed is max(sum of inputs, largest output)
            total = max(total_inputs, largest_output)

            node_memory.append({
                'node': node.name,
                'index': i,
                'input_size': total_inputs,
                'output_size': largest_output,
                'total': total,
                'inputs': list(input_tensors),
                'outputs': list(output_tensors),
            })

        # Step 2: Find peak memory requirement
        if not node_memory:
            # Empty graph
            return MemoryAllocationPlan(
                strategy_name=self.name,
                total_fast_memory=0,
                peak_memory=0,
                num_buffers=0,
            )

        peak_node = max(node_memory, key=lambda x: x['total'])

        # Step 3: Validate that peak fits in max_memory
        if peak_node['total'] > max_memory:
            raise RuntimeError(
                f"Cannot fit model in fast memory with aggressive spill. "
                f"Node '{peak_node['node']}' requires {peak_node['total']} bytes "
                f"(inputs: {peak_node['input_size']}, outputs: {peak_node['output_size']}) "
                f"but max_memory is {max_memory} bytes. "
                f"Reduce tensor sizes or increase max_memory."
            )

        # Step 4: Allocate memory for all tensors
        # Key insight: With aggressive spill, ALL tensors can share the SAME
        # memory location because they're never in fast memory at the same time.
        # We only need to allocate enough memory for the peak requirement.
        buffers: List[MemoryBuffer] = []
        tensor_to_buffer: Dict[str, int] = {}
        tensor_allocations: Dict[str, TensorAllocation] = {}

        # All tensors start at offset 0 since they can share memory
        # The buffer size is the peak memory requirement
        peak_size = peak_node['total']

        # Align peak_size
        aligned_peak_size = self._align(peak_size, self.DEFAULT_ALIGNMENT)

        # Create a single shared buffer that all tensors use
        buffer = MemoryBuffer(
            id=0,
            offset=0,
            size=aligned_peak_size,
            alignment=self.DEFAULT_ALIGNMENT,
            tensors=[],
        )
        buffers.append(buffer)

        # Collect all tensors that need allocation (non-constants)
        all_tensors = set()
        for node_info in node_memory:
            all_tensors.update(node_info['inputs'])
            all_tensors.update(node_info['outputs'])

        # Allocate all tensors at offset 0 (they share the buffer)
        for tensor_name in all_tensors:
            if tensor_name in graph.constants:
                continue

            size = self._get_tensor_size(tensor_name, graph)
            if size == 0:
                continue

            # All tensors use the same buffer at offset 0
            # With aggressive spill, they're never in fast memory simultaneously
            tensor_allocations[tensor_name] = TensorAllocation(
                tensor_name=tensor_name,
                buffer_id=0,
                offset=0,
                size=size,
            )
            tensor_to_buffer[tensor_name] = 0

        # Verify largest tensor fits
        largest_tensor = max(
            (self._get_tensor_size(t, graph) for t in all_tensors if t not in graph.constants),
            default=0
        )
        if largest_tensor > aligned_peak_size:
            raise RuntimeError(
                f"Internal error: Largest tensor ({largest_tensor} bytes) "
                f"exceeds allocated buffer ({aligned_peak_size} bytes)"
            )

        # Step 5: Generate spill and reload points
        # In aggressive strategy, we spill immediately after last use
        # and reload just before first use
        spill_points, reload_points = self._generate_spill_reload_points(
            node_memory, tensor_allocations, graph, nodes
        )

        # Calculate total and peak memory
        total_fast_memory = aligned_peak_size
        peak_memory = peak_node['total']

        plan = MemoryAllocationPlan(
            strategy_name=self.name,
            total_fast_memory=total_fast_memory,
            peak_memory=peak_memory,
            num_buffers=len(buffers),
            buffers=buffers,
            tensor_to_buffer=tensor_to_buffer,
            tensor_allocations=tensor_allocations,
            spill_points=spill_points,
            reload_points=reload_points,
            node_memory_usage=[nm['total'] for nm in node_memory],
        )

        return plan

    def _get_non_constant_inputs(self, node, graph) -> Set[str]:
        """Get non-constant input tensors for a node."""
        return {t for t in node.inputs if t not in graph.constants}

    def _get_tensor_size(self, tensor_name: str, graph) -> int:
        """Get the byte size of a tensor."""
        tensor = graph.get_tensor(tensor_name)
        if tensor is None:
            return 0
        return tensor.byte_size()

    def _align(self, size: int, alignment: int) -> int:
        """Align size to alignment boundary."""
        return ((size + alignment - 1) // alignment) * alignment

    def _generate_spill_reload_points(
        self,
        node_memory: List[Dict],
        tensor_allocations: Dict[str, TensorAllocation],
        graph,
        nodes,
    ) -> tuple[List[SpillPoint], List[ReloadPoint]]:
        """Generate spill and reload points for aggressive strategy.

        Strategy:
        - Spill immediately after each node's outputs are produced
        - Reload before each node's inputs are needed (if not already in fast memory)
        """
        spill_points: List[SpillPoint] = []
        reload_points: List[ReloadPoint] = []

        # Track which tensors are "in fast memory" at each point
        # Initially, only input tensors are in fast memory
        fast_tensors: Set[str] = set()

        # Get graph inputs (graph.inputs is a List[str])
        graph_inputs = set(graph.inputs)
        fast_tensors.update(graph_inputs)

        # Slow memory offset for spilled tensors
        slow_offset = 0

        # First pass: mark when tensors are first used and last used
        first_use: Dict[str, int] = {}
        last_use: Dict[str, int] = {}

        for node_info in node_memory:
            idx = node_info['index']
            for t in node_info['inputs']:
                if t not in first_use:
                    first_use[t] = idx
                last_use[t] = idx
            for t in node_info['outputs']:
                if t not in first_use:
                    first_use[t] = idx
                last_use[t] = idx

        # Second pass: generate spill/reload points
        for node_info in node_memory:
            idx = node_info['index']
            node_name = node_info['node']
            inputs = node_info['inputs']
            outputs = node_info['outputs']

            # Reload inputs before this node (if they were spilled)
            for input_tensor in inputs:
                if input_tensor in tensor_allocations:
                    alloc = tensor_allocations[input_tensor]

                    # Need reload if this tensor was spilled before
                    # In aggressive strategy, we assume everything not actively
                    # being computed is in slow memory
                    if input_tensor not in fast_tensors and input_tensor not in graph_inputs:
                        reload_points.append(ReloadPoint(
                            tensor_name=input_tensor,
                            before_node=node_name,
                            before_node_idx=idx,
                            from_slow_offset=slow_offset,
                            to_buffer_id=alloc.buffer_id,
                            to_fast_offset=alloc.offset,
                            size=alloc.size,
                        ))
                        slow_offset += alloc.size
                        fast_tensors.add(input_tensor)

            # Execute node (inputs consumed, outputs produced)
            # Remove inputs from fast memory (they've been consumed)
            for input_tensor in inputs:
                fast_tensors.discard(input_tensor)

            # Add outputs to fast memory
            fast_tensors.update(outputs)

            # Spill outputs immediately after production
            # (unless they're the final graph outputs)
            graph_outputs = set(graph.outputs)
            for output_tensor in outputs:
                if output_tensor in tensor_allocations and output_tensor not in graph_outputs:
                    alloc = tensor_allocations[output_tensor]
                    spill_points.append(SpillPoint(
                        tensor_name=output_tensor,
                        after_node=node_name,
                        after_node_idx=idx,
                        from_buffer_id=alloc.buffer_id,
                        from_fast_offset=alloc.offset,
                        to_slow_offset=slow_offset,
                        size=alloc.size,
                    ))
                    slow_offset += alloc.size
                    # Remove from fast after spilling
                    fast_tensors.discard(output_tensor)

        return spill_points, reload_points
