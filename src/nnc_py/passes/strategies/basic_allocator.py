"""Basic sequential memory allocation with simple spill-all strategy.

This module implements the simplest possible memory allocation algorithm:

1. Sequential allocation: Allocate memory for tensors as needed
2. Spill-all: When memory is full, spill ALL existing tensors to slow memory
3. Continue: After spill, allocate for new tensors

Key properties:
- Simple: No complex graph analysis or coloring
- Predictable: Always spills everything when full
- Naive: Not optimal, but very easy to understand and verify
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional

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


@dataclass
class ActiveTensor:
    """A tensor currently in fast memory."""
    name: str
    size: int
    offset: int


class BasicAllocator(MemoryAllocationStrategy):
    """Basic sequential allocator with simple spill-all strategy.

    Algorithm:
    1. Process nodes in topological order
    2. For each node, ensure inputs are in fast memory
    3. If not enough space, spill ALL active tensors
    4. Reload spilled tensors before use
    """

    DEFAULT_ALIGNMENT = 16

    @property
    def name(self) -> str:
        return "basic"

    @property
    def strategy_type(self) -> AllocationStrategy:
        return AllocationStrategy.BASIC

    def allocate(
        self,
        ctx: CompileContext,
        liveness_map: Dict[str, TensorLiveness],
        max_memory: Optional[int] = None,
    ) -> MemoryAllocationPlan:
        """Execute basic allocation."""
        if max_memory is None:
            max_memory = float("inf")

        nodes = ctx.graph.topological_sort()
        node_index = {n.name: i for i, n in enumerate(nodes)}

        # Get tensor sizes
        tensor_sizes: Dict[str, int] = {}
        for tensor_name in liveness_map.keys():
            tensor = ctx.graph.get_tensor(tensor_name)
            if tensor:
                tensor_sizes[tensor_name] = tensor.byte_size()

        # Track state
        active_tensors: List[ActiveTensor] = []  # Tensors in fast memory
        used_memory = 0
        slow_offset = 0  # Offset in slow memory for spilled tensors

        # Track allocations
        tensor_allocations: Dict[str, TensorAllocation] = {}
        buffers: List[MemoryBuffer] = []
        spill_points: List[SpillPoint] = []
        reload_points: List[ReloadPoint] = []

        # Create a single buffer that holds one tensor at a time
        max_tensor_size = max(tensor_sizes.values()) if tensor_sizes else 0
        buffer_size = max(max_tensor_size, self.DEFAULT_ALIGNMENT)
        buffer = MemoryBuffer(
            id=0,
            offset=0,
            size=buffer_size,
            alignment=self.DEFAULT_ALIGNMENT,
            tensors=[],
        )
        buffers.append(buffer)

        # Track which tensors are in slow memory
        in_slow_memory: Set[str] = set()
        slow_memory_offsets: Dict[str, int] = {}

        def spill_all(node_idx: int) -> None:
            """Spill all active tensors to slow memory."""
            nonlocal active_tensors, used_memory, slow_offset

            for at in active_tensors:
                if at.name not in in_slow_memory:
                    in_slow_memory.add(at.name)
                    slow_memory_offsets[at.name] = slow_offset
                    slow_offset += at.size

                spill_points.append(SpillPoint(
                    tensor_name=at.name,
                    after_node=nodes[node_idx].name,
                    after_node_idx=node_idx,
                    from_buffer_id=0,
                    from_fast_offset=at.offset,
                    to_slow_offset=slow_memory_offsets[at.name],
                    size=at.size,
                ))

            active_tensors.clear()
            used_memory = 0

        def ensure_in_fast_memory(tensor_name: str, before_node_idx: int) -> None:
            """Ensure a tensor is in fast memory, reloading if necessary."""
            nonlocal active_tensors, used_memory

            size = tensor_sizes.get(tensor_name, 0)
            self._validate_tensor_fits(size, max_memory)

            # Check if already in fast memory
            for at in active_tensors:
                if at.name == tensor_name:
                    return

            # If in slow memory, reload it
            if tensor_name in in_slow_memory:
                reload_points.append(ReloadPoint(
                    tensor_name=tensor_name,
                    before_node=nodes[before_node_idx].name,
                    before_node_idx=before_node_idx,
                    from_slow_offset=slow_memory_offsets[tensor_name],
                    to_buffer_id=0,
                    to_fast_offset=0,
                    size=size,
                    reload_slot_id=0,
                ))

            # Allocate in fast memory
            offset = 0  # Always use offset 0 since we have one slot
            active_tensors.append(ActiveTensor(
                name=tensor_name,
                size=size,
                offset=offset,
            ))
            used_memory += size

            # Record allocation
            is_spilled = tensor_name in in_slow_memory
            tensor_allocations[tensor_name] = TensorAllocation(
                tensor_name=tensor_name,
                buffer_id=0,
                offset=offset,
                size=size,
                is_spilled=is_spilled,
            )

        def fits_in_memory(additional_size: int) -> bool:
            """Check if additional memory fits."""
            return used_memory + additional_size <= max_memory

        # Process each node
        for i, node in enumerate(nodes):
            # Get all input tensors for this node
            input_tensors = []
            for tensor_name in node.inputs:
                if tensor_name in liveness_map:
                    input_tensors.append(tensor_name)

            # Calculate memory needed for all inputs
            needed_memory = sum(
                tensor_sizes.get(t, 0) for t in input_tensors
                if not any(at.name == t for at in active_tensors)
            )

            # If we need to spill, spill all
            if not fits_in_memory(needed_memory):
                # Get previous node index for spill point
                spill_node_idx = max(0, i - 1)
                spill_all(spill_node_idx)

            # Ensure all inputs are in fast memory
            for tensor_name in input_tensors:
                ensure_in_fast_memory(tensor_name, i)

            # Output tensor will be produced, also needs space
            for tensor_name in node.outputs:
                if tensor_name in liveness_map:
                    output_size = tensor_sizes.get(tensor_name, 0)
                    if not fits_in_memory(output_size):
                        spill_all(i)
                    ensure_in_fast_memory(tensor_name, i)

        # Final spill at the end
        if active_tensors and nodes:
            spill_all(len(nodes) - 1)

        # Build tensor_to_buffer map
        tensor_to_buffer = {
            t: a.buffer_id
            for t, a in tensor_allocations.items()
        }

        return MemoryAllocationPlan(
            strategy_name=self.name,
            total_fast_memory=buffer_size,
            total_slow_memory=slow_offset,
            peak_memory=buffer_size,
            num_buffers=len(buffers),
            buffers=buffers,
            tensor_allocations=tensor_allocations,
            tensor_to_buffer=tensor_to_buffer,
            spill_points=spill_points,
            reload_points=reload_points,
        )
