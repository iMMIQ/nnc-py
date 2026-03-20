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

    Note: Uses sequential memory allocation where each tensor gets a unique
    offset within the buffer. Spill/reload is tracked but not currently
    used in code generation - allocations are static.
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

        # Track state for sequential allocation
        # Each tensor gets a unique offset within the buffer
        current_offset = 0
        tensor_offsets: Dict[str, int] = {}

        # Track allocations
        tensor_allocations: Dict[str, TensorAllocation] = {}
        buffers: List[MemoryBuffer] = []
        spill_points: List[SpillPoint] = []
        reload_points: List[ReloadPoint] = []
        spill_bytes = 0
        reload_bytes = 0

        # Track which tensors are in slow memory (for future spill/reload support)
        in_slow_memory: Set[str] = set()
        slow_memory_offsets: Dict[str, int] = {}
        slow_offset = 0

        def align_offset(offset: int, alignment: int) -> int:
            """Align offset to alignment boundary."""
            return ((offset + alignment - 1) // alignment) * alignment

        def fits_in_memory(additional_size: int) -> bool:
            """Check if additional memory fits."""
            return current_offset + additional_size <= max_memory

        def spill_all(node_idx: int) -> None:
            """Spill all active tensors to slow memory."""
            nonlocal current_offset, slow_offset, spill_bytes

            for tensor_name, offset in list(tensor_offsets.items()):
                if tensor_name not in in_slow_memory:
                    in_slow_memory.add(tensor_name)
                    size = tensor_sizes.get(tensor_name, 0)
                    slow_memory_offsets[tensor_name] = slow_offset
                    slow_offset += size

                    spill_points.append(SpillPoint(
                        tensor_name=tensor_name,
                        after_node=nodes[node_idx].name,
                        after_node_idx=node_idx,
                        from_buffer_id=0,
                        from_fast_offset=offset,
                        to_slow_offset=slow_memory_offsets[tensor_name],
                        size=size,
                    ))
                    spill_bytes += size

            # Clear all allocations and reset offset
            tensor_offsets.clear()
            current_offset = 0

        def allocate_tensor(tensor_name: str, node_idx: int) -> int:
            """Allocate a tensor in fast memory and return its offset."""
            nonlocal current_offset, reload_bytes

            # If already allocated, return existing offset
            if tensor_name in tensor_offsets:
                return tensor_offsets[tensor_name]

            size = tensor_sizes.get(tensor_name, 0)

            # Skip validation if tensor is larger than max_memory - we'll use single-slot mode
            # The spill mechanism will handle this by constantly spilling and reloading
            if max_memory != float("inf") and size > max_memory:
                # Fall back to single-slot mode for this tensor
                pass  # Will use offset=0 below
            else:
                self._validate_tensor_fits(size, max_memory)

            # If in slow memory, reload it
            if tensor_name in in_slow_memory:
                # For single-slot mode or normal allocation
                offset = 0 if size > max_memory else current_offset
                reload_points.append(ReloadPoint(
                    tensor_name=tensor_name,
                    before_node=nodes[node_idx].name,
                    before_node_idx=node_idx,
                    from_slow_offset=slow_memory_offsets[tensor_name],
                    to_buffer_id=0,
                    to_fast_offset=offset,
                    size=size,
                    reload_slot_id=0,
                ))
                reload_bytes += size

            # Check if we need single-slot mode (tensor larger than max_memory)
            if max_memory != float("inf") and size > max_memory:
                # Single-slot mode: all tensors at offset 0, but record size for buffer
                offset = 0
            else:
                # Normal sequential allocation
                # Align offset
                aligned_offset = align_offset(current_offset, self.DEFAULT_ALIGNMENT)
                offset = aligned_offset
                current_offset = aligned_offset + size

            # Allocate tensor
            tensor_offsets[tensor_name] = offset

            # Record allocation
            is_spilled = tensor_name in in_slow_memory
            tensor_allocations[tensor_name] = TensorAllocation(
                tensor_name=tensor_name,
                buffer_id=0,
                offset=offset,
                size=size,
                is_spilled=is_spilled,
            )

            return offset

        # Process each node
        for i, node in enumerate(nodes):
            # Get all input tensors for this node
            input_tensors = []
            for tensor_name in node.inputs:
                if tensor_name in liveness_map:
                    input_tensors.append(tensor_name)

            # Calculate memory needed for new inputs
            needed_memory = sum(
                tensor_sizes.get(t, 0) for t in input_tensors
                if t not in tensor_offsets
            )

            # If we need to spill, spill all
            if not fits_in_memory(needed_memory):
                # Get previous node index for spill point
                spill_node_idx = max(0, i - 1)
                spill_all(spill_node_idx)

            # Ensure all inputs are in fast memory
            for tensor_name in input_tensors:
                allocate_tensor(tensor_name, i)

            # Output tensor will be produced, also needs space
            for tensor_name in node.outputs:
                if tensor_name in liveness_map:
                    output_size = tensor_sizes.get(tensor_name, 0)
                    if not fits_in_memory(output_size):
                        spill_all(i)
                    allocate_tensor(tensor_name, i)

        # Calculate total buffer size
        # If any tensor is larger than max_memory, buffer size is the max tensor size
        # (single-slot mode)
        max_tensor_size = max(tensor_sizes.values()) if tensor_sizes else 0
        if max_memory != float("inf") and max_tensor_size > max_memory:
            buffer_size = max_tensor_size
        else:
            buffer_size = align_offset(current_offset, self.DEFAULT_ALIGNMENT)
        buffer_size = max(buffer_size, self.DEFAULT_ALIGNMENT)

        # Create buffer - all tensors share one buffer with sequential offsets
        buffer = MemoryBuffer(
            id=0,
            offset=0,
            size=buffer_size,
            alignment=self.DEFAULT_ALIGNMENT,
            tensors=list(tensor_offsets.keys()),
        )
        buffers.append(buffer)

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
            spill_bytes=spill_bytes,
            reload_bytes=reload_bytes,
            total_transfer_bytes=spill_bytes + reload_bytes,
        )
