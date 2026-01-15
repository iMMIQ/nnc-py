"""Liveness-based memory allocation strategy adapter.

This module adapts the original MemoryPlanningPass algorithm to work
with the new MemoryAllocationStrategy interface.
"""

from typing import Dict, List, Optional

from nnc_py.ir.context import CompileContext
from nnc_py.passes.liveness import TensorLiveness
from nnc_py.passes.memory_plan import MemoryBuffer
from nnc_py.passes.memory_strategy import (
    AllocationStrategy,
    MemoryAllocationPlan,
    MemoryAllocationStrategy,
    TensorAllocation,
)


class LivenessAllocationStrategy(MemoryAllocationStrategy):
    """Adapter for the original MemoryPlanningPass algorithm.

    This strategy uses liveness interval analysis to allocate memory
    buffers, allowing tensors with non-overlapping lifetimes to share
    the same buffer.
    """

    DEFAULT_ALIGNMENT = 16
    MIN_BUFFER_SIZE = 16

    @property
    def name(self) -> str:
        return "liveness"

    @property
    def strategy_type(self) -> AllocationStrategy:
        return AllocationStrategy.LIVENESS_BASED

    def allocate(
        self,
        ctx: CompileContext,
        liveness_map: Dict[str, TensorLiveness],
        max_memory: Optional[int] = None,
    ) -> MemoryAllocationPlan:
        """Execute liveness-based allocation."""
        if max_memory is None:
            max_memory = float("inf")

        # Collect tensor information
        tensor_info = self._collect_tensor_info(ctx, liveness_map)

        # Validate largest tensor fits
        largest_size = max(info["size"] for info in tensor_info.values())
        self._validate_tensor_fits(largest_size, max_memory)

        # Sort tensors by size (descending) for better packing
        sorted_tensors = sorted(
            tensor_info.items(),
            key=lambda x: x[1]["size"],
            reverse=True
        )

        # Allocate buffers
        buffers: List[MemoryBuffer] = []
        tensor_to_buffer: Dict[str, int] = {}
        current_offset = 0

        for tensor_name, info in sorted_tensors:
            liveness = info["liveness"]
            size = info["size"]

            # Try to reuse existing buffer
            assigned_buffer = None

            for buf in buffers:
                if self._can_reuse_buffer(buf, size, liveness, tensor_info, tensor_to_buffer):
                    assigned_buffer = buf
                    break

            if assigned_buffer is None:
                # Create new buffer
                aligned_offset = self._align(current_offset, self.DEFAULT_ALIGNMENT)
                buffer_size = max(self._align(size, self.DEFAULT_ALIGNMENT), self.MIN_BUFFER_SIZE)

                new_buffer = MemoryBuffer(
                    id=len(buffers),
                    offset=aligned_offset,
                    size=buffer_size,
                    alignment=self.DEFAULT_ALIGNMENT,
                    tensors=[],
                )
                buffers.append(new_buffer)

                current_offset = aligned_offset + buffer_size
                assigned_buffer = new_buffer

            # Add tensor to buffer
            if hasattr(assigned_buffer, 'add_tensor'):
                assigned_buffer.add_tensor(tensor_name)
            else:
                assigned_buffer.tensors.append(tensor_name)

            tensor_to_buffer[tensor_name] = assigned_buffer.id

        # Create plan
        total_fast_memory = sum(b.size for b in buffers) if buffers else 0

        plan = MemoryAllocationPlan(
            strategy_name=self.name,
            total_fast_memory=total_fast_memory,
            num_buffers=len(buffers),
            buffers=buffers,
            tensor_to_buffer=tensor_to_buffer,
        )

        # Fill tensor allocations
        for tensor_name, info in tensor_info.items():
            buffer_id = tensor_to_buffer[tensor_name]
            buffer = buffers[buffer_id]
            plan.tensor_allocations[tensor_name] = TensorAllocation(
                tensor_name=tensor_name,
                buffer_id=buffer_id,
                offset=buffer.offset,
                size=info["size"],
            )

        # Calculate peak memory
        plan.peak_memory = self._calculate_peak_memory(ctx, liveness_map, plan)

        return plan

    def _collect_tensor_info(
        self,
        ctx: CompileContext,
        liveness_map: Dict[str, TensorLiveness],
    ) -> Dict[str, Dict]:
        """Collect tensor information."""
        info = {}
        for tensor_name, liveness in liveness_map.items():
            tensor = ctx.graph.get_tensor(tensor_name)
            size = tensor.byte_size()
            if size < 0:
                continue
            info[tensor_name] = {
                "size": size,
                "liveness": liveness,
            }
        return info

    def _can_reuse_buffer(
        self,
        buffer: MemoryBuffer,
        tensor_size: int,
        tensor_liveness: TensorLiveness,
        tensor_info: Dict,
        tensor_to_buffer: Dict[str, int],
    ) -> bool:
        """Check if a buffer can be reused for this tensor."""
        # Check size
        if not self._buffer_can_hold(buffer, tensor_size):
            return False

        # Check for liveness overlap with existing tensors in buffer
        for existing_name in buffer.tensors:
            if existing_name in tensor_info:
                existing_liveness = tensor_info[existing_name]["liveness"]
                if self._liveness_overlaps(tensor_liveness, existing_liveness):
                    return False

        return True

    def _buffer_can_hold(self, buffer: MemoryBuffer, size: int) -> bool:
        """Check if buffer can hold a tensor of given size."""
        return buffer.size >= size

    def _liveness_overlaps(
        self,
        l1: TensorLiveness,
        l2: TensorLiveness,
    ) -> bool:
        """Check if two liveness ranges overlap."""
        return not (l1.live_end < l2.live_start or l2.live_end < l1.live_start)

    def _align(self, size: int, alignment: int) -> int:
        """Align size to alignment boundary."""
        return ((size + alignment - 1) // alignment) * alignment

    def _calculate_peak_memory(
        self,
        ctx: CompileContext,
        liveness_map: Dict[str, TensorLiveness],
        plan: MemoryAllocationPlan,
    ) -> int:
        """Calculate peak memory usage across all nodes."""
        nodes = ctx.graph.topological_sort()
        peak = 0

        for i, node in enumerate(nodes):
            # Find all tensors live at this node
            live_tensors = set()
            for tensor_name, liveness in liveness_map.items():
                if liveness.live_start <= i <= liveness.live_end:
                    live_tensors.add(tensor_name)

            # Sum up their buffer sizes
            used_buffers = set()
            for t in live_tensors:
                bid = plan.get_buffer_for_tensor(t)
                if bid >= 0:
                    used_buffers.add(bid)

            memory = sum(
                plan.buffers[bid].size
                for bid in used_buffers
                if 0 <= bid < len(plan.buffers)
            )
            peak = max(peak, memory)

        return peak
