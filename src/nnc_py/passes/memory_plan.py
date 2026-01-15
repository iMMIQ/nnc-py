"""Memory planning pass for static memory allocation.

This pass performs buffer allocation and sharing based on liveness analysis.
Tensors with non-overlapping lifetimes can share the same memory buffer.

This enables static memory allocation for embedded devices without dynamic malloc.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.types import DataType
from nnc_py.passes.base import PassBase
from nnc_py.passes.liveness import TensorLiveness, get_liveness


@dataclass
class MemoryBuffer:
    """A memory buffer that can be shared by multiple tensors."""

    id: int
    offset: int              # Offset in memory pool (bytes)
    size: int                # Buffer size (bytes)
    alignment: int = 16      # Required alignment

    # Tensors assigned to this buffer (in order of use)
    tensors: List[str] = field(default_factory=list)

    def can_hold(self, tensor_size: int, alignment: int = 16) -> bool:
        """Check if this buffer can hold a tensor of given size."""
        return self.size >= tensor_size and self.alignment >= alignment

    def add_tensor(self, tensor_name: str) -> None:
        """Add a tensor to this buffer."""
        if tensor_name not in self.tensors:
            self.tensors.append(tensor_name)


@dataclass
class TensorMemoryInfo:
    """Memory allocation information for a single tensor."""

    tensor_name: str
    buffer_id: int           # Which buffer this tensor uses
    offset: int              # Offset within buffer (bytes)
    size: int                # Tensor size (bytes)
    pool_offset: int         # Absolute offset in memory pool (bytes)


@dataclass
class MemoryPlan:
    """Complete memory plan for a computation graph."""

    buffers: List[MemoryBuffer]
    tensor_info: Dict[str, TensorMemoryInfo]
    total_size: int
    alignment: int = 16

    # Statistics
    num_tensors: int = 0
    num_buffers: int = 0
    savings_without_sharing: float = 0.0  # Percentage

    def get_tensor_info(self, tensor_name: str) -> TensorMemoryInfo:
        """Get memory info for a tensor."""
        return self.tensor_info[tensor_name]

    def get_buffer(self, buffer_id: int) -> MemoryBuffer:
        """Get a buffer by ID."""
        for buf in self.buffers:
            if buf.id == buffer_id:
                return buf
        raise KeyError(f"Buffer {buffer_id} not found")

    def get_summary(self) -> str:
        """Get a summary of the memory plan."""
        lines = [
            "Memory Planning Summary:",
            f"  Total tensors: {self.num_tensors}",
            f"  Shared buffers: {self.num_buffers}",
            f"  Total memory: {self._format_size(self.total_size)}",
            f"  Alignment: {self.alignment} bytes",
            f"  Savings vs no-sharing: {self.savings_without_sharing:.1f}%",
        ]
        return "\n".join(lines)

    def _format_size(self, size: int) -> str:
        """Format size in human-readable form."""
        if size >= 1024 * 1024:
            return f"{size / (1024 * 1024):.2f} MB"
        elif size >= 1024:
            return f"{size / 1024:.2f} KB"
        return f"{size} bytes"


class MemoryPlanningPass(PassBase):
    """Memory planning pass for static allocation.

    This pass:
    1. Collects all tensors with their sizes and liveness information
    2. Allocates memory buffers, allowing reuse for non-overlapping lifetimes
    3. Assigns each tensor to a buffer
    4. Calculates total memory pool size
    """

    # Default alignment for SIMD/NPU operations
    DEFAULT_ALIGNMENT = 16

    # Minimum buffer size (avoid tiny buffers)
    MIN_BUFFER_SIZE = 16

    @property
    def name(self) -> str:
        return "MemoryPlanning"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute memory planning."""
        # Ensure liveness analysis has been run
        if "tensor_liveness" not in ctx.metadata:
            raise RuntimeError("LivenessAnalysisPass must be run before MemoryPlanningPass")

        graph = ctx.graph

        # Collect tensor information
        tensor_list = self._collect_tensor_info(ctx)

        # Sort by size (descending) for better packing
        tensor_list.sort(key=lambda x: x["size"], reverse=True)

        # Allocate buffers with sharing
        buffers = self._allocate_buffers(ctx, tensor_list)

        # Calculate total memory size
        total_size = self._calculate_total_size(buffers)

        # Build tensor info mapping
        tensor_info = self._build_tensor_info(buffers, ctx)

        # Calculate statistics
        num_tensors = len(tensor_list)
        num_buffers = len(buffers)
        total_without_sharing = sum(t["size"] for t in tensor_list)
        savings = (1 - total_size / max(total_without_sharing, 1)) * 100

        # Create memory plan
        plan = MemoryPlan(
            buffers=buffers,
            tensor_info=tensor_info,
            total_size=total_size,
            alignment=self.DEFAULT_ALIGNMENT,
            num_tensors=num_tensors,
            num_buffers=num_buffers,
            savings_without_sharing=savings,
        )

        # Store in context
        ctx.metadata["memory_plan"] = plan

        # Log summary
        self._log_summary(ctx, plan)

    def _collect_tensor_info(self, ctx: CompileContext) -> List[Dict]:
        """Collect information about all tensors that need memory allocation."""
        graph = ctx.graph
        tensor_list = []

        for tensor_name, tensor in graph.tensors.items():
            # Constants are stored separately in constants.c
            if tensor_name in graph.constants:
                continue

            # Calculate size
            size = tensor.byte_size()
            if size < 0:
                # Unknown size (dynamic shape) - skip for now
                if ctx.debug:
                    print(f"Warning: Tensor {tensor_name} has unknown size, skipping memory planning")
                continue

            # Get liveness info
            liveness = get_liveness(ctx, tensor_name)

            tensor_list.append({
                "name": tensor_name,
                "size": size,
                "liveness": liveness,
            })

        return tensor_list

    def _allocate_buffers(self, ctx: CompileContext, tensor_list: List[Dict]) -> List[MemoryBuffer]:
        """Allocate memory buffers with reuse based on liveness."""
        buffers: List[MemoryBuffer] = []
        current_offset = 0
        buffer_id = 0

        for tensor_info in tensor_list:
            tensor_name = tensor_info["name"]
            tensor_size = tensor_info["size"]
            liveness = tensor_info["liveness"]

            # Try to find an existing buffer that can hold this tensor
            assigned_buffer = None

            for buffer in buffers:
                if self._can_reuse_buffer(buffer, tensor_size, liveness, ctx):
                    assigned_buffer = buffer
                    break

            if assigned_buffer is None:
                # Need to allocate a new buffer
                # Align the offset
                aligned_offset = self._align(current_offset, self.DEFAULT_ALIGNMENT)
                buffer_size = max(self._align(tensor_size, self.DEFAULT_ALIGNMENT), self.MIN_BUFFER_SIZE)

                new_buffer = MemoryBuffer(
                    id=buffer_id,
                    offset=aligned_offset,
                    size=buffer_size,
                    alignment=self.DEFAULT_ALIGNMENT,
                )
                buffers.append(new_buffer)

                current_offset = aligned_offset + buffer_size
                buffer_id += 1
                assigned_buffer = new_buffer

            # Assign tensor to buffer
            assigned_buffer.add_tensor(tensor_name)

        return buffers

    def _can_reuse_buffer(
        self,
        buffer: MemoryBuffer,
        tensor_size: int,
        tensor_liveness: TensorLiveness,
        ctx: CompileContext,
    ) -> bool:
        """Check if a tensor can reuse an existing buffer.

        A tensor can reuse a buffer if:
        1. The buffer is large enough
        2. The tensor's lifetime doesn't overlap with any tensor already using the buffer
        """
        # Size check
        if not buffer.can_hold(tensor_size, self.DEFAULT_ALIGNMENT):
            return False

        # Liveness overlap check
        liveness_map = ctx.metadata["tensor_liveness"]

        for existing_tensor in buffer.tensors:
            existing_liveness = liveness_map[existing_tensor]

            # Check for lifetime overlap
            if self._liveness_overlaps(tensor_liveness, existing_liveness):
                return False

        return True

    def _liveness_overlaps(self, a: TensorLiveness, b: TensorLiveness) -> bool:
        """Check if two liveness ranges overlap.

        We use a conservative definition: ranges overlap if they intersect.
        For reuse, we also need to consider the last use carefully - a tensor
        can be reused immediately after its last use.
        """
        # Check for intersection
        return not (a.live_end < b.live_start or b.live_end < a.live_start)

    def _align(self, size: int, alignment: int) -> int:
        """Align size to the given alignment boundary."""
        return ((size + alignment - 1) // alignment) * alignment

    def _calculate_total_size(self, buffers: List[MemoryBuffer]) -> int:
        """Calculate total memory pool size."""
        if not buffers:
            return 0
        # Last buffer's end
        last_buffer = max(buffers, key=lambda b: b.offset)
        return last_buffer.offset + last_buffer.size

    def _build_tensor_info(self, buffers: List[MemoryBuffer], ctx: CompileContext) -> Dict[str, TensorMemoryInfo]:
        """Build tensor info mapping from buffer allocation."""
        tensor_info: Dict[str, TensorMemoryInfo] = {}

        for buffer in buffers:
            for tensor_name in buffer.tensors:
                # Get tensor size
                tensor = ctx.graph.get_tensor(tensor_name)
                size = tensor.byte_size()

                tensor_info[tensor_name] = TensorMemoryInfo(
                    tensor_name=tensor_name,
                    buffer_id=buffer.id,
                    offset=0,  # Offset within buffer (always 0 for single-use)
                    pool_offset=buffer.offset,
                    size=size,
                )

        return tensor_info

    def _log_summary(self, ctx: CompileContext, plan: MemoryPlan) -> None:
        """Log memory planning summary."""
        if not ctx.debug:
            return

        print(f"\n{'='*80}")
        print(plan.get_summary())
        print(f"{'='*80}")

        # Detailed tensor breakdown
        print(f"\n{'Tensor':<20} {'Size':>10} {'Offset':>10} {'Buffer':>8} {'Shared With'}")
        print(f"{'-'*80}")

        for tensor_name in sorted(plan.tensor_info.keys(), key=lambda n: plan.tensor_info[n].pool_offset):
            info = plan.tensor_info[tensor_name]
            buffer = plan.get_buffer(info.buffer_id)

            # Get other tensors sharing this buffer
            sharing_with = [t for t in buffer.tensors if t != tensor_name]
            sharing_str = ",".join(sharing_with[:2])  # Show first 2
            if len(sharing_with) > 2:
                sharing_str += f"... ({len(sharing_with)} total)"

            if not sharing_with:
                sharing_str = "-"

            size_str = plan._format_size(info.size)
            print(
                f"{tensor_name:<20} "
                f"{size_str:>10} "
                f"{info.pool_offset:>10} "
                f"#{info.buffer_id:>7} "
                f"{sharing_str}"
            )

        print(f"{'='*80}\n")


def get_memory_plan(ctx: CompileContext) -> MemoryPlan:
    """Get the memory plan from the context.

    Args:
        ctx: Compilation context

    Returns:
        MemoryPlan object

    Raises:
        RuntimeError: If MemoryPlanningPass hasn't been run
    """
    plan = ctx.metadata.get("memory_plan")
    if plan is None:
        raise RuntimeError("MemoryPlanningPass must be run before calling get_memory_plan")
    return plan
