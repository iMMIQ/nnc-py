"""Memory planning pass for static memory allocation.

This pass performs buffer allocation and sharing based on liveness analysis.
Tensors with non-overlapping lifetimes can share the same memory buffer.

This enables static memory allocation for embedded devices without dynamic malloc.
"""

from dataclasses import dataclass, field

from nnc_py.ir.context import CompileContext


@dataclass
class MemoryBuffer:
    """A memory buffer that can be shared by multiple tensors."""

    id: int
    offset: int              # Offset in memory pool (bytes)
    size: int                # Buffer size (bytes)
    alignment: int = 16      # Required alignment

    # Tensors assigned to this buffer (in order of use)
    tensors: list[str] = field(default_factory=list)

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

    buffers: list[MemoryBuffer]
    tensor_info: dict[str, TensorMemoryInfo]
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
