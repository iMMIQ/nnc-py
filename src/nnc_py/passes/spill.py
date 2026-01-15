"""Spill analysis for memory overflow handling.

When the total memory required exceeds the available fast memory,
this pass analyzes which tensors should be spilled to slow memory
and generates spill/reload points.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.passes.base import PassBase
from nnc_py.passes.liveness import TensorLiveness, get_liveness
from nnc_py.passes.memory_plan import (
    MemoryBuffer,
    MemoryPlan,
    TensorMemoryInfo,
    get_memory_plan,
)


@dataclass
class SpillSlot:
    """Information about a spilled tensor in slow memory."""

    tensor_name: str
    slow_offset: int           # Offset in slow memory pool
    size: int

    # When to spill/reload (node names)
    spill_after_node: str      # Spill after this node executes
    reload_before_nodes: List[str] = field(default_factory=list)

    # Original fast memory info
    original_buffer_id: int = 0
    original_fast_offset: int = 0


@dataclass
class ReloadPoint:
    """A point where a tensor needs to be reloaded from slow memory."""

    tensor_name: str
    before_node: str           # Reload before this node
    from_slow_offset: int      # Source offset in slow memory
    to_fast_offset: int        # Destination offset in fast memory
    size: int


@dataclass
class SpillPoint:
    """A point where a tensor needs to be spilled to slow memory."""

    tensor_name: str
    after_node: str            # Spill after this node
    from_fast_offset: int      # Source offset in fast memory
    to_slow_offset: int        # Destination offset in slow memory
    size: int


@dataclass
class SpillPlan:
    """Complete spill plan for memory overflow."""

    original_memory_size: int     # Original memory required
    fast_memory_size: int          # Fast memory after spills
    slow_memory_size: int
    max_memory: int

    # Tensors that are spilled
    spilled_tensors: Dict[str, SpillSlot]

    # Spill and reload points (in execution order)
    spill_points: List[SpillPoint]
    reload_points: List[ReloadPoint]

    # Modified memory plan with spill info
    memory_plan: MemoryPlan

    @property
    def overflow_amount(self) -> int:
        """Amount of memory that exceeds the limit."""
        return max(0, self.original_memory_size - self.max_memory)

    @property
    def has_overflow(self) -> bool:
        """Whether memory overflow occurs."""
        return self.original_memory_size > self.max_memory


class SpillAnalysisPass(PassBase):
    """Analyze and plan memory spill to slow memory.

    This pass:
    1. Checks if memory limit is exceeded
    2. Selects tensors to spill based on spill priority
    3. Generates spill/reload points
    4. Updates the memory plan with spill information
    """

    @property
    def name(self) -> str:
        return "SpillAnalysis"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute spill analysis."""
        # Only run if max_memory is set and memory planning was done
        if "max_memory" not in ctx.metadata:
            return

        if "memory_plan" not in ctx.metadata:
            # Memory planning must be run first
            raise RuntimeError("MemoryPlanningPass must be run before SpillAnalysisPass")

        max_memory = ctx.metadata["max_memory"]
        plan = get_memory_plan(ctx)

        # CRITICAL CHECK: Verify that max_memory is at least as large as
        # the largest individual tensor. Each tensor must fit in fast memory
        # when it is being computed (we don't support tensor chunking yet).
        largest_tensor = max(
            (tensor.byte_size() for tensor in ctx.graph.tensors.values()
             if tensor.name not in ctx.graph.constants),
            default=0
        )

        if max_memory < largest_tensor:
            # Cannot fit even a single tensor in fast memory
            raise RuntimeError(
                f"Memory limit ({max_memory} bytes) is too small to fit "
                f"the largest tensor ({largest_tensor} bytes). "
                f"Each tensor must fit in fast memory when being computed. "
                f"Minimum required: {largest_tensor} bytes."
            )

        # Check if overflow occurs
        if plan.total_size <= max_memory:
            # No overflow needed
            ctx.metadata["spill_plan"] = None
            if ctx.debug:
                self._log_no_overflow(plan, max_memory)
            return

        # Perform spill analysis
        spill_plan = self._analyze_spill(ctx, plan, max_memory)

        # Store in context
        ctx.metadata["spill_plan"] = spill_plan

        if ctx.debug:
            self._log_summary(spill_plan)

    def _analyze_spill(
        self,
        ctx: CompileContext,
        plan: MemoryPlan,
        max_memory: int,
    ) -> SpillPlan:
        """Analyze which tensors to spill and generate spill plan."""
        liveness_map = ctx.metadata["tensor_liveness"]

        # Step 1: Calculate spill priority for each tensor
        tensor_priorities = self._calculate_spill_priorities(
            plan, liveness_map, ctx.graph
        )

        # Step 2: Select tensors to spill until we fit in memory
        spilled_tensors = self._select_spill_candidates(
            plan, tensor_priorities, max_memory
        )

        # Step 3: Generate spill and reload points
        spill_points, reload_points = self._generate_spill_reload_points(
            ctx, plan, spilled_tensors
        )

        # Step 4: Assign slow memory offsets
        slow_offset = 0
        for tensor_name, slot in spilled_tensors.items():
            slot.slow_offset = slow_offset
            slow_offset += self._align(slot.size, 16)

        slow_memory_size = slow_offset

        # Update spill_points and reload_points with correct slow offsets
        for point in spill_points:
            if point.tensor_name in spilled_tensors:
                point.to_slow_offset = spilled_tensors[point.tensor_name].slow_offset

        for point in reload_points:
            if point.tensor_name in spilled_tensors:
                point.from_slow_offset = spilled_tensors[point.tensor_name].slow_offset

        # Create spill plan
        spill_plan = SpillPlan(
            original_memory_size=plan.total_size,  # Original requirement
            fast_memory_size=max_memory,  # Will fit after spills
            slow_memory_size=slow_memory_size,
            max_memory=max_memory,
            spilled_tensors=spilled_tensors,
            spill_points=spill_points,
            reload_points=reload_points,
            memory_plan=plan,
        )

        return spill_plan

    def _calculate_spill_priorities(
        self,
        plan: MemoryPlan,
        liveness_map: Dict[str, TensorLiveness],
        graph: Graph,
    ) -> List[Tuple[str, float]]:
        """Calculate spill priority for each tensor.

        Higher priority = should be spilled first
        Priority = size / lifetime_length (larger, longer-lived tensors first)

        Also consider:
        - Don't spill constants (they're in ROM)
        - Prefer spilling tensors with gaps between uses
        """
        priorities = []

        for tensor_name, mem_info in plan.tensor_info.items():
            # Skip constants
            if tensor_name in graph.constants:
                continue

            liveness = liveness_map[tensor_name]
            size = mem_info.size
            lifetime = liveness.lifetime_range

            # Base priority: bytes per lifetime unit
            priority = size / max(lifetime, 1)

            # Boost priority for tensors with gaps between uses
            # (tensors that are not continuously used)
            consumers = graph.get_consumers(tensor_name)
            if consumers:
                # Check for gaps in usage
                node_indices = {
                    node.name: i for i, node in enumerate(graph.topological_sort())
                }
                use_indices = sorted(node_indices[c.name] for c in consumers)

                # Calculate gaps
                gaps = 0
                for i in range(len(use_indices) - 1):
                    gap = use_indices[i + 1] - use_indices[i] - 1
                    gaps += gap

                if gaps > 0:
                    priority *= (1 + gaps * 0.5)  # Boost priority

            priorities.append((tensor_name, priority))

        # Sort by priority (descending)
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities

    def _select_spill_candidates(
        self,
        plan: MemoryPlan,
        priorities: List[Tuple[str, float]],
        max_memory: int,
    ) -> Dict[str, SpillSlot]:
        """Select tensors to spill until we fit in memory."""
        # Calculate current memory usage by buffer
        buffer_usage = {}
        for buf in plan.buffers:
            buffer_usage[buf.id] = buf.size

        # Total memory is sum of all buffers
        total_memory = sum(buffer_usage.values())

        # Select tensors to spill
        spilled: Dict[str, SpillSlot] = {}
        current_memory = total_memory
        spilled_buffers = set()  # Track which buffers have had tensors spilled

        for tensor_name, _ in priorities:
            if current_memory <= max_memory:
                break

            mem_info = plan.tensor_info.get(tensor_name)
            if mem_info is None:
                continue

            buffer = plan.get_buffer(mem_info.buffer_id)

            # Create spill slot
            slot = SpillSlot(
                tensor_name=tensor_name,
                slow_offset=0,  # Will be assigned later
                size=mem_info.size,
                spill_after_node="",  # Will be assigned later
                original_buffer_id=buffer.id,
                original_fast_offset=mem_info.pool_offset,
            )
            spilled[tensor_name] = slot

            # Reduce memory usage by tensor size, not buffer size
            # Only count the buffer reduction once per buffer
            if buffer.id not in spilled_buffers:
                current_memory -= buffer.size
                spilled_buffers.add(buffer.id)

        return spilled

    def _generate_spill_reload_points(
        self,
        ctx: CompileContext,
        plan: MemoryPlan,
        spilled_tensors: Dict[str, SpillSlot],
    ) -> Tuple[List[SpillPoint], List[ReloadPoint]]:
        """Generate spill and reload points for spilled tensors."""
        graph = ctx.graph
        nodes = graph.topological_sort()
        node_index = {node.name: i for i, node in enumerate(nodes)}

        liveness_map = ctx.metadata["tensor_liveness"]

        spill_points: List[SpillPoint] = []
        reload_points: List[ReloadPoint] = []

        for tensor_name, slot in spilled_tensors.items():
            liveness = liveness_map[tensor_name]
            mem_info = plan.tensor_info[tensor_name]
            consumers = graph.get_consumers(tensor_name)

            # Spill after last use
            # Find the node that is the last consumer
            if consumers:
                last_consumer = max(consumers, key=lambda c: node_index[c.name])
                spill_after = last_consumer.name

                spill_points.append(SpillPoint(
                    tensor_name=tensor_name,
                    after_node=spill_after,
                    from_fast_offset=mem_info.pool_offset,
                    to_slow_offset=0,  # Will be assigned later
                    size=mem_info.size,
                ))
                slot.spill_after_node = spill_after

            # Reload before each use (except first use if still in fast memory)
            # For simplicity, reload before each use after spill
            for i, consumer in enumerate(consumers):
                use_node_idx = node_index[consumer.name]
                spill_after_idx = node_index[slot.spill_after_node]

                # Need reload if this use is after the spill point
                if use_node_idx > spill_after_idx:
                    reload_points.append(ReloadPoint(
                        tensor_name=tensor_name,
                        before_node=consumer.name,
                        from_slow_offset=0,  # Will be assigned later
                        to_fast_offset=mem_info.pool_offset,
                        size=mem_info.size,
                    ))
                    slot.reload_before_nodes.append(consumer.name)

        # Sort by execution order
        spill_points.sort(key=lambda p: node_index.get(p.after_node, 0))
        reload_points.sort(key=lambda p: node_index.get(p.before_node, 0))

        return spill_points, reload_points

    def _align(self, size: int, alignment: int) -> int:
        """Align size to the given alignment boundary."""
        return ((size + alignment - 1) // alignment) * alignment

    def _log_no_overflow(self, plan: MemoryPlan, max_memory: int):
        """Log that no overflow occurs."""
        print(f"\n{'='*60}")
        print("Spill Analysis: No Overflow")
        print(f"{'='*60}")
        print(f"Fast memory required: {plan.total_size:,} bytes")
        print(f"Fast memory limit:   {max_memory:,} bytes")
        print(f"Margin:              {max_memory - plan.total_size:,} bytes")
        print("No spill needed.")
        print(f"{'='*60}\n")

    def _log_summary(self, spill_plan: SpillPlan):
        """Log spill plan summary."""
        print(f"\n{'='*80}")
        print("Spill Analysis Summary")
        print(f"{'='*80}")
        print(f"Fast memory limit:   {spill_plan.max_memory:,} bytes ({spill_plan.max_memory / 1024:.2f} KB)")
        print(f"Original requirement: {spill_plan.memory_plan.total_size:,} bytes ({spill_plan.memory_plan.total_size / 1024:.2f} KB)")
        print(f"Overflow:            {spill_plan.overflow_amount:,} bytes ({spill_plan.overflow_amount / 1024:.2f} KB)")
        print()
        print(f"Tensors spilled:     {len(spill_plan.spilled_tensors)}")
        print(f"Slow memory used:    {spill_plan.slow_memory_size:,} bytes ({spill_plan.slow_memory_size / 1024:.2f} KB)")
        print(f"Spill points:        {len(spill_plan.spill_points)}")
        print(f"Reload points:       {len(spill_plan.reload_points)}")
        print()

        if spill_plan.spilled_tensors:
            print("Spilled Tensors:")
            for tensor_name, slot in spill_plan.spilled_tensors.items():
                print(f"  - {tensor_name}: {slot.size} bytes "
                      f"(spill after {slot.spill_after_node}, "
                      f"{len(slot.reload_before_nodes)} reloads)")

        print(f"{'='*80}\n")


def get_spill_plan(ctx: CompileContext) -> Optional[SpillPlan]:
    """Get the spill plan from the context.

    Args:
        ctx: Compilation context

    Returns:
        SpillPlan object or None if no spill needed
    """
    return ctx.metadata.get("spill_plan")
