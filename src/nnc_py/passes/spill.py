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

        # Step 4: Recalculate fast memory layout with compaction
        # This ensures all tensor offsets fit within max_memory
        new_fast_offsets = self._recalculate_fast_memory_layout(
            plan, spilled_tensors, max_memory, ctx
        )

        # Step 5: Assign slow memory offsets
        slow_offset = 0
        for tensor_name, slot in spilled_tensors.items():
            slot.slow_offset = slow_offset
            slow_offset += self._align(slot.size, 16)

        slow_memory_size = slow_offset

        # Step 6: Update spill_points and reload_points with correct offsets
        for point in spill_points:
            if point.tensor_name in spilled_tensors:
                point.to_slow_offset = spilled_tensors[point.tensor_name].slow_offset
            # Update fast offset to use recalculated layout
            if point.tensor_name in new_fast_offsets:
                point.from_fast_offset = new_fast_offsets[point.tensor_name]

        for point in reload_points:
            if point.tensor_name in spilled_tensors:
                point.from_slow_offset = spilled_tensors[point.tensor_name].slow_offset
            # Update fast offset to use recalculated layout
            if point.tensor_name in new_fast_offsets:
                point.to_fast_offset = new_fast_offsets[point.tensor_name]

        # Store recalculated offsets in spill plan for codegen use
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
        # Attach recalculated fast offsets for code generation
        spill_plan.fast_tensor_offsets = new_fast_offsets

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
        """Select tensors to spill until we fit in memory.

        This is a simplified approach that selects tensors based on priority.
        The actual memory reduction is achieved through the compact layout
        recalculation in _recalculate_fast_memory_layout.
        """
        # Calculate current memory usage by buffer
        buffer_usage = {}
        for buf in plan.buffers:
            buffer_usage[buf.id] = buf.size

        # Total memory is sum of all buffers
        total_memory = sum(buffer_usage.values())

        # If already within limit, no spill needed
        if total_memory <= max_memory:
            return {}

        # Select tensors to spill based on priority
        spilled: Dict[str, SpillSlot] = {}

        for tensor_name, _ in priorities:
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

    def _recalculate_fast_memory_layout(
        self,
        plan: MemoryPlan,
        spilled_tensors: Dict[str, SpillSlot],
        max_memory: int,
        ctx: CompileContext = None,
    ) -> Dict[str, int]:
        """Recalculate fast memory layout with lifetime-aware reuse.

        Uses tensor liveness information to allow non-overlapping tensors
        to share the same memory locations. This ensures all offsets stay
        within max_memory without wraparound.

        Note: spilled_tensors is ignored for now. All tensors are allocated
        in fast memory using lifetime-aware placement. The spilled_tensors
        mechanism would require generating spill/reload code, which is
        a more complex feature.

        Returns:
            Dictionary mapping tensor_name -> new_fast_offset
        """
        if ctx is None:
            # Fallback to simple sequential layout if no context
            return self._sequential_layout(plan, spilled_tensors, max_memory)

        new_offsets = {}
        alignment = 16

        # Get liveness information
        liveness_map = ctx.metadata.get("tensor_liveness", {})

        # Build interval map: tensor -> (live_start, live_end)
        intervals = {}
        for tensor_name, mem_info in plan.tensor_info.items():
            if tensor_name in liveness_map:
                liv = liveness_map[tensor_name]
                intervals[tensor_name] = (liv.live_start, liv.live_end)
            else:
                intervals[tensor_name] = (0, float("inf"))

        # Assign offsets using first-fit with lifetime checking
        # Track each allocation: (offset, size, live_start, live_end, tensor_name)
        allocations = []

        # Process tensors in order of their live_start to avoid ordering issues
        # Large tensors first within each live_start group to avoid fragmentation
        tensor_list = [
            (name, mem_info, intervals.get(name, (0, float("inf"))))
            for name, mem_info in plan.tensor_info.items()
        ]
        # Sort by live_start, then by size (descending), then by name
        tensor_list.sort(key=lambda x: (x[2][0], -x[1].size, x[0]))

        for tensor_name, mem_info, (live_start, live_end) in tensor_list:
            size = mem_info.size

            # Find first non-overlapping position
            offset = 0
            while True:
                # Check if this position conflicts with any existing allocation
                conflict = False
                for alloc_offset, alloc_size, alloc_start, alloc_end, _ in allocations:
                    # Check memory overlap
                    if offset < alloc_offset + alloc_size and offset + size > alloc_offset:
                        # Check lifetime overlap
                        if not (live_end <= alloc_start or live_start >= alloc_end):
                            conflict = True
                            break

                if not conflict and offset + size <= max_memory:
                    break

                # Try next aligned position
                offset = ((offset + alignment - 1) // alignment) * alignment
                offset += ((size + alignment - 1) // alignment) * alignment

                if offset >= max_memory:
                    raise RuntimeError(
                        f"Cannot fit tensor {tensor_name} (size={size}) "
                        f"in fast memory (limit={max_memory}). "
                        f"This tensor overlaps with {len(allocations)} other tensors that need "
                        f"to be alive simultaneously. "
                        f"Consider increasing max_memory or restructuring the model."
                    )

            allocations.append((offset, size, live_start, live_end, tensor_name))
            new_offsets[tensor_name] = offset

        return new_offsets

    def _sequential_layout(
        self,
        plan: MemoryPlan,
        spilled_tensors: Dict[str, SpillSlot],
        max_memory: int,
    ) -> Dict[str, int]:
        """Fallback sequential layout without liveness analysis."""
        new_offsets = {}
        current_offset = 0
        alignment = 16

        for buf in plan.buffers:
            buf_tensors = [t for t in buf.tensors if t in plan.tensor_info]
            if not buf_tensors:
                continue

            aligned_offset = ((current_offset + alignment - 1) // alignment) * alignment

            if aligned_offset + buf.size > max_memory:
                raise RuntimeError(
                    f"Sequential layout exceeds max_memory ({max_memory}). "
                    f"Need {aligned_offset + buf.size} bytes. "
                    f"Pass CompileContext to enable lifetime-aware reuse."
                )

            for tensor_name in buf_tensors:
                new_offsets[tensor_name] = aligned_offset

            current_offset = aligned_offset + buf.size

        return new_offsets

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
