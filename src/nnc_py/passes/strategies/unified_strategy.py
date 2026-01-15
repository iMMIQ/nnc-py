"""Unified memory allocation strategy adapter.

This module adapts the UnifiedMemoryPass algorithm to work
with the new MemoryAllocationStrategy interface.
"""

from typing import Dict, List, Optional, Set, Tuple

from nnc_py.ir.context import CompileContext
from nnc_py.passes.liveness import TensorLiveness
from nnc_py.passes.memory_plan import MemoryBuffer
from nnc_py.passes.memory_strategy import (
    AllocationStrategy,
    MemoryAllocationPlan,
    MemoryAllocationStrategy,
    SpillPoint,
    ReloadPoint,
    TensorAllocation,
)


class UsePointInfo:
    """Tensor use point information."""

    def __init__(
        self,
        tensor_name: str,
        produce_at: int,
        use_at: List[int],
        size: int,
        is_input: bool = False,
        is_output: bool = False,
    ):
        self.tensor_name = tensor_name
        self.produce_at = produce_at
        self.use_at = use_at
        self.size = size
        self.is_input = is_input
        self.is_output = is_output

    @property
    def last_use(self) -> int:
        """Last use node index."""
        return max(self.use_at) if self.use_at else self.produce_at

    def distance_to_next_use(self, node_idx: int) -> int:
        """Distance to next use."""
        future_uses = [u for u in self.use_at if u > node_idx]
        return min(future_uses) - node_idx if future_uses else float('inf')


class ResidencyDecision:
    """Residency decision for a tensor."""

    def __init__(self, tensor_name: str):
        self.tensor_name = tensor_name
        self.resident_ranges: List[Tuple[int, int]] = []
        self.spill_after: Set[int] = set()
        self.reload_before: Set[int] = set()
        self.slow_pool_offset: int = 0

    def is_resident_at(self, node_idx: int) -> bool:
        """Check if tensor is resident at node."""
        for start, end in self.resident_ranges:
            if start <= node_idx <= end:
                return True
        return False

    def add_range(self, start: int, end: int) -> None:
        """Add resident range."""
        self.resident_ranges.append((start, end))
        self.resident_ranges.sort(key=lambda x: x[0])
        self._merge_overlapping_ranges()

    def _merge_overlapping_ranges(self) -> None:
        """Merge overlapping ranges."""
        if not self.resident_ranges:
            return

        merged = [self.resident_ranges[0]]
        for current_start, current_end in self.resident_ranges[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        self.resident_ranges = merged

    def truncate_after(self, node_idx: int) -> None:
        """Truncate ranges after node."""
        new_ranges = []
        for start, end in self.resident_ranges:
            if start <= node_idx <= end:
                new_ranges.append((start, node_idx))
            elif end < node_idx:
                new_ranges.append((start, end))
        self.resident_ranges = new_ranges


class UnifiedAllocationStrategy(MemoryAllocationStrategy):
    """Adapter for the UnifiedMemoryPass algorithm.

    This strategy uses use-point analysis for precise memory management,
    actively using spill/reload to meet memory constraints.
    """

    DEFAULT_ALIGNMENT = 16
    MIN_BUFFER_SIZE = 16
    MAX_OPTIMIZATION_ITERATIONS = 100

    @property
    def name(self) -> str:
        return "unified"

    @property
    def strategy_type(self) -> AllocationStrategy:
        return AllocationStrategy.UNIFIED

    def allocate(
        self,
        ctx: CompileContext,
        liveness_map: Dict[str, TensorLiveness],
        max_memory: Optional[int] = None,
    ) -> MemoryAllocationPlan:
        """Execute unified memory allocation."""
        if max_memory is None:
            max_memory = float("inf")

        nodes = ctx.graph.topological_sort()
        num_nodes = len(nodes)
        node_index = {node.name: i for i, node in enumerate(nodes)}

        # Phase 1: Use point analysis
        use_points = self._analyze_use_points(ctx, nodes, node_index)

        # Validate largest tensor fits
        largest_size = max(
            (info.size for info in use_points.values() if not info.is_input),
            default=0
        )
        self._validate_tensor_fits(largest_size, max_memory)

        # Phase 2: Initial residency
        residency = self._initial_residency(use_points, num_nodes)

        # Phase 3: Peak constraint optimization
        if max_memory != float("inf"):
            residency = self._optimize_for_peak(
                nodes, use_points, residency, max_memory
            )

        # Phase 4: Buffer allocation
        buffers, tensor_to_buffer = self._allocate_buffers(
            residency, use_points
        )

        # Phase 5: Generate spill/reload points
        spill_points, reload_points = self._generate_spill_reload(
            nodes, node_index, use_points, residency, tensor_to_buffer, buffers
        )

        # Calculate memory usage
        total_fast_memory = sum(b.size for b in buffers) if buffers else 0
        slow_memory_size = self._calculate_slow_memory_size(use_points, residency)

        # Create plan
        plan = MemoryAllocationPlan(
            strategy_name=self.name,
            total_fast_memory=total_fast_memory,
            total_slow_memory=slow_memory_size,
            num_buffers=len(buffers),
            buffers=buffers,
            tensor_to_buffer=tensor_to_buffer,
            spill_points=spill_points,
            reload_points=reload_points,
        )

        # Fill tensor allocations
        for tensor_name, info in use_points.items():
            if tensor_name in tensor_to_buffer:
                buffer_id = tensor_to_buffer[tensor_name]
                buffer = buffers[buffer_id]
                decision = residency.get(tensor_name)

                alloc = TensorAllocation(
                    tensor_name=tensor_name,
                    buffer_id=buffer_id,
                    offset=buffer.offset,
                    size=info.size,
                )

                if decision and (decision.spill_after or decision.reload_before):
                    alloc.is_spilled = True
                    if decision.spill_after:
                        alloc.spill_after = min(decision.spill_after)
                    if decision.reload_before:
                        alloc.reload_before = list(decision.reload_before)

                plan.tensor_allocations[tensor_name] = alloc

        # Calculate peak memory
        plan.peak_memory = self._calculate_peak_memory(nodes, use_points, residency, buffers, tensor_to_buffer)

        return plan

    def _analyze_use_points(
        self,
        ctx: CompileContext,
        nodes: List,
        node_index: Dict[str, int],
    ) -> Dict[str, UsePointInfo]:
        """Analyze use points for each tensor."""
        graph = ctx.graph
        use_points = {}

        for tensor_name in graph.tensors:
            tensor = graph.get_tensor(tensor_name)
            size = tensor.byte_size()
            if size < 0:
                continue

            # Production node
            producers = graph.get_producers(tensor_name)
            produce_at = node_index[producers[0].name] if producers else 0

            # Use nodes
            consumers = graph.get_consumers(tensor_name)
            use_at = [node_index[c.name] for c in consumers]

            is_input = tensor_name in graph.inputs
            is_output = tensor_name in graph.outputs

            use_points[tensor_name] = UsePointInfo(
                tensor_name=tensor_name,
                produce_at=produce_at,
                use_at=use_at,
                size=size,
                is_input=is_input,
                is_output=is_output,
            )

        return use_points

    def _initial_residency(
        self,
        use_points: Dict[str, UsePointInfo],
        num_nodes: int,
    ) -> Dict[str, ResidencyDecision]:
        """Generate initial residency decisions."""
        residency = {}

        for tensor_name, info in use_points.items():
            if info.is_input:
                resident_ranges = [(0, num_nodes - 1)]
            elif info.is_output:
                resident_ranges = [(info.produce_at, num_nodes - 1)]
            else:
                resident_ranges = [(info.produce_at, info.last_use)]

            decision = ResidencyDecision(tensor_name)
            decision.resident_ranges = resident_ranges
            residency[tensor_name] = decision

        return residency

    def _optimize_for_peak(
        self,
        nodes: List,
        use_points: Dict[str, UsePointInfo],
        residency: Dict[str, ResidencyDecision],
        memory_limit: int,
    ) -> Dict[str, ResidencyDecision]:
        """Iteratively optimize to meet peak constraint."""
        for _iteration in range(self.MAX_OPTIMIZATION_ITERATIONS):
            node_memory = self._calculate_node_memory(nodes, use_points, residency)
            peak_memory = max(nm["usage"] for nm in node_memory)

            if peak_memory <= memory_limit:
                break

            # Find peak nodes
            peak_idx = max(range(len(node_memory)), key=lambda i: node_memory[i]["usage"])
            peak_tensors = node_memory[peak_idx]["tensors"]

            # Select spill candidates
            candidates = []
            excess = peak_memory - memory_limit

            for tensor_name in peak_tensors:
                info = use_points.get(tensor_name)
                if not info or peak_idx in info.use_at:
                    continue

                distance = info.distance_to_next_use(peak_idx)
                priority = info.size / max(distance, 1)
                candidates.append((tensor_name, priority, info.size))

            candidates.sort(key=lambda x: x[1], reverse=True)

            # Apply spill
            released = 0
            for name, _, size in candidates:
                self._apply_spill(name, peak_idx, residency, use_points)
                released += size
                if released >= excess:
                    break

        return residency

    def _calculate_node_memory(
        self,
        nodes: List,
        use_points: Dict[str, UsePointInfo],
        residency: Dict[str, ResidencyDecision],
    ) -> List[Dict]:
        """Calculate memory usage per node."""
        node_memory = []

        for i, node in enumerate(nodes):
            tensors_resident = set()
            for tensor_name, decision in residency.items():
                if decision.is_resident_at(i):
                    tensors_resident.add(tensor_name)

            usage = sum(
                use_points[t].size
                for t in tensors_resident
                if t in use_points
            )

            node_memory.append({
                "node": node,
                "tensors": tensors_resident,
                "usage": usage,
            })

        return node_memory

    def _apply_spill(
        self,
        tensor_name: str,
        spill_at_idx: int,
        residency: Dict[str, ResidencyDecision],
        use_points: Dict[str, UsePointInfo],
    ) -> None:
        """Apply spill decision."""
        decision = residency[tensor_name]
        info = use_points[tensor_name]

        decision.truncate_after(spill_at_idx)
        decision.spill_after.add(spill_at_idx)

        for use_idx in info.use_at:
            if use_idx > spill_at_idx:
                decision.reload_before.add(use_idx)
                decision.resident_ranges.append((use_idx, use_idx))

        decision._merge_overlapping_ranges()

    def _allocate_buffers(
        self,
        residency: Dict[str, ResidencyDecision],
        use_points: Dict[str, UsePointInfo],
    ) -> Tuple[List[MemoryBuffer], Dict[str, int]]:
        """Allocate buffers based on residency."""
        buffers: List[MemoryBuffer] = []
        tensor_to_buffer: Dict[str, int] = {}
        current_offset = 0

        sorted_tensors = sorted(
            [(name, info) for name, info in use_points.items() if name in residency],
            key=lambda x: x[1].size,
            reverse=True
        )

        for tensor_name, info in sorted_tensors:
            decision = residency[tensor_name]
            tensor_size = info.size

            assigned_buffer = None
            for buf in buffers:
                if self._can_reuse_buffer(buf, tensor_size, decision, residency):
                    assigned_buffer = buf
                    break

            if assigned_buffer is None:
                aligned_offset = self._align(current_offset, self.DEFAULT_ALIGNMENT)
                buffer_size = max(self._align(tensor_size, self.DEFAULT_ALIGNMENT), self.MIN_BUFFER_SIZE)

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

            if hasattr(assigned_buffer, 'add_tensor'):
                assigned_buffer.add_tensor(tensor_name)
            else:
                assigned_buffer.tensors.append(tensor_name)

            tensor_to_buffer[tensor_name] = assigned_buffer.id

        return buffers, tensor_to_buffer

    def _can_reuse_buffer(
        self,
        buffer: MemoryBuffer,
        tensor_size: int,
        decision: ResidencyDecision,
        residency: Dict[str, ResidencyDecision],
    ) -> bool:
        """Check if buffer can be reused."""
        if buffer.size < tensor_size:
            return False

        for existing_tensor in buffer.tensors:
            existing_decision = residency.get(existing_tensor)
            if existing_decision and self._residency_ranges_overlap(decision, existing_decision):
                return False

        return True

    def _residency_ranges_overlap(
        self,
        a: ResidencyDecision,
        b: ResidencyDecision,
    ) -> bool:
        """Check if residency ranges overlap."""
        for a_start, a_end in a.resident_ranges:
            for b_start, b_end in b.resident_ranges:
                if not (a_end < b_start or b_end < a_start):
                    return True
        return False

    def _align(self, size: int, alignment: int) -> int:
        """Align size."""
        return ((size + alignment - 1) // alignment) * alignment

    def _generate_spill_reload(
        self,
        nodes: List,
        node_index: Dict[str, int],
        use_points: Dict[str, UsePointInfo],
        residency: Dict[str, ResidencyDecision],
        tensor_to_buffer: Dict[str, int],
        buffers: List[MemoryBuffer],
    ) -> Tuple[List[SpillPoint], List[ReloadPoint]]:
        """Generate spill/reload points."""
        spill_points = []
        reload_points = []

        slow_offset = 0
        for tensor_name, decision in residency.items():
            if decision.spill_after or decision.reload_before:
                decision.slow_pool_offset = slow_offset
                info = use_points.get(tensor_name)
                if info:
                    slow_offset += self._align(info.size, 16)

        for tensor_name, decision in residency.items():
            if not decision.spill_after:
                continue

            info = use_points.get(tensor_name)
            buffer_id = tensor_to_buffer.get(tensor_name, 0)
            buf = buffers[buffer_id] if 0 <= buffer_id < len(buffers) else None

            if not info or not buf:
                continue

            for spill_after_idx in decision.spill_after:
                node = nodes[spill_after_idx]
                spill_points.append(SpillPoint(
                    tensor_name=tensor_name,
                    after_node=node.name,
                    after_node_idx=spill_after_idx,
                    from_buffer_id=buffer_id,
                    from_fast_offset=buf.offset,
                    to_slow_offset=decision.slow_pool_offset,
                    size=info.size,
                ))

        for tensor_name, decision in residency.items():
            if not decision.reload_before:
                continue

            info = use_points.get(tensor_name)
            buffer_id = tensor_to_buffer.get(tensor_name, 0)
            buf = buffers[buffer_id] if 0 <= buffer_id < len(buffers) else None

            if not info or not buf:
                continue

            for reload_before_idx in decision.reload_before:
                node = nodes[reload_before_idx]
                reload_points.append(ReloadPoint(
                    tensor_name=tensor_name,
                    before_node=node.name,
                    before_node_idx=reload_before_idx,
                    from_slow_offset=decision.slow_pool_offset,
                    to_buffer_id=buffer_id,
                    to_fast_offset=buf.offset,
                    size=info.size,
                ))

        spill_points.sort(key=lambda p: p.after_node_idx)
        reload_points.sort(key=lambda p: p.before_node_idx)

        return spill_points, reload_points

    def _calculate_slow_memory_size(
        self,
        use_points: Dict[str, UsePointInfo],
        residency: Dict[str, ResidencyDecision],
    ) -> int:
        """Calculate slow memory size."""
        total = 0
        for tensor_name, decision in residency.items():
            if decision.spill_after or decision.reload_before:
                info = use_points.get(tensor_name)
                if info:
                    total += self._align(info.size, 16)
        return total

    def _calculate_peak_memory(
        self,
        nodes: List,
        use_points: Dict[str, UsePointInfo],
        residency: Dict[str, ResidencyDecision],
        buffers: List[MemoryBuffer],
        tensor_to_buffer: Dict[str, int],
    ) -> int:
        """Calculate peak memory."""
        peak = 0

        for i in range(len(nodes)):
            resident_tensors = set()
            for tensor_name, decision in residency.items():
                if decision.is_resident_at(i):
                    resident_tensors.add(tensor_name)

            used_buffers = set()
            for t in resident_tensors:
                bid = tensor_to_buffer.get(t, -1)
                if bid >= 0:
                    used_buffers.add(bid)

            memory = sum(
                buffers[bid].size
                for bid in used_buffers
                if 0 <= bid < len(buffers)
            )
            peak = max(peak, memory)

        return peak
