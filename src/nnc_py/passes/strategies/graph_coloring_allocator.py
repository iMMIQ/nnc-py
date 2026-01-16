"""Graph coloring memory allocation with spill support.

This module implements register allocation for memory tensors using
graph coloring with the simplify-spill-select algorithm:

1. Build: Construct interference graph from liveness analysis
   - Nodes = tensors, Edges = liveness overlap
2. Simplify: Remove nodes with degree < K (K = available colors/buffers)
3. Spill: When all nodes have degree >= K, select a node to spill
4. Select: Assign colors in reverse order of removal

Key properties:
- Guarantees no overlap: Interfering tensors never share memory
- Optimal K-coloring: If K colors suffice, finds the coloring
- Proper spill: Selects appropriate tensors to spill to slow memory
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple

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
class SimplifyStackEntry:
    """Entry on the simplify stack during graph simplification."""
    tensor: str
    was_spilled: bool = False
    spill_reason: str = ""


class GraphColoringAllocator(MemoryAllocationStrategy):
    """Graph coloring allocator with spill support for memory allocation.

    This implements the graph coloring algorithm adapted for memory
    allocation. The key insight is that by simplifying the interference
    graph (removing low-degree nodes first), we can either:
    1. Find a K-coloring if one exists
    2. Identify nodes that must be spilled

    The algorithm guarantees that interfering tensors (overlapping lifetimes)
    never share the same memory location.
    """

    DEFAULT_ALIGNMENT = 16

    @property
    def name(self) -> str:
        return "graph_coloring"

    @property
    def strategy_type(self) -> AllocationStrategy:
        return AllocationStrategy.GRAPH_COLORING

    def allocate(
        self,
        ctx: CompileContext,
        liveness_map: Dict[str, TensorLiveness],
        max_memory: Optional[int] = None,
    ) -> MemoryAllocationPlan:
        """Execute LLVM-style allocation."""
        if max_memory is None:
            max_memory = float("inf")

        # Step 1: Build interference graph
        ig = self._build_interference_graph(ctx, liveness_map)

        # Step 2: Calculate K (max colors we can use)
        K = self._calculate_K(ig, max_memory)

        # Step 3: Iterated coloring with spill
        colors, spilled_tensors = self._iterated_coloring(ig, K, max_memory)

        # Step 4: Create buffers and assign offsets
        buffers, tensor_allocations = self._create_buffers_and_allocations(
            ig, colors, spilled_tensors
        )

        # Step 5: Generate spill/reload points for spilled tensors
        spill_points, reload_points = self._generate_spill_reload(
            ctx, spilled_tensors, liveness_map, tensor_allocations
        )

        # Calculate memory usage
        total_fast_memory = sum(b.size for b in buffers)
        slow_memory = sum(ig['tensor_sizes'].get(t, 0) for t in spilled_tensors)

        return MemoryAllocationPlan(
            strategy_name=self.name,
            total_fast_memory=total_fast_memory,
            total_slow_memory=slow_memory,
            peak_memory=total_fast_memory,
            num_buffers=len(buffers),
            buffers=buffers,
            tensor_allocations=tensor_allocations,
            tensor_to_buffer={t: a.buffer_id for t, a in tensor_allocations.items()},
            spill_points=spill_points,
            reload_points=reload_points,
        )

    def _build_interference_graph(
        self,
        ctx: CompileContext,
        liveness_map: Dict[str, TensorLiveness],
    ) -> Dict:
        """Build interference graph from liveness information.

        Returns:
            Dict with keys: 'nodes', 'edges', 'tensor_sizes', 'tensor_liveness'
        """
        nodes = set(liveness_map.keys())
        edges = {t: set() for t in nodes}
        tensor_sizes = {}
        tensor_liveness = {}

        for t, liv in liveness_map.items():
            tensor_liveness[t] = liv
            tensor = ctx.graph.get_tensor(t)
            if tensor:
                tensor_sizes[t] = tensor.byte_size()

        # Add edges for overlapping lifetimes
        tensor_list = list(nodes)
        for i, t1 in enumerate(tensor_list):
            for t2 in tensor_list[i + 1:]:
                liv1 = liveness_map[t1]
                liv2 = liveness_map[t2]
                if self._liveness_overlaps(liv1, liv2):
                    edges[t1].add(t2)
                    edges[t2].add(t1)

        return {
            'nodes': nodes,
            'edges': edges,
            'tensor_sizes': tensor_sizes,
            'tensor_liveness': tensor_liveness,
        }

    def _liveness_overlaps(self, l1: TensorLiveness, l2: TensorLiveness) -> bool:
        """Check if two liveness ranges overlap."""
        return not (l1.live_end < l2.live_start or l2.live_end < l1.live_start)

    def _calculate_K(self, ig: Dict, max_memory: int) -> int:
        """Calculate maximum number of colors (buffers) we can use.

        K represents the number of memory buffers (colors) available.
        Each buffer can hold one tensor at a time, but non-interfering
        tensors can share the same buffer.

        Args:
            ig: Interference graph
            max_memory: Maximum fast memory available

        Returns:
            Number of colors (buffers) we can use
        """
        if max_memory == float("inf"):
            return len(ig['nodes'])  # Unlimited colors

        max_size = max(ig['tensor_sizes'].values()) if ig['tensor_sizes'] else 1
        if max_size == 0:
            return 1

        # K is how many max-sized tensors can fit in memory
        # Each color needs its own buffer slot
        K = max(1, max_memory // max_size)
        return K

    def _iterated_coloring(
        self,
        ig: Dict,
        K: int,
        max_memory: int,
    ) -> Tuple[Dict[str, int], Set[str]]:
        """Execute iterated coloring with spill.

        This is the core LLVM algorithm:
        1. Simplify: Remove nodes with degree < K
        2. If no such nodes exist, spill one with largest size/degree
        3. Continue until graph is empty
        4. Select: Assign colors in reverse order

        Args:
            ig: Interference graph
            K: Number of colors available
            max_memory: Memory limit

        Returns:
            (colors, spilled) where colors maps tensor->color (-1 if spilled)
        """
        colors = {}
        spilled = set()
        stack: List[SimplifyStackEntry] = []

        # Make a copy of the graph we'll modify
        current_nodes = ig['nodes'].copy()
        current_edges = {t: s.copy() for t, s in ig['edges'].items()}

        def degree(t: str) -> int:
            return len(current_edges.get(t, set()))

        # Simplify and spill loop
        while current_nodes:
            # Find nodes with degree < K
            low_degree = [t for t in current_nodes if degree(t) < K]

            if low_degree:
                # Simplify: remove a low-degree node
                # Prefer larger nodes first (they're harder to place later)
                node = max(low_degree, key=lambda t: ig['tensor_sizes'].get(t, 0))
                stack.append(SimplifyStackEntry(tensor=node, was_spilled=False))
                current_nodes.remove(node)
                # Remove edges connected to this node
                for neighbor in list(current_edges.get(node, set())):
                    current_edges[neighbor].discard(node)
            else:
                # All nodes have degree >= K - need to spill
                # Spill heuristic: spill largest tensor with most conflicts
                candidates = [
                    (t, degree(t), ig['tensor_sizes'].get(t, 0))
                    for t in current_nodes
                ]
                # Sort by: size (desc), then degree (desc)
                candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)

                node = candidates[0][0]
                stack.append(SimplifyStackEntry(
                    tensor=node,
                    was_spilled=True,
                    spill_reason=f"degree={degree(node)} >= K={K}"
                ))
                spilled.add(node)
                current_nodes.remove(node)
                for neighbor in list(current_edges.get(node, set())):
                    current_edges[neighbor].discard(node)

        # Select phase: assign colors in reverse order of removal
        for entry in reversed(stack):
            if entry.was_spilled:
                # Spilled tensors still get a color for buffer allocation
                # but we track them separately for spill/reload generation
                colors[entry.tensor] = -1  # Mark as spilled
            else:
                # Find available color (not used by interfering neighbors)
                used_colors = set()
                for neighbor in ig['edges'].get(entry.tensor, set()):
                    if neighbor in colors:
                        neigh_color = colors[neighbor]
                        # Only respect colors of non-spilled neighbors
                        if neigh_color >= 0:
                            used_colors.add(neigh_color)

                # Find first available color
                for color in range(K):
                    if color not in used_colors:
                        colors[entry.tensor] = color
                        break
                else:
                    # No color available - spill this tensor too
                    colors[entry.tensor] = -1
                    spilled.add(entry.tensor)

        return colors, spilled

    def _create_buffers_and_allocations(
        self,
        ig: Dict,
        colors: Dict[str, int],
        spilled: Set[str],
    ) -> Tuple[List[MemoryBuffer], Dict[str, TensorAllocation]]:
        """Create memory buffers and tensor allocations from coloring result.

        Each color becomes one buffer. Tensors with the same color share
        that buffer (at different times, since they don't interfere).

        Spilled tensors are NOT allocated in fast memory - they use slow memory.

        Args:
            ig: Interference graph with tensor sizes
            colors: Tensor -> color mapping (-1 for spilled)
            spilled: Set of spilled tensor names

        Returns:
            (buffers, tensor_allocations)
        """
        # Group non-spilled tensors by color
        color_groups: Dict[int, List[str]] = {}

        for tensor, color in colors.items():
            # Only allocate non-spilled tensors in fast memory
            if color >= 0 and tensor not in spilled:
                color_groups.setdefault(color, []).append(tensor)

        # Create fast memory buffers
        buffers: List[MemoryBuffer] = []
        tensor_allocations: Dict[str, TensorAllocation] = {}
        current_offset = 0
        alignment = self.DEFAULT_ALIGNMENT

        for color in sorted(color_groups.keys()):
            tensors = color_groups[color]
            # Buffer size = max tensor size in this color
            max_size = max(
                (ig['tensor_sizes'].get(t, 0) for t in tensors),
                default=0
            )
            buffer_size = max(max_size, alignment)

            buffer = MemoryBuffer(
                id=len(buffers),
                offset=current_offset,
                size=buffer_size,
                alignment=alignment,
                tensors=tensors,
            )
            buffers.append(buffer)

            for tensor in tensors:
                tensor_allocations[tensor] = TensorAllocation(
                    tensor_name=tensor,
                    buffer_id=len(buffers) - 1,
                    offset=0,  # Offset within buffer is 0 (tensors share buffer)
                    size=ig['tensor_sizes'].get(tensor, 0),
                    is_spilled=False,
                )

            current_offset += buffer_size

        # Create allocations for spilled tensors (in slow memory, buffer_id=-1)
        slow_offset = 0
        for tensor in spilled:
            size = ig['tensor_sizes'].get(tensor, 0)
            tensor_allocations[tensor] = TensorAllocation(
                tensor_name=tensor,
                buffer_id=-1,  # Slow memory
                offset=slow_offset,
                size=size,
                is_spilled=True,
            )
            slow_offset += size

        return buffers, tensor_allocations

    def _generate_spill_reload(
        self,
        ctx: CompileContext,
        spilled: Set[str],
        liveness_map: Dict[str, TensorLiveness],
        allocations: Dict[str, TensorAllocation],
    ) -> Tuple[List[SpillPoint], List[ReloadPoint]]:
        """Generate spill and reload points for spilled tensors.

        Spilled tensors need to be:
        - Written to slow memory after their last use
        - Read back to fast memory before each use

        Args:
            ctx: Compilation context
            spilled: Set of spilled tensor names
            liveness_map: Tensor liveness information
            allocations: Tensor allocation information

        Returns:
            (spill_points, reload_points)
        """
        spill_points: List[SpillPoint] = []
        reload_points: List[ReloadPoint] = []

        nodes = ctx.graph.topological_sort()
        node_index = {n.name: i for i, n in enumerate(nodes)}

        for tensor in spilled:
            liv = liveness_map[tensor]
            alloc = allocations[tensor]

            # Spill after last use (producer creates it, then we spill after last consumer)
            spill_after_idx = liv.live_end
            if 0 <= spill_after_idx < len(nodes):
                spill_points.append(SpillPoint(
                    tensor_name=tensor,
                    after_node=nodes[spill_after_idx].name,
                    after_node_idx=spill_after_idx,
                    from_buffer_id=-1,  # Will need temp buffer
                    from_fast_offset=0,  # Will need temp offset
                    to_slow_offset=alloc.offset,
                    size=alloc.size,
                ))

            # Reload before each use after the producer
            consumers = ctx.graph.get_consumers(tensor)
            for consumer in consumers:
                use_idx = node_index.get(consumer.name, -1)
                if use_idx > liv.live_start:
                    reload_points.append(ReloadPoint(
                        tensor_name=tensor,
                        before_node=consumer.name,
                        before_node_idx=use_idx,
                        from_slow_offset=alloc.offset,
                        to_buffer_id=-1,  # Will get temp buffer
                        to_fast_offset=0,  # Will get temp offset
                        size=alloc.size,
                        reload_slot_id=-1,  # Will be assigned later
                    ))

        # Assign reload slot IDs to each reload point
        # Reloads at the same node need different slots if they're for different tensors
        reload_points = self._assign_reload_slots(reload_points)

        return spill_points, reload_points

    def _assign_reload_slots(
        self,
        reload_points: List[ReloadPoint],
    ) -> List[ReloadPoint]:
        """Assign reload slot IDs to reload points.

        For each node, assigns unique slot IDs to concurrent reloads.
        Slot IDs are per-node (0, 1, 2, ...) to allow temp buffer allocation.

        Args:
            reload_points: List of reload points

        Returns:
            List of reload points with slot IDs assigned
        """
        # Group reloads by node
        reloads_by_node: Dict[int, List[ReloadPoint]] = {}
        for rp in reload_points:
            reloads_by_node.setdefault(rp.before_node_idx, []).append(rp)

        # Assign slot IDs within each node
        result = []
        for rp in reload_points:
            node_reloads = reloads_by_node[rp.before_node_idx]
            # Find the index of this reload point in the node's reload list
            slot_id = node_reloads.index(rp)
            # Create a new ReloadPoint with the slot_id assigned
            result.append(ReloadPoint(
                tensor_name=rp.tensor_name,
                before_node=rp.before_node,
                before_node_idx=rp.before_node_idx,
                from_slow_offset=rp.from_slow_offset,
                to_buffer_id=rp.to_buffer_id,
                to_fast_offset=rp.to_fast_offset,
                size=rp.size,
                reload_slot_id=slot_id,
            ))

        return result
