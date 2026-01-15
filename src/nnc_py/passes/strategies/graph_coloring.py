"""Graph coloring based memory allocation strategy.

This module implements memory allocation using graph coloring algorithms.
The interference graph represents tensor liveness overlaps, and coloring
determines which tensors can share the same memory buffer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum

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


class ColoringHeuristic(Enum):
    """Graph coloring heuristic algorithms."""
    WELSH_POWELL = "welsh_powell"    # Largest degree first
    DSATUR = "dsatur"                # Degree of saturation
    LARGEST_FIRST = "largest_first"  # Largest tensor first
    SMALLEST_LAST = "smallest_last"  # Smallest last (for packing)


@dataclass
class InterferenceGraph:
    """Interference graph for memory allocation.

    Nodes = tensors
    Edges = liveness overlap (cannot share same buffer/color)
    """
    # Adjacency list representation
    nodes: Set[str] = field(default_factory=set)
    edges: Dict[str, Set[str]] = field(default_factory=dict)

    # Tensor metadata
    tensor_sizes: Dict[str, int] = field(default_factory=dict)
    tensor_liveness: Dict[str, TensorLiveness] = field(default_factory=dict)

    # Coloring result
    colors: Dict[str, int] = field(default_factory=dict)  # tensor -> color (buffer_id)
    num_colors: int = 0

    def build_from_liveness(
        self,
        liveness_map: Dict[str, TensorLiveness],
        tensor_sizes: Dict[str, int],
    ) -> None:
        """Build interference graph from liveness information."""
        self.tensor_liveness = liveness_map.copy()
        self.tensor_sizes = tensor_sizes.copy()

        # Add all tensors as nodes
        for tensor_name in liveness_map:
            self.add_node(tensor_name)

        # Add edges for overlapping lifetimes
        tensor_names = list(liveness_map.keys())
        for i, t1 in enumerate(tensor_names):
            for t2 in tensor_names[i + 1:]:
                if self._liveness_overlaps(liveness_map[t1], liveness_map[t2]):
                    self.add_edge(t1, t2)

    def add_node(self, tensor_name: str) -> None:
        """Add a tensor node to the graph."""
        self.nodes.add(tensor_name)
        if tensor_name not in self.edges:
            self.edges[tensor_name] = set()

    def add_edge(self, t1: str, t2: str) -> None:
        """Add an edge between two tensors (they interfere)."""
        self.edges.setdefault(t1, set()).add(t2)
        self.edges.setdefault(t2, set()).add(t1)

    def _liveness_overlaps(self, l1: TensorLiveness, l2: TensorLiveness) -> bool:
        """Check if two liveness ranges overlap."""
        return not (l1.live_end < l2.live_start or l2.live_end < l1.live_start)

    def degree(self, tensor: str) -> int:
        """Get degree (number of conflicts) for a tensor."""
        return len(self.edges.get(tensor, set()))

    def get_conflicts(self, tensor: str) -> Set[str]:
        """Get all tensors that conflict with this tensor."""
        return self.edges.get(tensor, set()).copy()

    def get_color(self, tensor: str) -> Optional[int]:
        """Get the color assigned to a tensor."""
        return self.colors.get(tensor)

    def set_color(self, tensor: str, color: int) -> None:
        """Assign a color to a tensor."""
        self.colors[tensor] = color

    def is_colored(self, tensor: str) -> bool:
        """Check if a tensor has been colored."""
        return tensor in self.colors

    def get_uncolored_nodes(self) -> Set[str]:
        """Get all nodes that haven't been colored yet."""
        return self.nodes - self.colors.keys()


class GraphColoringStrategy(MemoryAllocationStrategy):
    """Memory allocation using graph coloring.

    Algorithm:
    1. Build interference graph where edges = liveness overlap
    2. Color the graph using heuristic (Welsh-Powell, DSatur, etc.)
    3. Each color = one shared buffer
    4. Assign offsets within each buffer based on size alignment
    """

    # Configuration
    DEFAULT_HEURISTIC = ColoringHeuristic.WELSH_POWELL
    DEFAULT_ALIGNMENT = 16
    MIN_BUFFER_SIZE = 16

    def __init__(self, heuristic: ColoringHeuristic = DEFAULT_HEURISTIC):
        self.heuristic = heuristic

    @property
    def name(self) -> str:
        return f"graph_coloring_{self.heuristic.value}"

    @property
    def strategy_type(self) -> AllocationStrategy:
        return AllocationStrategy.GRAPH_COLORING

    def allocate(
        self,
        ctx: CompileContext,
        liveness_map: Dict[str, TensorLiveness],
        max_memory: Optional[int] = None,
    ) -> MemoryAllocationPlan:
        """Execute graph coloring allocation."""
        if max_memory is None:
            max_memory = float("inf")

        # Step 1: Collect tensor sizes
        tensor_sizes = self._collect_tensor_sizes(ctx, liveness_map)

        # Step 2: Validate largest tensor fits
        largest_size = max(tensor_sizes.values(), default=0)
        self._validate_tensor_fits(largest_size, max_memory)

        # Step 3: Build interference graph
        ig = InterferenceGraph()
        ig.build_from_liveness(liveness_map, tensor_sizes)

        # Step 4: Color the graph
        if self.heuristic == ColoringHeuristic.WELSH_POWELL:
            colors = self._welsh_powell_coloring(ig)
        elif self.heuristic == ColoringHeuristic.DSATUR:
            colors = self._dsatur_coloring(ig)
        elif self.heuristic == ColoringHeuristic.LARGEST_FIRST:
            colors = self._largest_first_coloring(ig)
        elif self.heuristic == ColoringHeuristic.SMALLEST_LAST:
            colors = self._smallest_last_coloring(ig)
        else:
            colors = self._greedy_coloring(ig)

        ig.colors = colors
        ig.num_colors = max(colors.values()) + 1 if colors else 0

        # Step 5: Assign buffers (one per color)
        buffers, tensor_to_buffer = self._create_buffers_from_colors(
            ig, ctx.metadata.get("memory_alignment", self.DEFAULT_ALIGNMENT)
        )

        # Step 6: Create allocation plan
        total_fast_memory = sum(b.size for b in buffers) if buffers else 0

        plan = MemoryAllocationPlan(
            strategy_name=self.name,
            total_fast_memory=total_fast_memory,
            num_buffers=len(buffers),
            buffers=buffers,
            tensor_to_buffer=tensor_to_buffer,
        )

        # Step 7: Fill in tensor allocations
        for tensor_name, buffer_id in tensor_to_buffer.items():
            buffer = buffers[buffer_id]
            plan.tensor_allocations[tensor_name] = TensorAllocation(
                tensor_name=tensor_name,
                buffer_id=buffer_id,
                offset=buffer.offset,
                size=tensor_sizes[tensor_name],
            )

        # Step 8: Calculate peak memory
        plan.peak_memory = self._calculate_peak_memory(ctx, liveness_map, plan)

        # Step 9: Handle overflow if needed
        if max_memory != float("inf") and plan.peak_memory > max_memory:
            plan = self._handle_overflow(ctx, plan, liveness_map, max_memory)

        return plan

    def _welsh_powell_coloring(self, ig: InterferenceGraph) -> Dict[str, int]:
        """Welsh-Powell algorithm: sort by degree, color in order.

        This is a classic graph coloring heuristic that often gives
        good results for interference graphs.
        """
        colors: Dict[str, int] = {}

        # Sort nodes by degree (descending)
        nodes_by_degree = sorted(
            ig.nodes,
            key=lambda n: ig.degree(n),
            reverse=True
        )

        for node in nodes_by_degree:
            # Find used colors among neighbors
            used_colors = set()
            for neighbor in ig.get_conflicts(node):
                if neighbor in colors:
                    used_colors.add(colors[neighbor])

            # Assign lowest available color
            color = 0
            while color in used_colors:
                color += 1
            colors[node] = color

        return colors

    def _dsatur_coloring(self, ig: InterferenceGraph) -> Dict[str, int]:
        """DSatur (Degree of Saturation) algorithm.

        Choose the node with highest saturation (number of different
        colors used by neighbors), breaking ties by degree.
        """
        colors: Dict[str, int] = {}
        saturation: Dict[str, int] = {n: 0 for n in ig.nodes}

        while len(colors) < len(ig.nodes):
            # Find uncolored node with highest saturation
            uncolored = list(ig.get_uncolored_nodes())

            # Sort by saturation (desc), then by degree (desc)
            node = max(
                uncolored,
                key=lambda n: (saturation[n], ig.degree(n))
            )

            # Find used colors among neighbors
            used_colors = set()
            for neighbor in ig.get_conflicts(node):
                if neighbor in colors:
                    used_colors.add(colors[neighbor])

            # Assign lowest available color
            color = 0
            while color in used_colors:
                color += 1
            colors[node] = color

            # Update saturation for neighbors
            for neighbor in ig.get_conflicts(node):
                if neighbor not in colors:
                    # Count distinct colors in neighborhood
                    neighbor_colors = set()
                    for nn in ig.get_conflicts(neighbor):
                        if nn in colors:
                            neighbor_colors.add(colors[nn])
                    saturation[neighbor] = len(neighbor_colors)

        return colors

    def _largest_first_coloring(self, ig: InterferenceGraph) -> Dict[str, int]:
        """Color largest tensors first (for better packing)."""
        colors: Dict[str, int] = {}

        # Sort by size (descending)
        nodes_by_size = sorted(
            ig.nodes,
            key=lambda n: ig.tensor_sizes.get(n, 0),
            reverse=True
        )

        for node in nodes_by_size:
            used_colors = set()
            for neighbor in ig.get_conflicts(node):
                if neighbor in colors:
                    used_colors.add(colors[neighbor])

            color = 0
            while color in used_colors:
                color += 1
            colors[node] = color

        return colors

    def _smallest_last_coloring(self, ig: InterferenceGraph) -> Dict[str, int]:
        """Color smallest tensors last - for compact packing."""
        colors: Dict[str, int] = {}

        # Sort by size (ascending)
        nodes_by_size = sorted(
            ig.nodes,
            key=lambda n: ig.tensor_sizes.get(n, 0),
        )

        for node in nodes_by_size:
            used_colors = set()
            for neighbor in ig.get_conflicts(node):
                if neighbor in colors:
                    used_colors.add(colors[neighbor])

            color = 0
            while color in used_colors:
                color += 1
            colors[node] = color

        return colors

    def _greedy_coloring(self, ig: InterferenceGraph) -> Dict[str, int]:
        """Simple greedy coloring in given order."""
        colors: Dict[str, int] = {}

        for node in ig.nodes:
            used_colors = set()
            for neighbor in ig.get_conflicts(node):
                if neighbor in colors:
                    used_colors.add(colors[neighbor])

            color = 0
            while color in used_colors:
                color += 1
            colors[node] = color

        return colors

    def _create_buffers_from_colors(
        self,
        ig: InterferenceGraph,
        alignment: int,
    ) -> Tuple[List[MemoryBuffer], Dict[str, int]]:
        """Create buffers from coloring result."""
        # Group tensors by color
        color_to_tensors: Dict[int, List[str]] = {}
        for tensor, color in ig.colors.items():
            color_to_tensors.setdefault(color, []).append(tensor)

        # For each color, find max size needed
        buffers: List[MemoryBuffer] = []
        tensor_to_buffer: Dict[str, int] = {}

        sorted_colors = sorted(color_to_tensors.keys())

        current_offset = 0
        for color in sorted_colors:
            tensors = color_to_tensors[color]
            max_size = max(ig.tensor_sizes.get(t, 0) for t in tensors)
            buffer_size = max(
                self._align(max_size, alignment),
                self.MIN_BUFFER_SIZE
            )

            buffer = MemoryBuffer(
                id=color,
                offset=current_offset,
                size=buffer_size,
                alignment=alignment,
                tensors=[],
            )
            # Use add_tensor method if available, otherwise modify directly
            if hasattr(buffer, 'add_tensor'):
                for tensor in tensors:
                    buffer.add_tensor(tensor)
            else:
                buffer.tensors = tensors
            buffers.append(buffer)

            for tensor in tensors:
                tensor_to_buffer[tensor] = color

            current_offset += buffer_size

        return buffers, tensor_to_buffer

    def _collect_tensor_sizes(
        self,
        ctx: CompileContext,
        liveness_map: Dict[str, TensorLiveness],
    ) -> Dict[str, int]:
        """Collect sizes for all tensors in liveness map."""
        sizes = {}
        for tensor_name in liveness_map:
            tensor = ctx.graph.get_tensor(tensor_name)
            size = tensor.byte_size()
            if size < 0:
                # Skip tensors with unknown size
                continue
            sizes[tensor_name] = size
        return sizes

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

    def _handle_overflow(
        self,
        ctx: CompileContext,
        plan: MemoryAllocationPlan,
        liveness_map: Dict[str, TensorLiveness],
        max_memory: int,
    ) -> MemoryAllocationPlan:
        """Handle memory overflow by selective spilling.

        This implements a simple spill strategy - select tensors with
        the largest size * lifetime product.
        """
        excess = plan.peak_memory - max_memory

        # Find candidate tensors for spilling
        candidates = self._select_spill_candidates(
            ctx, liveness_map, plan, excess
        )

        # Generate spill/reload points
        spill_points, reload_points = self._generate_spill_reload(
            ctx, candidates, liveness_map, plan
        )

        # Update plan with spill information
        plan.spill_points = spill_points
        plan.reload_points = reload_points
        plan.total_slow_memory = sum(
            ctx.graph.get_tensor(t).byte_size()
            for t in candidates
        )

        # Mark tensors as spilled in allocation
        for tensor_name in candidates:
            if tensor_name in plan.tensor_allocations:
                plan.tensor_allocations[tensor_name].is_spilled = True

        # Recalculate peak (simplified - actual peak may differ)
        plan.peak_memory = max_memory

        return plan

    def _select_spill_candidates(
        self,
        ctx: CompileContext,
        liveness_map: Dict[str, TensorLiveness],
        plan: MemoryAllocationPlan,
        excess: int,
    ) -> Set[str]:
        """Select tensors to spill based on cost/benefit."""
        candidates = []

        for tensor_name, liveness in liveness_map.items():
            # Skip input tensors
            if tensor_name in ctx.graph.inputs:
                continue

            # Calculate spill priority
            tensor = ctx.graph.get_tensor(tensor_name)
            tensor_size = tensor.byte_size()
            if tensor_size < 0:
                continue

            lifetime = liveness.live_end - liveness.live_start + 1

            # Priority: larger size, longer lifetime
            priority = tensor_size / max(lifetime, 1)

            # Boost if tensor has multiple uses (gaps)
            consumers = ctx.graph.get_consumers(tensor_name)
            if len(consumers) > 1:
                priority *= 1.5

            candidates.append((tensor_name, priority, tensor_size))

        # Sort by priority and select until excess is covered
        candidates.sort(key=lambda x: x[1], reverse=True)

        selected = set()
        total_released = 0
        for name, _, size in candidates:
            selected.add(name)
            total_released += size
            if total_released >= excess:
                break

        return selected

    def _generate_spill_reload(
        self,
        ctx: CompileContext,
        spilled: Set[str],
        liveness_map: Dict[str, TensorLiveness],
        plan: MemoryAllocationPlan,
    ) -> Tuple[List[SpillPoint], List[ReloadPoint]]:
        """Generate spill and reload points."""
        nodes = ctx.graph.topological_sort()
        node_index = {node.name: i for i, node in enumerate(nodes)}

        spill_points = []
        reload_points = []

        slow_offset = 0

        for tensor_name in spilled:
            liveness = liveness_map[tensor_name]
            tensor = ctx.graph.get_tensor(tensor_name)
            size = tensor.byte_size()
            buffer_id = plan.get_buffer_for_tensor(tensor_name)
            buffer = plan.buffers[buffer_id] if 0 <= buffer_id < len(plan.buffers) else None

            if buffer is None:
                continue

            # Spill after last use
            spill_node_idx = liveness.live_end
            if 0 <= spill_node_idx < len(nodes):
                spill_points.append(SpillPoint(
                    tensor_name=tensor_name,
                    after_node=nodes[spill_node_idx].name,
                    after_node_idx=spill_node_idx,
                    from_buffer_id=buffer_id,
                    from_fast_offset=buffer.offset,
                    to_slow_offset=slow_offset,
                    size=size,
                ))

            # Reload at first use (if there are multiple uses)
            consumers = ctx.graph.get_consumers(tensor_name)
            for consumer in consumers:
                reload_idx = node_index.get(consumer.name)
                if reload_idx is not None and reload_idx > liveness.live_start:
                    reload_points.append(ReloadPoint(
                        tensor_name=tensor_name,
                        before_node=consumer.name,
                        before_node_idx=reload_idx,
                        from_slow_offset=slow_offset,
                        to_buffer_id=buffer_id,
                        to_fast_offset=buffer.offset,
                        size=size,
                    ))

            slow_offset += size

        # Sort by execution order
        spill_points.sort(key=lambda p: p.after_node_idx)
        reload_points.sort(key=lambda p: p.before_node_idx)

        return spill_points, reload_points
