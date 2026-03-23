"""Cost-aware memory allocator with reusable fast-memory regions."""

import heapq
from dataclasses import dataclass
from typing import Dict, Optional

from nnc_py.ir.context import CompileContext
from nnc_py.ir.node import Node, OpType
from nnc_py.passes.liveness import TensorLiveness
from nnc_py.passes.memory_plan import MemoryBuffer
from nnc_py.passes.memory_strategy import (
    AllocationStrategy,
    MemoryAllocationPlan,
    MemoryAllocationStrategy,
    ReloadPoint,
    SpillPoint,
    TensorAllocation,
)


@dataclass
class _ResidentTensor:
    """A tensor currently resident in fast memory."""

    name: str
    offset: int
    size: int


class CostAwareAllocator(MemoryAllocationStrategy):
    """Transfer-aware allocator that treats fast memory as a constrained cache."""

    DEFAULT_ALIGNMENT = 16
    MAX_EXACT_EVICTION_CANDIDATES = 12
    # These kernels are expected to tolerate input/output aliasing when the
    # reused input dies at the current node. Keep this list aligned with
    # runtime kernel behavior and the memory-limit regression tests.
    INPLACE_REUSE_OPS = {
        OpType.RELU,
        OpType.SIGMOID,
        OpType.TANH,
        OpType.ADD,
        OpType.SUB,
        OpType.MUL,
        OpType.DIV,
        OpType.CLIP,
    }

    def __init__(self) -> None:
        self._spill_bytes = 0
        self._reload_bytes = 0

    @property
    def name(self) -> str:
        return "cost_aware"

    @property
    def strategy_type(self) -> AllocationStrategy:
        return AllocationStrategy.COST_AWARE

    def allocate(
        self,
        ctx: CompileContext,
        liveness_map: Dict[str, TensorLiveness],
        max_memory: Optional[int] = None,
    ) -> MemoryAllocationPlan:
        """Allocate fast memory with reuse, selective eviction, and transfer tracking."""
        self._spill_bytes = 0
        self._reload_bytes = 0

        capacity = float("inf") if max_memory is None else max_memory
        nodes = ctx.graph.topological_sort()

        tensor_sizes: Dict[str, int] = {}
        for tensor_name in liveness_map:
            tensor = ctx.graph.get_tensor(tensor_name)
            if tensor is None:
                continue

            size = tensor.byte_size()
            if size >= 0:
                tensor_sizes[tensor_name] = size

        if capacity != float("inf"):
            max_node_demand = 0
            for node in nodes:
                unique_inputs = {
                    tensor_name for tensor_name in node.inputs if tensor_name in tensor_sizes
                }
                unique_outputs = {
                    tensor_name for tensor_name in node.outputs if tensor_name in tensor_sizes
                }
                demand = sum(
                    self._align(tensor_sizes[tensor_name], self.DEFAULT_ALIGNMENT)
                    for tensor_name in unique_inputs
                )
                demand += sum(
                    self._align(tensor_sizes[tensor_name], self.DEFAULT_ALIGNMENT)
                    for tensor_name in unique_outputs
                )
                max_node_demand = max(max_node_demand, demand)
            if max_node_demand > capacity:
                raise ValueError(
                    f"max_memory ({capacity}) < peak node demand ({max_node_demand}). "
                    f"Minimum required: {max_node_demand} bytes for the aligned fast-memory pool."
                )

        resident: Dict[str, _ResidentTensor] = {}
        free_regions: list[tuple[int, int]] = []
        tensor_allocations: Dict[str, TensorAllocation] = {}
        spill_points: list[SpillPoint] = []
        reload_points: list[ReloadPoint] = []
        spill_after: Dict[str, int] = {}
        reload_before: Dict[str, list[int]] = {}
        reload_slots_by_node: Dict[int, int] = {}
        slow_offsets: Dict[str, int] = {}
        node_memory_usage: list[int] = []

        next_fast_offset = 0
        next_slow_offset = 0
        peak_fast_end = 0
        peak_fast_footprint = 0

        def current_fast_footprint() -> int:
            if not resident:
                return 0
            return max(slot.offset + slot.size for slot in resident.values())

        def update_peak_fast_footprint() -> None:
            nonlocal peak_fast_footprint
            peak_fast_footprint = max(peak_fast_footprint, current_fast_footprint())

        def align_offset(offset: int) -> int:
            return self._align(offset, self.DEFAULT_ALIGNMENT)

        def can_extend_fast_pool(size: int) -> bool:
            candidate_offset = align_offset(next_fast_offset)
            candidate_end = candidate_offset + size
            return self._fits_fast_memory_budget(candidate_end, capacity)

        def record_allocation(tensor_name: str, offset: int) -> None:
            existing = tensor_allocations.get(tensor_name)
            if existing is not None:
                tensor_allocations[tensor_name] = TensorAllocation(
                    tensor_name=tensor_name,
                    buffer_id=existing.buffer_id,
                    offset=existing.offset,
                    size=existing.size,
                    is_spilled=existing.is_spilled,
                    spill_after=spill_after.get(tensor_name, existing.spill_after),
                    reload_before=reload_before.get(tensor_name, existing.reload_before),
                )
                return

            tensor_allocations[tensor_name] = TensorAllocation(
                tensor_name=tensor_name,
                buffer_id=0,
                offset=offset,
                size=tensor_sizes[tensor_name],
                is_spilled=tensor_name in slow_offsets,
                spill_after=spill_after.get(tensor_name),
                reload_before=reload_before.get(tensor_name),
            )

        def assign_slow_offset(tensor_name: str) -> int:
            nonlocal next_slow_offset

            if tensor_name in slow_offsets:
                return slow_offsets[tensor_name]

            slow_offset = align_offset(next_slow_offset)
            slow_offsets[tensor_name] = slow_offset
            next_slow_offset = slow_offset + tensor_sizes[tensor_name]
            return slow_offset

        def next_reload_slot(node_idx: int) -> int:
            slot_id = reload_slots_by_node.get(node_idx, 0)
            reload_slots_by_node[node_idx] = slot_id + 1
            return slot_id

        def free_tensor(tensor_name: str) -> None:
            slot = resident.pop(tensor_name, None)
            if slot is None:
                return

            self._free_fast_region(free_regions, slot.offset, slot.size)

        def try_allocate_fast_region(size: int) -> Optional[int]:
            nonlocal next_fast_offset, peak_fast_end

            offset = self._allocate_from_free_list(free_regions, size, self.DEFAULT_ALIGNMENT)
            if offset is not None:
                peak_fast_end = max(peak_fast_end, offset + size)
                return offset

            offset = align_offset(next_fast_offset)
            if not self._fits_fast_memory_budget(offset + size, capacity):
                return None

            padding = offset - next_fast_offset
            if padding > 0:
                # Preserve bump-path alignment holes for later reuse.
                self._free_fast_region(free_regions, next_fast_offset, padding)

            next_fast_offset = offset + size
            peak_fast_end = max(peak_fast_end, next_fast_offset)
            return offset

        def evict_tensor(tensor_name: str, node_idx: int, spill_after_idx: int) -> None:
            slot = resident.get(tensor_name)
            if slot is None:
                return

            liveness = liveness_map[tensor_name]
            next_use = liveness.next_use_after(node_idx)
            if next_use is None and not liveness.is_output:
                free_tensor(tensor_name)
                return
            self._validate_spill_after_idx(tensor_name, nodes[node_idx].name, spill_after_idx)

            slow_offset = assign_slow_offset(tensor_name)
            spill_points.append(
                SpillPoint(
                    tensor_name=tensor_name,
                    after_node=nodes[spill_after_idx].name,
                    after_node_idx=spill_after_idx,
                    from_buffer_id=0,
                    from_fast_offset=slot.offset,
                    to_slow_offset=slow_offset,
                    size=slot.size,
                )
            )
            self._record_spill(slot.size)
            spill_after[tensor_name] = spill_after_idx
            free_tensor(tensor_name)

        def make_space(
            size: int,
            node_idx: int,
            protected: set[str],
            spill_after_idx: int,
        ) -> int:
            offset = try_allocate_fast_region(size)
            if offset is not None:
                return offset

            while True:
                candidates = []
                for tensor_name, slot in resident.items():
                    if tensor_name in protected:
                        continue

                    liveness = liveness_map[tensor_name]
                    next_use = liveness.next_use_after(node_idx)
                    next_use_distance = None if next_use is None else next_use - node_idx
                    candidates.append(
                        {
                            "tensor_name": tensor_name,
                            "offset": slot.offset,
                            "size": slot.size,
                            "next_use_distance": next_use_distance,
                            "transfer_cost": self._estimated_transfer_cost(
                                size=slot.size,
                                next_use_distance=next_use_distance,
                                is_output=liveness.is_output,
                            ),
                        }
                    )

                if not candidates:
                    if not can_extend_fast_pool(size):
                        raise ValueError(
                            f"Cannot allocate {size} bytes at node {nodes[node_idx].name}: "
                            f"aligned fast-memory pool would exceed max_memory ({capacity})"
                        )
                    raise ValueError(
                        f"Cannot allocate {size} bytes at node {nodes[node_idx].name} "
                        "without evicting required tensors"
                    )

                eviction_plan = self._select_eviction_candidates(
                    size=size,
                    candidates=candidates,
                    free_regions=free_regions,
                    next_fast_offset=next_fast_offset,
                    max_memory=capacity,
                )
                if eviction_plan is None:
                    if not can_extend_fast_pool(size):
                        raise ValueError(
                            f"Cannot allocate {size} bytes at node {nodes[node_idx].name}: "
                            f"aligned fast-memory pool would exceed max_memory ({capacity})"
                        )
                    raise ValueError(
                        f"Cannot allocate {size} bytes at node {nodes[node_idx].name} "
                        "without evicting required tensors"
                    )

                for candidate in eviction_plan:
                    evict_tensor(candidate["tensor_name"], node_idx, spill_after_idx)

                offset = try_allocate_fast_region(size)
                if offset is not None:
                    return offset

                raise ValueError(
                    f"Cannot allocate {size} bytes at node {nodes[node_idx].name} "
                    "after applying eviction plan"
                )

        def allocate_or_reload(
            tensor_name: str,
            node_idx: int,
            protected: set[str],
            spill_after_idx: int,
        ) -> None:
            if tensor_name in resident:
                return

            size = tensor_sizes[tensor_name]
            self._validate_tensor_fits(size, capacity)
            offset = make_space(size, node_idx, protected, spill_after_idx)

            if tensor_name in slow_offsets:
                reload_points.append(
                    ReloadPoint(
                        tensor_name=tensor_name,
                        before_node=nodes[node_idx].name,
                        before_node_idx=node_idx,
                        from_slow_offset=slow_offsets[tensor_name],
                        to_buffer_id=0,
                        to_fast_offset=offset,
                        size=size,
                        reload_slot_id=next_reload_slot(node_idx),
                    )
                )
                self._record_reload(size)
                reload_before.setdefault(tensor_name, []).append(node_idx)

            resident[tensor_name] = _ResidentTensor(tensor_name, offset, size)
            record_allocation(tensor_name, offset)
            update_peak_fast_footprint()

        def take_inplace_input_slot(
            node: Node,
            node_idx: int,
            output_name: str,
        ) -> Optional[int]:
            if node.op_type not in self.INPLACE_REUSE_OPS:
                return None

            output_size = tensor_sizes[output_name]
            for input_name in node.inputs:
                slot = resident.get(input_name)
                if slot is None:
                    continue

                liveness = liveness_map[input_name]
                if liveness.live_end > node_idx or liveness.is_output:
                    continue
                if slot.size != output_size:
                    continue

                resident.pop(input_name)
                resident[output_name] = _ResidentTensor(output_name, slot.offset, output_size)
                return slot.offset

            return None

        for node_idx, node in enumerate(nodes):
            current_inputs = {
                tensor_name
                for tensor_name in node.inputs
                if tensor_name in tensor_sizes
            }

            for tensor_name in node.inputs:
                if tensor_name not in tensor_sizes:
                    continue

                allocate_or_reload(
                    tensor_name,
                    node_idx,
                    current_inputs,
                    max(node_idx - 1, 0),
                )

            protected = set(current_inputs)
            for tensor_name in node.outputs:
                if tensor_name not in tensor_sizes:
                    continue

                size = tensor_sizes[tensor_name]
                self._validate_tensor_fits(size, capacity)
                offset = take_inplace_input_slot(node, node_idx, tensor_name)
                if offset is None:
                    offset = make_space(size, node_idx, protected, node_idx - 1)
                resident[tensor_name] = _ResidentTensor(tensor_name, offset, size)
                record_allocation(tensor_name, offset)
                protected.add(tensor_name)
                update_peak_fast_footprint()

            for tensor_name in list(resident):
                if (
                    liveness_map[tensor_name].live_end <= node_idx
                    and not liveness_map[tensor_name].is_output
                ):
                    free_tensor(tensor_name)

            node_memory_usage.append(current_fast_footprint())

        for tensor_name, alloc in list(tensor_allocations.items()):
            spill_after_idx = spill_after.get(tensor_name)
            has_reload = tensor_name in reload_before
            is_output = liveness_map[tensor_name].is_output
            is_resident = tensor_name in resident
            is_slow_backed = tensor_name in slow_offsets and (
                not has_reload or (is_output and not is_resident)
            )

            if is_slow_backed:
                buffer_id = -1
                offset = slow_offsets[tensor_name]
            elif has_reload:
                buffer_id = alloc.buffer_id
                offset = alloc.offset
            elif is_resident:
                buffer_id = 0
                offset = resident[tensor_name].offset
            else:
                buffer_id = alloc.buffer_id
                offset = alloc.offset

            tensor_allocations[tensor_name] = TensorAllocation(
                tensor_name=tensor_name,
                buffer_id=buffer_id,
                offset=offset,
                size=alloc.size,
                is_spilled=is_slow_backed,
                spill_after=spill_after_idx,
                reload_before=reload_before.get(tensor_name),
            )

        buffer_size = 0
        if tensor_allocations:
            buffer_size = max(align_offset(peak_fast_end), self.DEFAULT_ALIGNMENT)
            if not self._fits_fast_memory_budget(peak_fast_end, capacity):
                raise ValueError(
                    f"Cannot finalize plan: aligned fast-memory pool size {buffer_size} "
                    f"exceeds max_memory ({capacity})"
                )

        buffers = []
        if tensor_allocations:
            buffers.append(
                MemoryBuffer(
                    id=0,
                    offset=0,
                    size=buffer_size,
                    alignment=self.DEFAULT_ALIGNMENT,
                    tensors=[
                        tensor_name
                        for tensor_name, alloc in tensor_allocations.items()
                        if alloc.buffer_id >= 0
                    ],
                )
            )

        tensor_to_buffer = {
            tensor_name: alloc.buffer_id
            for tensor_name, alloc in tensor_allocations.items()
        }

        return MemoryAllocationPlan(
            strategy_name=self.name,
            total_fast_memory=buffer_size,
            total_slow_memory=next_slow_offset,
            peak_memory=peak_fast_footprint,
            num_buffers=len(buffers),
            buffers=buffers,
            tensor_allocations=tensor_allocations,
            tensor_to_buffer=tensor_to_buffer,
            spill_points=spill_points,
            reload_points=reload_points,
            spill_bytes=self._spill_bytes,
            reload_bytes=self._reload_bytes,
            total_transfer_bytes=self._spill_bytes + self._reload_bytes,
            node_memory_usage=node_memory_usage,
        )

    def _align(self, offset: int, alignment: int) -> int:
        """Align an offset to the requested boundary."""
        return ((offset + alignment - 1) // alignment) * alignment

    def _aligned_pool_size(self, fast_end: int) -> int:
        """Return the aligned fast-pool size needed to cover `fast_end`."""
        if fast_end <= 0:
            return 0
        return max(self._align(fast_end, self.DEFAULT_ALIGNMENT), self.DEFAULT_ALIGNMENT)

    def _fits_fast_memory_budget(self, fast_end: int, max_memory: int | float) -> bool:
        """Check whether the aligned fast-pool size remains within budget."""
        if max_memory == float("inf"):
            return True
        return self._aligned_pool_size(fast_end) <= max_memory

    def _validate_spill_after_idx(
        self,
        tensor_name: str,
        node_name: str,
        spill_after_idx: int,
    ) -> None:
        """Reject spills that would need to happen before the first graph node."""
        if spill_after_idx < 0:
            raise ValueError(
                f"Cannot schedule spill of {tensor_name} early enough before node {node_name}"
            )

    def _allocate_from_free_list(
        self,
        free_regions: list[tuple[int, int]],
        size: int,
        alignment: int,
    ) -> Optional[int]:
        """Allocate the first aligned region large enough to satisfy `size`."""
        for idx, (start, region_size) in enumerate(list(free_regions)):
            aligned_start = self._align(start, alignment)
            padding = aligned_start - start
            if padding + size > region_size:
                continue

            region_end = start + region_size
            alloc_end = aligned_start + size
            replacement: list[tuple[int, int]] = []
            if padding > 0:
                replacement.append((start, padding))
            if alloc_end < region_end:
                replacement.append((alloc_end, region_end - alloc_end))

            free_regions[idx:idx + 1] = replacement
            return aligned_start

        return None

    def _free_fast_region(
        self,
        free_regions: list[tuple[int, int]],
        offset: int,
        size: int,
    ) -> None:
        """Return a fast-memory region to the free list and merge neighbors."""
        if size <= 0:
            return

        free_regions.append((offset, size))
        free_regions.sort()

        merged: list[tuple[int, int]] = []
        for region_start, region_size in free_regions:
            if not merged:
                merged.append((region_start, region_size))
                continue

            last_start, last_size = merged[-1]
            last_end = last_start + last_size
            region_end = region_start + region_size
            if region_start <= last_end:
                merged[-1] = (last_start, max(last_end, region_end) - last_start)
                continue

            merged.append((region_start, region_size))

        free_regions[:] = merged

    def _evict_score(self, *, next_use_distance: Optional[int], size: int) -> float:
        """Higher score means the tensor is a better eviction candidate."""
        if next_use_distance is None:
            return float("inf")
        return next_use_distance / max(size, 1)

    def _estimated_transfer_cost(
        self,
        *,
        size: int,
        next_use_distance: Optional[int],
        is_output: bool,
    ) -> int:
        """Estimate total transfer bytes caused by evicting a tensor now."""
        if next_use_distance is None:
            return size if is_output else 0
        return size * 2

    def _sort_eviction_candidates(
        self,
        candidates: list[dict[str, int | str | None]],
    ) -> list[dict[str, int | str | None]]:
        """Sort tensors from most-evictable to least-evictable."""
        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.get("transfer_cost", 0),  # type: ignore[arg-type]
                -self._evict_score(
                    next_use_distance=candidate["next_use_distance"],  # type: ignore[arg-type]
                    size=candidate["size"],  # type: ignore[arg-type]
                ),
                -(
                    float("inf")
                    if candidate["next_use_distance"] is None
                    else candidate["next_use_distance"]
                ),
                candidate["size"],  # type: ignore[index]
            ),
        )

    def _can_allocate_with_state(
        self,
        *,
        size: int,
        free_regions: list[tuple[int, int]],
        next_fast_offset: int,
        max_memory: int | float,
    ) -> bool:
        """Check whether the current free-list/bump state can satisfy an allocation."""
        simulated_free_regions = list(free_regions)
        offset = self._allocate_from_free_list(
            simulated_free_regions,
            size,
            self.DEFAULT_ALIGNMENT,
        )
        if offset is not None:
            return True

        offset = self._align(next_fast_offset, self.DEFAULT_ALIGNMENT)
        return self._fits_fast_memory_budget(offset + size, max_memory)

    def _select_eviction_candidates(
        self,
        *,
        size: int,
        candidates: list[dict[str, int | str | None]],
        free_regions: list[tuple[int, int]],
        next_fast_offset: int,
        max_memory: int | float,
    ) -> list[dict[str, int | str | None]] | None:
        """Find the minimum-transfer eviction set that makes the allocation fit."""
        ordered_candidates = self._sort_eviction_candidates(candidates)
        allocation_cache: dict[tuple[int, ...], bool] = {}
        heap: list[tuple[int, int, tuple[int, ...], int]] = [(0, 0, tuple(), 0)]

        def can_allocate_after(indices: tuple[int, ...]) -> bool:
            if indices in allocation_cache:
                return allocation_cache[indices]

            simulated_free_regions = list(free_regions)
            for candidate_idx in indices:
                candidate = ordered_candidates[candidate_idx]
                self._free_fast_region(
                    simulated_free_regions,
                    int(candidate["offset"]),
                    int(candidate["size"]),
                )

            fits = self._can_allocate_with_state(
                size=size,
                free_regions=simulated_free_regions,
                next_fast_offset=next_fast_offset,
                max_memory=max_memory,
            )
            allocation_cache[indices] = fits
            return fits

        if len(ordered_candidates) > self.MAX_EXACT_EVICTION_CANDIDATES:
            for candidate_idx in range(len(ordered_candidates)):
                if can_allocate_after((candidate_idx,)):
                    return [ordered_candidates[candidate_idx]]

            for left_idx in range(len(ordered_candidates)):
                for right_idx in range(left_idx + 1, len(ordered_candidates)):
                    indices = (left_idx, right_idx)
                    if can_allocate_after(indices):
                        return [ordered_candidates[idx] for idx in indices]

            for first_idx in range(len(ordered_candidates)):
                for second_idx in range(first_idx + 1, len(ordered_candidates)):
                    for third_idx in range(second_idx + 1, len(ordered_candidates)):
                        indices = (first_idx, second_idx, third_idx)
                        if can_allocate_after(indices):
                            return [ordered_candidates[idx] for idx in indices]

            greedy_indices: tuple[int, ...] = tuple()
            for candidate_idx in range(len(ordered_candidates)):
                greedy_indices += (candidate_idx,)
                if can_allocate_after(greedy_indices):
                    return [ordered_candidates[idx] for idx in greedy_indices]

            return None

        while heap:
            total_transfer_cost, selected_count, selected_indices, start_idx = heapq.heappop(
                heap
            )
            if can_allocate_after(selected_indices):
                return [ordered_candidates[idx] for idx in selected_indices]

            for candidate_idx in range(start_idx, len(ordered_candidates)):
                candidate = ordered_candidates[candidate_idx]
                new_indices = selected_indices + (candidate_idx,)
                heapq.heappush(
                    heap,
                    (
                        total_transfer_cost + int(candidate.get("transfer_cost", 0)),
                        selected_count + 1,
                        new_indices,
                        candidate_idx + 1,
                    ),
                )

        return None

    def _record_spill(self, size: int) -> None:
        """Accumulate bytes transferred from fast to slow memory."""
        self._spill_bytes += size

    def _record_reload(self, size: int) -> None:
        """Accumulate bytes transferred from slow to fast memory."""
        self._reload_bytes += size
