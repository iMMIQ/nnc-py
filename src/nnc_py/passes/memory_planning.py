"""Unified memory planning pass with pluggable strategies.

This pass provides a single entry point for memory allocation with
runtime-selectable strategies.
"""

from typing import TYPE_CHECKING

from nnc_py.ir.context import CompileContext
from nnc_py.ir.execution_plan import MemoryRegionKind, get_node_execution_plans
from nnc_py.ir.node import OpType
from nnc_py.passes.base import PassBase
from nnc_py.passes.liveness import TensorLiveness
from nnc_py.passes.memory_plan import MemoryBuffer
from nnc_py.passes.memory_strategy import (
    AllocationStrategy,
    LogicalMemoryRegion,
    MemoryAllocationPlan,
    MemoryAllocationStrategy,
    StrategyRegistry,
    _register_default_strategies,
    get_default_allocation_strategy,
)

if TYPE_CHECKING:
    from typing import Any


_DEFAULT_ALIGNMENT = 16
_ADVANCED_PIPELINE_FLAGS = (
    "advanced_pipeline",
    "advanced_pipeline_enabled",
    "enable_advanced_pipeline",
)
_NODE_REGION_SIZE_HINTS_METADATA_KEY = "node_execution_plan_region_sizes"
_FAST_TILE_REGION_KINDS = {
    MemoryRegionKind.TILE,
    MemoryRegionKind.SCRATCH,
    MemoryRegionKind.PACK,
    MemoryRegionKind.STAGE,
}
_FAST_TENSOR_REGION_KINDS = {
    MemoryRegionKind.TILE,
    MemoryRegionKind.PACK,
    MemoryRegionKind.STAGE,
}
_TILE_COMPATIBLE_GROUP_OPS = {
    OpType.ADD,
    OpType.RELU,
    OpType.FUSED_ADD_RELU,
}


class MemoryPlanningPassV2(PassBase):
    """Unified memory planning pass with pluggable strategies.

    This pass replaces the original MemoryPlanningPass and provides
    a single entry point for memory allocation using the basic
    sequential allocation algorithm.

    Usage:
        ctx.metadata["memory_strategy"] = "basic"
        pass = MemoryPlanningPassV2()
        pass.run(ctx)
        plan = ctx.metadata["memory_allocation_plan"]  # MemoryAllocationPlan
    """

    @property
    def name(self) -> str:
        return "MemoryPlanningV2"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute memory planning with the selected strategy."""
        # Ensure liveness analysis has been run
        if "tensor_liveness" not in ctx.metadata:
            raise RuntimeError(
                "LivenessAnalysisPass must be run before MemoryPlanningPassV2"
            )

        liveness_map: dict[str, TensorLiveness] = ctx.metadata["tensor_liveness"]

        # Get strategy from metadata or use default
        strategy_config: str | AllocationStrategy | None = ctx.metadata.get(
            "memory_strategy"
        )
        max_memory_raw: float | int | None = ctx.metadata.get("max_memory", float("inf"))

        # Get strategy instance
        strategy = self._get_strategy(strategy_config, ctx.optimization_level)

        # Convert float infinity to None for the allocate call
        max_memory_for_alloc: int | None
        if max_memory_raw == float("inf") or max_memory_raw is None:
            max_memory_for_alloc = None
        else:
            max_memory_for_alloc = int(max_memory_raw)

        if ctx.debug:
            self._log_start(strategy, max_memory_raw)

        # Run allocation
        plan = strategy.allocate(ctx, liveness_map, max_memory_for_alloc)

        # Store result
        ctx.metadata["memory_allocation_plan"] = plan

        # For backward compatibility, also store as legacy formats
        self._store_legacy_formats(ctx, plan)

        if ctx.debug:
            self._log_summary(ctx, plan)

    def _get_strategy(
        self,
        strategy_config: str | AllocationStrategy | None,
        optimization_level: int,
    ) -> MemoryAllocationStrategy:
        """Get strategy instance from config."""
        if strategy_config is None:
            strategy_config = get_default_allocation_strategy(optimization_level)

        if isinstance(strategy_config, MemoryAllocationStrategy):
            return strategy_config

        if not isinstance(strategy_config, (str, AllocationStrategy)):
            raise ValueError(f"Invalid strategy configuration: {strategy_config}")

        return StrategyRegistry.get(strategy_config)

    def _store_legacy_formats(
        self,
        ctx: CompileContext,
        plan: MemoryAllocationPlan,
    ) -> None:
        """Store results in legacy formats for backward compatibility."""
        # Create MemoryPlan-like structure for old code
        from nnc_py.passes.memory_plan import MemoryPlan, TensorMemoryInfo

        legacy_tensor_info = {}
        for tensor_name, alloc in plan.tensor_allocations.items():
            # Spilled tensors have buffer_id=-1, skip them for fast memory pool
            if alloc.buffer_id < 0:
                continue

            buffer = plan.buffers[alloc.buffer_id] if 0 <= alloc.buffer_id < len(plan.buffers) else None
            if buffer:
                # pool_offset is the absolute offset from memory pool start
                # = buffer.offset (where this buffer starts)
                # Note: alloc.offset is 0 for non-spilled tensors (they share the buffer)
                legacy_tensor_info[tensor_name] = TensorMemoryInfo(
                    tensor_name=tensor_name,
                    buffer_id=alloc.buffer_id,
                    offset=alloc.offset,
                    size=alloc.size,
                    pool_offset=buffer.offset,
                )

        # Calculate savings (approximate)
        total_without_sharing = sum(info.size for info in legacy_tensor_info.values())
        savings = 0.0
        if total_without_sharing > 0:
            savings = (1 - plan.total_fast_memory / total_without_sharing) * 100

        legacy_plan = MemoryPlan(
            buffers=plan.buffers,
            tensor_info=legacy_tensor_info,
            total_size=plan.total_fast_memory,
            alignment=16,
            num_tensors=len(legacy_tensor_info),
            num_buffers=plan.num_buffers,
            savings_without_sharing=savings,
        )

        ctx.metadata["memory_plan"] = legacy_plan

    def _log_start(
        self, strategy: MemoryAllocationStrategy, max_memory: float | int | None
    ) -> None:
        """Log start of memory planning."""
        print(f"\n{'='*60}")
        print(f"Memory Planning with strategy: {strategy.name}")
        if max_memory == float("inf") or max_memory is None:
            print("Max memory: unlimited")
        else:
            print(f"Max memory: {max_memory} bytes ({max_memory / 1024:.2f} KB)")
        print(f"{'='*60}")

    def _log_summary(self, ctx: CompileContext, plan: MemoryAllocationPlan) -> None:
        """Log allocation summary."""
        print("\nMemory Allocation Plan Summary:")
        print(f"  Strategy: {plan.strategy_name}")
        print(f"  Buffers: {plan.num_buffers}")
        print(f"  Total fast memory: {plan.total_fast_memory} bytes ({plan.total_fast_memory / 1024:.2f} KB)")
        print(f"  Peak memory: {plan.peak_memory} bytes ({plan.peak_memory / 1024:.2f} KB)")
        if plan.total_slow_memory > 0:
            print(f"  Slow memory: {plan.total_slow_memory} bytes ({plan.total_slow_memory / 1024:.2f} KB)")
        print(f"  Tensors: {len(plan.tensor_allocations)}")
        if plan.has_spill:
            print(f"  Spill points: {plan.spill_count}")
            print(f"  Reload points: {plan.reload_count}")


class MemoryPlanningPassV3(MemoryPlanningPassV2):
    """Tile-aware memory planning with V2 fallback for legacy paths."""

    @property
    def name(self) -> str:
        return "MemoryPlanningV3"

    def _execute(self, ctx: CompileContext) -> None:
        execution_plans = get_node_execution_plans(ctx)
        if _should_use_tile_aware_v3(ctx, execution_plans):
            plan = allocate_tile_regions(ctx)
            max_memory = _resolve_max_memory_budget(ctx)
            if max_memory is None or plan.total_fast_memory <= max_memory:
                ctx.metadata["memory_allocation_plan"] = plan
                ctx.metadata.pop("memory_plan", None)
                if max_memory is not None:
                    # The V3 tile-aware plan already satisfies the user's fast-memory
                    # budget, so downstream legacy spill logic must not re-interpret the
                    # graph with full-tensor sizing assumptions.
                    ctx.metadata["memory_budget_satisfied_by_v3"] = max_memory
                    ctx.metadata.pop("max_memory", None)
                if ctx.debug:
                    self._log_summary(ctx, plan)
                return

        if _advanced_pipeline_enabled(ctx) and not execution_plans:
            raise RuntimeError(
                "MemoryPlanningPassV3 requires node_execution_plans when the advanced pipeline is enabled"
            )

        super()._execute(ctx)


def allocate_tile_regions(ctx: CompileContext) -> MemoryAllocationPlan:
    """Build a tile-aware plan from typed execution plans and size hints."""

    execution_plans = get_node_execution_plans(ctx)
    region_hints_by_node = ctx.metadata.get(_NODE_REGION_SIZE_HINTS_METADATA_KEY, {})
    region_peaks: dict[str, int] = {}
    region_presence: set[str] = set()
    per_node_region_sizes: list[dict[str, int]] = []
    pool_backed_region_names = _collect_pool_backed_region_names(execution_plans)

    for node_name, plan in execution_plans.items():
        node_region_sizes = _collect_node_region_sizes(
            ctx,
            plan,
            region_hints_by_node.get(node_name, {}),
        )
        per_node_region_sizes.append(node_region_sizes)
        for region_name, size_bytes in node_region_sizes.items():
            region_presence.add(region_name)
            region_peaks[region_name] = max(region_peaks.get(region_name, 0), size_bytes)

    logical_regions = _build_logical_regions(region_presence, region_peaks)
    buffers, logical_regions, total_fast_memory, peak_memory = _build_region_buffers(
        logical_regions,
        per_node_region_sizes,
        pool_backed_region_names,
    )

    return MemoryAllocationPlan(
        strategy_name="tile_regions_v3",
        total_fast_memory=total_fast_memory,
        peak_memory=peak_memory,
        num_buffers=len(buffers),
        buffers=buffers,
        logical_regions=logical_regions,
    )


def _advanced_pipeline_enabled(ctx: CompileContext) -> bool:
    return any(bool(ctx.metadata.get(flag)) for flag in _ADVANCED_PIPELINE_FLAGS)


def _should_use_tile_aware_v3(
    ctx: CompileContext,
    execution_plans: dict[str, "Any"],
) -> bool:
    if not execution_plans:
        return False
    if not _has_full_execution_plan_coverage(ctx, execution_plans):
        return False
    return _has_complete_tile_region_metadata(ctx, execution_plans)


def _resolve_max_memory_budget(ctx: CompileContext) -> int | None:
    max_memory_raw = ctx.metadata.get("max_memory")
    if max_memory_raw is None or max_memory_raw == float("inf"):
        return None
    return int(max_memory_raw)


def _collect_pool_backed_region_names(
    execution_plans: dict[str, "Any"],
) -> set[str]:
    pool_backed_region_names: set[str] = set()
    for plan in execution_plans.values():
        for access in (*plan.input_accesses, *plan.output_accesses):
            if access.memory_region in _FAST_TENSOR_REGION_KINDS:
                pool_backed_region_names.add(access.memory_region.value)
    return pool_backed_region_names


def _has_full_execution_plan_coverage(
    ctx: CompileContext,
    execution_plans: dict[str, "Any"],
) -> bool:
    required_node_names = {
        node.name for node in ctx.graph.topological_sort() if node.is_computational()
    }
    covered_node_names = _covered_execution_group_node_names(ctx, execution_plans)
    return required_node_names.issubset(covered_node_names)


def _covered_execution_group_node_names(
    ctx: CompileContext,
    execution_plans: dict[str, "Any"],
) -> set[str]:
    covered_node_names = set(execution_plans)

    for node in ctx.graph.topological_sort():
        if node.name not in execution_plans:
            continue
        if len(node.outputs) != 1:
            continue

        flow_tensor_name = node.outputs[0]
        while True:
            consumers = [
                consumer
                for consumer in ctx.graph.get_consumers(flow_tensor_name)
                if consumer.is_computational()
            ]
            if len(consumers) != 1:
                break

            consumer = consumers[0]
            if consumer.name in covered_node_names or consumer.name in execution_plans:
                break
            if not _is_tile_compatible_group_successor(ctx, flow_tensor_name, consumer):
                break

            covered_node_names.add(consumer.name)
            if len(consumer.outputs) != 1:
                break
            flow_tensor_name = consumer.outputs[0]

    return covered_node_names


def _is_tile_compatible_group_successor(
    ctx: CompileContext,
    flow_tensor_name: str,
    node: "Any",
) -> bool:
    if node.op_type not in _TILE_COMPATIBLE_GROUP_OPS:
        return False
    if len(node.outputs) != 1:
        return False

    flow_tensor = ctx.graph.tensors.get(flow_tensor_name)
    output_tensor = ctx.graph.tensors.get(node.outputs[0])
    if flow_tensor is None or output_tensor is None:
        return False
    if flow_tensor.dtype != output_tensor.dtype:
        return False
    if flow_tensor.byte_size() != output_tensor.byte_size():
        return False

    if node.op_type == OpType.RELU:
        return tuple(node.inputs) == (flow_tensor_name,)

    if node.op_type in {OpType.ADD, OpType.FUSED_ADD_RELU}:
        if len(node.inputs) != 2 or flow_tensor_name not in node.inputs:
            return False
        other_input_name = next(
            input_name for input_name in node.inputs if input_name != flow_tensor_name
        )
        other_input = ctx.graph.tensors.get(other_input_name)
        if other_input is None:
            return False
        if other_input.dtype != flow_tensor.dtype:
            return False
        return other_input.byte_size() == flow_tensor.byte_size()

    return False


def _has_complete_tile_region_metadata(
    ctx: CompileContext,
    execution_plans: dict[str, "Any"],
) -> bool:
    region_hints_by_node = ctx.metadata.get(_NODE_REGION_SIZE_HINTS_METADATA_KEY)
    if not isinstance(region_hints_by_node, dict):
        return False

    for node_name, plan in execution_plans.items():
        node_hints = region_hints_by_node.get(node_name)
        if not isinstance(node_hints, dict):
            return False
        if not _node_plan_has_required_region_hints(plan, node_hints):
            return False

    return True


def _node_plan_has_required_region_hints(plan: "Any", node_hints: dict[str, object]) -> bool:
    tensor_hints = node_hints.get("tensor_bytes", {})
    region_hints = node_hints.get("region_bytes", {})
    if not isinstance(tensor_hints, dict) or not isinstance(region_hints, dict):
        return False

    for access in (*plan.input_accesses, *plan.output_accesses):
        if access.memory_region in _FAST_TENSOR_REGION_KINDS:
            size_bytes = tensor_hints.get(access.tensor_name)
            if size_bytes is None or int(size_bytes) <= 0:
                return False

    for region in plan.memory_regions:
        if region in {MemoryRegionKind.SCRATCH, MemoryRegionKind.PACK, MemoryRegionKind.STAGE}:
            if region.value not in region_hints:
                return False

    return True


def _collect_node_region_sizes(
    ctx: CompileContext,
    plan: "Any",
    size_hints: object,
) -> dict[str, int]:
    tensor_hints: dict[str, int] = {}
    region_hints: dict[str, int] = {}
    if isinstance(size_hints, dict):
        raw_tensor_hints = size_hints.get("tensor_bytes", {})
        raw_region_hints = size_hints.get("region_bytes", {})
        if isinstance(raw_tensor_hints, dict):
            tensor_hints = {str(name): int(size) for name, size in raw_tensor_hints.items()}
        if isinstance(raw_region_hints, dict):
            region_hints = {str(name): int(size) for name, size in raw_region_hints.items()}

    node_region_sizes = {
        region.value: 0 for region in plan.memory_regions if region in _FAST_TILE_REGION_KINDS
    }
    for access in (*plan.input_accesses, *plan.output_accesses):
        if access.memory_region not in _FAST_TENSOR_REGION_KINDS:
            continue
        region_name = access.memory_region.value
        node_region_sizes.setdefault(region_name, 0)
        node_region_sizes[region_name] += _resolve_access_size_bytes(
            ctx,
            access.tensor_name,
            tensor_hints,
        )

    for region_name, size_bytes in region_hints.items():
        node_region_sizes[region_name] = max(node_region_sizes.get(region_name, 0), size_bytes)

    return node_region_sizes


def _resolve_access_size_bytes(
    ctx: CompileContext,
    tensor_name: str,
    tensor_hints: dict[str, int],
) -> int:
    hint_size = tensor_hints.get(tensor_name)
    if hint_size is not None:
        return int(hint_size)

    tensor = ctx.graph.tensors.get(tensor_name)
    if tensor is None:
        return 0

    tensor_size = tensor.byte_size()
    if tensor_size < 0:
        return 0
    return int(tensor_size)


def _build_logical_regions(
    region_presence: set[str],
    region_peaks: dict[str, int],
) -> dict[str, LogicalMemoryRegion]:
    ordered_region_names = [region.value for region in MemoryRegionKind]
    logical_regions: dict[str, LogicalMemoryRegion] = {}

    for region_name in ordered_region_names:
        if region_name not in region_presence:
            continue
        logical_regions[region_name] = LogicalMemoryRegion(
            name=region_name,
            size_bytes=region_peaks.get(region_name, 0),
        )

    return logical_regions


def _build_region_buffers(
    logical_regions: dict[str, LogicalMemoryRegion],
    per_node_region_sizes: list[dict[str, int]] | None = None,
    pool_backed_region_names: set[str] | None = None,
) -> tuple[list[MemoryBuffer], dict[str, LogicalMemoryRegion], int, int]:
    if pool_backed_region_names is None:
        pool_backed_region_names = set(logical_regions)
    return _schedule_region_buffers(
        logical_regions,
        per_node_region_sizes or [],
        pool_backed_region_names,
    )


def _schedule_region_buffers(
    logical_regions: dict[str, LogicalMemoryRegion],
    per_node_region_sizes: list[dict[str, int]],
    pool_backed_region_names: set[str],
) -> tuple[list[MemoryBuffer], dict[str, LogicalMemoryRegion], int, int]:
    buffers: list[MemoryBuffer] = []
    regions_with_offsets: dict[str, LogicalMemoryRegion] = {}
    region_offsets: dict[str, int] = {}

    for region_name, region in logical_regions.items():
        if region_name not in pool_backed_region_names:
            continue
        aligned_size = _align(region.size_bytes, _DEFAULT_ALIGNMENT)
        aligned_offset = _find_region_offset(
            region_name,
            aligned_size,
            region_offsets,
            per_node_region_sizes,
        )
        region_offsets[region_name] = aligned_offset
        aligned_size = _align(region.size_bytes, _DEFAULT_ALIGNMENT)
        buffers.append(
            MemoryBuffer(
                id=len(buffers),
                offset=aligned_offset,
                size=aligned_size,
                alignment=_DEFAULT_ALIGNMENT,
                tensors=[region.name],
            )
        )
        regions_with_offsets[region.name] = LogicalMemoryRegion(
            name=region.name,
            size_bytes=region.size_bytes,
            offset=aligned_offset,
        )

    execution_peak = 0
    pool_backed_execution_peak = 0
    if per_node_region_sizes:
        for node_region_sizes in per_node_region_sizes:
            node_peak = sum(
                _align(size_bytes, _DEFAULT_ALIGNMENT)
                for size_bytes in node_region_sizes.values()
                if size_bytes > 0
            )
            pool_backed_node_peak = sum(
                _align(size_bytes, _DEFAULT_ALIGNMENT)
                for region_name, size_bytes in node_region_sizes.items()
                if region_name in pool_backed_region_names and size_bytes > 0
            )
            execution_peak = max(execution_peak, node_peak)
            pool_backed_execution_peak = max(
                pool_backed_execution_peak,
                pool_backed_node_peak,
            )
    else:
        for region_name, region in logical_regions.items():
            if region_name in region_offsets:
                region_end = region_offsets[region_name] + _align(
                    region.size_bytes,
                    _DEFAULT_ALIGNMENT,
                )
                execution_peak = max(execution_peak, region_end)
                if region_name in pool_backed_region_names:
                    pool_backed_execution_peak = max(
                        pool_backed_execution_peak,
                        region_end,
                    )
            else:
                execution_peak = max(
                    execution_peak,
                    _align(region.size_bytes, _DEFAULT_ALIGNMENT),
                )

    layout_upper_bound = max(
        (buffer.offset + buffer.size for buffer in buffers),
        default=0,
    )
    total_fast_memory = max(pool_backed_execution_peak, layout_upper_bound)

    for region_name, region in logical_regions.items():
        if region_name in regions_with_offsets:
            continue
        aligned_size = _align(region.size_bytes, _DEFAULT_ALIGNMENT)
        offset = max(0, total_fast_memory - aligned_size)
        regions_with_offsets[region_name] = LogicalMemoryRegion(
            name=region.name,
            size_bytes=region.size_bytes,
            offset=offset,
        )

    return buffers, regions_with_offsets, total_fast_memory, execution_peak


def _find_region_offset(
    region_name: str,
    aligned_region_size: int,
    placed_region_offsets: dict[str, int],
    per_node_region_sizes: list[dict[str, int]],
) -> int:
    candidate_offsets = {0}

    if per_node_region_sizes:
        for node_region_sizes in per_node_region_sizes:
            for placed_region_name, placed_offset in placed_region_offsets.items():
                placed_size = _align(
                    node_region_sizes.get(placed_region_name, 0),
                    _DEFAULT_ALIGNMENT,
                )
                if placed_size > 0:
                    candidate_offsets.add(placed_offset + placed_size)
    else:
        for placed_offset in placed_region_offsets.values():
            candidate_offsets.add(placed_offset)

    for candidate_offset in sorted(candidate_offsets):
        if _region_offset_is_valid(
            region_name,
            candidate_offset,
            aligned_region_size,
            placed_region_offsets,
            per_node_region_sizes,
        ):
            return candidate_offset

    if not placed_region_offsets:
        return 0
    return max(placed_region_offsets.values()) + aligned_region_size


def _region_offset_is_valid(
    region_name: str,
    candidate_offset: int,
    aligned_region_size: int,
    placed_region_offsets: dict[str, int],
    per_node_region_sizes: list[dict[str, int]],
) -> bool:
    if not per_node_region_sizes:
        return True

    for node_region_sizes in per_node_region_sizes:
        candidate_size = _align(node_region_sizes.get(region_name, 0), _DEFAULT_ALIGNMENT)
        if candidate_size <= 0:
            continue
        candidate_end = candidate_offset + candidate_size

        for placed_region_name, placed_offset in placed_region_offsets.items():
            placed_size = _align(
                node_region_sizes.get(placed_region_name, 0),
                _DEFAULT_ALIGNMENT,
            )
            if placed_size <= 0:
                continue
            placed_end = placed_offset + placed_size
            if candidate_offset < placed_end and placed_offset < candidate_end:
                return False

        if candidate_offset + aligned_region_size < candidate_end:
            return False

    return True


def _align(value: int, alignment: int) -> int:
    if value <= 0:
        return 0
    return ((value + alignment - 1) // alignment) * alignment


# Register default strategies when this module is imported
def _initialize_strategies() -> None:
    """Import and register default strategies."""
    _register_default_strategies()


# Auto-initialize on import
_initialize_strategies()


def get_memory_allocation_plan(ctx: CompileContext) -> MemoryAllocationPlan | None:
    """Get the memory allocation plan from context.

    Args:
        ctx: Compilation context

    Returns:
        MemoryAllocationPlan if available, None otherwise
    """
    return ctx.metadata.get("memory_allocation_plan")
