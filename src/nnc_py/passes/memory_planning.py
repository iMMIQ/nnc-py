"""Unified memory planning pass with pluggable strategies.

This pass provides a single entry point for memory allocation with
runtime-selectable strategies.
"""

from typing import TYPE_CHECKING, Dict, Optional

from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassBase
from nnc_py.passes.liveness import TensorLiveness
from nnc_py.passes.memory_strategy import (
    AllocationStrategy,
    MemoryAllocationPlan,
    MemoryAllocationStrategy,
    StrategyRegistry,
    get_default_allocation_strategy,
    _register_default_strategies,
)

if TYPE_CHECKING:
    from typing import Any


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

        liveness_map: Dict[str, TensorLiveness] = ctx.metadata["tensor_liveness"]

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
            print(f"Max memory: unlimited")
        else:
            print(f"Max memory: {max_memory} bytes ({max_memory / 1024:.2f} KB)")
        print(f"{'='*60}")

    def _log_summary(self, ctx: CompileContext, plan: MemoryAllocationPlan) -> None:
        """Log allocation summary."""
        print(f"\nMemory Allocation Plan Summary:")
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


# Register default strategies when this module is imported
def _initialize_strategies() -> None:
    """Import and register default strategies."""
    _register_default_strategies()


# Auto-initialize on import
_initialize_strategies()


def get_memory_allocation_plan(ctx: CompileContext) -> Optional[MemoryAllocationPlan]:
    """Get the memory allocation plan from context.

    Args:
        ctx: Compilation context

    Returns:
        MemoryAllocationPlan if available, None otherwise
    """
    return ctx.metadata.get("memory_allocation_plan")
