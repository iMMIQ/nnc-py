"""Placeholder cost-aware allocator.

This adapter reserves the strategy identity and registration for the future
cost-aware planner while delegating allocation behavior to the basic allocator.
"""

from dataclasses import replace
from typing import Dict, Optional

from nnc_py.ir.context import CompileContext
from nnc_py.passes.liveness import TensorLiveness
from nnc_py.passes.memory_strategy import (
    AllocationStrategy,
    MemoryAllocationPlan,
    MemoryAllocationStrategy,
)
from nnc_py.passes.strategies.basic_allocator import BasicAllocator


class CostAwareAllocator(MemoryAllocationStrategy):
    """Temporary adapter for the future cost-aware allocator."""

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
        plan = BasicAllocator().allocate(ctx, liveness_map, max_memory)
        return replace(plan, strategy_name=self.name)
