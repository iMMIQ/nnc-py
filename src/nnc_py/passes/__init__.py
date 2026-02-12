"""Optimization passes module."""

from nnc_py.passes.base import PassBase, PassManager
# New memory strategy interface
from nnc_py.passes.memory_strategy import (
    AllocationStrategy,
    MemoryAllocationPlan,
    MemoryAllocationStrategy,
    StrategyRegistry,
    SpillPoint,
    ReloadPoint,
    TensorAllocation,
    get_allocation_plan,
    get_memory_strategy,
)

# New unified memory planning pass
from nnc_py.passes.memory_planning import MemoryPlanningPassV2, get_memory_allocation_plan

# Optimization passes
from nnc_py.passes.dead_code_elimination import DeadCodeEliminationPass
from nnc_py.passes.identity_elimination import IdentityEliminationPass
from nnc_py.passes.pattern_fusion import PatternFusionPass

__all__ = [
    # Core pass infrastructure
    "PassBase",
    "PassManager",

    # New memory strategy interface
    "AllocationStrategy",
    "MemoryAllocationPlan",
    "MemoryAllocationStrategy",
    "StrategyRegistry",
    "SpillPoint",
    "ReloadPoint",
    "TensorAllocation",
    "get_allocation_plan",
    "get_memory_strategy",

    # New unified pass
    "MemoryPlanningPassV2",
    "get_memory_allocation_plan",

    # Optimization passes
    "DeadCodeEliminationPass",
    "IdentityEliminationPass",
    "PatternFusionPass",
]
