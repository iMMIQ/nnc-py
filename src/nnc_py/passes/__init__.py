"""Optimization passes module."""

from nnc_py.passes.indexed_forward_graph import Edge, IndexedForwardGraph, NodeEntry
from nnc_py.passes.dominator_tree import DominatorTree
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
from nnc_py.passes.fusion_groups import FusionGroup, GroupArena

__all__ = [
    # Indexed forward graph
    "Edge",
    "IndexedForwardGraph",
    "NodeEntry",

    # Post-dominator tree
    "DominatorTree",

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

    # Union-Find fusion groups
    "FusionGroup",
    "GroupArena",
]
