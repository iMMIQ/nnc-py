"""Optimization passes module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nnc_py.ir.execution_plan import get_node_execution_plan, get_node_execution_plans
from nnc_py.passes.indexed_forward_graph import Edge, IndexedForwardGraph, NodeEntry
from nnc_py.passes.dominator_tree import DominatorTree
from nnc_py.passes.base import PassBase, PassManager
# New memory strategy interface
from nnc_py.passes.memory_strategy import (
    AllocationStrategy,
    LogicalMemoryRegion,
    MemoryAllocationPlan,
    MemoryAllocationStrategy,
    StrategyRegistry,
    SpillPoint,
    ReloadPoint,
    TensorAllocation,
    get_allocation_plan,
    get_default_allocation_strategy,
    get_memory_strategy,
)

# New unified memory planning pass
from nnc_py.passes.memory_planning import (
    MemoryPlanningPassV2,
    MemoryPlanningPassV3,
    get_memory_allocation_plan,
)
from nnc_py.passes.memory_planning_v4 import MemoryPlanningPassV4
from nnc_py.passes.scheduled_memory_planning import (
    ScheduledFastAllocation,
    ScheduledMemoryPlan,
    ScheduledMemoryPlanningPass,
    ScheduledSlowAllocation,
    ScheduledTransferPoint,
)

# Optimization passes
from nnc_py.passes.dead_code_elimination import DeadCodeEliminationPass
from nnc_py.passes.identity_elimination import IdentityEliminationPass
from nnc_py.passes.pattern_fusion import PatternFusionPass
from nnc_py.passes.prepack_lowering import PrepackLoweringPass
from nnc_py.passes.layout_planning import LayoutPlanningPass, LayoutPlan
from nnc_py.passes.schedule_analysis import ScheduleAnalysisPass, ScheduleCandidate
from nnc_py.passes.scheduled_memory_expansion import ScheduledMemoryExpansionPass
from nnc_py.passes.pipeline_step_lowering import PipelineStepLoweringPass
from nnc_py.passes.pipeline_scheduling import PipelineSchedulingPass
from nnc_py.passes.tiled_lowering import TiledLoweringPass
from nnc_py.passes.dominator_fusion import DominatorFusionPass
from nnc_py.passes.fusion_groups import FusionGroup, GroupArena
from nnc_py.passes.path_validator import PathValidator

if TYPE_CHECKING:
    from nnc_py.passes.joint_tiling_schedule import (
        JointTilingScheduleMaterializationPass as JointTilingScheduleMaterializationPass,
    )
    from nnc_py.passes.joint_tiling_schedule import (
        JointTilingScheduleProblemPass as JointTilingScheduleProblemPass,
    )
    from nnc_py.passes.joint_tiling_schedule import (
        JointTilingScheduleSolvePass as JointTilingScheduleSolvePass,
    )

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
    "get_node_execution_plan",
    "get_node_execution_plans",

    # New memory strategy interface
    "AllocationStrategy",
    "LogicalMemoryRegion",
    "MemoryAllocationPlan",
    "MemoryAllocationStrategy",
    "StrategyRegistry",
    "SpillPoint",
    "ReloadPoint",
    "TensorAllocation",
    "get_allocation_plan",
    "get_default_allocation_strategy",
    "get_memory_strategy",

    # New unified pass
    "MemoryPlanningPassV2",
    "MemoryPlanningPassV3",
    "MemoryPlanningPassV4",
    "ScheduledFastAllocation",
    "ScheduledMemoryPlan",
    "ScheduledMemoryPlanningPass",
    "ScheduledSlowAllocation",
    "ScheduledTransferPoint",
    "get_memory_allocation_plan",

    # Optimization passes
    "DeadCodeEliminationPass",
    "IdentityEliminationPass",
    "PatternFusionPass",
    "PrepackLoweringPass",
    "LayoutPlanningPass",
    "LayoutPlan",
    "ScheduleAnalysisPass",
    "ScheduleCandidate",
    "ScheduledMemoryExpansionPass",
    "JointTilingScheduleMaterializationPass",
    "JointTilingScheduleProblemPass",
    "JointTilingScheduleSolvePass",
    "PipelineStepLoweringPass",
    "PipelineSchedulingPass",
    "TiledLoweringPass",
    "DominatorFusionPass",

    # Union-Find fusion groups
    "FusionGroup",
    "GroupArena",

    # Path validation
    "PathValidator",
]


def __getattr__(name: str) -> Any:
    if name not in {
        "JointTilingScheduleMaterializationPass",
        "JointTilingScheduleProblemPass",
        "JointTilingScheduleSolvePass",
    }:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from nnc_py.passes.joint_tiling_schedule import (
        JointTilingScheduleMaterializationPass,
        JointTilingScheduleProblemPass,
        JointTilingScheduleSolvePass,
    )

    exports = {
        "JointTilingScheduleMaterializationPass": JointTilingScheduleMaterializationPass,
        "JointTilingScheduleProblemPass": JointTilingScheduleProblemPass,
        "JointTilingScheduleSolvePass": JointTilingScheduleSolvePass,
    }
    return exports[name]
