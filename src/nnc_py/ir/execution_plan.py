"""Execution-plan IR primitives for schedule-aware lowering."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from nnc_py.ir.types import AxisNames

if TYPE_CHECKING:
    from nnc_py.ir.context import CompileContext


NODE_EXECUTION_PLANS_METADATA_KEY = "node_execution_plans"


class LayoutClass(Enum):
    """Generic layout classes used before backend physical mapping."""

    PLAIN = "plain"
    BLOCKED_ACTIVATION = "blocked_activation"
    BLOCKED_WEIGHT = "blocked_weight"
    TARGET_PHYSICAL = "target_physical"


class MemoryRegionKind(Enum):
    """Logical memory regions required by a node execution step."""

    PERSISTENT = "persistent"
    TILE = "tile"
    SCRATCH = "scratch"
    PACK = "pack"
    STAGE = "stage"


@dataclass(frozen=True)
class TileRegion:
    """Logical tile extents and constraints for one execution step."""

    logical_extents: tuple[int, ...] = ()
    halo_extents: tuple[int, ...] = ()
    block_alignment: tuple[int, ...] = ()


@dataclass(frozen=True)
class TensorAccessPlan:
    """Materialization contract for an input or output tensor."""

    tensor_name: str
    layout_class: LayoutClass = LayoutClass.PLAIN
    tile_region: TileRegion = field(default_factory=TileRegion)
    memory_region: MemoryRegionKind = MemoryRegionKind.PERSISTENT


@dataclass(frozen=True)
class NodeExecutionPlan:
    """Execution contract for one semantic node."""

    node_name: str
    op_family: str
    tile_axes: AxisNames = ()
    layout_class: LayoutClass = LayoutClass.PLAIN
    memory_regions: tuple[MemoryRegionKind, ...] = ()
    input_accesses: tuple[TensorAccessPlan, ...] = ()
    output_accesses: tuple[TensorAccessPlan, ...] = ()


def get_node_execution_plans(ctx: CompileContext) -> dict[str, NodeExecutionPlan]:
    """Return the typed node execution-plan map stored in compile metadata."""

    plans = ctx.metadata.get(NODE_EXECUTION_PLANS_METADATA_KEY)
    if plans is None:
        plans = {}
        ctx.metadata[NODE_EXECUTION_PLANS_METADATA_KEY] = plans
    return plans


def get_node_execution_plan(
    ctx: CompileContext, node_name: str
) -> NodeExecutionPlan | None:
    """Return the execution plan for a node when present."""

    return get_node_execution_plans(ctx).get(node_name)


def set_node_execution_plan(ctx: CompileContext, plan: NodeExecutionPlan) -> None:
    """Store a typed execution plan keyed by node name."""

    get_node_execution_plans(ctx)[plan.node_name] = plan


__all__ = [
    "NODE_EXECUTION_PLANS_METADATA_KEY",
    "LayoutClass",
    "MemoryRegionKind",
    "NodeExecutionPlan",
    "TensorAccessPlan",
    "TileRegion",
    "get_node_execution_plan",
    "get_node_execution_plans",
    "set_node_execution_plan",
]
