"""Generic blocked layout planning for schedule-aware lowering."""

from __future__ import annotations

from dataclasses import dataclass

from nnc_py.ir.context import CompileContext
from nnc_py.ir.types import AxisNames, GenericBlockedLayoutKind
from nnc_py.passes.base import PassBase
from nnc_py.passes.schedule_analysis import ScheduleCandidate

LAYOUT_PLANS_METADATA_KEY = "layout_plans"
_WEIGHTED_OP_FAMILIES = {"conv2d", "gemm", "matmul"}


@dataclass(frozen=True)
class GenericBlockedLayout:
    """Target-agnostic blocked layout choice for one tensor role."""

    name: str
    blocked_axes: AxisNames


@dataclass(frozen=True)
class LayoutPlan:
    """Per-node generic layout selections for phase-1 lowering."""

    node_name: str
    op_family: str
    input_layout: GenericBlockedLayout
    weight_layout: GenericBlockedLayout | None = None
    target_physical_layout: str | None = None


class LayoutPlanningPass(PassBase):
    """Assign conservative generic blocked layouts from schedule candidates."""

    @property
    def name(self) -> str:
        return "LayoutPlanning"

    def _execute(self, ctx: CompileContext) -> None:
        ctx.metadata[LAYOUT_PLANS_METADATA_KEY] = build_generic_blocked_layouts(ctx)


def build_generic_blocked_layouts(ctx: CompileContext) -> dict[str, LayoutPlan]:
    """Choose generic blocked layouts for supported scheduled nodes."""

    schedule_candidates: dict[str, ScheduleCandidate] = ctx.metadata.get(
        "schedule_candidates", {}
    )
    plans: dict[str, LayoutPlan] = {}

    for node_name, candidate in schedule_candidates.items():
        input_layout = GenericBlockedLayout(
            name=GenericBlockedLayoutKind.BLOCKED_ACTIVATION.value,
            blocked_axes=("C",),
        )
        weight_layout = None
        if candidate.op_family in _WEIGHTED_OP_FAMILIES:
            weight_layout = GenericBlockedLayout(
                name=GenericBlockedLayoutKind.BLOCKED_WEIGHT.value,
                blocked_axes=("K", "C"),
            )

        plans[node_name] = LayoutPlan(
            node_name=node_name,
            op_family=candidate.op_family,
            input_layout=input_layout,
            weight_layout=weight_layout,
        )

    return plans


__all__ = [
    "GenericBlockedLayout",
    "LAYOUT_PLANS_METADATA_KEY",
    "LayoutPlan",
    "LayoutPlanningPass",
    "build_generic_blocked_layouts",
]
