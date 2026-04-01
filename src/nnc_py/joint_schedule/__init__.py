"""Compiler-owned builders for the external joint tiling/schedule contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nnc_py.joint_schedule.recipes import build_joint_problem as build_joint_problem
    from nnc_py.joint_schedule.regions import (
        JointProblemBuilderError as JointProblemBuilderError,
    )
    from nnc_py.joint_schedule.regions import build_joint_regions as build_joint_regions

__all__ = ["JointProblemBuilderError", "build_joint_problem", "build_joint_regions"]


def __getattr__(name: str) -> Any:
    if name == "build_joint_problem":
        from nnc_py.joint_schedule.recipes import build_joint_problem

        return build_joint_problem
    if name in {"JointProblemBuilderError", "build_joint_regions"}:
        from nnc_py.joint_schedule.regions import (
            JointProblemBuilderError,
            build_joint_regions,
        )

        exports = {
            "JointProblemBuilderError": JointProblemBuilderError,
            "build_joint_regions": build_joint_regions,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
