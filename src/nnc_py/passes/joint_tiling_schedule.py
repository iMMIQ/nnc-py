"""Passes for the joint tiling/schedule O3 contract path."""

from __future__ import annotations

from collections.abc import Iterable
import shlex

from nnc_py.ir.context import CompileContext
from nnc_py.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_FAILURE_METADATA_KEY,
    JOINT_TILING_SCHEDULE_SOLUTION_METADATA_KEY,
    JointFailure,
    JointProblem,
    JointSolution,
    set_joint_tiling_schedule_failure,
    set_joint_tiling_schedule_problem,
    set_joint_tiling_schedule_solution,
)
from nnc_py.ir.pipeline_schedule import (
    set_pipeline_schedule_problem,
    set_pipeline_schedule_result,
)
from nnc_py.joint_schedule.materialize import materialize_joint_solution
from nnc_py.joint_schedule.recipes import build_joint_problem
from nnc_py.joint_schedule.solver import (
    BaselineJointScheduleSolver,
    CliJointScheduleSolver,
    JointScheduleSolver,
)
from nnc_py.joint_schedule.validation import (
    validate_joint_problem,
    validate_joint_solution,
)
from nnc_py.passes.base import PassBase


class JointTilingScheduleProblemPass(PassBase):
    """Build and validate the external joint tiling/schedule problem."""

    @property
    def name(self) -> str:
        return "JointTilingScheduleProblem"

    def _execute(self, ctx: CompileContext) -> None:
        problem = build_joint_problem(ctx)
        failure = validate_joint_problem(problem)
        set_joint_tiling_schedule_problem(ctx, problem)
        ctx.metadata.pop(JOINT_TILING_SCHEDULE_SOLUTION_METADATA_KEY, None)
        ctx.metadata.pop(JOINT_TILING_SCHEDULE_FAILURE_METADATA_KEY, None)
        if failure is not None:
            set_joint_tiling_schedule_failure(ctx, failure)
            return


class JointTilingScheduleSolvePass(PassBase):
    """Solve the joint tiling/schedule problem with either the baseline or CLI solver."""

    @property
    def name(self) -> str:
        return "JointTilingScheduleSolve"

    def _execute(self, ctx: CompileContext) -> None:
        if ctx.joint_tiling_schedule_failure is not None:
            return

        problem = ctx.joint_tiling_schedule_problem
        if problem is None:
            return

        solver = _build_solver(ctx)
        result = solver.solve(problem)
        if isinstance(result, JointFailure):
            set_joint_tiling_schedule_failure(ctx, result)
            ctx.metadata.pop(JOINT_TILING_SCHEDULE_SOLUTION_METADATA_KEY, None)
            return
        if not isinstance(result, JointSolution):
            raise TypeError("joint tiling schedule solver must return JointSolution or JointFailure")

        failure = validate_joint_solution(problem, result)
        if failure is not None:
            set_joint_tiling_schedule_failure(ctx, failure)
            ctx.metadata.pop(JOINT_TILING_SCHEDULE_SOLUTION_METADATA_KEY, None)
            return

        set_joint_tiling_schedule_solution(ctx, result)
        ctx.metadata.pop(JOINT_TILING_SCHEDULE_FAILURE_METADATA_KEY, None)


class JointTilingScheduleMaterializationPass(PassBase):
    """Materialize a validated joint solution into the internal scheduled IR."""

    @property
    def name(self) -> str:
        return "JointTilingScheduleMaterialization"

    def _execute(self, ctx: CompileContext) -> None:
        if ctx.joint_tiling_schedule_failure is not None:
            return

        problem = ctx.joint_tiling_schedule_problem
        solution = ctx.joint_tiling_schedule_solution
        if problem is None or solution is None:
            return

        failure = validate_joint_solution(problem, solution)
        if failure is not None:
            set_joint_tiling_schedule_failure(ctx, failure)
            return

        pipeline_problem, pipeline_result = materialize_joint_solution(problem, solution)
        set_pipeline_schedule_problem(ctx, pipeline_problem)
        set_pipeline_schedule_result(ctx, pipeline_result)
        ctx.metadata.pop("scheduled_memory_plan", None)
        ctx.metadata.pop("memory_allocation_plan", None)
        ctx.metadata.pop("memory_plan", None)
        ctx.metadata.pop("spill_plan", None)


def _build_solver(ctx: CompileContext) -> JointScheduleSolver:
    command = _normalize_command(ctx.metadata.get("joint_tiling_schedule_solver_command"))
    if command:
        return CliJointScheduleSolver(command=command)
    return BaselineJointScheduleSolver()


def _normalize_command(command: object) -> list[str] | None:
    if command is None:
        return None
    if isinstance(command, str):
        normalized = shlex.split(command)
        return normalized or None
    if isinstance(command, Iterable):
        normalized = [str(part) for part in command]
        return normalized or None
    return None

__all__ = [
    "JointTilingScheduleMaterializationPass",
    "JointTilingScheduleProblemPass",
    "JointTilingScheduleSolvePass",
]
