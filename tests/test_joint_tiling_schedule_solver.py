from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nnc_py.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
    JointAction,
    JointBoundaryConstraint,
    JointDependencyEdge,
    JointFailure,
    JointFailureStatus,
    JointLayoutSpec,
    JointProblem,
    JointRecipe,
    JointRegion,
    JointResource,
    JointSelectedRecipe,
    JointSolution,
    JointTileSpec,
    JointValue,
    JointValueFootprint,
    JointValueTier,
    JointCostParameters,
)
from nnc_py.joint_schedule.solver import (
    CliJointScheduleSolver,
    JointSolverTransportError,
)


def _minimal_joint_problem() -> JointProblem:
    return JointProblem(
        schema_version=JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
        regions=(
            JointRegion(
                region_id="region0",
                kind="single_op",
                input_value_ids=("input0",),
                output_value_ids=("output0",),
            ),
        ),
        recipes=(
            JointRecipe(
                recipe_id="region0.recipe0",
                region_id="region0",
                tile_spec=JointTileSpec(axes=("h", "w"), shape=(8, 8)),
                layout_spec=JointLayoutSpec(layout_tags=("nchw",)),
                activates_action_ids=("region0.recipe0.compute",),
                value_footprint=JointValueFootprint(
                    resident_bytes=128,
                    scratch_bytes=64,
                    transfer_bytes=32,
                ),
                cost_parameters=JointCostParameters(latency=10, launch_overhead=2),
            ),
        ),
        values=(
            JointValue(
                value_id="input0",
                size_bytes=64,
                initial_tier=JointValueTier.INPUT,
                required_final_tier=JointValueTier.INPUT,
                must_keep=False,
                spillable=False,
                allows_multiple_sram_windows=False,
                producer=None,
                consumers=(),
            ),
            JointValue(
                value_id="output0",
                size_bytes=64,
                initial_tier=JointValueTier.UNMATERIALIZED,
                required_final_tier=JointValueTier.SLOW,
                must_keep=False,
                spillable=False,
                allows_multiple_sram_windows=False,
                producer=None,
                consumers=(),
            ),
        ),
        actions=(
            JointAction(
                action_id="region0.recipe0.compute",
                kind="compute",
                resource_kind="OTHER",
                duration=10,
                launch_overhead=2,
                reads=("input0",),
                writes=("output0",),
                temp_bytes=64,
                is_optional=False,
                region_id="region0",
                recipe_id="region0.recipe0",
                optional_value_id=None,
            ),
        ),
        boundary_constraints=(),
        dependency_edges=(
            JointDependencyEdge(
                src_action_id="region0.recipe0.compute",
                dst_action_id="region0.recipe0.compute",
                kind="order",
            ),
        ),
        resources=(JointResource(resource_kind="OTHER", slot_count=1),),
        sram_capacity_bytes=1024,
        objective="min_makespan",
    )


@pytest.fixture
def minimal_joint_problem() -> JointProblem:
    problem = _minimal_joint_problem()
    return JointProblem(
        schema_version=problem.schema_version,
        regions=problem.regions,
        recipes=problem.recipes,
        values=problem.values,
        actions=(
            JointAction(
                action_id="region0.recipe0.compute",
                kind="compute",
                resource_kind="OTHER",
                duration=10,
                launch_overhead=2,
                reads=("input0",),
                writes=("output0",),
                temp_bytes=64,
                is_optional=False,
                region_id="region0",
                recipe_id="region0.recipe0",
                optional_value_id=None,
            ),
        ),
        boundary_constraints=problem.boundary_constraints,
        dependency_edges=(),
        resources=problem.resources,
        sram_capacity_bytes=problem.sram_capacity_bytes,
        objective=problem.objective,
    )


def _solver_command(mode: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("fake_joint_solver.py")),
        mode,
    ]


def test_cli_solver_parses_solution_payload(minimal_joint_problem: JointProblem):
    result = CliJointScheduleSolver(_solver_command("solution")).solve(
        minimal_joint_problem
    )

    assert isinstance(result, JointSolution)
    assert result.schema_version == JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION
    assert result.selected_recipes == (
        JointSelectedRecipe(region_id="region0", recipe_id="region0.recipe0"),
    )


def test_cli_solver_attaches_stderr_to_solution_diagnostics(
    minimal_joint_problem: JointProblem,
):
    result = CliJointScheduleSolver(_solver_command("solution_stderr")).solve(
        minimal_joint_problem
    )

    assert isinstance(result, JointSolution)
    assert result.diagnostics["_solver_transport"]["stderr"] == "solver emitted warning"


def test_cli_solver_parses_infeasible_failure_payload(
    minimal_joint_problem: JointProblem,
):
    result = CliJointScheduleSolver(_solver_command("infeasible")).solve(
        minimal_joint_problem
    )

    assert isinstance(result, JointFailure)
    assert result.schema_version == JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION
    assert result.status is JointFailureStatus.INFEASIBLE


def test_cli_solver_parses_timeout_failure_payload(
    minimal_joint_problem: JointProblem,
):
    result = CliJointScheduleSolver(_solver_command("timeout")).solve(
        minimal_joint_problem
    )

    assert isinstance(result, JointFailure)
    assert result.schema_version == JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION
    assert result.status is JointFailureStatus.TIMEOUT


def test_cli_solver_parses_error_failure_payload(
    minimal_joint_problem: JointProblem,
):
    result = CliJointScheduleSolver(_solver_command("error")).solve(
        minimal_joint_problem
    )

    assert isinstance(result, JointFailure)
    assert result.schema_version == JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION
    assert result.status is JointFailureStatus.ERROR


def test_cli_solver_raises_transport_error_on_non_zero_exit(
    minimal_joint_problem: JointProblem,
):
    solver = CliJointScheduleSolver(_solver_command("crash"))

    with pytest.raises(JointSolverTransportError, match="exited with code 7"):
        solver.solve(minimal_joint_problem)


def test_cli_solver_prioritizes_transport_error_over_failure_payload_on_non_zero_exit(
    minimal_joint_problem: JointProblem,
):
    solver = CliJointScheduleSolver(_solver_command("crash_with_payload"))

    with pytest.raises(JointSolverTransportError, match="exited with code 7"):
        solver.solve(minimal_joint_problem)


def test_cli_solver_raises_transport_error_on_malformed_failure_payload(
    minimal_joint_problem: JointProblem,
):
    solver = CliJointScheduleSolver(_solver_command("malformed_failure"))

    with pytest.raises(JointSolverTransportError, match="malformed failure payload"):
        solver.solve(minimal_joint_problem)


def test_cli_solver_raises_transport_error_on_malformed_solution_payload(
    minimal_joint_problem: JointProblem,
):
    solver = CliJointScheduleSolver(_solver_command("malformed_solution"))

    with pytest.raises(JointSolverTransportError, match="malformed solution payload"):
        solver.solve(minimal_joint_problem)
