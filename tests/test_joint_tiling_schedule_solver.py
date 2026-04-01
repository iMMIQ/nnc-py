from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests import fake_joint_solver as fake_solver_module
from nnc_py.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
    JointAction,
    JointActionKind,
    JointBoundaryConstraint,
    JointDependencyEdge,
    JointDependencyEdgeKind,
    JointFailure,
    JointFailureCategory,
    JointFailureStatus,
    JointLayoutSpec,
    JointProblem,
    JointRecipe,
    JointRegion,
    JointResourceKind,
    JointSramAllocation,
    JointSramItem,
    JointSramItemKind,
    JointResource,
    JointSelectedRecipe,
    JointSolution,
    JointTileSpec,
    JointValue,
    JointValueConsumer,
    JointValueFootprint,
    JointValueProducer,
    JointValueTier,
    JointCostParameters,
)
from nnc_py.joint_schedule.solver import (
    CliJointScheduleSolver,
    JointSolverTransportError,
)
from tests.joint_solver_helpers import joint_solver_cli_command


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


def _allocatable_joint_problem() -> JointProblem:
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
                activates_action_ids=(
                    "region0.recipe0.dma_in.input0",
                    "region0.recipe0.compute",
                    "region0.recipe0.dma_out.output0",
                ),
                value_footprint=JointValueFootprint(
                    resident_bytes=160,
                    scratch_bytes=64,
                    transfer_bytes=160,
                ),
                cost_parameters=JointCostParameters(latency=9, launch_overhead=3),
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
                consumers=(
                    JointValueConsumer(action_id="region0.recipe0.dma_in.input0"),
                    JointValueConsumer(action_id="region0.recipe0.compute"),
                ),
            ),
            JointValue(
                value_id="output0",
                size_bytes=96,
                initial_tier=JointValueTier.UNMATERIALIZED,
                required_final_tier=JointValueTier.SLOW,
                must_keep=False,
                spillable=False,
                allows_multiple_sram_windows=False,
                producer=JointValueProducer(action_id="region0.recipe0.compute"),
                consumers=(
                    JointValueConsumer(action_id="region0.recipe0.dma_out.output0"),
                ),
            ),
        ),
        actions=(
            JointAction(
                action_id="region0.recipe0.dma_in.input0",
                kind=JointActionKind.DMA_IN,
                resource_kind=JointResourceKind.DMA,
                duration=2,
                launch_overhead=1,
                reads=("input0",),
                writes=("input0",),
                temp_bytes=0,
                is_optional=False,
                region_id="region0",
                recipe_id="region0.recipe0",
                optional_value_id=None,
            ),
            JointAction(
                action_id="region0.recipe0.compute",
                kind=JointActionKind.COMPUTE,
                resource_kind=JointResourceKind.OTHER,
                duration=5,
                launch_overhead=1,
                reads=("input0",),
                writes=("output0",),
                temp_bytes=64,
                is_optional=False,
                region_id="region0",
                recipe_id="region0.recipe0",
                optional_value_id=None,
            ),
            JointAction(
                action_id="region0.recipe0.dma_out.output0",
                kind=JointActionKind.DMA_OUT,
                resource_kind=JointResourceKind.DMA,
                duration=2,
                launch_overhead=1,
                reads=("output0",),
                writes=("output0",),
                temp_bytes=0,
                is_optional=False,
                region_id="region0",
                recipe_id="region0.recipe0",
                optional_value_id=None,
            ),
        ),
        boundary_constraints=(),
        dependency_edges=(
            JointDependencyEdge(
                src_action_id="region0.recipe0.dma_in.input0",
                dst_action_id="region0.recipe0.compute",
                kind=JointDependencyEdgeKind.DATA,
            ),
            JointDependencyEdge(
                src_action_id="region0.recipe0.compute",
                dst_action_id="region0.recipe0.dma_out.output0",
                kind=JointDependencyEdgeKind.DATA,
            ),
        ),
        resources=(
            JointResource(resource_kind=JointResourceKind.DMA, slot_count=1),
            JointResource(resource_kind=JointResourceKind.MATMUL, slot_count=1),
            JointResource(resource_kind=JointResourceKind.SHAPE, slot_count=1),
            JointResource(resource_kind=JointResourceKind.OTHER, slot_count=1),
        ),
        sram_capacity_bytes=192,
        sram_items=(
            JointSramItem(
                item_id="region0.recipe0.compute.temp",
                kind=JointSramItemKind.TEMP_INTERVAL,
                size_bytes=64,
                alignment_bytes=16,
                is_optional=False,
                owner_action_id="region0.recipe0.compute",
                owner_value_id=None,
                owner_residency_id=None,
            ),
            JointSramItem(
                item_id="region0.recipe0.compute.pack",
                kind=JointSramItemKind.TRANSFER_BUFFER,
                size_bytes=32,
                alignment_bytes=16,
                is_optional=False,
                owner_action_id="region0.recipe0.dma_in.input0",
                owner_value_id=None,
                owner_residency_id=None,
            ),
        ),
        default_alignment_bytes=16,
        objective="min_makespan",
    )


@pytest.fixture
def allocatable_joint_problem() -> JointProblem:
    return _allocatable_joint_problem()


@pytest.fixture
def minimal_joint_problem() -> JointProblem:
    return _allocatable_joint_problem()


def _fake_solver_command(mode: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("fake_joint_solver.py")),
        mode,
    ]


def test_cli_solver_parses_solution_payload(minimal_joint_problem: JointProblem):
    result = CliJointScheduleSolver(joint_solver_cli_command()).solve(
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
    result = CliJointScheduleSolver(_fake_solver_command("solution_stderr")).solve(
        minimal_joint_problem
    )

    assert isinstance(result, JointSolution)
    assert result.diagnostics["_solver_transport"]["stderr"] == "solver emitted warning"


def test_cli_solver_submodule_emits_generated_residency_items_and_sram_allocations(
    allocatable_joint_problem: JointProblem,
):
    result = CliJointScheduleSolver(joint_solver_cli_command()).solve(
        allocatable_joint_problem
    )

    assert isinstance(result, JointSolution)
    assert {window.residency_id for window in result.residency_windows} == {"input0@3", "output0@9"}
    assert {item.item_id for item in result.generated_sram_items} == {
        "input0@3.item",
        "output0@9.item",
    }
    assert {allocation.item_id for allocation in result.sram_allocations} == {
        "region0.recipe0.compute.temp",
        "region0.recipe0.compute.pack",
        "input0@3.item",
        "output0@9.item",
    }
    offsets = {allocation.item_id: allocation.offset for allocation in result.sram_allocations}
    assert offsets["output0@9.item"] == offsets["region0.recipe0.compute.temp"]
    assert offsets["output0@9.item"] != offsets["input0@3.item"]


def test_cli_solver_parses_infeasible_failure_payload(
    minimal_joint_problem: JointProblem,
):
    result = CliJointScheduleSolver(_fake_solver_command("infeasible")).solve(
        minimal_joint_problem
    )

    assert isinstance(result, JointFailure)
    assert result.schema_version == JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION
    assert result.status is JointFailureStatus.INFEASIBLE


def test_cli_solver_parses_timeout_failure_payload(
    minimal_joint_problem: JointProblem,
):
    result = CliJointScheduleSolver(_fake_solver_command("timeout")).solve(
        minimal_joint_problem
    )

    assert isinstance(result, JointFailure)
    assert result.schema_version == JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION
    assert result.status is JointFailureStatus.TIMEOUT


def test_cli_solver_parses_error_failure_payload(
    minimal_joint_problem: JointProblem,
):
    result = CliJointScheduleSolver(_fake_solver_command("error")).solve(
        minimal_joint_problem
    )

    assert isinstance(result, JointFailure)
    assert result.schema_version == JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION
    assert result.status is JointFailureStatus.ERROR


def test_cli_solver_raises_transport_error_on_non_zero_exit(
    minimal_joint_problem: JointProblem,
):
    solver = CliJointScheduleSolver(_fake_solver_command("crash"))

    with pytest.raises(JointSolverTransportError, match="exited with code 7"):
        solver.solve(minimal_joint_problem)


def test_cli_solver_prioritizes_transport_error_over_failure_payload_on_non_zero_exit(
    minimal_joint_problem: JointProblem,
):
    solver = CliJointScheduleSolver(_fake_solver_command("crash_with_payload"))

    with pytest.raises(JointSolverTransportError, match="exited with code 7"):
        solver.solve(minimal_joint_problem)


def test_cli_solver_raises_transport_error_on_malformed_failure_payload(
    minimal_joint_problem: JointProblem,
):
    solver = CliJointScheduleSolver(_fake_solver_command("malformed_failure"))

    with pytest.raises(JointSolverTransportError, match="malformed failure payload"):
        solver.solve(minimal_joint_problem)


def test_cli_solver_raises_transport_error_on_malformed_solution_payload(
    minimal_joint_problem: JointProblem,
):
    solver = CliJointScheduleSolver(_fake_solver_command("malformed_solution"))

    with pytest.raises(JointSolverTransportError, match="malformed solution payload"):
        solver.solve(minimal_joint_problem)


def test_fake_solver_solution_mode_rejects_non_solution_payload(monkeypatch):
    class _Result:
        returncode = 0
        stdout = (
            '{"schema_version":"'
            + JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION
            + '","status":"error","error_category":"invalid_solution","diagnostics":{}}'
        )
        stderr = ""

    monkeypatch.setattr(fake_solver_module.subprocess, "run", lambda *args, **kwargs: _Result())

    with pytest.raises(RuntimeError, match="expected solution payload"):
        fake_solver_module._build_solution(_allocatable_joint_problem().to_json())
