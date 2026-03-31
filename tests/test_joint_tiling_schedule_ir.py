import json

import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
    JointAction,
    JointActionKind,
    JointBoundaryConstraint,
    JointCompatibleRecipePair,
    JointDependencyEdge,
    JointDependencyEdgeKind,
    JointFailure,
    JointFailureCategory,
    JointFailureStatus,
    JointLayoutSpec,
    JointProblem,
    JointRecipe,
    JointRegion,
    JointRegionKind,
    JointResidencyWindow,
    JointResource,
    JointResourceKind,
    JointScheduledAction,
    JointSelectedRecipe,
    JointSolution,
    JointTileSpec,
    JointValue,
    JointValueConsumer,
    JointValueFootprint,
    JointValueProducer,
    JointValueTier,
    JointCostParameters,
    get_joint_tiling_schedule_failure,
    get_joint_tiling_schedule_problem,
    get_joint_tiling_schedule_solution,
    set_joint_tiling_schedule_failure,
    set_joint_tiling_schedule_problem,
    set_joint_tiling_schedule_solution,
)


def _sample_problem() -> JointProblem:
    return JointProblem(
        schema_version=JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
        regions=(
            JointRegion(
                region_id="region0",
                kind=JointRegionKind.SINGLE_OP,
                member_nodes=("conv0",),
                input_value_ids=("input0", "weight0"),
                output_value_ids=("output0",),
                predecessor_region_ids=(),
                successor_region_ids=("region1",),
            ),
            JointRegion(
                region_id="region1",
                kind="fused_group",
                member_nodes=("relu0", "add0"),
                input_value_ids=("output0",),
                output_value_ids=("output1",),
                predecessor_region_ids=("region0",),
                successor_region_ids=(),
            ),
        ),
        recipes=(
            JointRecipe(
                recipe_id="recipe0",
                region_id="region0",
                tile_spec=JointTileSpec(axes=("n", "h"), shape=(1, 8)),
                layout_spec=JointLayoutSpec(layout_tags=("nhwc",)),
                activates_action_ids=("action0", "action1"),
                value_footprint=JointValueFootprint(
                    resident_bytes=128,
                    scratch_bytes=64,
                    transfer_bytes=32,
                ),
                cost_parameters=JointCostParameters(latency=17, launch_overhead=3),
            ),
            JointRecipe(
                recipe_id="recipe1",
                region_id="region1",
                tile_spec=JointTileSpec(axes=("n", "h"), shape=(1, 8)),
                layout_spec=JointLayoutSpec(layout_tags=("nhwc", "fused")),
                activates_action_ids=("action2",),
                value_footprint=JointValueFootprint(
                    resident_bytes=64,
                    scratch_bytes=16,
                    transfer_bytes=0,
                ),
                cost_parameters=JointCostParameters(latency=9, launch_overhead=1),
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
                consumers=(JointValueConsumer(action_id="action0"),),
            ),
            JointValue(
                value_id="weight0",
                size_bytes=32,
                initial_tier="const",
                required_final_tier=JointValueTier.CONST,
                must_keep=False,
                spillable=False,
                allows_multiple_sram_windows=False,
                producer=None,
                consumers=(JointValueConsumer(action_id="action0"),),
            ),
            JointValue(
                value_id="output0",
                size_bytes=96,
                initial_tier=JointValueTier.UNMATERIALIZED,
                required_final_tier=JointValueTier.SRAM,
                must_keep=False,
                spillable=True,
                allows_multiple_sram_windows=True,
                producer=JointValueProducer(action_id="action0"),
                consumers=(
                    JointValueConsumer(action_id="action1"),
                    JointValueConsumer(action_id="action2"),
                ),
            ),
            JointValue(
                value_id="output1",
                size_bytes=96,
                initial_tier=JointValueTier.UNMATERIALIZED,
                required_final_tier=JointValueTier.SLOW,
                must_keep=False,
                spillable=False,
                allows_multiple_sram_windows=False,
                producer=JointValueProducer(action_id="action2"),
                consumers=(),
            ),
        ),
        actions=(
            JointAction(
                action_id="action0",
                kind=JointActionKind.DMA_IN,
                resource_kind=JointResourceKind.DMA,
                duration=5,
                launch_overhead=1,
                reads=("input0",),
                writes=("input0",),
                temp_bytes=0,
                is_optional=False,
                region_id="region0",
                recipe_id="recipe0",
                optional_value_id=None,
            ),
            JointAction(
                action_id="action1",
                kind="compute",
                resource_kind="MATMUL",
                duration=17,
                launch_overhead=3,
                reads=("input0", "weight0"),
                writes=("output0",),
                temp_bytes=64,
                is_optional=False,
                region_id="region0",
                recipe_id="recipe0",
                optional_value_id=None,
            ),
            JointAction(
                action_id="action2",
                kind=JointActionKind.COMPUTE,
                resource_kind=JointResourceKind.OTHER,
                duration=9,
                launch_overhead=1,
                reads=("output0",),
                writes=("output1",),
                temp_bytes=16,
                is_optional=False,
                region_id="region1",
                recipe_id="recipe1",
                optional_value_id=None,
            ),
            JointAction(
                action_id="spill0",
                kind=JointActionKind.SPILL,
                resource_kind=JointResourceKind.DMA,
                duration=4,
                launch_overhead=1,
                reads=("output0",),
                writes=("output0",),
                temp_bytes=0,
                is_optional=True,
                region_id=None,
                recipe_id=None,
                optional_value_id="output0",
            ),
        ),
        boundary_constraints=(
            JointBoundaryConstraint(
                boundary_id="boundary0",
                src_region_id="region0",
                dst_region_id="region1",
                compatible_recipe_pairs=(
                    JointCompatibleRecipePair(
                        src_recipe_id="recipe0",
                        dst_recipe_id="recipe1",
                    ),
                ),
                required_layout_relations=("same_layout",),
                required_tile_domain_relations=("same_domain",),
            ),
        ),
        dependency_edges=(
            JointDependencyEdge(
                src_action_id="action0",
                dst_action_id="action1",
                kind=JointDependencyEdgeKind.DATA,
            ),
            JointDependencyEdge(
                src_action_id="action1",
                dst_action_id="action2",
                kind="order",
            ),
        ),
        resources=(
            JointResource(resource_kind=JointResourceKind.DMA, slot_count=1),
            JointResource(resource_kind="MATMUL", slot_count=1),
            JointResource(resource_kind=JointResourceKind.SHAPE, slot_count=1),
            JointResource(resource_kind=JointResourceKind.OTHER, slot_count=1),
        ),
        sram_capacity_bytes=1024,
        objective="min_makespan",
    )


def _sample_solution() -> JointSolution:
    return JointSolution(
        schema_version=JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
        selected_recipes=(
            JointSelectedRecipe(region_id="region0", recipe_id="recipe0"),
            JointSelectedRecipe(region_id="region1", recipe_id="recipe1"),
        ),
        scheduled_actions=(
            JointScheduledAction(action_id="action0", start_time=0),
            JointScheduledAction(action_id="action1", start_time=6),
            JointScheduledAction(action_id="action2", start_time=26),
        ),
        residency_windows=(
            JointResidencyWindow(value_id="input0", start_time=0, end_time=26),
            JointResidencyWindow(value_id="output0", start_time=26, end_time=36),
        ),
        objective_value=36,
        diagnostics={"solver": {"status": "ok"}},
    )


def _sample_failure() -> JointFailure:
    return JointFailure(
        schema_version=JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
        status=JointFailureStatus.INFEASIBLE,
        error_category=JointFailureCategory.SOLVER_REPORTED_INFEASIBLE,
        diagnostics={"reason": "search exhausted"},
    )


def test_joint_problem_keeps_required_top_level_arrays():
    problem = _sample_problem()

    assert problem.schema_version == JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION
    assert problem.objective == "min_makespan"
    assert problem.regions[0].kind is JointRegionKind.SINGLE_OP
    assert problem.regions[1].kind is JointRegionKind.FUSED_GROUP
    assert problem.dependency_edges[0].kind is JointDependencyEdgeKind.DATA
    assert problem.values[0].initial_tier is JointValueTier.INPUT
    assert problem.actions[1].kind is JointActionKind.COMPUTE
    assert problem.actions[1].resource_kind is JointResourceKind.MATMUL


def test_joint_contract_json_round_trips_and_ignores_unknown_fields():
    problem_payload = _sample_problem().to_json()
    problem_payload["unknown_extension"] = {"future": True}
    problem_payload["regions"][0]["extra_region_field"] = "ignored"
    problem_payload["recipes"][0]["tile_spec"]["future_axis_tags"] = ["x"]

    restored_problem = JointProblem.from_json(problem_payload)

    assert restored_problem == _sample_problem()
    assert json.loads(json.dumps(problem_payload))["schema_version"] == (
        JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION
    )

    solution_payload = _sample_solution().to_json()
    solution_payload["ignored"] = [1, 2, 3]
    solution_payload["scheduled_actions"][0]["end_time"] = 6

    restored_solution = JointSolution.from_json(solution_payload)

    assert restored_solution == _sample_solution()

    failure_payload = _sample_failure().to_json()
    failure_payload["ignored"] = "future"

    restored_failure = JointFailure.from_json(failure_payload)

    assert restored_failure == _sample_failure()


def test_joint_contract_requires_expected_fields_in_from_json():
    problem_payload = _sample_problem().to_json()
    del problem_payload["regions"]

    with pytest.raises(ValueError, match="regions"):
        JointProblem.from_json(problem_payload)

    value_payload = _sample_problem().to_json()["values"][0]
    del value_payload["producer"]
    broken_problem_payload = _sample_problem().to_json()
    broken_problem_payload["values"][0] = value_payload

    with pytest.raises(ValueError, match="producer"):
        JointProblem.from_json(broken_problem_payload)

    action_payload = _sample_problem().to_json()["actions"][0]
    del action_payload["recipe_id"]
    broken_problem_payload = _sample_problem().to_json()
    broken_problem_payload["actions"][0] = action_payload

    with pytest.raises(ValueError, match="recipe_id"):
        JointProblem.from_json(broken_problem_payload)


def test_joint_contract_rejects_invalid_literals_and_duplicate_ids():
    with pytest.raises(ValueError, match="schema_version"):
        JointProblem(
            schema_version="bad",
            regions=(),
            recipes=(),
            values=(),
            actions=(),
            boundary_constraints=(),
            dependency_edges=(),
            resources=(),
            sram_capacity_bytes=0,
            objective="min_makespan",
        )

    with pytest.raises(ValueError, match="JointRegion.kind"):
        JointRegion(
            region_id="region0",
            kind="bad",
            input_value_ids=(),
            output_value_ids=(),
        )

    with pytest.raises(ValueError, match="JointDependencyEdge.kind"):
        JointDependencyEdge(src_action_id="a", dst_action_id="b", kind="bad")

    with pytest.raises(ValueError, match="JointValue.initial_tier"):
        JointValue(
            value_id="value0",
            size_bytes=1,
            initial_tier="bad",
            required_final_tier=JointValueTier.SLOW,
            must_keep=False,
            spillable=False,
            allows_multiple_sram_windows=False,
            producer=None,
            consumers=(),
        )

    with pytest.raises(ValueError, match="duplicate region_id"):
        JointProblem(
            schema_version=JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
            regions=(
                JointRegion(
                    region_id="region0",
                    kind=JointRegionKind.SINGLE_OP,
                    input_value_ids=(),
                    output_value_ids=(),
                ),
                JointRegion(
                    region_id="region0",
                    kind=JointRegionKind.FUSED_GROUP,
                    input_value_ids=(),
                    output_value_ids=(),
                ),
            ),
            recipes=(),
            values=(),
            actions=(),
            boundary_constraints=(),
            dependency_edges=(),
            resources=(),
            sram_capacity_bytes=0,
            objective="min_makespan",
        )


def test_joint_value_and_action_validate_required_nullable_fields():
    value = JointValue(
        value_id="value0",
        size_bytes=32,
        initial_tier=JointValueTier.SLOW,
        required_final_tier=JointValueTier.SRAM,
        must_keep=False,
        spillable=True,
        allows_multiple_sram_windows=True,
        producer=None,
        consumers=(),
    )
    action = JointAction(
        action_id="reload0",
        kind=JointActionKind.RELOAD,
        resource_kind=JointResourceKind.DMA,
        duration=4,
        launch_overhead=1,
        reads=("value0",),
        writes=("value0",),
        temp_bytes=0,
        is_optional=True,
        region_id=None,
        recipe_id=None,
        optional_value_id="value0",
    )

    assert value.producer is None
    assert action.region_id is None
    assert action.recipe_id is None
    assert action.optional_value_id == "value0"


def test_joint_solution_and_failure_validate_unique_entries_and_json_payloads():
    solution = _sample_solution()
    failure = _sample_failure()

    assert json.loads(json.dumps(solution.to_json())) == {
        "schema_version": JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
        "selected_recipes": [
            {"region_id": "region0", "recipe_id": "recipe0"},
            {"region_id": "region1", "recipe_id": "recipe1"},
        ],
        "scheduled_actions": [
            {"action_id": "action0", "start_time": 0},
            {"action_id": "action1", "start_time": 6},
            {"action_id": "action2", "start_time": 26},
        ],
        "residency_windows": [
            {"value_id": "input0", "start_time": 0, "end_time": 26},
            {"value_id": "output0", "start_time": 26, "end_time": 36},
        ],
        "objective_value": 36,
        "diagnostics": {"solver": {"status": "ok"}},
    }
    assert json.loads(json.dumps(failure.to_json())) == {
        "schema_version": JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
        "status": "infeasible",
        "error_category": "solver_reported_infeasible",
        "diagnostics": {"reason": "search exhausted"},
    }

    with pytest.raises(ValueError, match="duplicate region_id"):
        JointSolution(
            schema_version=JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
            selected_recipes=(
                JointSelectedRecipe(region_id="region0", recipe_id="recipe0"),
                JointSelectedRecipe(region_id="region0", recipe_id="recipe1"),
            ),
            scheduled_actions=(),
            residency_windows=(),
            objective_value=0,
            diagnostics={},
        )

    with pytest.raises(ValueError, match="JointFailure.error_category"):
        JointFailure(
            schema_version=JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
            status=JointFailureStatus.ERROR,
            error_category="bad",
            diagnostics={},
        )


def test_joint_tiling_schedule_metadata_helpers_validate_runtime_types():
    ctx = CompileContext(graph=Graph("typed_joint_ctx"), target="x86")
    problem = _sample_problem()
    solution = _sample_solution()
    failure = _sample_failure()

    set_joint_tiling_schedule_problem(ctx, problem)
    set_joint_tiling_schedule_solution(ctx, solution)
    set_joint_tiling_schedule_failure(ctx, failure)

    assert get_joint_tiling_schedule_problem(ctx) is problem
    assert get_joint_tiling_schedule_solution(ctx) is solution
    assert get_joint_tiling_schedule_failure(ctx) is failure
    assert ctx.joint_tiling_schedule_problem is problem
    assert ctx.joint_tiling_schedule_solution is solution
    assert ctx.joint_tiling_schedule_failure is failure
    assert ctx.get_joint_tiling_schedule_problem() is problem
    assert ctx.get_joint_tiling_schedule_solution() is solution
    assert ctx.get_joint_tiling_schedule_failure() is failure

    with pytest.raises(TypeError):
        set_joint_tiling_schedule_problem(ctx, object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        set_joint_tiling_schedule_solution(ctx, object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        set_joint_tiling_schedule_failure(ctx, object())  # type: ignore[arg-type]

    ctx.metadata["joint_tiling_schedule_problem"] = "bad"
    ctx.metadata["joint_tiling_schedule_solution"] = 1
    ctx.metadata["joint_tiling_schedule_failure"] = []

    with pytest.raises(TypeError):
        get_joint_tiling_schedule_problem(ctx)

    with pytest.raises(TypeError):
        get_joint_tiling_schedule_solution(ctx)

    with pytest.raises(TypeError):
        get_joint_tiling_schedule_failure(ctx)


def test_joint_tiling_schedule_accessors_do_not_mutate_missing_metadata_on_read():
    ctx = CompileContext(graph=Graph("empty_joint_ctx"), target="x86")

    assert "joint_tiling_schedule_problem" not in ctx.metadata
    assert "joint_tiling_schedule_solution" not in ctx.metadata
    assert "joint_tiling_schedule_failure" not in ctx.metadata
    assert get_joint_tiling_schedule_problem(ctx) is None
    assert get_joint_tiling_schedule_solution(ctx) is None
    assert get_joint_tiling_schedule_failure(ctx) is None
    assert ctx.joint_tiling_schedule_problem is None
    assert ctx.joint_tiling_schedule_solution is None
    assert ctx.joint_tiling_schedule_failure is None
    assert ctx.get_joint_tiling_schedule_problem() is None
    assert ctx.get_joint_tiling_schedule_solution() is None
    assert ctx.get_joint_tiling_schedule_failure() is None
    assert "joint_tiling_schedule_problem" not in ctx.metadata
    assert "joint_tiling_schedule_solution" not in ctx.metadata
    assert "joint_tiling_schedule_failure" not in ctx.metadata
