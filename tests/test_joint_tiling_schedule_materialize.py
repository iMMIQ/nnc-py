from __future__ import annotations

from nnc_py.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
    JointAction,
    JointBoundaryConstraint,
    JointCompatibleRecipePair,
    JointCostParameters,
    JointDependencyEdge,
    JointLayoutSpec,
    JointProblem,
    JointRecipe,
    JointRegion,
    JointResidencyWindow,
    JointResource,
    JointScheduledAction,
    JointSelectedRecipe,
    JointSolution,
    JointTileSpec,
    JointValue,
    JointValueConsumer,
    JointValueFootprint,
    JointValueProducer,
)
from nnc_py.ir.pipeline_schedule import PipelineScheduleProblem, PipelineScheduleResult
from nnc_py.joint_schedule.materialize import materialize_joint_solution
from nnc_py.joint_schedule.solver import BaselineJointScheduleSolver
from nnc_py.joint_schedule.validation import validate_joint_solution


def _valid_joint_problem() -> JointProblem:
    return JointProblem(
        schema_version=JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
        regions=(
            JointRegion(
                region_id="r0",
                kind="single_op",
                input_value_ids=("input0",),
                output_value_ids=("out",),
            ),
        ),
        recipes=(
            JointRecipe(
                recipe_id="r0.recipe0",
                region_id="r0",
                tile_spec=JointTileSpec(axes=("h", "w"), shape=(8, 8)),
                layout_spec=JointLayoutSpec(layout_tags=("nchw",)),
                activates_action_ids=("r0.dma_in", "r0.compute", "r0.dma_out"),
                value_footprint=JointValueFootprint(
                    resident_bytes=96,
                    scratch_bytes=8,
                    transfer_bytes=64,
                ),
                cost_parameters=JointCostParameters(latency=5, launch_overhead=0),
            ),
        ),
        values=(
            JointValue(
                value_id="input0",
                size_bytes=32,
                initial_tier="input",
                required_final_tier="input",
                must_keep=False,
                spillable=False,
                allows_multiple_sram_windows=False,
                producer=None,
                consumers=(
                    JointValueConsumer(action_id="r0.dma_in"),
                    JointValueConsumer(action_id="r0.compute"),
                ),
            ),
            JointValue(
                value_id="out",
                size_bytes=64,
                initial_tier="unmaterialized",
                required_final_tier="slow",
                must_keep=False,
                spillable=False,
                allows_multiple_sram_windows=False,
                producer=JointValueProducer(action_id="r0.compute"),
                consumers=(JointValueConsumer(action_id="r0.dma_out"),),
            ),
        ),
        actions=(
            JointAction(
                action_id="r0.dma_in",
                kind="dma_in",
                resource_kind="DMA",
                duration=1,
                launch_overhead=0,
                reads=("input0",),
                writes=("input0",),
                temp_bytes=0,
                is_optional=False,
                region_id="r0",
                recipe_id="r0.recipe0",
                optional_value_id=None,
            ),
            JointAction(
                action_id="r0.compute",
                kind="compute",
                resource_kind="MATMUL",
                duration=3,
                launch_overhead=0,
                reads=("input0",),
                writes=("out",),
                temp_bytes=8,
                is_optional=False,
                region_id="r0",
                recipe_id="r0.recipe0",
                optional_value_id=None,
            ),
            JointAction(
                action_id="r0.dma_out",
                kind="dma_out",
                resource_kind="DMA",
                duration=1,
                launch_overhead=0,
                reads=("out",),
                writes=("out",),
                temp_bytes=0,
                is_optional=False,
                region_id="r0",
                recipe_id="r0.recipe0",
                optional_value_id=None,
            ),
        ),
        boundary_constraints=(),
        dependency_edges=(
            JointDependencyEdge(src_action_id="r0.dma_in", dst_action_id="r0.compute", kind="data"),
            JointDependencyEdge(src_action_id="r0.compute", dst_action_id="r0.dma_out", kind="data"),
        ),
        resources=(
            JointResource(resource_kind="DMA", slot_count=1),
            JointResource(resource_kind="MATMUL", slot_count=1),
            JointResource(resource_kind="SHAPE", slot_count=1),
            JointResource(resource_kind="OTHER", slot_count=1),
        ),
        sram_capacity_bytes=128,
        objective="min_makespan",
    )


def _valid_joint_solution() -> JointSolution:
    return JointSolution(
        schema_version=JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
        selected_recipes=(JointSelectedRecipe(region_id="r0", recipe_id="r0.recipe0"),),
        scheduled_actions=(
            JointScheduledAction(action_id="r0.dma_in", start_time=0),
            JointScheduledAction(action_id="r0.compute", start_time=1),
            JointScheduledAction(action_id="r0.dma_out", start_time=4),
        ),
        residency_windows=(
            JointResidencyWindow(value_id="input0", start_time=1, end_time=4),
            JointResidencyWindow(value_id="out", start_time=4, end_time=5),
        ),
        objective_value=5,
        diagnostics={},
    )


def _initial_sram_problem() -> JointProblem:
    return JointProblem(
        schema_version=JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
        regions=(
            JointRegion(
                region_id="r0",
                kind="single_op",
                input_value_ids=("resident0",),
                output_value_ids=("out",),
            ),
        ),
        recipes=(
            JointRecipe(
                recipe_id="r0.recipe0",
                region_id="r0",
                tile_spec=JointTileSpec(axes=("h", "w"), shape=(4, 4)),
                layout_spec=JointLayoutSpec(layout_tags=("nchw",)),
                activates_action_ids=("r0.compute", "r0.dma_out"),
                value_footprint=JointValueFootprint(
                    resident_bytes=96,
                    scratch_bytes=4,
                    transfer_bytes=32,
                ),
                cost_parameters=JointCostParameters(latency=4, launch_overhead=0),
            ),
        ),
        values=(
            JointValue(
                value_id="resident0",
                size_bytes=32,
                initial_tier="sram",
                required_final_tier="sram",
                must_keep=False,
                spillable=False,
                allows_multiple_sram_windows=False,
                producer=None,
                consumers=(JointValueConsumer(action_id="r0.compute"),),
            ),
            JointValue(
                value_id="out",
                size_bytes=32,
                initial_tier="unmaterialized",
                required_final_tier="slow",
                must_keep=False,
                spillable=False,
                allows_multiple_sram_windows=False,
                producer=JointValueProducer(action_id="r0.compute"),
                consumers=(JointValueConsumer(action_id="r0.dma_out"),),
            ),
        ),
        actions=(
            JointAction(
                action_id="r0.compute",
                kind="compute",
                resource_kind="MATMUL",
                duration=2,
                launch_overhead=0,
                reads=("resident0",),
                writes=("out",),
                temp_bytes=4,
                is_optional=False,
                region_id="r0",
                recipe_id="r0.recipe0",
                optional_value_id=None,
            ),
            JointAction(
                action_id="r0.dma_out",
                kind="dma_out",
                resource_kind="DMA",
                duration=1,
                launch_overhead=0,
                reads=("out",),
                writes=("out",),
                temp_bytes=0,
                is_optional=False,
                region_id="r0",
                recipe_id="r0.recipe0",
                optional_value_id=None,
            ),
        ),
        boundary_constraints=(),
        dependency_edges=(
            JointDependencyEdge(src_action_id="r0.compute", dst_action_id="r0.dma_out", kind="data"),
        ),
        resources=(
            JointResource(resource_kind="DMA", slot_count=1),
            JointResource(resource_kind="MATMUL", slot_count=1),
            JointResource(resource_kind="SHAPE", slot_count=1),
            JointResource(resource_kind="OTHER", slot_count=1),
        ),
        sram_capacity_bytes=128,
        objective="min_makespan",
    )


def _initial_sram_solution() -> JointSolution:
    return JointSolution(
        schema_version=JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
        selected_recipes=(JointSelectedRecipe(region_id="r0", recipe_id="r0.recipe0"),),
        scheduled_actions=(
            JointScheduledAction(action_id="r0.compute", start_time=0),
            JointScheduledAction(action_id="r0.dma_out", start_time=2),
        ),
        residency_windows=(
            JointResidencyWindow(value_id="resident0", start_time=0, end_time=3),
            JointResidencyWindow(value_id="out", start_time=2, end_time=3),
        ),
        objective_value=3,
        diagnostics={},
    )


def _spill_reload_problem() -> JointProblem:
    return JointProblem(
        schema_version=JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
        regions=(
            JointRegion(region_id="r0", kind="single_op", input_value_ids=(), output_value_ids=("mid",)),
            JointRegion(region_id="r1", kind="single_op", input_value_ids=("mid",), output_value_ids=("out",)),
        ),
        recipes=(
            JointRecipe(
                recipe_id="r0.recipe0",
                region_id="r0",
                tile_spec=JointTileSpec(axes=("h", "w"), shape=(4, 4)),
                layout_spec=JointLayoutSpec(layout_tags=("nchw",)),
                activates_action_ids=("r0.compute",),
                value_footprint=JointValueFootprint(resident_bytes=32, scratch_bytes=4, transfer_bytes=0),
                cost_parameters=JointCostParameters(latency=3, launch_overhead=0),
            ),
            JointRecipe(
                recipe_id="r1.recipe0",
                region_id="r1",
                tile_spec=JointTileSpec(axes=("h", "w"), shape=(4, 4)),
                layout_spec=JointLayoutSpec(layout_tags=("nchw",)),
                activates_action_ids=("r1.compute", "r1.dma_out"),
                value_footprint=JointValueFootprint(resident_bytes=32, scratch_bytes=4, transfer_bytes=16),
                cost_parameters=JointCostParameters(latency=4, launch_overhead=0),
            ),
        ),
        values=(
            JointValue(
                value_id="mid",
                size_bytes=32,
                initial_tier="unmaterialized",
                required_final_tier="slow",
                must_keep=False,
                spillable=True,
                allows_multiple_sram_windows=True,
                producer=JointValueProducer(action_id="r0.compute"),
                consumers=(
                    JointValueConsumer(action_id="mid.spill"),
                    JointValueConsumer(action_id="mid.reload"),
                    JointValueConsumer(action_id="r1.compute"),
                ),
            ),
            JointValue(
                value_id="out",
                size_bytes=16,
                initial_tier="unmaterialized",
                required_final_tier="slow",
                must_keep=False,
                spillable=False,
                allows_multiple_sram_windows=False,
                producer=JointValueProducer(action_id="r1.compute"),
                consumers=(JointValueConsumer(action_id="r1.dma_out"),),
            ),
        ),
        actions=(
            JointAction(
                action_id="r0.compute",
                kind="compute",
                resource_kind="MATMUL",
                duration=3,
                launch_overhead=0,
                reads=(),
                writes=("mid",),
                temp_bytes=4,
                is_optional=False,
                region_id="r0",
                recipe_id="r0.recipe0",
                optional_value_id=None,
            ),
            JointAction(
                action_id="r1.compute",
                kind="compute",
                resource_kind="MATMUL",
                duration=3,
                launch_overhead=0,
                reads=("mid",),
                writes=("out",),
                temp_bytes=4,
                is_optional=False,
                region_id="r1",
                recipe_id="r1.recipe0",
                optional_value_id=None,
            ),
            JointAction(
                action_id="r1.dma_out",
                kind="dma_out",
                resource_kind="DMA",
                duration=1,
                launch_overhead=0,
                reads=("out",),
                writes=("out",),
                temp_bytes=0,
                is_optional=False,
                region_id="r1",
                recipe_id="r1.recipe0",
                optional_value_id=None,
            ),
            JointAction(
                action_id="mid.spill",
                kind="spill",
                resource_kind="DMA",
                duration=1,
                launch_overhead=0,
                reads=("mid",),
                writes=("mid",),
                temp_bytes=0,
                is_optional=True,
                region_id=None,
                recipe_id=None,
                optional_value_id="mid",
            ),
            JointAction(
                action_id="mid.reload",
                kind="reload",
                resource_kind="DMA",
                duration=1,
                launch_overhead=0,
                reads=("mid",),
                writes=("mid",),
                temp_bytes=0,
                is_optional=True,
                region_id=None,
                recipe_id=None,
                optional_value_id="mid",
            ),
        ),
        boundary_constraints=(
            JointBoundaryConstraint(
                boundary_id="r0->r1",
                src_region_id="r0",
                dst_region_id="r1",
                compatible_recipe_pairs=(JointCompatibleRecipePair(src_recipe_id="r0.recipe0", dst_recipe_id="r1.recipe0"),),
            ),
        ),
        dependency_edges=(
            JointDependencyEdge(src_action_id="r0.compute", dst_action_id="r1.compute", kind="data"),
            JointDependencyEdge(src_action_id="mid.spill", dst_action_id="mid.reload", kind="order"),
            JointDependencyEdge(src_action_id="mid.reload", dst_action_id="r1.compute", kind="order"),
            JointDependencyEdge(src_action_id="r1.compute", dst_action_id="r1.dma_out", kind="data"),
        ),
        resources=(
            JointResource(resource_kind="DMA", slot_count=1),
            JointResource(resource_kind="MATMUL", slot_count=1),
            JointResource(resource_kind="SHAPE", slot_count=1),
            JointResource(resource_kind="OTHER", slot_count=1),
        ),
        sram_capacity_bytes=128,
        objective="min_makespan",
    )


def _spill_reload_solution() -> JointSolution:
    return JointSolution(
        schema_version=JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
        selected_recipes=(
            JointSelectedRecipe(region_id="r0", recipe_id="r0.recipe0"),
            JointSelectedRecipe(region_id="r1", recipe_id="r1.recipe0"),
        ),
        scheduled_actions=(
            JointScheduledAction(action_id="r0.compute", start_time=0),
            JointScheduledAction(action_id="mid.spill", start_time=3),
            JointScheduledAction(action_id="mid.reload", start_time=5),
            JointScheduledAction(action_id="r1.compute", start_time=6),
            JointScheduledAction(action_id="r1.dma_out", start_time=9),
        ),
        residency_windows=(
            JointResidencyWindow(value_id="mid", start_time=3, end_time=4),
            JointResidencyWindow(value_id="mid", start_time=6, end_time=9),
            JointResidencyWindow(value_id="out", start_time=9, end_time=10),
        ),
        objective_value=10,
        diagnostics={},
    )

def test_materialize_joint_solution_returns_internal_schedule_pair():
    problem, result = materialize_joint_solution(
        _valid_joint_problem(),
        _valid_joint_solution(),
    )

    assert isinstance(problem, PipelineScheduleProblem)
    assert isinstance(result, PipelineScheduleResult)
    assert result.feasible is True
    assert tuple(step.id for step in problem.steps) == (
        "r0.dma_in",
        "r0.compute",
        "r0.dma_out",
    )
    assert tuple(step.step_id for step in result.scheduled_steps) == (
        "r0.dma_in",
        "r0.compute",
        "r0.dma_out",
    )
    scheduled_values = {value.name: value for value in problem.scheduled_values}
    assert scheduled_values["input0"].home_tier.value == "input"
    assert scheduled_values["input0.resident@1"].home_tier.value == "sram"
    assert scheduled_values["out"].home_tier.value == "slow"
    assert scheduled_values["out.resident@4"].home_tier.value == "sram"

    steps = {step.id: step for step in problem.steps}
    assert steps["r0.dma_in"].sram_input_names == ()
    assert steps["r0.dma_in"].sram_output_names == ("input0.resident@1",)
    assert steps["r0.compute"].sram_input_names == ("input0.resident@1",)
    assert steps["r0.compute"].sram_output_names == ("out.resident@4",)
    assert steps["r0.dma_out"].sram_input_names == ("out.resident@4",)
    assert steps["r0.dma_out"].sram_output_names == ()


def test_baseline_solver_returns_solution_shape():
    problem = _valid_joint_problem()

    result = BaselineJointScheduleSolver().solve(problem)

    assert isinstance(result, JointSolution)
    assert result.schema_version == JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION
    assert result.selected_recipes
    assert validate_joint_solution(problem, result) is None


def test_materialize_joint_solution_supports_initial_sram_windows():
    problem, result = materialize_joint_solution(
        _initial_sram_problem(),
        _initial_sram_solution(),
    )

    assert isinstance(problem, PipelineScheduleProblem)
    assert isinstance(result, PipelineScheduleResult)
    assert result.feasible is True


def test_baseline_solver_emits_initial_sram_window_when_required():
    problem = _initial_sram_problem()

    result = BaselineJointScheduleSolver().solve(problem)

    assert isinstance(result, JointSolution)
    assert any(
        window.value_id == "resident0" and window.start_time == 0
        for window in result.residency_windows
    )
    assert validate_joint_solution(problem, result) is None


def test_materialize_joint_solution_uses_base_value_name_for_reload_move():
    problem, _ = materialize_joint_solution(
        _spill_reload_problem(),
        _spill_reload_solution(),
    )

    steps = {step.id: step for step in problem.steps}
    scheduled_values = {value.name: value for value in problem.scheduled_values}
    assert steps["mid.reload"].moved_value_name == "mid.resident@3"
    assert steps["mid.reload"].sram_output_names == ("mid.resident@6",)
    assert "mid.reload" not in scheduled_values["mid"].consumer_step_ids
