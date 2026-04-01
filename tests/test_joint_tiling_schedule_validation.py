from __future__ import annotations

from dataclasses import replace

from nnc_py.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
    JointAction,
    JointBoundaryConstraint,
    JointCompatibleRecipePair,
    JointCostParameters,
    JointDependencyEdge,
    JointFailureCategory,
    JointFailureStatus,
    JointLayoutSpec,
    JointProblem,
    JointRecipe,
    JointRegion,
    JointResidencyWindow,
    JointResource,
    JointScheduledAction,
    JointSelectedRecipe,
    JointSolution,
    JointSramAllocation,
    JointSramItem,
    JointSramItemKind,
    JointTileSpec,
    JointValue,
    JointValueConsumer,
    JointValueFootprint,
    JointValueProducer,
)
from nnc_py.joint_schedule.validation import (
    validate_joint_problem,
    validate_joint_solution,
)


def _window(value_id: str, start_time: int, end_time: int) -> JointResidencyWindow:
    return JointResidencyWindow(
        residency_id=f"{value_id}@{start_time}",
        value_id=value_id,
        start_time=start_time,
        end_time=end_time,
    )


def _generated_resident_items(
    problem: JointProblem,
    windows: tuple[JointResidencyWindow, ...],
) -> tuple[JointSramItem, ...]:
    size_by_value = {value.value_id: value.size_bytes for value in problem.values}
    return tuple(
        JointSramItem(
            item_id=f"{window.residency_id}.item",
            kind=JointSramItemKind.RESIDENT_WINDOW,
            size_bytes=size_by_value[window.value_id],
            alignment_bytes=problem.default_alignment_bytes,
            is_optional=False,
            owner_action_id=None,
            owner_value_id=window.value_id,
            owner_residency_id=window.residency_id,
        )
        for window in windows
    )


def _valid_problem() -> JointProblem:
    return JointProblem(
        schema_version=JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
        regions=(
            JointRegion(
                region_id="r0",
                kind="single_op",
                input_value_ids=("input0",),
                output_value_ids=("mid",),
            ),
            JointRegion(
                region_id="r1",
                kind="single_op",
                input_value_ids=("mid",),
                output_value_ids=("out",),
            ),
        ),
        recipes=(
            JointRecipe(
                recipe_id="r0.recipe0",
                region_id="r0",
                tile_spec=JointTileSpec(axes=("h", "w"), shape=(8, 8)),
                layout_spec=JointLayoutSpec(layout_tags=("nchw",)),
                activates_action_ids=("r0.dma_in", "r0.compute"),
                value_footprint=JointValueFootprint(
                    resident_bytes=128,
                    scratch_bytes=16,
                    transfer_bytes=64,
                ),
                cost_parameters=JointCostParameters(latency=8, launch_overhead=2),
            ),
            JointRecipe(
                recipe_id="r1.recipe0",
                region_id="r1",
                tile_spec=JointTileSpec(axes=("h", "w"), shape=(8, 8)),
                layout_spec=JointLayoutSpec(layout_tags=("nchw",)),
                activates_action_ids=("r1.compute", "r1.dma_out"),
                value_footprint=JointValueFootprint(
                    resident_bytes=128,
                    scratch_bytes=8,
                    transfer_bytes=64,
                ),
                cost_parameters=JointCostParameters(latency=6, launch_overhead=2),
            ),
            JointRecipe(
                recipe_id="r1.recipe1",
                region_id="r1",
                tile_spec=JointTileSpec(axes=("h", "w"), shape=(4, 4)),
                layout_spec=JointLayoutSpec(layout_tags=("nhwc",)),
                activates_action_ids=("r1.alt_compute", "r1.alt_dma_out"),
                value_footprint=JointValueFootprint(
                    resident_bytes=96,
                    scratch_bytes=8,
                    transfer_bytes=64,
                ),
                cost_parameters=JointCostParameters(latency=5, launch_overhead=2),
            ),
        ),
        values=(
            JointValue(
                value_id="input0",
                size_bytes=64,
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
                value_id="mid",
                size_bytes=96,
                initial_tier="unmaterialized",
                required_final_tier="sram",
                must_keep=False,
                spillable=True,
                allows_multiple_sram_windows=True,
                producer=JointValueProducer(action_id="r0.compute"),
                consumers=(
                    JointValueConsumer(action_id="r1.compute"),
                    JointValueConsumer(action_id="r1.alt_compute"),
                    JointValueConsumer(action_id="mid.spill"),
                    JointValueConsumer(action_id="mid.reload"),
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
                producer=JointValueProducer(action_id="r1.compute"),
                consumers=(JointValueConsumer(action_id="r1.dma_out"),),
            ),
            JointValue(
                value_id="out_alt",
                size_bytes=64,
                initial_tier="unmaterialized",
                required_final_tier="slow",
                must_keep=False,
                spillable=False,
                allows_multiple_sram_windows=False,
                producer=JointValueProducer(action_id="r1.alt_compute"),
                consumers=(JointValueConsumer(action_id="r1.alt_dma_out"),),
            ),
        ),
        actions=(
            JointAction(
                action_id="r0.dma_in",
                kind="dma_in",
                resource_kind="DMA",
                duration=2,
                launch_overhead=1,
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
                duration=5,
                launch_overhead=1,
                reads=("input0",),
                writes=("mid",),
                temp_bytes=16,
                is_optional=False,
                region_id="r0",
                recipe_id="r0.recipe0",
                optional_value_id=None,
            ),
            JointAction(
                action_id="r1.compute",
                kind="compute",
                resource_kind="MATMUL",
                duration=4,
                launch_overhead=1,
                reads=("mid",),
                writes=("out",),
                temp_bytes=8,
                is_optional=False,
                region_id="r1",
                recipe_id="r1.recipe0",
                optional_value_id=None,
            ),
            JointAction(
                action_id="r1.dma_out",
                kind="dma_out",
                resource_kind="DMA",
                duration=2,
                launch_overhead=1,
                reads=("out",),
                writes=("out",),
                temp_bytes=0,
                is_optional=False,
                region_id="r1",
                recipe_id="r1.recipe0",
                optional_value_id=None,
            ),
            JointAction(
                action_id="r1.alt_compute",
                kind="compute",
                resource_kind="MATMUL",
                duration=3,
                launch_overhead=1,
                reads=("mid",),
                writes=("out_alt",),
                temp_bytes=8,
                is_optional=False,
                region_id="r1",
                recipe_id="r1.recipe1",
                optional_value_id=None,
            ),
            JointAction(
                action_id="r1.alt_dma_out",
                kind="dma_out",
                resource_kind="DMA",
                duration=2,
                launch_overhead=1,
                reads=("out_alt",),
                writes=("out_alt",),
                temp_bytes=0,
                is_optional=False,
                region_id="r1",
                recipe_id="r1.recipe1",
                optional_value_id=None,
            ),
            JointAction(
                action_id="mid.spill",
                kind="spill",
                resource_kind="DMA",
                duration=2,
                launch_overhead=1,
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
                duration=2,
                launch_overhead=1,
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
                compatible_recipe_pairs=(
                    JointCompatibleRecipePair(
                        src_recipe_id="r0.recipe0",
                        dst_recipe_id="r1.recipe0",
                    ),
                ),
            ),
        ),
        dependency_edges=(
            JointDependencyEdge(
                src_action_id="r0.dma_in",
                dst_action_id="r0.compute",
                kind="data",
            ),
            JointDependencyEdge(
                src_action_id="r0.compute",
                dst_action_id="r1.compute",
                kind="data",
            ),
            JointDependencyEdge(
                src_action_id="r1.compute",
                dst_action_id="r1.dma_out",
                kind="data",
            ),
            JointDependencyEdge(
                src_action_id="mid.spill",
                dst_action_id="mid.reload",
                kind="order",
            ),
            JointDependencyEdge(
                src_action_id="mid.reload",
                dst_action_id="r1.compute",
                kind="order",
            ),
        ),
        resources=(
            JointResource(resource_kind="DMA", slot_count=1),
            JointResource(resource_kind="MATMUL", slot_count=1),
            JointResource(resource_kind="SHAPE", slot_count=1),
            JointResource(resource_kind="OTHER", slot_count=1),
        ),
        sram_capacity_bytes=200,
        sram_items=(
            JointSramItem(
                item_id="r0.compute.temp",
                kind=JointSramItemKind.TEMP_INTERVAL,
                size_bytes=16,
                alignment_bytes=16,
                is_optional=False,
                owner_action_id="r0.compute",
                owner_value_id=None,
                owner_residency_id=None,
            ),
            JointSramItem(
                item_id="r1.compute.temp",
                kind=JointSramItemKind.TEMP_INTERVAL,
                size_bytes=8,
                alignment_bytes=16,
                is_optional=False,
                owner_action_id="r1.compute",
                owner_value_id=None,
                owner_residency_id=None,
            ),
            JointSramItem(
                item_id="r1.alt_compute.temp",
                kind=JointSramItemKind.TEMP_INTERVAL,
                size_bytes=8,
                alignment_bytes=16,
                is_optional=False,
                owner_action_id="r1.alt_compute",
                owner_value_id=None,
                owner_residency_id=None,
            ),
        ),
        default_alignment_bytes=16,
        objective="min_makespan",
    )


def _valid_solution() -> JointSolution:
    problem = _valid_problem()
    residency_windows = (
        _window("input0", 3, 9),
        _window("mid", 9, 17),
        _window("out", 14, 17),
    )
    return JointSolution(
        schema_version=JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
        selected_recipes=(
            JointSelectedRecipe(region_id="r0", recipe_id="r0.recipe0"),
            JointSelectedRecipe(region_id="r1", recipe_id="r1.recipe0"),
        ),
        scheduled_actions=(
            JointScheduledAction(action_id="r0.dma_in", start_time=0),
            JointScheduledAction(action_id="r0.compute", start_time=3),
            JointScheduledAction(action_id="r1.compute", start_time=9),
            JointScheduledAction(action_id="r1.dma_out", start_time=14),
        ),
        residency_windows=residency_windows,
        generated_sram_items=_generated_resident_items(problem, residency_windows),
        sram_allocations=(
            JointSramAllocation(item_id="r0.compute.temp", offset=0),
            JointSramAllocation(item_id="r1.compute.temp", offset=0),
            JointSramAllocation(item_id="input0@3.item", offset=16),
            JointSramAllocation(item_id="mid@9.item", offset=16),
            JointSramAllocation(item_id="out@14.item", offset=112),
        ),
        objective_value=17,
        diagnostics={},
    )


def _problem_with_transfer_buffer() -> JointProblem:
    problem = _valid_problem()
    return replace(
        problem,
        sram_items=problem.sram_items
        + (
            JointSramItem(
                item_id="r0.compute.pack",
                kind=JointSramItemKind.TRANSFER_BUFFER,
                size_bytes=32,
                alignment_bytes=16,
                is_optional=False,
                owner_action_id="r0.dma_in",
                owner_value_id=None,
                owner_residency_id=None,
            ),
        ),
    )


def _solution_with_transfer_buffer() -> JointSolution:
    solution = _valid_solution()
    return replace(
        solution,
        sram_allocations=solution.sram_allocations
        + (JointSramAllocation(item_id="r0.compute.pack", offset=80),),
    )


def test_validate_joint_problem_accepts_valid_problem():
    assert validate_joint_problem(_valid_problem()) is None


def test_validator_rejects_missing_boundary_constraint():
    problem = replace(_valid_problem(), boundary_constraints=())

    failure = validate_joint_problem(problem)

    assert failure is not None
    assert failure.schema_version == JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION
    assert failure.status is JointFailureStatus.INVALID_PROBLEM


def test_validator_rejects_duplicate_boundary_pairs():
    problem = replace(
        _valid_problem(),
        boundary_constraints=_valid_problem().boundary_constraints
        + (
            JointBoundaryConstraint(
                boundary_id="r0->r1.dup",
                src_region_id="r0",
                dst_region_id="r1",
                compatible_recipe_pairs=(
                    JointCompatibleRecipePair(
                        src_recipe_id="r0.recipe0",
                        dst_recipe_id="r1.recipe0",
                    ),
                ),
            ),
        ),
    )

    failure = validate_joint_problem(problem)

    assert failure is not None
    assert failure.status is JointFailureStatus.INVALID_PROBLEM


def test_validator_rejects_dangling_recipe_activation_ids():
    problem = replace(
        _valid_problem(),
        recipes=tuple(
            replace(recipe, activates_action_ids=recipe.activates_action_ids + ("ghost.action",))
            if recipe.recipe_id == "r0.recipe0"
            else recipe
            for recipe in _valid_problem().recipes
        ),
    )

    failure = validate_joint_problem(problem)

    assert failure is not None
    assert failure.status is JointFailureStatus.INVALID_PROBLEM


def test_validator_rejects_unknown_region_interface_value():
    problem = replace(
        _valid_problem(),
        regions=(
            replace(
                _valid_problem().regions[0],
                input_value_ids=("ghost",),
            ),
            _valid_problem().regions[1],
        ),
    )

    failure = validate_joint_problem(problem)

    assert failure is not None
    assert failure.status is JointFailureStatus.INVALID_PROBLEM


def test_validate_joint_solution_accepts_valid_solution():
    assert validate_joint_solution(_valid_problem(), _valid_solution()) is None


def test_validator_rejects_missing_reverse_consumer_metadata():
    problem = replace(
        _valid_problem(),
        values=tuple(
            replace(
                value,
                consumers=(JointValueConsumer(action_id="r0.dma_in"),),
            )
            if value.value_id == "input0"
            else value
            for value in _valid_problem().values
        ),
    )

    failure = validate_joint_problem(problem)

    assert failure is not None
    assert failure.status is JointFailureStatus.INVALID_PROBLEM


def test_validator_rejects_mandatory_action_region_recipe_mismatch():
    problem = replace(
        _valid_problem(),
        actions=tuple(
            replace(action, region_id="r0")
            if action.action_id == "r1.compute"
            else action
            for action in _valid_problem().actions
        ),
    )

    failure = validate_joint_problem(problem)

    assert failure is not None
    assert failure.status is JointFailureStatus.INVALID_PROBLEM


def test_validator_rejects_region_interface_action_mismatch():
    problem = replace(
        _valid_problem(),
        regions=(
            replace(
                _valid_problem().regions[0],
                input_value_ids=("out",),
            ),
            _valid_problem().regions[1],
        ),
    )

    failure = validate_joint_problem(problem)

    assert failure is not None
    assert failure.status is JointFailureStatus.INVALID_PROBLEM


def test_validator_rejects_optional_value_reference_to_unknown_value():
    problem = replace(
        _valid_problem(),
        actions=tuple(
            replace(action, optional_value_id="ghost")
            if action.action_id == "mid.reload"
            else action
            for action in _valid_problem().actions
        ),
    )

    failure = validate_joint_problem(problem)

    assert failure is not None
    assert failure.status is JointFailureStatus.INVALID_PROBLEM


def test_validator_rejects_overlapping_same_resource_actions():
    bad_solution = replace(
        _valid_solution(),
        scheduled_actions=(
            JointScheduledAction(action_id="r0.dma_in", start_time=0),
            JointScheduledAction(action_id="r0.compute", start_time=3),
            JointScheduledAction(action_id="r1.compute", start_time=6),
            JointScheduledAction(action_id="r1.dma_out", start_time=14),
        ),
    )

    failure = validate_joint_solution(_valid_problem(), bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.RESOURCE_OVERLAP


def test_validator_rejects_dependency_edges_with_unknown_actions():
    problem = replace(
        _valid_problem(),
        dependency_edges=_valid_problem().dependency_edges
        + (
            JointDependencyEdge(
                src_action_id="ghost.src",
                dst_action_id="r1.compute",
                kind="data",
            ),
        ),
    )

    failure = validate_joint_problem(problem)

    assert failure is not None
    assert failure.status is JointFailureStatus.INVALID_PROBLEM


def test_validator_rejects_incompatible_recipe_boundary():
    problem = _valid_problem()
    residency_windows = (
        _window("input0", 3, 9),
        _window("mid", 9, 16),
        _window("out_alt", 13, 16),
    )
    bad_solution = replace(
        _valid_solution(),
        selected_recipes=(
            JointSelectedRecipe(region_id="r0", recipe_id="r0.recipe0"),
            JointSelectedRecipe(region_id="r1", recipe_id="r1.recipe1"),
        ),
        scheduled_actions=(
            JointScheduledAction(action_id="r0.dma_in", start_time=0),
            JointScheduledAction(action_id="r0.compute", start_time=3),
            JointScheduledAction(action_id="r1.alt_compute", start_time=9),
            JointScheduledAction(action_id="r1.alt_dma_out", start_time=13),
        ),
        residency_windows=residency_windows,
        generated_sram_items=_generated_resident_items(problem, residency_windows),
        sram_allocations=(
            JointSramAllocation(item_id="r0.compute.temp", offset=0),
            JointSramAllocation(item_id="r1.alt_compute.temp", offset=0),
            JointSramAllocation(item_id="input0@3.item", offset=16),
            JointSramAllocation(item_id="mid@9.item", offset=16),
            JointSramAllocation(item_id="out_alt@13.item", offset=112),
        ),
        objective_value=16,
    )

    failure = validate_joint_solution(problem, bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.INCOMPATIBLE_RECIPE_BOUNDARY


def test_validator_rejects_boundary_pairs_bound_to_wrong_regions():
    problem = replace(
        _valid_problem(),
        boundary_constraints=(
            JointBoundaryConstraint(
                boundary_id="r0->r1",
                src_region_id="r0",
                dst_region_id="r1",
                compatible_recipe_pairs=(
                    JointCompatibleRecipePair(
                        src_recipe_id="r1.recipe0",
                        dst_recipe_id="r0.recipe0",
                    ),
                ),
            ),
        ),
    )

    failure = validate_joint_problem(problem)

    assert failure is not None
    assert failure.status is JointFailureStatus.INVALID_PROBLEM


def test_validator_rejects_missing_final_output():
    problem = _valid_problem()
    residency_windows = (
        _window("input0", 3, 9),
        _window("mid", 9, 14),
    )
    bad_solution = replace(
        _valid_solution(),
        scheduled_actions=(
            JointScheduledAction(action_id="r0.dma_in", start_time=0),
            JointScheduledAction(action_id="r0.compute", start_time=3),
            JointScheduledAction(action_id="r1.compute", start_time=9),
        ),
        residency_windows=residency_windows,
        generated_sram_items=_generated_resident_items(problem, residency_windows),
        sram_allocations=(
            JointSramAllocation(item_id="r0.compute.temp", offset=0),
            JointSramAllocation(item_id="r1.compute.temp", offset=0),
            JointSramAllocation(item_id="input0@3.item", offset=16),
            JointSramAllocation(item_id="mid@9.item", offset=16),
        ),
        objective_value=14,
    )

    failure = validate_joint_solution(problem, bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.INCOMPLETE_SOLUTION


def test_validator_rejects_invalid_solution_shape():
    bad_solution = replace(
        _valid_solution(),
        scheduled_actions=_valid_solution().scheduled_actions
        + (JointScheduledAction(action_id="ghost.action", start_time=20),),
    )

    failure = validate_joint_solution(_valid_problem(), bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION


def test_validator_rejects_dependency_violation():
    problem = _valid_problem()
    residency_windows = (
        _window("input0", 3, 9),
        _window("mid", 9, 17),
        _window("out", 13, 16),
    )
    bad_solution = replace(
        _valid_solution(),
        scheduled_actions=(
            JointScheduledAction(action_id="r0.dma_in", start_time=0),
            JointScheduledAction(action_id="r0.compute", start_time=3),
            JointScheduledAction(action_id="r1.compute", start_time=9),
            JointScheduledAction(action_id="r1.dma_out", start_time=13),
        ),
        residency_windows=residency_windows,
        generated_sram_items=_generated_resident_items(problem, residency_windows),
        sram_allocations=(
            JointSramAllocation(item_id="r0.compute.temp", offset=0),
            JointSramAllocation(item_id="r1.compute.temp", offset=0),
            JointSramAllocation(item_id="input0@3.item", offset=16),
            JointSramAllocation(item_id="mid@9.item", offset=16),
            JointSramAllocation(item_id="out@13.item", offset=112),
        ),
        objective_value=16,
    )

    failure = validate_joint_solution(problem, bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.DEPENDENCY_VIOLATION


def test_validator_rejects_sram_capacity_exceeded():
    problem = replace(_valid_problem(), sram_capacity_bytes=150)

    failure = validate_joint_solution(problem, _valid_solution())

    assert failure is not None
    assert failure.error_category is JointFailureCategory.SRAM_CAPACITY_EXCEEDED


def test_validator_rejects_illegal_transfer():
    problem = _valid_problem()
    residency_windows = (
        _window("input0", 3, 9),
        _window("mid", 12, 20),
        _window("out", 17, 20),
    )
    bad_solution = replace(
        _valid_solution(),
        scheduled_actions=(
            JointScheduledAction(action_id="r0.dma_in", start_time=0),
            JointScheduledAction(action_id="r0.compute", start_time=3),
            JointScheduledAction(action_id="mid.reload", start_time=9),
            JointScheduledAction(action_id="r1.compute", start_time=12),
            JointScheduledAction(action_id="r1.dma_out", start_time=17),
        ),
        residency_windows=residency_windows,
        generated_sram_items=_generated_resident_items(problem, residency_windows),
        sram_allocations=(
            JointSramAllocation(item_id="r0.compute.temp", offset=0),
            JointSramAllocation(item_id="r1.compute.temp", offset=0),
            JointSramAllocation(item_id="input0@3.item", offset=16),
            JointSramAllocation(item_id="mid@12.item", offset=16),
            JointSramAllocation(item_id="out@17.item", offset=112),
        ),
        objective_value=20,
    )

    failure = validate_joint_solution(problem, bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.ILLEGAL_TRANSFER


def test_validator_rejects_unjustified_first_residency_window():
    problem = _valid_problem()
    residency_windows = (
        _window("input0", 3, 9),
        _window("mid", 0, 17),
        _window("out", 14, 17),
    )
    bad_solution = replace(
        _valid_solution(),
        residency_windows=residency_windows,
        generated_sram_items=_generated_resident_items(problem, residency_windows),
        sram_allocations=(
            JointSramAllocation(item_id="r0.compute.temp", offset=0),
            JointSramAllocation(item_id="r1.compute.temp", offset=0),
            JointSramAllocation(item_id="input0@3.item", offset=16),
            JointSramAllocation(item_id="mid@0.item", offset=16),
            JointSramAllocation(item_id="out@14.item", offset=112),
        ),
    )

    failure = validate_joint_solution(problem, bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION


def test_validator_rejects_must_keep_gap_from_first_availability():
    problem = replace(
        _valid_problem(),
        values=tuple(
            replace(value, must_keep=True, spillable=False)
            if value.value_id == "mid"
            else value
            for value in _valid_problem().values
        ),
    )
    residency_windows = (
        _window("input0", 3, 9),
        _window("mid", 10, 17),
        _window("out", 14, 17),
    )
    bad_solution = replace(
        _valid_solution(),
        residency_windows=residency_windows,
        generated_sram_items=_generated_resident_items(problem, residency_windows),
        sram_allocations=(
            JointSramAllocation(item_id="r0.compute.temp", offset=0),
            JointSramAllocation(item_id="r1.compute.temp", offset=0),
            JointSramAllocation(item_id="input0@3.item", offset=16),
            JointSramAllocation(item_id="mid@10.item", offset=16),
            JointSramAllocation(item_id="out@14.item", offset=112),
        ),
    )

    failure = validate_joint_solution(problem, bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION


def test_validator_rejects_missing_allocation_for_active_fixed_temp_item():
    bad_solution = replace(
        _valid_solution(),
        sram_allocations=tuple(
            allocation
            for allocation in _valid_solution().sram_allocations
            if allocation.item_id != "r1.compute.temp"
        ),
    )

    failure = validate_joint_solution(_valid_problem(), bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION
    assert "missing allocation" in str(failure.diagnostics["reason"])


def test_validator_rejects_allocation_that_exceeds_sram_capacity():
    bad_solution = replace(
        _valid_solution(),
        sram_allocations=tuple(
            replace(allocation, offset=144)
            if allocation.item_id == "out@14.item"
            else allocation
            for allocation in _valid_solution().sram_allocations
        ),
    )

    failure = validate_joint_solution(_valid_problem(), bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.SRAM_CAPACITY_EXCEEDED
    assert "exceeds SRAM capacity" in str(failure.diagnostics["reason"])


def test_validator_rejects_overlapping_allocations_for_time_overlapping_items():
    bad_solution = replace(
        _valid_solution(),
        sram_allocations=tuple(
            replace(allocation, offset=80)
            if allocation.item_id == "out@14.item"
            else allocation
            for allocation in _valid_solution().sram_allocations
        ),
    )

    failure = validate_joint_solution(_valid_problem(), bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION
    assert "overlap" in str(failure.diagnostics["reason"])


def test_validator_rejects_misaligned_allocation_offset():
    bad_solution = replace(
        _valid_solution(),
        sram_allocations=tuple(
            replace(allocation, offset=18)
            if allocation.item_id == "input0@3.item"
            else allocation
            for allocation in _valid_solution().sram_allocations
        ),
    )

    failure = validate_joint_solution(_valid_problem(), bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION
    assert "alignment" in str(failure.diagnostics["reason"])


def test_validator_rejects_resident_item_cardinality_mismatch():
    bad_solution = replace(
        _valid_solution(),
        generated_sram_items=tuple(
            item
            for item in _valid_solution().generated_sram_items
            if item.item_id != "out@14.item"
        ),
        sram_allocations=tuple(
            allocation
            for allocation in _valid_solution().sram_allocations
            if allocation.item_id != "out@14.item"
        ),
    )

    failure = validate_joint_solution(_valid_problem(), bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION
    assert "cardinality" in str(failure.diagnostics["reason"])


def test_validator_rejects_generated_resident_item_ownership_mismatch():
    bad_solution = replace(
        _valid_solution(),
        generated_sram_items=tuple(
            replace(item, owner_value_id="mid")
            if item.item_id == "out@14.item"
            else item
            for item in _valid_solution().generated_sram_items
        ),
    )

    failure = validate_joint_solution(_valid_problem(), bad_solution)

    assert failure is not None
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION
    assert "ownership" in str(failure.diagnostics["reason"])


def test_validator_rejects_temp_interval_owned_by_non_compute_action():
    bad_problem = replace(
        _valid_problem(),
        sram_items=tuple(
            replace(item, owner_action_id="r0.dma_in")
            if item.item_id == "r0.compute.temp"
            else item
            for item in _valid_problem().sram_items
        ),
    )

    failure = validate_joint_solution(bad_problem, _valid_solution())

    assert failure is not None
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION
    assert "temp_interval" in str(failure.diagnostics["reason"])


def test_validator_rejects_transfer_buffer_with_non_action_ownership():
    problem = _problem_with_transfer_buffer()
    bad_problem = replace(
        problem,
        sram_items=tuple(
            replace(
                item,
                owner_action_id=None,
                owner_value_id="input0",
                owner_residency_id="input0@3",
            )
            if item.item_id == "r0.compute.pack"
            else item
            for item in problem.sram_items
        ),
    )

    failure = validate_joint_solution(bad_problem, _solution_with_transfer_buffer())

    assert failure is not None
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION
    assert "transfer_buffer" in str(failure.diagnostics["reason"])


def test_validator_rejects_transfer_buffer_owned_by_compute_action():
    problem = replace(
        _problem_with_transfer_buffer(),
        sram_items=tuple(
            replace(item, owner_action_id="r0.compute")
            if item.item_id == "r0.compute.pack"
            else item
            for item in _problem_with_transfer_buffer().sram_items
        ),
    )

    failure = validate_joint_solution(problem, _solution_with_transfer_buffer())

    assert failure is not None
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION
    assert "transfer_buffer" in str(failure.diagnostics["reason"])
