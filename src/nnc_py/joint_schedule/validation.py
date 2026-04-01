"""Validation for external joint tiling/schedule problems and solutions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from nnc_py.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
    JointAction,
    JointActionKind,
    JointDependencyEdgeKind,
    JointFailure,
    JointFailureCategory,
    JointFailureStatus,
    JointProblem,
    JointResidencyWindow,
    JointResourceKind,
    JointSolution,
    JointSramAllocation,
    JointSramItem,
    JointSramItemKind,
    JointValue,
    JointValueTier,
)


def validate_joint_problem(problem: JointProblem) -> JointFailure | None:
    """Return a structured failure when the problem is invalid."""

    try:
        _validate_problem_shape(problem)
    except _ValidationError as exc:
        return _make_failure(
            status=JointFailureStatus.INVALID_PROBLEM,
            error_category=JointFailureCategory.INVALID_SOLUTION,
            reason=str(exc),
        )
    return None


def validate_joint_solution(
    problem: JointProblem,
    solution: JointSolution,
) -> JointFailure | None:
    """Return a structured failure when the solution is invalid."""

    problem_failure = validate_joint_problem(problem)
    if problem_failure is not None:
        return problem_failure

    if solution.schema_version != JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION:
        return _solution_failure(
            JointFailureCategory.INVALID_SOLUTION,
            "solution schema_version must be joint_tiling_schedule_solution_v1",
        )

    try:
        return _validate_solution(problem, solution)
    except _ValidationError as exc:
        return _solution_failure(exc.error_category, str(exc))


class _ValidationError(ValueError):
    def __init__(self, error_category: JointFailureCategory, message: str) -> None:
        super().__init__(message)
        self.error_category = error_category


@dataclass(frozen=True)
class _ActiveSramItem:
    item: JointSramItem
    allocation: JointSramAllocation
    start_time: int
    end_time: int


def _validate_problem_shape(problem: JointProblem) -> None:
    if problem.schema_version != JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION:
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            "problem schema_version must be joint_tiling_schedule_problem_v1",
        )

    regions_by_id = {region.region_id: region for region in problem.regions}
    recipes_by_id = {recipe.recipe_id: recipe for recipe in problem.recipes}
    values_by_id = {value.value_id: value for value in problem.values}
    actions_by_id = {action.action_id: action for action in problem.actions}
    resources_by_kind = {resource.resource_kind: resource for resource in problem.resources}

    for resource_kind in (
        JointResourceKind.DMA,
        JointResourceKind.MATMUL,
        JointResourceKind.SHAPE,
        JointResourceKind.OTHER,
    ):
        if resource_kind not in resources_by_kind:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"missing resource entry for {resource_kind.value}",
            )

    recipe_ids_by_action_id: dict[str, list[str]] = defaultdict(list)
    for recipe in problem.recipes:
        if recipe.region_id not in regions_by_id:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"recipe {recipe.recipe_id!r} references unknown region",
            )
        for action_id in recipe.activates_action_ids:
            if action_id not in actions_by_id:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"recipe {recipe.recipe_id!r} activates unknown action {action_id!r}",
                )
            recipe_ids_by_action_id[action_id].append(recipe.recipe_id)

    for action in problem.actions:
        if not action.is_optional:
            if action.region_id is None or action.recipe_id is None:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"mandatory action {action.action_id!r} must declare region_id and recipe_id",
                )
            if action.region_id not in regions_by_id:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"action {action.action_id!r} references unknown region",
                )
            if action.recipe_id not in recipes_by_id:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"action {action.action_id!r} references unknown recipe",
                )
            if recipes_by_id[action.recipe_id].region_id != action.region_id:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"action {action.action_id!r} region_id does not match its recipe region",
                )
            if recipe_ids_by_action_id.get(action.action_id) != [action.recipe_id]:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"mandatory action {action.action_id!r} must be activated by exactly one recipe",
                )
        else:
            if action.optional_value_id is None:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"optional action {action.action_id!r} must declare optional_value_id",
                )
            if action.region_id is not None or action.recipe_id is not None:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"optional action {action.action_id!r} must not be bound to a region or recipe",
                )
            if action.optional_value_id not in values_by_id:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"optional action {action.action_id!r} references unknown optional value {action.optional_value_id!r}",
                )

        for value_id in (*action.reads, *action.writes):
            if value_id not in values_by_id:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"action {action.action_id!r} references unknown value {value_id!r}",
                )
            if value_id in action.reads and not any(
                consumer.action_id == action.action_id
                for consumer in values_by_id[value_id].consumers
            ):
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"action {action.action_id!r} reads value {value_id!r} without matching value consumer metadata",
                )

    for value in problem.values:
        _validate_value_tiers(value)
        if value.producer is not None:
            producer_action = actions_by_id.get(value.producer.action_id)
            if producer_action is None or value.value_id not in producer_action.writes:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"value {value.value_id!r} producer does not match action writes",
                )
        for consumer in value.consumers:
            consumer_action = actions_by_id.get(consumer.action_id)
            if consumer_action is None or value.value_id not in consumer_action.reads:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"value {value.value_id!r} consumer does not match action reads",
                )

    for region in problem.regions:
        region_actions = tuple(
            action for action in problem.actions if action.region_id == region.region_id
        )
        for value_id in (*region.input_value_ids, *region.output_value_ids):
            if value_id not in values_by_id:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"region {region.region_id!r} references unknown value {value_id!r}",
                )
        if any(
            not any(value_id in action.reads for action in region_actions)
            for value_id in region.input_value_ids
        ):
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"region {region.region_id!r} input interface does not match action reads",
            )
        if any(
            not any(value_id in action.writes for action in region_actions)
            for value_id in region.output_value_ids
        ):
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"region {region.region_id!r} output interface does not match action writes",
            )

    adjacent_pairs = {
        (src.region_id, dst.region_id)
        for src in problem.regions
        for dst in problem.regions
        if src.region_id != dst.region_id
        and set(src.output_value_ids).intersection(dst.input_value_ids)
    }
    boundary_pairs = {
        (boundary.src_region_id, boundary.dst_region_id)
        for boundary in problem.boundary_constraints
    }
    if len(boundary_pairs) != len(problem.boundary_constraints):
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            "boundary constraints must be unique per adjacent region pair",
        )
    if adjacent_pairs != boundary_pairs:
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            "boundary constraints must cover every adjacent region pair exactly once",
        )

    for boundary in problem.boundary_constraints:
        if boundary.src_region_id not in regions_by_id or boundary.dst_region_id not in regions_by_id:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"boundary {boundary.boundary_id!r} references unknown regions",
            )
        for pair in boundary.compatible_recipe_pairs:
            if pair.src_recipe_id not in recipes_by_id or pair.dst_recipe_id not in recipes_by_id:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"boundary {boundary.boundary_id!r} references unknown recipes",
                )
            if recipes_by_id[pair.src_recipe_id].region_id != boundary.src_region_id:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"boundary {boundary.boundary_id!r} references source recipe bound to the wrong region",
                )
            if recipes_by_id[pair.dst_recipe_id].region_id != boundary.dst_region_id:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"boundary {boundary.boundary_id!r} references destination recipe bound to the wrong region",
                )

    for edge in problem.dependency_edges:
        if edge.src_action_id not in actions_by_id or edge.dst_action_id not in actions_by_id:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"dependency edge {edge.src_action_id!r}->{edge.dst_action_id!r} references unknown actions",
            )


def _validate_value_tiers(value: JointValue) -> None:
    if value.required_final_tier is JointValueTier.UNMATERIALIZED:
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            f"value {value.value_id!r} cannot require final tier unmaterialized",
        )
    if value.producer is None:
        if value.initial_tier is JointValueTier.UNMATERIALIZED:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"value {value.value_id!r} without producer cannot start unmaterialized",
            )
    else:
        if value.initial_tier is not JointValueTier.UNMATERIALIZED:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"value {value.value_id!r} with producer must start unmaterialized",
            )


def _validate_solution(
    problem: JointProblem,
    solution: JointSolution,
) -> JointFailure | None:
    regions_by_id = {region.region_id: region for region in problem.regions}
    recipes_by_id = {recipe.recipe_id: recipe for recipe in problem.recipes}
    actions_by_id = {action.action_id: action for action in problem.actions}
    values_by_id = {value.value_id: value for value in problem.values}
    resources_by_kind = {resource.resource_kind: resource for resource in problem.resources}

    selected_by_region = {item.region_id: item.recipe_id for item in solution.selected_recipes}
    if set(selected_by_region) != set(regions_by_id):
        missing = sorted(set(regions_by_id) - set(selected_by_region))
        extra = sorted(set(selected_by_region) - set(regions_by_id))
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            f"selected recipes must cover each region exactly once (missing={missing}, extra={extra})",
        )
    for region_id, recipe_id in selected_by_region.items():
        recipe = recipes_by_id.get(recipe_id)
        if recipe is None or recipe.region_id != region_id:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"selected recipe {recipe_id!r} is invalid for region {region_id!r}",
            )

    mandatory_action_ids = {
        action_id
        for recipe_id in selected_by_region.values()
        for action_id in recipes_by_id[recipe_id].activates_action_ids
    }
    scheduled_by_id = {item.action_id: item for item in solution.scheduled_actions}
    if not set(scheduled_by_id).issuperset(mandatory_action_ids):
        missing_action_ids = sorted(mandatory_action_ids - set(scheduled_by_id))
        if any(actions_by_id[action_id].kind is JointActionKind.DMA_OUT for action_id in missing_action_ids):
            raise _ValidationError(
                JointFailureCategory.INCOMPLETE_SOLUTION,
                f"missing mandatory finalization actions: {missing_action_ids}",
            )
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            f"missing mandatory active actions: {missing_action_ids}",
        )

    allowed_action_ids = mandatory_action_ids | {
        action.action_id for action in problem.actions if action.is_optional
    }
    for action_id in scheduled_by_id:
        if action_id not in allowed_action_ids:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"scheduled unknown or illegal action {action_id!r}",
            )

    active_actions = {action_id: actions_by_id[action_id] for action_id in scheduled_by_id}
    start_by_action = {item.action_id: item.start_time for item in solution.scheduled_actions}
    end_by_action = {
        action_id: start_by_action[action_id] + action.duration + action.launch_overhead
        for action_id, action in active_actions.items()
    }
    makespan = max(end_by_action.values(), default=0)
    if makespan != solution.objective_value:
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            f"objective_value {solution.objective_value} does not match makespan {makespan}",
        )

    _validate_resource_overlap(active_actions, start_by_action, end_by_action)
    _validate_dependencies(problem, start_by_action, end_by_action)
    _validate_boundary_choices(problem, selected_by_region)

    windows_by_value = _normalize_windows(solution, values_by_id)
    _validate_optional_transfer_legality(
        active_actions,
        values_by_id,
        start_by_action,
        end_by_action,
        windows_by_value,
    )
    _validate_residency_constraints(
        values_by_id,
        active_actions,
        start_by_action,
        end_by_action,
        windows_by_value,
        solution.objective_value,
    )
    _validate_sram_placement(
        problem,
        solution,
        actions_by_id=actions_by_id,
        start_by_action=start_by_action,
        end_by_action=end_by_action,
    )
    _validate_capacity(
        problem,
        active_actions,
        start_by_action,
        end_by_action,
        windows_by_value,
        resources_by_kind,
    )
    _validate_final_outputs(values_by_id, active_actions, end_by_action, solution.objective_value)
    return None


def _validate_dependencies(
    problem: JointProblem,
    start_by_action: Mapping[str, int],
    end_by_action: Mapping[str, int],
) -> None:
    for edge in problem.dependency_edges:
        if edge.src_action_id not in end_by_action or edge.dst_action_id not in start_by_action:
            continue
        if end_by_action[edge.src_action_id] > start_by_action[edge.dst_action_id]:
            raise _ValidationError(
                JointFailureCategory.DEPENDENCY_VIOLATION,
                f"dependency violated: {edge.src_action_id!r} -> {edge.dst_action_id!r}",
            )


def _validate_resource_overlap(
    active_actions: Mapping[str, JointAction],
    start_by_action: Mapping[str, int],
    end_by_action: Mapping[str, int],
) -> None:
    actions_by_resource: dict[object, list[str]] = defaultdict(list)
    for action_id, action in active_actions.items():
        actions_by_resource[action.resource_kind].append(action_id)
    for action_ids in actions_by_resource.values():
        for index, left_id in enumerate(action_ids):
            for right_id in action_ids[index + 1 :]:
                if _intervals_overlap(
                    start_by_action[left_id],
                    end_by_action[left_id],
                    start_by_action[right_id],
                    end_by_action[right_id],
                ):
                    raise _ValidationError(
                        JointFailureCategory.RESOURCE_OVERLAP,
                        f"resource overlap between {left_id!r} and {right_id!r}",
                    )


def _validate_boundary_choices(
    problem: JointProblem, selected_by_region: Mapping[str, str]
) -> None:
    for boundary in problem.boundary_constraints:
        chosen_pair = (
            selected_by_region[boundary.src_region_id],
            selected_by_region[boundary.dst_region_id],
        )
        compatible_pairs = {
            (pair.src_recipe_id, pair.dst_recipe_id)
            for pair in boundary.compatible_recipe_pairs
        }
        if chosen_pair not in compatible_pairs:
            raise _ValidationError(
                JointFailureCategory.INCOMPATIBLE_RECIPE_BOUNDARY,
                f"selected recipes {chosen_pair} violate boundary {boundary.boundary_id!r}",
            )


def _normalize_windows(
    solution: JointSolution,
    values_by_id: Mapping[str, JointValue],
) -> dict[str, tuple[JointResidencyWindow, ...]]:
    windows_by_value: dict[str, list[JointResidencyWindow]] = defaultdict(list)
    for window in solution.residency_windows:
        if window.value_id not in values_by_id:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"residency window references unknown value {window.value_id!r}",
            )
        windows_by_value[window.value_id].append(window)
    normalized: dict[str, tuple[JointResidencyWindow, ...]] = {}
    for value_id, windows in windows_by_value.items():
        ordered = tuple(sorted(windows, key=lambda item: (item.start_time, item.end_time)))
        for prev, nxt in zip(ordered, ordered[1:]):
            if _intervals_overlap(
                prev.start_time,
                prev.end_time,
                nxt.start_time,
                nxt.end_time,
            ):
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"residency windows overlap for value {value_id!r}",
                )
        normalized[value_id] = ordered
    return normalized


def _validate_optional_transfer_legality(
    active_actions: Mapping[str, JointAction],
    values_by_id: Mapping[str, JointValue],
    start_by_action: Mapping[str, int],
    end_by_action: Mapping[str, int],
    windows_by_value: Mapping[str, tuple[JointResidencyWindow, ...]],
) -> None:
    spill_end_by_value = {
        action.optional_value_id: end_by_action[action_id]
        for action_id, action in active_actions.items()
        if action.kind is JointActionKind.SPILL and action.optional_value_id is not None
    }
    for action_id, action in active_actions.items():
        if action.kind not in (JointActionKind.SPILL, JointActionKind.RELOAD):
            continue
        value_id = action.optional_value_id
        if value_id is None:
            raise _ValidationError(
                JointFailureCategory.ILLEGAL_TRANSFER,
                f"optional action {action_id!r} is missing optional_value_id",
            )
        value = values_by_id[value_id]
        if not value.spillable:
            raise _ValidationError(
                JointFailureCategory.ILLEGAL_TRANSFER,
                f"value {value_id!r} is not spillable",
            )
        if action.kind is JointActionKind.SPILL:
            if not _is_resident_for_interval(
                windows_by_value.get(value_id, ()),
                start_by_action[action_id],
                end_by_action[action_id],
            ):
                raise _ValidationError(
                    JointFailureCategory.ILLEGAL_TRANSFER,
                    f"spill action {action_id!r} requires SRAM residency",
                )
        if action.kind is JointActionKind.RELOAD:
            spill_end = spill_end_by_value.get(value_id)
            if spill_end is None or spill_end > start_by_action[action_id]:
                raise _ValidationError(
                    JointFailureCategory.ILLEGAL_TRANSFER,
                    f"reload action {action_id!r} has no preceding completed spill",
                )


def _validate_residency_constraints(
    values_by_id: Mapping[str, JointValue],
    active_actions: Mapping[str, JointAction],
    start_by_action: Mapping[str, int],
    end_by_action: Mapping[str, int],
    windows_by_value: Mapping[str, tuple[JointResidencyWindow, ...]],
    objective_value: int,
) -> None:
    writer_actions_by_value: dict[str, list[JointAction]] = defaultdict(list)
    for action in active_actions.values():
        for value_id in action.writes:
            writer_actions_by_value[value_id].append(action)

    for value_id, value in values_by_id.items():
        windows = windows_by_value.get(value_id, ())
        if value.initial_tier is JointValueTier.SRAM:
            if not windows or windows[0].start_time != 0:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"value {value_id!r} requires an initial SRAM window starting at 0",
                )
        if value.required_final_tier is JointValueTier.SRAM:
            if not windows or windows[-1].end_time != objective_value:
                raise _ValidationError(
                    JointFailureCategory.INCOMPLETE_SOLUTION,
                    f"value {value_id!r} requires final SRAM residency through objective_value",
                )
        if not value.allows_multiple_sram_windows and len(windows) > 1:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"value {value_id!r} cannot have multiple SRAM windows",
            )
        if windows:
            first_window_start = windows[0].start_time
            anchored_starts = _valid_window_open_times(
                value_id=value_id,
                value=value,
                active_actions=active_actions,
                end_by_action=end_by_action,
            )
            if first_window_start not in anchored_starts:
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"value {value_id!r} has an unjustified first SRAM window",
                )
        if value.must_keep and windows:
            relevant_consumer_ends = [
                end_by_action[consumer.action_id]
                for consumer in value.consumers
                if consumer.action_id in end_by_action
            ]
            if relevant_consumer_ends:
                first_available = min(_valid_window_open_times(
                    value_id=value_id,
                    value=value,
                    active_actions=active_actions,
                    end_by_action=end_by_action,
                ))
                if (
                    len(windows) != 1
                    or windows[0].start_time != first_available
                    or windows[0].end_time < max(relevant_consumer_ends)
                ):
                    raise _ValidationError(
                        JointFailureCategory.INVALID_SOLUTION,
                        f"value {value_id!r} must_keep residency is not continuous",
                    )

        for window in windows[1:]:
            if not any(
                end_by_action[action.action_id] == window.start_time
                for action in writer_actions_by_value.get(value_id, ())
                if action.kind in (
                    JointActionKind.COMPUTE,
                    JointActionKind.DMA_IN,
                    JointActionKind.RELOAD,
                )
            ):
                raise _ValidationError(
                    JointFailureCategory.ILLEGAL_TRANSFER,
                    f"value {value_id!r} re-enters SRAM without a matching writer",
                )
        for prev, nxt in zip(windows, windows[1:]):
            if not any(
                action.kind is JointActionKind.SPILL
                and action.optional_value_id == value_id
                and end_by_action[action_id] == prev.end_time
                for action_id, action in active_actions.items()
            ):
                raise _ValidationError(
                    JointFailureCategory.ILLEGAL_TRANSFER,
                    f"value {value_id!r} leaves SRAM without a matching spill",
                )
            if not any(
                action.kind is JointActionKind.RELOAD
                and action.optional_value_id == value_id
                and end_by_action[action_id] == nxt.start_time
                for action_id, action in active_actions.items()
            ):
                raise _ValidationError(
                    JointFailureCategory.ILLEGAL_TRANSFER,
                    f"value {value_id!r} re-enters SRAM without a matching reload",
                )

    for action_id, action in active_actions.items():
        if action.kind in (JointActionKind.COMPUTE, JointActionKind.DMA_OUT, JointActionKind.SPILL):
            for value_id in action.reads:
                if not _is_resident_for_interval(
                    windows_by_value.get(value_id, ()),
                    start_by_action[action_id],
                    end_by_action[action_id],
                ):
                    raise _ValidationError(
                        JointFailureCategory.ILLEGAL_TRANSFER
                        if action.kind is not JointActionKind.COMPUTE
                        else JointFailureCategory.INVALID_SOLUTION,
                        f"action {action_id!r} reads value {value_id!r} without full SRAM residency",
                    )


def _validate_sram_placement(
    problem: JointProblem,
    solution: JointSolution,
    *,
    actions_by_id: Mapping[str, JointAction],
    start_by_action: Mapping[str, int],
    end_by_action: Mapping[str, int],
) -> None:
    windows_by_residency = {
        window.residency_id: window for window in solution.residency_windows
    }
    generated_items = solution.generated_sram_items
    if any(item.kind is not JointSramItemKind.RESIDENT_WINDOW for item in generated_items):
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            "generated_sram_items must contain only resident_window items",
        )
    if len(generated_items) != len(solution.residency_windows):
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            "resident_window cardinality must match residency_windows exactly",
        )

    allocations_by_item = {
        allocation.item_id: allocation for allocation in solution.sram_allocations
    }
    active_items: list[_ActiveSramItem] = []
    for item in problem.sram_items:
        _validate_sram_item_ownership(
            item,
            actions_by_id=actions_by_id,
            windows_by_residency=windows_by_residency,
        )
        lifetime = _item_lifetime(
            item,
            start_by_action=start_by_action,
            end_by_action=end_by_action,
            windows_by_residency=windows_by_residency,
        )
        if lifetime is None:
            continue
        allocation = allocations_by_item.get(item.item_id)
        if allocation is None:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"missing allocation for active SRAM item {item.item_id!r}",
            )
        active_items.append(
            _ActiveSramItem(
                item=item,
                allocation=allocation,
                start_time=lifetime[0],
                end_time=lifetime[1],
            )
        )

    resident_item_ids_by_residency: dict[str, str] = {}
    for item in generated_items:
        _validate_sram_item_ownership(
            item,
            actions_by_id=actions_by_id,
            windows_by_residency=windows_by_residency,
        )
        assert item.owner_residency_id is not None
        if item.owner_residency_id in resident_item_ids_by_residency:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                "resident_window cardinality must match residency_windows exactly",
            )
        resident_item_ids_by_residency[item.owner_residency_id] = item.item_id
        lifetime = _item_lifetime(
            item,
            start_by_action=start_by_action,
            end_by_action=end_by_action,
            windows_by_residency=windows_by_residency,
        )
        if lifetime is None:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"resident_window item {item.item_id!r} is missing its residency window",
            )
        allocation = allocations_by_item.get(item.item_id)
        if allocation is None:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"missing allocation for active SRAM item {item.item_id!r}",
            )
        active_items.append(
            _ActiveSramItem(
                item=item,
                allocation=allocation,
                start_time=lifetime[0],
                end_time=lifetime[1],
            )
        )

    if set(resident_item_ids_by_residency) != set(windows_by_residency):
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            "resident_window cardinality must match residency_windows exactly",
        )

    active_item_ids = {active_item.item.item_id for active_item in active_items}
    extra_allocations = sorted(set(allocations_by_item) - active_item_ids)
    if extra_allocations:
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            f"allocations reference inactive SRAM items: {extra_allocations}",
        )

    for active_item in active_items:
        if active_item.allocation.offset < 0:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"allocation for {active_item.item.item_id!r} has negative offset",
            )
        if active_item.allocation.offset % active_item.item.alignment_bytes != 0:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"allocation for {active_item.item.item_id!r} violates alignment",
            )
        end_offset = active_item.allocation.offset + active_item.item.size_bytes
        if end_offset > problem.sram_capacity_bytes:
            raise _ValidationError(
                JointFailureCategory.SRAM_CAPACITY_EXCEEDED,
                f"allocation for {active_item.item.item_id!r} exceeds SRAM capacity",
            )

    for index, left in enumerate(active_items):
        left_start = left.allocation.offset
        left_end = left_start + left.item.size_bytes
        for right in active_items[index + 1 :]:
            if not _intervals_overlap(
                left.start_time,
                left.end_time,
                right.start_time,
                right.end_time,
            ):
                continue
            right_start = right.allocation.offset
            right_end = right_start + right.item.size_bytes
            if _intervals_overlap(left_start, left_end, right_start, right_end):
                raise _ValidationError(
                    JointFailureCategory.INVALID_SOLUTION,
                    f"SRAM allocations overlap for time-overlapping items {left.item.item_id!r} and {right.item.item_id!r}",
                )


def _validate_sram_item_ownership(
    item: JointSramItem,
    *,
    actions_by_id: Mapping[str, JointAction],
    windows_by_residency: Mapping[str, JointResidencyWindow],
) -> None:
    if item.kind in (
        JointSramItemKind.TEMP_INTERVAL,
        JointSramItemKind.TRANSFER_BUFFER,
    ):
        if item.owner_action_id is None:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"{item.kind.value} item {item.item_id!r} must declare owner_action_id",
            )
        if item.owner_action_id not in actions_by_id:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"{item.kind.value} item {item.item_id!r} references unknown action {item.owner_action_id!r}",
            )
        if item.owner_value_id is not None or item.owner_residency_id is not None:
            raise _ValidationError(
                JointFailureCategory.INVALID_SOLUTION,
                f"{item.kind.value} item {item.item_id!r} has invalid ownership metadata",
            )
        return

    if item.owner_action_id is not None:
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            f"resident_window item {item.item_id!r} must not declare owner_action_id",
        )
    if item.owner_value_id is None or item.owner_residency_id is None:
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            f"resident_window item {item.item_id!r} has incomplete ownership metadata",
        )
    window = windows_by_residency.get(item.owner_residency_id)
    if window is None or window.value_id != item.owner_value_id:
        raise _ValidationError(
            JointFailureCategory.INVALID_SOLUTION,
            f"resident_window item {item.item_id!r} has ownership mismatch",
        )


def _validate_capacity(
    problem: JointProblem,
    active_actions: Mapping[str, JointAction],
    start_by_action: Mapping[str, int],
    end_by_action: Mapping[str, int],
    windows_by_value: Mapping[str, tuple[JointResidencyWindow, ...]],
    resources_by_kind: Mapping[JointResourceKind, object],
) -> None:
    if not resources_by_kind:
        return
    event_times = sorted(
        {
            0,
            *start_by_action.values(),
            *end_by_action.values(),
            *(
                time
                for windows in windows_by_value.values()
                for window in windows
                for time in (window.start_time, window.end_time)
            ),
        }
    )
    for event_time, next_time in zip(event_times, event_times[1:]):
        if event_time == next_time:
            continue
        resident_bytes = 0
        for value_id, windows in windows_by_value.items():
            if any(window.start_time <= event_time < window.end_time for window in windows):
                resident_bytes += _value_size(problem, value_id)
        temp_bytes = sum(
            action.temp_bytes
            for action_id, action in active_actions.items()
            if start_by_action[action_id] <= event_time < end_by_action[action_id]
        )
        if resident_bytes + temp_bytes > problem.sram_capacity_bytes:
            raise _ValidationError(
                JointFailureCategory.SRAM_CAPACITY_EXCEEDED,
                f"SRAM capacity exceeded at time {event_time}",
            )


def _validate_final_outputs(
    values_by_id: Mapping[str, JointValue],
    active_actions: Mapping[str, JointAction],
    end_by_action: Mapping[str, int],
    objective_value: int,
) -> None:
    for value in values_by_id.values():
        if value.producer is None or value.producer.action_id not in active_actions:
            continue
        if value.required_final_tier is JointValueTier.SLOW:
            if not any(
                action.kind in (JointActionKind.DMA_OUT, JointActionKind.SPILL)
                and value.value_id in action.writes
                for action in active_actions.values()
            ):
                raise _ValidationError(
                    JointFailureCategory.INCOMPLETE_SOLUTION,
                    f"value {value.value_id!r} is missing required final slow-tier materialization",
                )
        if value.required_final_tier is JointValueTier.SRAM:
            if end_by_action[value.producer.action_id] > objective_value:
                raise _ValidationError(
                    JointFailureCategory.INCOMPLETE_SOLUTION,
                    f"value {value.value_id!r} is not available by objective_value",
                )


def _is_resident_for_interval(
    windows: tuple[object, ...], start_time: int, end_time: int
) -> bool:
    return any(
        window.start_time <= start_time and end_time <= window.end_time
        for window in windows
    )


def _value_size(problem: JointProblem, value_id: str) -> int:
    for value in problem.values:
        if value.value_id == value_id:
            return value.size_bytes
    return 0


def _valid_window_open_times(
    *,
    value_id: str,
    value: JointValue,
    active_actions: Mapping[str, JointAction],
    end_by_action: Mapping[str, int],
) -> set[int]:
    starts: set[int] = set()
    if value.initial_tier is JointValueTier.SRAM:
        starts.add(0)
    for action_id, action in active_actions.items():
        if value_id not in action.writes:
            continue
        if action.kind in (
            JointActionKind.COMPUTE,
            JointActionKind.DMA_IN,
            JointActionKind.RELOAD,
        ):
            starts.add(end_by_action[action_id])
    return starts


def _item_lifetime(
    item: JointSramItem,
    *,
    start_by_action: Mapping[str, int],
    end_by_action: Mapping[str, int],
    windows_by_residency: Mapping[str, JointResidencyWindow],
) -> tuple[int, int] | None:
    if item.owner_residency_id is not None:
        window = windows_by_residency.get(item.owner_residency_id)
        if window is None:
            return None
        return window.start_time, window.end_time
    if item.owner_action_id is not None:
        start_time = start_by_action.get(item.owner_action_id)
        end_time = end_by_action.get(item.owner_action_id)
        if start_time is None or end_time is None:
            return None
        return start_time, end_time
    return None


def _intervals_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _make_failure(
    *,
    status: JointFailureStatus,
    error_category: JointFailureCategory,
    reason: str,
) -> JointFailure:
    return JointFailure(
        schema_version=JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
        status=status,
        error_category=error_category,
        diagnostics={"reason": reason},
    )


def _solution_failure(
    error_category: JointFailureCategory, reason: str
) -> JointFailure:
    return _make_failure(
        status=JointFailureStatus.ERROR,
        error_category=error_category,
        reason=reason,
    )


__all__ = ["validate_joint_problem", "validate_joint_solution"]
