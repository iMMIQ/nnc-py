"""Materialize validated joint schedules into the current internal schedule IR."""

from __future__ import annotations

from nnc_py.ir.joint_tiling_schedule import (
    JointAction,
    JointActionKind,
    JointProblem,
    JointResidencyWindow,
    JointSolution,
    JointSramItem,
    JointValue,
    JointValueTier,
)
from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    PipelineScheduleProblem,
    PipelineScheduleResult,
    ResidencyWindow,
    ScheduleDependencyKind,
    ScheduleEdge,
    ScheduleStep,
    ScheduleStepKind,
    ScheduledStep,
    ScheduledValue,
    ScheduledValueHomeTier,
    SramAllocationInterval,
    TransferStep,
    TransferStepKind,
)
from nnc_py.joint_schedule.validation import validate_joint_solution


def materialize_joint_solution(
    problem: JointProblem,
    solution: JointSolution,
) -> tuple[PipelineScheduleProblem, PipelineScheduleResult]:
    """Lower a validated joint solution into the internal schedule IR."""

    failure = validate_joint_solution(problem, solution)
    if failure is not None:
        raise ValueError(f"cannot materialize invalid joint solution: {failure.diagnostics}")

    active_action_ids = {item.action_id for item in solution.scheduled_actions}
    active_actions = tuple(
        action for action in problem.actions if action.action_id in active_action_ids
    )
    value_by_id = {value.value_id: value for value in problem.values}
    windows_by_value = _windows_by_value(solution)
    start_by_action = {item.action_id: item.start_time for item in solution.scheduled_actions}
    end_by_action = {
        action.action_id: start_by_action[action.action_id] + action.duration + action.launch_overhead
        for action in active_actions
    }
    home_tier_by_value = {
        value.value_id: _base_home_tier(value)
        for value in problem.values
    }

    steps = tuple(
        _materialize_step(
            action,
            start_by_action=start_by_action,
            end_by_action=end_by_action,
            windows_by_value=windows_by_value,
            value_by_id=value_by_id,
            home_tier_by_value=home_tier_by_value,
        )
        for action in active_actions
    )
    edges = tuple(
        ScheduleEdge(
            src_step_id=edge.src_action_id,
            dst_step_id=edge.dst_action_id,
            kind=(
                ScheduleDependencyKind.DATA
                if edge.kind.value == "data"
                else ScheduleDependencyKind.ORDER
            ),
        )
        for edge in problem.dependency_edges
        if edge.src_action_id in active_action_ids and edge.dst_action_id in active_action_ids
    )
    scheduled_values = _materialize_values(
        problem,
        active_action_ids=active_action_ids,
        windows_by_value=windows_by_value,
        start_by_action=start_by_action,
        end_by_action=end_by_action,
    )
    residency_windows = tuple(
        _materialize_residency_window(
            window=window,
            value=value_by_id[window.value_id],
            active_actions=active_actions,
            end_by_action=end_by_action,
            objective_value=solution.objective_value,
            windows_by_value=windows_by_value,
        )
        for window in solution.residency_windows
        if not (
            window.start_time == 0
            and value_by_id[window.value_id].initial_tier is JointValueTier.SRAM
        )
    )
    scheduled_steps = tuple(
        ScheduledStep(
            step_id=action.action_id,
            resource_kind=_pipeline_resource_kind(action),
            resource_slot=0,
            start_time=start_by_action[action.action_id],
            end_time=end_by_action[action.action_id],
        )
        for action in active_actions
    )
    sram_intervals = _materialize_sram_intervals(
        problem,
        solution,
        start_by_action=start_by_action,
        end_by_action=end_by_action,
        value_by_id=value_by_id,
        windows_by_value=windows_by_value,
    )

    internal_problem = PipelineScheduleProblem(
        steps=steps,
        edges=edges,
        scheduled_values=scheduled_values,
        residency_windows=residency_windows,
        resources=tuple(_pipeline_resource_kind(resource) for resource in problem.resources),
        sram_capacity_bytes=problem.sram_capacity_bytes,
        metadata={
            "origin": "joint_tiling_schedule_materialize",
            "joint_objective": solution.objective_value,
        },
    )
    internal_result = PipelineScheduleResult(
        scheduled_steps=scheduled_steps,
        sram_intervals=sram_intervals,
        scheduled_values=scheduled_values,
        residency_windows=residency_windows,
        makespan=solution.objective_value,
        feasible=True,
        solver_name="joint_materialized",
        diagnostics={"active_actions": tuple(active_action_ids)},
    )
    return internal_problem, internal_result


def _materialize_sram_intervals(
    problem: JointProblem,
    solution: JointSolution,
    *,
    start_by_action: dict[str, int],
    end_by_action: dict[str, int],
    value_by_id: dict[str, JointValue],
    windows_by_value: dict[str, tuple[object, ...]],
) -> tuple[SramAllocationInterval, ...]:
    windows_by_residency = {
        window.residency_id: window for window in solution.residency_windows
    }
    allocations_by_item = {
        allocation.item_id: allocation for allocation in solution.sram_allocations
    }
    intervals: list[SramAllocationInterval] = []

    for item in (*problem.sram_items, *solution.generated_sram_items):
        lifetime = _sram_item_lifetime(
            item,
            start_by_action=start_by_action,
            end_by_action=end_by_action,
            windows_by_residency=windows_by_residency,
        )
        if lifetime is None:
            continue
        allocation = allocations_by_item.get(item.item_id)
        if allocation is None:
            raise ValueError(f"active SRAM item {item.item_id!r} is missing an allocation")
        value_name = _sram_interval_value_name(
            item,
            windows_by_residency=windows_by_residency,
            value_by_id=value_by_id,
            windows_by_value=windows_by_value,
        )
        intervals.append(
            SramAllocationInterval(
                value_name=value_name,
                item_id=item.item_id,
                item_kind=item.kind.value,
                buffer_id=item.item_id,
                offset=allocation.offset,
                start_time=lifetime[0],
                end_time=lifetime[1],
                size_bytes=item.size_bytes,
            )
        )

    return tuple(intervals)


def _sram_item_lifetime(
    item: JointSramItem,
    *,
    start_by_action: dict[str, int],
    end_by_action: dict[str, int],
    windows_by_residency: dict[str, JointResidencyWindow],
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


def _sram_interval_value_name(
    item: JointSramItem,
    *,
    windows_by_residency: dict[str, JointResidencyWindow],
    value_by_id: dict[str, JointValue],
    windows_by_value: dict[str, tuple[object, ...]],
) -> str:
    if item.owner_residency_id is None:
        return item.item_id
    window = windows_by_residency.get(item.owner_residency_id)
    if window is None:
        raise ValueError(f"resident SRAM item {item.item_id!r} is missing its residency window")
    return _resident_name_for_window(
        window.value_id,
        window.start_time,
        value_by_id,
        windows_by_value,
    )


def _materialize_step(
    action: JointAction,
    *,
    start_by_action: dict[str, int],
    end_by_action: dict[str, int],
    windows_by_value: dict[str, tuple[object, ...]],
    value_by_id: dict[str, JointValue],
    home_tier_by_value: dict[str, ScheduledValueHomeTier],
) -> ScheduleStep:
    if action.kind is JointActionKind.COMPUTE:
        return ScheduleStep(
            id=action.action_id,
            node_name=action.region_id or action.action_id,
            step_kind=ScheduleStepKind.COMPUTE,
            resource_kind=_pipeline_resource_kind(action),
            duration=action.duration,
            launch_overhead=action.launch_overhead,
            sram_input_names=tuple(
                _resident_name_for_read(
                    value_id,
                    action_start=start_by_action[action.action_id],
                    action_end=end_by_action[action.action_id],
                    windows_by_value=windows_by_value,
                    value_by_id=value_by_id,
                )
                for value_id in action.reads
            ),
            sram_output_names=tuple(
                _resident_name_for_write(
                    value_id,
                    writer_end=end_by_action[action.action_id],
                    windows_by_value=windows_by_value,
                    value_by_id=value_by_id,
                )
                for value_id in action.writes
            ),
            sram_temp_bytes=action.temp_bytes,
            attrs={"joint_action_kind": action.kind.value},
        )

    logical_value_id = action.optional_value_id or (action.reads[0] if action.reads else action.writes[0])
    if logical_value_id is None:
        raise ValueError(f"joint transfer action {action.action_id!r} has no logical value binding")
    resident_read_name = _resident_name_for_read(
        logical_value_id,
        action_start=start_by_action[action.action_id],
        action_end=end_by_action[action.action_id],
        windows_by_value=windows_by_value,
        value_by_id=value_by_id,
    )
    resident_write_name = _resident_name_for_write(
        logical_value_id,
        writer_end=end_by_action[action.action_id],
        windows_by_value=windows_by_value,
        value_by_id=value_by_id,
    )

    if action.kind in (JointActionKind.DMA_IN, JointActionKind.RELOAD):
        sram_input_names: tuple[str, ...] = ()
        sram_output_names = (resident_write_name,)
        moved_value_name = (
            logical_value_id
            if action.kind is JointActionKind.DMA_IN
            else _resident_name_for_previous_window(
                logical_value_id,
                action_start=start_by_action[action.action_id],
                windows_by_value=windows_by_value,
                value_by_id=value_by_id,
            )
        )
    else:
        sram_input_names = (resident_read_name,)
        sram_output_names = ()
        moved_value_name = resident_read_name

    return TransferStep(
        id=action.action_id,
        node_name=action.region_id or action.action_id,
        transfer_kind=_transfer_step_kind(action),
        moved_value_name=moved_value_name,
        src_tier=_src_tier(action, home_tier_by_value[logical_value_id]),
        dst_tier=_dst_tier(action),
        bytes=value_by_id[logical_value_id].size_bytes,
        duration=action.duration,
        launch_overhead=action.launch_overhead,
        sram_input_names=sram_input_names,
        sram_output_names=sram_output_names,
        attrs={"joint_action_kind": action.kind.value},
    )


def _materialize_values(
    problem: JointProblem,
    *,
    active_action_ids: set[str],
    windows_by_value: dict[str, tuple[object, ...]],
    start_by_action: dict[str, int],
    end_by_action: dict[str, int],
) -> tuple[ScheduledValue, ...]:
    active_actions = {
        action.action_id: action
        for action in problem.actions
        if action.action_id in active_action_ids
    }
    materialized: list[ScheduledValue] = []
    for value in problem.values:
        home_tier = _base_home_tier(value)
        resident_windows = windows_by_value.get(value.value_id, ())
        if resident_windows and _uses_distinct_window_aliases(
            value.value_id,
            value_by_id={item.value_id: item for item in problem.values},
            windows_by_value=windows_by_value,
        ):
            materialized.append(
                ScheduledValue(
                    name=value.value_id,
                    graph_tensor_name=value.value_id,
                    size_bytes=value.size_bytes,
                    producer_step_id=None,
                    consumer_step_ids=tuple(
                        action_id
                        for action_id, action in active_actions.items()
                        if action.kind is JointActionKind.DMA_IN
                        and value.value_id in action.reads
                    ),
                    home_tier=(
                        home_tier
                        if home_tier is not ScheduledValueHomeTier.SRAM
                        else ScheduledValueHomeTier.SLOW
                    ),
                )
            )
            for window in resident_windows:
                alias_name = _resident_name_for_window(
                    value.value_id,
                    window.start_time,
                    {value.value_id: value},
                    {value.value_id: resident_windows},
                )
                materialized.append(
                    ScheduledValue(
                        name=alias_name,
                        graph_tensor_name=value.value_id,
                        size_bytes=value.size_bytes,
                        producer_step_id=_resident_producer_step_id(
                            value,
                            window_start=window.start_time,
                            active_actions=active_actions,
                            end_by_action=end_by_action,
                        ),
                    consumer_step_ids=tuple(
                        action_id
                        for action_id, action in active_actions.items()
                        if _action_consumes_window(
                            action,
                            action_id=action_id,
                            value_id=value.value_id,
                            window_start=window.start_time,
                            start_by_action=start_by_action,
                            end_by_action=end_by_action,
                            windows_by_value=windows_by_value,
                            value_by_id={item.value_id: item for item in problem.values},
                        )
                    ),
                        must_reside_in_sram=(
                            value.must_keep or value.required_final_tier is JointValueTier.SRAM
                        ),
                        home_tier=ScheduledValueHomeTier.SRAM,
                    )
                )
            continue

        materialized.append(
            ScheduledValue(
                name=value.value_id,
                graph_tensor_name=value.value_id,
                size_bytes=value.size_bytes,
                producer_step_id=_resident_producer_step_id(
                    value,
                    window_start=None,
                    active_actions=active_actions,
                    end_by_action=end_by_action,
                ),
                consumer_step_ids=tuple(
                    action_id
                    for action_id, action in active_actions.items()
                    if value.value_id in action.reads and action.kind is not JointActionKind.DMA_IN
                ),
                must_reside_in_sram=(
                    value.must_keep or value.required_final_tier is JointValueTier.SRAM
                ),
                home_tier=home_tier,
            )
        )
    return tuple(materialized)


def _materialize_residency_window(
    *,
    window,
    value: JointValue,
    active_actions: tuple[JointAction, ...],
    end_by_action: dict[str, int],
    objective_value: int,
    windows_by_value: dict[str, tuple[object, ...]],
) -> ResidencyWindow:
    if window.start_time == 0 and value.initial_tier is JointValueTier.SRAM:
        raise ValueError("initial SRAM windows starting at 0 should use default internal residency")
    opened_by = _find_open_action_id(window.value_id, window.start_time, active_actions, end_by_action)
    closed_by = None
    if window.end_time != objective_value:
        closed_by = _find_close_action_id(window.value_id, window.end_time, active_actions, end_by_action)
    value_name = _resident_name_for_window(
        window.value_id,
        window.start_time,
        {value.value_id: value},
        {value.value_id: windows_by_value.get(value.value_id, ())},
    )
    return ResidencyWindow(
        value_name=value_name,
        residency_id=f"{value_name}@{window.start_time}",
        opened_by_step_id=opened_by,
        closed_by_step_id=closed_by,
    )


def _windows_by_value(solution: JointSolution) -> dict[str, tuple[object, ...]]:
    grouped: dict[str, list[object]] = {}
    for window in solution.residency_windows:
        grouped.setdefault(window.value_id, []).append(window)
    return {
        value_id: tuple(sorted(windows, key=lambda item: (item.start_time, item.end_time)))
        for value_id, windows in grouped.items()
    }


def _resident_name_for_window(
    value_id: str,
    start_time: int,
    value_by_id: dict[str, JointValue],
    windows_by_value: dict[str, tuple[object, ...]],
) -> str:
    if not _uses_distinct_window_aliases(
        value_id,
        value_by_id=value_by_id,
        windows_by_value=windows_by_value,
    ):
        return value_id
    return f"{value_id}.resident@{start_time}"


def _resident_name_for_read(
    value_id: str,
    *,
    action_start: int,
    action_end: int,
    windows_by_value: dict[str, tuple[object, ...]],
    value_by_id: dict[str, JointValue],
) -> str:
    if not _uses_distinct_window_aliases(
        value_id,
        value_by_id=value_by_id,
        windows_by_value=windows_by_value,
    ):
        return value_id
    for window in windows_by_value.get(value_id, ()):
        if window.start_time <= action_start and action_end <= window.end_time:
            return _resident_name_for_window(
                value_id,
                window.start_time,
                value_by_id,
                windows_by_value,
            )
    return value_id


def _resident_name_for_write(
    value_id: str,
    *,
    writer_end: int,
    windows_by_value: dict[str, tuple[object, ...]],
    value_by_id: dict[str, JointValue],
) -> str:
    if not _uses_distinct_window_aliases(
        value_id,
        value_by_id=value_by_id,
        windows_by_value=windows_by_value,
    ):
        return value_id
    for window in windows_by_value.get(value_id, ()):
        if window.start_time == writer_end:
            return _resident_name_for_window(
                value_id,
                window.start_time,
                value_by_id,
                windows_by_value,
            )
    return value_id


def _read_window_start(
    action: JointAction,
    value_id: str,
    *,
    start_by_action: dict[str, int],
    end_by_action: dict[str, int],
    windows_by_value: dict[str, tuple[object, ...]],
    value_by_id: dict[str, JointValue],
) -> int | None:
    if value_id not in action.reads:
        return None
    if not _uses_distinct_window_aliases(
        value_id,
        value_by_id=value_by_id,
        windows_by_value=windows_by_value,
    ) and _base_home_tier(value_by_id[value_id]) is ScheduledValueHomeTier.SRAM:
        return 0
    for window in windows_by_value.get(value_id, ()):
        if (
            window.start_time <= start_by_action[action.action_id]
            and end_by_action[action.action_id] <= window.end_time
        ):
            return window.start_time
    return None


def _resident_name_for_previous_window(
    value_id: str,
    *,
    action_start: int,
    windows_by_value: dict[str, tuple[object, ...]],
    value_by_id: dict[str, JointValue],
) -> str:
    if not _uses_distinct_window_aliases(
        value_id,
        value_by_id=value_by_id,
        windows_by_value=windows_by_value,
    ):
        return value_id
    candidate = None
    for window in windows_by_value.get(value_id, ()):
        if window.end_time <= action_start:
            candidate = window
    if candidate is None:
        return value_id
    return _resident_name_for_window(
        value_id,
        candidate.start_time,
        value_by_id,
        windows_by_value,
    )


def _action_consumes_window(
    action: JointAction,
    *,
    action_id: str,
    value_id: str,
    window_start: int,
    start_by_action: dict[str, int],
    end_by_action: dict[str, int],
    windows_by_value: dict[str, tuple[object, ...]],
    value_by_id: dict[str, JointValue],
) -> bool:
    if action.kind is JointActionKind.RELOAD:
        return _resident_name_for_previous_window(
            value_id,
            action_start=start_by_action[action_id],
            windows_by_value=windows_by_value,
            value_by_id=value_by_id,
        ) == _resident_name_for_window(
            value_id,
            window_start,
            value_by_id,
            windows_by_value,
        )
    return _read_window_start(
        action,
        value_id,
        start_by_action=start_by_action,
        end_by_action=end_by_action,
        windows_by_value=windows_by_value,
        value_by_id=value_by_id,
    ) == window_start


def _resident_producer_step_id(
    value: JointValue,
    *,
    window_start: int | None,
    active_actions: dict[str, JointAction],
    end_by_action: dict[str, int],
) -> str | None:
    if value.producer is not None and value.producer.action_id in active_actions:
        if window_start is None or end_by_action[value.producer.action_id] == window_start:
            return value.producer.action_id
    for action_id, action in active_actions.items():
        if action.kind in (JointActionKind.DMA_IN, JointActionKind.RELOAD) and value.value_id in action.writes:
            if window_start is None or end_by_action[action_id] == window_start:
                return action_id
    return None


def _find_open_action_id(
    value_id: str,
    start_time: int,
    actions: tuple[JointAction, ...],
    end_by_action: dict[str, int],
) -> str:
    matches = sorted(
        action.action_id
        for action in actions
        if value_id in action.writes
        and action.kind in (JointActionKind.COMPUTE, JointActionKind.DMA_IN, JointActionKind.RELOAD)
        and end_by_action[action.action_id] == start_time
    )
    if not matches:
        raise ValueError(f"cannot resolve residency open for {value_id!r} at {start_time}")
    return matches[0]


def _find_close_action_id(
    value_id: str,
    end_time: int,
    actions: tuple[JointAction, ...],
    end_by_action: dict[str, int],
) -> str:
    matches = sorted(
        action.action_id
        for action in actions
        if value_id in action.reads and end_by_action[action.action_id] == end_time
    )
    if not matches:
        raise ValueError(f"cannot resolve residency close for {value_id!r} at {end_time}")
    return matches[0]


def _uses_distinct_window_aliases(
    value_id: str,
    *,
    value_by_id: dict[str, JointValue],
    windows_by_value: dict[str, tuple[object, ...]],
) -> bool:
    windows = windows_by_value.get(value_id, ())
    if not windows:
        return False
    if _base_home_tier(value_by_id[value_id]) is not ScheduledValueHomeTier.SRAM:
        return True
    return len(windows) > 1


def _pipeline_resource_kind(resource: JointAction | object) -> PipelineResourceKind:
    value = getattr(resource, "resource_kind", resource)
    value_str = getattr(value, "value", value)
    return {
        "DMA": PipelineResourceKind.DMA,
        "MATMUL": PipelineResourceKind.MATMUL,
        "SHAPE": PipelineResourceKind.SHAPE,
        "OTHER": PipelineResourceKind.OTHER,
    }[value_str]


def _transfer_step_kind(action: JointAction) -> TransferStepKind:
    return {
        JointActionKind.DMA_IN: TransferStepKind.DMA_IN,
        JointActionKind.DMA_OUT: TransferStepKind.DMA_OUT,
        JointActionKind.SPILL: TransferStepKind.SPILL_DMA,
        JointActionKind.RELOAD: TransferStepKind.RELOAD_DMA,
    }[action.kind]


def _base_home_tier(value: JointValue) -> ScheduledValueHomeTier:
    if value.producer is not None:
        if value.required_final_tier is JointValueTier.SRAM:
            return ScheduledValueHomeTier.SRAM
        return ScheduledValueHomeTier.SLOW
    return {
        JointValueTier.INPUT: ScheduledValueHomeTier.INPUT,
        JointValueTier.CONST: ScheduledValueHomeTier.CONST,
        JointValueTier.SLOW: ScheduledValueHomeTier.SLOW,
        JointValueTier.SRAM: ScheduledValueHomeTier.SRAM,
        JointValueTier.UNMATERIALIZED: ScheduledValueHomeTier.SLOW,
    }[value.initial_tier]


def _src_tier(
    action: JointAction,
    value_home_tier: ScheduledValueHomeTier,
) -> ScheduledValueHomeTier:
    return {
        JointActionKind.DMA_IN: value_home_tier,
        JointActionKind.DMA_OUT: ScheduledValueHomeTier.SRAM,
        JointActionKind.SPILL: ScheduledValueHomeTier.SRAM,
        JointActionKind.RELOAD: ScheduledValueHomeTier.SLOW,
    }[action.kind]


def _dst_tier(action: JointAction) -> ScheduledValueHomeTier:
    return {
        JointActionKind.DMA_IN: ScheduledValueHomeTier.SRAM,
        JointActionKind.DMA_OUT: ScheduledValueHomeTier.SLOW,
        JointActionKind.SPILL: ScheduledValueHomeTier.SLOW,
        JointActionKind.RELOAD: ScheduledValueHomeTier.SRAM,
    }[action.kind]


__all__ = ["materialize_joint_solution"]
