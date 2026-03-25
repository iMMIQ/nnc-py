"""Baseline heuristic list scheduler for pipeline scheduling."""

from __future__ import annotations

from collections.abc import Mapping
import math

from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    PipelineScheduleProblem,
    PipelineScheduleResult,
    ResidencyWindow,
    ScheduledStep,
    ScheduleStep,
    ScheduleStepKind,
    ScheduledValue,
    ScheduledValueHomeTier,
    SramAllocationInterval,
    TransferStep,
)
from nnc_py.scheduler.base import PipelineScheduler


class ListPipelineScheduler(PipelineScheduler):
    """Simple deterministic list scheduler with conservative SRAM checks."""

    def solve(self, problem: PipelineScheduleProblem) -> PipelineScheduleResult:
        step_by_id = {step.id: step for step in problem.steps}
        if len(step_by_id) != len(problem.steps):
            return _infeasible_result(
                problem,
                diagnostics={"reason": "duplicate_step_id"},
            )

        value_by_name = {value.name: value for value in problem.scheduled_values}
        if len(value_by_name) != len(problem.scheduled_values):
            return _infeasible_result(
                problem,
                diagnostics={"reason": "duplicate_sram_value_name"},
            )

        successors: dict[str, list[str]] = {step.id: [] for step in problem.steps}
        predecessor_counts: dict[str, int] = {step.id: 0 for step in problem.steps}
        predecessors: dict[str, list[str]] = {step.id: [] for step in problem.steps}
        for edge in problem.edges:
            if edge.src_step_id not in step_by_id or edge.dst_step_id not in step_by_id:
                return _infeasible_result(
                    problem,
                    diagnostics={
                        "reason": "unknown_step_reference",
                        "src_step_id": edge.src_step_id,
                        "dst_step_id": edge.dst_step_id,
                    },
                )
            successors[edge.src_step_id].append(edge.dst_step_id)
            predecessors[edge.dst_step_id].append(edge.src_step_id)
            predecessor_counts[edge.dst_step_id] += 1

        validation_error = _validate_schedule_metadata(
            problem=problem,
            step_by_id=step_by_id,
            value_by_name=value_by_name,
        )
        if validation_error is not None:
            return _infeasible_result(problem, diagnostics=validation_error)

        topological_order = _topological_order(
            step_ids=tuple(step.id for step in problem.steps),
            successors=successors,
            predecessor_counts=predecessor_counts,
        )
        if topological_order is None:
            return _infeasible_result(
                problem,
                diagnostics={"reason": "cyclic_dependencies"},
            )

        step_index = {step.id: index for index, step in enumerate(problem.steps)}
        critical_path_lengths = _compute_critical_path_lengths(
            problem.steps,
            successors,
            topological_order,
        )
        resource_available = _initial_resource_availability(problem)
        scheduled_steps: dict[str, ScheduledStep] = {}
        ready = {
            step.id
            for step in problem.steps
            if predecessor_counts[step.id] == 0
        }
        scheduled_order: list[str] = []

        while len(scheduled_steps) < len(problem.steps):
            if not ready:
                return _infeasible_result(
                    problem,
                    diagnostics={"reason": "cyclic_dependencies"},
                )

            candidates: list[tuple[int, tuple[object, ...], ScheduleStep]] = []
            for step_id in sorted(ready, key=step_index.__getitem__):
                step = step_by_id[step_id]
                earliest_start = _earliest_feasible_start(
                    step=step,
                    step_by_id=step_by_id,
                    predecessors=predecessors[step.id],
                    scheduled_steps=scheduled_steps,
                    resource_available=resource_available,
                    problem=problem,
                    value_by_name=value_by_name,
                )
                if earliest_start is None:
                    continue
                priority = (
                    -critical_path_lengths[step.id],
                    0 if _effective_resource_kind(step) is PipelineResourceKind.MATMUL else 1,
                    step_index[step.id],
                )
                candidates.append((earliest_start, priority, step))

            if not candidates:
                blocker = _pick_blocked_ready_step(
                    ready=ready,
                    step_by_id=step_by_id,
                    critical_path_lengths=critical_path_lengths,
                    step_index=step_index,
                )
                return _budget_failure_result(problem, step_id=blocker.id)

            _, _, selected_step = min(candidates, key=lambda item: (item[0], item[1]))
            start_time = _earliest_feasible_start(
                step=selected_step,
                step_by_id=step_by_id,
                predecessors=predecessors[selected_step.id],
                scheduled_steps=scheduled_steps,
                resource_available=resource_available,
                problem=problem,
                value_by_name=value_by_name,
            )
            if start_time is None:
                return _budget_failure_result(problem, step_id=selected_step.id)

            end_time = start_time + selected_step.duration + selected_step.launch_overhead
            resource_kind = _effective_resource_kind(selected_step)
            scheduled = ScheduledStep(
                step_id=selected_step.id,
                resource_kind=resource_kind,
                resource_slot=0,
                start_time=start_time,
                end_time=end_time,
            )
            scheduled_steps[selected_step.id] = scheduled
            scheduled_order.append(selected_step.id)
            ready.remove(selected_step.id)
            resource_available[resource_kind] = end_time
            for successor_id in successors[selected_step.id]:
                predecessor_counts[successor_id] -= 1
                if predecessor_counts[successor_id] == 0:
                    ready.add(successor_id)

        ordered_scheduled_steps = tuple(
            scheduled_steps[step.id] for step in problem.steps if step.id in scheduled_steps
        )
        makespan = max((step.end_time for step in scheduled_steps.values()), default=0)
        sram_intervals = _build_sram_intervals(
            problem=problem,
            scheduled_steps=scheduled_steps,
            makespan=makespan,
        )
        return PipelineScheduleResult(
            scheduled_steps=ordered_scheduled_steps,
            sram_intervals=sram_intervals,
            scheduled_values=problem.scheduled_values,
            residency_windows=problem.residency_windows,
            makespan=makespan,
            feasible=True,
            solver_name="list",
            diagnostics={
                "scheduled_order": tuple(scheduled_order),
                "critical_path_lengths": critical_path_lengths,
            },
        )


def _infeasible_result(
    problem: PipelineScheduleProblem,
    *,
    diagnostics: Mapping[str, object],
) -> PipelineScheduleResult:
    return PipelineScheduleResult(
        feasible=False,
        solver_name="list",
        scheduled_values=problem.scheduled_values,
        residency_windows=problem.residency_windows,
        diagnostics=diagnostics,
    )


def _compute_critical_path_lengths(
    steps: tuple[ScheduleStep, ...],
    successors: Mapping[str, list[str]],
    topological_order: tuple[str, ...],
) -> dict[str, int]:
    step_by_id = {step.id: step for step in steps}
    critical_path_lengths: dict[str, int] = {}
    for step_id in reversed(topological_order):
        successor_lengths = [
            critical_path_lengths[successor_id]
            for successor_id in successors[step_id]
        ]
        critical_path_lengths[step_id] = step_by_id[step_id].duration + (
            max(successor_lengths) if successor_lengths else 0
        )
    return critical_path_lengths


def _budget_failure_result(
    problem: PipelineScheduleProblem,
    *,
    step_id: str,
) -> PipelineScheduleResult:
    return _infeasible_result(
        problem,
        diagnostics={
            "reason": "no_feasible_schedule_under_budget",
            "step_id": step_id,
        },
    )


def _topological_order(
    *,
    step_ids: tuple[str, ...],
    successors: Mapping[str, list[str]],
    predecessor_counts: Mapping[str, int],
) -> tuple[str, ...] | None:
    remaining_predecessors = dict(predecessor_counts)
    ready = [
        step_id
        for step_id in step_ids
        if remaining_predecessors.get(step_id, 0) == 0
    ]
    order: list[str] = []
    while ready:
        step_id = ready.pop(0)
        order.append(step_id)
        for successor_id in successors[step_id]:
            remaining_predecessors[successor_id] -= 1
            if remaining_predecessors[successor_id] == 0:
                ready.append(successor_id)
    if len(order) != len(step_ids):
        return None
    return tuple(order)


def _validate_schedule_metadata(
    *,
    problem: PipelineScheduleProblem,
    step_by_id: Mapping[str, ScheduleStep],
    value_by_name: Mapping[str, ScheduledValue],
) -> dict[str, object] | None:
    produced_by_step: dict[str, set[str]] = {
        step.id: set(step.sram_output_names) for step in problem.steps
    }

    for step in problem.steps:
        if isinstance(step, TransferStep) and step.moved_value_name not in value_by_name:
            return {
                "reason": "unknown_scheduled_value_reference",
                "step_id": step.id,
                "value_name": step.moved_value_name,
            }

        for value_name in step.sram_output_names:
            if value_name not in value_by_name:
                return {
                    "reason": "unknown_sram_value_reference",
                    "step_id": step.id,
                    "value_name": value_name,
                }
            if value_by_name[value_name].producer_step_id != step.id:
                return {
                    "reason": "sram_producer_mismatch",
                    "step_id": step.id,
                    "value_name": value_name,
                }

        for value_name in step.sram_input_names:
            if value_name not in value_by_name:
                return {
                    "reason": "unknown_sram_value_reference",
                    "step_id": step.id,
                    "value_name": value_name,
                }
            if step.id not in value_by_name[value_name].consumer_step_ids:
                return {
                    "reason": "sram_consumer_mismatch",
                    "step_id": step.id,
                    "value_name": value_name,
                }

    for value in problem.scheduled_values:
        if value.producer_step_id is not None and value.producer_step_id not in step_by_id:
            return {
                "reason": "unknown_sram_step_reference",
                "value_name": value.name,
                "step_id": value.producer_step_id,
            }
        for consumer_step_id in value.consumer_step_ids:
            if consumer_step_id not in step_by_id:
                return {
                    "reason": "unknown_sram_step_reference",
                    "value_name": value.name,
                    "step_id": consumer_step_id,
                }
        if (
            value.producer_step_id is not None
            and value.name not in produced_by_step[value.producer_step_id]
        ):
            return {
                "reason": "sram_producer_mismatch",
                "step_id": value.producer_step_id,
                "value_name": value.name,
            }
        for consumer_step_id in value.consumer_step_ids:
            if not _step_consumes_value(step_by_id[consumer_step_id], value.name):
                return {
                    "reason": "sram_consumer_mismatch",
                    "step_id": consumer_step_id,
                    "value_name": value.name,
                }

    for window in problem.residency_windows:
        if window.value_name not in value_by_name:
            return {
                "reason": "unknown_residency_value_reference",
                "value_name": window.value_name,
                "residency_id": window.residency_id,
            }
        if window.opened_by_step_id not in step_by_id:
            return {
                "reason": "unknown_residency_step_reference",
                "step_id": window.opened_by_step_id,
                "residency_id": window.residency_id,
            }
        if (
            window.closed_by_step_id is not None
            and window.closed_by_step_id not in step_by_id
        ):
            return {
                "reason": "unknown_residency_step_reference",
                "step_id": window.closed_by_step_id,
                "residency_id": window.residency_id,
            }

    return None


def _step_consumes_value(step: ScheduleStep, value_name: str) -> bool:
    if value_name in step.sram_input_names:
        return True
    return isinstance(step, TransferStep) and step.moved_value_name == value_name


def _initial_resource_availability(
    problem: PipelineScheduleProblem,
) -> dict[PipelineResourceKind, int]:
    resource_available: dict[PipelineResourceKind, int] = {
        resource: 0 for resource in problem.resources
    }
    for step in problem.steps:
        resource_available.setdefault(_effective_resource_kind(step), 0)
    return resource_available


def _effective_resource_kind(step: ScheduleStep) -> PipelineResourceKind:
    if step.step_kind in {
        ScheduleStepKind.DMA_IN,
        ScheduleStepKind.SPILL_DMA,
        ScheduleStepKind.RELOAD_DMA,
        ScheduleStepKind.DMA_OUT,
    }:
        return PipelineResourceKind.DMA
    return step.resource_kind


def _earliest_feasible_start(
    *,
    step: ScheduleStep,
    step_by_id: Mapping[str, ScheduleStep],
    predecessors: list[str],
    scheduled_steps: Mapping[str, ScheduledStep],
    resource_available: Mapping[PipelineResourceKind, int],
    problem: PipelineScheduleProblem,
    value_by_name: Mapping[str, ScheduledValue],
) -> int | None:
    dependency_ready_time = max(
        (scheduled_steps[step_id].end_time for step_id in predecessors),
        default=0,
    )
    resource_kind = _effective_resource_kind(step)
    start_time = max(
        dependency_ready_time,
        resource_available.get(resource_kind, 0),
    )
    duration = step.duration + step.launch_overhead
    capacity_bytes = problem.sram_capacity_bytes
    if capacity_bytes < 0:
        return None
    if capacity_bytes == 0:
        return start_time
    if step.sram_temp_bytes > capacity_bytes:
        return None

    while True:
        end_time = start_time + duration
        tentative_scheduled_steps = dict(scheduled_steps)
        tentative_scheduled_steps[step.id] = ScheduledStep(
            step_id=step.id,
            resource_kind=resource_kind,
            resource_slot=0,
            start_time=start_time,
            end_time=end_time,
        )
        occupancy_intervals = _occupied_sram_intervals(
            problem=problem,
            step_by_id=step_by_id,
            scheduled_steps=tentative_scheduled_steps,
            value_by_name=value_by_name,
            candidate_step_id=step.id,
        )
        violating_end_times = _violating_interval_end_times(
            start_time=start_time,
            occupancy_intervals=occupancy_intervals,
            capacity_bytes=capacity_bytes,
        )
        if violating_end_times == []:
            return start_time
        if violating_end_times is None:
            return None
        start_time = min(violating_end_times)


def _occupied_sram_intervals(
    *,
    problem: PipelineScheduleProblem,
    step_by_id: Mapping[str, ScheduleStep],
    scheduled_steps: Mapping[str, ScheduledStep],
    value_by_name: Mapping[str, ScheduledValue],
    candidate_step_id: str | None,
) -> tuple[tuple[int, float, int, bool], ...]:
    intervals: list[tuple[int, float, int, bool]] = []
    scheduled_step_ids = set(scheduled_steps)

    for step_id, scheduled_step in scheduled_steps.items():
        temp_bytes = step_by_id[step_id].sram_temp_bytes
        if temp_bytes <= 0 or scheduled_step.start_time >= scheduled_step.end_time:
            continue
        intervals.append(
            (
                scheduled_step.start_time,
                float(scheduled_step.end_time),
                temp_bytes,
                step_id != candidate_step_id,
            )
        )

    explicit_window_names = {window.value_name for window in problem.residency_windows}
    for step_id, scheduled_step in scheduled_steps.items():
        producer_output_bytes = _explicit_window_output_bytes(
            step=step_by_id[step_id],
            value_by_name=value_by_name,
            explicit_window_names=explicit_window_names,
        )
        if (
            producer_output_bytes <= 0
            or scheduled_step.start_time >= scheduled_step.end_time
        ):
            continue
        intervals.append(
            (
                scheduled_step.start_time,
                float(scheduled_step.end_time),
                producer_output_bytes,
                step_id != candidate_step_id,
            )
        )

    if candidate_step_id is not None:
        candidate_schedule = scheduled_steps.get(candidate_step_id)
        candidate_step = step_by_id.get(candidate_step_id)
        if candidate_schedule is not None and candidate_step is not None:
            output_bytes = _candidate_output_bytes(
                step=candidate_step,
                value_by_name=value_by_name,
                explicit_window_names=explicit_window_names,
            )
            if (
                output_bytes > 0
                and candidate_schedule.start_time < candidate_schedule.end_time
            ):
                intervals.append(
                    (
                        candidate_schedule.start_time,
                        float(candidate_schedule.end_time),
                        output_bytes,
                        False,
                    )
                )

    for window in problem.residency_windows:
        interval = _timed_residency_window(
            window=window,
            value_by_name=value_by_name,
            scheduled_steps=scheduled_steps,
            candidate_step_id=candidate_step_id,
        )
        if interval is not None and interval[0] < interval[1]:
            intervals.append(interval)

    for value in problem.scheduled_values:
        if value.name in explicit_window_names:
            continue
        interval = _timed_default_value_interval(
            value=value,
            scheduled_step_ids=scheduled_step_ids,
            scheduled_steps=scheduled_steps,
            candidate_step_id=candidate_step_id,
        )
        if interval is not None and interval[0] < interval[1]:
            intervals.append(interval)

    return tuple(intervals)


def _explicit_window_output_bytes(
    *,
    step: ScheduleStep,
    value_by_name: Mapping[str, ScheduledValue],
    explicit_window_names: set[str],
) -> int:
    return sum(
        value.size_bytes
        for value_name in step.sram_output_names
        if (value := value_by_name.get(value_name)) is not None
        and value.name in explicit_window_names
        and value.home_tier is ScheduledValueHomeTier.SRAM
        and value.size_bytes > 0
    )


def _candidate_output_bytes(
    *,
    step: ScheduleStep,
    value_by_name: Mapping[str, ScheduledValue],
    explicit_window_names: set[str],
) -> int:
    return sum(
        value.size_bytes
        for value_name in step.sram_output_names
        if (value := value_by_name.get(value_name)) is not None
        and value.name not in explicit_window_names
        and value.home_tier is ScheduledValueHomeTier.SRAM
        and value.size_bytes > 0
    )


def _timed_residency_window(
    *,
    window: ResidencyWindow,
    value_by_name: Mapping[str, ScheduledValue],
    scheduled_steps: Mapping[str, ScheduledStep],
    candidate_step_id: str | None,
) -> tuple[int, float, int, bool] | None:
    value = value_by_name.get(window.value_name)
    if value is None or value.size_bytes <= 0:
        return None
    if value.home_tier is not ScheduledValueHomeTier.SRAM:
        return None

    opened_by = scheduled_steps.get(window.opened_by_step_id)
    if opened_by is None:
        return None
    start_time = opened_by.end_time

    if window.closed_by_step_id is None:
        end_time = math.inf
    else:
        closed_by = scheduled_steps.get(window.closed_by_step_id)
        end_time = math.inf if closed_by is None else float(closed_by.end_time)

    return (
        start_time,
        end_time,
        value.size_bytes,
        window.opened_by_step_id != candidate_step_id
        and window.closed_by_step_id != candidate_step_id,
    )


def _timed_default_value_interval(
    *,
    value: ScheduledValue,
    scheduled_step_ids: set[str],
    scheduled_steps: Mapping[str, ScheduledStep],
    candidate_step_id: str | None,
) -> tuple[int, float, int, bool] | None:
    if value.size_bytes <= 0 or value.home_tier is not ScheduledValueHomeTier.SRAM:
        return None

    if value.producer_step_id is None:
        start_time = 0
    else:
        producer_schedule = scheduled_steps.get(value.producer_step_id)
        if producer_schedule is None:
            return None
        start_time = producer_schedule.end_time

    scheduled_consumer_end_times = [
        scheduled_steps[consumer_id].end_time
        for consumer_id in value.consumer_step_ids
        if consumer_id in scheduled_step_ids
    ]
    has_unscheduled_consumers = any(
        consumer_id not in scheduled_step_ids
        for consumer_id in value.consumer_step_ids
    )
    if has_unscheduled_consumers:
        end_time = math.inf
    elif scheduled_consumer_end_times:
        end_time = float(max(scheduled_consumer_end_times))
    elif value.must_reside_in_sram:
        end_time = math.inf
    else:
        end_time = float(start_time)

    candidate_schedule = (
        None if candidate_step_id is None else scheduled_steps.get(candidate_step_id)
    )
    fixed = True
    if value.producer_step_id == candidate_step_id:
        fixed = False
    elif candidate_step_id is not None and candidate_step_id in value.consumer_step_ids:
        if has_unscheduled_consumers:
            fixed = True
        elif candidate_schedule is None:
            fixed = False
        else:
            other_consumer_end_times = [
                scheduled_steps[consumer_id].end_time
                for consumer_id in value.consumer_step_ids
                if consumer_id in scheduled_step_ids and consumer_id != candidate_step_id
            ]
            fixed = bool(
                other_consumer_end_times
                and max(other_consumer_end_times) >= candidate_schedule.end_time
            )

    return (
        start_time,
        end_time,
        value.size_bytes,
        fixed,
    )


def _violating_interval_end_times(
    *,
    start_time: int,
    occupancy_intervals: tuple[tuple[int, float, int, bool], ...],
    capacity_bytes: int,
) -> list[int] | None:
    checkpoints = {start_time}
    for interval_start, interval_end, _, _ in occupancy_intervals:
        if interval_end <= start_time:
            continue
        checkpoints.add(max(start_time, interval_start))
        if interval_end < math.inf:
            checkpoints.add(max(start_time, int(interval_end)))

    violates_capacity = False
    violating_end_times: set[int] = set()
    for checkpoint in sorted(checkpoints):
        occupied_bytes = 0
        for interval_start, interval_end, interval_bytes, _ in occupancy_intervals:
            if interval_start <= checkpoint < interval_end:
                occupied_bytes += interval_bytes

        if occupied_bytes <= capacity_bytes:
            continue

        violates_capacity = True
        for interval_start, interval_end, _interval_bytes, is_fixed in occupancy_intervals:
            if (
                interval_start <= checkpoint < interval_end
                and is_fixed
                and interval_end < math.inf
                and interval_end > start_time
            ):
                violating_end_times.add(int(interval_end))

    if not violates_capacity:
        return []
    if not violating_end_times:
        return None
    return sorted(violating_end_times)


def _build_sram_intervals(
    *,
    problem: PipelineScheduleProblem,
    scheduled_steps: Mapping[str, ScheduledStep],
    makespan: int,
) -> tuple[SramAllocationInterval, ...]:
    value_by_name = {value.name: value for value in problem.scheduled_values}
    intervals: list[SramAllocationInterval] = []
    explicit_window_names = {window.value_name for window in problem.residency_windows}
    next_buffer_id = 0

    for window in problem.residency_windows:
        value = value_by_name.get(window.value_name)
        if value is None or value.size_bytes <= 0:
            continue
        if value.home_tier is not ScheduledValueHomeTier.SRAM:
            continue
        opened_by = scheduled_steps.get(window.opened_by_step_id)
        if opened_by is None:
            continue
        start_time = opened_by.end_time
        if window.closed_by_step_id is None:
            end_time = makespan
        else:
            closed_by = scheduled_steps.get(window.closed_by_step_id)
            if closed_by is None:
                continue
            end_time = closed_by.end_time
        intervals.append(
            SramAllocationInterval(
                value_name=window.value_name,
                buffer_id=f"buf{next_buffer_id}",
                start_time=start_time,
                end_time=end_time,
                size_bytes=value.size_bytes,
            )
        )
        next_buffer_id += 1

    for value in problem.scheduled_values:
        if value.name in explicit_window_names:
            continue
        if value.size_bytes <= 0 or value.home_tier is not ScheduledValueHomeTier.SRAM:
            continue
        if value.producer_step_id is None:
            start_time = 0
        else:
            producer_schedule = scheduled_steps.get(value.producer_step_id)
            if producer_schedule is None:
                continue
            start_time = producer_schedule.end_time

        consumer_end_times = [
            scheduled_steps[consumer_id].end_time
            for consumer_id in value.consumer_step_ids
            if consumer_id in scheduled_steps
        ]
        if consumer_end_times:
            end_time = max(consumer_end_times)
        elif value.must_reside_in_sram:
            end_time = makespan
        else:
            end_time = start_time

        intervals.append(
            SramAllocationInterval(
                value_name=value.name,
                buffer_id=f"buf{next_buffer_id}",
                start_time=start_time,
                end_time=end_time,
                size_bytes=value.size_bytes,
            )
        )
        next_buffer_id += 1

    return tuple(intervals)


def _pick_blocked_ready_step(
    *,
    ready: set[str],
    step_by_id: Mapping[str, ScheduleStep],
    critical_path_lengths: Mapping[str, int],
    step_index: Mapping[str, int],
) -> ScheduleStep:
    return min(
        (step_by_id[step_id] for step_id in ready),
        key=lambda step: (
            -critical_path_lengths[step.id],
            0 if _effective_resource_kind(step) is PipelineResourceKind.MATMUL else 1,
            step_index[step.id],
        ),
    )
