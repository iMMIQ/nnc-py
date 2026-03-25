"""Baseline heuristic list scheduler for pipeline scheduling."""

from __future__ import annotations

from collections.abc import Mapping
import math

from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    PipelineScheduleProblem,
    PipelineScheduleResult,
    ScheduledStep,
    ScheduleStep,
    SramAllocationInterval,
    SramValue,
)
from nnc_py.scheduler.base import PipelineScheduler


class ListPipelineScheduler(PipelineScheduler):
    """Simple deterministic list scheduler with conservative SRAM checks."""

    def solve(self, problem: PipelineScheduleProblem) -> PipelineScheduleResult:
        step_by_id = {step.id: step for step in problem.steps}
        if len(step_by_id) != len(problem.steps):
            return PipelineScheduleResult(
                feasible=False,
                solver_name="list",
                diagnostics={"reason": "duplicate_step_id"},
            )

        value_by_name = {value.name: value for value in problem.sram_values}
        if len(value_by_name) != len(problem.sram_values):
            return PipelineScheduleResult(
                feasible=False,
                solver_name="list",
                diagnostics={"reason": "duplicate_sram_value_name"},
            )

        successors: dict[str, list[str]] = {step.id: [] for step in problem.steps}
        predecessor_counts: dict[str, int] = {step.id: 0 for step in problem.steps}
        predecessors: dict[str, list[str]] = {step.id: [] for step in problem.steps}
        for edge in problem.edges:
            if edge.src_step_id not in step_by_id or edge.dst_step_id not in step_by_id:
                return PipelineScheduleResult(
                    feasible=False,
                    solver_name="list",
                    diagnostics={
                        "reason": "unknown_step_reference",
                        "src_step_id": edge.src_step_id,
                        "dst_step_id": edge.dst_step_id,
                    },
                )
            successors[edge.src_step_id].append(edge.dst_step_id)
            predecessors[edge.dst_step_id].append(edge.src_step_id)
            predecessor_counts[edge.dst_step_id] += 1

        validation_error = _validate_sram_metadata(
            problem=problem,
            step_by_id=step_by_id,
            value_by_name=value_by_name,
        )
        if validation_error is not None:
            return PipelineScheduleResult(
                feasible=False,
                solver_name="list",
                diagnostics=validation_error,
            )

        topological_order = _topological_order(
            step_ids=tuple(step.id for step in problem.steps),
            successors=successors,
            predecessor_counts=predecessor_counts,
        )
        if topological_order is None:
            return PipelineScheduleResult(
                feasible=False,
                solver_name="list",
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
                return PipelineScheduleResult(
                    feasible=False,
                    solver_name="list",
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
                    0 if step.resource_kind is PipelineResourceKind.MATMUL else 1,
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
                return PipelineScheduleResult(
                    feasible=False,
                    solver_name="list",
                    diagnostics={
                        "reason": "sram_capacity_exceeded",
                        "step_id": blocker.id,
                    },
                )

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
                return PipelineScheduleResult(
                    feasible=False,
                    solver_name="list",
                    diagnostics={
                        "reason": "sram_capacity_exceeded",
                        "step_id": selected_step.id,
                    },
                )
            end_time = start_time + selected_step.duration + selected_step.launch_overhead
            scheduled = ScheduledStep(
                step_id=selected_step.id,
                resource_kind=selected_step.resource_kind,
                resource_slot=0,
                start_time=start_time,
                end_time=end_time,
            )
            scheduled_steps[selected_step.id] = scheduled
            scheduled_order.append(selected_step.id)
            ready.remove(selected_step.id)
            resource_available[selected_step.resource_kind] = end_time
            for successor_id in successors[selected_step.id]:
                predecessor_counts[successor_id] -= 1
                if predecessor_counts[successor_id] == 0:
                    ready.add(successor_id)

        ordered_scheduled_steps = tuple(
            scheduled_steps[step.id] for step in problem.steps if step.id in scheduled_steps
        )
        makespan = max((step.end_time for step in scheduled_steps.values()), default=0)
        sram_intervals = _build_sram_intervals(
            problem.sram_values,
            scheduled_steps,
            makespan,
        )
        return PipelineScheduleResult(
            scheduled_steps=ordered_scheduled_steps,
            sram_intervals=sram_intervals,
            makespan=makespan,
            feasible=True,
            solver_name="list",
            diagnostics={
                "scheduled_order": tuple(scheduled_order),
                "critical_path_lengths": critical_path_lengths,
            },
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


def _validate_sram_metadata(
    *,
    problem: PipelineScheduleProblem,
    step_by_id: Mapping[str, ScheduleStep],
    value_by_name: Mapping[str, SramValue],
) -> dict[str, object] | None:
    produced_by_step: dict[str, set[str]] = {
        step.id: set(step.sram_output_names) for step in problem.steps
    }
    consumed_by_step: dict[str, set[str]] = {
        step.id: set(step.sram_input_names) for step in problem.steps
    }

    for step in problem.steps:
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

    for value in problem.sram_values:
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
            if value.name not in consumed_by_step[consumer_step_id]:
                return {
                    "reason": "sram_consumer_mismatch",
                    "step_id": consumer_step_id,
                    "value_name": value.name,
                }
    return None


def _initial_resource_availability(
    problem: PipelineScheduleProblem,
) -> dict[PipelineResourceKind, int]:
    resource_available: dict[PipelineResourceKind, int] = {
        resource: 0 for resource in problem.resources
    }
    for step in problem.steps:
        resource_available.setdefault(step.resource_kind, 0)
    return resource_available


def _earliest_feasible_start(
    *,
    step: ScheduleStep,
    step_by_id: Mapping[str, ScheduleStep],
    predecessors: list[str],
    scheduled_steps: Mapping[str, ScheduledStep],
    resource_available: Mapping[PipelineResourceKind, int],
    problem: PipelineScheduleProblem,
    value_by_name: Mapping[str, SramValue],
) -> int | None:
    dependency_ready_time = max(
        (scheduled_steps[step_id].end_time for step_id in predecessors),
        default=0,
    )
    start_time = max(
        dependency_ready_time,
        resource_available.get(step.resource_kind, 0),
    )
    duration = step.duration + step.launch_overhead
    if problem.sram_capacity_bytes < 0:
        return None

    reserved_bytes = step.sram_temp_bytes + sum(
        value_by_name[name].size_bytes
        for name in step.sram_output_names
        if name in value_by_name
    )
    if reserved_bytes > problem.sram_capacity_bytes:
        return None

    occupancy_intervals = _occupied_sram_intervals(
        problem=problem,
        step_by_id=step_by_id,
        scheduled_steps=scheduled_steps,
    )
    while True:
        end_time = start_time + duration
        violating_end_times = _violating_interval_end_times(
            start_time=start_time,
            end_time=end_time,
            reserved_bytes=reserved_bytes,
            occupancy_intervals=occupancy_intervals,
            capacity_bytes=problem.sram_capacity_bytes,
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
) -> tuple[tuple[int, float, int], ...]:
    intervals: list[tuple[int, float, int]] = []
    scheduled_step_ids = set(scheduled_steps)
    for step_id, scheduled_step in scheduled_steps.items():
        temp_bytes = step_by_id[step_id].sram_temp_bytes
        if temp_bytes <= 0 or scheduled_step.start_time >= scheduled_step.end_time:
            continue
        intervals.append(
            (scheduled_step.start_time, float(scheduled_step.end_time), temp_bytes)
        )

    for value in problem.sram_values:
        if value.producer_step_id is None:
            start_time = 0
        else:
            producer_schedule = scheduled_steps.get(value.producer_step_id)
            if producer_schedule is None:
                continue
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
            end_time = max(scheduled_consumer_end_times)
        elif value.must_reside_in_sram:
            end_time = math.inf
        else:
            end_time = float(start_time)
        if start_time < end_time:
            intervals.append((start_time, end_time, value.size_bytes))
    return tuple(intervals)


def _violating_interval_end_times(
    *,
    start_time: int,
    end_time: int,
    reserved_bytes: int,
    occupancy_intervals: tuple[tuple[int, float, int], ...],
    capacity_bytes: int,
) -> list[int] | None:
    if start_time >= end_time:
        return []

    checkpoints = {start_time}
    for interval_start, interval_end, _ in occupancy_intervals:
        if interval_start < end_time and start_time < interval_end:
            checkpoints.add(max(start_time, interval_start))
            if interval_end < end_time:
                checkpoints.add(int(interval_end))

    violates_capacity = False
    violating_end_times: set[int] = set()
    for checkpoint in sorted(checkpoints):
        occupied_bytes = 0
        for interval_start, interval_end, interval_bytes in occupancy_intervals:
            if interval_start <= checkpoint < interval_end:
                occupied_bytes += interval_bytes
        if occupied_bytes + reserved_bytes > capacity_bytes:
            violates_capacity = True
            for interval_start, interval_end, _interval_bytes in occupancy_intervals:
                if (
                    interval_start <= checkpoint < interval_end
                    and interval_end < math.inf
                ):
                    violating_end_times.add(int(interval_end))

    if not violates_capacity:
        return []
    if not violating_end_times:
        return None
    return sorted(end_time for end_time in violating_end_times if end_time > start_time)


def _build_sram_intervals(
    values: tuple[SramValue, ...],
    scheduled_steps: Mapping[str, ScheduledStep],
    makespan: int,
) -> tuple[SramAllocationInterval, ...]:
    intervals: list[SramAllocationInterval] = []
    for index, value in enumerate(values):
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
                buffer_id=f"buf{index}",
                start_time=start_time,
                end_time=end_time,
                size_bytes=value.size_bytes,
            )
        )
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
            0 if step.resource_kind is PipelineResourceKind.MATMUL else 1,
            step_index[step.id],
        ),
    )
