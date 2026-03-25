"""Scheduled-native memory planning for the explicit O3 pipeline path."""

from __future__ import annotations

from dataclasses import dataclass

from nnc_py.ir.context import CompileContext
from nnc_py.ir.pipeline_schedule import (
    PipelineScheduleProblem,
    PipelineScheduleResult,
    ResidencyWindow,
    ScheduledStep,
    ScheduledValue,
    ScheduledValueHomeTier,
    TransferStep,
    TransferStepKind,
)
from nnc_py.passes.base import PassBase
from nnc_py.passes.memory_plan import MemoryBuffer
from nnc_py.passes.memory_strategy import (
    LogicalMemoryRegion,
    MemoryAllocationPlan,
    ReloadPoint,
    SpillPoint,
    TensorAllocation,
)


_DEFAULT_ALIGNMENT = 16
_COMPAT_STRATEGY_NAME = "schedule_time_v4"


@dataclass(frozen=True)
class ScheduledFastAllocation:
    """Final SRAM placement for one scheduled value residency."""

    value_name: str
    buffer_id: int
    offset: int
    size_bytes: int
    start_time: int
    end_time: int


@dataclass(frozen=True)
class ScheduledSlowAllocation:
    """Final slow-tier placement for one spilled scheduled value."""

    value_name: str
    offset: int
    size_bytes: int


@dataclass(frozen=True)
class ScheduledTransferPoint:
    """Resolved transfer binding between fast and slow memory."""

    step_id: str
    transfer_kind: TransferStepKind
    value_name: str
    size_bytes: int
    start_time: int
    end_time: int
    fast_offset: int | None = None
    slow_offset: int | None = None
    resident_value_name: str | None = None
    after_node_name: str | None = None
    before_node_name: str | None = None


@dataclass(frozen=True)
class ScheduledMemoryPlan:
    """Canonical scheduled-O3 memory plan."""

    total_fast_memory: int
    total_slow_memory: int
    value_allocations: dict[str, ScheduledFastAllocation]
    slow_allocations: dict[str, ScheduledSlowAllocation]
    transfer_points: tuple[ScheduledTransferPoint, ...]


@dataclass(frozen=True)
class _TimedValueInterval:
    name: str
    start_time: int
    end_time: int
    size_bytes: int


@dataclass
class _BufferState:
    buffer: MemoryBuffer
    available_at: int = 0


class ScheduledMemoryPlanningPass(PassBase):
    """Build scheduled-native SRAM and slow-tier allocations."""

    @property
    def name(self) -> str:
        return "ScheduledMemoryPlanning"

    def _execute(self, ctx: CompileContext) -> None:
        scheduled_plan, compat_plan = self._build_plans(ctx)
        ctx.metadata["scheduled_memory_plan"] = scheduled_plan
        ctx.metadata["memory_allocation_plan"] = compat_plan
        ctx.metadata.pop("memory_plan", None)
        ctx.metadata.pop("spill_plan", None)

    def _build_plans(
        self,
        ctx: CompileContext,
    ) -> tuple[ScheduledMemoryPlan, MemoryAllocationPlan]:
        schedule_inputs = _collect_schedule_inputs(ctx)
        if schedule_inputs is None:
            return _empty_scheduled_plan(), _empty_compat_plan()

        problem, result = schedule_inputs
        scheduled_steps = {
            step.step_id: step
            for step in result.scheduled_steps
        }
        problem_steps = {
            step.id: step
            for step in problem.steps
        }
        scheduled_values = _resolve_scheduled_values(problem, result)
        residency_windows = _resolve_residency_windows(problem, result)
        intervals = _collect_fast_intervals(
            scheduled_values=scheduled_values,
            residency_windows=residency_windows,
            scheduled_steps=scheduled_steps,
            makespan=result.makespan,
        )
        buffer_states, interval_assignments = _allocate_fast_buffers(intervals)
        fast_allocations = _build_fast_allocations(
            intervals=intervals,
            buffer_states=buffer_states,
            interval_assignments=interval_assignments,
        )
        slow_allocations = _build_slow_allocations(
            scheduled_values=scheduled_values,
            problem_steps=problem_steps,
        )
        transfer_points = _build_transfer_points(
            scheduled_steps=scheduled_steps,
            problem_steps=problem_steps,
            scheduled_values=scheduled_values,
            fast_allocations=fast_allocations,
            slow_allocations=slow_allocations,
        )
        compat_plan = _build_compat_plan(
            ctx=ctx,
            buffer_states=buffer_states,
            intervals=intervals,
            interval_assignments=interval_assignments,
            fast_allocations=fast_allocations,
            slow_allocations=slow_allocations,
            transfer_points=transfer_points,
        )
        scheduled_plan = ScheduledMemoryPlan(
            total_fast_memory=compat_plan.total_fast_memory,
            total_slow_memory=compat_plan.total_slow_memory,
            value_allocations=fast_allocations,
            slow_allocations=slow_allocations,
            transfer_points=transfer_points,
        )
        return scheduled_plan, compat_plan


def _collect_schedule_inputs(
    ctx: CompileContext,
) -> tuple[PipelineScheduleProblem, PipelineScheduleResult] | None:
    try:
        problem = ctx.pipeline_schedule_problem
        result = ctx.pipeline_schedule_result
    except TypeError:
        return None

    if not isinstance(problem, PipelineScheduleProblem):
        return None
    if not isinstance(result, PipelineScheduleResult):
        return None
    if not result.feasible:
        return None
    return problem, result


def _resolve_scheduled_values(
    problem: PipelineScheduleProblem,
    result: PipelineScheduleResult,
) -> dict[str, ScheduledValue]:
    values = result.scheduled_values or problem.scheduled_values
    return {value.name: value for value in values}


def _resolve_residency_windows(
    problem: PipelineScheduleProblem,
    result: PipelineScheduleResult,
) -> tuple[ResidencyWindow, ...]:
    return result.residency_windows or problem.residency_windows


def _collect_fast_intervals(
    *,
    scheduled_values: dict[str, ScheduledValue],
    residency_windows: tuple[ResidencyWindow, ...],
    scheduled_steps: dict[str, ScheduledStep],
    makespan: int,
) -> list[_TimedValueInterval]:
    intervals: list[_TimedValueInterval] = []
    explicit_window_names = {window.value_name for window in residency_windows}

    for window in residency_windows:
        value = scheduled_values.get(window.value_name)
        if value is None or value.home_tier is not ScheduledValueHomeTier.SRAM:
            continue
        opened_by = scheduled_steps.get(window.opened_by_step_id)
        if opened_by is None:
            continue
        if window.closed_by_step_id is None:
            end_time = makespan
        else:
            closed_by = scheduled_steps.get(window.closed_by_step_id)
            if closed_by is None:
                continue
            end_time = closed_by.end_time
        interval = _TimedValueInterval(
            name=value.name,
            start_time=opened_by.end_time,
            end_time=end_time,
            size_bytes=value.size_bytes,
        )
        if _is_usable_interval(interval):
            intervals.append(interval)

    for value in scheduled_values.values():
        if value.name in explicit_window_names:
            continue
        interval = _default_interval_for_value(
            value=value,
            scheduled_steps=scheduled_steps,
            makespan=makespan,
        )
        if interval is not None:
            intervals.append(interval)

    intervals.sort(
        key=lambda interval: (
            interval.start_time,
            interval.end_time,
            interval.name,
        )
    )
    return intervals


def _default_interval_for_value(
    *,
    value: ScheduledValue,
    scheduled_steps: dict[str, ScheduledStep],
    makespan: int,
) -> _TimedValueInterval | None:
    if value.size_bytes <= 0 or value.home_tier is not ScheduledValueHomeTier.SRAM:
        return None

    if value.producer_step_id is None:
        start_time = 0
    else:
        producer = scheduled_steps.get(value.producer_step_id)
        if producer is None:
            return None
        start_time = producer.end_time

    consumer_end_times = [
        scheduled_steps[consumer_step_id].end_time
        for consumer_step_id in value.consumer_step_ids
        if consumer_step_id in scheduled_steps
    ]
    if consumer_end_times:
        end_time = max(consumer_end_times)
    elif value.must_reside_in_sram:
        end_time = makespan
    else:
        end_time = start_time

    interval = _TimedValueInterval(
        name=value.name,
        start_time=start_time,
        end_time=end_time,
        size_bytes=value.size_bytes,
    )
    if not _is_usable_interval(interval):
        return None
    return interval


def _allocate_fast_buffers(
    intervals: list[_TimedValueInterval],
) -> tuple[list[_BufferState], dict[str, int]]:
    buffer_states: list[_BufferState] = []
    interval_assignments: dict[str, int] = {}

    for interval in intervals:
        buffer_state = _pick_buffer(buffer_states, interval)
        if buffer_state is None:
            buffer_state = _BufferState(
                buffer=MemoryBuffer(
                    id=len(buffer_states),
                    offset=0,
                    size=0,
                    alignment=_DEFAULT_ALIGNMENT,
                )
            )
            buffer_states.append(buffer_state)

        aligned_size = _align(interval.size_bytes, _DEFAULT_ALIGNMENT)
        buffer_state.buffer.size = max(buffer_state.buffer.size, aligned_size)
        buffer_state.buffer.add_tensor(interval.name)
        buffer_state.available_at = interval.end_time
        interval_assignments[interval.name] = buffer_state.buffer.id

    _assign_offsets(buffer_states)
    return buffer_states, interval_assignments


def _build_fast_allocations(
    *,
    intervals: list[_TimedValueInterval],
    buffer_states: list[_BufferState],
    interval_assignments: dict[str, int],
) -> dict[str, ScheduledFastAllocation]:
    buffers_by_id = {state.buffer.id: state.buffer for state in buffer_states}
    allocations: dict[str, ScheduledFastAllocation] = {}
    for interval in intervals:
        buffer_id = interval_assignments[interval.name]
        buffer = buffers_by_id[buffer_id]
        allocations[interval.name] = ScheduledFastAllocation(
            value_name=interval.name,
            buffer_id=buffer_id,
            offset=buffer.offset,
            size_bytes=interval.size_bytes,
            start_time=interval.start_time,
            end_time=interval.end_time,
        )
    return allocations


def _build_slow_allocations(
    *,
    scheduled_values: dict[str, ScheduledValue],
    problem_steps: dict[str, object],
) -> dict[str, ScheduledSlowAllocation]:
    slow_sizes: dict[str, int] = {}
    for step in problem_steps.values():
        if not isinstance(step, TransferStep):
            continue
        if step.transfer_kind not in (
            TransferStepKind.SPILL_DMA,
            TransferStepKind.RELOAD_DMA,
        ):
            continue
        value = scheduled_values.get(step.moved_value_name)
        size_bytes = step.bytes
        if value is not None:
            size_bytes = max(size_bytes, value.size_bytes)
        if size_bytes <= 0:
            continue
        previous_size = slow_sizes.get(step.moved_value_name, 0)
        slow_sizes[step.moved_value_name] = max(previous_size, size_bytes)

    slow_allocations: dict[str, ScheduledSlowAllocation] = {}
    current_offset = 0
    for value_name in sorted(slow_sizes):
        slow_allocations[value_name] = ScheduledSlowAllocation(
            value_name=value_name,
            offset=current_offset,
            size_bytes=slow_sizes[value_name],
        )
        current_offset += _align(slow_sizes[value_name], _DEFAULT_ALIGNMENT)
    return slow_allocations


def _build_transfer_points(
    *,
    scheduled_steps: dict[str, ScheduledStep],
    problem_steps: dict[str, object],
    scheduled_values: dict[str, ScheduledValue],
    fast_allocations: dict[str, ScheduledFastAllocation],
    slow_allocations: dict[str, ScheduledSlowAllocation],
) -> tuple[ScheduledTransferPoint, ...]:
    transfer_points: list[ScheduledTransferPoint] = []

    for step_id, problem_step in problem_steps.items():
        if not isinstance(problem_step, TransferStep):
            continue
        if problem_step.transfer_kind not in (
            TransferStepKind.SPILL_DMA,
            TransferStepKind.RELOAD_DMA,
        ):
            continue
        scheduled_step = scheduled_steps.get(step_id)
        if scheduled_step is None:
            continue

        resident_value_name = None
        fast_offset = None
        after_node_name = None
        before_node_name = None
        if problem_step.transfer_kind is TransferStepKind.SPILL_DMA:
            fast_offset = _lookup_fast_offset(
                fast_allocations,
                problem_step.moved_value_name,
            )
            moved_value = scheduled_values.get(problem_step.moved_value_name)
            if moved_value is not None and moved_value.producer_step_id is not None:
                producer_step = problem_steps.get(moved_value.producer_step_id)
                after_node_name = getattr(producer_step, "node_name", None)
        else:
            resident_value_name = next(
                iter(problem_step.sram_output_names),
                None,
            )
            if resident_value_name is not None:
                fast_offset = _lookup_fast_offset(
                    fast_allocations,
                    resident_value_name,
                )
                resident_value = scheduled_values.get(resident_value_name)
                if resident_value is not None and resident_value.consumer_step_ids:
                    consumer_step = problem_steps.get(resident_value.consumer_step_ids[0])
                    before_node_name = getattr(consumer_step, "node_name", None)

        slow_offset = None
        slow_allocation = slow_allocations.get(problem_step.moved_value_name)
        if slow_allocation is not None:
            slow_offset = slow_allocation.offset

        size_bytes = problem_step.bytes
        if size_bytes <= 0:
            value = scheduled_values.get(problem_step.moved_value_name)
            size_bytes = 0 if value is None else value.size_bytes

        transfer_points.append(
            ScheduledTransferPoint(
                step_id=step_id,
                transfer_kind=problem_step.transfer_kind,
                value_name=problem_step.moved_value_name,
                size_bytes=size_bytes,
                start_time=scheduled_step.start_time,
                end_time=scheduled_step.end_time,
                fast_offset=fast_offset,
                slow_offset=slow_offset,
                resident_value_name=resident_value_name,
                after_node_name=after_node_name,
                before_node_name=before_node_name,
            )
        )

    transfer_points.sort(key=lambda point: (point.start_time, point.end_time, point.step_id))
    return tuple(transfer_points)


def _build_compat_plan(
    *,
    ctx: CompileContext,
    buffer_states: list[_BufferState],
    intervals: list[_TimedValueInterval],
    interval_assignments: dict[str, int],
    fast_allocations: dict[str, ScheduledFastAllocation],
    slow_allocations: dict[str, ScheduledSlowAllocation],
    transfer_points: tuple[ScheduledTransferPoint, ...],
) -> MemoryAllocationPlan:
    buffers = [state.buffer for state in buffer_states]
    tensor_allocations: dict[str, TensorAllocation] = {}
    tensor_to_buffer: dict[str, int] = {}
    logical_regions: dict[str, LogicalMemoryRegion] = {}

    for value_name, allocation in fast_allocations.items():
        tensor_allocations[value_name] = TensorAllocation(
            tensor_name=value_name,
            buffer_id=allocation.buffer_id,
            offset=allocation.offset,
            size=allocation.size_bytes,
        )
        tensor_to_buffer[value_name] = allocation.buffer_id
        logical_regions[value_name] = LogicalMemoryRegion(
            name=value_name,
            size_bytes=allocation.size_bytes,
            offset=allocation.offset,
        )

    spill_points, reload_points = _build_compat_transfer_points(
        ctx=ctx,
        transfer_points=transfer_points,
        fast_allocations=fast_allocations,
        slow_allocations=slow_allocations,
    )

    spill_bytes = sum(point.size for point in spill_points)
    reload_bytes = sum(point.size for point in reload_points)
    total_slow_memory = 0
    if slow_allocations:
        total_slow_memory = max(
            allocation.offset + _align(allocation.size_bytes, _DEFAULT_ALIGNMENT)
            for allocation in slow_allocations.values()
        )

    return MemoryAllocationPlan(
        strategy_name=_COMPAT_STRATEGY_NAME,
        total_fast_memory=sum(buffer.size for buffer in buffers),
        total_slow_memory=total_slow_memory,
        peak_memory=_calculate_peak_memory(intervals, buffer_states, interval_assignments),
        num_buffers=len(buffers),
        buffers=buffers,
        tensor_allocations=tensor_allocations,
        tensor_to_buffer=tensor_to_buffer,
        spill_points=spill_points,
        reload_points=reload_points,
        spill_bytes=spill_bytes,
        reload_bytes=reload_bytes,
        total_transfer_bytes=spill_bytes + reload_bytes,
        logical_regions=logical_regions,
    )


def _build_compat_transfer_points(
    *,
    ctx: CompileContext,
    transfer_points: tuple[ScheduledTransferPoint, ...],
    fast_allocations: dict[str, ScheduledFastAllocation],
    slow_allocations: dict[str, ScheduledSlowAllocation],
) -> tuple[list[SpillPoint], list[ReloadPoint]]:
    node_index_by_name = {
        node.name: index
        for index, node in enumerate(ctx.graph.topological_sort())
    }
    spill_points: list[SpillPoint] = []
    reload_points: list[ReloadPoint] = []

    for transfer_point in transfer_points:
        if transfer_point.slow_offset is None or transfer_point.fast_offset is None:
            continue

        if transfer_point.transfer_kind is TransferStepKind.SPILL_DMA:
            fast_allocation = fast_allocations.get(transfer_point.value_name)
            if fast_allocation is None:
                continue
            after_node = transfer_point.after_node_name or transfer_point.step_id
            spill_points.append(
                SpillPoint(
                    tensor_name=transfer_point.value_name,
                    after_node=after_node,
                    after_node_idx=node_index_by_name.get(after_node, -1),
                    from_buffer_id=fast_allocation.buffer_id,
                    from_fast_offset=transfer_point.fast_offset,
                    to_slow_offset=transfer_point.slow_offset,
                    size=transfer_point.size_bytes,
                )
            )
            continue

        if transfer_point.transfer_kind is TransferStepKind.RELOAD_DMA:
            resident_value_name = transfer_point.resident_value_name
            resident_allocation = (
                None
                if resident_value_name is None
                else fast_allocations.get(resident_value_name)
            )
            if resident_value_name is None or resident_allocation is None:
                continue
            before_node = transfer_point.before_node_name or transfer_point.step_id
            reload_points.append(
                ReloadPoint(
                    tensor_name=transfer_point.value_name,
                    before_node=before_node,
                    before_node_idx=node_index_by_name.get(before_node, -1),
                    from_slow_offset=transfer_point.slow_offset,
                    to_buffer_id=resident_allocation.buffer_id,
                    to_fast_offset=resident_allocation.offset,
                    size=transfer_point.size_bytes,
                )
            )

    return spill_points, reload_points


def _lookup_fast_offset(
    fast_allocations: dict[str, ScheduledFastAllocation],
    value_name: str,
) -> int | None:
    allocation = fast_allocations.get(value_name)
    if allocation is None:
        return None
    return allocation.offset


def _empty_scheduled_plan() -> ScheduledMemoryPlan:
    return ScheduledMemoryPlan(
        total_fast_memory=0,
        total_slow_memory=0,
        value_allocations={},
        slow_allocations={},
        transfer_points=(),
    )


def _empty_compat_plan() -> MemoryAllocationPlan:
    return MemoryAllocationPlan(
        strategy_name=_COMPAT_STRATEGY_NAME,
        total_fast_memory=0,
        peak_memory=0,
        num_buffers=0,
    )


def _is_usable_interval(interval: _TimedValueInterval) -> bool:
    return (
        bool(interval.name)
        and interval.start_time >= 0
        and interval.end_time >= interval.start_time
        and interval.size_bytes > 0
    )


def _pick_buffer(
    buffer_states: list[_BufferState],
    interval: _TimedValueInterval,
) -> _BufferState | None:
    reusable = [
        state for state in buffer_states if state.available_at <= interval.start_time
    ]
    if not reusable:
        return None

    aligned_size = _align(interval.size_bytes, _DEFAULT_ALIGNMENT)
    already_fits = [
        state for state in reusable if state.buffer.size >= aligned_size
    ]
    if already_fits:
        return min(
            already_fits,
            key=lambda state: (state.buffer.size, state.buffer.id),
        )
    return max(reusable, key=lambda state: (state.buffer.size, -state.buffer.id))


def _assign_offsets(buffer_states: list[_BufferState]) -> None:
    current_offset = 0
    for state in buffer_states:
        state.buffer.offset = current_offset
        current_offset += state.buffer.size


def _calculate_peak_memory(
    intervals: list[_TimedValueInterval],
    buffer_states: list[_BufferState],
    interval_assignments: dict[str, int],
) -> int:
    if not intervals:
        return 0

    buffer_sizes = {
        state.buffer.id: state.buffer.size
        for state in buffer_states
    }
    checkpoints = sorted(
        {interval.start_time for interval in intervals}
        | {interval.end_time for interval in intervals}
    )
    peak_memory = 0
    for checkpoint in checkpoints:
        active_buffers = {
            interval_assignments[interval.name]
            for interval in intervals
            if interval.start_time <= checkpoint < interval.end_time
        }
        peak_memory = max(
            peak_memory,
            sum(buffer_sizes[buffer_id] for buffer_id in active_buffers),
        )
    return peak_memory


def _align(value: int, alignment: int) -> int:
    if value <= 0:
        return 0
    return ((value + alignment - 1) // alignment) * alignment


__all__ = [
    "ScheduledFastAllocation",
    "ScheduledMemoryPlan",
    "ScheduledMemoryPlanningPass",
    "ScheduledSlowAllocation",
    "ScheduledTransferPoint",
]
