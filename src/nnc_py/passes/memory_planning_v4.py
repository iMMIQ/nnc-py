"""Time-aware SRAM memory planning using scheduled live intervals."""

from __future__ import annotations

from dataclasses import dataclass

from nnc_py.ir.context import CompileContext
from nnc_py.ir.pipeline_schedule import (
    PipelineScheduleProblem,
    PipelineScheduleResult,
    ScheduledStep,
    SramValue,
)
from nnc_py.passes.base import PassBase
from nnc_py.passes.memory_plan import MemoryBuffer
from nnc_py.passes.memory_planning import (
    MemoryPlanningPassV2,
    _resolve_max_memory_budget,
    _should_use_tile_aware_v3,
    allocate_tile_regions,
)
from nnc_py.passes.memory_strategy import (
    LogicalMemoryRegion,
    MemoryAllocationPlan,
    TensorAllocation,
)


_DEFAULT_ALIGNMENT = 16
_STRATEGY_NAME = "schedule_time_v4"


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


class MemoryPlanningPassV4(PassBase):
    """Build a fast-SRAM allocation plan from scheduled producer/consumer timing."""

    @property
    def name(self) -> str:
        return "MemoryPlanningV4"

    def _execute(self, ctx: CompileContext) -> None:
        plan = self._build_plan(ctx)
        max_memory = _resolve_max_memory_budget(ctx)
        if max_memory is not None:
            execution_plans = ctx.node_execution_plans
            if plan.total_fast_memory > max_memory and _should_use_tile_aware_v3(
                ctx, execution_plans
            ):
                tile_plan = allocate_tile_regions(ctx)
                if tile_plan.total_fast_memory <= max_memory:
                    ctx.metadata["memory_allocation_plan"] = tile_plan
                    ctx.metadata["memory_budget_satisfied_by_v3"] = max_memory
                    ctx.metadata.pop("memory_plan", None)
                    ctx.metadata.pop("max_memory", None)
                    return

            if 0 < plan.total_fast_memory <= max_memory:
                ctx.metadata["memory_allocation_plan"] = plan
                ctx.metadata["memory_budget_satisfied_by_v4"] = max_memory
                MemoryPlanningPassV2()._store_legacy_formats(ctx, plan)
                ctx.metadata.pop("max_memory", None)
                return

        ctx.metadata["memory_allocation_plan"] = plan
        MemoryPlanningPassV2()._store_legacy_formats(ctx, plan)

    def _build_plan(self, ctx: CompileContext) -> MemoryAllocationPlan:
        intervals = self._collect_intervals(ctx)
        if intervals is None:
            return _empty_plan()

        buffer_states: list[_BufferState] = []
        assignments: dict[str, int] = {}

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
            assignments[interval.name] = buffer_state.buffer.id

        _assign_offsets(buffer_states)
        tensor_allocations, tensor_to_buffer, logical_regions = _build_tensor_mappings(
            intervals,
            buffer_states,
            assignments,
        )
        peak_memory = _calculate_peak_memory(intervals, buffer_states, assignments)
        total_fast_memory = sum(state.buffer.size for state in buffer_states)

        return MemoryAllocationPlan(
            strategy_name=_STRATEGY_NAME,
            total_fast_memory=total_fast_memory,
            peak_memory=peak_memory,
            num_buffers=len(buffer_states),
            buffers=[state.buffer for state in buffer_states],
            tensor_allocations=tensor_allocations,
            tensor_to_buffer=tensor_to_buffer,
            logical_regions=logical_regions,
        )

    def _collect_intervals(
        self,
        ctx: CompileContext,
    ) -> list[_TimedValueInterval] | None:
        try:
            schedule_problem = ctx.pipeline_schedule_problem
            schedule_result = ctx.pipeline_schedule_result
        except TypeError:
            return None

        if not isinstance(schedule_problem, PipelineScheduleProblem):
            return None
        if not isinstance(schedule_result, PipelineScheduleResult):
            return None
        if not schedule_result.feasible:
            return None

        scheduled_steps = {
            step.step_id: step
            for step in schedule_result.scheduled_steps
        }
        normalized: list[_TimedValueInterval] = []
        seen_names: set[str] = set()
        for value in schedule_problem.sram_values:
            size_bytes = _resolve_value_size_bytes(ctx, value)
            if size_bytes <= 0:
                return None
            interval = _derive_interval(
                SramValue(
                    name=value.name,
                    size_bytes=size_bytes,
                    producer_step_id=value.producer_step_id,
                    consumer_step_ids=value.consumer_step_ids,
                    must_reside_in_sram=value.must_reside_in_sram,
                    can_alias=value.can_alias,
                ),
                scheduled_steps,
                schedule_result.makespan,
            )
            if interval is None:
                return None
            if interval.name in seen_names:
                return None
            seen_names.add(interval.name)
            normalized.append(interval)

        normalized.sort(
            key=lambda interval: (
                interval.start_time,
                interval.end_time,
                interval.name,
            )
        )
        return normalized


def _resolve_value_size_bytes(ctx: CompileContext, value: SramValue) -> int:
    if value.size_bytes > 0:
        return value.size_bytes

    tensor = ctx.graph.tensors.get(value.name)
    if tensor is None:
        return 0

    return max(tensor.byte_size(), 0)


def _empty_plan() -> MemoryAllocationPlan:
    return MemoryAllocationPlan(
        strategy_name=_STRATEGY_NAME,
        total_fast_memory=0,
        peak_memory=0,
        num_buffers=0,
    )


def _derive_interval(
    value: SramValue,
    scheduled_steps: dict[str, ScheduledStep],
    makespan: int,
) -> _TimedValueInterval | None:
    if not value.name or value.size_bytes <= 0:
        return None

    start_time: int
    if value.producer_step_id is None:
        start_time = 0
    else:
        producer = scheduled_steps.get(value.producer_step_id)
        if producer is None:
            return None
        start_time = producer.end_time

    consumer_end_times: list[int] = []
    for consumer_step_id in value.consumer_step_ids:
        consumer = scheduled_steps.get(consumer_step_id)
        if consumer is None:
            return None
        consumer_end_times.append(consumer.end_time)

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
        return min(already_fits, key=lambda state: (state.buffer.size, state.buffer.id))
    return max(reusable, key=lambda state: (state.buffer.size, -state.buffer.id))


def _assign_offsets(buffer_states: list[_BufferState]) -> None:
    current_offset = 0
    for state in buffer_states:
        state.buffer.offset = current_offset
        current_offset += state.buffer.size


def _build_tensor_mappings(
    intervals: list[_TimedValueInterval],
    buffer_states: list[_BufferState],
    assignments: dict[str, int],
) -> tuple[
    dict[str, TensorAllocation],
    dict[str, int],
    dict[str, LogicalMemoryRegion],
]:
    buffers_by_id = {state.buffer.id: state.buffer for state in buffer_states}
    tensor_allocations: dict[str, TensorAllocation] = {}
    tensor_to_buffer: dict[str, int] = {}
    logical_regions: dict[str, LogicalMemoryRegion] = {}

    for interval in intervals:
        buffer_id = assignments[interval.name]
        buffer = buffers_by_id[buffer_id]
        tensor_allocations[interval.name] = TensorAllocation(
            tensor_name=interval.name,
            buffer_id=buffer_id,
            offset=buffer.offset,
            size=interval.size_bytes,
        )
        tensor_to_buffer[interval.name] = buffer_id
        logical_regions[interval.name] = LogicalMemoryRegion(
            name=interval.name,
            size_bytes=interval.size_bytes,
            offset=buffer.offset,
        )

    return tensor_allocations, tensor_to_buffer, logical_regions


def _calculate_peak_memory(
    intervals: list[_TimedValueInterval],
    buffer_states: list[_BufferState],
    assignments: dict[str, int],
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
            assignments[interval.name]
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


__all__ = ["MemoryPlanningPassV4"]
