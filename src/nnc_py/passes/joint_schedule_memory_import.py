"""Import joint-solver SRAM placement into scheduled-memory compatibility metadata."""

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
    SramAllocationInterval,
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
from nnc_py.passes.scheduled_memory_planning import (
    ScheduledFastAllocation,
    ScheduledMemoryPlan,
    ScheduledSlowAllocation,
    ScheduledTransferPoint,
)


_DEFAULT_ALIGNMENT = 16
_STRATEGY_NAME = "joint_solver_import"


@dataclass(frozen=True)
class _ImportedFastAllocation:
    value_name: str
    offset: int
    size_bytes: int
    start_time: int
    end_time: int
    opened_by_step_id: str | None
    closed_by_step_id: str | None


class JointScheduleMemoryImportPass(PassBase):
    """Build scheduled-memory compatibility metadata from imported joint offsets."""

    @property
    def name(self) -> str:
        return "JointScheduleMemoryImport"

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
        scheduled_steps = {step.step_id: step for step in result.scheduled_steps}
        problem_steps = {step.id: step for step in problem.steps}
        scheduled_values = _resolve_scheduled_values(problem, result)
        windows_by_value_name = _resolve_windows_by_value_name(problem, result)

        fast_allocations = _build_fast_allocations(
            scheduled_values=scheduled_values,
            scheduled_steps=scheduled_steps,
            windows_by_value_name=windows_by_value_name,
            imported_intervals=result.sram_intervals,
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
        total_fast_memory = _calculate_total_fast_memory(result.sram_intervals)
        total_slow_memory = _calculate_total_slow_memory(slow_allocations)
        compat_plan = _build_compat_plan(
            ctx=ctx,
            fast_allocations=fast_allocations,
            total_fast_memory=total_fast_memory,
            total_slow_memory=total_slow_memory,
            transfer_points=transfer_points,
        )
        return (
            ScheduledMemoryPlan(
                total_fast_memory=total_fast_memory,
                total_slow_memory=total_slow_memory,
                fast_allocations=fast_allocations,
                slow_allocations=slow_allocations,
                transfer_points=transfer_points,
            ),
            compat_plan,
        )


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
    by_name = {value.name: value for value in values}
    if len(by_name) != len(values):
        raise RuntimeError(
            "Malformed feasible joint schedule import metadata: duplicate scheduled value names."
        )
    return by_name


def _resolve_windows_by_value_name(
    problem: PipelineScheduleProblem,
    result: PipelineScheduleResult,
) -> dict[str, ResidencyWindow]:
    windows = result.residency_windows or problem.residency_windows
    by_value_name = {window.value_name: window for window in windows}
    if len(by_value_name) != len(windows):
        raise RuntimeError(
            "Malformed feasible joint schedule import metadata: duplicate residency window value names."
        )
    return by_value_name


def _build_fast_allocations(
    *,
    scheduled_values: dict[str, ScheduledValue],
    scheduled_steps: dict[str, ScheduledStep],
    windows_by_value_name: dict[str, ResidencyWindow],
    imported_intervals: tuple[SramAllocationInterval, ...],
) -> dict[str, ScheduledFastAllocation]:
    expected_sram_values = {
        name: value
        for name, value in scheduled_values.items()
        if value.home_tier is ScheduledValueHomeTier.SRAM
    }
    imported_by_value_name: dict[str, list[SramAllocationInterval]] = {}

    for interval in imported_intervals:
        _validate_imported_interval(interval)
        if interval.value_name in expected_sram_values:
            imported_by_value_name.setdefault(interval.value_name, []).append(interval)

    if not expected_sram_values:
        return {}
    if not imported_intervals:
        raise RuntimeError(
            "Malformed feasible joint schedule import metadata: missing imported sram_intervals."
        )

    allocations: dict[str, ScheduledFastAllocation] = {}
    for value_name, value in expected_sram_values.items():
        intervals = imported_by_value_name.get(value_name, [])
        if not intervals:
            raise RuntimeError(
                "Malformed feasible joint schedule import metadata: "
                f"missing imported interval for SRAM value '{value_name}'."
            )
        if len(intervals) != 1:
            raise RuntimeError(
                "Malformed feasible joint schedule import metadata: "
                f"expected exactly one imported interval for SRAM value '{value_name}'."
            )

        imported = intervals[0]
        if imported.offset is None:
            raise RuntimeError(
                "Malformed feasible joint schedule import metadata: "
                f"imported interval for SRAM value '{value_name}' is missing offset."
            )
        if imported.size_bytes != value.size_bytes:
            raise RuntimeError(
                "Malformed feasible joint schedule import metadata: "
                f"imported interval for SRAM value '{value_name}' has size_bytes "
                f"{imported.size_bytes}, expected {value.size_bytes}."
            )
        timing = _resolve_value_timing(
            value=value,
            value_name=value_name,
            scheduled_steps=scheduled_steps,
            windows_by_value_name=windows_by_value_name,
        )
        allocation = _ImportedFastAllocation(
            value_name=value_name,
            offset=imported.offset,
            size_bytes=imported.size_bytes,
            start_time=imported.start_time,
            end_time=imported.end_time,
            opened_by_step_id=timing.opened_by_step_id,
            closed_by_step_id=timing.closed_by_step_id,
        )
        allocations[value_name] = ScheduledFastAllocation(
            residency_id=value_name,
            value_name=value_name,
            buffer_id=0,
            offset=allocation.offset,
            size_bytes=allocation.size_bytes,
            start_time=allocation.start_time,
            end_time=allocation.end_time,
            opened_by_step_id=allocation.opened_by_step_id,
            closed_by_step_id=allocation.closed_by_step_id,
        )

    return allocations


def _validate_imported_interval(interval: SramAllocationInterval) -> None:
    if not interval.value_name:
        raise RuntimeError(
            "Malformed feasible joint schedule import metadata: imported interval is missing value_name."
        )
    if not interval.item_id:
        raise RuntimeError(
            "Malformed feasible joint schedule import metadata: imported interval "
            f"for value '{interval.value_name}' is missing item_id."
        )
    if not interval.item_kind:
        raise RuntimeError(
            "Malformed feasible joint schedule import metadata: imported interval "
            f"for value '{interval.value_name}' is missing item_kind."
        )
    if interval.offset is None:
        raise RuntimeError(
            "Malformed feasible joint schedule import metadata: imported interval "
            f"for value '{interval.value_name}' is missing offset."
        )
    if interval.size_bytes <= 0:
        raise RuntimeError(
            "Malformed feasible joint schedule import metadata: imported interval "
            f"for value '{interval.value_name}' has no positive size."
        )
    if interval.end_time < interval.start_time:
        raise RuntimeError(
            "Malformed feasible joint schedule import metadata: imported interval "
            f"for value '{interval.value_name}' has invalid timing."
        )


@dataclass(frozen=True)
class _ValueTiming:
    opened_by_step_id: str | None
    closed_by_step_id: str | None


def _resolve_value_timing(
    *,
    value: ScheduledValue,
    value_name: str,
    scheduled_steps: dict[str, ScheduledStep],
    windows_by_value_name: dict[str, ResidencyWindow],
) -> _ValueTiming:
    window = windows_by_value_name.get(value_name)
    if window is not None:
        _require_known_step(
            scheduled_steps,
            window.opened_by_step_id,
            value_name=value_name,
            label="open",
        )
        _require_known_step(
            scheduled_steps,
            window.closed_by_step_id,
            value_name=value_name,
            label="close",
        )
        return _ValueTiming(
            opened_by_step_id=window.opened_by_step_id,
            closed_by_step_id=window.closed_by_step_id,
        )

    _require_known_step(
        scheduled_steps,
        value.producer_step_id,
        value_name=value_name,
        label="producer",
    )
    consumer_steps = []
    for consumer_step_id in value.consumer_step_ids:
        consumer = scheduled_steps.get(consumer_step_id)
        if consumer is None:
            raise RuntimeError(
                "Malformed feasible joint schedule import metadata: "
                f"value '{value_name}' references unknown consumer step '{consumer_step_id}'."
            )
        consumer_steps.append(consumer)

    if not consumer_steps:
        return _ValueTiming(
            opened_by_step_id=value.producer_step_id,
            closed_by_step_id=None,
        )

    latest_end = max(step.end_time for step in consumer_steps)
    closed_by_step_id = min(
        step.step_id
        for step in consumer_steps
        if step.end_time == latest_end
    )
    return _ValueTiming(
        opened_by_step_id=value.producer_step_id,
        closed_by_step_id=closed_by_step_id,
    )


def _require_known_step(
    scheduled_steps: dict[str, ScheduledStep],
    step_id: str | None,
    *,
    value_name: str,
    label: str,
) -> None:
    if step_id is None:
        return
    if step_id not in scheduled_steps:
        raise RuntimeError(
            "Malformed feasible joint schedule import metadata: "
            f"value '{value_name}' references unknown {label} step '{step_id}'."
        )


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
        if value is None:
            raise RuntimeError(
                "Malformed feasible joint schedule import metadata: transfer step "
                f"'{step.id}' references unknown value '{step.moved_value_name}'."
            )

        size_bytes = max(step.bytes, value.size_bytes)
        previous = slow_sizes.get(step.moved_value_name, 0)
        slow_sizes[step.moved_value_name] = max(previous, size_bytes)

    allocations: dict[str, ScheduledSlowAllocation] = {}
    current_offset = 0
    for value_name in sorted(slow_sizes):
        allocations[value_name] = ScheduledSlowAllocation(
            value_name=value_name,
            offset=current_offset,
            size_bytes=slow_sizes[value_name],
        )
        current_offset += _align(slow_sizes[value_name], _DEFAULT_ALIGNMENT)
    return allocations


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
            raise RuntimeError(
                "Malformed feasible joint schedule import metadata: "
                f"missing scheduled placement for transfer step '{step_id}'."
            )

        slow_allocation = slow_allocations.get(problem_step.moved_value_name)
        if slow_allocation is None:
            raise RuntimeError(
                "Malformed feasible joint schedule import metadata: transfer step "
                f"'{step_id}' has no slow allocation for value '{problem_step.moved_value_name}'."
            )

        resident_value_name = None
        after_node_name = None
        before_node_name = None

        if problem_step.transfer_kind is TransferStepKind.SPILL_DMA:
            fast_allocation = _lookup_fast_allocation(
                fast_allocations=fast_allocations,
                value_name=problem_step.moved_value_name,
                step_id=step_id,
                by_open_step=False,
            )
            moved_value = scheduled_values.get(problem_step.moved_value_name)
            if moved_value is None or moved_value.producer_step_id is None:
                raise RuntimeError(
                    "Malformed feasible joint schedule import metadata: spill step "
                    f"'{step_id}' cannot resolve producer for value '{problem_step.moved_value_name}'."
                )
            producer_step = problem_steps.get(moved_value.producer_step_id)
            after_node_name = getattr(producer_step, "node_name", None)
            if after_node_name is None:
                raise RuntimeError(
                    "Malformed feasible joint schedule import metadata: "
                    f"spill step '{step_id}' cannot resolve producer node."
                )
        else:
            resident_value_name = next(iter(problem_step.sram_output_names), None)
            if resident_value_name is None:
                raise RuntimeError(
                    "Malformed feasible joint schedule import metadata: reload step "
                    f"'{step_id}' has no SRAM output binding."
                )
            fast_allocation = _lookup_fast_allocation(
                fast_allocations=fast_allocations,
                value_name=resident_value_name,
                step_id=step_id,
                by_open_step=True,
            )
            resident_value = scheduled_values.get(resident_value_name)
            if resident_value is not None and resident_value.consumer_step_ids:
                consumer_step = problem_steps.get(resident_value.consumer_step_ids[0])
                before_node_name = getattr(consumer_step, "node_name", None)
            if before_node_name is None and fast_allocation.closed_by_step_id is not None:
                consumer_step = problem_steps.get(fast_allocation.closed_by_step_id)
                before_node_name = getattr(consumer_step, "node_name", None)
            if before_node_name is None:
                raise RuntimeError(
                    "Malformed feasible joint schedule import metadata: "
                    f"reload step '{step_id}' cannot resolve consumer node."
                )

        moved_value = scheduled_values.get(problem_step.moved_value_name)
        if moved_value is None:
            raise RuntimeError(
                "Malformed feasible joint schedule import metadata: transfer step "
                f"'{step_id}' references unknown value '{problem_step.moved_value_name}'."
            )
        size_bytes = max(problem_step.bytes, moved_value.size_bytes)

        transfer_points.append(
            ScheduledTransferPoint(
                step_id=step_id,
                transfer_kind=problem_step.transfer_kind,
                value_name=problem_step.moved_value_name,
                size_bytes=size_bytes,
                start_time=scheduled_step.start_time,
                end_time=scheduled_step.end_time,
                fast_offset=fast_allocation.offset,
                slow_offset=slow_allocation.offset,
                resident_value_name=resident_value_name,
                after_node_name=after_node_name,
                before_node_name=before_node_name,
            )
        )

    transfer_points.sort(key=lambda point: (point.start_time, point.end_time, point.step_id))
    return tuple(transfer_points)


def _lookup_fast_allocation(
    *,
    fast_allocations: dict[str, ScheduledFastAllocation],
    value_name: str,
    step_id: str,
    by_open_step: bool,
) -> ScheduledFastAllocation:
    for allocation in fast_allocations.values():
        if allocation.value_name != value_name:
            continue
        if by_open_step and allocation.opened_by_step_id == step_id:
            return allocation
        if not by_open_step and allocation.closed_by_step_id == step_id:
            return allocation
    allocation = fast_allocations.get(value_name)
    if allocation is not None:
        return allocation
    raise RuntimeError(
        "Malformed feasible joint schedule import metadata: transfer step "
        f"'{step_id}' cannot resolve imported fast allocation for value '{value_name}'."
    )


def _calculate_total_fast_memory(
    intervals: tuple[SramAllocationInterval, ...],
) -> int:
    if not intervals:
        return 0
    max_extent = 0
    for interval in intervals:
        if interval.offset is None:
            raise RuntimeError(
                "Malformed feasible joint schedule import metadata: imported interval "
                f"for value '{interval.value_name}' is missing offset."
            )
        max_extent = max(
            max_extent,
            interval.offset + _align(interval.size_bytes, _DEFAULT_ALIGNMENT),
        )
    return max_extent


def _calculate_total_slow_memory(
    slow_allocations: dict[str, ScheduledSlowAllocation],
) -> int:
    if not slow_allocations:
        return 0
    return max(
        allocation.offset + _align(allocation.size_bytes, _DEFAULT_ALIGNMENT)
        for allocation in slow_allocations.values()
    )


def _build_compat_plan(
    *,
    ctx: CompileContext,
    fast_allocations: dict[str, ScheduledFastAllocation],
    total_fast_memory: int,
    total_slow_memory: int,
    transfer_points: tuple[ScheduledTransferPoint, ...],
) -> MemoryAllocationPlan:
    buffers = []
    if total_fast_memory > 0:
        buffers.append(
            MemoryBuffer(
                id=0,
                offset=0,
                size=total_fast_memory,
                alignment=_DEFAULT_ALIGNMENT,
            )
        )

    tensor_allocations: dict[str, TensorAllocation] = {}
    tensor_to_buffer: dict[str, int] = {}
    logical_regions: dict[str, LogicalMemoryRegion] = {}
    for allocation in fast_allocations.values():
        tensor_allocations[allocation.value_name] = TensorAllocation(
            tensor_name=allocation.value_name,
            buffer_id=0,
            offset=allocation.offset,
            size=allocation.size_bytes,
        )
        tensor_to_buffer[allocation.value_name] = 0
        logical_regions[allocation.value_name] = LogicalMemoryRegion(
            name=allocation.value_name,
            size_bytes=allocation.size_bytes,
            offset=allocation.offset,
        )

    spill_points, reload_points = _build_compat_transfer_points(
        ctx=ctx,
        fast_allocations=fast_allocations,
        transfer_points=transfer_points,
    )
    spill_bytes = sum(point.size for point in spill_points)
    reload_bytes = sum(point.size for point in reload_points)

    return MemoryAllocationPlan(
        strategy_name=_STRATEGY_NAME,
        total_fast_memory=total_fast_memory,
        total_slow_memory=total_slow_memory,
        peak_memory=total_fast_memory,
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
    fast_allocations: dict[str, ScheduledFastAllocation],
    transfer_points: tuple[ScheduledTransferPoint, ...],
) -> tuple[list[SpillPoint], list[ReloadPoint]]:
    node_index_by_name = {
        node.name: index
        for index, node in enumerate(ctx.graph.topological_sort())
    }
    spill_points: list[SpillPoint] = []
    reload_points: list[ReloadPoint] = []

    for transfer_point in transfer_points:
        if transfer_point.transfer_kind is TransferStepKind.SPILL_DMA:
            fast_allocation = _lookup_fast_allocation(
                fast_allocations=fast_allocations,
                value_name=transfer_point.value_name,
                step_id=transfer_point.step_id,
                by_open_step=False,
            )
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

        resident_value_name = transfer_point.resident_value_name
        if resident_value_name is None:
            continue
        resident_allocation = _lookup_fast_allocation(
            fast_allocations=fast_allocations,
            value_name=resident_value_name,
            step_id=transfer_point.step_id,
            by_open_step=True,
        )
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


def _empty_scheduled_plan() -> ScheduledMemoryPlan:
    return ScheduledMemoryPlan(
        total_fast_memory=0,
        total_slow_memory=0,
        fast_allocations={},
        slow_allocations={},
        transfer_points=(),
    )


def _empty_compat_plan() -> MemoryAllocationPlan:
    return MemoryAllocationPlan(
        strategy_name=_STRATEGY_NAME,
        total_fast_memory=0,
        peak_memory=0,
        num_buffers=0,
    )


def _align(value: int, alignment: int) -> int:
    if value <= 0:
        return 0
    return ((value + alignment - 1) // alignment) * alignment


__all__ = ["JointScheduleMemoryImportPass"]
