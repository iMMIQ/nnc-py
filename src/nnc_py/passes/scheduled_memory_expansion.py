"""Expand scheduled problems with explicit DMA spill and reload steps."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from nnc_py.ir.context import CompileContext
from nnc_py.ir.pipeline_schedule import (
    PipelineScheduleProblem,
    ResidencyWindow,
    ScheduleDependencyKind,
    ScheduleEdge,
    ScheduleStep,
    ScheduledValue,
    ScheduledValueHomeTier,
    TransferStep,
    TransferStepKind,
    set_pipeline_schedule_problem,
)
from nnc_py.passes.base import PassBase


@dataclass(frozen=True)
class _ValueLifetime:
    value: ScheduledValue
    producer_index: int
    consumer_indices: tuple[int, ...]

    def is_live_after(self, step_index: int) -> bool:
        return self.producer_index <= step_index and any(
            consumer_index > step_index for consumer_index in self.consumer_indices
        )

    def next_consumer_after(self, step_index: int) -> int:
        return min(
            consumer_index
            for consumer_index in self.consumer_indices
            if consumer_index > step_index
        )


class ScheduledMemoryExpansionPass(PassBase):
    """Insert conservative spill and reload DMA steps before scheduling."""

    @property
    def name(self) -> str:
        return "ScheduledMemoryExpansion"

    def _execute(self, ctx: CompileContext) -> None:
        problem = ctx.pipeline_schedule_problem
        budget = _resolve_max_memory_budget(ctx)
        if problem is None or budget is None:
            return

        expanded_problem = expand_schedule_problem(problem, budget)
        set_pipeline_schedule_problem(ctx, expanded_problem)


def expand_schedule_problem(
    problem: PipelineScheduleProblem,
    budget: int,
) -> PipelineScheduleProblem:
    _validate_individual_value_fit(problem.scheduled_values, budget)
    spill_names = _select_spill_candidates(problem, budget)

    steps = list(problem.steps)
    edges = list(problem.edges)
    edge_keys = {
        (edge.src_step_id, edge.dst_step_id, edge.kind)
        for edge in edges
    }
    scheduled_values = list(problem.scheduled_values)
    value_index_by_name = {
        value.name: index for index, value in enumerate(scheduled_values)
    }
    residency_windows = list(problem.residency_windows)

    for spill_name in spill_names:
        value_index = value_index_by_name[spill_name]
        value = scheduled_values[value_index]
        if value.producer_step_id is None or not value.consumer_step_ids:
            continue

        spill_step_id = f"{value.name}.spill0"
        spill_duration = _transfer_duration(value.size_bytes)
        spill_step = TransferStep(
            id=spill_step_id,
            node_name=f"spill:{value.name}",
            transfer_kind=TransferStepKind.SPILL_DMA,
            moved_value_name=value.name,
            src_tier=ScheduledValueHomeTier.SRAM,
            dst_tier=ScheduledValueHomeTier.SLOW,
            bytes=value.size_bytes,
            duration=spill_duration,
            sram_input_names=(value.name,),
        )
        _insert_step_after(steps, anchor_step_id=value.producer_step_id, step=spill_step)
        _append_edge(
            edges,
            edge_keys,
            src_step_id=value.producer_step_id,
            dst_step_id=spill_step_id,
            kind=ScheduleDependencyKind.DATA,
        )

        scheduled_values[value_index] = replace(
            value,
            consumer_step_ids=(spill_step_id,),
        )
        residency_windows.append(
            ResidencyWindow(
                value_name=value.name,
                residency_id=f"{value.name}@spill0",
                opened_by_step_id=value.producer_step_id,
                closed_by_step_id=spill_step_id,
            )
        )

        for consumer_index, consumer_step_id in enumerate(
            _unique_consumer_step_ids(value.consumer_step_ids)
        ):
            reload_step_id = f"{value.name}.reload{consumer_index}"
            resident_value_name = f"{reload_step_id}.resident"
            reload_step = TransferStep(
                id=reload_step_id,
                node_name=f"reload:{value.name}",
                transfer_kind=TransferStepKind.RELOAD_DMA,
                moved_value_name=value.name,
                src_tier=ScheduledValueHomeTier.SLOW,
                dst_tier=ScheduledValueHomeTier.SRAM,
                bytes=value.size_bytes,
                duration=spill_duration,
                sram_output_names=(resident_value_name,),
            )

            predecessor_ids = [
                edge.src_step_id for edge in edges if edge.dst_step_id == consumer_step_id
            ]
            _insert_step_before(steps, anchor_step_id=consumer_step_id, step=reload_step)
            for predecessor_id in predecessor_ids:
                _append_edge(
                    edges,
                    edge_keys,
                    src_step_id=predecessor_id,
                    dst_step_id=reload_step_id,
                    kind=ScheduleDependencyKind.ORDER,
                )
            _append_edge(
                edges,
                edge_keys,
                src_step_id=spill_step_id,
                dst_step_id=reload_step_id,
                kind=ScheduleDependencyKind.ORDER,
            )
            _append_edge(
                edges,
                edge_keys,
                src_step_id=reload_step_id,
                dst_step_id=consumer_step_id,
                kind=ScheduleDependencyKind.DATA,
            )

            _replace_step_inputs(
                steps,
                step_id=consumer_step_id,
                old_value_name=value.name,
                new_value_name=resident_value_name,
            )
            scheduled_values.append(
                ScheduledValue(
                    name=resident_value_name,
                    graph_tensor_name=value.graph_tensor_name,
                    size_bytes=value.size_bytes,
                    producer_step_id=reload_step_id,
                    consumer_step_ids=(consumer_step_id,),
                    can_alias=value.can_alias,
                    home_tier=ScheduledValueHomeTier.SRAM,
                )
            )
            residency_windows.append(
                ResidencyWindow(
                    value_name=resident_value_name,
                    residency_id=f"{resident_value_name}@0",
                    opened_by_step_id=reload_step_id,
                    closed_by_step_id=consumer_step_id,
                )
            )

    metadata = dict(problem.metadata)
    metadata["scheduled_memory_expansion"] = {
        "max_memory": budget,
        "spilled_values": tuple(spill_names),
    }
    return PipelineScheduleProblem(
        steps=tuple(steps),
        edges=tuple(edges),
        scheduled_values=tuple(scheduled_values),
        residency_windows=tuple(residency_windows),
        resources=problem.resources,
        sram_capacity_bytes=budget,
        objective=problem.objective,
        metadata=metadata,
    )


def _resolve_max_memory_budget(ctx: CompileContext) -> int | None:
    max_memory = ctx.metadata.get("max_memory")
    if max_memory is None or max_memory == float("inf"):
        return None
    return int(max_memory)


def _validate_individual_value_fit(
    scheduled_values: tuple[ScheduledValue, ...],
    budget: int,
) -> None:
    for value in scheduled_values:
        if value.home_tier is not ScheduledValueHomeTier.SRAM:
            continue
        if value.size_bytes <= budget:
            continue
        if value.must_reside_in_sram:
            raise RuntimeError(
                f"Value '{value.name}' must reside in SRAM but needs "
                f"{value.size_bytes} bytes with budget {budget}."
            )
        raise RuntimeError(
            f"Value '{value.name}' cannot fit in SRAM under scheduled budget {budget}."
        )


def _select_spill_candidates(
    problem: PipelineScheduleProblem,
    budget: int,
) -> tuple[str, ...]:
    step_index_by_id = {
        step.id: index for index, step in enumerate(problem.steps)
    }
    lifetimes: list[_ValueLifetime] = []
    for value in problem.scheduled_values:
        if value.home_tier is not ScheduledValueHomeTier.SRAM:
            continue
        if value.producer_step_id is None or not value.consumer_step_ids:
            continue
        producer_index = step_index_by_id.get(value.producer_step_id)
        consumer_indices = tuple(
            sorted(
                step_index_by_id[consumer_step_id]
                for consumer_step_id in value.consumer_step_ids
                if consumer_step_id in step_index_by_id
            )
        )
        if producer_index is None or not consumer_indices:
            continue
        lifetimes.append(
            _ValueLifetime(
                value=value,
                producer_index=producer_index,
                consumer_indices=consumer_indices,
            )
        )

    spill_names: list[str] = []
    spilled_name_set: set[str] = set()
    for boundary_index in range(len(problem.steps) - 1):
        live_values = [
            lifetime for lifetime in lifetimes
            if lifetime.value.name not in spilled_name_set
            and lifetime.is_live_after(boundary_index)
        ]
        live_bytes = sum(lifetime.value.size_bytes for lifetime in live_values)
        while live_bytes > budget:
            candidate = _pick_spill_candidate(live_values, boundary_index)
            if candidate is None:
                raise RuntimeError(
                    "Scheduled value must reside in SRAM but cannot fit under "
                    f"budget {budget}."
                )
            spill_names.append(candidate.value.name)
            spilled_name_set.add(candidate.value.name)
            live_values = [
                lifetime for lifetime in live_values
                if lifetime.value.name != candidate.value.name
            ]
            live_bytes -= candidate.value.size_bytes
    return tuple(spill_names)


def _pick_spill_candidate(
    live_values: list[_ValueLifetime],
    boundary_index: int,
) -> _ValueLifetime | None:
    spillable_values = [
        lifetime for lifetime in live_values if not lifetime.value.must_reside_in_sram
    ]
    if not spillable_values:
        return None
    return max(
        spillable_values,
        key=lambda lifetime: (
            lifetime.next_consumer_after(boundary_index),
            lifetime.value.size_bytes,
            -lifetime.producer_index,
        ),
    )


def _unique_consumer_step_ids(
    consumer_step_ids: tuple[str, ...],
) -> tuple[str, ...]:
    unique_consumer_step_ids: list[str] = []
    seen_step_ids: set[str] = set()
    for consumer_step_id in consumer_step_ids:
        if consumer_step_id in seen_step_ids:
            continue
        unique_consumer_step_ids.append(consumer_step_id)
        seen_step_ids.add(consumer_step_id)
    return tuple(unique_consumer_step_ids)


def _transfer_duration(size_bytes: int) -> int:
    return max(1, math.ceil(size_bytes / 64))


def _insert_step_after(
    steps: list[ScheduleStep],
    *,
    anchor_step_id: str,
    step: ScheduleStep,
) -> None:
    anchor_index = _step_index(steps, anchor_step_id)
    steps.insert(anchor_index + 1, step)


def _insert_step_before(
    steps: list[ScheduleStep],
    *,
    anchor_step_id: str,
    step: ScheduleStep,
) -> None:
    anchor_index = _step_index(steps, anchor_step_id)
    steps.insert(anchor_index, step)


def _replace_step_inputs(
    steps: list[ScheduleStep],
    *,
    step_id: str,
    old_value_name: str,
    new_value_name: str,
) -> None:
    index = _step_index(steps, step_id)
    step = steps[index]
    steps[index] = replace(
        step,
        sram_input_names=tuple(
            new_value_name if value_name == old_value_name else value_name
            for value_name in step.sram_input_names
        ),
    )


def _step_index(steps: list[ScheduleStep], step_id: str) -> int:
    for index, step in enumerate(steps):
        if step.id == step_id:
            return index
    raise KeyError(f"unknown step id: {step_id}")


def _append_edge(
    edges: list[ScheduleEdge],
    edge_keys: set[tuple[str, str, ScheduleDependencyKind]],
    *,
    src_step_id: str,
    dst_step_id: str,
    kind: ScheduleDependencyKind,
) -> None:
    edge_key = (src_step_id, dst_step_id, kind)
    if edge_key in edge_keys:
        return
    edges.append(
        ScheduleEdge(
            src_step_id=src_step_id,
            dst_step_id=dst_step_id,
            kind=kind,
        )
    )
    edge_keys.add(edge_key)


__all__ = ["ScheduledMemoryExpansionPass", "expand_schedule_problem"]
