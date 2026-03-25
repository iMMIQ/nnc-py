"""Pipeline scheduling IR primitives and compile-context metadata helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from nnc_py.ir.context import CompileContext


JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]
EnumT = TypeVar("EnumT", bound=Enum)


PIPELINE_SCHEDULE_PROBLEM_METADATA_KEY = "pipeline_schedule_problem"
PIPELINE_SCHEDULE_RESULT_METADATA_KEY = "pipeline_schedule_result"


def _freeze_json_mapping(
    value: object, *, field_name: str
) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    frozen_items: dict[str, JsonValue] = {}
    keys = tuple(value.keys())
    for key in keys:
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings, got {type(key).__name__}")
    for key in sorted(keys):
        frozen_items[key] = _freeze_json_value(
            value[key], path=f"{field_name}.{key}"
        )
    return MappingProxyType(frozen_items)


def _freeze_json_value(value: object, *, path: str) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError(f"{path} must contain only finite float values")
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, field_name=path)
    if isinstance(value, list | tuple):
        return tuple(
            _freeze_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain only JSON-compatible values, got {type(value).__name__}"
    )


def _to_json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    raise TypeError(f"Unsupported JSON conversion for {type(value).__name__}")


def _coerce_enum(value: object, enum_type: type[EnumT], *, field_name: str) -> EnumT:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be one of {tuple(member.value for member in enum_type)}") from exc
    raise TypeError(f"{field_name} must be a {enum_type.__name__} or str")


def _coerce_str_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a sequence of strings")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of strings") from exc
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise TypeError(f"{field_name}[{index}] must be str")
    return items


def _coerce_tuple_of_type(
    value: object, expected_type: type, *, field_name: str
) -> tuple[object, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a sequence of {expected_type.__name__}")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be a sequence of {expected_type.__name__}"
        ) from exc
    for index, item in enumerate(items):
        if not isinstance(item, expected_type):
            raise TypeError(f"{field_name}[{index}] must be {expected_type.__name__}")
    return items


def _coerce_enum_tuple(
    value: object, enum_type: type[EnumT], *, field_name: str
) -> tuple[EnumT, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a sequence of {enum_type.__name__}")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be a sequence of {enum_type.__name__}"
        ) from exc
    return tuple(
        _coerce_enum(item, enum_type, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(items)
    )


def _get_typed_metadata(
    ctx: CompileContext, key: str, expected_type: type[object]
) -> object | None:
    value = ctx.metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, expected_type):
        raise TypeError(
            f"metadata[{key!r}] must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


class PipelineResourceKind(str, Enum):
    """Abstract execution resources the scheduler can assign work to."""

    MATMUL = "matmul"
    SHAPE = "shape"
    DMA = "dma"
    OTHER = "other"


class ScheduleStepKind(str, Enum):
    """Semantic categories for schedule steps."""

    DMA_IN = "dma_in"
    SHAPE_PREP = "shape_prep"
    COMPUTE = "compute"
    SPILL_DMA = "spill_dma"
    RELOAD_DMA = "reload_dma"
    DMA_OUT = "dma_out"


class ScheduleDependencyKind(str, Enum):
    """Supported dependency edge kinds between schedule steps."""

    DATA = "data"
    ORDER = "order"
    SAME_NODE_SEQUENCE = "same_node_sequence"


class TransferStepKind(str, Enum):
    """Semantic categories for transfer-oriented schedule steps."""

    DMA_IN = "dma_in"
    SPILL_DMA = "spill_dma"
    RELOAD_DMA = "reload_dma"
    DMA_OUT = "dma_out"


class ScheduledValueHomeTier(str, Enum):
    """Where a scheduled value naturally resides when not in active SRAM use."""

    INPUT = "input"
    CONST = "const"
    SLOW = "slow"
    SRAM = "sram"


def _transfer_kind_to_step_kind(transfer_kind: TransferStepKind) -> ScheduleStepKind:
    return ScheduleStepKind(transfer_kind.value)


@dataclass(frozen=True)
class ScheduleStep:
    """Smallest non-preemptive unit seen by the scheduler."""

    id: str
    node_name: str
    tile_id: str | None = None
    step_kind: ScheduleStepKind = ScheduleStepKind.COMPUTE
    resource_kind: PipelineResourceKind = PipelineResourceKind.OTHER
    duration: int = 0
    launch_overhead: int = 0
    sram_input_names: tuple[str, ...] = ()
    sram_output_names: tuple[str, ...] = ()
    sram_temp_bytes: int = 0
    attrs: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step_kind",
            _coerce_enum(self.step_kind, ScheduleStepKind, field_name="ScheduleStep.step_kind"),
        )
        object.__setattr__(
            self,
            "resource_kind",
            _coerce_enum(
                self.resource_kind,
                PipelineResourceKind,
                field_name="ScheduleStep.resource_kind",
            ),
        )
        object.__setattr__(
            self,
            "sram_input_names",
            _coerce_str_tuple(
                self.sram_input_names,
                field_name="ScheduleStep.sram_input_names",
            ),
        )
        object.__setattr__(
            self,
            "sram_output_names",
            _coerce_str_tuple(
                self.sram_output_names,
                field_name="ScheduleStep.sram_output_names",
            ),
        )
        object.__setattr__(
            self,
            "attrs",
            _freeze_json_mapping(self.attrs, field_name="ScheduleStep.attrs"),
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready representation of the schedule step."""

        return {
            "id": self.id,
            "node_name": self.node_name,
            "tile_id": self.tile_id,
            "step_kind": self.step_kind.value,
            "resource_kind": self.resource_kind.value,
            "duration": self.duration,
            "launch_overhead": self.launch_overhead,
            "sram_input_names": list(self.sram_input_names),
            "sram_output_names": list(self.sram_output_names),
            "sram_temp_bytes": self.sram_temp_bytes,
            "attrs": _to_json_value(self.attrs),
        }


@dataclass(frozen=True)
class TransferStep(ScheduleStep):
    """Schedule step that moves one value between storage tiers via DMA."""

    transfer_kind: TransferStepKind = TransferStepKind.DMA_IN
    moved_value_name: str = ""
    src_tier: ScheduledValueHomeTier = ScheduledValueHomeTier.SLOW
    dst_tier: ScheduledValueHomeTier = ScheduledValueHomeTier.SRAM
    bytes: int = 0

    def __post_init__(self) -> None:
        transfer_kind = _coerce_enum(
            self.transfer_kind,
            TransferStepKind,
            field_name="TransferStep.transfer_kind",
        )
        object.__setattr__(self, "transfer_kind", transfer_kind)
        object.__setattr__(
            self,
            "src_tier",
            _coerce_enum(
                self.src_tier,
                ScheduledValueHomeTier,
                field_name="TransferStep.src_tier",
            ),
        )
        object.__setattr__(
            self,
            "dst_tier",
            _coerce_enum(
                self.dst_tier,
                ScheduledValueHomeTier,
                field_name="TransferStep.dst_tier",
            ),
        )
        object.__setattr__(
            self,
            "step_kind",
            _transfer_kind_to_step_kind(transfer_kind),
        )
        object.__setattr__(self, "resource_kind", PipelineResourceKind.DMA)
        super().__post_init__()

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready representation of the transfer step."""

        payload = super().to_json()
        payload.update(
            {
                "transfer_kind": self.transfer_kind.value,
                "moved_value_name": self.moved_value_name,
                "src_tier": self.src_tier.value,
                "dst_tier": self.dst_tier.value,
                "bytes": self.bytes,
            }
        )
        return payload


@dataclass(frozen=True)
class SramValue:
    """Value that may remain live in SRAM across schedule-step boundaries."""

    name: str
    size_bytes: int
    producer_step_id: str | None = None
    consumer_step_ids: tuple[str, ...] = ()
    must_reside_in_sram: bool = False
    can_alias: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "consumer_step_ids",
            _coerce_str_tuple(
                self.consumer_step_ids,
                field_name="SramValue.consumer_step_ids",
            ),
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready representation of the SRAM value."""

        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "producer_step_id": self.producer_step_id,
            "consumer_step_ids": list(self.consumer_step_ids),
            "must_reside_in_sram": self.must_reside_in_sram,
            "can_alias": self.can_alias,
        }


@dataclass(frozen=True)
class ScheduledValue:
    """Value tracked by the scheduler across SRAM and slow-tier residency."""

    name: str
    graph_tensor_name: str | None
    size_bytes: int
    producer_step_id: str | None = None
    consumer_step_ids: tuple[str, ...] = ()
    must_reside_in_sram: bool = False
    can_alias: bool = False
    home_tier: ScheduledValueHomeTier = ScheduledValueHomeTier.SRAM

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "consumer_step_ids",
            _coerce_str_tuple(
                self.consumer_step_ids,
                field_name="ScheduledValue.consumer_step_ids",
            ),
        )
        object.__setattr__(
            self,
            "home_tier",
            _coerce_enum(
                self.home_tier,
                ScheduledValueHomeTier,
                field_name="ScheduledValue.home_tier",
            ),
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready representation of the scheduled value."""

        return {
            "name": self.name,
            "graph_tensor_name": self.graph_tensor_name,
            "size_bytes": self.size_bytes,
            "producer_step_id": self.producer_step_id,
            "consumer_step_ids": list(self.consumer_step_ids),
            "must_reside_in_sram": self.must_reside_in_sram,
            "can_alias": self.can_alias,
            "home_tier": self.home_tier.value,
        }


@dataclass(frozen=True)
class ScheduleEdge:
    """Dependency edge between two schedule steps."""

    src_step_id: str
    dst_step_id: str
    kind: ScheduleDependencyKind = ScheduleDependencyKind.DATA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _coerce_enum(
                self.kind,
                ScheduleDependencyKind,
                field_name="ScheduleEdge.kind",
            ),
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready representation of the schedule edge."""

        return {
            "src_step_id": self.src_step_id,
            "dst_step_id": self.dst_step_id,
            "kind": self.kind.value,
        }


@dataclass(frozen=True)
class ResidencyWindow:
    """Logical lifetime window for one scheduled value residency."""

    value_name: str
    residency_id: str
    opened_by_step_id: str
    closed_by_step_id: str | None = None

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready representation of the residency window."""

        return {
            "value_name": self.value_name,
            "residency_id": self.residency_id,
            "opened_by_step_id": self.opened_by_step_id,
            "closed_by_step_id": self.closed_by_step_id,
        }


def _scheduled_value_from_sram_value(value: SramValue) -> ScheduledValue:
    return ScheduledValue(
        name=value.name,
        graph_tensor_name=None,
        size_bytes=value.size_bytes,
        producer_step_id=value.producer_step_id,
        consumer_step_ids=value.consumer_step_ids,
        must_reside_in_sram=value.must_reside_in_sram,
        can_alias=value.can_alias,
    )


def _sram_value_from_scheduled_value(value: ScheduledValue) -> SramValue:
    return SramValue(
        name=value.name,
        size_bytes=(
            value.size_bytes
            if value.home_tier is ScheduledValueHomeTier.SRAM
            else 0
        ),
        producer_step_id=value.producer_step_id,
        consumer_step_ids=value.consumer_step_ids,
        must_reside_in_sram=value.must_reside_in_sram,
        can_alias=value.can_alias,
    )


def _validate_value_shim_consistency(
    sram_values: tuple[SramValue, ...],
    scheduled_values: tuple[ScheduledValue, ...],
) -> None:
    projected_sram_values = tuple(
        _sram_value_from_scheduled_value(value) for value in scheduled_values
    )
    if sram_values != projected_sram_values:
        raise ValueError("scheduled_values must match sram_values")


@dataclass(frozen=True)
class ScheduledStep:
    """Scheduled placement for one step on a resource slot."""

    step_id: str
    resource_kind: PipelineResourceKind
    resource_slot: int = 0
    start_time: int = 0
    end_time: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_kind",
            _coerce_enum(
                self.resource_kind,
                PipelineResourceKind,
                field_name="ScheduledStep.resource_kind",
            ),
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready representation of the scheduled step."""

        return {
            "step_id": self.step_id,
            "resource_kind": self.resource_kind.value,
            "resource_slot": self.resource_slot,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass(frozen=True)
class SramAllocationInterval:
    """Lifetime of one logical SRAM value within an assigned buffer."""

    value_name: str
    buffer_id: str
    start_time: int
    end_time: int
    size_bytes: int

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready representation of the SRAM interval."""

        return {
            "value_name": self.value_name,
            "buffer_id": self.buffer_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class PipelineScheduleProblem:
    """Serializable solver input for pipeline scheduling."""

    steps: tuple[ScheduleStep, ...] = ()
    edges: tuple[ScheduleEdge, ...] = ()
    sram_values: tuple[SramValue, ...] = ()
    scheduled_values: tuple[ScheduledValue, ...] = ()
    residency_windows: tuple[ResidencyWindow, ...] = ()
    resources: tuple[PipelineResourceKind, ...] = ()
    sram_capacity_bytes: int = 0
    objective: str = "min_makespan"
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sram_values = _coerce_tuple_of_type(
            self.sram_values,
            SramValue,
            field_name="PipelineScheduleProblem.sram_values",
        )
        scheduled_values = _coerce_tuple_of_type(
            self.scheduled_values,
            ScheduledValue,
            field_name="PipelineScheduleProblem.scheduled_values",
        )
        if not scheduled_values and sram_values:
            scheduled_values = tuple(
                _scheduled_value_from_sram_value(value) for value in sram_values
            )
        elif not sram_values and scheduled_values:
            sram_values = tuple(
                _sram_value_from_scheduled_value(value) for value in scheduled_values
            )
        elif sram_values and scheduled_values:
            _validate_value_shim_consistency(sram_values, scheduled_values)
        object.__setattr__(
            self,
            "steps",
            _coerce_tuple_of_type(
                self.steps, ScheduleStep, field_name="PipelineScheduleProblem.steps"
            ),
        )
        object.__setattr__(
            self,
            "edges",
            _coerce_tuple_of_type(
                self.edges, ScheduleEdge, field_name="PipelineScheduleProblem.edges"
            ),
        )
        object.__setattr__(
            self,
            "sram_values",
            sram_values,
        )
        object.__setattr__(
            self,
            "scheduled_values",
            scheduled_values,
        )
        object.__setattr__(
            self,
            "residency_windows",
            _coerce_tuple_of_type(
                self.residency_windows,
                ResidencyWindow,
                field_name="PipelineScheduleProblem.residency_windows",
            ),
        )
        object.__setattr__(
            self,
            "resources",
            _coerce_enum_tuple(
                self.resources,
                PipelineResourceKind,
                field_name="PipelineScheduleProblem.resources",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_mapping(
                self.metadata, field_name="PipelineScheduleProblem.metadata"
            ),
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready representation of the schedule problem."""

        return {
            "steps": [step.to_json() for step in self.steps],
            "edges": [edge.to_json() for edge in self.edges],
            "sram_values": [value.to_json() for value in self.sram_values],
            "scheduled_values": [
                value.to_json() for value in self.scheduled_values
            ],
            "residency_windows": [
                window.to_json() for window in self.residency_windows
            ],
            "resources": [resource.value for resource in self.resources],
            "sram_capacity_bytes": self.sram_capacity_bytes,
            "objective": self.objective,
            "metadata": _to_json_value(self.metadata),
        }


@dataclass(frozen=True)
class PipelineScheduleResult:
    """Serializable solver output for pipeline scheduling."""

    scheduled_steps: tuple[ScheduledStep, ...] = ()
    sram_intervals: tuple[SramAllocationInterval, ...] = ()
    scheduled_values: tuple[ScheduledValue, ...] = ()
    residency_windows: tuple[ResidencyWindow, ...] = ()
    makespan: int = 0
    feasible: bool = False
    solver_name: str = ""
    diagnostics: Mapping[str, JsonValue] = field(default_factory=dict)
    transfer_diagnostics: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scheduled_steps",
            _coerce_tuple_of_type(
                self.scheduled_steps,
                ScheduledStep,
                field_name="PipelineScheduleResult.scheduled_steps",
            ),
        )
        object.__setattr__(
            self,
            "sram_intervals",
            _coerce_tuple_of_type(
                self.sram_intervals,
                SramAllocationInterval,
                field_name="PipelineScheduleResult.sram_intervals",
            ),
        )
        object.__setattr__(
            self,
            "scheduled_values",
            _coerce_tuple_of_type(
                self.scheduled_values,
                ScheduledValue,
                field_name="PipelineScheduleResult.scheduled_values",
            ),
        )
        object.__setattr__(
            self,
            "residency_windows",
            _coerce_tuple_of_type(
                self.residency_windows,
                ResidencyWindow,
                field_name="PipelineScheduleResult.residency_windows",
            ),
        )
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_json_mapping(
                self.diagnostics, field_name="PipelineScheduleResult.diagnostics"
            ),
        )
        object.__setattr__(
            self,
            "transfer_diagnostics",
            _freeze_json_mapping(
                self.transfer_diagnostics,
                field_name="PipelineScheduleResult.transfer_diagnostics",
            ),
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready representation of the schedule result."""

        return {
            "scheduled_steps": [step.to_json() for step in self.scheduled_steps],
            "sram_intervals": [interval.to_json() for interval in self.sram_intervals],
            "scheduled_values": [
                value.to_json() for value in self.scheduled_values
            ],
            "residency_windows": [
                window.to_json() for window in self.residency_windows
            ],
            "makespan": self.makespan,
            "feasible": self.feasible,
            "solver_name": self.solver_name,
            "diagnostics": _to_json_value(self.diagnostics),
            "transfer_diagnostics": _to_json_value(self.transfer_diagnostics),
        }


def get_pipeline_schedule_problem(
    ctx: CompileContext,
) -> PipelineScheduleProblem | None:
    """Return the pipeline schedule problem stored in compile metadata."""

    problem = _get_typed_metadata(
        ctx,
        PIPELINE_SCHEDULE_PROBLEM_METADATA_KEY,
        PipelineScheduleProblem,
    )
    if problem is None:
        return None
    return problem


def set_pipeline_schedule_problem(
    ctx: CompileContext, problem: PipelineScheduleProblem
) -> None:
    """Store the typed pipeline schedule problem in compile metadata."""

    if not isinstance(problem, PipelineScheduleProblem):
        raise TypeError(
            "problem must be PipelineScheduleProblem"
        )
    ctx.metadata[PIPELINE_SCHEDULE_PROBLEM_METADATA_KEY] = problem


def get_pipeline_schedule_result(
    ctx: CompileContext,
) -> PipelineScheduleResult | None:
    """Return the pipeline schedule result stored in compile metadata."""

    result = _get_typed_metadata(
        ctx,
        PIPELINE_SCHEDULE_RESULT_METADATA_KEY,
        PipelineScheduleResult,
    )
    if result is None:
        return None
    return result


def get_pipeline_scheduled_values(
    ctx: CompileContext,
) -> tuple[ScheduledValue, ...]:
    """Return scheduled values from the stored schedule result when available."""

    result = get_pipeline_schedule_result(ctx)
    if result is not None:
        return result.scheduled_values
    problem = get_pipeline_schedule_problem(ctx)
    if problem is not None:
        return problem.scheduled_values
    return ()


def get_pipeline_residency_windows(
    ctx: CompileContext,
) -> tuple[ResidencyWindow, ...]:
    """Return residency windows from the stored schedule result when available."""

    result = get_pipeline_schedule_result(ctx)
    if result is not None:
        return result.residency_windows
    problem = get_pipeline_schedule_problem(ctx)
    if problem is not None:
        return problem.residency_windows
    return ()


def get_pipeline_transfer_diagnostics(
    ctx: CompileContext,
) -> Mapping[str, JsonValue]:
    """Return transfer-specific diagnostics from the stored schedule result."""

    result = get_pipeline_schedule_result(ctx)
    if result is None:
        return MappingProxyType({})
    return result.transfer_diagnostics


def set_pipeline_schedule_result(
    ctx: CompileContext, result: PipelineScheduleResult
) -> None:
    """Store the typed pipeline schedule result in compile metadata."""

    if not isinstance(result, PipelineScheduleResult):
        raise TypeError(
            "result must be PipelineScheduleResult"
        )
    ctx.metadata[PIPELINE_SCHEDULE_RESULT_METADATA_KEY] = result


__all__ = [
    "PIPELINE_SCHEDULE_PROBLEM_METADATA_KEY",
    "PIPELINE_SCHEDULE_RESULT_METADATA_KEY",
    "PipelineResourceKind",
    "PipelineScheduleProblem",
    "PipelineScheduleResult",
    "JsonScalar",
    "JsonValue",
    "ResidencyWindow",
    "ScheduleDependencyKind",
    "ScheduleEdge",
    "ScheduledStep",
    "ScheduleStep",
    "ScheduleStepKind",
    "ScheduledValue",
    "ScheduledValueHomeTier",
    "SramAllocationInterval",
    "SramValue",
    "TransferStep",
    "TransferStepKind",
    "get_pipeline_schedule_problem",
    "get_pipeline_residency_windows",
    "get_pipeline_schedule_result",
    "get_pipeline_scheduled_values",
    "get_pipeline_transfer_diagnostics",
    "set_pipeline_schedule_problem",
    "set_pipeline_schedule_result",
]
