"""External joint tiling and scheduling IR plus compile-context helpers."""

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


JOINT_TILING_SCHEDULE_PROBLEM_METADATA_KEY = "joint_tiling_schedule_problem"
JOINT_TILING_SCHEDULE_SOLUTION_METADATA_KEY = "joint_tiling_schedule_solution"
JOINT_TILING_SCHEDULE_FAILURE_METADATA_KEY = "joint_tiling_schedule_failure"

JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION = "joint_tiling_schedule_problem_v1"
JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION = "joint_tiling_schedule_solution_v1"
JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION = "joint_tiling_schedule_failure_v1"

JOINT_TILING_SCHEDULE_OBJECTIVE = "min_makespan"


def _freeze_json_mapping(
    value: object, *, field_name: str
) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    frozen_items: dict[str, JsonValue] = {}
    keys = tuple(value.keys())
    for key in keys:
        if not isinstance(key, str):
            raise TypeError(
                f"{field_name} keys must be strings, got {type(key).__name__}"
            )
    for key in sorted(keys):
        frozen_items[key] = _freeze_json_value(value[key], path=f"{field_name}.{key}")
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
            raise ValueError(
                f"{field_name} must be one of {tuple(member.value for member in enum_type)}"
            ) from exc
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


def _coerce_int_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a sequence of integers")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of integers") from exc
    for index, item in enumerate(items):
        _validate_non_negative_int(item, field_name=f"{field_name}[{index}]")
    return items  # type: ignore[return-value]


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


def _validate_non_negative_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _validate_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool")
    return value


def _validate_optional_str(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str or None")
    return value


def _validate_schema_version(value: str, *, expected: str, field_name: str) -> str:
    if value != expected:
        raise ValueError(f"{field_name} must be {expected!r}")
    return value


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    keys = tuple(value.keys())
    for key in keys:
        if not isinstance(key, str):
            raise TypeError(
                f"{field_name} keys must be strings, got {type(key).__name__}"
            )
    return value


def _require_field(payload: Mapping[str, object], field_name: str) -> object:
    if field_name not in payload:
        raise ValueError(f"{field_name} is required")
    return payload[field_name]


def _parse_object_list(
    payload: Mapping[str, object],
    field_name: str,
    factory: type,
) -> tuple[object, ...]:
    value = _require_field(payload, field_name)
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be an array")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an array") from exc
    parsed: list[object] = []
    for index, item in enumerate(items):
        mapping = _require_mapping(item, field_name=f"{field_name}[{index}]")
        parsed.append(factory.from_json(mapping))
    return tuple(parsed)


def _validate_unique_attr(
    items: tuple[object, ...], *, attr_name: str, field_name: str
) -> None:
    seen: set[str] = set()
    for item in items:
        value = getattr(item, attr_name)
        if value in seen:
            raise ValueError(f"duplicate {attr_name} in {field_name}")
        seen.add(value)


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


class JointRegionKind(str, Enum):
    SINGLE_OP = "single_op"
    FUSED_GROUP = "fused_group"


class JointDependencyEdgeKind(str, Enum):
    DATA = "data"
    ORDER = "order"


class JointValueTier(str, Enum):
    UNMATERIALIZED = "unmaterialized"
    INPUT = "input"
    CONST = "const"
    SLOW = "slow"
    SRAM = "sram"


class JointActionKind(str, Enum):
    COMPUTE = "compute"
    DMA_IN = "dma_in"
    DMA_OUT = "dma_out"
    SPILL = "spill"
    RELOAD = "reload"


class JointResourceKind(str, Enum):
    DMA = "DMA"
    MATMUL = "MATMUL"
    SHAPE = "SHAPE"
    OTHER = "OTHER"


class JointFailureStatus(str, Enum):
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    INVALID_PROBLEM = "invalid_problem"
    ERROR = "error"


class JointFailureCategory(str, Enum):
    INVALID_SOLUTION = "invalid_solution"
    INCOMPLETE_SOLUTION = "incomplete_solution"
    DEPENDENCY_VIOLATION = "dependency_violation"
    RESOURCE_OVERLAP = "resource_overlap"
    SRAM_CAPACITY_EXCEEDED = "sram_capacity_exceeded"
    ILLEGAL_TRANSFER = "illegal_transfer"
    INCOMPATIBLE_RECIPE_BOUNDARY = "incompatible_recipe_boundary"
    SOLVER_REPORTED_INFEASIBLE = "solver_reported_infeasible"


@dataclass(frozen=True)
class JointTileSpec:
    axes: tuple[str, ...]
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "axes", _coerce_str_tuple(self.axes, field_name="JointTileSpec.axes")
        )
        object.__setattr__(
            self, "shape", _coerce_int_tuple(self.shape, field_name="JointTileSpec.shape")
        )

    def to_json(self) -> dict[str, object]:
        return {"axes": list(self.axes), "shape": list(self.shape)}

    @classmethod
    def from_json(cls, payload: object) -> JointTileSpec:
        mapping = _require_mapping(payload, field_name="JointTileSpec")
        return cls(
            axes=_require_field(mapping, "axes"),
            shape=_require_field(mapping, "shape"),
        )


@dataclass(frozen=True)
class JointLayoutSpec:
    layout_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layout_tags",
            _coerce_str_tuple(self.layout_tags, field_name="JointLayoutSpec.layout_tags"),
        )

    def to_json(self) -> dict[str, object]:
        return {"layout_tags": list(self.layout_tags)}

    @classmethod
    def from_json(cls, payload: object) -> JointLayoutSpec:
        mapping = _require_mapping(payload, field_name="JointLayoutSpec")
        return cls(layout_tags=_require_field(mapping, "layout_tags"))


@dataclass(frozen=True)
class JointValueFootprint:
    resident_bytes: int
    scratch_bytes: int
    transfer_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resident_bytes",
            _validate_non_negative_int(
                self.resident_bytes, field_name="JointValueFootprint.resident_bytes"
            ),
        )
        object.__setattr__(
            self,
            "scratch_bytes",
            _validate_non_negative_int(
                self.scratch_bytes, field_name="JointValueFootprint.scratch_bytes"
            ),
        )
        object.__setattr__(
            self,
            "transfer_bytes",
            _validate_non_negative_int(
                self.transfer_bytes, field_name="JointValueFootprint.transfer_bytes"
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "resident_bytes": self.resident_bytes,
            "scratch_bytes": self.scratch_bytes,
            "transfer_bytes": self.transfer_bytes,
        }

    @classmethod
    def from_json(cls, payload: object) -> JointValueFootprint:
        mapping = _require_mapping(payload, field_name="JointValueFootprint")
        return cls(
            resident_bytes=_require_field(mapping, "resident_bytes"),
            scratch_bytes=_require_field(mapping, "scratch_bytes"),
            transfer_bytes=_require_field(mapping, "transfer_bytes"),
        )


@dataclass(frozen=True)
class JointCostParameters:
    latency: int
    launch_overhead: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "latency",
            _validate_non_negative_int(
                self.latency, field_name="JointCostParameters.latency"
            ),
        )
        object.__setattr__(
            self,
            "launch_overhead",
            _validate_non_negative_int(
                self.launch_overhead,
                field_name="JointCostParameters.launch_overhead",
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "latency": self.latency,
            "launch_overhead": self.launch_overhead,
        }

    @classmethod
    def from_json(cls, payload: object) -> JointCostParameters:
        mapping = _require_mapping(payload, field_name="JointCostParameters")
        return cls(
            latency=_require_field(mapping, "latency"),
            launch_overhead=_require_field(mapping, "launch_overhead"),
        )


@dataclass(frozen=True)
class JointValueProducer:
    action_id: str

    def to_json(self) -> dict[str, object]:
        return {"action_id": self.action_id}

    @classmethod
    def from_json(cls, payload: object) -> JointValueProducer:
        mapping = _require_mapping(payload, field_name="JointValueProducer")
        return cls(action_id=_require_field(mapping, "action_id"))


@dataclass(frozen=True)
class JointValueConsumer:
    action_id: str

    def to_json(self) -> dict[str, object]:
        return {"action_id": self.action_id}

    @classmethod
    def from_json(cls, payload: object) -> JointValueConsumer:
        mapping = _require_mapping(payload, field_name="JointValueConsumer")
        return cls(action_id=_require_field(mapping, "action_id"))


@dataclass(frozen=True)
class JointCompatibleRecipePair:
    src_recipe_id: str
    dst_recipe_id: str

    def to_json(self) -> dict[str, object]:
        return {
            "src_recipe_id": self.src_recipe_id,
            "dst_recipe_id": self.dst_recipe_id,
        }

    @classmethod
    def from_json(cls, payload: object) -> JointCompatibleRecipePair:
        mapping = _require_mapping(payload, field_name="JointCompatibleRecipePair")
        return cls(
            src_recipe_id=_require_field(mapping, "src_recipe_id"),
            dst_recipe_id=_require_field(mapping, "dst_recipe_id"),
        )


@dataclass(frozen=True)
class JointRegion:
    region_id: str
    kind: JointRegionKind
    input_value_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]
    member_nodes: tuple[str, ...] = ()
    predecessor_region_ids: tuple[str, ...] = ()
    successor_region_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _coerce_enum(self.kind, JointRegionKind, field_name="JointRegion.kind"),
        )
        object.__setattr__(
            self,
            "input_value_ids",
            _coerce_str_tuple(
                self.input_value_ids, field_name="JointRegion.input_value_ids"
            ),
        )
        object.__setattr__(
            self,
            "output_value_ids",
            _coerce_str_tuple(
                self.output_value_ids, field_name="JointRegion.output_value_ids"
            ),
        )
        object.__setattr__(
            self,
            "member_nodes",
            _coerce_str_tuple(self.member_nodes, field_name="JointRegion.member_nodes"),
        )
        object.__setattr__(
            self,
            "predecessor_region_ids",
            _coerce_str_tuple(
                self.predecessor_region_ids,
                field_name="JointRegion.predecessor_region_ids",
            ),
        )
        object.__setattr__(
            self,
            "successor_region_ids",
            _coerce_str_tuple(
                self.successor_region_ids,
                field_name="JointRegion.successor_region_ids",
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "kind": self.kind.value,
            "member_nodes": list(self.member_nodes),
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "predecessor_region_ids": list(self.predecessor_region_ids),
            "successor_region_ids": list(self.successor_region_ids),
        }

    @classmethod
    def from_json(cls, payload: object) -> JointRegion:
        mapping = _require_mapping(payload, field_name="JointRegion")
        return cls(
            region_id=_require_field(mapping, "region_id"),
            kind=_require_field(mapping, "kind"),
            input_value_ids=_require_field(mapping, "input_value_ids"),
            output_value_ids=_require_field(mapping, "output_value_ids"),
            member_nodes=mapping.get("member_nodes", ()),
            predecessor_region_ids=mapping.get("predecessor_region_ids", ()),
            successor_region_ids=mapping.get("successor_region_ids", ()),
        )


@dataclass(frozen=True)
class JointRecipe:
    recipe_id: str
    region_id: str
    tile_spec: JointTileSpec
    layout_spec: JointLayoutSpec
    activates_action_ids: tuple[str, ...]
    value_footprint: JointValueFootprint
    cost_parameters: JointCostParameters

    def __post_init__(self) -> None:
        if not isinstance(self.tile_spec, JointTileSpec):
            raise TypeError("JointRecipe.tile_spec must be JointTileSpec")
        if not isinstance(self.layout_spec, JointLayoutSpec):
            raise TypeError("JointRecipe.layout_spec must be JointLayoutSpec")
        if not isinstance(self.value_footprint, JointValueFootprint):
            raise TypeError("JointRecipe.value_footprint must be JointValueFootprint")
        if not isinstance(self.cost_parameters, JointCostParameters):
            raise TypeError("JointRecipe.cost_parameters must be JointCostParameters")
        object.__setattr__(
            self,
            "activates_action_ids",
            _coerce_str_tuple(
                self.activates_action_ids,
                field_name="JointRecipe.activates_action_ids",
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "recipe_id": self.recipe_id,
            "region_id": self.region_id,
            "tile_spec": self.tile_spec.to_json(),
            "layout_spec": self.layout_spec.to_json(),
            "activates_action_ids": list(self.activates_action_ids),
            "value_footprint": self.value_footprint.to_json(),
            "cost_parameters": self.cost_parameters.to_json(),
        }

    @classmethod
    def from_json(cls, payload: object) -> JointRecipe:
        mapping = _require_mapping(payload, field_name="JointRecipe")
        return cls(
            recipe_id=_require_field(mapping, "recipe_id"),
            region_id=_require_field(mapping, "region_id"),
            tile_spec=JointTileSpec.from_json(_require_field(mapping, "tile_spec")),
            layout_spec=JointLayoutSpec.from_json(_require_field(mapping, "layout_spec")),
            activates_action_ids=_require_field(mapping, "activates_action_ids"),
            value_footprint=JointValueFootprint.from_json(
                _require_field(mapping, "value_footprint")
            ),
            cost_parameters=JointCostParameters.from_json(
                _require_field(mapping, "cost_parameters")
            ),
        )


@dataclass(frozen=True)
class JointValue:
    value_id: str
    size_bytes: int
    initial_tier: JointValueTier
    required_final_tier: JointValueTier
    must_keep: bool
    spillable: bool
    allows_multiple_sram_windows: bool
    producer: JointValueProducer | None
    consumers: tuple[JointValueConsumer, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "size_bytes",
            _validate_non_negative_int(self.size_bytes, field_name="JointValue.size_bytes"),
        )
        object.__setattr__(
            self,
            "initial_tier",
            _coerce_enum(
                self.initial_tier,
                JointValueTier,
                field_name="JointValue.initial_tier",
            ),
        )
        object.__setattr__(
            self,
            "required_final_tier",
            _coerce_enum(
                self.required_final_tier,
                JointValueTier,
                field_name="JointValue.required_final_tier",
            ),
        )
        object.__setattr__(
            self,
            "must_keep",
            _validate_bool(self.must_keep, field_name="JointValue.must_keep"),
        )
        object.__setattr__(
            self,
            "spillable",
            _validate_bool(self.spillable, field_name="JointValue.spillable"),
        )
        object.__setattr__(
            self,
            "allows_multiple_sram_windows",
            _validate_bool(
                self.allows_multiple_sram_windows,
                field_name="JointValue.allows_multiple_sram_windows",
            ),
        )
        if self.producer is not None and not isinstance(self.producer, JointValueProducer):
            raise TypeError("JointValue.producer must be JointValueProducer or None")
        object.__setattr__(
            self,
            "consumers",
            _coerce_tuple_of_type(
                self.consumers,
                JointValueConsumer,
                field_name="JointValue.consumers",
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "value_id": self.value_id,
            "size_bytes": self.size_bytes,
            "initial_tier": self.initial_tier.value,
            "required_final_tier": self.required_final_tier.value,
            "must_keep": self.must_keep,
            "spillable": self.spillable,
            "allows_multiple_sram_windows": self.allows_multiple_sram_windows,
            "producer": None if self.producer is None else self.producer.to_json(),
            "consumers": [consumer.to_json() for consumer in self.consumers],
        }

    @classmethod
    def from_json(cls, payload: object) -> JointValue:
        mapping = _require_mapping(payload, field_name="JointValue")
        producer_payload = _require_field(mapping, "producer")
        consumers_payload = _require_field(mapping, "consumers")
        if isinstance(consumers_payload, str):
            raise TypeError("consumers must be an array")
        try:
            consumer_items = tuple(consumers_payload)
        except TypeError as exc:
            raise TypeError("consumers must be an array") from exc
        return cls(
            value_id=_require_field(mapping, "value_id"),
            size_bytes=_require_field(mapping, "size_bytes"),
            initial_tier=_require_field(mapping, "initial_tier"),
            required_final_tier=_require_field(mapping, "required_final_tier"),
            must_keep=_require_field(mapping, "must_keep"),
            spillable=_require_field(mapping, "spillable"),
            allows_multiple_sram_windows=_require_field(
                mapping, "allows_multiple_sram_windows"
            ),
            producer=(
                None
                if producer_payload is None
                else JointValueProducer.from_json(producer_payload)
            ),
            consumers=tuple(
                JointValueConsumer.from_json(
                    _require_mapping(item, field_name=f"consumers[{index}]")
                )
                for index, item in enumerate(consumer_items)
            ),
        )


@dataclass(frozen=True)
class JointAction:
    action_id: str
    kind: JointActionKind
    resource_kind: JointResourceKind
    duration: int
    launch_overhead: int
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    temp_bytes: int
    is_optional: bool
    region_id: str | None
    recipe_id: str | None
    optional_value_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _coerce_enum(self.kind, JointActionKind, field_name="JointAction.kind"),
        )
        object.__setattr__(
            self,
            "resource_kind",
            _coerce_enum(
                self.resource_kind,
                JointResourceKind,
                field_name="JointAction.resource_kind",
            ),
        )
        object.__setattr__(
            self,
            "duration",
            _validate_non_negative_int(self.duration, field_name="JointAction.duration"),
        )
        object.__setattr__(
            self,
            "launch_overhead",
            _validate_non_negative_int(
                self.launch_overhead, field_name="JointAction.launch_overhead"
            ),
        )
        object.__setattr__(
            self, "reads", _coerce_str_tuple(self.reads, field_name="JointAction.reads")
        )
        object.__setattr__(
            self,
            "writes",
            _coerce_str_tuple(self.writes, field_name="JointAction.writes"),
        )
        object.__setattr__(
            self,
            "temp_bytes",
            _validate_non_negative_int(self.temp_bytes, field_name="JointAction.temp_bytes"),
        )
        object.__setattr__(
            self,
            "is_optional",
            _validate_bool(self.is_optional, field_name="JointAction.is_optional"),
        )
        object.__setattr__(
            self,
            "region_id",
            _validate_optional_str(self.region_id, field_name="JointAction.region_id"),
        )
        object.__setattr__(
            self,
            "recipe_id",
            _validate_optional_str(self.recipe_id, field_name="JointAction.recipe_id"),
        )
        object.__setattr__(
            self,
            "optional_value_id",
            _validate_optional_str(
                self.optional_value_id, field_name="JointAction.optional_value_id"
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "resource_kind": self.resource_kind.value,
            "duration": self.duration,
            "launch_overhead": self.launch_overhead,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "temp_bytes": self.temp_bytes,
            "is_optional": self.is_optional,
            "region_id": self.region_id,
            "recipe_id": self.recipe_id,
            "optional_value_id": self.optional_value_id,
        }

    @classmethod
    def from_json(cls, payload: object) -> JointAction:
        mapping = _require_mapping(payload, field_name="JointAction")
        return cls(
            action_id=_require_field(mapping, "action_id"),
            kind=_require_field(mapping, "kind"),
            resource_kind=_require_field(mapping, "resource_kind"),
            duration=_require_field(mapping, "duration"),
            launch_overhead=_require_field(mapping, "launch_overhead"),
            reads=_require_field(mapping, "reads"),
            writes=_require_field(mapping, "writes"),
            temp_bytes=_require_field(mapping, "temp_bytes"),
            is_optional=_require_field(mapping, "is_optional"),
            region_id=_require_field(mapping, "region_id"),
            recipe_id=_require_field(mapping, "recipe_id"),
            optional_value_id=_require_field(mapping, "optional_value_id"),
        )


@dataclass(frozen=True)
class JointBoundaryConstraint:
    boundary_id: str
    src_region_id: str
    dst_region_id: str
    compatible_recipe_pairs: tuple[JointCompatibleRecipePair, ...]
    required_layout_relations: tuple[str, ...] = ()
    required_tile_domain_relations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compatible_recipe_pairs",
            _coerce_tuple_of_type(
                self.compatible_recipe_pairs,
                JointCompatibleRecipePair,
                field_name="JointBoundaryConstraint.compatible_recipe_pairs",
            ),
        )
        object.__setattr__(
            self,
            "required_layout_relations",
            _coerce_str_tuple(
                self.required_layout_relations,
                field_name="JointBoundaryConstraint.required_layout_relations",
            ),
        )
        object.__setattr__(
            self,
            "required_tile_domain_relations",
            _coerce_str_tuple(
                self.required_tile_domain_relations,
                field_name="JointBoundaryConstraint.required_tile_domain_relations",
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "boundary_id": self.boundary_id,
            "src_region_id": self.src_region_id,
            "dst_region_id": self.dst_region_id,
            "compatible_recipe_pairs": [
                pair.to_json() for pair in self.compatible_recipe_pairs
            ],
            "required_layout_relations": list(self.required_layout_relations),
            "required_tile_domain_relations": list(
                self.required_tile_domain_relations
            ),
        }

    @classmethod
    def from_json(cls, payload: object) -> JointBoundaryConstraint:
        mapping = _require_mapping(payload, field_name="JointBoundaryConstraint")
        pairs_payload = _require_field(mapping, "compatible_recipe_pairs")
        if isinstance(pairs_payload, str):
            raise TypeError("compatible_recipe_pairs must be an array")
        try:
            pair_items = tuple(pairs_payload)
        except TypeError as exc:
            raise TypeError("compatible_recipe_pairs must be an array") from exc
        return cls(
            boundary_id=_require_field(mapping, "boundary_id"),
            src_region_id=_require_field(mapping, "src_region_id"),
            dst_region_id=_require_field(mapping, "dst_region_id"),
            compatible_recipe_pairs=tuple(
                JointCompatibleRecipePair.from_json(
                    _require_mapping(item, field_name=f"compatible_recipe_pairs[{index}]")
                )
                for index, item in enumerate(pair_items)
            ),
            required_layout_relations=mapping.get("required_layout_relations", ()),
            required_tile_domain_relations=mapping.get(
                "required_tile_domain_relations", ()
            ),
        )


@dataclass(frozen=True)
class JointDependencyEdge:
    src_action_id: str
    dst_action_id: str
    kind: JointDependencyEdgeKind = JointDependencyEdgeKind.DATA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _coerce_enum(
                self.kind,
                JointDependencyEdgeKind,
                field_name="JointDependencyEdge.kind",
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "src_action_id": self.src_action_id,
            "dst_action_id": self.dst_action_id,
            "kind": self.kind.value,
        }

    @classmethod
    def from_json(cls, payload: object) -> JointDependencyEdge:
        mapping = _require_mapping(payload, field_name="JointDependencyEdge")
        return cls(
            src_action_id=_require_field(mapping, "src_action_id"),
            dst_action_id=_require_field(mapping, "dst_action_id"),
            kind=_require_field(mapping, "kind"),
        )


@dataclass(frozen=True)
class JointResource:
    resource_kind: JointResourceKind
    slot_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_kind",
            _coerce_enum(
                self.resource_kind,
                JointResourceKind,
                field_name="JointResource.resource_kind",
            ),
        )
        slot_count = _validate_non_negative_int(
            self.slot_count, field_name="JointResource.slot_count"
        )
        if slot_count != 1:
            raise ValueError("JointResource.slot_count must be 1")
        object.__setattr__(self, "slot_count", slot_count)

    def to_json(self) -> dict[str, object]:
        return {
            "resource_kind": self.resource_kind.value,
            "slot_count": self.slot_count,
        }

    @classmethod
    def from_json(cls, payload: object) -> JointResource:
        mapping = _require_mapping(payload, field_name="JointResource")
        return cls(
            resource_kind=_require_field(mapping, "resource_kind"),
            slot_count=_require_field(mapping, "slot_count"),
        )


@dataclass(frozen=True)
class JointSelectedRecipe:
    region_id: str
    recipe_id: str

    def to_json(self) -> dict[str, object]:
        return {"region_id": self.region_id, "recipe_id": self.recipe_id}

    @classmethod
    def from_json(cls, payload: object) -> JointSelectedRecipe:
        mapping = _require_mapping(payload, field_name="JointSelectedRecipe")
        return cls(
            region_id=_require_field(mapping, "region_id"),
            recipe_id=_require_field(mapping, "recipe_id"),
        )


@dataclass(frozen=True)
class JointScheduledAction:
    action_id: str
    start_time: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_time",
            _validate_non_negative_int(
                self.start_time, field_name="JointScheduledAction.start_time"
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {"action_id": self.action_id, "start_time": self.start_time}

    @classmethod
    def from_json(cls, payload: object) -> JointScheduledAction:
        mapping = _require_mapping(payload, field_name="JointScheduledAction")
        return cls(
            action_id=_require_field(mapping, "action_id"),
            start_time=_require_field(mapping, "start_time"),
        )


@dataclass(frozen=True)
class JointResidencyWindow:
    value_id: str
    start_time: int
    end_time: int

    def __post_init__(self) -> None:
        start_time = _validate_non_negative_int(
            self.start_time, field_name="JointResidencyWindow.start_time"
        )
        end_time = _validate_non_negative_int(
            self.end_time, field_name="JointResidencyWindow.end_time"
        )
        if end_time <= start_time:
            raise ValueError("JointResidencyWindow.end_time must be greater than start_time")
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "end_time", end_time)

    def to_json(self) -> dict[str, object]:
        return {
            "value_id": self.value_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_json(cls, payload: object) -> JointResidencyWindow:
        mapping = _require_mapping(payload, field_name="JointResidencyWindow")
        return cls(
            value_id=_require_field(mapping, "value_id"),
            start_time=_require_field(mapping, "start_time"),
            end_time=_require_field(mapping, "end_time"),
        )


@dataclass(frozen=True)
class JointProblem:
    schema_version: str
    regions: tuple[JointRegion, ...]
    recipes: tuple[JointRecipe, ...]
    values: tuple[JointValue, ...]
    actions: tuple[JointAction, ...]
    boundary_constraints: tuple[JointBoundaryConstraint, ...]
    dependency_edges: tuple[JointDependencyEdge, ...]
    resources: tuple[JointResource, ...]
    sram_capacity_bytes: int
    objective: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _validate_schema_version(
                self.schema_version,
                expected=JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
                field_name="JointProblem.schema_version",
            ),
        )
        object.__setattr__(
            self,
            "regions",
            _coerce_tuple_of_type(self.regions, JointRegion, field_name="JointProblem.regions"),
        )
        object.__setattr__(
            self,
            "recipes",
            _coerce_tuple_of_type(self.recipes, JointRecipe, field_name="JointProblem.recipes"),
        )
        object.__setattr__(
            self,
            "values",
            _coerce_tuple_of_type(self.values, JointValue, field_name="JointProblem.values"),
        )
        object.__setattr__(
            self,
            "actions",
            _coerce_tuple_of_type(self.actions, JointAction, field_name="JointProblem.actions"),
        )
        object.__setattr__(
            self,
            "boundary_constraints",
            _coerce_tuple_of_type(
                self.boundary_constraints,
                JointBoundaryConstraint,
                field_name="JointProblem.boundary_constraints",
            ),
        )
        object.__setattr__(
            self,
            "dependency_edges",
            _coerce_tuple_of_type(
                self.dependency_edges,
                JointDependencyEdge,
                field_name="JointProblem.dependency_edges",
            ),
        )
        object.__setattr__(
            self,
            "resources",
            _coerce_tuple_of_type(
                self.resources,
                JointResource,
                field_name="JointProblem.resources",
            ),
        )
        object.__setattr__(
            self,
            "sram_capacity_bytes",
            _validate_non_negative_int(
                self.sram_capacity_bytes,
                field_name="JointProblem.sram_capacity_bytes",
            ),
        )
        if self.objective != JOINT_TILING_SCHEDULE_OBJECTIVE:
            raise ValueError(
                f"JointProblem.objective must be {JOINT_TILING_SCHEDULE_OBJECTIVE!r}"
            )
        _validate_unique_attr(self.regions, attr_name="region_id", field_name="regions")
        _validate_unique_attr(self.recipes, attr_name="recipe_id", field_name="recipes")
        _validate_unique_attr(self.values, attr_name="value_id", field_name="values")
        _validate_unique_attr(self.actions, attr_name="action_id", field_name="actions")
        _validate_unique_attr(
            self.boundary_constraints,
            attr_name="boundary_id",
            field_name="boundary_constraints",
        )
        _validate_unique_attr(
            self.resources, attr_name="resource_kind", field_name="resources"
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "regions": [region.to_json() for region in self.regions],
            "recipes": [recipe.to_json() for recipe in self.recipes],
            "values": [value.to_json() for value in self.values],
            "actions": [action.to_json() for action in self.actions],
            "boundary_constraints": [
                boundary.to_json() for boundary in self.boundary_constraints
            ],
            "dependency_edges": [edge.to_json() for edge in self.dependency_edges],
            "resources": [resource.to_json() for resource in self.resources],
            "sram_capacity_bytes": self.sram_capacity_bytes,
            "objective": self.objective,
        }

    @classmethod
    def from_json(cls, payload: object) -> JointProblem:
        mapping = _require_mapping(payload, field_name="JointProblem")
        schema_version = _require_field(mapping, "schema_version")
        if schema_version != JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION:
            raise ValueError(
                "JointProblem.schema_version must be "
                f"{JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION!r}"
            )
        return cls(
            schema_version=schema_version,
            regions=_parse_object_list(mapping, "regions", JointRegion),
            recipes=_parse_object_list(mapping, "recipes", JointRecipe),
            values=_parse_object_list(mapping, "values", JointValue),
            actions=_parse_object_list(mapping, "actions", JointAction),
            boundary_constraints=_parse_object_list(
                mapping, "boundary_constraints", JointBoundaryConstraint
            ),
            dependency_edges=_parse_object_list(
                mapping, "dependency_edges", JointDependencyEdge
            ),
            resources=_parse_object_list(mapping, "resources", JointResource),
            sram_capacity_bytes=_require_field(mapping, "sram_capacity_bytes"),
            objective=_require_field(mapping, "objective"),
        )


@dataclass(frozen=True)
class JointSolution:
    schema_version: str
    selected_recipes: tuple[JointSelectedRecipe, ...]
    scheduled_actions: tuple[JointScheduledAction, ...]
    residency_windows: tuple[JointResidencyWindow, ...]
    objective_value: int
    diagnostics: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _validate_schema_version(
                self.schema_version,
                expected=JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
                field_name="JointSolution.schema_version",
            ),
        )
        object.__setattr__(
            self,
            "selected_recipes",
            _coerce_tuple_of_type(
                self.selected_recipes,
                JointSelectedRecipe,
                field_name="JointSolution.selected_recipes",
            ),
        )
        object.__setattr__(
            self,
            "scheduled_actions",
            _coerce_tuple_of_type(
                self.scheduled_actions,
                JointScheduledAction,
                field_name="JointSolution.scheduled_actions",
            ),
        )
        object.__setattr__(
            self,
            "residency_windows",
            _coerce_tuple_of_type(
                self.residency_windows,
                JointResidencyWindow,
                field_name="JointSolution.residency_windows",
            ),
        )
        object.__setattr__(
            self,
            "objective_value",
            _validate_non_negative_int(
                self.objective_value, field_name="JointSolution.objective_value"
            ),
        )
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_json_mapping(self.diagnostics, field_name="JointSolution.diagnostics"),
        )
        _validate_unique_attr(
            self.selected_recipes,
            attr_name="region_id",
            field_name="selected_recipes",
        )
        _validate_unique_attr(
            self.scheduled_actions,
            attr_name="action_id",
            field_name="scheduled_actions",
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selected_recipes": [
                recipe.to_json() for recipe in self.selected_recipes
            ],
            "scheduled_actions": [
                action.to_json() for action in self.scheduled_actions
            ],
            "residency_windows": [
                window.to_json() for window in self.residency_windows
            ],
            "objective_value": self.objective_value,
            "diagnostics": _to_json_value(self.diagnostics),
        }

    @classmethod
    def from_json(cls, payload: object) -> JointSolution:
        mapping = _require_mapping(payload, field_name="JointSolution")
        schema_version = _require_field(mapping, "schema_version")
        if schema_version != JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION:
            raise ValueError(
                "JointSolution.schema_version must be "
                f"{JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION!r}"
            )
        return cls(
            schema_version=schema_version,
            selected_recipes=_parse_object_list(
                mapping, "selected_recipes", JointSelectedRecipe
            ),
            scheduled_actions=_parse_object_list(
                mapping, "scheduled_actions", JointScheduledAction
            ),
            residency_windows=_parse_object_list(
                mapping, "residency_windows", JointResidencyWindow
            ),
            objective_value=_require_field(mapping, "objective_value"),
            diagnostics=_require_field(mapping, "diagnostics"),
        )


@dataclass(frozen=True)
class JointFailure:
    schema_version: str
    status: JointFailureStatus
    error_category: JointFailureCategory
    diagnostics: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _validate_schema_version(
                self.schema_version,
                expected=JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
                field_name="JointFailure.schema_version",
            ),
        )
        object.__setattr__(
            self,
            "status",
            _coerce_enum(
                self.status, JointFailureStatus, field_name="JointFailure.status"
            ),
        )
        object.__setattr__(
            self,
            "error_category",
            _coerce_enum(
                self.error_category,
                JointFailureCategory,
                field_name="JointFailure.error_category",
            ),
        )
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_json_mapping(self.diagnostics, field_name="JointFailure.diagnostics"),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "error_category": self.error_category.value,
            "diagnostics": _to_json_value(self.diagnostics),
        }

    @classmethod
    def from_json(cls, payload: object) -> JointFailure:
        mapping = _require_mapping(payload, field_name="JointFailure")
        schema_version = _require_field(mapping, "schema_version")
        if schema_version != JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION:
            raise ValueError(
                "JointFailure.schema_version must be "
                f"{JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION!r}"
            )
        return cls(
            schema_version=schema_version,
            status=_require_field(mapping, "status"),
            error_category=_require_field(mapping, "error_category"),
            diagnostics=_require_field(mapping, "diagnostics"),
        )


def get_joint_tiling_schedule_problem(ctx: CompileContext) -> JointProblem | None:
    problem = _get_typed_metadata(
        ctx,
        JOINT_TILING_SCHEDULE_PROBLEM_METADATA_KEY,
        JointProblem,
    )
    if problem is None:
        return None
    return problem


def set_joint_tiling_schedule_problem(ctx: CompileContext, problem: JointProblem) -> None:
    if not isinstance(problem, JointProblem):
        raise TypeError("problem must be JointProblem")
    ctx.metadata[JOINT_TILING_SCHEDULE_PROBLEM_METADATA_KEY] = problem


def get_joint_tiling_schedule_solution(ctx: CompileContext) -> JointSolution | None:
    solution = _get_typed_metadata(
        ctx,
        JOINT_TILING_SCHEDULE_SOLUTION_METADATA_KEY,
        JointSolution,
    )
    if solution is None:
        return None
    return solution


def set_joint_tiling_schedule_solution(
    ctx: CompileContext, solution: JointSolution
) -> None:
    if not isinstance(solution, JointSolution):
        raise TypeError("solution must be JointSolution")
    ctx.metadata[JOINT_TILING_SCHEDULE_SOLUTION_METADATA_KEY] = solution


def get_joint_tiling_schedule_failure(ctx: CompileContext) -> JointFailure | None:
    failure = _get_typed_metadata(
        ctx,
        JOINT_TILING_SCHEDULE_FAILURE_METADATA_KEY,
        JointFailure,
    )
    if failure is None:
        return None
    return failure


def set_joint_tiling_schedule_failure(ctx: CompileContext, failure: JointFailure) -> None:
    if not isinstance(failure, JointFailure):
        raise TypeError("failure must be JointFailure")
    ctx.metadata[JOINT_TILING_SCHEDULE_FAILURE_METADATA_KEY] = failure


__all__ = [
    "JOINT_TILING_SCHEDULE_FAILURE_METADATA_KEY",
    "JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION",
    "JOINT_TILING_SCHEDULE_OBJECTIVE",
    "JOINT_TILING_SCHEDULE_PROBLEM_METADATA_KEY",
    "JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION",
    "JOINT_TILING_SCHEDULE_SOLUTION_METADATA_KEY",
    "JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION",
    "JsonScalar",
    "JsonValue",
    "JointAction",
    "JointActionKind",
    "JointBoundaryConstraint",
    "JointCompatibleRecipePair",
    "JointCostParameters",
    "JointDependencyEdge",
    "JointDependencyEdgeKind",
    "JointFailure",
    "JointFailureCategory",
    "JointFailureStatus",
    "JointLayoutSpec",
    "JointProblem",
    "JointRecipe",
    "JointRegion",
    "JointRegionKind",
    "JointResidencyWindow",
    "JointResource",
    "JointResourceKind",
    "JointScheduledAction",
    "JointSelectedRecipe",
    "JointSolution",
    "JointTileSpec",
    "JointValue",
    "JointValueConsumer",
    "JointValueFootprint",
    "JointValueProducer",
    "JointValueTier",
    "get_joint_tiling_schedule_failure",
    "get_joint_tiling_schedule_problem",
    "get_joint_tiling_schedule_solution",
    "set_joint_tiling_schedule_failure",
    "set_joint_tiling_schedule_problem",
    "set_joint_tiling_schedule_solution",
]
