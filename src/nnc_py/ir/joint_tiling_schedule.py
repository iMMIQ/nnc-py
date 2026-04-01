"""External joint tiling/schedule IR and compile-context metadata helpers."""

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


JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION = "joint_tiling_schedule_problem_v1"
JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION = "joint_tiling_schedule_solution_v1"
JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION = "joint_tiling_schedule_failure_v1"
JOINT_TILING_SCHEDULE_OBJECTIVE = "min_makespan"

JOINT_TILING_SCHEDULE_PROBLEM_METADATA_KEY = "joint_tiling_schedule_problem"
JOINT_TILING_SCHEDULE_SOLUTION_METADATA_KEY = "joint_tiling_schedule_solution"
JOINT_TILING_SCHEDULE_FAILURE_METADATA_KEY = "joint_tiling_schedule_failure"


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
            members = tuple(member.value for member in enum_type)
            raise ValueError(f"{field_name} must be one of {members}") from exc
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


def _coerce_non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _coerce_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _coerce_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool")
    return value


def _coerce_optional_str(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str or None")
    return value


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


def _require_mapping(payload: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    for key in payload:
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
    return payload


def _require_field(payload: Mapping[str, object], key: str, *, owner: str) -> object:
    if key not in payload:
        raise ValueError(f"{owner}.{key} is required")
    return payload[key]


def _ensure_unique_ids(
    items: tuple[object, ...], attr_name: str, *, label: str | None = None
) -> None:
    seen: set[str] = set()
    field_label = label or attr_name
    for item in items:
        item_id = getattr(item, attr_name)
        if item_id in seen:
            raise ValueError(f"duplicate {field_label}: {item_id}")
        seen.add(item_id)


def _parse_object_array(
    payload: Mapping[str, object], key: str, parser, *, owner: str
) -> tuple[object, ...]:
    raw = _require_field(payload, key, owner=owner)
    if isinstance(raw, str):
        raise TypeError(f"{owner}.{key} must be an array")
    try:
        items = tuple(raw)
    except TypeError as exc:
        raise TypeError(f"{owner}.{key} must be an array") from exc
    return tuple(
        parser(item, field_name=f"{owner}.{key}[{index}]")
        for index, item in enumerate(items)
    )


class JointRegionKind(str, Enum):
    SINGLE_OP = "single_op"
    FUSED_GROUP = "fused_group"


class JointValueTier(str, Enum):
    UNMATERIALIZED = "unmaterialized"
    INPUT = "input"
    CONST = "const"
    SLOW = "slow"
    SRAM = "sram"


class JointSramItemKind(str, Enum):
    TEMP_INTERVAL = "temp_interval"
    TRANSFER_BUFFER = "transfer_buffer"
    RESIDENT_WINDOW = "resident_window"


class JointActionKind(str, Enum):
    COMPUTE = "compute"
    DMA_IN = "dma_in"
    DMA_OUT = "dma_out"
    SPILL = "spill"
    RELOAD = "reload"


class JointDependencyEdgeKind(str, Enum):
    DATA = "data"
    ORDER = "order"


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
        if isinstance(self.shape, str):
            raise TypeError("JointTileSpec.shape must be a sequence of int")
        try:
            shape_items = tuple(self.shape)
        except TypeError as exc:
            raise TypeError("JointTileSpec.shape must be a sequence of int") from exc
        object.__setattr__(
            self,
            "shape",
            tuple(
                _coerce_non_negative_int(
                    item, field_name=f"JointTileSpec.shape[{index}]"
                )
                for index, item in enumerate(shape_items)
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {"axes": list(self.axes), "shape": list(self.shape)}

    @classmethod
    def from_json(
        cls, payload: object, *, field_name: str = "JointTileSpec"
    ) -> "JointTileSpec":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            axes=_require_field(mapping, "axes", owner=field_name),
            shape=_require_field(mapping, "shape", owner=field_name),
        )


@dataclass(frozen=True)
class JointLayoutSpec:
    layout_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layout_tags",
            _coerce_str_tuple(
                self.layout_tags, field_name="JointLayoutSpec.layout_tags"
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {"layout_tags": list(self.layout_tags)}

    @classmethod
    def from_json(
        cls, payload: object, *, field_name: str = "JointLayoutSpec"
    ) -> "JointLayoutSpec":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            layout_tags=_require_field(mapping, "layout_tags", owner=field_name)
        )


@dataclass(frozen=True)
class JointValueFootprint:
    resident_bytes: int
    scratch_bytes: int
    transfer_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resident_bytes",
            _coerce_non_negative_int(
                self.resident_bytes, field_name="JointValueFootprint.resident_bytes"
            ),
        )
        object.__setattr__(
            self,
            "scratch_bytes",
            _coerce_non_negative_int(
                self.scratch_bytes, field_name="JointValueFootprint.scratch_bytes"
            ),
        )
        object.__setattr__(
            self,
            "transfer_bytes",
            _coerce_non_negative_int(
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
    def from_json(
        cls, payload: object, *, field_name: str = "JointValueFootprint"
    ) -> "JointValueFootprint":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            resident_bytes=_require_field(mapping, "resident_bytes", owner=field_name),
            scratch_bytes=_require_field(mapping, "scratch_bytes", owner=field_name),
            transfer_bytes=_require_field(mapping, "transfer_bytes", owner=field_name),
        )


@dataclass(frozen=True)
class JointCostParameters:
    latency: int
    launch_overhead: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "latency",
            _coerce_non_negative_int(
                self.latency, field_name="JointCostParameters.latency"
            ),
        )
        object.__setattr__(
            self,
            "launch_overhead",
            _coerce_non_negative_int(
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
    def from_json(
        cls, payload: object, *, field_name: str = "JointCostParameters"
    ) -> "JointCostParameters":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            latency=_require_field(mapping, "latency", owner=field_name),
            launch_overhead=_require_field(mapping, "launch_overhead", owner=field_name),
        )


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
    def from_json(
        cls, payload: object, *, field_name: str = "JointCompatibleRecipePair"
    ) -> "JointCompatibleRecipePair":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            src_recipe_id=_require_field(mapping, "src_recipe_id", owner=field_name),
            dst_recipe_id=_require_field(mapping, "dst_recipe_id", owner=field_name),
        )


@dataclass(frozen=True)
class JointValueProducer:
    action_id: str

    def to_json(self) -> dict[str, object]:
        return {"action_id": self.action_id}

    @classmethod
    def from_json(
        cls, payload: object, *, field_name: str = "JointValueProducer"
    ) -> "JointValueProducer":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(action_id=_require_field(mapping, "action_id", owner=field_name))


@dataclass(frozen=True)
class JointValueConsumer:
    action_id: str

    def to_json(self) -> dict[str, object]:
        return {"action_id": self.action_id}

    @classmethod
    def from_json(
        cls, payload: object, *, field_name: str = "JointValueConsumer"
    ) -> "JointValueConsumer":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(action_id=_require_field(mapping, "action_id", owner=field_name))


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
    def from_json(
        cls, payload: object, *, field_name: str = "JointRegion"
    ) -> "JointRegion":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            region_id=_require_field(mapping, "region_id", owner=field_name),
            kind=_require_field(mapping, "kind", owner=field_name),
            member_nodes=mapping.get("member_nodes", ()),
            input_value_ids=_require_field(mapping, "input_value_ids", owner=field_name),
            output_value_ids=_require_field(
                mapping, "output_value_ids", owner=field_name
            ),
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
        object.__setattr__(
            self,
            "activates_action_ids",
            _coerce_str_tuple(
                self.activates_action_ids,
                field_name="JointRecipe.activates_action_ids",
            ),
        )
        if not isinstance(self.value_footprint, JointValueFootprint):
            raise TypeError(
                "JointRecipe.value_footprint must be JointValueFootprint"
            )
        if not isinstance(self.cost_parameters, JointCostParameters):
            raise TypeError(
                "JointRecipe.cost_parameters must be JointCostParameters"
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
    def from_json(
        cls, payload: object, *, field_name: str = "JointRecipe"
    ) -> "JointRecipe":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            recipe_id=_require_field(mapping, "recipe_id", owner=field_name),
            region_id=_require_field(mapping, "region_id", owner=field_name),
            tile_spec=JointTileSpec.from_json(
                _require_field(mapping, "tile_spec", owner=field_name),
                field_name=f"{field_name}.tile_spec",
            ),
            layout_spec=JointLayoutSpec.from_json(
                _require_field(mapping, "layout_spec", owner=field_name),
                field_name=f"{field_name}.layout_spec",
            ),
            activates_action_ids=_require_field(
                mapping, "activates_action_ids", owner=field_name
            ),
            value_footprint=JointValueFootprint.from_json(
                _require_field(mapping, "value_footprint", owner=field_name),
                field_name=f"{field_name}.value_footprint",
            ),
            cost_parameters=JointCostParameters.from_json(
                _require_field(mapping, "cost_parameters", owner=field_name),
                field_name=f"{field_name}.cost_parameters",
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
            _coerce_non_negative_int(self.size_bytes, field_name="JointValue.size_bytes"),
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
        object.__setattr__(
            self,
            "must_keep",
            _coerce_bool(self.must_keep, field_name="JointValue.must_keep"),
        )
        object.__setattr__(
            self,
            "spillable",
            _coerce_bool(self.spillable, field_name="JointValue.spillable"),
        )
        object.__setattr__(
            self,
            "allows_multiple_sram_windows",
            _coerce_bool(
                self.allows_multiple_sram_windows,
                field_name="JointValue.allows_multiple_sram_windows",
            ),
        )
        if self.must_keep and self.spillable:
            raise ValueError("JointValue.must_keep cannot be true when spillable is true")

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
    def from_json(
        cls, payload: object, *, field_name: str = "JointValue"
    ) -> "JointValue":
        mapping = _require_mapping(payload, field_name=field_name)
        producer_payload = _require_field(mapping, "producer", owner=field_name)
        if producer_payload is None:
            producer = None
        else:
            producer = JointValueProducer.from_json(
                producer_payload, field_name=f"{field_name}.producer"
            )
        consumers_payload = _parse_object_array(
            mapping, "consumers", JointValueConsumer.from_json, owner=field_name
        )
        return cls(
            value_id=_require_field(mapping, "value_id", owner=field_name),
            size_bytes=_require_field(mapping, "size_bytes", owner=field_name),
            initial_tier=_require_field(mapping, "initial_tier", owner=field_name),
            required_final_tier=_require_field(
                mapping, "required_final_tier", owner=field_name
            ),
            must_keep=_require_field(mapping, "must_keep", owner=field_name),
            spillable=_require_field(mapping, "spillable", owner=field_name),
            allows_multiple_sram_windows=_require_field(
                mapping, "allows_multiple_sram_windows", owner=field_name
            ),
            producer=producer,
            consumers=consumers_payload,
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
            _coerce_non_negative_int(self.duration, field_name="JointAction.duration"),
        )
        object.__setattr__(
            self,
            "launch_overhead",
            _coerce_non_negative_int(
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
            _coerce_non_negative_int(
                self.temp_bytes, field_name="JointAction.temp_bytes"
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "region_id": self.region_id,
            "recipe_id": self.recipe_id,
            "kind": self.kind.value,
            "resource_kind": self.resource_kind.value,
            "duration": self.duration,
            "launch_overhead": self.launch_overhead,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "temp_bytes": self.temp_bytes,
            "is_optional": self.is_optional,
            "optional_value_id": self.optional_value_id,
        }

    @classmethod
    def from_json(
        cls, payload: object, *, field_name: str = "JointAction"
    ) -> "JointAction":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            action_id=_require_field(mapping, "action_id", owner=field_name),
            region_id=_require_field(mapping, "region_id", owner=field_name),
            recipe_id=_require_field(mapping, "recipe_id", owner=field_name),
            kind=_require_field(mapping, "kind", owner=field_name),
            resource_kind=_require_field(mapping, "resource_kind", owner=field_name),
            duration=_require_field(mapping, "duration", owner=field_name),
            launch_overhead=_require_field(
                mapping, "launch_overhead", owner=field_name
            ),
            reads=_require_field(mapping, "reads", owner=field_name),
            writes=_require_field(mapping, "writes", owner=field_name),
            temp_bytes=_require_field(mapping, "temp_bytes", owner=field_name),
            is_optional=_require_field(mapping, "is_optional", owner=field_name),
            optional_value_id=_require_field(
                mapping, "optional_value_id", owner=field_name
            ),
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
    def from_json(
        cls, payload: object, *, field_name: str = "JointBoundaryConstraint"
    ) -> "JointBoundaryConstraint":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            boundary_id=_require_field(mapping, "boundary_id", owner=field_name),
            src_region_id=_require_field(mapping, "src_region_id", owner=field_name),
            dst_region_id=_require_field(mapping, "dst_region_id", owner=field_name),
            compatible_recipe_pairs=_parse_object_array(
                mapping,
                "compatible_recipe_pairs",
                JointCompatibleRecipePair.from_json,
                owner=field_name,
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
    def from_json(
        cls, payload: object, *, field_name: str = "JointDependencyEdge"
    ) -> "JointDependencyEdge":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            src_action_id=_require_field(mapping, "src_action_id", owner=field_name),
            dst_action_id=_require_field(mapping, "dst_action_id", owner=field_name),
            kind=_require_field(mapping, "kind", owner=field_name),
        )


@dataclass(frozen=True)
class JointResource:
    resource_kind: JointResourceKind
    slot_count: int = 1

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
        slot_count = _coerce_non_negative_int(
            self.slot_count, field_name="JointResource.slot_count"
        )
        if slot_count != 1:
            raise ValueError("JointResource.slot_count must be 1 in v1")
        object.__setattr__(self, "slot_count", slot_count)

    def to_json(self) -> dict[str, object]:
        return {
            "resource_kind": self.resource_kind.value,
            "slot_count": self.slot_count,
        }

    @classmethod
    def from_json(
        cls, payload: object, *, field_name: str = "JointResource"
    ) -> "JointResource":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            resource_kind=_require_field(mapping, "resource_kind", owner=field_name),
            slot_count=_require_field(mapping, "slot_count", owner=field_name),
        )


@dataclass(frozen=True)
class JointSramItem:
    item_id: str
    kind: JointSramItemKind
    size_bytes: int
    alignment_bytes: int
    is_optional: bool
    owner_action_id: str | None
    owner_value_id: str | None
    owner_residency_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _coerce_enum(self.kind, JointSramItemKind, field_name="JointSramItem.kind"),
        )
        object.__setattr__(
            self,
            "size_bytes",
            _coerce_non_negative_int(
                self.size_bytes, field_name="JointSramItem.size_bytes"
            ),
        )
        object.__setattr__(
            self,
            "alignment_bytes",
            _coerce_positive_int(
                self.alignment_bytes, field_name="JointSramItem.alignment_bytes"
            ),
        )
        object.__setattr__(
            self,
            "is_optional",
            _coerce_bool(self.is_optional, field_name="JointSramItem.is_optional"),
        )
        object.__setattr__(
            self,
            "owner_action_id",
            _coerce_optional_str(
                self.owner_action_id, field_name="JointSramItem.owner_action_id"
            ),
        )
        object.__setattr__(
            self,
            "owner_value_id",
            _coerce_optional_str(
                self.owner_value_id, field_name="JointSramItem.owner_value_id"
            ),
        )
        object.__setattr__(
            self,
            "owner_residency_id",
            _coerce_optional_str(
                self.owner_residency_id,
                field_name="JointSramItem.owner_residency_id",
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "kind": self.kind.value,
            "size_bytes": self.size_bytes,
            "alignment_bytes": self.alignment_bytes,
            "is_optional": self.is_optional,
            "owner_action_id": self.owner_action_id,
            "owner_value_id": self.owner_value_id,
            "owner_residency_id": self.owner_residency_id,
        }

    @classmethod
    def from_json(
        cls, payload: object, *, field_name: str = "JointSramItem"
    ) -> "JointSramItem":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            item_id=_require_field(mapping, "item_id", owner=field_name),
            kind=_require_field(mapping, "kind", owner=field_name),
            size_bytes=_require_field(mapping, "size_bytes", owner=field_name),
            alignment_bytes=_require_field(
                mapping, "alignment_bytes", owner=field_name
            ),
            is_optional=_require_field(mapping, "is_optional", owner=field_name),
            owner_action_id=_require_field(
                mapping, "owner_action_id", owner=field_name
            ),
            owner_value_id=_require_field(mapping, "owner_value_id", owner=field_name),
            owner_residency_id=_require_field(
                mapping, "owner_residency_id", owner=field_name
            ),
        )


@dataclass(frozen=True)
class JointSramAllocation:
    item_id: str
    offset: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "offset",
            _coerce_non_negative_int(
                self.offset, field_name="JointSramAllocation.offset"
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {"item_id": self.item_id, "offset": self.offset}

    @classmethod
    def from_json(
        cls, payload: object, *, field_name: str = "JointSramAllocation"
    ) -> "JointSramAllocation":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            item_id=_require_field(mapping, "item_id", owner=field_name),
            offset=_require_field(mapping, "offset", owner=field_name),
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
    sram_items: tuple[JointSramItem, ...] = ()
    default_alignment_bytes: int = 1
    objective: str = JOINT_TILING_SCHEDULE_OBJECTIVE

    def __post_init__(self) -> None:
        if self.schema_version != JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION:
            raise ValueError("JointProblem.schema_version must be joint_tiling_schedule_problem_v1")
        object.__setattr__(
            self, "regions", _coerce_tuple_of_type(self.regions, JointRegion, field_name="JointProblem.regions")
        )
        object.__setattr__(
            self, "recipes", _coerce_tuple_of_type(self.recipes, JointRecipe, field_name="JointProblem.recipes")
        )
        object.__setattr__(
            self, "values", _coerce_tuple_of_type(self.values, JointValue, field_name="JointProblem.values")
        )
        object.__setattr__(
            self, "actions", _coerce_tuple_of_type(self.actions, JointAction, field_name="JointProblem.actions")
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
                self.resources, JointResource, field_name="JointProblem.resources"
            ),
        )
        object.__setattr__(
            self,
            "sram_capacity_bytes",
            _coerce_non_negative_int(
                self.sram_capacity_bytes,
                field_name="JointProblem.sram_capacity_bytes",
            ),
        )
        object.__setattr__(
            self,
            "sram_items",
            _coerce_tuple_of_type(
                self.sram_items,
                JointSramItem,
                field_name="JointProblem.sram_items",
            ),
        )
        object.__setattr__(
            self,
            "default_alignment_bytes",
            _coerce_positive_int(
                self.default_alignment_bytes,
                field_name="JointProblem.default_alignment_bytes",
            ),
        )
        if self.objective != JOINT_TILING_SCHEDULE_OBJECTIVE:
            raise ValueError("JointProblem.objective must be min_makespan")
        _ensure_unique_ids(self.regions, "region_id")
        _ensure_unique_ids(self.recipes, "recipe_id")
        _ensure_unique_ids(self.values, "value_id")
        _ensure_unique_ids(self.actions, "action_id")
        _ensure_unique_ids(self.boundary_constraints, "boundary_id")
        _ensure_unique_ids(self.resources, "resource_kind")
        _ensure_unique_ids(self.sram_items, "item_id")

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
            "sram_items": [item.to_json() for item in self.sram_items],
            "default_alignment_bytes": self.default_alignment_bytes,
            "objective": self.objective,
        }

    @classmethod
    def from_json(cls, payload: object) -> "JointProblem":
        mapping = _require_mapping(payload, field_name="JointProblem")
        return cls(
            schema_version=_require_field(
                mapping, "schema_version", owner="JointProblem"
            ),
            regions=_parse_object_array(
                mapping, "regions", JointRegion.from_json, owner="JointProblem"
            ),
            recipes=_parse_object_array(
                mapping, "recipes", JointRecipe.from_json, owner="JointProblem"
            ),
            values=_parse_object_array(
                mapping, "values", JointValue.from_json, owner="JointProblem"
            ),
            actions=_parse_object_array(
                mapping, "actions", JointAction.from_json, owner="JointProblem"
            ),
            boundary_constraints=_parse_object_array(
                mapping,
                "boundary_constraints",
                JointBoundaryConstraint.from_json,
                owner="JointProblem",
            ),
            dependency_edges=_parse_object_array(
                mapping,
                "dependency_edges",
                JointDependencyEdge.from_json,
                owner="JointProblem",
            ),
            resources=_parse_object_array(
                mapping, "resources", JointResource.from_json, owner="JointProblem"
            ),
            sram_capacity_bytes=_require_field(
                mapping, "sram_capacity_bytes", owner="JointProblem"
            ),
            sram_items=_parse_object_array(
                mapping, "sram_items", JointSramItem.from_json, owner="JointProblem"
            ),
            default_alignment_bytes=_require_field(
                mapping, "default_alignment_bytes", owner="JointProblem"
            ),
            objective=_require_field(mapping, "objective", owner="JointProblem"),
        )


@dataclass(frozen=True)
class JointSelectedRecipe:
    region_id: str
    recipe_id: str

    def to_json(self) -> dict[str, object]:
        return {"region_id": self.region_id, "recipe_id": self.recipe_id}

    @classmethod
    def from_json(
        cls, payload: object, *, field_name: str = "JointSelectedRecipe"
    ) -> "JointSelectedRecipe":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            region_id=_require_field(mapping, "region_id", owner=field_name),
            recipe_id=_require_field(mapping, "recipe_id", owner=field_name),
        )


@dataclass(frozen=True)
class JointScheduledAction:
    action_id: str
    start_time: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_time",
            _coerce_non_negative_int(
                self.start_time, field_name="JointScheduledAction.start_time"
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {"action_id": self.action_id, "start_time": self.start_time}

    @classmethod
    def from_json(
        cls, payload: object, *, field_name: str = "JointScheduledAction"
    ) -> "JointScheduledAction":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            action_id=_require_field(mapping, "action_id", owner=field_name),
            start_time=_require_field(mapping, "start_time", owner=field_name),
        )


@dataclass(frozen=True)
class JointResidencyWindow:
    value_id: str
    start_time: int
    end_time: int
    residency_id: str | None = None

    def __post_init__(self) -> None:
        start_time = _coerce_non_negative_int(
            self.start_time, field_name="JointResidencyWindow.start_time"
        )
        end_time = _coerce_non_negative_int(
            self.end_time, field_name="JointResidencyWindow.end_time"
        )
        if end_time <= start_time:
            raise ValueError("JointResidencyWindow.end_time must be > start_time")
        residency_id = _coerce_optional_str(
            self.residency_id, field_name="JointResidencyWindow.residency_id"
        )
        if residency_id is None:
            raise ValueError("JointResidencyWindow.residency_id is required")
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "end_time", end_time)
        object.__setattr__(self, "residency_id", residency_id)

    def to_json(self) -> dict[str, object]:
        return {
            "residency_id": self.residency_id,
            "value_id": self.value_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_json(
        cls, payload: object, *, field_name: str = "JointResidencyWindow"
    ) -> "JointResidencyWindow":
        mapping = _require_mapping(payload, field_name=field_name)
        return cls(
            residency_id=_require_field(mapping, "residency_id", owner=field_name),
            value_id=_require_field(mapping, "value_id", owner=field_name),
            start_time=_require_field(mapping, "start_time", owner=field_name),
            end_time=_require_field(mapping, "end_time", owner=field_name),
        )


@dataclass(frozen=True)
class JointSolution:
    schema_version: str
    selected_recipes: tuple[JointSelectedRecipe, ...]
    scheduled_actions: tuple[JointScheduledAction, ...]
    residency_windows: tuple[JointResidencyWindow, ...]
    objective_value: int
    generated_sram_items: tuple[JointSramItem, ...] = ()
    sram_allocations: tuple[JointSramAllocation, ...] = ()
    diagnostics: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION:
            raise ValueError("JointSolution.schema_version must be joint_tiling_schedule_solution_v1")
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
            _coerce_non_negative_int(
                self.objective_value, field_name="JointSolution.objective_value"
            ),
        )
        object.__setattr__(
            self,
            "generated_sram_items",
            _coerce_tuple_of_type(
                self.generated_sram_items,
                JointSramItem,
                field_name="JointSolution.generated_sram_items",
            ),
        )
        object.__setattr__(
            self,
            "sram_allocations",
            _coerce_tuple_of_type(
                self.sram_allocations,
                JointSramAllocation,
                field_name="JointSolution.sram_allocations",
            ),
        )
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_json_mapping(self.diagnostics, field_name="JointSolution.diagnostics"),
        )
        _ensure_unique_ids(self.selected_recipes, "region_id")
        _ensure_unique_ids(self.scheduled_actions, "action_id")
        _ensure_unique_ids(self.residency_windows, "residency_id")
        _ensure_unique_ids(self.generated_sram_items, "item_id")
        _ensure_unique_ids(self.sram_allocations, "item_id")

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selected_recipes": [
                selected_recipe.to_json()
                for selected_recipe in self.selected_recipes
            ],
            "scheduled_actions": [
                scheduled_action.to_json()
                for scheduled_action in self.scheduled_actions
            ],
            "residency_windows": [
                residency_window.to_json()
                for residency_window in self.residency_windows
            ],
            "objective_value": self.objective_value,
            "generated_sram_items": [
                item.to_json() for item in self.generated_sram_items
            ],
            "sram_allocations": [
                allocation.to_json() for allocation in self.sram_allocations
            ],
            "diagnostics": _to_json_value(self.diagnostics),
        }

    @classmethod
    def from_json(cls, payload: object) -> "JointSolution":
        mapping = _require_mapping(payload, field_name="JointSolution")
        return cls(
            schema_version=_require_field(
                mapping, "schema_version", owner="JointSolution"
            ),
            selected_recipes=_parse_object_array(
                mapping,
                "selected_recipes",
                JointSelectedRecipe.from_json,
                owner="JointSolution",
            ),
            scheduled_actions=_parse_object_array(
                mapping,
                "scheduled_actions",
                JointScheduledAction.from_json,
                owner="JointSolution",
            ),
            residency_windows=_parse_object_array(
                mapping,
                "residency_windows",
                JointResidencyWindow.from_json,
                owner="JointSolution",
            ),
            objective_value=_require_field(
                mapping, "objective_value", owner="JointSolution"
            ),
            generated_sram_items=_parse_object_array(
                mapping,
                "generated_sram_items",
                JointSramItem.from_json,
                owner="JointSolution",
            ),
            sram_allocations=_parse_object_array(
                mapping,
                "sram_allocations",
                JointSramAllocation.from_json,
                owner="JointSolution",
            ),
            diagnostics=_require_field(mapping, "diagnostics", owner="JointSolution"),
        )


@dataclass(frozen=True)
class JointFailure:
    schema_version: str
    status: JointFailureStatus
    error_category: JointFailureCategory
    diagnostics: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION:
            raise ValueError("JointFailure.schema_version must be joint_tiling_schedule_failure_v1")
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
    def from_json(cls, payload: object) -> "JointFailure":
        mapping = _require_mapping(payload, field_name="JointFailure")
        return cls(
            schema_version=_require_field(
                mapping, "schema_version", owner="JointFailure"
            ),
            status=_require_field(mapping, "status", owner="JointFailure"),
            error_category=_require_field(
                mapping, "error_category", owner="JointFailure"
            ),
            diagnostics=_require_field(mapping, "diagnostics", owner="JointFailure"),
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
    "JsonScalar",
    "JsonValue",
    "get_joint_tiling_schedule_failure",
    "get_joint_tiling_schedule_problem",
    "get_joint_tiling_schedule_solution",
    "set_joint_tiling_schedule_failure",
    "set_joint_tiling_schedule_problem",
    "set_joint_tiling_schedule_solution",
]
