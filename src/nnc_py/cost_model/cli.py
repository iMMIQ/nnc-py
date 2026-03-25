"""External CLI-backed cost model with safe fallback behavior."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from enum import Enum
from types import MappingProxyType

from nnc_py.cost_model.base import AttrMapping, CostEstimate, CostModelProvider, ShapeSeq
from nnc_py.cost_model.simple import SimpleCostModelProvider
from nnc_py.ir.pipeline_schedule import PipelineResourceKind, ScheduleStepKind


class CliCostModelProvider(CostModelProvider):
    """Provider that asks an external CLI for estimates and caches results."""

    def __init__(
        self,
        command: list[str] | None = None,
        fallback: CostModelProvider | None = None,
        *,
        timeout_seconds: float = 1.0,
    ) -> None:
        self.command = tuple(command or ())
        self.fallback = fallback or SimpleCostModelProvider()
        self.timeout_seconds = max(float(timeout_seconds), 0.001)
        self._cache: dict[tuple[object, ...], CostEstimate] = {}

    def estimate_step(
        self,
        *,
        op_type: str,
        step_kind: ScheduleStepKind,
        resource_kind: PipelineResourceKind,
        input_shapes: ShapeSeq,
        output_shapes: ShapeSeq,
        dtypes: tuple[str, ...],
        tensor_bytes: int,
        attrs: AttrMapping | None = None,
    ) -> CostEstimate:
        normalized_attrs = self._normalize_attrs(attrs)
        cache_key = self._make_cache_key(
            op_type=op_type,
            step_kind=step_kind,
            resource_kind=resource_kind,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            dtypes=dtypes,
            tensor_bytes=tensor_bytes,
            attrs=normalized_attrs,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        estimate = self._estimate_cli(
            op_type=op_type,
            step_kind=step_kind,
            resource_kind=resource_kind,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            dtypes=dtypes,
            tensor_bytes=tensor_bytes,
            attrs=normalized_attrs,
        )
        if estimate is not None:
            self._cache[cache_key] = estimate
            return estimate
        return self._fallback_estimate(
            op_type=op_type,
            step_kind=step_kind,
            resource_kind=resource_kind,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            dtypes=dtypes,
            tensor_bytes=tensor_bytes,
            attrs=normalized_attrs,
        )

    def _estimate_cli(
        self,
        *,
        op_type: str,
        step_kind: ScheduleStepKind,
        resource_kind: PipelineResourceKind,
        input_shapes: ShapeSeq,
        output_shapes: ShapeSeq,
        dtypes: tuple[str, ...],
        tensor_bytes: int,
        attrs: AttrMapping,
    ) -> CostEstimate | None:
        if not self.command:
            return None

        payload = {
            "op_type": op_type,
            "step_kind": ScheduleStepKind(step_kind).value,
            "resource_kind": PipelineResourceKind(resource_kind).value,
            "input_shapes": [list(shape) for shape in input_shapes],
            "output_shapes": [list(shape) for shape in output_shapes],
            "dtypes": list(dtypes),
            "tensor_bytes": int(tensor_bytes),
            "attrs": self._to_json_value(attrs),
        }

        try:
            result = subprocess.run(
                list(self.command),
                input=json.dumps(payload, sort_keys=True),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (FileNotFoundError, OSError, TypeError, subprocess.TimeoutExpired):
            return None

        if result.returncode != 0:
            return None

        try:
            payload = json.loads(result.stdout)
            return self._parse_cli_estimate(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def _fallback_estimate(
        self,
        *,
        op_type: str,
        step_kind: ScheduleStepKind,
        resource_kind: PipelineResourceKind,
        input_shapes: ShapeSeq,
        output_shapes: ShapeSeq,
        dtypes: tuple[str, ...],
        tensor_bytes: int,
        attrs: AttrMapping,
    ) -> CostEstimate:
        return self.fallback.estimate_step(
            op_type=op_type,
            step_kind=step_kind,
            resource_kind=resource_kind,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            dtypes=dtypes,
            tensor_bytes=tensor_bytes,
            attrs=attrs,
        )

    @staticmethod
    def _parse_cli_estimate(payload: object) -> CostEstimate:
        if not isinstance(payload, Mapping):
            raise ValueError("CLI cost model must return a JSON object")

        latency = CliCostModelProvider._read_positive_int(payload, "latency")
        launch_overhead = CliCostModelProvider._read_positive_int(
            payload, "launch_overhead"
        )
        raw_breakdown = payload.get("breakdown", {})
        if not isinstance(raw_breakdown, Mapping):
            raise ValueError("CLI breakdown must be a mapping")

        breakdown: dict[str, int] = {}
        for key, value in raw_breakdown.items():
            if not isinstance(key, str):
                raise ValueError("CLI breakdown keys must be strings")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("CLI breakdown values must be non-negative integers")
            breakdown[key] = value

        return CostEstimate(
            latency=latency,
            launch_overhead=launch_overhead,
            source="cli",
            breakdown=breakdown,
        )

    @staticmethod
    def _read_positive_int(mapping: Mapping[object, object], key: str) -> int:
        value = mapping.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        return value

    @classmethod
    def _make_cache_key(
        cls,
        *,
        op_type: str,
        step_kind: ScheduleStepKind,
        resource_kind: PipelineResourceKind,
        input_shapes: ShapeSeq,
        output_shapes: ShapeSeq,
        dtypes: tuple[str, ...],
        tensor_bytes: int,
        attrs: dict[str, object],
    ) -> tuple[object, ...]:
        return (
            op_type,
            ScheduleStepKind(step_kind).value,
            PipelineResourceKind(resource_kind).value,
            tuple(tuple(shape) for shape in input_shapes),
            tuple(tuple(shape) for shape in output_shapes),
            tuple(dtypes),
            int(tensor_bytes),
            cls._freeze_value(attrs),
        )

    @classmethod
    def _normalize_attrs(cls, attrs: AttrMapping | None) -> AttrMapping:
        if attrs is None:
            return MappingProxyType({})
        if not isinstance(attrs, Mapping):
            raise TypeError("attrs must be a mapping")
        normalized: dict[str, object] = {}
        for key, value in attrs.items():
            if not isinstance(key, str):
                raise TypeError("attrs keys must be strings")
            normalized[key] = value
        return MappingProxyType(normalized)

    @classmethod
    def _freeze_value(cls, value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if value is None or isinstance(value, bool | int | float | str):
            return value
        if isinstance(value, Mapping):
            return tuple(
                (cls._validate_attr_key(key), cls._freeze_value(item))
                for key, item in sorted(value.items(), key=lambda entry: cls._validate_attr_key(entry[0]))
            )
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return tuple(cls._freeze_value(item) for item in value)
        raise TypeError(
            f"unsupported attr value of type {type(value).__name__}"
        )

    @classmethod
    def _to_json_value(cls, value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if value is None or isinstance(value, bool | int | float | str):
            return value
        if isinstance(value, Mapping):
            return {
                cls._validate_attr_key(key): cls._to_json_value(item)
                for key, item in sorted(value.items(), key=lambda entry: cls._validate_attr_key(entry[0]))
            }
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return [cls._to_json_value(item) for item in value]
        raise TypeError(
            f"unsupported attr value of type {type(value).__name__}"
        )

    @staticmethod
    def _validate_attr_key(key: object) -> str:
        if not isinstance(key, str):
            raise TypeError("attrs keys must be strings")
        return key
