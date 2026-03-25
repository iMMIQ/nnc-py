"""Base interfaces for schedule-step cost estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from nnc_py.ir.pipeline_schedule import PipelineResourceKind, ScheduleStepKind


Shape = tuple[int, ...]
ShapeSeq = tuple[Shape, ...]
AttrMapping = Mapping[str, object]


def _freeze_breakdown(value: Mapping[str, int]) -> Mapping[str, int]:
    frozen_items: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("CostEstimate.breakdown keys must be strings")
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError("CostEstimate.breakdown values must be integers")
        frozen_items[key] = item
    return MappingProxyType(frozen_items)


@dataclass(frozen=True)
class CostEstimate:
    """Estimated latency for a scheduled step."""

    latency: int
    launch_overhead: int
    source: str
    breakdown: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "breakdown", _freeze_breakdown(self.breakdown))


class CostModelProvider(ABC):
    """Abstract provider for step-level latency estimates."""

    @abstractmethod
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
        """Estimate the latency of a schedule step."""
        raise NotImplementedError
