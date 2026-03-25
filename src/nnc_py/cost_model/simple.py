"""Deterministic fallback cost model."""

from __future__ import annotations

import math

from nnc_py.cost_model.base import AttrMapping, CostEstimate, CostModelProvider, ShapeSeq
from nnc_py.ir.pipeline_schedule import PipelineResourceKind, ScheduleStepKind


class SimpleCostModelProvider(CostModelProvider):
    """Small heuristic provider used when no external model is available."""

    DMA_LAUNCH = 12
    SHAPE_LAUNCH = 8
    MATMUL_LAUNCH = 16
    OTHER_LAUNCH = 10

    DMA_BW = 32
    SHAPE_TPUT = 64
    MATMUL_TPUT = 128
    OTHER_TPUT = 64

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
        del op_type, step_kind, dtypes

        attrs = attrs or {}
        resource = PipelineResourceKind(resource_kind)

        if resource is PipelineResourceKind.DMA:
            launch = self.DMA_LAUNCH
            work_units = self._ceil_div(max(tensor_bytes, 0), self.DMA_BW)
            metric = "bytes"
        elif resource is PipelineResourceKind.SHAPE:
            launch = self.SHAPE_LAUNCH
            work = self._infer_elements(input_shapes, output_shapes)
            work_units = self._ceil_div(work, self.SHAPE_TPUT)
            metric = "elements"
        elif resource is PipelineResourceKind.MATMUL:
            launch = self.MATMUL_LAUNCH
            work = self._read_positive_int(attrs, "macs")
            if work is not None:
                metric = "macs"
            else:
                work = self._infer_elements(input_shapes, output_shapes)
                metric = "inferred_work"
            work_units = self._ceil_div(work, self.MATMUL_TPUT)
        else:
            launch = self.OTHER_LAUNCH
            work = self._read_positive_int(attrs, "work")
            if work is not None:
                metric = "work"
            else:
                work = self._infer_elements(input_shapes, output_shapes)
                metric = "inferred_work"
            work_units = self._ceil_div(work, self.OTHER_TPUT)

        latency = launch + work_units
        return CostEstimate(
            latency=latency,
            launch_overhead=launch,
            source="simple",
            breakdown={
                "launch_overhead": launch,
                "work_units": work_units,
                metric: max(tensor_bytes, 0) if resource is PipelineResourceKind.DMA else work,
            },
        )

    @staticmethod
    def _ceil_div(numerator: int, denominator: int) -> int:
        if numerator <= 0:
            return 0
        return int(math.ceil(numerator / denominator))

    @staticmethod
    def _read_positive_int(attrs: AttrMapping, key: str) -> int | None:
        value = attrs.get(key)
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value > 0:
            return value
        return None

    @classmethod
    def _infer_elements(
        cls, input_shapes: ShapeSeq, output_shapes: ShapeSeq
    ) -> int:
        shapes = output_shapes or input_shapes
        total = sum(cls._shape_elements(shape) for shape in shapes)
        return max(total, 1)

    @staticmethod
    def _shape_elements(shape: tuple[int, ...]) -> int:
        if not shape:
            return 1
        total = 1
        for dim in shape:
            total *= max(int(dim), 0)
        return total
