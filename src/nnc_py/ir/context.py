"""Compilation context for passing information through the pipeline."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nnc_py.ir.graph import Graph

if TYPE_CHECKING:
    from nnc_py.ir.execution_plan import NodeExecutionPlan
    from nnc_py.ir.pipeline_schedule import (
        PipelineScheduleProblem,
        PipelineScheduleResult,
    )


@dataclass
class CompileContext:
    """Compilation context - passes information through the compilation pipeline."""

    graph: Graph
    target: str  # "x86" or "npu"
    optimization_level: int = 0
    debug: bool = False

    # Symbol tables
    tensor_symbols: dict[str, str] = field(default_factory=dict)  # ONNX -> C
    node_symbols: dict[str, str] = field(default_factory=dict)  # ONNX -> C

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if self.tensor_symbols is None:
            self.tensor_symbols = {}
        if self.node_symbols is None:
            self.node_symbols = {}
        if self.metadata is None:
            self.metadata = {}

    @property
    def node_execution_plans(self) -> dict[str, "NodeExecutionPlan"]:
        """Read-only typed access to node execution plans stored in metadata."""

        from nnc_py.ir.execution_plan import get_node_execution_plans

        return get_node_execution_plans(self)

    def get_node_execution_plan(self, node_name: str) -> "NodeExecutionPlan | None":
        """Return a single node execution plan by name without mutating metadata."""

        from nnc_py.ir.execution_plan import get_node_execution_plan

        return get_node_execution_plan(self, node_name)

    @property
    def pipeline_schedule_problem(self) -> "PipelineScheduleProblem | None":
        """Read-only typed access to the pipeline schedule problem in metadata."""

        from nnc_py.ir.pipeline_schedule import get_pipeline_schedule_problem

        return get_pipeline_schedule_problem(self)

    def get_pipeline_schedule_problem(self) -> "PipelineScheduleProblem | None":
        """Return the pipeline schedule problem without mutating metadata."""

        from nnc_py.ir.pipeline_schedule import get_pipeline_schedule_problem

        return get_pipeline_schedule_problem(self)

    @property
    def pipeline_schedule_result(self) -> "PipelineScheduleResult | None":
        """Read-only typed access to the pipeline schedule result in metadata."""

        from nnc_py.ir.pipeline_schedule import get_pipeline_schedule_result

        return get_pipeline_schedule_result(self)

    def get_pipeline_schedule_result(self) -> "PipelineScheduleResult | None":
        """Return the pipeline schedule result without mutating metadata."""

        from nnc_py.ir.pipeline_schedule import get_pipeline_schedule_result

        return get_pipeline_schedule_result(self)
