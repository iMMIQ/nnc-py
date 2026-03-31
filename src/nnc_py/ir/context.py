"""Compilation context for passing information through the pipeline."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nnc_py.ir.graph import Graph

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nnc_py.ir.execution_plan import NodeExecutionPlan
    from nnc_py.ir.joint_tiling_schedule import (
        JointFailure,
        JointProblem,
        JointSolution,
    )
    from nnc_py.ir.pipeline_schedule import (
        JsonValue,
        PipelineScheduleProblem,
        PipelineScheduleResult,
        ResidencyWindow,
        ScheduledValue,
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

    @property
    def pipeline_scheduled_values(self) -> "tuple[ScheduledValue, ...]":
        """Read-only typed access to scheduled values stored in metadata."""

        from nnc_py.ir.pipeline_schedule import get_pipeline_scheduled_values

        return get_pipeline_scheduled_values(self)

    def get_pipeline_scheduled_values(self) -> "tuple[ScheduledValue, ...]":
        """Return scheduled values without mutating metadata."""

        from nnc_py.ir.pipeline_schedule import get_pipeline_scheduled_values

        return get_pipeline_scheduled_values(self)

    @property
    def pipeline_residency_windows(self) -> "tuple[ResidencyWindow, ...]":
        """Read-only typed access to residency windows stored in metadata."""

        from nnc_py.ir.pipeline_schedule import get_pipeline_residency_windows

        return get_pipeline_residency_windows(self)

    def get_pipeline_residency_windows(self) -> "tuple[ResidencyWindow, ...]":
        """Return residency windows without mutating metadata."""

        from nnc_py.ir.pipeline_schedule import get_pipeline_residency_windows

        return get_pipeline_residency_windows(self)

    @property
    def pipeline_transfer_diagnostics(self) -> "Mapping[str, JsonValue]":
        """Read-only typed access to transfer diagnostics stored in metadata."""

        from nnc_py.ir.pipeline_schedule import get_pipeline_transfer_diagnostics

        return get_pipeline_transfer_diagnostics(self)

    def get_pipeline_transfer_diagnostics(self) -> "Mapping[str, JsonValue]":
        """Return transfer diagnostics without mutating metadata."""

        from nnc_py.ir.pipeline_schedule import get_pipeline_transfer_diagnostics

        return get_pipeline_transfer_diagnostics(self)

    @property
    def joint_tiling_schedule_problem(self) -> "JointProblem | None":
        """Read-only typed access to the joint tiling schedule problem metadata."""

        from nnc_py.ir.joint_tiling_schedule import get_joint_tiling_schedule_problem

        return get_joint_tiling_schedule_problem(self)

    def get_joint_tiling_schedule_problem(self) -> "JointProblem | None":
        """Return the joint tiling schedule problem without mutating metadata."""

        from nnc_py.ir.joint_tiling_schedule import get_joint_tiling_schedule_problem

        return get_joint_tiling_schedule_problem(self)

    @property
    def joint_tiling_schedule_solution(self) -> "JointSolution | None":
        """Read-only typed access to the joint tiling schedule solution metadata."""

        from nnc_py.ir.joint_tiling_schedule import get_joint_tiling_schedule_solution

        return get_joint_tiling_schedule_solution(self)

    def get_joint_tiling_schedule_solution(self) -> "JointSolution | None":
        """Return the joint tiling schedule solution without mutating metadata."""

        from nnc_py.ir.joint_tiling_schedule import get_joint_tiling_schedule_solution

        return get_joint_tiling_schedule_solution(self)

    @property
    def joint_tiling_schedule_failure(self) -> "JointFailure | None":
        """Read-only typed access to the joint tiling schedule failure metadata."""

        from nnc_py.ir.joint_tiling_schedule import get_joint_tiling_schedule_failure

        return get_joint_tiling_schedule_failure(self)

    def get_joint_tiling_schedule_failure(self) -> "JointFailure | None":
        """Return the joint tiling schedule failure without mutating metadata."""

        from nnc_py.ir.joint_tiling_schedule import get_joint_tiling_schedule_failure

        return get_joint_tiling_schedule_failure(self)
