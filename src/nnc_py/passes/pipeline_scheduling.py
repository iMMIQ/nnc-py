"""Pipeline scheduling pass wrapper."""

from __future__ import annotations

from nnc_py.ir.context import CompileContext
from nnc_py.ir.pipeline_schedule import set_pipeline_schedule_result
from nnc_py.passes.base import PassBase
from nnc_py.scheduler import ListPipelineScheduler, PipelineScheduler


class PipelineSchedulingPass(PassBase):
    """Run a pipeline scheduler on the lowered scheduling problem."""

    def __init__(self, scheduler: PipelineScheduler | None = None) -> None:
        self._scheduler = scheduler if scheduler is not None else ListPipelineScheduler()

    @property
    def name(self) -> str:
        return "PipelineScheduling"

    def _execute(self, ctx: CompileContext) -> None:
        problem = ctx.pipeline_schedule_problem
        if problem is None:
            return
        result = self._scheduler.solve(problem)
        set_pipeline_schedule_result(ctx, result)
        if (
            ctx.optimization_level >= 3
            and bool(ctx.metadata.get("pipeline_scheduler_enabled"))
            and not result.feasible
        ):
            raise RuntimeError(_format_strict_o3_schedule_error(result.diagnostics))


def _format_strict_o3_schedule_error(diagnostics: dict[str, object]) -> str:
    reason = diagnostics.get("reason")
    detail_parts: list[str] = []
    if isinstance(reason, str) and reason:
        detail_parts.append(reason)
    for key in sorted(diagnostics):
        if key == "reason":
            continue
        detail_parts.append(f"{key}={diagnostics[key]}")
    detail = ", ".join(detail_parts) if detail_parts else "infeasible_schedule_result"
    return f"O3 scheduled pipeline path failed ({detail})."
