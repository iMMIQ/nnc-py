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
        set_pipeline_schedule_result(ctx, self._scheduler.solve(problem))
