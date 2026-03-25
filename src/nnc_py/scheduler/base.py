"""Base interfaces for pipeline scheduling solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from nnc_py.ir.pipeline_schedule import PipelineScheduleProblem, PipelineScheduleResult


class PipelineScheduler(ABC):
    """Abstract interface for pipeline scheduling solvers."""

    @abstractmethod
    def solve(self, problem: PipelineScheduleProblem) -> PipelineScheduleResult:
        """Return a schedule result for the given scheduling problem."""
