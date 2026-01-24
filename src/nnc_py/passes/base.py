"""Base classes for optimization passes."""

from abc import ABC, abstractmethod
from typing import List

from nnc_py.ir.context import CompileContext


class PassBase(ABC):
    """Base class for optimization passes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the pass name."""
        pass

    def run(self, ctx: CompileContext) -> None:
        """Execute the pass.

        Args:
            ctx: Compilation context to transform.
        """
        self._before_pass(ctx)
        self._execute(ctx)
        self._after_pass(ctx)

    def _before_pass(self, ctx: CompileContext) -> None:
        """Hook called before pass execution."""
        pass

    @abstractmethod
    def _execute(self, ctx: CompileContext) -> None:
        """Core pass logic to be implemented by subclasses."""
        pass

    def _after_pass(self, ctx: CompileContext) -> None:
        """Hook called after pass execution."""
        pass


class PassManager:
    """Pass manager for running optimization passes."""

    def __init__(self) -> None:
        self.passes: List[PassBase] = []
        self.applied_passes: List[str] = []

    def register(self, pass_obj: PassBase) -> None:
        """Register a pass.

        Args:
            pass_obj: Pass to register.
        """
        self.passes.append(pass_obj)

    def run(self, ctx: CompileContext) -> None:
        """Run all registered passes.

        Args:
            ctx: Compilation context to transform.
        """
        for pass_obj in self.passes:
            pass_obj.run(ctx)
            self.applied_passes.append(pass_obj.name)

    @classmethod
    def get_default_passes(cls, opt_level: int) -> List[PassBase]:
        """Get default pass sequence for optimization level.

        Args:
            opt_level: Optimization level (0-3).

        Returns:
            List of passes to run.
        """
        from nnc_py.passes.liveness import LivenessAnalysisPass
        from nnc_py.passes.memory_planning import MemoryPlanningPassV2
        from nnc_py.passes.spill import SpillAnalysisPass

        # O0: Essential passes only (liveness + memory planning)
        # These are needed for code generation even without optimization
        # Note: constant folding is now handled by onnxsim in the frontend
        if opt_level == 0:
            return [LivenessAnalysisPass(), MemoryPlanningPassV2()]

        # O1: Basic optimizations
        # Note: constant folding is now handled by onnxsim in the frontend
        if opt_level == 1:
            return [
                LivenessAnalysisPass(),
                MemoryPlanningPassV2(),
            ]

        # O2: Intermediate optimizations
        if opt_level == 2:
            return [
                LivenessAnalysisPass(),
                MemoryPlanningPassV2(),
                SpillAnalysisPass(),  # Handles overflow if max_memory set
            ]

        # O3: Advanced optimizations
        if opt_level >= 3:
            # TODO: Add advanced passes (operator fusion, etc.)
            return [
                LivenessAnalysisPass(),
                MemoryPlanningPassV2(),
                SpillAnalysisPass(),
            ]

        return []
