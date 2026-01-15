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

    def _before_pass(self, ctx: CompileContext):
        """Hook called before pass execution."""
        pass

    @abstractmethod
    def _execute(self, ctx: CompileContext) -> None:
        """Core pass logic to be implemented by subclasses."""
        pass

    def _after_pass(self, ctx: CompileContext):
        """Hook called after pass execution."""
        pass


class PassManager:
    """Pass manager for running optimization passes."""

    def __init__(self):
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
        from nnc_py.passes.constant_folding import ConstantFoldingPass
        from nnc_py.passes.liveness import LivenessAnalysisPass
        from nnc_py.passes.memory_plan import MemoryPlanningPass
        from nnc_py.passes.spill import SpillAnalysisPass

        passes = []

        # O0: No optimization
        if opt_level == 0:
            return passes

        # O1: Basic optimizations
        if opt_level >= 1:
            passes.append(ConstantFoldingPass())

        # O2: Intermediate optimizations (with memory planning)
        if opt_level >= 2:
            passes.append(LivenessAnalysisPass())
            passes.append(MemoryPlanningPass())
            passes.append(SpillAnalysisPass())  # Handles overflow if max_memory set

        # O3: Advanced optimizations
        if opt_level >= 3:
            # TODO: Add advanced passes (operator fusion, etc.)
            pass

        return passes
