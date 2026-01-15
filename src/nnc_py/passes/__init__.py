"""Optimization passes module."""

from nnc_py.passes.base import PassBase, PassManager
from nnc_py.passes.constant_folding import ConstantFoldingPass
from nnc_py.passes.unified_memory import UnifiedMemoryPass, get_unified_memory_plan, UnifiedMemoryPlan

__all__ = [
    "PassBase",
    "PassManager",
    "ConstantFoldingPass",
    "UnifiedMemoryPass",
    "get_unified_memory_plan",
    "UnifiedMemoryPlan",
]
