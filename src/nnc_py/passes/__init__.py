"""Optimization passes module."""

from nnc_py.passes.base import PassBase, PassManager
from nnc_py.passes.constant_folding import ConstantFoldingPass

__all__ = ["PassBase", "PassManager", "ConstantFoldingPass"]
