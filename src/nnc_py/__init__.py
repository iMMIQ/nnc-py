"""
NNC - Neural Network Compiler
ONNX to C compiler for edge inference devices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

if TYPE_CHECKING:
    from nnc_py.compiler import Compiler as Compiler

__all__ = ["Compiler"]


def __getattr__(name: str) -> Any:
    if name != "Compiler":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from nnc_py.compiler import Compiler

    return Compiler
