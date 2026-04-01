"""ONNX frontend module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nnc_py.frontend.onnx_loader import ONNXFrontend as ONNXFrontend

__all__ = ["ONNXFrontend"]


def __getattr__(name: str) -> Any:
    if name != "ONNXFrontend":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from nnc_py.frontend.onnx_loader import ONNXFrontend

    return ONNXFrontend
