"""Tensor emitter for x86 codegen packages."""

from __future__ import annotations

from typing import Any

from nnc_py.codegen.x86_ir import X86CodegenPackage


def emit_tensors(package: X86CodegenPackage, backend: Any) -> str:
    """Emit tensor definitions from a lowered package."""
    return backend._generate_tensors(package.ctx)
