"""Model source emitter for x86 codegen packages."""

from __future__ import annotations

from typing import Any

from nnc_py.codegen.x86_ir import X86CodegenPackage


def emit_model_source(package: X86CodegenPackage, backend: Any) -> str:
    """Emit model source from a lowered package."""
    return backend._generate_source(package.ctx)
