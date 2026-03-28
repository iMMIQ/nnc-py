"""Constants loader emitter for x86 codegen packages."""

from __future__ import annotations

from typing import Any

from nnc_py.codegen.x86_ir import X86CodegenPackage


def emit_constants_loader(package: X86CodegenPackage, backend: Any) -> str:
    """Emit the constants loader source from a lowered package."""
    return backend._generate_constants_loader(package.ctx, package.constants_metadata)
