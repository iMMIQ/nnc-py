"""Build artifact emitters for x86 codegen packages."""

from __future__ import annotations

from typing import Any

from nnc_py.codegen.x86_ir import X86CodegenPackage


def emit_makefile(package: X86CodegenPackage, backend: Any) -> str:
    """Emit the build Makefile from a lowered package."""
    return backend._generate_makefile(package.ctx)


def emit_test_runner(package: X86CodegenPackage, backend: Any) -> str:
    """Emit the x86 test runner from a lowered package."""
    return backend._generate_test_runner(package.ctx)
