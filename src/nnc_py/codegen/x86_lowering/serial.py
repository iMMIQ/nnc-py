"""Serial x86 lowering for code generation."""

from __future__ import annotations

from typing import Any

from nnc_py.codegen.x86_ir import X86CodegenPackage
from nnc_py.ir.context import CompileContext


def lower_serial_x86_codegen(
    ctx: CompileContext,
    backend: Any,
    *,
    alloc_plan: Any | None = None,
) -> X86CodegenPackage:
    """Build serial x86 codegen metadata behind a dedicated lowering entry point."""
    pipeline_codegen_metadata = backend._build_pipeline_codegen_metadata(
        ctx,
        alloc_plan,
        scheduled_plan=None,
    )
    return X86CodegenPackage(
        mode="serial",
        entry_point=backend._get_public_entry_point(ctx),
        pipeline_summary_lines=list(pipeline_codegen_metadata.get("summary_lines", ())),
        pipeline_codegen_metadata=pipeline_codegen_metadata,
    )
