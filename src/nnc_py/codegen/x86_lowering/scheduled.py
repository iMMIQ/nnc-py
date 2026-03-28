"""Scheduled-O3 lowering for x86 code generation."""

from __future__ import annotations

from typing import Any

from nnc_py.codegen.x86_ir import X86CodegenPackage
from nnc_py.codegen.x86_lowering.common import resolve_scheduled_codegen_inputs
from nnc_py.ir.context import CompileContext


def lower_scheduled_x86_codegen(
    ctx: CompileContext,
    backend: Any,
    *,
    alloc_plan: Any | None = None,
) -> X86CodegenPackage:
    """Build schedule-aware codegen metadata behind a dedicated lowering entry point."""
    scheduled_plan, prefer_scheduled_plan = resolve_scheduled_codegen_inputs(backend, ctx)
    pipeline_codegen_metadata = backend._build_pipeline_codegen_metadata(
        ctx,
        alloc_plan,
        scheduled_plan=scheduled_plan if prefer_scheduled_plan else None,
    )
    if prefer_scheduled_plan and scheduled_plan is not None:
        pipeline_codegen_metadata = backend._augment_parallel_runtime_for_scheduled_spill(
            ctx,
            scheduled_plan,
            pipeline_codegen_metadata,
        )
        if not bool(getattr(scheduled_plan, "transfer_points", ())):
            pipeline_codegen_metadata = backend._augment_parallel_runtime_for_scheduled_tile_streaming(
                ctx,
                pipeline_codegen_metadata,
            )
            pipeline_codegen_metadata = backend._augment_parallel_runtime_for_scheduled_home_execution(
                ctx,
                pipeline_codegen_metadata,
            )

    return X86CodegenPackage(
        mode="scheduled",
        entry_point=backend._get_public_entry_point(ctx),
        ctx=ctx,
        pipeline_summary_lines=list(pipeline_codegen_metadata.get("summary_lines", ())),
        pipeline_codegen_metadata=pipeline_codegen_metadata,
        scheduled_plan=scheduled_plan if prefer_scheduled_plan else None,
    )
