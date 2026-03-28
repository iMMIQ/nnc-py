"""Shared helpers for x86 codegen lowering."""

from __future__ import annotations

from typing import Any

from nnc_py.ir.context import CompileContext


def resolve_scheduled_codegen_inputs(
    backend: Any,
    ctx: CompileContext,
) -> tuple[Any | None, bool]:
    """Resolve scheduled-plan inputs from the current compile context."""
    scheduled_plan = backend._get_scheduled_memory_plan(ctx)
    prefer_scheduled_plan = backend._prefer_scheduled_memory_plan(ctx, scheduled_plan)
    return scheduled_plan, prefer_scheduled_plan
