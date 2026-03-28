"""Lowered x86 codegen IR types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class X86CodegenPackage:
    """Minimal codegen package shared by x86 lowerers and emitters."""

    mode: str
    entry_point: str
    ctx: Any | None = None
    files: dict[str, Any] = field(default_factory=dict)
    pipeline_summary_lines: list[str] = field(default_factory=list)
    pipeline_codegen_metadata: dict[str, Any] = field(default_factory=dict)
    scheduled_plan: Any | None = None
    constants_metadata: dict[str, Any] = field(default_factory=dict)
