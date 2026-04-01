"""Helpers for exporting joint tiling/schedule problem JSON from ONNX models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nnc_py.compiler import parse_memory_size
from nnc_py.frontend import ONNXFrontend
from nnc_py.ir.context import CompileContext
from nnc_py.ir.joint_tiling_schedule import JointProblem
from nnc_py.passes.base import PassManager
from nnc_py.passes.joint_tiling_schedule import JointTilingScheduleProblemPass


def build_joint_problem_from_onnx(
    onnx_path: str | Path,
    *,
    target: str = "x86",
    opt_level: int = 3,
    max_memory: str | None = None,
    enable_constant_folding: bool = True,
    metadata: dict[str, Any] | None = None,
) -> JointProblem:
    """Run the O3 preprocessing path up to JointTilingScheduleProblemPass."""
    if opt_level < 3:
        raise ValueError("joint problem export requires opt_level >= 3")

    frontend = ONNXFrontend(enable_simplify=enable_constant_folding)
    graph = frontend.load(str(onnx_path))
    ctx = CompileContext(graph, target, opt_level)
    ctx.metadata.update(dict(metadata or {}))
    ctx.metadata["enable_joint_tiling_schedule_contract"] = True
    ctx.metadata["joint_tiling_schedule_contract_enabled"] = True
    ctx.metadata["pipeline_scheduler_enabled"] = True
    if max_memory is not None:
        ctx.metadata["max_memory"] = parse_memory_size(max_memory)

    for pass_obj in PassManager.get_joint_tiling_schedule_o3_passes():
        pass_obj.run(ctx)
        if isinstance(pass_obj, JointTilingScheduleProblemPass):
            break

    if ctx.joint_tiling_schedule_failure is not None:
        raise RuntimeError(
            f"failed to build joint tiling schedule problem: {ctx.joint_tiling_schedule_failure}"
        )
    if ctx.joint_tiling_schedule_problem is None:
        raise RuntimeError("joint tiling schedule problem was not generated")
    return ctx.joint_tiling_schedule_problem


def export_joint_problem_to_path(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    target: str = "x86",
    opt_level: int = 3,
    max_memory: str | None = None,
    enable_constant_folding: bool = True,
    metadata: dict[str, Any] | None = None,
) -> JointProblem:
    """Build a joint problem and persist it as formatted JSON."""
    problem = build_joint_problem_from_onnx(
        onnx_path,
        target=target,
        opt_level=opt_level,
        max_memory=max_memory,
        enable_constant_folding=enable_constant_folding,
        metadata=metadata,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(problem.to_json(), indent=2, sort_keys=True) + "\n")
    return problem


__all__ = [
    "build_joint_problem_from_onnx",
    "export_joint_problem_to_path",
]
