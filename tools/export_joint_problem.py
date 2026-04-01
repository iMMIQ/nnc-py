#!/usr/bin/env python3
"""Export the O3 joint tiling/schedule problem for an ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nnc_py.tools.joint_problem_export import export_joint_problem_to_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the joint tiling/schedule problem JSON produced by the O3 preprocessing path."
    )
    parser.add_argument(
        "--model",
        default="models/resnet18.onnx",
        help="Path to the source ONNX model",
    )
    parser.add_argument(
        "--output",
        default="joint_solver/benchmarks/problems/resnet18_o3_1m.problem.json",
        help="Path to write the exported problem JSON",
    )
    parser.add_argument(
        "--max-memory",
        default="1M",
        help="Fast SRAM budget used while building the joint problem",
    )
    parser.add_argument(
        "--target",
        default="x86",
        help="Compilation target used for preprocessing",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=3,
        help="Optimization level used for preprocessing",
    )
    parser.add_argument(
        "--disable-constant-folding",
        action="store_true",
        help="Disable ONNX constant folding before preprocessing",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    problem = export_joint_problem_to_path(
        args.model,
        args.output,
        target=args.target,
        opt_level=args.opt_level,
        max_memory=args.max_memory,
        enable_constant_folding=not args.disable_constant_folding,
    )
    output_path = Path(args.output).resolve()
    print(f"Wrote {output_path}")
    print(f"schema_version={problem.schema_version}")
    print(f"regions={len(problem.regions)} recipes={len(problem.recipes)} actions={len(problem.actions)}")
    print(f"sram_capacity_bytes={problem.sram_capacity_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
