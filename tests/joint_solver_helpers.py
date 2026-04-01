from __future__ import annotations

import sys
from pathlib import Path


def joint_solver_cli_command() -> list[str]:
    root = Path(__file__).resolve().parents[1]
    solver = root / "joint_solver" / "bin" / "nnc-joint-solver"
    if not solver.is_file():
        raise RuntimeError(
            "joint_solver submodule CLI is unavailable; run "
            "'git submodule update --init --recursive'."
        )
    return [sys.executable, str(solver)]

