from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from nnc_py.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
)


def _build_solution(problem_payload: dict[str, object]) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    solver = root / "joint_solver" / "bin" / "nnc-joint-solver"
    result = subprocess.run(
        [sys.executable, str(solver)],
        input=json.dumps(problem_payload),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "external joint solver CLI failed: "
            + (result.stderr.strip() or f"exit {result.returncode}")
        )
    payload = json.loads(result.stdout)
    if payload.get("schema_version") != JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION:
        raise RuntimeError(
            "external joint solver CLI expected solution payload, got "
            f"{payload.get('schema_version')!r}"
        )
    payload["schema_version"] = JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION
    payload["diagnostics"] = {
        **dict(payload.get("diagnostics", {})),
        "mode": "solution",
    }
    return payload


def _build_failure(status: str) -> dict[str, object]:
    category = "solver_reported_infeasible" if status == "infeasible" else "invalid_solution"
    return {
        "schema_version": JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
        "status": status,
        "error_category": category,
        "diagnostics": {"mode": status},
    }


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "solution"
    problem = json.load(sys.stdin)

    if mode == "solution":
        json.dump(_build_solution(problem), sys.stdout)
        return 0
    if mode == "solution_stderr":
        json.dump(_build_solution(problem), sys.stdout)
        sys.stderr.write("solver emitted warning\n")
        return 0
    if mode == "malformed_solution":
        json.dump(
            {
                "schema_version": JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
                "selected_recipes": [],
                "scheduled_actions": [],
                "residency_windows": [],
                "objective_value": 1,
            },
            sys.stdout,
        )
        return 0
    if mode in {"infeasible", "timeout", "error"}:
        json.dump(_build_failure(mode), sys.stdout)
        return 0
    if mode == "malformed_failure":
        json.dump(
            {
                "schema_version": JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
                "status": "error",
                "error_category": "invalid_solution",
            },
            sys.stdout,
        )
        return 0
    if mode == "crash_with_payload":
        json.dump(_build_failure("error"), sys.stdout)
        sys.stderr.write("solver crashed after payload\n")
        return 7
    if mode == "crash":
        sys.stderr.write("solver crashed on purpose\n")
        return 7
    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
