from __future__ import annotations

import json
import sys

from nnc_py.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
    JointFailure,
    JointProblem,
    JointSolution,
)
from nnc_py.joint_schedule.solver import BaselineJointScheduleSolver


def _build_solution(problem_payload: dict[str, object]) -> dict[str, object]:
    problem = JointProblem.from_json(problem_payload)
    result = BaselineJointScheduleSolver().solve(problem)
    if isinstance(result, JointFailure):
        return _build_fallback_solution(problem_payload)
    if not isinstance(result, JointSolution):
        raise TypeError("baseline solver returned unexpected payload type")

    payload = result.to_json()
    payload["schema_version"] = JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION
    payload["diagnostics"] = {
        **dict(result.diagnostics),
        "mode": "solution",
    }
    return payload


def _build_fallback_solution(problem: dict[str, object]) -> dict[str, object]:
    recipes = problem.get("recipes", [])
    actions = problem.get("actions", [])

    recipes_by_region: dict[str, str] = {}
    for recipe in recipes:
        if not isinstance(recipe, dict):
            continue
        region_id = recipe.get("region_id")
        recipe_id = recipe.get("recipe_id")
        if isinstance(region_id, str) and isinstance(recipe_id, str):
            recipes_by_region.setdefault(region_id, recipe_id)

    mandatory_actions: list[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_id = action.get("action_id")
        is_optional = action.get("is_optional", False)
        if isinstance(action_id, str) and is_optional is False:
            mandatory_actions.append(action_id)

    return {
        "schema_version": JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
        "selected_recipes": [
            {"region_id": region_id, "recipe_id": recipe_id}
            for region_id, recipe_id in recipes_by_region.items()
        ],
        "scheduled_actions": [
            {"action_id": action_id, "start_time": index}
            for index, action_id in enumerate(mandatory_actions)
        ],
        "residency_windows": [],
        "objective_value": max(len(mandatory_actions), 1),
        "diagnostics": {"mode": "solution", "fallback": "simple"},
    }


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
