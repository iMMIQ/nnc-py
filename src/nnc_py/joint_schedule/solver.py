"""Solver transport interfaces for the external joint tiling/schedule contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import subprocess
from typing import Final

from nnc_py.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
    JointFailure,
    JointProblem,
    JointSolution,
)


DEFAULT_SOLVER_TIMEOUT_SECONDS: Final[float] = 5.0


class JointSolverTransportError(RuntimeError):
    """Raised when the external solver transport or wire protocol fails."""


class JointScheduleSolver(ABC):
    """Abstract solver for joint tiling/schedule problems."""

    @abstractmethod
    def solve(self, problem: JointProblem) -> JointSolution | JointFailure:
        raise NotImplementedError


class CliJointScheduleSolver(JointScheduleSolver):
    """Ask an external CLI to solve the joint problem over JSON stdin/stdout.

    The wire contract is strict:
    - successful solutions must exit `0` and print `joint_tiling_schedule_solution_v1`
    - structured failures must exit `0` and print `joint_tiling_schedule_failure_v1`
    - any non-zero exit is treated as a transport/protocol failure even if stdout
      contains structured JSON
    - transport-side stderr is attached under diagnostics['_solver_transport']['stderr']
      when exit code is 0
    """

    def __init__(
        self,
        command: list[str] | tuple[str, ...],
        *,
        timeout_seconds: float = DEFAULT_SOLVER_TIMEOUT_SECONDS,
    ) -> None:
        self.command = tuple(command)
        self.timeout_seconds = max(float(timeout_seconds), 0.001)

    def solve(self, problem: JointProblem) -> JointSolution | JointFailure:
        if not self.command:
            raise JointSolverTransportError("solver command must not be empty")

        try:
            result = subprocess.run(
                list(self.command),
                input=json.dumps(problem.to_json(), sort_keys=True),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise JointSolverTransportError(
                f"solver command not found: {self.command[0]!r}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise JointSolverTransportError(
                f"solver command timed out after {self.timeout_seconds:.3f}s"
            ) from exc
        except (OSError, TypeError) as exc:
            raise JointSolverTransportError("failed to invoke solver command") from exc

        if result.returncode != 0:
            raise JointSolverTransportError(
                _format_transport_error(
                    f"solver command exited with code {result.returncode}",
                    stderr=result.stderr,
                )
            )

        payload = _load_json_payload(result.stdout)
        schema_version = payload.get("schema_version")
        if schema_version == JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION:
            solution = _parse_solution_payload(payload)
            if result.stderr.strip():
                diagnostics = _attach_solver_stderr(
                    solution.diagnostics, result.stderr.strip()
                )
                return JointSolution(
                    schema_version=solution.schema_version,
                    selected_recipes=solution.selected_recipes,
                    scheduled_actions=solution.scheduled_actions,
                    residency_windows=solution.residency_windows,
                    objective_value=solution.objective_value,
                    generated_sram_items=solution.generated_sram_items,
                    sram_allocations=solution.sram_allocations,
                    diagnostics=diagnostics,
                )
            return solution
        if schema_version == JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION:
            failure = _parse_failure_payload(payload)
            if result.stderr.strip():
                diagnostics = _attach_solver_stderr(
                    failure.diagnostics, result.stderr.strip()
                )
                return JointFailure(
                    schema_version=failure.schema_version,
                    status=failure.status,
                    error_category=failure.error_category,
                    diagnostics=diagnostics,
                )
            return failure
        raise JointSolverTransportError(
            _format_transport_error(
                f"solver returned unsupported schema_version {schema_version!r}",
                stderr=result.stderr,
            )
        )


def _load_json_payload(stdout: str) -> dict[str, object]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise JointSolverTransportError("solver stdout must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise JointSolverTransportError("solver stdout must be a JSON object")
    return payload


def _format_transport_error(message: str, *, stderr: str) -> str:
    stderr_text = stderr.strip()
    if not stderr_text:
        return message
    return f"{message}: {stderr_text}"


def _attach_solver_stderr(diagnostics: object, stderr: str) -> dict[str, object]:
    updated = dict(diagnostics) if isinstance(diagnostics, dict) else dict(diagnostics)
    transport_payload = updated.get("_solver_transport")
    if isinstance(transport_payload, dict):
        transport = dict(transport_payload)
    else:
        transport = {}
        if transport_payload is not None:
            transport["existing"] = transport_payload
    transport["stderr"] = stderr
    updated["_solver_transport"] = transport
    return updated


def _parse_solution_payload(payload: dict[str, object]) -> JointSolution:
    try:
        return JointSolution.from_json(payload)
    except (TypeError, ValueError) as exc:
        raise JointSolverTransportError("solver returned malformed solution payload") from exc


def _parse_failure_payload(payload: dict[str, object]) -> JointFailure:
    try:
        return JointFailure.from_json(payload)
    except (TypeError, ValueError) as exc:
        raise JointSolverTransportError("solver returned malformed failure payload") from exc


__all__ = [
    "CliJointScheduleSolver",
    "DEFAULT_SOLVER_TIMEOUT_SECONDS",
    "JointScheduleSolver",
    "JointSolverTransportError",
]
