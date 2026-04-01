"""Solver interfaces and adapters for the joint tiling/schedule contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import json
import subprocess
from typing import Final

from nnc_py.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
    JointActionKind,
    JointFailureCategory,
    JointFailureStatus,
    JointFailure,
    JointProblem,
    JointResidencyWindow,
    JointSramAllocation,
    JointSramItem,
    JointSramItemKind,
    JointScheduledAction,
    JointSelectedRecipe,
    JointSolution,
)
from nnc_py.joint_schedule.validation import (
    validate_joint_problem,
    validate_joint_solution,
)


DEFAULT_SOLVER_TIMEOUT_SECONDS: Final[float] = 5.0


@dataclass(frozen=True)
class _LiveSramItem:
    order_index: int
    item: JointSramItem
    start_time: int
    end_time: int


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


class BaselineJointScheduleSolver(JointScheduleSolver):
    """Deterministic internal baseline for the external joint contract."""

    def solve(self, problem: JointProblem) -> JointSolution | JointFailure:
        problem_failure = validate_joint_problem(problem)
        if problem_failure is not None:
            return problem_failure

        recipe_by_region: dict[str, str] = {}
        for region in problem.regions:
            region_recipes = [
                recipe for recipe in problem.recipes if recipe.region_id == region.region_id
            ]
            if not region_recipes:
                return _baseline_failure(
                    JointFailureStatus.INVALID_PROBLEM,
                    JointFailureCategory.INVALID_SOLUTION,
                    f"region {region.region_id!r} has no recipes",
                )
            recipe_by_region[region.region_id] = region_recipes[0].recipe_id

        selected_recipes = tuple(
            JointSelectedRecipe(region_id=region.region_id, recipe_id=recipe_by_region[region.region_id])
            for region in problem.regions
        )

        recipes_by_id = {recipe.recipe_id: recipe for recipe in problem.recipes}
        actions_by_id = {action.action_id: action for action in problem.actions}
        mandatory_action_ids = {
            action_id
            for recipe_id in recipe_by_region.values()
            for action_id in recipes_by_id[recipe_id].activates_action_ids
        }
        active_actions = {
            action_id: actions_by_id[action_id]
            for action_id in mandatory_action_ids
        }
        schedule_order = _topological_action_order(problem, mandatory_action_ids)
        if schedule_order is None:
            return _baseline_failure(
                JointFailureStatus.ERROR,
                JointFailureCategory.DEPENDENCY_VIOLATION,
                "mandatory action graph is cyclic",
            )

        start_by_action: dict[str, int] = {}
        end_by_action: dict[str, int] = {}
        resource_available: dict[str, int] = defaultdict(int)
        predecessor_ids: dict[str, list[str]] = defaultdict(list)
        for edge in problem.dependency_edges:
            if edge.src_action_id in mandatory_action_ids and edge.dst_action_id in mandatory_action_ids:
                predecessor_ids[edge.dst_action_id].append(edge.src_action_id)

        for action_id in schedule_order:
            action = active_actions[action_id]
            earliest = max(
                [resource_available[action.resource_kind.value]]
                + [end_by_action[pred] for pred in predecessor_ids.get(action_id, ())]
            )
            start_by_action[action_id] = earliest
            end_by_action[action_id] = earliest + action.duration + action.launch_overhead
            resource_available[action.resource_kind.value] = end_by_action[action_id]

        objective_value = max(end_by_action.values(), default=0)
        residency_windows = tuple(
            _minimal_residency_windows(problem, active_actions, end_by_action, objective_value)
        )
        generated_sram_items = _generated_residency_items(problem, residency_windows)
        sram_allocations = _pack_sram_allocations(
            problem,
            fixed_items=problem.sram_items,
            generated_items=generated_sram_items,
            start_by_action=start_by_action,
            end_by_action=end_by_action,
            residency_windows=residency_windows,
            objective_value=objective_value,
        )
        if sram_allocations is None:
            return _baseline_failure(
                JointFailureStatus.ERROR,
                JointFailureCategory.SRAM_CAPACITY_EXCEEDED,
                "baseline SRAM allocator could not place active items within capacity",
            )
        solution = JointSolution(
            schema_version=JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
            selected_recipes=selected_recipes,
            scheduled_actions=tuple(
                JointScheduledAction(action_id=action_id, start_time=start_by_action[action_id])
                for action_id in schedule_order
            ),
            residency_windows=residency_windows,
            objective_value=objective_value,
            generated_sram_items=generated_sram_items,
            sram_allocations=sram_allocations,
            diagnostics={"solver": "baseline"},
        )
        solution_failure = validate_joint_solution(problem, solution)
        if solution_failure is not None:
            return solution_failure
        return solution


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


def _topological_action_order(
    problem: JointProblem,
    active_action_ids: set[str],
) -> tuple[str, ...] | None:
    successors: dict[str, list[str]] = {action_id: [] for action_id in active_action_ids}
    predecessor_counts: dict[str, int] = {action_id: 0 for action_id in active_action_ids}
    order_index = {action.action_id: index for index, action in enumerate(problem.actions)}
    for edge in problem.dependency_edges:
        if edge.src_action_id not in active_action_ids or edge.dst_action_id not in active_action_ids:
            continue
        successors[edge.src_action_id].append(edge.dst_action_id)
        predecessor_counts[edge.dst_action_id] += 1
    ready = sorted(
        [action_id for action_id, count in predecessor_counts.items() if count == 0],
        key=order_index.__getitem__,
    )
    order: list[str] = []
    while ready:
        action_id = ready.pop(0)
        order.append(action_id)
        for successor_id in sorted(successors[action_id], key=order_index.__getitem__):
            predecessor_counts[successor_id] -= 1
            if predecessor_counts[successor_id] == 0:
                ready.append(successor_id)
                ready.sort(key=order_index.__getitem__)
    if len(order) != len(active_action_ids):
        return None
    return tuple(order)


def _generated_residency_items(
    problem: JointProblem,
    residency_windows: tuple[JointResidencyWindow, ...],
) -> tuple[JointSramItem, ...]:
    size_by_value = {value.value_id: value.size_bytes for value in problem.values}
    return tuple(
        JointSramItem(
            item_id=f"{window.residency_id}.item",
            kind=JointSramItemKind.RESIDENT_WINDOW,
            size_bytes=size_by_value[window.value_id],
            alignment_bytes=problem.default_alignment_bytes,
            is_optional=False,
            owner_action_id=None,
            owner_value_id=window.value_id,
            owner_residency_id=window.residency_id,
        )
        for window in residency_windows
    )


def _pack_sram_allocations(
    problem: JointProblem,
    *,
    fixed_items: tuple[JointSramItem, ...],
    generated_items: tuple[JointSramItem, ...],
    start_by_action: dict[str, int],
    end_by_action: dict[str, int],
    residency_windows: tuple[JointResidencyWindow, ...],
    objective_value: int,
) -> tuple[JointSramAllocation, ...] | None:
    live_items = _collect_live_sram_items(
        fixed_items=fixed_items,
        generated_items=generated_items,
        start_by_action=start_by_action,
        end_by_action=end_by_action,
        residency_windows=residency_windows,
        objective_value=objective_value,
    )
    allocations: list[JointSramAllocation] = []
    active: list[tuple[int, int, int, int]] = []

    for live_item in live_items:
        active = [
            (start_time, end_time, offset, end_offset)
            for start_time, end_time, offset, end_offset in active
            if end_time > live_item.start_time
        ]
        candidate_offset = 0
        placed_offset: int | None = None
        for _, _, offset, end_offset in sorted(active, key=lambda interval: interval[2]):
            aligned_offset = _align(candidate_offset, live_item.item.alignment_bytes)
            if aligned_offset + live_item.item.size_bytes <= offset:
                placed_offset = aligned_offset
                break
            candidate_offset = max(candidate_offset, end_offset)
        if placed_offset is None:
            placed_offset = _align(candidate_offset, live_item.item.alignment_bytes)
        if placed_offset + live_item.item.size_bytes > problem.sram_capacity_bytes:
            return None
        allocations.append(
            JointSramAllocation(item_id=live_item.item.item_id, offset=placed_offset)
        )
        active.append(
            (
                live_item.start_time,
                live_item.end_time,
                placed_offset,
                placed_offset + live_item.item.size_bytes,
            )
        )

    return tuple(allocations)


def _collect_live_sram_items(
    *,
    fixed_items: tuple[JointSramItem, ...],
    generated_items: tuple[JointSramItem, ...],
    start_by_action: dict[str, int],
    end_by_action: dict[str, int],
    residency_windows: tuple[JointResidencyWindow, ...],
    objective_value: int,
) -> tuple[_LiveSramItem, ...]:
    windows_by_residency = {
        window.residency_id: window for window in residency_windows
    }
    live_items: list[_LiveSramItem] = []
    for order_index, item in enumerate((*fixed_items, *generated_items)):
        lifetime = _item_lifetime(
            item,
            start_by_action=start_by_action,
            end_by_action=end_by_action,
            windows_by_residency=windows_by_residency,
            objective_value=objective_value,
        )
        if lifetime is None:
            continue
        start_time, end_time = lifetime
        live_items.append(
            _LiveSramItem(
                order_index=order_index,
                item=item,
                start_time=start_time,
                end_time=end_time,
            )
        )
    return tuple(
        sorted(
            live_items,
            key=lambda live_item: (
                live_item.start_time,
                live_item.end_time,
                live_item.order_index,
            ),
        )
    )


def _item_lifetime(
    item: JointSramItem,
    *,
    start_by_action: dict[str, int],
    end_by_action: dict[str, int],
    windows_by_residency: dict[str, JointResidencyWindow],
    objective_value: int,
) -> tuple[int, int] | None:
    if item.owner_residency_id is not None:
        window = windows_by_residency.get(item.owner_residency_id)
        if window is None:
            return None
        return window.start_time, window.end_time
    if item.owner_action_id is not None:
        start_time = start_by_action.get(item.owner_action_id)
        end_time = end_by_action.get(item.owner_action_id)
        if start_time is None or end_time is None:
            return None
        return start_time, end_time
    return 0, objective_value


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _minimal_residency_windows(
    problem: JointProblem,
    active_actions: dict[str, object],
    end_by_action: dict[str, int],
    objective_value: int,
) -> list[JointResidencyWindow]:
    windows: list[JointResidencyWindow] = []
    active_action_ids = set(active_actions)
    for value in problem.values:
        active_consumers = [
            consumer.action_id
            for consumer in value.consumers
            if consumer.action_id in active_action_ids
        ]
        if not active_consumers:
            continue
        if value.producer is None:
            if value.initial_tier.value == "sram":
                open_end = 0
            else:
                open_end = next(
                    (
                        end_by_action[action_id]
                        for action_id, action in active_actions.items()
                        if action.kind is JointActionKind.DMA_IN and value.value_id in action.writes
                    ),
                    None,
                )
                if open_end is None:
                    continue
        else:
            if value.producer.action_id not in active_action_ids:
                continue
            open_end = end_by_action[value.producer.action_id]
        close_end = max(end_by_action[action_id] for action_id in active_consumers)
        if value.required_final_tier.value == "sram":
            close_end = objective_value
        if close_end > open_end:
            windows.append(
                JointResidencyWindow(
                    residency_id=f"{value.value_id}@{open_end}",
                    value_id=value.value_id,
                    start_time=open_end,
                    end_time=close_end,
                )
            )
    return windows


def _baseline_failure(
    status: JointFailureStatus,
    error_category: JointFailureCategory,
    reason: str,
) -> JointFailure:
    return JointFailure(
        schema_version=JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
        status=status,
        error_category=error_category,
        diagnostics={"reason": reason},
    )


__all__ = [
    "BaselineJointScheduleSolver",
    "CliJointScheduleSolver",
    "DEFAULT_SOLVER_TIMEOUT_SECONDS",
    "JointScheduleSolver",
    "JointSolverTransportError",
]
