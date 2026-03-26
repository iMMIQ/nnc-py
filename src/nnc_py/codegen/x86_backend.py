"""x86 backend for simulation."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Union

import numpy as np

from nnc_py.codegen.base import BackendBase, CodeGenResult
from nnc_py.codegen.c_emitter import CEmitter
from nnc_py.ir.context import CompileContext
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.types import DataType
from nnc_py.passes.memory_strategy import (
    MemoryAllocationPlan,
    MovePoint,
    ReloadPoint,
    SpillPoint,
    TensorAllocation,
)
from nnc_py.utils.name_manager import NameManager

if TYPE_CHECKING:
    from nnc_py.passes.spill import SpillPlan


class X86Backend(BackendBase):
    """x86 target backend - generates code for simulation."""

    # Map DataType to NNC_DTYPE constant
    DTYPE_MAP = {
        DataType.FLOAT32: "NNC_DTYPE_FLOAT32",
        DataType.FLOAT16: "NNC_DTYPE_FLOAT16",
        DataType.INT32: "NNC_DTYPE_INT32",
        DataType.INT64: "NNC_DTYPE_INT64",
        DataType.INT8: "NNC_DTYPE_INT8",
        DataType.UINT8: "NNC_DTYPE_UINT8",
        DataType.BOOL: "NNC_DTYPE_BOOL",
    }

    def __init__(self, debug_mode: bool = False):
        """Initialize the x86 backend.

        Args:
            debug_mode: Whether to enable debug mode with intermediate tensor dumps.
        """
        self.debug_mode = debug_mode

    def generate(self, ctx: CompileContext) -> CodeGenResult:
        """Generate x86 C code."""
        result = CodeGenResult()
        self._prune_unused_tensor_defs(ctx)

        # Assign C symbol names
        self._assign_symbols(ctx)

        # Generate header file
        header = self._generate_header(ctx)
        result.add_file("model.h", header, "header")

        # Generate source file
        source = self._generate_source(ctx)
        result.add_file("model.c", source, "source")

        # Generate tensors definition file
        tensors = self._generate_tensors(ctx)
        result.add_file("tensors.c", tensors, "source")

        # Generate constants binary file and loader code
        if ctx.graph.constants:
            constants_binary, constants_metadata = self._generate_constants_binary(ctx)
            result.add_file("constants.bin", constants_binary, "binary")

            # Generate constants loader code
            constants_loader = self._generate_constants_loader(ctx, constants_metadata)
            result.add_file("constants_loader.c", constants_loader, "source")

        # Generate Makefile
        makefile = self._generate_makefile(ctx)
        result.add_file("Makefile", makefile, "build")

        # Generate test runner
        test_runner = self._generate_test_runner(ctx)
        result.add_file("test_runner.c", test_runner, "source")

        return result

    def _get_public_entry_point(self, ctx: CompileContext) -> str:
        """Get the public entry-point symbol name for generated code."""
        entry_point = ctx.metadata.get("entry_point", "nnc_run")
        if not isinstance(entry_point, str) or not entry_point:
            return "nnc_run"
        return entry_point

    def _build_pipeline_codegen_metadata(
        self,
        ctx: CompileContext,
        alloc_plan: "MemoryAllocationPlan | None",
        *,
        scheduled_plan: Any | None = None,
    ) -> dict[str, Any]:
        """Build schedule-visible metadata for generated simulation code."""
        try:
            schedule_problem = ctx.pipeline_schedule_problem
            schedule_result = ctx.pipeline_schedule_result
        except TypeError as exc:
            return {
                "summary_lines": [
                    "schedule_metadata=invalid",
                    f"schedule_metadata_error={self._sanitize_c_comment_text(str(exc))}",
                    self._build_memory_plan_summary_line(
                        alloc_plan,
                        ctx,
                        scheduled_plan=scheduled_plan,
                    ),
                ],
                "node_comments": {},
                "parallel_runtime": None,
            }

        summary_lines = ["schedule_metadata=absent"]
        node_comments: dict[str, list[str]] = {}

        if schedule_result is None:
            fallback_reason = ctx.metadata.get("pipeline_scheduler_fallback")
            if fallback_reason is not None:
                summary_lines.append(
                    f"fallback={self._sanitize_c_comment_text(str(fallback_reason))}"
                )
            summary_lines.append(
                self._build_memory_plan_summary_line(
                    alloc_plan,
                    ctx,
                    scheduled_plan=scheduled_plan,
                )
            )
            return {
                "summary_lines": summary_lines,
                "node_comments": node_comments,
                "parallel_runtime": None,
            }

        summary_lines = [
            "schedule_metadata=present",
            f"solver={self._sanitize_c_comment_text(schedule_result.solver_name or 'unknown')}",
            f"feasible={'yes' if schedule_result.feasible else 'no'}",
            f"makespan={schedule_result.makespan}",
            f"scheduled_steps={len(schedule_result.scheduled_steps)}",
            self._build_memory_plan_summary_line(
                alloc_plan,
                ctx,
                scheduled_plan=scheduled_plan,
            ),
        ]
        diagnostics_summary = self._format_json_mapping(schedule_result.diagnostics)
        if diagnostics_summary:
            summary_lines.append(f"diagnostics={diagnostics_summary}")

        parallel_runtime = self._build_pipeline_parallel_runtime_metadata(
            ctx,
            schedule_problem=schedule_problem,
            schedule_result=schedule_result,
            scheduled_plan=scheduled_plan,
        )
        if parallel_runtime is not None:
            summary_lines.append("parallel_runtime=enabled(workers=4)")
        else:
            summary_lines.append("parallel_runtime=disabled")

        if schedule_problem is None or not schedule_result.feasible:
            return {
                "summary_lines": summary_lines,
                "node_comments": node_comments,
                "parallel_runtime": parallel_runtime,
            }

        problem_steps = {step.id: step for step in schedule_problem.steps}
        value_bindings = self._build_schedule_value_bindings(
            ctx,
            alloc_plan,
            schedule_result,
            scheduled_plan=scheduled_plan,
        )

        for scheduled_step in sorted(
            schedule_result.scheduled_steps,
            key=lambda step: (step.start_time, step.end_time, step.step_id),
        ):
            problem_step = problem_steps.get(scheduled_step.step_id)
            if problem_step is None:
                continue
            node_comments.setdefault(problem_step.node_name, []).append(
                self._format_pipeline_step_comment(
                    scheduled_step=scheduled_step,
                    problem_step=problem_step,
                    value_bindings=value_bindings,
                )
            )

        return {
            "summary_lines": summary_lines,
            "node_comments": node_comments,
            "parallel_runtime": parallel_runtime,
        }

    def _build_pipeline_parallel_runtime_metadata(
        self,
        ctx: CompileContext,
        *,
        schedule_problem: Any,
        schedule_result: Any,
        scheduled_plan: Any | None = None,
    ) -> dict[str, Any] | None:
        """Build structured metadata used to emit a true parallel C simulation runtime."""
        if self.debug_mode:
            return None
        if schedule_problem is None:
            return None
        if schedule_result is None or not schedule_result.feasible:
            return None

        problem_steps = {step.id: step for step in schedule_problem.steps}
        if not problem_steps:
            return None

        nodes_with_compute = {
            step.node_name
            for step in schedule_problem.steps
            if getattr(getattr(step, "step_kind", None), "value", None) == "compute"
        }
        resource_to_index = {
            "dma": 0,
            "shape": 1,
            "matmul": 2,
            "other": 3,
        }

        ordered_scheduled_steps = sorted(
            schedule_result.scheduled_steps,
            key=lambda step: (step.start_time, step.end_time, step.step_id),
        )
        step_records: list[dict[str, Any]] = []
        for scheduled_step in ordered_scheduled_steps:
            problem_step = problem_steps.get(scheduled_step.step_id)
            if problem_step is None:
                continue

            resource_kind = getattr(scheduled_step.resource_kind, "value", "other")
            resource_index = resource_to_index.get(resource_kind)
            if resource_index is None:
                continue

            step_kind = getattr(getattr(problem_step, "step_kind", None), "value", "")
            invoke_node = False
            if step_kind == "compute":
                invoke_node = True
            elif step_kind == "shape_prep" and problem_step.node_name not in nodes_with_compute:
                invoke_node = True

            invoke_symbol = None
            invoke_symbol = self._parallel_step_function_name(problem_step.id)
            input_tensor_symbols = self._collect_parallel_step_tensor_symbols(
                ctx,
                tuple(getattr(problem_step, "sram_input_names", ())),
            )
            output_tensor_symbols = self._collect_parallel_step_tensor_symbols(
                ctx,
                tuple(getattr(problem_step, "sram_output_names", ())),
            )
            input_value_records = self._collect_parallel_step_value_records(
                ctx,
                tuple(getattr(problem_step, "sram_input_names", ())),
                step_id=str(problem_step.id),
                scheduled_plan=scheduled_plan,
            )
            output_value_records = self._collect_parallel_step_value_records(
                ctx,
                tuple(getattr(problem_step, "sram_output_names", ())),
                step_id=str(problem_step.id),
                scheduled_plan=scheduled_plan,
            )

            step_records.append(
                {
                    "step_id": scheduled_step.step_id,
                    "node_name": problem_step.node_name,
                    "step_kind": step_kind,
                    "resource_index": resource_index,
                    "start_time": int(scheduled_step.start_time),
                    "end_time": int(scheduled_step.end_time),
                    "invoke_symbol": invoke_symbol,
                    "invoke_node": invoke_node,
                    "node_symbol": ctx.node_symbols.get(problem_step.node_name, problem_step.node_name),
                    "input_tensor_symbols": input_tensor_symbols,
                    "output_tensor_symbols": output_tensor_symbols,
                    "input_value_records": input_value_records,
                    "output_value_records": output_value_records,
                }
            )

        if not step_records:
            return None

        step_index_by_id = {
            record["step_id"]: index
            for index, record in enumerate(step_records)
        }
        predecessor_indices: list[set[int]] = [set() for _ in step_records]
        successor_indices: list[set[int]] = [set() for _ in step_records]
        for edge in getattr(schedule_problem, "edges", ()):
            src_index = step_index_by_id.get(edge.src_step_id)
            dst_index = step_index_by_id.get(edge.dst_step_id)
            if src_index is None or dst_index is None:
                continue
            predecessor_indices[dst_index].add(src_index)
            successor_indices[src_index].add(dst_index)

        return {
            "enabled": True,
            "worker_count": 4,
            "steps": tuple(step_records),
            "predecessor_indices": tuple(
                tuple(sorted(indices)) for indices in predecessor_indices
            ),
            "successor_indices": tuple(
                tuple(sorted(indices)) for indices in successor_indices
            ),
        }

    def _has_parallel_runtime(self, pipeline_codegen_metadata: dict[str, Any]) -> bool:
        runtime = pipeline_codegen_metadata.get("parallel_runtime")
        return bool(isinstance(runtime, dict) and runtime.get("enabled") is True)

    def _parallel_step_function_name(self, step_id: str) -> str:
        sanitized = []
        for char in step_id:
            if char.isalnum():
                sanitized.append(char)
            else:
                sanitized.append("_")
        return "nnc_pipeline_step_" + "".join(sanitized)

    def _get_scheduled_memory_plan(self, ctx: CompileContext) -> Any | None:
        plan = ctx.metadata.get("scheduled_memory_plan")
        if plan is None:
            return None
        if not hasattr(plan, "fast_allocations") or not hasattr(plan, "transfer_points"):
            return None
        return plan

    def _prefer_scheduled_memory_plan(
        self,
        ctx: CompileContext,
        scheduled_plan: Any | None = None,
    ) -> bool:
        if scheduled_plan is None:
            scheduled_plan = self._get_scheduled_memory_plan(ctx)
        return (
            scheduled_plan is not None
            and ctx.optimization_level >= 3
            and bool(ctx.metadata.get("pipeline_scheduler_enabled"))
        )

    def _build_schedule_value_map(
        self,
        ctx: CompileContext,
    ) -> dict[str, Any]:
        values_by_name: dict[str, Any] = {}
        for values in (
            getattr(ctx.metadata.get("pipeline_schedule_problem"), "scheduled_values", ()),
            getattr(ctx.metadata.get("pipeline_schedule_result"), "scheduled_values", ()),
        ):
            for value in values or ():
                value_name = getattr(value, "name", None)
                if isinstance(value_name, str) and value_name:
                    values_by_name[value_name] = value
        return values_by_name

    def _build_schedule_value_graph_tensor_map(
        self,
        ctx: CompileContext,
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for values in (
            getattr(ctx.metadata.get("pipeline_schedule_result"), "scheduled_values", ()),
            getattr(ctx.metadata.get("pipeline_schedule_problem"), "scheduled_values", ()),
        ):
            for value in values or ():
                value_name = getattr(value, "name", None)
                graph_tensor_name = getattr(value, "graph_tensor_name", None)
                if not isinstance(value_name, str) or not isinstance(graph_tensor_name, str):
                    continue
                if not graph_tensor_name:
                    continue
                mapping.setdefault(value_name, graph_tensor_name)
        return mapping

    def _resolve_schedule_value_graph_tensor_name(
        self,
        ctx: CompileContext,
        value_name: str,
    ) -> str | None:
        candidates: list[str] = []
        graph_tensor_name = self._build_schedule_value_graph_tensor_map(ctx).get(value_name)
        if isinstance(graph_tensor_name, str) and graph_tensor_name:
            candidates.append(graph_tensor_name)

        decoded_name = self._decode_schedule_value_graph_tensor_name(value_name)
        if isinstance(decoded_name, str) and decoded_name:
            candidates.append(decoded_name)

        inferred_name = self._infer_schedule_value_graph_tensor_name(value_name)
        if isinstance(inferred_name, str) and inferred_name:
            candidates.append(inferred_name)

        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate in ctx.graph.tensors:
                return candidate
        return None

    def _infer_schedule_value_graph_tensor_name(self, value_name: str) -> str | None:
        if not value_name:
            return None
        for marker in (".reload", ".spill"):
            if marker in value_name:
                candidate = value_name.split(marker, 1)[0]
                if candidate:
                    return candidate
        return None

    def _decode_schedule_value_graph_tensor_name(self, value_name: str) -> str | None:
        if value_name.startswith("sram|node|") and "|tensor|" in value_name:
            encoded_tensor = value_name.split("|tensor|", 1)[1]
            name_parts = encoded_tensor.split(":", 1)
            if len(name_parts) == 2:
                return name_parts[1]
            return encoded_tensor
        if value_name.startswith("sram|"):
            return None
        return value_name

    def _collect_parallel_step_tensor_symbols(
        self,
        ctx: CompileContext,
        value_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        symbols: list[str] = []
        for value_name in value_names:
            graph_tensor_name = self._resolve_schedule_value_graph_tensor_name(
                ctx,
                value_name,
            )
            if graph_tensor_name is None:
                continue
            if graph_tensor_name not in ctx.graph.tensors:
                continue
            symbol = ctx.tensor_symbols.get(graph_tensor_name, graph_tensor_name)
            if symbol not in symbols:
                symbols.append(symbol)
        return tuple(symbols)

    def _parallel_value_storage_name(self, value_name: str) -> str:
        sanitized = []
        for char in value_name:
            if char.isalnum():
                sanitized.append(char)
            else:
                sanitized.append("_")
        return "_nnc_pipeline_value_" + "".join(sanitized)

    def _parallel_tensor_saved_data_name(self, tensor_symbol: str) -> str:
        return f"_nnc_pipeline_saved_{tensor_symbol}_data"

    def _resolve_schedule_value_home_tier(
        self,
        ctx: CompileContext,
        value_name: str,
    ) -> str:
        value = self._build_schedule_value_map(ctx).get(value_name)
        if value is not None:
            tier = getattr(getattr(value, "home_tier", None), "value", None)
            if isinstance(tier, str) and tier:
                return tier

        graph_tensor_name = self._resolve_schedule_value_graph_tensor_name(ctx, value_name)
        if graph_tensor_name in ctx.graph.inputs:
            return "input"
        if graph_tensor_name in ctx.graph.constants:
            return "const"
        return "sram"

    def _resolve_scheduled_fast_allocation_for_step(
        self,
        ctx: CompileContext,
        scheduled_plan: Any | None,
        *,
        value_name: str,
        step_id: str,
    ) -> Any | None:
        if scheduled_plan is None:
            return None

        candidates = [
            allocation
            for allocation in getattr(scheduled_plan, "fast_allocations", {}).values()
            if getattr(allocation, "value_name", None) == value_name
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        schedule_result = ctx.metadata.get("pipeline_schedule_result")
        step_placement = next(
            (
                scheduled_step
                for scheduled_step in getattr(schedule_result, "scheduled_steps", ())
                if getattr(scheduled_step, "step_id", None) == step_id
            ),
            None,
        )

        exact_matches = [
            allocation
            for allocation in candidates
            if getattr(allocation, "opened_by_step_id", None) == step_id
            or getattr(allocation, "closed_by_step_id", None) == step_id
        ]
        if exact_matches:
            exact_matches.sort(
                key=lambda allocation: (
                    0 if getattr(allocation, "opened_by_step_id", None) == step_id else 1,
                    int(getattr(allocation, "start_time", 0)),
                    int(getattr(allocation, "end_time", 0)),
                    str(getattr(allocation, "residency_id", "")),
                )
            )
            return exact_matches[0]

        if step_placement is not None:
            overlapping = [
                allocation
                for allocation in candidates
                if int(getattr(allocation, "start_time", 0)) <= int(getattr(step_placement, "end_time", 0))
                and int(getattr(allocation, "end_time", 0)) >= int(getattr(step_placement, "start_time", 0))
            ]
            if overlapping:
                overlapping.sort(
                    key=lambda allocation: (
                        abs(int(getattr(allocation, "start_time", 0)) - int(getattr(step_placement, "start_time", 0))),
                        abs(int(getattr(allocation, "end_time", 0)) - int(getattr(step_placement, "end_time", 0))),
                        str(getattr(allocation, "residency_id", "")),
                    )
                )
                return overlapping[0]

        candidates.sort(
            key=lambda allocation: (
                int(getattr(allocation, "start_time", 0)),
                int(getattr(allocation, "end_time", 0)),
                str(getattr(allocation, "residency_id", "")),
            )
        )
        return candidates[0]

    def _clone_pipeline_codegen_metadata(
        self,
        pipeline_codegen_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        cloned = dict(pipeline_codegen_metadata)
        runtime = pipeline_codegen_metadata.get("parallel_runtime")
        if not isinstance(runtime, dict):
            return cloned

        cloned_runtime = dict(runtime)
        cloned_runtime["steps"] = tuple(dict(step) for step in runtime.get("steps", ()))
        custom_declarations = runtime.get("custom_declarations", ())
        cloned_runtime["custom_declarations"] = tuple(str(line) for line in custom_declarations)
        cloned["parallel_runtime"] = cloned_runtime
        return cloned

    def _augment_parallel_runtime_for_legacy_spill(
        self,
        ctx: CompileContext,
        spill_plan: "SpillPlan",
        pipeline_codegen_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._has_parallel_runtime(pipeline_codegen_metadata):
            return pipeline_codegen_metadata

        cloned = self._clone_pipeline_codegen_metadata(pipeline_codegen_metadata)
        runtime = cloned.get("parallel_runtime")
        if not isinstance(runtime, dict):
            return pipeline_codegen_metadata

        steps_by_id = {
            str(step["step_id"]): step
            for step in tuple(runtime.get("steps", ()))
        }
        nodes = ctx.graph.topological_sort()

        for node in nodes:
            if node.op_type == OpType.CONSTANT:
                continue

            func_name = ctx.node_symbols.get(node.name, node.name)
            dma_in_lines: list[str] = []
            for reload_point in spill_plan.reload_points:
                if reload_point.before_node != node.name:
                    continue
                dma_in_lines.extend(
                    [
                        "memcpy(",
                        f"    _nnc_fast_pool + {reload_point.to_fast_offset},",
                        f"    _nnc_slow_pool + {reload_point.from_slow_offset},",
                        f"    {reload_point.size}",
                        ");",
                    ]
                )

            dma_out_lines: list[str] = []
            for spill_point in spill_plan.spill_points:
                if spill_point.after_node != node.name:
                    continue
                dma_out_lines.extend(
                    [
                        "memcpy(",
                        f"    _nnc_slow_pool + {spill_point.to_slow_offset},",
                        f"    _nnc_fast_pool + {spill_point.from_fast_offset},",
                        f"    {spill_point.size}",
                        ");",
                    ]
                )

            compute_step = steps_by_id.get(f"{node.name}.compute")
            if compute_step is not None:
                compute_step["custom_body_lines"] = (f"{func_name}_body();",)

            dma_in_step = steps_by_id.get(f"{node.name}.dma_in")
            if dma_in_step is not None and dma_in_lines:
                dma_in_step["custom_body_lines"] = tuple(dma_in_lines)

            dma_out_step = steps_by_id.get(f"{node.name}.dma_out")
            if dma_out_step is not None and dma_out_lines:
                dma_out_step["custom_body_lines"] = tuple(dma_out_lines)

        return cloned

    def _augment_parallel_runtime_for_scheduled_spill(
        self,
        ctx: CompileContext,
        scheduled_plan: Any,
        pipeline_codegen_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._has_parallel_runtime(pipeline_codegen_metadata):
            return pipeline_codegen_metadata
        if not getattr(scheduled_plan, "transfer_points", ()):
            return pipeline_codegen_metadata

        cloned = self._clone_pipeline_codegen_metadata(pipeline_codegen_metadata)
        runtime = cloned.get("parallel_runtime")
        if not isinstance(runtime, dict):
            return pipeline_codegen_metadata

        steps_by_id = {
            str(step["step_id"]): step
            for step in tuple(runtime.get("steps", ()))
        }
        for transfer_point in tuple(getattr(scheduled_plan, "transfer_points", ())):
            step = steps_by_id.get(str(getattr(transfer_point, "step_id", "")))
            if step is None:
                continue

            custom_body_lines = self._build_scheduled_transfer_body_lines(
                ctx,
                transfer_point,
            )
            if not custom_body_lines:
                continue

            step["custom_body_lines"] = tuple(custom_body_lines)

        return cloned

    def _augment_parallel_runtime_for_scheduled_home_execution(
        self,
        ctx: CompileContext,
        pipeline_codegen_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._has_parallel_runtime(pipeline_codegen_metadata):
            return pipeline_codegen_metadata
        if not self._should_use_scheduled_home_execution(ctx):
            return pipeline_codegen_metadata

        cloned = self._clone_pipeline_codegen_metadata(pipeline_codegen_metadata)
        runtime = cloned.get("parallel_runtime")
        if not isinstance(runtime, dict):
            return pipeline_codegen_metadata

        for step in tuple(runtime.get("steps", ())):
            if step.get("custom_body_lines") is not None:
                continue
            node_name = step.get("node_name")
            if isinstance(node_name, str) and node_name:
                if not self._scheduled_step_requires_home_execution(ctx, node_name):
                    continue
            step_kind = str(step.get("step_kind", ""))
            node_symbol = step.get("node_symbol")
            if step_kind == "compute" and isinstance(node_symbol, str) and node_symbol:
                step["custom_body_lines"] = (f"{node_symbol}();",)
            elif (
                step_kind == "shape_prep"
                and bool(step.get("invoke_node"))
                and isinstance(node_symbol, str)
                and node_symbol
            ):
                step["custom_body_lines"] = (f"{node_symbol}();",)
            else:
                step["custom_body_lines"] = ("/* scheduled home execution */",)

        return cloned

    def _augment_parallel_runtime_for_scheduled_tile_streaming(
        self,
        ctx: CompileContext,
        pipeline_codegen_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._has_parallel_runtime(pipeline_codegen_metadata):
            return pipeline_codegen_metadata

        streaming_plan = self._build_scheduled_tile_streaming_plan(ctx, pipeline_codegen_metadata)
        if not streaming_plan["groups"]:
            return pipeline_codegen_metadata

        cloned = self._clone_pipeline_codegen_metadata(pipeline_codegen_metadata)
        runtime = cloned.get("parallel_runtime")
        if not isinstance(runtime, dict):
            return pipeline_codegen_metadata

        steps_by_id = {
            str(step["step_id"]): step
            for step in tuple(runtime.get("steps", ()))
        }

        custom_declarations = list(runtime.get("custom_declarations", ()))
        custom_declarations.extend(streaming_plan["helper_definitions"])
        runtime["custom_declarations"] = tuple(custom_declarations)

        for group in streaming_plan["groups"]:
            for step_id in group["noop_step_ids"]:
                step = steps_by_id.get(step_id)
                if step is None:
                    continue
                step["custom_body_lines"] = ("/* scheduled tile streaming */",)
            invoke_step = steps_by_id.get(group["invoke_step_id"])
            if invoke_step is not None:
                invoke_step["custom_body_lines"] = (f"{group['helper_name']}();",)
                if "invoke_resource_kind" in group:
                    invoke_step["resource_kind"] = group["invoke_resource_kind"]

        return cloned

    def _build_scheduled_tile_streaming_plan(
        self,
        ctx: CompileContext,
        pipeline_codegen_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        runtime = pipeline_codegen_metadata.get("parallel_runtime")
        if not isinstance(runtime, dict) or runtime.get("enabled") is not True:
            return {"groups": [], "helper_definitions": [], "internal_tensor_names": set()}

        execution_plans = ctx.metadata.get("node_execution_plans")
        if not isinstance(execution_plans, dict) or not execution_plans:
            return {"groups": [], "helper_definitions": [], "internal_tensor_names": set()}

        node_by_name = {node.name: node for node in ctx.graph.topological_sort()}
        step_records_by_node: dict[str, list[dict[str, Any]]] = {}
        for step in tuple(runtime.get("steps", ())):
            node_name = step.get("node_name")
            if not isinstance(node_name, str) or not node_name:
                continue
            step_records_by_node.setdefault(node_name, []).append(step)

        groups: list[dict[str, Any]] = []
        helper_definitions: list[str] = []
        internal_tensor_names: set[str] = set()
        max_required_fast_memory = 0
        for execution_group in self._collect_scheduled_tile_streaming_execution_groups(
            ctx,
            execution_plans,
        ):
            group_plan = self._build_scheduled_tile_streaming_group_plan(
                ctx,
                execution_group,
                execution_plans=execution_plans,
                node_by_name=node_by_name,
                step_records_by_node=step_records_by_node,
            )
            if group_plan is None:
                continue
            groups.append(group_plan)
            helper_definitions.extend(group_plan["helper_definition"])
            internal_tensor_names.update(group_plan["internal_tensor_names"])
            max_required_fast_memory = max(
                max_required_fast_memory,
                int(group_plan["required_fast_memory"]),
            )

        return {
            "groups": groups,
            "helper_definitions": helper_definitions,
            "internal_tensor_names": internal_tensor_names,
            "required_fast_memory": max_required_fast_memory,
        }

    def _build_scheduled_tile_streaming_group_plan(
        self,
        ctx: CompileContext,
        execution_group: dict[str, Any],
        *,
        execution_plans: dict[str, Any],
        node_by_name: dict[str, Node],
        step_records_by_node: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        node_names = tuple(execution_group.get("node_names", ()))
        if not node_names:
            return None

        root_name = node_names[0]
        root_node = node_by_name.get(root_name)
        root_plan = execution_plans.get(root_name)
        if root_node is None or root_plan is None:
            return None
        root_family = getattr(root_plan, "op_family", None)

        root_compute_step = self._find_parallel_runtime_step(step_records_by_node, root_name, "compute")
        if root_compute_step is None:
            return None

        final_node = node_by_name[node_names[-1]]
        final_compute_step = self._find_parallel_runtime_step(
            step_records_by_node,
            final_node.name,
            "compute",
        )
        if final_compute_step is None:
            return None
        final_output_record = self._find_value_record(
            tuple(final_compute_step.get("output_value_records", ())),
            tensor_name=final_node.outputs[0],
            staged_only=True,
        )
        if final_output_record is None:
            return None

        if root_family == "conv2d":
            if any(
                node_by_name.get(node_name) is None
                or node_by_name[node_name].op_type
                not in {
                    OpType.CONV2D,
                    OpType.FUSED_CONV_RELU,
                    OpType.ADD,
                    OpType.FUSED_ADD_RELU,
                    OpType.RELU,
                }
                for node_name in node_names
            ):
                return None
            conv_input_record = self._find_value_record(
                tuple(root_compute_step.get("input_value_records", ())),
                tensor_name=root_node.inputs[0],
                staged_only=True,
            )
            conv_output_record = self._find_value_record(
                tuple(root_compute_step.get("output_value_records", ())),
                tensor_name=root_node.outputs[0],
                staged_only=True,
            )
            if conv_input_record is None or conv_output_record is None:
                return None

            primary_buffer_size = self._align_c_buffer_size(int(conv_input_record["size_bytes"]))
            secondary_buffer_size = self._align_c_buffer_size(
                max(int(conv_output_record["size_bytes"]), int(final_output_record["size_bytes"]))
            )
            required_fast_memory = primary_buffer_size + secondary_buffer_size
            max_memory = ctx.metadata.get("max_memory")
            if isinstance(max_memory, int) and max_memory > 0 and required_fast_memory > max_memory:
                return None

            helper_lines = self._render_scheduled_tile_streaming_helper(
                ctx,
                execution_group=execution_group,
                node_by_name=node_by_name,
                execution_plans=execution_plans,
                step_records_by_node=step_records_by_node,
                conv_input_record=conv_input_record,
                conv_output_record=conv_output_record,
                extra_stage_record=conv_input_record,
                primary_buffer_offset=0,
                secondary_buffer_offset=primary_buffer_size,
            )
            if not helper_lines:
                return None
        elif root_family in {"maxpool", "average_pool", "global_average_pool"}:
            if len(node_names) != 1:
                return None
            helper_lines, required_fast_memory = self._render_scheduled_pool_tile_streaming_helper(
                ctx,
                root_node=root_node,
                root_plan=root_plan,
                primary_buffer_offset=0,
            )
            if not helper_lines:
                return None
        else:
            return None

        invoke_step = root_compute_step
        invoke_resource_kind = root_compute_step.get("resource_kind")
        if len(node_names) > 1:
            invoke_step = final_compute_step

        noop_step_ids: list[str] = []
        for node_name in node_names:
            for step in step_records_by_node.get(node_name, ()):
                step_id = str(step.get("step_id", ""))
                if not step_id:
                    continue
                if step_id == str(invoke_step.get("step_id", "")):
                    continue
                noop_step_ids.append(step_id)

        internal_tensor_names = {
            tensor_name
            for tensor_name in execution_group.get("fast_tensors", ())
            if tensor_name in ctx.graph.tensors and tensor_name not in ctx.graph.outputs
        }
        final_output_names = set(node_by_name[node_names[-1]].outputs)
        internal_tensor_names.difference_update(final_output_names)

        return {
            "node_names": node_names,
            "invoke_step_id": str(invoke_step["step_id"]),
            "invoke_resource_kind": invoke_resource_kind,
            "noop_step_ids": tuple(dict.fromkeys(noop_step_ids)),
            "helper_name": helper_lines[0].removeprefix("static void ").split("(")[0],
            "helper_definition": helper_lines,
            "internal_tensor_names": internal_tensor_names,
            "required_fast_memory": required_fast_memory,
        }

    def _render_scheduled_pool_tile_streaming_helper(
        self,
        ctx: CompileContext,
        *,
        root_node: Node,
        root_plan: Any,
        primary_buffer_offset: int,
    ) -> tuple[list[str], int]:
        def nchw_tensor_tile_nbytes(tensor: Any, tile_h: int, tile_w: int) -> int:
            dims = tuple(int(dim) for dim in tensor.shape.dims)
            if len(dims) != 4:
                return 0
            total_elems = max(1, dims[0] * dims[1] * dims[2] * dims[3])
            elem_size = max(1, int(tensor.byte_size() // total_elems))
            return elem_size * dims[1] * tile_h * tile_w

        if len(root_node.inputs) != 1 or len(root_node.outputs) != 1:
            return [], 0

        input_symbol = ctx.tensor_symbols.get(root_node.inputs[0], root_node.inputs[0])
        output_symbol = ctx.tensor_symbols.get(root_node.outputs[0], root_node.outputs[0])
        input_tensor = ctx.graph.tensors.get(root_node.inputs[0])
        output_tensor = ctx.graph.tensors.get(root_node.outputs[0])
        if input_tensor is None or output_tensor is None:
            return [], 0

        output_access = root_plan.output_accesses[0] if root_plan.output_accesses else None
        if output_access is None:
            return [], 0
        output_tile_extents = tuple(output_access.tile_region.logical_extents)
        if len(output_tile_extents) != 2:
            return [], 0

        root_family = getattr(root_plan, "op_family", None)
        if root_family == "global_average_pool":
            input_tile_h = int(input_tensor.shape.dims[2])
            input_tile_w = int(input_tensor.shape.dims[3])
            kernel_h = input_tile_h
            kernel_w = input_tile_w
            stride_h = input_tile_h
            stride_w = input_tile_w
            pad_h = 0
            pad_w = 0
            input_origin_h_expr = "0"
            input_origin_w_expr = "0"
            input_h_expr = str(input_tile_h)
            input_w_expr = str(input_tile_w)
        else:
            input_access = root_plan.input_accesses[0] if root_plan.input_accesses else None
            if input_access is None:
                return [], 0
            input_tile_extents = tuple(input_access.tile_region.logical_extents)
            if len(input_tile_extents) != 2:
                return [], 0
            input_tile_h = int(input_tile_extents[0])
            input_tile_w = int(input_tile_extents[1])
            kernel_shape = root_node.attrs.get("kernel_shape", [1, 1])
            strides = root_node.attrs.get("strides", [1, 1])
            pads = root_node.attrs.get("pads", [0, 0, 0, 0])
            kernel_h = int(kernel_shape[0]) if len(kernel_shape) >= 1 else 1
            kernel_w = int(kernel_shape[1]) if len(kernel_shape) >= 2 else kernel_h
            stride_h = int(strides[0]) if len(strides) >= 1 else 1
            stride_w = int(strides[1]) if len(strides) >= 2 else stride_h
            pad_h = int(pads[0]) if len(pads) >= 1 else 0
            pad_w = int(pads[1]) if len(pads) >= 2 else pad_h
            input_origin_h_expr = f"_nnc_h0 * {stride_h} - {pad_h}"
            input_origin_w_expr = f"_nnc_w0 * {stride_w} - {pad_w}"
            input_h_expr = f"(_nnc_tile_h_0 - 1) * {stride_h} + {kernel_h}"
            input_w_expr = f"(_nnc_tile_w_0 - 1) * {stride_w} + {kernel_w}"

        input_buffer_size = self._align_c_buffer_size(
            nchw_tensor_tile_nbytes(input_tensor, input_tile_h, input_tile_w)
        )
        output_buffer_size = self._align_c_buffer_size(
            nchw_tensor_tile_nbytes(
                output_tensor,
                int(output_tile_extents[0]),
                int(output_tile_extents[1]),
            )
        )
        required_fast_memory = input_buffer_size + output_buffer_size
        max_memory = ctx.metadata.get("max_memory")
        if isinstance(max_memory, int) and max_memory > 0 and required_fast_memory > max_memory:
            return [], 0

        input_stage_expr = f"_nnc_fast_pool + {int(primary_buffer_offset)}"
        output_stage_expr = f"_nnc_fast_pool + {int(primary_buffer_offset + input_buffer_size)}"
        helper_name = f"nnc_pipeline_tile_stream_{self._sanitize_c_ident(root_node.name)}"
        pool_call = "nnc_maxpool2d" if root_family == "maxpool" else "nnc_avgpool2d"
        stage_call = (
            "_nnc_pipeline_stage_nchw_tile_with_origin_maxpool"
            if root_family == "maxpool"
            else "_nnc_pipeline_stage_nchw_tile_with_origin"
        )

        helper_lines = [
            f"static void {helper_name}(void) {{",
            f"    Tensor* _nnc_group_input = &{input_symbol};",
            f"    Tensor* _nnc_group_terminal = &{output_symbol};",
            f"    uint8_t* _nnc_input_stage = (uint8_t*)({input_stage_expr});",
            f"    uint8_t* _nnc_output_stage = (uint8_t*)({output_stage_expr});",
            f"    for (int64_t _nnc_n = 0; _nnc_n < {output_symbol}.shape[0]; ++_nnc_n) {{",
            f"        for (int64_t _nnc_h0 = 0; _nnc_h0 < {output_symbol}.shape[2]; _nnc_h0 += {int(output_tile_extents[0])}) {{",
            f"            int64_t _nnc_tile_h_0 = _nnc_min_i64({int(output_tile_extents[0])}, {output_symbol}.shape[2] - _nnc_h0);",
            f"            for (int64_t _nnc_w0 = 0; _nnc_w0 < {output_symbol}.shape[3]; _nnc_w0 += {int(output_tile_extents[1])}) {{",
            f"                int64_t _nnc_tile_w_0 = _nnc_min_i64({int(output_tile_extents[1])}, {output_symbol}.shape[3] - _nnc_w0);",
            f"                int64_t _nnc_input_h = {input_h_expr};",
            f"                int64_t _nnc_input_w = {input_w_expr};",
            f"                {stage_call}(_nnc_group_input, _nnc_input_stage, _nnc_n, {input_origin_h_expr}, {input_origin_w_expr}, _nnc_input_h, _nnc_input_w);",
            f"                int64_t _nnc_input_shape[4] = {{1, {input_symbol}.shape[1], _nnc_input_h, _nnc_input_w}};",
            f"                int64_t _nnc_output_shape[4] = {{1, {output_symbol}.shape[1], _nnc_tile_h_0, _nnc_tile_w_0}};",
            "                Tensor _nnc_input = {",
            "                    .data = _nnc_input_stage,",
            f"                    .dtype = {input_symbol}.dtype,",
            "                    .shape = _nnc_input_shape,",
            "                    .ndim = 4,",
            f"                    .nbytes = _nnc_pipeline_tile_nbytes(&{input_symbol}, _nnc_input_h, _nnc_input_w),",
            "                };",
            "                Tensor _nnc_output = {",
            "                    .data = _nnc_output_stage,",
            f"                    .dtype = {output_symbol}.dtype,",
            "                    .shape = _nnc_output_shape,",
            "                    .ndim = 4,",
            f"                    .nbytes = _nnc_pipeline_tile_nbytes(&{output_symbol}, _nnc_tile_h_0, _nnc_tile_w_0),",
            "                };",
            f"                {pool_call}(&_nnc_input, &_nnc_output, {kernel_h}, {kernel_w}, {stride_h}, {stride_w}, 0, 0);",
            "                _nnc_pipeline_commit_nchw_tile(_nnc_output_stage, _nnc_group_terminal, _nnc_n, _nnc_h0, _nnc_w0, _nnc_tile_h_0, _nnc_tile_w_0);",
            "            }",
            "        }",
            "    }",
            "}",
            "",
        ]
        return helper_lines, required_fast_memory

    def _align_c_buffer_size(self, size_bytes: int) -> int:
        alignment = 16
        return ((max(size_bytes, 1) + alignment - 1) // alignment) * alignment

    def _scheduled_tile_streaming_internal_tensor_names(
        self,
        ctx: CompileContext,
    ) -> set[str]:
        metadata = self._scheduled_tile_streaming_metadata(ctx)
        return set(metadata["internal_tensor_names"])

    def _scheduled_tile_streaming_metadata(
        self,
        ctx: CompileContext,
    ) -> dict[str, Any]:
        schedule_problem = ctx.pipeline_schedule_problem
        schedule_result = ctx.pipeline_schedule_result
        if schedule_problem is None or schedule_result is None or not schedule_result.feasible:
            return {"internal_tensor_names": set(), "streamed_node_names": set()}
        runtime = self._build_pipeline_parallel_runtime_metadata(
            ctx,
            schedule_problem=schedule_problem,
            schedule_result=schedule_result,
            scheduled_plan=self._get_scheduled_memory_plan(ctx),
        )
        if runtime is None:
            return {"internal_tensor_names": set(), "streamed_node_names": set()}
        plan = self._build_scheduled_tile_streaming_plan(
            ctx,
            {"parallel_runtime": runtime},
        )
        streamed_node_names = {
            node_name
            for group in plan.get("groups", ())
            for node_name in group.get("node_names", ())
        }
        return {
            "internal_tensor_names": set(plan.get("internal_tensor_names", set())),
            "streamed_node_names": streamed_node_names,
        }

    def _scheduled_step_requires_home_execution(
        self,
        ctx: CompileContext,
        node_name: str,
    ) -> bool:
        execution_plans = ctx.metadata.get("node_execution_plans")
        if not isinstance(execution_plans, dict) or not execution_plans:
            return False
        plan = execution_plans.get(node_name)
        if plan is None:
            return False
        if getattr(plan, "op_family", None) in {"gemm"}:
            return False
        if node_name in self._scheduled_tile_streaming_metadata(ctx)["streamed_node_names"]:
            return False
        return self._node_execution_plan_uses_tiled_storage(plan)

    def _node_execution_plan_uses_tiled_storage(self, plan: Any) -> bool:
        return (
            bool(getattr(plan, "tile_axes", ()))
            or any(
                bool(access.tile_region.logical_extents)
                for access in getattr(plan, "input_accesses", ())
            )
            or any(
                bool(access.tile_region.logical_extents)
                for access in getattr(plan, "output_accesses", ())
            )
        )

    def _collect_scheduled_tile_streaming_execution_groups(
        self,
        ctx: CompileContext,
        execution_plans: dict[str, Any],
    ) -> list[dict[str, Any]]:
        groups = list(self._collect_tile_aware_execution_groups(ctx, execution_plans))
        visited_node_names = {
            node_name
            for group in groups
            for node_name in group.get("node_names", ())
        }
        region_sizes = ctx.metadata.get("node_execution_plan_region_sizes", {})
        for node in ctx.graph.topological_sort():
            if node.name in visited_node_names:
                continue
            plan = execution_plans.get(node.name)
            if plan is None:
                continue
            if getattr(plan, "op_family", None) not in {"maxpool", "average_pool", "global_average_pool"}:
                continue
            if node.name not in region_sizes:
                continue
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                continue
            groups.append(
                {
                    "node_names": [node.name],
                    "external_inputs": [node.inputs[0]],
                    "fast_tensors": [],
                    "static_tensors": [node.outputs[0]],
                }
            )
            visited_node_names.add(node.name)
        return groups

    def _scheduled_tile_streaming_required_fast_memory(
        self,
        ctx: CompileContext,
    ) -> int:
        schedule_problem = ctx.pipeline_schedule_problem
        schedule_result = ctx.pipeline_schedule_result
        if schedule_problem is None or schedule_result is None or not schedule_result.feasible:
            return 0
        runtime = self._build_pipeline_parallel_runtime_metadata(
            ctx,
            schedule_problem=schedule_problem,
            schedule_result=schedule_result,
            scheduled_plan=self._get_scheduled_memory_plan(ctx),
        )
        if runtime is None:
            return 0
        plan = self._build_scheduled_tile_streaming_plan(
            ctx,
            {"parallel_runtime": runtime},
        )
        return int(plan.get("required_fast_memory", 0) or 0)

    def _find_parallel_runtime_step(
        self,
        step_records_by_node: dict[str, list[dict[str, Any]]],
        node_name: str,
        step_kind: str,
    ) -> dict[str, Any] | None:
        for step in step_records_by_node.get(node_name, ()):
            if str(step.get("step_kind", "")) == step_kind:
                return step
        return None

    def _find_value_record(
        self,
        records: tuple[dict[str, Any], ...],
        *,
        tensor_name: str,
        staged_only: bool = False,
    ) -> dict[str, Any] | None:
        for record in records:
            if str(record.get("graph_tensor_name", "")) != tensor_name:
                continue
            if staged_only and not bool(record.get("is_staged")):
                continue
            return dict(record)
        return None

    def _record_buffer_expr(self, record: dict[str, Any]) -> str | None:
        fast_expr = record.get("fast_expr")
        if isinstance(fast_expr, str) and fast_expr:
            return fast_expr
        if bool(record.get("is_staged")):
            return f"{record['storage_symbol']}_buffer"
        return None

    def _sanitize_c_ident(self, value: str) -> str:
        chars = []
        for char in value:
            if char.isalnum():
                chars.append(char)
            else:
                chars.append("_")
        return "".join(chars)

    def _render_scheduled_tile_streaming_helper(
        self,
        ctx: CompileContext,
        *,
        execution_group: dict[str, Any],
        node_by_name: dict[str, Node],
        execution_plans: dict[str, Any],
        step_records_by_node: dict[str, list[dict[str, Any]]],
        conv_input_record: dict[str, Any],
        conv_output_record: dict[str, Any],
        extra_stage_record: dict[str, Any],
        primary_buffer_offset: int,
        secondary_buffer_offset: int,
    ) -> list[str]:
        node_names = tuple(execution_group.get("node_names", ()))
        if not node_names:
            return []
        root_node = node_by_name[node_names[0]]
        root_plan = execution_plans[node_names[0]]
        output_access = root_plan.output_accesses[0]
        output_tile_extents = tuple(output_access.tile_region.logical_extents)
        if len(output_tile_extents) != 2:
            return []
        input_access = root_plan.input_accesses[0]
        input_tile_extents = tuple(input_access.tile_region.logical_extents)
        if len(input_tile_extents) != 2:
            return []

        conv_input_stage_expr = f"_nnc_fast_pool + {int(primary_buffer_offset)}"
        conv_output_stage_expr = f"_nnc_fast_pool + {int(secondary_buffer_offset)}"
        extra_stage_expr = f"_nnc_fast_pool + {int(primary_buffer_offset)}"

        root_output_symbol = ctx.tensor_symbols.get(root_node.outputs[0], root_node.outputs[0])
        final_node = node_by_name[node_names[-1]]
        final_output_symbol = ctx.tensor_symbols.get(final_node.outputs[0], final_node.outputs[0])
        final_compute_step = self._find_parallel_runtime_step(
            step_records_by_node,
            final_node.name,
            "compute",
        )
        if final_compute_step is None:
            return []
        final_output_record = self._find_value_record(
            tuple(final_compute_step.get("output_value_records", ())),
            tensor_name=final_node.outputs[0],
            staged_only=True,
        )
        final_output_stage_expr = f"_nnc_fast_pool + {int(primary_buffer_offset)}"
        if final_output_record is None:
            return []

        helper_name = f"nnc_pipeline_tile_stream_{self._sanitize_c_ident(node_names[0])}"
        input_symbol = ctx.tensor_symbols.get(root_node.inputs[0], root_node.inputs[0])
        weight_symbol = ctx.tensor_symbols.get(root_node.inputs[1], root_node.inputs[1])
        bias_expr = "NULL"
        if len(root_node.inputs) >= 3:
            bias_symbol = ctx.tensor_symbols.get(root_node.inputs[2], root_node.inputs[2])
            bias_expr = f"&{bias_symbol}"

        kernel_shape = root_node.attrs.get("kernel_shape", [1, 1])
        strides = root_node.attrs.get("strides", [1, 1])
        pads = root_node.attrs.get("pads", [0, 0, 0, 0])
        kernel_h = int(kernel_shape[0]) if len(kernel_shape) >= 1 else 1
        kernel_w = int(kernel_shape[1]) if len(kernel_shape) >= 2 else kernel_h
        stride_h = int(strides[0]) if len(strides) >= 1 else 1
        stride_w = int(strides[1]) if len(strides) >= 2 else stride_h
        pad_h = int(pads[0]) if len(pads) >= 1 else 0
        pad_w = int(pads[1]) if len(pads) >= 2 else pad_h

        follower_lines: list[str] = []
        current_stage_tensor_name = root_node.outputs[0]
        current_stage_expr = conv_output_stage_expr
        for index, node_name in enumerate(node_names[1:], start=1):
            follower = node_by_name[node_name]
            follower_plan = execution_plans[node_name]
            follower_output_access = follower_plan.output_accesses[0]
            follower_tile_extents = tuple(follower_output_access.tile_region.logical_extents)
            if len(follower_tile_extents) != 2:
                return []
            tile_h_expr = f"_nnc_tile_h_{index}"
            tile_w_expr = f"_nnc_tile_w_{index}"
            follower_lines.extend(
                [
                    f"            int64_t {tile_h_expr} = _nnc_min_i64({follower_tile_extents[0]}, {ctx.tensor_symbols.get(follower.outputs[0], follower.outputs[0])}.shape[2] - _nnc_h0);",
                    f"            int64_t {tile_w_expr} = _nnc_min_i64({follower_tile_extents[1]}, {ctx.tensor_symbols.get(follower.outputs[0], follower.outputs[0])}.shape[3] - _nnc_w0);",
                ]
            )
            lhs_symbol = f"_nnc_group_{self._sanitize_c_ident(follower.inputs[0])}"
            rhs_symbol = f"_nnc_group_{self._sanitize_c_ident(follower.inputs[1])}" if len(follower.inputs) > 1 else None
            out_symbol = f"_nnc_group_{self._sanitize_c_ident(follower.outputs[0])}"
            follower_lines.extend(
                [
                    f"            int64_t {lhs_symbol}_shape[4] = {{1, {root_output_symbol}.shape[1], {tile_h_expr}, {tile_w_expr}}};",
                    f"            Tensor {lhs_symbol} = {{",
                    f"                .data = {current_stage_expr},",
                    f"                .dtype = {root_output_symbol}.dtype,",
                    f"                .shape = {lhs_symbol}_shape,",
                    "                .ndim = 4,",
                    f"                .nbytes = _nnc_pipeline_tile_nbytes(&{root_output_symbol}, {tile_h_expr}, {tile_w_expr}),",
                    "            };",
                ]
            )
            follower_output_expr = final_output_stage_expr
            follower_output_symbol = ctx.tensor_symbols.get(follower.outputs[0], follower.outputs[0])
            if len(follower.inputs) > 1:
                other_input_name = next(
                    input_name for input_name in follower.inputs if input_name != current_stage_tensor_name
                )
                other_input_symbol = ctx.tensor_symbols.get(other_input_name, other_input_name)
                follower_lines.extend(
                    [
                        f"            _nnc_pipeline_stage_nchw_tile(&{other_input_symbol}, {extra_stage_expr}, _nnc_n, _nnc_h0, _nnc_w0, {tile_h_expr}, {tile_w_expr});",
                        f"            int64_t {rhs_symbol}_shape[4] = {{1, {other_input_symbol}.shape[1], {tile_h_expr}, {tile_w_expr}}};",
                        f"            Tensor {rhs_symbol} = {{",
                        f"                .data = {extra_stage_expr},",
                        f"                .dtype = {other_input_symbol}.dtype,",
                        f"                .shape = {rhs_symbol}_shape,",
                        "                .ndim = 4,",
                        f"                .nbytes = _nnc_pipeline_tile_nbytes(&{other_input_symbol}, {tile_h_expr}, {tile_w_expr}),",
                        "            };",
                    ]
                )
            follower_lines.extend(
                [
                    f"            int64_t {out_symbol}_shape[4] = {{1, {follower_output_symbol}.shape[1], {tile_h_expr}, {tile_w_expr}}};",
                    f"            Tensor {out_symbol} = {{",
                    f"                .data = {follower_output_expr},",
                    f"                .dtype = {follower_output_symbol}.dtype,",
                    f"                .shape = {out_symbol}_shape,",
                    "                .ndim = 4,",
                    f"                .nbytes = _nnc_pipeline_tile_nbytes(&{follower_output_symbol}, {tile_h_expr}, {tile_w_expr}),",
                    "            };",
                ]
            )
            if follower.op_type == OpType.FUSED_ADD_RELU:
                follower_lines.append(f"            nnc_add_relu(&{lhs_symbol}, &{rhs_symbol}, &{out_symbol});")
            elif follower.op_type == OpType.ADD:
                follower_lines.append(f"            nnc_add(&{lhs_symbol}, &{rhs_symbol}, &{out_symbol});")
            elif follower.op_type == OpType.RELU:
                follower_lines.append(f"            nnc_relu(&{lhs_symbol}, &{out_symbol});")
            else:
                return []
            current_stage_tensor_name = follower.outputs[0]
            current_stage_expr = follower_output_expr

        terminal_tensor_symbol = ctx.tensor_symbols.get(current_stage_tensor_name, current_stage_tensor_name)
        terminal_tensor = ctx.graph.tensors.get(current_stage_tensor_name)
        if terminal_tensor is None:
            return []
        root_conv_entry = "nnc_conv_relu" if root_node.op_type == OpType.FUSED_CONV_RELU else "nnc_conv"
        helper_lines = [
            f"static void {helper_name}(void) {{",
            f"    Tensor* _nnc_group_input = &{input_symbol};",
            f"    Tensor* _nnc_group_weight = &{weight_symbol};",
            f"    Tensor* _nnc_group_terminal = &{terminal_tensor_symbol};",
            f"    uint8_t* _nnc_conv_input_stage = (uint8_t*)({conv_input_stage_expr});",
            f"    uint8_t* _nnc_conv_output_stage = (uint8_t*)({conv_output_stage_expr});",
            f"    uint8_t* _nnc_extra_stage = (uint8_t*)({extra_stage_expr});",
            f"    uint8_t* _nnc_terminal_stage = (uint8_t*)({final_output_stage_expr});",
            f"    for (int64_t _nnc_n = 0; _nnc_n < {input_symbol}.shape[0]; ++_nnc_n) {{",
            f"        for (int64_t _nnc_h0 = 0; _nnc_h0 < {root_output_symbol}.shape[2]; _nnc_h0 += {output_tile_extents[0]}) {{",
            f"            int64_t _nnc_tile_h_0 = _nnc_min_i64({output_tile_extents[0]}, {root_output_symbol}.shape[2] - _nnc_h0);",
            f"            for (int64_t _nnc_w0 = 0; _nnc_w0 < {root_output_symbol}.shape[3]; _nnc_w0 += {output_tile_extents[1]}) {{",
            f"                int64_t _nnc_tile_w_0 = _nnc_min_i64({output_tile_extents[1]}, {root_output_symbol}.shape[3] - _nnc_w0);",
            f"                int64_t _nnc_input_h = (_nnc_tile_h_0 - 1) * {stride_h} + {kernel_h};",
            f"                int64_t _nnc_input_w = (_nnc_tile_w_0 - 1) * {stride_w} + {kernel_w};",
            f"                _nnc_pipeline_stage_nchw_tile_with_origin(_nnc_group_input, _nnc_conv_input_stage, _nnc_n, _nnc_h0 * {stride_h} - {pad_h}, _nnc_w0 * {stride_w} - {pad_w}, _nnc_input_h, _nnc_input_w);",
            f"                int64_t _nnc_conv_input_shape[4] = {{1, {input_symbol}.shape[1], _nnc_input_h, _nnc_input_w}};",
            f"                int64_t _nnc_conv_output_shape[4] = {{1, {root_output_symbol}.shape[1], _nnc_tile_h_0, _nnc_tile_w_0}};",
            "                Tensor _nnc_conv_input = {",
            "                    .data = _nnc_conv_input_stage,",
            f"                    .dtype = {input_symbol}.dtype,",
            "                    .shape = _nnc_conv_input_shape,",
            "                    .ndim = 4,",
            f"                    .nbytes = _nnc_pipeline_tile_nbytes(&{input_symbol}, _nnc_input_h, _nnc_input_w),",
            "                };",
            "                Tensor _nnc_conv_output = {",
            "                    .data = _nnc_conv_output_stage,",
            f"                    .dtype = {root_output_symbol}.dtype,",
            "                    .shape = _nnc_conv_output_shape,",
            "                    .ndim = 4,",
            f"                    .nbytes = _nnc_pipeline_tile_nbytes(&{root_output_symbol}, _nnc_tile_h_0, _nnc_tile_w_0),",
            "                };",
            f"                {root_conv_entry}(&_nnc_conv_input, _nnc_group_weight, {bias_expr}, &_nnc_conv_output, {kernel_h}, {kernel_w}, {stride_h}, {stride_w}, 0, 0);",
        ]
        helper_lines.extend(follower_lines)
        commit_src_expr = (
            "_nnc_conv_output_stage"
            if len(node_names) == 1
            else "_nnc_terminal_stage"
        )
        commit_tile_index = len(node_names) - 1 if len(node_names) > 1 else 0
        helper_lines.extend(
            [
                f"                _nnc_pipeline_commit_nchw_tile({commit_src_expr}, _nnc_group_terminal, _nnc_n, _nnc_h0, _nnc_w0, _nnc_tile_h_{commit_tile_index}, _nnc_tile_w_{commit_tile_index});",
                "            }",
                "        }",
                "    }",
                "}",
                "",
            ]
        )
        return helper_lines

    def _should_use_scheduled_home_execution(self, ctx: CompileContext) -> bool:
        execution_plans = ctx.metadata.get("node_execution_plans", {})
        if not isinstance(execution_plans, dict):
            return False
        return any(
            bool(getattr(plan, "tile_axes", ()))
            or any(
                bool(access.tile_region.logical_extents)
                for access in getattr(plan, "input_accesses", ())
            )
            or any(
                bool(access.tile_region.logical_extents)
                for access in getattr(plan, "output_accesses", ())
            )
            for plan in execution_plans.values()
        )

    def _build_scheduled_transfer_body_lines(
        self,
        ctx: CompileContext,
        transfer_point: Any,
    ) -> list[str]:
        fast_expr = f"_nnc_fast_pool + {int(transfer_point.fast_offset)}"
        slow_expr = f"_nnc_slow_pool + {int(transfer_point.slow_offset)}"
        size_bytes = int(transfer_point.size_bytes)
        tensor_value_name = transfer_point.resident_value_name or transfer_point.value_name
        graph_tensor_name = self._resolve_schedule_value_graph_tensor_name(
            ctx,
            str(tensor_value_name),
        )
        tensor_symbol = None
        if graph_tensor_name is not None:
            tensor_symbol = ctx.tensor_symbols.get(graph_tensor_name, graph_tensor_name)

        transfer_kind = getattr(getattr(transfer_point, "transfer_kind", None), "value", "")
        custom_body_lines: list[str] = [f"/* {transfer_kind} */"]
        if transfer_kind == "spill_dma":
            if tensor_symbol is not None:
                custom_body_lines.append(f"{tensor_symbol}.data = {fast_expr};")
            custom_body_lines.append(
                f"memcpy({slow_expr}, {fast_expr}, {size_bytes});"
            )
        elif transfer_kind == "reload_dma":
            custom_body_lines.append(
                f"memcpy({fast_expr}, {slow_expr}, {size_bytes});"
            )
            if tensor_symbol is not None:
                custom_body_lines.append(f"{tensor_symbol}.data = {fast_expr};")
        else:
            return []

        return custom_body_lines

    def _get_scheduled_transfer_points_for_node(
        self,
        scheduled_plan: Any,
        *,
        before_node_name: str | None = None,
        after_node_name: str | None = None,
        transfer_kind: str | None = None,
    ) -> list[Any]:
        transfer_points: list[Any] = []
        for transfer_point in tuple(getattr(scheduled_plan, "transfer_points", ())):
            kind_value = getattr(getattr(transfer_point, "transfer_kind", None), "value", "")
            if transfer_kind is not None and kind_value != transfer_kind:
                continue
            if before_node_name is not None and getattr(transfer_point, "before_node_name", None) != before_node_name:
                continue
            if after_node_name is not None and getattr(transfer_point, "after_node_name", None) != after_node_name:
                continue
            transfer_points.append(transfer_point)
        transfer_points.sort(
            key=lambda item: (
                int(getattr(item, "start_time", 0)),
                int(getattr(item, "end_time", 0)),
                str(getattr(item, "step_id", "")),
            )
        )
        return transfer_points

    def _augment_parallel_runtime_for_unified_spill(
        self,
        ctx: CompileContext,
        plan: MemoryAllocationPlan,
        pipeline_codegen_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._has_parallel_runtime(pipeline_codegen_metadata):
            return pipeline_codegen_metadata

        cloned = self._clone_pipeline_codegen_metadata(pipeline_codegen_metadata)
        runtime = cloned.get("parallel_runtime")
        if not isinstance(runtime, dict):
            return pipeline_codegen_metadata

        steps_by_id = {
            str(step["step_id"]): step
            for step in tuple(runtime.get("steps", ()))
        }
        custom_declarations = list(runtime.get("custom_declarations", ()))
        seen_declarations = set(custom_declarations)

        def add_declaration(line: str) -> None:
            if line in seen_declarations:
                return
            seen_declarations.add(line)
            custom_declarations.append(line)

        def get_saved_var(tensor_symbol: str) -> str:
            name = self._parallel_tensor_saved_data_name(tensor_symbol)
            add_declaration(f"static void* {name} = NULL;")
            return name

        def get_output_spill_points(node_idx: int) -> dict[str, SpillPoint]:
            return {
                point.tensor_name: point
                for point in plan.get_spill_points_after(node_idx)
            }

        def get_move_points_at(node_idx: int) -> list[MovePoint]:
            return sorted(
                plan.get_move_points_at(node_idx),
                key=lambda move: move.from_offset,
            )

        def get_internal_storage_expr(tensor_name: str) -> str | None:
            alloc = plan.tensor_allocations.get(tensor_name)
            if alloc is None:
                return None
            pool_name = "_nnc_slow_pool" if alloc.is_spilled else "_nnc_fast_pool"
            return f"{pool_name} + {alloc.offset}"

        def get_saved_input_binding_name(tensor_name: str) -> str:
            symbol = ctx.tensor_symbols.get(tensor_name, tensor_name)
            return f"_nnc_bound_input_{symbol}"

        input_first_use: dict[str, int] = {}
        liveness_map = ctx.metadata.get("tensor_liveness", {})
        for tensor_name in ctx.graph.inputs:
            liveness = liveness_map.get(tensor_name)
            if liveness is None or not liveness.use_positions:
                continue
            input_first_use[tensor_name] = liveness.use_positions[0]

        nodes = ctx.graph.topological_sort()
        for node_idx, node in enumerate(nodes):
            if node.op_type == OpType.CONSTANT:
                continue

            func_name = ctx.node_symbols.get(node.name, node.name)
            spill_points_after = get_output_spill_points(node_idx)
            move_points_at = get_move_points_at(node_idx)
            move_points_by_tensor = {
                move.tensor_name: move for move in move_points_at
            }
            reloads_before = plan.get_reload_points_before(node_idx)
            inputs_need_reload: dict[str, Union[ReloadPoint, SpillPoint]] = {
                rp.tensor_name: rp for rp in reloads_before
            }

            additional_inputs: list[tuple[str, TensorAllocation]] = []
            for input_name in node.inputs:
                if input_name in inputs_need_reload:
                    continue
                alloc = plan.tensor_allocations.get(input_name)
                if alloc is None or not alloc.is_spilled:
                    continue
                existing_reload = next(
                    (rp for rp in plan.reload_points if rp.tensor_name == input_name),
                    None,
                )
                slow_offset = existing_reload.from_slow_offset if existing_reload else 0
                additional_inputs.append((input_name, alloc))
                inputs_need_reload[input_name] = ReloadPoint(
                    tensor_name=input_name,
                    before_node=node.name,
                    before_node_idx=node_idx,
                    from_slow_offset=slow_offset,
                    to_buffer_id=alloc.buffer_id,
                    to_fast_offset=0,
                    size=alloc.size,
                    reload_slot_id=-1,
                )

            current_slot = max(
                [
                    rp.reload_slot_id
                    for rp in inputs_need_reload.values()
                    if isinstance(rp, ReloadPoint)
                ],
                default=-1,
            ) + 1
            for input_name, _ in additional_inputs:
                rp = inputs_need_reload[input_name]
                if isinstance(rp, ReloadPoint):
                    rp.reload_slot_id = current_slot
                current_slot += 1

            outputs_need_temp: dict[str, Union[ReloadPoint, SpillPoint]] = {}
            for output_name in node.outputs:
                spill_point = spill_points_after.get(output_name)
                if spill_point is not None:
                    outputs_need_temp[output_name] = spill_point
                    continue

                alloc = plan.tensor_allocations.get(output_name)
                if alloc is None or not alloc.is_spilled:
                    continue
                existing_spill = next(
                    (sp for sp in plan.spill_points if sp.tensor_name == output_name),
                    None,
                )
                slow_offset = existing_spill.to_slow_offset if existing_spill else 0
                outputs_need_temp[output_name] = SpillPoint(
                    tensor_name=output_name,
                    after_node=node.name,
                    after_node_idx=node_idx,
                    from_buffer_id=alloc.buffer_id,
                    from_fast_offset=0,
                    to_slow_offset=slow_offset,
                    size=alloc.size,
                )

            output_slot_ids: dict[str, int] = {}
            output_slot_idx = len(inputs_need_reload)
            for output_name in node.outputs:
                if output_name not in outputs_need_temp:
                    continue
                output_slot_ids[output_name] = output_slot_idx
                output_slot_idx += 1

            dma_in_lines: list[str] = []
            moved_first_use_inputs = [
                input_name
                for input_name in node.inputs
                if input_first_use.get(input_name) == node_idx
                and input_name in move_points_by_tensor
            ]
            for input_name in moved_first_use_inputs:
                alloc = plan.tensor_allocations.get(input_name)
                move = move_points_by_tensor[input_name]
                if alloc is None:
                    continue
                tensor_symbol = ctx.tensor_symbols.get(input_name, input_name)
                from_expr = f"_nnc_fast_pool + {move.from_offset}"
                dma_in_lines.extend(
                    [
                        f"memcpy({from_expr}, {get_saved_input_binding_name(input_name)}, {alloc.size});",
                        f"{tensor_symbol}.data = {from_expr};",
                    ]
                )

            for move in move_points_at:
                tensor_symbol = ctx.tensor_symbols.get(move.tensor_name, move.tensor_name)
                to_expr = f"_nnc_fast_pool + {move.to_offset}"
                from_expr = f"_nnc_fast_pool + {move.from_offset}"
                dma_in_lines.extend(
                    [
                        f"memmove({to_expr}, {from_expr}, {move.size});",
                        f"{tensor_symbol}.data = {to_expr};",
                    ]
                )

            for input_name in node.inputs:
                if input_first_use.get(input_name) != node_idx:
                    continue
                if input_name in move_points_by_tensor:
                    continue
                alloc = plan.tensor_allocations.get(input_name)
                target_expr = get_internal_storage_expr(input_name)
                if alloc is None or target_expr is None:
                    continue
                tensor_symbol = ctx.tensor_symbols.get(input_name, input_name)
                dma_in_lines.extend(
                    [
                        f"memcpy({target_expr}, {get_saved_input_binding_name(input_name)}, {alloc.size});",
                        f"{tensor_symbol}.data = {target_expr};",
                    ]
                )

            for input_name in node.inputs:
                if input_name not in inputs_need_reload:
                    continue
                rp = inputs_need_reload[input_name]
                if not isinstance(rp, ReloadPoint):
                    continue
                tensor_symbol = ctx.tensor_symbols.get(input_name, input_name)
                saved_var = get_saved_var(tensor_symbol)
                dma_in_lines.extend(
                    [
                        f"{saved_var} = {tensor_symbol}.data;",
                        f"memcpy(_nnc_reload_buffer_{rp.reload_slot_id},",
                        f"       _nnc_slow_pool + {rp.from_slow_offset}, {rp.size});",
                        f"{tensor_symbol}.data = _nnc_reload_buffer_{rp.reload_slot_id};",
                    ]
                )

            compute_lines: list[str] = []
            for output_name in node.outputs:
                if output_name not in outputs_need_temp:
                    continue
                tensor_symbol = ctx.tensor_symbols.get(output_name, output_name)
                saved_var = get_saved_var(tensor_symbol)
                slot_id = output_slot_ids[output_name]
                compute_lines.extend(
                    [
                        f"{saved_var} = {tensor_symbol}.data;",
                        f"{tensor_symbol}.data = _nnc_reload_buffer_{slot_id};",
                    ]
                )

            compute_lines.append(f"{func_name}_body();")

            for input_name in node.inputs:
                if input_name not in inputs_need_reload:
                    continue
                tensor_symbol = ctx.tensor_symbols.get(input_name, input_name)
                saved_var = get_saved_var(tensor_symbol)
                compute_lines.extend(
                    [
                        f"if ({saved_var} != NULL) {{",
                        f"    {tensor_symbol}.data = {saved_var};",
                        "}",
                    ]
                )

            dma_out_lines: list[str] = []
            for output_name in node.outputs:
                if output_name not in outputs_need_temp:
                    continue
                tensor_symbol = ctx.tensor_symbols.get(output_name, output_name)
                saved_var = get_saved_var(tensor_symbol)
                slot_id = output_slot_ids[output_name]
                spill_value = outputs_need_temp[output_name]
                if isinstance(spill_value, SpillPoint):
                    dma_out_lines.extend(
                        [
                            f"memcpy(_nnc_slow_pool + {spill_value.to_slow_offset},",
                            f"       _nnc_reload_buffer_{slot_id}, {spill_value.size});",
                        ]
                    )
                dma_out_lines.extend(
                    [
                        f"if ({saved_var} != NULL) {{",
                        f"    {tensor_symbol}.data = {saved_var};",
                        "}",
                    ]
                )

            dma_in_step = steps_by_id.get(f"{node.name}.dma_in")
            if dma_in_step is not None and dma_in_lines:
                dma_in_step["custom_body_lines"] = tuple(dma_in_lines)

            compute_step = steps_by_id.get(f"{node.name}.compute")
            if compute_step is not None:
                compute_step["custom_body_lines"] = tuple(compute_lines)

            dma_out_step = steps_by_id.get(f"{node.name}.dma_out")
            if dma_out_step is not None and dma_out_lines:
                dma_out_step["custom_body_lines"] = tuple(dma_out_lines)

        runtime["custom_declarations"] = tuple(custom_declarations)
        return cloned

    def _collect_parallel_step_value_records(
        self,
        ctx: CompileContext,
        value_names: tuple[str, ...],
        *,
        step_id: str,
        scheduled_plan: Any | None = None,
    ) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for value_name in value_names:
            graph_tensor_name = self._resolve_schedule_value_graph_tensor_name(
                ctx,
                value_name,
            )
            if graph_tensor_name is None:
                continue
            tensor = ctx.graph.tensors.get(graph_tensor_name)
            if tensor is None:
                continue
            symbol = ctx.tensor_symbols.get(graph_tensor_name, graph_tensor_name)
            key = (value_name, symbol)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            home_tier = self._resolve_schedule_value_home_tier(ctx, value_name)
            fast_allocation = self._resolve_scheduled_fast_allocation_for_step(
                ctx,
                scheduled_plan,
                value_name=value_name,
                step_id=step_id,
            )
            fast_expr = None
            if fast_allocation is not None:
                fast_expr = f"_nnc_fast_pool + {int(getattr(fast_allocation, 'offset', 0))}"
            storage_symbol = self._parallel_value_storage_name(value_name)
            scheduled_value = self._build_schedule_value_map(ctx).get(value_name)
            size_bytes = int(getattr(scheduled_value, "size_bytes", 0))
            if size_bytes <= 0:
                size_bytes = int(tensor.byte_size())
            records.append(
                {
                    "value_name": value_name,
                    "graph_tensor_name": graph_tensor_name,
                    "tensor_symbol": symbol,
                    "is_staged": value_name.startswith("sram|node|"),
                    "storage_symbol": storage_symbol,
                    "saved_data_symbol": f"{storage_symbol}_saved_data",
                    "size_bytes": max(size_bytes, 1),
                    "home_tier": home_tier,
                    "needs_restore": home_tier in {"input", "const", "slow"},
                    "fast_expr": fast_expr,
                }
            )
        return tuple(records)

    def _append_parallel_runtime_includes(
        self,
        lines: list[str],
        pipeline_codegen_metadata: dict[str, Any],
    ) -> None:
        """Append runtime headers needed by the parallel scheduler executor."""
        if not self._has_parallel_runtime(pipeline_codegen_metadata):
            return
        lines.extend(
            [
                "#ifndef _GNU_SOURCE",
                "#define _GNU_SOURCE",
                "#endif",
                "#include <string.h>",
                "#include <pthread.h>",
                "#include <stdint.h>",
                "#include <unistd.h>",
                "#if defined(__linux__)",
                "#include <sched.h>",
                "#endif",
                "",
            ]
        )

    def _render_parallel_runtime_block(
        self,
        pipeline_codegen_metadata: dict[str, Any],
    ) -> list[str]:
        """Render C helper code for dependency-driven 4-resource parallel execution."""
        runtime = pipeline_codegen_metadata.get("parallel_runtime")
        if not isinstance(runtime, dict) or runtime.get("enabled") is not True:
            return []

        steps = tuple(runtime.get("steps", ()))
        predecessor_indices = tuple(runtime.get("predecessor_indices", ()))
        successor_indices = tuple(runtime.get("successor_indices", ()))
        step_count = len(steps)
        if step_count == 0:
            return []

        dep_counts: list[int] = []
        for deps in predecessor_indices:
            deps_list = [int(dep) for dep in deps]
            dep_counts.append(len(deps_list))

        succ_offsets: list[int] = []
        succ_counts: list[int] = []
        succ_flat: list[int] = []
        succ_offset = 0
        for succs in successor_indices:
            succ_list = [int(succ) for succ in succs]
            succ_offsets.append(succ_offset)
            succ_counts.append(len(succ_list))
            succ_flat.extend(succ_list)
            succ_offset += len(succ_list)
        if not succ_flat:
            succ_flat = [-1]

        lines = [
            "/* Pipeline parallel runtime */",
            "#define NNC_PIPELINE_WORKER_COUNT 4",
            f"#define NNC_PIPELINE_STEP_COUNT {step_count}",
            f"#define NNC_PIPELINE_SUCC_INDEX_COUNT {len(succ_flat)}",
            "",
            "enum {",
            "    NNC_PIPE_RES_DMA = 0,",
            "    NNC_PIPE_RES_SHAPE = 1,",
            "    NNC_PIPE_RES_MATMUL = 2,",
            "    NNC_PIPE_RES_OTHER = 3,",
            "};",
            "",
            "typedef void (*NncPipelineStepFn)(void);",
            "typedef struct NncPipelineStepDesc {",
            "    int resource_kind;",
            "    int start_time;",
            "    int end_time;",
            "    NncPipelineStepFn invoke;",
            "} NncPipelineStepDesc;",
            "",
            "typedef struct NncPipelineWorkerArg {",
            "    int resource_kind;",
            "    int worker_index;",
            "} NncPipelineWorkerArg;",
            "",
            "static const NncPipelineStepDesc _nnc_pipeline_steps[NNC_PIPELINE_STEP_COUNT] = {",
        ]
        for step in steps:
            step_id = self._sanitize_c_comment_text(str(step["step_id"]))
            invoke_symbol = step["invoke_symbol"] if step["invoke_symbol"] is not None else "NULL"
            lines.append(
                f"    /* {step_id} */ {{{int(step['resource_index'])}, {int(step['start_time'])}, {int(step['end_time'])}, {invoke_symbol}}},"
            )
        lines.extend(
            [
                "};",
                "",
                "static const int _nnc_pipeline_dep_counts[NNC_PIPELINE_STEP_COUNT] = {"
                + ", ".join(str(value) for value in dep_counts)
                + "};",
                "static const int _nnc_pipeline_succ_offsets[NNC_PIPELINE_STEP_COUNT] = {"
                + ", ".join(str(value) for value in succ_offsets)
                + "};",
                "static const int _nnc_pipeline_succ_counts[NNC_PIPELINE_STEP_COUNT] = {"
                + ", ".join(str(value) for value in succ_counts)
                + "};",
                "static const int _nnc_pipeline_succ_indices[NNC_PIPELINE_SUCC_INDEX_COUNT] = {"
                + ", ".join(str(value) for value in succ_flat)
                + "};",
                "",
                "static int _nnc_pipeline_remaining_deps[NNC_PIPELINE_STEP_COUNT];",
                "static unsigned char _nnc_pipeline_started[NNC_PIPELINE_STEP_COUNT];",
                "static int _nnc_pipeline_completed_count = 0;",
                "static int _nnc_pipeline_start_flag = 0;",
                "static pthread_mutex_t _nnc_pipeline_mutex = PTHREAD_MUTEX_INITIALIZER;",
                "static pthread_cond_t _nnc_pipeline_cond = PTHREAD_COND_INITIALIZER;",
                "",
                "static void _nnc_pipeline_bind_worker_core(int worker_index) {",
                "#if defined(__linux__)",
                "    long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);",
                "    int cpu_index = worker_index;",
                "    if (cpu_count > 0) {",
                "        cpu_index = worker_index % (int)cpu_count;",
                "    }",
                "    cpu_set_t cpu_set;",
                "    CPU_ZERO(&cpu_set);",
                "    CPU_SET(cpu_index, &cpu_set);",
                "    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);",
                "#else",
                "    (void)worker_index;",
                "#endif",
                "}",
                "",
                "static int _nnc_pipeline_pick_ready_step_locked(int resource_kind) {",
                "    for (int step_index = 0; step_index < NNC_PIPELINE_STEP_COUNT; ++step_index) {",
                "        if (_nnc_pipeline_steps[step_index].resource_kind != resource_kind) {",
                "            continue;",
                "        }",
                "        if (_nnc_pipeline_started[step_index] != 0) {",
                "            continue;",
                "        }",
                "        if (_nnc_pipeline_remaining_deps[step_index] != 0) {",
                "            continue;",
                "        }",
                "        _nnc_pipeline_started[step_index] = 1;",
                "        return step_index;",
                "    }",
                "    return -1;",
                "}",
                "",
                "static void* _nnc_pipeline_worker_main(void* worker_arg_ptr) {",
                "    NncPipelineWorkerArg* worker_arg = (NncPipelineWorkerArg*)worker_arg_ptr;",
                "    _nnc_pipeline_bind_worker_core(worker_arg->worker_index);",
                "",
                "    pthread_mutex_lock(&_nnc_pipeline_mutex);",
                "    while (_nnc_pipeline_start_flag == 0 && _nnc_pipeline_completed_count < NNC_PIPELINE_STEP_COUNT) {",
                "        pthread_cond_wait(&_nnc_pipeline_cond, &_nnc_pipeline_mutex);",
                "    }",
                "    if (_nnc_pipeline_completed_count >= NNC_PIPELINE_STEP_COUNT) {",
                "        pthread_mutex_unlock(&_nnc_pipeline_mutex);",
                "        return NULL;",
                "    }",
                "    pthread_mutex_unlock(&_nnc_pipeline_mutex);",
                "",
                "    for (;;) {",
                "        int step_index = -1;",
                "",
                "        pthread_mutex_lock(&_nnc_pipeline_mutex);",
                "        while (step_index < 0) {",
                "            if (_nnc_pipeline_completed_count >= NNC_PIPELINE_STEP_COUNT) {",
                "                pthread_mutex_unlock(&_nnc_pipeline_mutex);",
                "                return NULL;",
                "            }",
                "            step_index = _nnc_pipeline_pick_ready_step_locked(worker_arg->resource_kind);",
                "            if (step_index < 0) {",
                "                pthread_cond_wait(&_nnc_pipeline_cond, &_nnc_pipeline_mutex);",
                "            }",
                "        }",
                "        pthread_mutex_unlock(&_nnc_pipeline_mutex);",
                "",
                "        NncPipelineStepFn invoke = _nnc_pipeline_steps[step_index].invoke;",
                "        if (invoke != NULL) {",
                "            invoke();",
                "        }",
                "",
                "        pthread_mutex_lock(&_nnc_pipeline_mutex);",
                "        int succ_offset = _nnc_pipeline_succ_offsets[step_index];",
                "        int succ_count = _nnc_pipeline_succ_counts[step_index];",
                "        for (int succ_i = 0; succ_i < succ_count; ++succ_i) {",
                "            int succ_index = _nnc_pipeline_succ_indices[succ_offset + succ_i];",
                "            if (_nnc_pipeline_remaining_deps[succ_index] > 0) {",
                "                _nnc_pipeline_remaining_deps[succ_index] -= 1;",
                "            }",
                "        }",
                "        _nnc_pipeline_completed_count += 1;",
                "        pthread_cond_broadcast(&_nnc_pipeline_cond);",
                "        pthread_mutex_unlock(&_nnc_pipeline_mutex);",
                "    }",
                "}",
                "",
                "static void nnc_pipeline_run_parallel(void) {",
                "    pthread_t workers[NNC_PIPELINE_WORKER_COUNT];",
                "    NncPipelineWorkerArg worker_args[NNC_PIPELINE_WORKER_COUNT] = {",
                "        {NNC_PIPE_RES_DMA, 0},",
                "        {NNC_PIPE_RES_SHAPE, 1},",
                "        {NNC_PIPE_RES_MATMUL, 2},",
                "        {NNC_PIPE_RES_OTHER, 3},",
                "    };",
                "",
                "    pthread_mutex_lock(&_nnc_pipeline_mutex);",
                "    _nnc_pipeline_completed_count = 0;",
                "    _nnc_pipeline_start_flag = 0;",
                "    for (int step_index = 0; step_index < NNC_PIPELINE_STEP_COUNT; ++step_index) {",
                "        _nnc_pipeline_remaining_deps[step_index] = _nnc_pipeline_dep_counts[step_index];",
                "        _nnc_pipeline_started[step_index] = 0;",
                "    }",
                "    pthread_mutex_unlock(&_nnc_pipeline_mutex);",
                "",
                "    int launched_count = 0;",
                "    for (int worker_index = 0; worker_index < NNC_PIPELINE_WORKER_COUNT; ++worker_index) {",
                "        if (pthread_create(",
                "                &workers[worker_index],",
                "                NULL,",
                "                _nnc_pipeline_worker_main,",
                "                &worker_args[worker_index]",
                "            ) != 0) {",
                "            break;",
                "        }",
                "        launched_count += 1;",
                "    }",
                "",
                "    if (launched_count != NNC_PIPELINE_WORKER_COUNT) {",
                "        pthread_mutex_lock(&_nnc_pipeline_mutex);",
                "        _nnc_pipeline_completed_count = NNC_PIPELINE_STEP_COUNT;",
                "        pthread_cond_broadcast(&_nnc_pipeline_cond);",
                "        pthread_mutex_unlock(&_nnc_pipeline_mutex);",
                "        for (int worker_index = 0; worker_index < launched_count; ++worker_index) {",
                "            pthread_join(workers[worker_index], NULL);",
                "        }",
                "        for (int step_index = 0; step_index < NNC_PIPELINE_STEP_COUNT; ++step_index) {",
                "            NncPipelineStepFn invoke = _nnc_pipeline_steps[step_index].invoke;",
                "            if (invoke != NULL) {",
                "                invoke();",
                "            }",
                "        }",
                "        return;",
                "    }",
                "",
                "    pthread_mutex_lock(&_nnc_pipeline_mutex);",
                "    _nnc_pipeline_start_flag = 1;",
                "    pthread_cond_broadcast(&_nnc_pipeline_cond);",
                "    pthread_mutex_unlock(&_nnc_pipeline_mutex);",
                "",
                "    for (int worker_index = 0; worker_index < NNC_PIPELINE_WORKER_COUNT; ++worker_index) {",
                "        pthread_join(workers[worker_index], NULL);",
                "    }",
                "}",
            ]
        )
        return lines

    def _render_parallel_step_helper_block(
        self,
        pipeline_codegen_metadata: dict[str, Any],
    ) -> list[str]:
        """Render per-step helper functions so each worker runs concrete C code."""
        runtime = pipeline_codegen_metadata.get("parallel_runtime")
        if not isinstance(runtime, dict) or runtime.get("enabled") is not True:
            return []

        steps = tuple(runtime.get("steps", ()))
        if not steps:
            return []

        declared_value_records: dict[str, dict[str, Any]] = {}
        used_buffer_symbols: set[str] = set()
        used_saved_data_symbols: set[str] = set()

        def remember_record(record: dict[str, Any]) -> None:
            declared_value_records.setdefault(str(record["value_name"]), dict(record))

        for step in steps:
            if step.get("custom_body_lines") is not None:
                continue
            input_value_records = tuple(step.get("input_value_records", ()))
            output_value_records = tuple(step.get("output_value_records", ()))
            step_kind = str(step.get("step_kind", ""))

            if step_kind == "dma_in":
                staged_outputs = [
                    record
                    for record in output_value_records
                    if bool(record.get("is_staged")) or record.get("fast_expr")
                ]
                direct_inputs_by_tensor = {
                    str(record["graph_tensor_name"]): record
                    for record in input_value_records
                    if not bool(record.get("is_staged"))
                }
                for record in staged_outputs:
                    if not bool(record.get("is_staged")):
                        continue
                    source_record = direct_inputs_by_tensor.get(str(record["graph_tensor_name"]))
                    if source_record is None:
                        continue
                    remember_record(record)
                    used_buffer_symbols.add(f"{record['storage_symbol']}_buffer")
                    used_saved_data_symbols.add(str(record["saved_data_symbol"]))
            elif step_kind == "dma_out":
                bridged_outputs = [
                    record
                    for record in output_value_records
                    if record.get("fast_expr")
                ]
                staged_inputs_by_tensor = {
                    str(record["graph_tensor_name"]): record
                    for record in input_value_records
                    if bool(record.get("is_staged")) or record.get("fast_expr")
                }
                for record in bridged_outputs:
                    source_record = staged_inputs_by_tensor.get(str(record["graph_tensor_name"]))
                    if source_record is None or not bool(source_record.get("is_staged")):
                        continue
                    remember_record(source_record)
                    used_buffer_symbols.add(f"{source_record['storage_symbol']}_buffer")
            elif step_kind == "compute":
                buffered_inputs = [
                    record for record in input_value_records if bool(record.get("is_staged"))
                ]
                fast_inputs = [
                    record
                    for record in input_value_records
                    if not bool(record.get("is_staged")) and record.get("fast_expr")
                ]
                buffered_outputs = [
                    record for record in output_value_records if bool(record.get("is_staged"))
                ]
                for record in buffered_inputs:
                    remember_record(record)
                    used_buffer_symbols.add(f"{record['storage_symbol']}_buffer")
                for record in fast_inputs:
                    if not bool(record.get("needs_restore")):
                        continue
                    remember_record(record)
                    used_saved_data_symbols.add(str(record["saved_data_symbol"]))
                for record in buffered_outputs:
                    remember_record(record)
                    used_buffer_symbols.add(f"{record['storage_symbol']}_buffer")
                    used_saved_data_symbols.add(str(record["saved_data_symbol"]))

        lines = [
            "/* Pipeline step helper functions */",
            "static volatile uint64_t _nnc_pipeline_touch_sink = 0;",
            "",
            "static int64_t _nnc_min_i64(int64_t lhs, int64_t rhs) {",
            "    return lhs < rhs ? lhs : rhs;",
            "}",
            "",
            "static int64_t _nnc_pipeline_dtype_size(nnc_dtype_t dtype) {",
            "    switch (dtype) {",
            "        case NNC_DTYPE_FLOAT16: return 2;",
            "        case NNC_DTYPE_INT8: return 1;",
            "        case NNC_DTYPE_UINT8: return 1;",
            "        case NNC_DTYPE_BOOL: return 1;",
            "        case NNC_DTYPE_INT64: return 8;",
            "        case NNC_DTYPE_FLOAT32:",
            "        case NNC_DTYPE_INT32:",
            "        default: return 4;",
            "    }",
            "}",
            "",
            "static int64_t _nnc_pipeline_tile_nbytes(const Tensor* tensor, int64_t tile_h, int64_t tile_w) {",
            "    if (tensor == NULL || tensor->shape == NULL || tensor->ndim != 4) {",
            "        return 0;",
            "    }",
            "    return _nnc_pipeline_dtype_size(tensor->dtype) * tensor->shape[1] * tile_h * tile_w;",
            "}",
            "",
            "static void _nnc_pipeline_stage_nchw_tile_with_origin(",
            "    const Tensor* src,",
            "    void* dst,",
            "    int64_t batch_index,",
            "    int64_t h_origin,",
            "    int64_t w_origin,",
            "    int64_t tile_h,",
            "    int64_t tile_w",
            ") {",
            "    if (src == NULL || dst == NULL || src->data == NULL || src->shape == NULL || src->ndim != 4) {",
            "        return;",
            "    }",
            "    int64_t elem_size = _nnc_pipeline_dtype_size(src->dtype);",
            "    int64_t channels = src->shape[1];",
            "    int64_t full_h = src->shape[2];",
            "    int64_t full_w = src->shape[3];",
            "    uint8_t* dst_bytes = (uint8_t*)dst;",
            "    const uint8_t* src_bytes = (const uint8_t*)src->data;",
            "    memset(dst_bytes, 0, (size_t)(elem_size * channels * tile_h * tile_w));",
            "    for (int64_t channel = 0; channel < channels; ++channel) {",
            "        for (int64_t tile_row = 0; tile_row < tile_h; ++tile_row) {",
            "            int64_t src_row = h_origin + tile_row;",
            "            if (src_row < 0 || src_row >= full_h) {",
            "                continue;",
            "            }",
            "            for (int64_t tile_col = 0; tile_col < tile_w; ++tile_col) {",
            "                int64_t src_col = w_origin + tile_col;",
            "                if (src_col < 0 || src_col >= full_w) {",
            "                    continue;",
            "                }",
            "                int64_t src_offset = ((((batch_index * channels) + channel) * full_h + src_row) * full_w + src_col) * elem_size;",
            "                int64_t dst_offset = (((channel * tile_h) + tile_row) * tile_w + tile_col) * elem_size;",
            "                memcpy(dst_bytes + dst_offset, src_bytes + src_offset, (size_t)elem_size);",
            "            }",
            "        }",
            "    }",
            "}",
            "",
            "static void _nnc_pipeline_stage_nchw_tile_with_origin_maxpool(",
            "    const Tensor* src,",
            "    void* dst,",
            "    int64_t batch_index,",
            "    int64_t h_origin,",
            "    int64_t w_origin,",
            "    int64_t tile_h,",
            "    int64_t tile_w",
            ") {",
            "    if (src == NULL || dst == NULL || src->data == NULL || src->shape == NULL || src->ndim != 4) {",
            "        return;",
            "    }",
            "    int64_t elem_size = _nnc_pipeline_dtype_size(src->dtype);",
            "    int64_t channels = src->shape[1];",
            "    int64_t full_h = src->shape[2];",
            "    int64_t full_w = src->shape[3];",
            "    uint8_t* dst_bytes = (uint8_t*)dst;",
            "    const uint8_t* src_bytes = (const uint8_t*)src->data;",
            "    int64_t total_elems = channels * tile_h * tile_w;",
            "    if (src->dtype == NNC_DTYPE_FLOAT32) {",
            "        float* dst_f32 = (float*)dst;",
            "        for (int64_t index = 0; index < total_elems; ++index) {",
            "            dst_f32[index] = -3.402823466e+38F;",
            "        }",
            "    } else {",
            "        memset(dst_bytes, 0, (size_t)(elem_size * total_elems));",
            "    }",
            "    for (int64_t channel = 0; channel < channels; ++channel) {",
            "        for (int64_t tile_row = 0; tile_row < tile_h; ++tile_row) {",
            "            int64_t src_row = h_origin + tile_row;",
            "            if (src_row < 0 || src_row >= full_h) {",
            "                continue;",
            "            }",
            "            for (int64_t tile_col = 0; tile_col < tile_w; ++tile_col) {",
            "                int64_t src_col = w_origin + tile_col;",
            "                if (src_col < 0 || src_col >= full_w) {",
            "                    continue;",
            "                }",
            "                int64_t src_offset = ((((batch_index * channels) + channel) * full_h + src_row) * full_w + src_col) * elem_size;",
            "                int64_t dst_offset = (((channel * tile_h) + tile_row) * tile_w + tile_col) * elem_size;",
            "                memcpy(dst_bytes + dst_offset, src_bytes + src_offset, (size_t)elem_size);",
            "            }",
            "        }",
            "    }",
            "}",
            "",
            "static void _nnc_pipeline_stage_nchw_tile(",
            "    const Tensor* src,",
            "    void* dst,",
            "    int64_t batch_index,",
            "    int64_t h_origin,",
            "    int64_t w_origin,",
            "    int64_t tile_h,",
            "    int64_t tile_w",
            ") {",
            "    _nnc_pipeline_stage_nchw_tile_with_origin(src, dst, batch_index, h_origin, w_origin, tile_h, tile_w);",
            "}",
            "",
            "static void _nnc_pipeline_commit_nchw_tile(",
            "    const void* src,",
            "    Tensor* dst,",
            "    int64_t batch_index,",
            "    int64_t h_origin,",
            "    int64_t w_origin,",
            "    int64_t tile_h,",
            "    int64_t tile_w",
            ") {",
            "    if (src == NULL || dst == NULL || dst->data == NULL || dst->shape == NULL || dst->ndim != 4) {",
            "        return;",
            "    }",
            "    int64_t elem_size = _nnc_pipeline_dtype_size(dst->dtype);",
            "    int64_t channels = dst->shape[1];",
            "    int64_t full_h = dst->shape[2];",
            "    int64_t full_w = dst->shape[3];",
            "    const uint8_t* src_bytes = (const uint8_t*)src;",
            "    uint8_t* dst_bytes = (uint8_t*)dst->data;",
            "    for (int64_t channel = 0; channel < channels; ++channel) {",
            "        for (int64_t tile_row = 0; tile_row < tile_h; ++tile_row) {",
            "            int64_t dst_row = h_origin + tile_row;",
            "            if (dst_row < 0 || dst_row >= full_h) {",
            "                continue;",
            "            }",
            "            for (int64_t tile_col = 0; tile_col < tile_w; ++tile_col) {",
            "                int64_t dst_col = w_origin + tile_col;",
            "                if (dst_col < 0 || dst_col >= full_w) {",
            "                    continue;",
            "                }",
            "                int64_t src_offset = (((channel * tile_h) + tile_row) * tile_w + tile_col) * elem_size;",
            "                int64_t dst_offset = ((((batch_index * channels) + channel) * full_h + dst_row) * full_w + dst_col) * elem_size;",
            "                memcpy(dst_bytes + dst_offset, src_bytes + src_offset, (size_t)elem_size);",
            "            }",
            "        }",
            "    }",
            "}",
            "",
            "static void _nnc_pipeline_touch_tensor_read(const Tensor* tensor) {",
            "    if (tensor == NULL || tensor->data == NULL || tensor->nbytes <= 0) {",
            "        return;",
            "    }",
            "    const unsigned char* bytes = (const unsigned char*)tensor->data;",
            "    uint64_t accum = 0;",
            "    for (int64_t index = 0; index < tensor->nbytes; ++index) {",
            "        accum += (uint64_t)bytes[index];",
            "    }",
            "    _nnc_pipeline_touch_sink ^= accum;",
            "}",
            "",
            "static void _nnc_pipeline_touch_tensor_write(Tensor* tensor) {",
            "    if (tensor == NULL || tensor->data == NULL || tensor->nbytes <= 0) {",
            "        return;",
            "    }",
            "    volatile unsigned char* bytes = (volatile unsigned char*)tensor->data;",
            "    for (int64_t index = 0; index < tensor->nbytes; ++index) {",
            "        bytes[index] = bytes[index];",
            "    }",
            "}",
            "",
            "static void _nnc_pipeline_shape_touch_tensor(const Tensor* tensor) {",
            "    if (tensor == NULL || tensor->shape == NULL || tensor->ndim <= 0) {",
            "        return;",
            "    }",
            "    int64_t shape_accum = 0;",
            "    for (int32_t dim = 0; dim < tensor->ndim; ++dim) {",
            "        shape_accum += tensor->shape[dim] * (dim + 1);",
            "    }",
            "    _nnc_pipeline_touch_sink ^= (uint64_t)shape_accum;",
            "}",
            "",
        ]

        for value_name in sorted(declared_value_records):
            record = declared_value_records[value_name]
            storage_symbol = str(record["storage_symbol"])
            saved_data_symbol = str(record["saved_data_symbol"])
            size_bytes = int(record["size_bytes"])
            if f"{storage_symbol}_buffer" in used_buffer_symbols:
                lines.append(f"static unsigned char {storage_symbol}_buffer[{size_bytes}];")
            if saved_data_symbol in used_saved_data_symbols:
                lines.append(f"static void* {saved_data_symbol} = NULL;")
        if declared_value_records:
            lines.append("")

        for declaration in tuple(runtime.get("custom_declarations", ())):
            lines.append(str(declaration))
        if runtime.get("custom_declarations"):
            lines.append("")

        for step in steps:
            invoke_symbol = step.get("invoke_symbol")
            if not isinstance(invoke_symbol, str) or not invoke_symbol:
                continue
            input_tensor_symbols = tuple(step.get("input_tensor_symbols", ()))
            output_tensor_symbols = tuple(step.get("output_tensor_symbols", ()))
            input_value_records = tuple(step.get("input_value_records", ()))
            output_value_records = tuple(step.get("output_value_records", ()))
            step_kind = str(step.get("step_kind", ""))
            invoke_node = bool(step.get("invoke_node"))
            node_symbol = str(step.get("node_symbol", ""))

            lines.append(f"static void {invoke_symbol}(void) {{")
            custom_body_lines = step.get("custom_body_lines")
            if custom_body_lines is not None:
                for body_line in tuple(custom_body_lines):
                    if body_line:
                        lines.append(f"    {body_line}")
                    else:
                        lines.append("")
            elif step_kind == "dma_in":
                staged_outputs = [
                    record
                    for record in output_value_records
                    if bool(record.get("is_staged")) or record.get("fast_expr")
                ]
                direct_inputs_by_tensor = {
                    str(record["graph_tensor_name"]): record
                    for record in input_value_records
                    if not bool(record.get("is_staged"))
                }
                if staged_outputs:
                    for record in staged_outputs:
                        tensor_symbol = str(record["tensor_symbol"])
                        storage_symbol = str(record["storage_symbol"])
                        size_bytes = int(record["size_bytes"])
                        source_record = direct_inputs_by_tensor.get(str(record["graph_tensor_name"]))
                        fast_expr = record.get("fast_expr")
                        if bool(record.get("is_staged")):
                            if source_record is not None:
                                lines.append(
                                    f"    {record['saved_data_symbol']} = {tensor_symbol}.data;"
                                )
                                lines.append(
                                    f"    memcpy({storage_symbol}_buffer, {tensor_symbol}.data, {size_bytes});"
                                )
                                lines.append(
                                    f"    {tensor_symbol}.data = {storage_symbol}_buffer;"
                                )
                            else:
                                lines.append(
                                    f"    _nnc_pipeline_touch_tensor_read(&{tensor_symbol});"
                                )
                        elif fast_expr:
                            source_fast_expr = (
                                source_record.get("fast_expr")
                                if source_record is not None
                                else None
                            )
                            if source_record is not None:
                                if source_fast_expr:
                                    if source_fast_expr != fast_expr:
                                        lines.append(
                                            f"    memcpy({fast_expr}, {source_fast_expr}, {size_bytes});"
                                        )
                                else:
                                    lines.append(
                                        f"    memcpy({fast_expr}, {tensor_symbol}.data, {size_bytes});"
                                    )
                        else:
                            lines.append(
                                f"    _nnc_pipeline_touch_tensor_read(&{tensor_symbol});"
                            )
                elif input_tensor_symbols:
                    for symbol in input_tensor_symbols:
                        lines.append(f"    _nnc_pipeline_touch_tensor_read(&{symbol});")
                else:
                    lines.append("    _nnc_pipeline_touch_sink ^= 1u;")
            elif step_kind == "dma_out":
                bridged_outputs = [
                    record
                    for record in output_value_records
                    if record.get("fast_expr")
                ]
                staged_inputs = [
                    record
                    for record in input_value_records
                    if bool(record.get("is_staged")) or record.get("fast_expr")
                ]
                if bridged_outputs:
                    staged_inputs_by_tensor = {
                        str(record["graph_tensor_name"]): record
                        for record in staged_inputs
                    }
                    for record in bridged_outputs:
                        tensor_symbol = str(record["tensor_symbol"])
                        size_bytes = int(record["size_bytes"])
                        target_fast_expr = str(record["fast_expr"])
                        source_record = staged_inputs_by_tensor.get(str(record["graph_tensor_name"]))
                        if source_record is not None and bool(source_record.get("is_staged")):
                            source_storage_symbol = str(source_record["storage_symbol"])
                            lines.append(
                                f"    memcpy({target_fast_expr}, {source_storage_symbol}_buffer, {size_bytes});"
                            )
                        else:
                            source_fast_expr = (
                                source_record.get("fast_expr")
                                if source_record is not None
                                else None
                            )
                            if source_fast_expr and source_fast_expr != target_fast_expr:
                                lines.append(
                                    f"    memcpy({target_fast_expr}, {source_fast_expr}, {size_bytes});"
                                )
                        lines.append(f"    {tensor_symbol}.data = {target_fast_expr};")
                elif staged_inputs:
                    fast_staged_inputs = [
                        record
                        for record in staged_inputs
                        if not bool(record.get("is_staged")) and record.get("fast_expr")
                    ]
                    for record in fast_staged_inputs:
                        tensor_symbol = str(record["tensor_symbol"])
                        lines.append(f"    {tensor_symbol}.data = {record['fast_expr']};")
                else:
                    dma_out_symbols = output_tensor_symbols or input_tensor_symbols
                    if dma_out_symbols:
                        for symbol in dma_out_symbols:
                            lines.append(f"    _nnc_pipeline_touch_tensor_write(&{symbol});")
                    else:
                        lines.append("    _nnc_pipeline_touch_sink ^= 1u;")
            elif step_kind == "shape_prep":
                touched = False
                for symbol in (*input_tensor_symbols, *output_tensor_symbols):
                    lines.append(f"    _nnc_pipeline_shape_touch_tensor(&{symbol});")
                    touched = True
                if not touched:
                    lines.append("    _nnc_pipeline_touch_sink ^= 1u;")
                if invoke_node and node_symbol:
                    lines.append(f"    {node_symbol}();")
            elif step_kind == "compute":
                buffered_inputs = [
                    record for record in input_value_records if bool(record.get("is_staged"))
                ]
                fast_inputs = [
                    record
                    for record in input_value_records
                    if not bool(record.get("is_staged")) and record.get("fast_expr")
                ]
                buffered_outputs = [
                    record for record in output_value_records if bool(record.get("is_staged"))
                ]
                fast_outputs = [
                    record
                    for record in output_value_records
                    if not bool(record.get("is_staged")) and record.get("fast_expr")
                ]
                for record in buffered_inputs:
                    tensor_symbol = str(record["tensor_symbol"])
                    storage_symbol = str(record["storage_symbol"])
                    lines.append(f"    {tensor_symbol}.data = {storage_symbol}_buffer;")
                for record in fast_inputs:
                    tensor_symbol = str(record["tensor_symbol"])
                    saved_data_symbol = str(record["saved_data_symbol"])
                    if bool(record.get("needs_restore")):
                        lines.append(f"    {saved_data_symbol} = {tensor_symbol}.data;")
                    lines.append(f"    {tensor_symbol}.data = {record['fast_expr']};")
                for record in buffered_outputs:
                    tensor_symbol = str(record["tensor_symbol"])
                    storage_symbol = str(record["storage_symbol"])
                    lines.append(f"    {record['saved_data_symbol']} = {tensor_symbol}.data;")
                    lines.append(f"    {tensor_symbol}.data = {storage_symbol}_buffer;")
                for record in fast_outputs:
                    tensor_symbol = str(record["tensor_symbol"])
                    lines.append(f"    {tensor_symbol}.data = {record['fast_expr']};")
                if node_symbol:
                    lines.append(f"    {node_symbol}();")
                else:
                    lines.append("    _nnc_pipeline_touch_sink ^= 1u;")
                for record in fast_inputs:
                    tensor_symbol = str(record["tensor_symbol"])
                    if bool(record.get("needs_restore")):
                        lines.append(f"    if ({record['saved_data_symbol']} != NULL) {{")
                        lines.append(
                            f"        {tensor_symbol}.data = {record['saved_data_symbol']};"
                        )
                        lines.append("    }")
            else:
                if node_symbol:
                    lines.append(f"    {node_symbol}();")
                else:
                    lines.append("    _nnc_pipeline_touch_sink ^= 1u;")
            lines.append("}")
            lines.append("")

        return lines

    def _inject_parallel_runtime_into_emitted_source(
        self,
        source: str,
        pipeline_codegen_metadata: dict[str, Any],
    ) -> str:
        """Inject parallel runtime helper and switch nnc_run to dependency-driven execution."""
        if not self._has_parallel_runtime(pipeline_codegen_metadata):
            return source

        include_block = "\n".join(
            [
                "#ifndef _GNU_SOURCE",
                "#define _GNU_SOURCE",
                "#endif",
                "#include <string.h>",
                "#include <pthread.h>",
                "#include <stdint.h>",
                "#include <unistd.h>",
                "#if defined(__linux__)",
                "#include <sched.h>",
                "#endif",
                "",
            ]
        )
        if "#include <pthread.h>" not in source:
            include_anchor = '#include "model.h"\n'
            if include_anchor in source:
                source = source.replace(include_anchor, include_block + include_anchor, 1)

        pool_decl_block = "\n".join(
            [
                "extern uint8_t _nnc_fast_pool[];",
                "extern uint8_t _nnc_slow_pool[];",
                "",
            ]
        )
        if "extern uint8_t _nnc_fast_pool[];" not in source:
            decl_anchor = '#include "nnc_ops.h"\n'
            if decl_anchor in source:
                source = source.replace(decl_anchor, decl_anchor + "\n" + pool_decl_block, 1)
            else:
                include_anchor = '#include "model.h"\n'
                if include_anchor in source:
                    source = source.replace(include_anchor, include_anchor + "\n" + pool_decl_block, 1)

        helper_block = "\n".join(self._render_parallel_step_helper_block(pipeline_codegen_metadata)).strip()
        parallel_block = "\n".join(self._render_parallel_runtime_block(pipeline_codegen_metadata)).strip()
        injected_blocks = "\n\n".join(block for block in (helper_block, parallel_block) if block)
        if injected_blocks:
            main_anchor = "/* Main inference entry point */"
            if main_anchor in source:
                source = source.replace(main_anchor, injected_blocks + "\n\n" + main_anchor, 1)
            else:
                source = source.rstrip() + "\n\n" + injected_blocks + "\n"

        lines = source.split("\n")
        run_start = None
        run_end = None
        for index, line in enumerate(lines):
            if line.strip().startswith("void nnc_run(void) {"):
                run_start = index
                break
        if run_start is None:
            return source

        brace_depth = 0
        for index in range(run_start, len(lines)):
            brace_depth += lines[index].count("{")
            brace_depth -= lines[index].count("}")
            if index > run_start and brace_depth == 0:
                run_end = index
                break
        if run_end is None:
            return source

        new_function_lines = [
            lines[run_start],
            "    nnc_pipeline_run_parallel();",
            "}",
        ]
        rewritten_lines = lines[:run_start] + new_function_lines + lines[run_end + 1 :]
        return "\n".join(rewritten_lines)

    def _build_memory_plan_summary_line(
        self,
        alloc_plan: "MemoryAllocationPlan | None",
        ctx: CompileContext,
        *,
        scheduled_plan: Any | None = None,
    ) -> str:
        """Return a compact summary line for memory planning metadata."""
        if scheduled_plan is not None:
            return "memory_plan_strategy=scheduled_native"
        if alloc_plan is not None:
            return f"memory_plan_strategy={alloc_plan.strategy_name}"
        if "memory_plan" in ctx.metadata:
            return "memory_plan_strategy=legacy"
        return "memory_plan_strategy=none"

    def _build_schedule_value_bindings(
        self,
        ctx: CompileContext,
        alloc_plan: "MemoryAllocationPlan | None",
        schedule_result: Any,
        *,
        scheduled_plan: Any | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Collect SRAM binding details keyed by scheduled value name."""
        value_bindings: dict[str, dict[str, Any]] = {}
        if scheduled_plan is not None:
            for allocation in sorted(
                getattr(scheduled_plan, "fast_allocations", {}).values(),
                key=lambda item: (
                    getattr(item, "start_time", 0),
                    getattr(item, "end_time", 0),
                    getattr(item, "residency_id", ""),
                ),
            ):
                binding = value_bindings.setdefault(allocation.value_name, {})
                binding["offset"] = allocation.offset
                binding["region"] = allocation.residency_id
                binding["source"] = "scheduled_memory_plan"
        if alloc_plan is not None:
            for value_name, allocation in alloc_plan.tensor_allocations.items():
                binding = value_bindings.setdefault(value_name, {})
                binding["offset"] = allocation.offset
                binding["source"] = "memory_plan"
            for region_name, region in alloc_plan.logical_regions.items():
                binding = value_bindings.setdefault(region_name, {})
                binding["offset"] = region.offset
                binding["region"] = region.name
                binding["source"] = "memory_plan"

        for interval in getattr(schedule_result, "sram_intervals", ()):
            binding = value_bindings.get(interval.value_name)
            if binding is not None and binding.get("source") == "memory_plan":
                continue
            binding = value_bindings.setdefault(interval.value_name, {})
            binding["buffer_id"] = interval.buffer_id
            binding["interval"] = (interval.start_time, interval.end_time)
            binding["size_bytes"] = interval.size_bytes
            binding["source"] = "schedule_interval"

        return value_bindings

    def _format_pipeline_step_comment(
        self,
        *,
        scheduled_step: Any,
        problem_step: Any,
        value_bindings: dict[str, dict[str, Any]],
    ) -> str:
        """Return one schedule annotation comment for a lowered step."""
        duration = scheduled_step.end_time - scheduled_step.start_time
        if duration <= 0:
            duration = getattr(problem_step, "duration", 0)

        cost_source = "unknown"
        attrs = getattr(problem_step, "attrs", {})
        if hasattr(attrs, "get") and attrs.get("cost_model") is not None:
            cost_source = str(attrs["cost_model"])

        binding_names = tuple(problem_step.sram_input_names) + tuple(problem_step.sram_output_names)
        binding_parts: list[str] = []
        for value_name in binding_names:
            binding = value_bindings.get(value_name)
            if not binding:
                continue
            base = f"{value_name}@{binding['offset']}" if "offset" in binding else value_name
            details = []
            if "buffer_id" in binding:
                details.append(f"buffer={binding['buffer_id']}")
            if "region" in binding:
                details.append(f"region={binding['region']}")
            if details:
                binding_parts.append(f"{base}[{', '.join(details)}]")
            else:
                binding_parts.append(base)

        parts = [
            f"step_id={scheduled_step.step_id}",
            f"kind={problem_step.step_kind.value}",
            f"resource={scheduled_step.resource_kind.value}",
            f"start={scheduled_step.start_time}",
            f"end={scheduled_step.end_time}",
            f"duration={duration}",
            f"cost_source={self._sanitize_c_comment_text(cost_source)}",
        ]
        if binding_parts:
            parts.append(
                "sram_bindings="
                + self._sanitize_c_comment_text("; ".join(binding_parts))
            )
        return "pipeline step: " + ", ".join(parts)

    def _format_json_mapping(self, value: Any) -> str:
        """Format small JSON-like mappings for comment output."""
        if not isinstance(value, dict):
            try:
                items = list(value.items())
            except AttributeError:
                return ""
        else:
            items = list(value.items())

        if not items:
            return ""
        return self._sanitize_c_comment_text(
            ", ".join(f"{key}={item}" for key, item in sorted(items))
        )

    def _sanitize_c_comment_text(self, value: str) -> str:
        """Avoid terminating generated C comments accidentally."""
        return value.replace("*/", "* /").replace("\n", " ")

    def _append_pipeline_schedule_summary_block(
        self,
        lines: list[str],
        pipeline_codegen_metadata: dict[str, Any],
    ) -> None:
        """Append a labeled pipeline schedule summary comment block."""
        lines.append("/* Pipeline schedule summary")
        for summary_line in pipeline_codegen_metadata.get("summary_lines", ()):
            lines.append(
                f" * {self._sanitize_c_comment_text(str(summary_line))}"
            )
        lines.append(" */")
        lines.append("")

    def _append_pipeline_step_comment_lines(
        self,
        lines: list[str],
        pipeline_codegen_metadata: dict[str, Any],
        node_name: str,
        *,
        indent: str = "",
    ) -> None:
        """Append grouped pipeline step comments for a node."""
        for comment in pipeline_codegen_metadata.get("node_comments", {}).get(node_name, ()):
            lines.append(f"{indent}/* {self._sanitize_c_comment_text(comment)} */")

    def _generate_source(self, ctx: CompileContext) -> str:
        """Generate main source file with spill/reload support."""
        from nnc_py.codegen.c_emitter import CEmitter
        from nnc_py.passes.memory_planning import get_memory_allocation_plan
        from nnc_py.passes.spill import get_spill_plan

        # Check for new MemoryAllocationPlan
        alloc_plan = get_memory_allocation_plan(ctx)
        scheduled_plan = self._get_scheduled_memory_plan(ctx)
        prefer_scheduled_plan = self._prefer_scheduled_memory_plan(ctx, scheduled_plan)
        tile_aware_runtime_plan = self._get_tile_aware_runtime_plan(ctx, alloc_plan)
        pipeline_codegen_metadata = self._build_pipeline_codegen_metadata(
            ctx,
            alloc_plan,
            scheduled_plan=scheduled_plan if prefer_scheduled_plan else None,
        )
        if prefer_scheduled_plan and scheduled_plan is not None:
            pipeline_codegen_metadata = self._augment_parallel_runtime_for_scheduled_spill(
                ctx,
                scheduled_plan,
                pipeline_codegen_metadata,
            )
            if not bool(getattr(scheduled_plan, "transfer_points", ())):
                pipeline_codegen_metadata = self._augment_parallel_runtime_for_scheduled_tile_streaming(
                    ctx,
                    pipeline_codegen_metadata,
                )
                pipeline_codegen_metadata = self._augment_parallel_runtime_for_scheduled_home_execution(
                    ctx,
                    pipeline_codegen_metadata,
                )

        # Check for legacy spill plan
        spill_plan = get_spill_plan(ctx)

        # Determine which plan to use
        used_standard_emitter = False
        if (
            prefer_scheduled_plan
            and self.debug_mode
            and scheduled_plan is not None
            and bool(getattr(scheduled_plan, "transfer_points", ()))
        ):
            source = self._generate_source_with_scheduled_spill(
                ctx,
                scheduled_plan,
                pipeline_codegen_metadata,
            )
        elif prefer_scheduled_plan:
            emitter = CEmitter(
                tile_aware_wrapper_nodes=tile_aware_runtime_plan.get("wrapper_nodes", {}),
                pipeline_schedule_metadata=pipeline_codegen_metadata,
            )
            source = emitter.emit(ctx)
            used_standard_emitter = True
        elif alloc_plan is not None and (alloc_plan.has_spill or bool(alloc_plan.move_points)):
            # Use new MemoryAllocationPlan with reload code generation
            source = self._generate_source_with_unified_spill(
                ctx,
                alloc_plan,
                pipeline_codegen_metadata,
            )
        elif spill_plan is not None and spill_plan.has_overflow:
            # Use legacy spill plan
            source = self._generate_source_with_spill(
                ctx,
                spill_plan,
                pipeline_codegen_metadata,
            )
        else:
            # No overflow, use standard emitter
            emitter = CEmitter(
                tile_aware_wrapper_nodes=tile_aware_runtime_plan.get("wrapper_nodes", {}),
                pipeline_schedule_metadata=pipeline_codegen_metadata,
            )
            source = emitter.emit(ctx)
            used_standard_emitter = True

        if used_standard_emitter:
            source = self._inject_parallel_runtime_into_emitted_source(
                source,
                pipeline_codegen_metadata,
            )

        # Add debug macros if in debug mode
        source = self._add_debug_macros(source)

        # Inject debug code if enabled
        source = self._inject_debug_into_nnc_run(source, ctx)
        return self._append_entry_point_alias(source, ctx)

    def _get_tile_aware_runtime_plan(
        self,
        ctx: CompileContext,
        alloc_plan: "MemoryAllocationPlan | None",
    ) -> dict[str, Any]:
        """Return a conservative tile-aware runtime plan when the graph is safely supported."""
        if alloc_plan is None:
            return {}
        if alloc_plan.strategy_name != "tile_regions_v3":
            return {}
        if not alloc_plan.logical_regions:
            return {}

        execution_plans = ctx.metadata.get("node_execution_plans")
        if not isinstance(execution_plans, dict) or not execution_plans:
            return {}

        region_sizes = ctx.metadata.get("node_execution_plan_region_sizes")
        if not isinstance(region_sizes, dict):
            return {}

        execution_groups = self._collect_tile_aware_execution_groups(ctx, execution_plans)
        if not execution_groups:
            return {}

        tensor_bindings: dict[str, dict[str, Any]] = {}
        wrapper_nodes: dict[str, dict[str, Any]] = {}
        logical_region_names = ", ".join(sorted(alloc_plan.logical_regions))

        for execution_group in execution_groups:
            required_fast_bytes = max(
                (
                    self._tile_aware_tensor_size_bytes(ctx, tensor_name)
                    for tensor_name in execution_group["fast_tensors"]
                ),
                default=0,
            )
            selected_region = self._select_tile_aware_region(alloc_plan, required_fast_bytes)
            if selected_region is None:
                return {}

            group_label = " -> ".join(execution_group["node_names"])
            for node_name in execution_group["node_names"]:
                plan = execution_plans.get(node_name)
                wrapper_nodes[node_name] = {
                    "comment": self._build_tile_aware_wrapper_comment(
                        group_label=group_label,
                        logical_region_names=logical_region_names,
                        plan=plan,
                    )
                }

            for tensor_name in execution_group["external_inputs"]:
                tensor_bindings.setdefault(
                    tensor_name,
                    self._make_tile_aware_external_binding(ctx, tensor_name),
                )
            for tensor_name in execution_group["static_tensors"]:
                tensor_bindings[tensor_name] = self._make_tile_aware_static_binding(ctx, tensor_name)
            for tensor_name in execution_group["fast_tensors"]:
                tensor_bindings[tensor_name] = {
                    "kind": "fast_pool",
                    "offset": selected_region.offset,
                    "region": selected_region.name,
                }

        for tensor_name in ctx.graph.tensors:
            if tensor_name in ctx.graph.constants:
                continue
            tensor_bindings.setdefault(
                tensor_name,
                self._make_tile_aware_external_binding(ctx, tensor_name),
            )

        return {
            "wrapper_nodes": wrapper_nodes,
            "tensor_bindings": tensor_bindings,
        }

    def _build_tile_aware_wrapper_comment(
        self,
        *,
        group_label: str,
        logical_region_names: str,
        plan: Any | None,
    ) -> str:
        """Describe the late physical-layout mapping intent without materializing it."""
        generic_layout = getattr(getattr(plan, "layout_class", None), "value", "unknown")
        target_physical_layout = getattr(plan, "target_physical_layout", None)
        return (
            f"tile-aware wrapper: execution_group={group_label}, "
            f"logical_regions={logical_region_names}, "
            f"generic_blocked_layout={generic_layout}, "
            f"target_physical_layout={target_physical_layout}, "
            "late_physical_mapping=deferred"
        )

    def _collect_tile_aware_execution_groups(
        self,
        ctx: CompileContext,
        execution_plans: dict[str, Any],
    ) -> list[dict[str, Any]]:
        supported_families = {"conv2d", "maxpool"}
        execution_groups: list[dict[str, Any]] = []
        visited_node_names: set[str] = set()
        topo_nodes = ctx.graph.topological_sort()
        node_order_by_name = {
            node.name: index
            for index, node in enumerate(topo_nodes)
        }

        for node in topo_nodes:
            if not node.is_computational():
                continue
            if node.name in visited_node_names:
                continue

            plan = execution_plans.get(node.name)
            if plan is None or getattr(plan, "op_family", None) not in supported_families:
                continue
            if node.name not in ctx.metadata.get("node_execution_plan_region_sizes", {}):
                continue
            if len(node.outputs) != 1:
                continue

            group_nodes = [node]
            flow_tensor_name = node.outputs[0]
            while True:
                successor = self._get_tile_compatible_successor(
                    ctx,
                    flow_tensor_name,
                    execution_plans,
                    visited_node_names,
                    node_order_by_name=node_order_by_name,
                )
                if successor is None:
                    break
                group_nodes.append(successor)
                flow_tensor_name = successor.outputs[0]

            group_node_names = {group_node.name for group_node in group_nodes}
            produced_tensor_names = {
                tensor_name
                for group_node in group_nodes
                for tensor_name in group_node.outputs
            }
            internal_tensor_names = {
                tensor_name
                for tensor_name in produced_tensor_names
                if any(
                    consumer.name in group_node_names
                    for consumer in ctx.graph.get_consumers(tensor_name)
                )
            }

            if any(
                any(
                    consumer.name not in group_node_names
                    for consumer in ctx.graph.get_consumers(tensor_name)
                )
                for tensor_name in internal_tensor_names
            ):
                continue

            external_inputs: list[str] = []
            for group_node in group_nodes:
                for input_name in group_node.inputs:
                    if input_name in produced_tensor_names or input_name in ctx.graph.constants:
                        continue
                    if input_name not in external_inputs:
                        external_inputs.append(input_name)

            fast_tensors = list(internal_tensor_names)
            static_tensors: list[str] = []
            for tensor_name in group_nodes[-1].outputs:
                outside_consumers = [
                    consumer
                    for consumer in ctx.graph.get_consumers(tensor_name)
                    if consumer.name not in group_node_names
                ]
                if outside_consumers:
                    static_tensors.append(tensor_name)
                    continue
                if tensor_name in ctx.graph.outputs or not ctx.graph.get_consumers(tensor_name):
                    fast_tensors.append(tensor_name)
                    continue
                static_tensors.append(tensor_name)

            execution_groups.append(
                {
                    "node_names": [group_node.name for group_node in group_nodes],
                    "external_inputs": external_inputs,
                    "fast_tensors": list(dict.fromkeys(fast_tensors)),
                    "static_tensors": list(dict.fromkeys(static_tensors)),
                }
            )
            visited_node_names.update(group_node_names)

        return execution_groups

    def _get_tile_compatible_successor(
        self,
        ctx: CompileContext,
        flow_tensor_name: str,
        execution_plans: dict[str, Any],
        visited_node_names: set[str],
        *,
        node_order_by_name: dict[str, int],
    ) -> Node | None:
        consumers = [
            consumer
            for consumer in ctx.graph.get_consumers(flow_tensor_name)
            if consumer.is_computational()
        ]
        if len(consumers) != 1:
            return None

        consumer = consumers[0]
        if consumer.name in visited_node_names:
            return None
        if len(consumer.outputs) != 1:
            return None

        if consumer.op_type == OpType.RELU:
            if tuple(consumer.inputs) != (flow_tensor_name,):
                return None
            if not self._tile_aware_tensors_match(ctx, flow_tensor_name, consumer.outputs[0]):
                return None
            return consumer

        if consumer.op_type in {OpType.ADD, OpType.FUSED_ADD_RELU}:
            if len(consumer.inputs) != 2 or flow_tensor_name not in consumer.inputs:
                return None
            other_input_name = next(
                input_name for input_name in consumer.inputs if input_name != flow_tensor_name
            )
            other_producers = [
                producer
                for producer in ctx.graph.get_producers(other_input_name)
                if producer.is_computational() and producer.name != consumer.name
            ]
            if other_producers:
                current_producer = next(
                    iter(ctx.graph.get_producers(flow_tensor_name)),
                    None,
                )
                if current_producer is None:
                    return None
                current_priority = self._tile_aware_group_root_priority(
                    current_producer,
                    node_order_by_name=node_order_by_name,
                )
                best_other_priority = max(
                    self._tile_aware_group_root_priority(
                        producer,
                        node_order_by_name=node_order_by_name,
                    )
                    for producer in other_producers
                )
                if current_priority < best_other_priority:
                    return None
            if not self._tile_aware_tensors_match(ctx, flow_tensor_name, consumer.outputs[0]):
                return None
            if not self._tile_aware_tensors_match(ctx, flow_tensor_name, other_input_name):
                return None
            return consumer

        return None

    def _tile_aware_group_root_priority(
        self,
        node: Node,
        *,
        node_order_by_name: dict[str, int],
    ) -> tuple[int, int]:
        kernel_shape = tuple(int(v) for v in node.attrs.get("kernel_shape", ()) if isinstance(v, int))
        kernel_area = 1
        if kernel_shape:
            for extent in kernel_shape:
                kernel_area *= max(extent, 1)
        else:
            kernel_area = 0
        return (
            kernel_area,
            node_order_by_name.get(node.name, -1),
        )

    def _tile_aware_tensors_match(
        self,
        ctx: CompileContext,
        lhs_tensor_name: str,
        rhs_tensor_name: str,
    ) -> bool:
        lhs_tensor = ctx.graph.tensors.get(lhs_tensor_name)
        rhs_tensor = ctx.graph.tensors.get(rhs_tensor_name)
        if lhs_tensor is None or rhs_tensor is None:
            return False
        if lhs_tensor.dtype != rhs_tensor.dtype:
            return False
        return lhs_tensor.byte_size() == rhs_tensor.byte_size()

    def _tile_aware_tensor_size_bytes(self, ctx: CompileContext, tensor_name: str) -> int:
        tensor = ctx.graph.tensors.get(tensor_name)
        if tensor is None:
            return 0
        return max(0, tensor.byte_size())

    def _select_tile_aware_region(
        self,
        alloc_plan: "MemoryAllocationPlan",
        required_size_bytes: int,
    ) -> Any | None:
        for region in sorted(
            alloc_plan.logical_regions.values(),
            key=lambda logical_region: (logical_region.offset, logical_region.name),
        ):
            if region.size_bytes >= required_size_bytes:
                return region
        return None

    def _make_tile_aware_external_binding(
        self,
        ctx: CompileContext,
        tensor_name: str,
    ) -> dict[str, Any]:
        symbol = ctx.tensor_symbols.get(tensor_name, tensor_name)
        if tensor_name in ctx.graph.inputs:
            return {
                "kind": "input_staging",
                "symbol": f"_nnc_input_buffer_{symbol}",
            }
        return self._make_tile_aware_static_binding(ctx, tensor_name)

    def _make_tile_aware_static_binding(
        self,
        ctx: CompileContext,
        tensor_name: str,
    ) -> dict[str, Any]:
        symbol = ctx.tensor_symbols.get(tensor_name, tensor_name)
        return {
            "kind": "static_buffer",
            "symbol": f"_nnc_tensor_buffer_{symbol}",
        }

    def _build_linear_tensor_fallback(
        self,
        ctx: CompileContext,
    ) -> tuple[dict[str, tuple[str, int]], int]:
        """Build a conservative sequential placement for non-constant tensors."""
        tensor_offsets: dict[str, tuple[str, int]] = {}
        current_offset = 0
        alignment = 16

        for tensor_name, tensor in ctx.graph.tensors.items():
            if tensor_name in ctx.graph.constants:
                continue
            aligned_offset = ((current_offset + alignment - 1) // alignment) * alignment
            tensor_offsets[tensor_name] = ("fast", aligned_offset)
            current_offset = aligned_offset + tensor.byte_size()

        total_size = ((current_offset + alignment - 1) // alignment) * alignment
        return tensor_offsets, total_size

    def _append_entry_point_alias(self, source_code: str, ctx: CompileContext) -> str:
        """Append a public wrapper when the requested entry point is not nnc_run."""
        entry_point = self._get_public_entry_point(ctx)
        if entry_point == "nnc_run":
            return source_code

        return (
            source_code
            + "\n"
            + f"/* Public entry point alias */\nvoid {entry_point}(void) {{\n    nnc_run();\n}}\n"
        )

    def _add_debug_macros(self, source_code: str) -> str:
        """Add debug macro definitions to source code.

        Args:
            source_code: Generated C source code.

        Returns:
            Modified source code with debug macros if debug mode is enabled.
        """
        if not self.debug_mode:
            return source_code

        # Define debug macros - use debug_file for output (declared as extern in test_runner.c)
        debug_macros = """
/* Debug mode: macros for intermediate tensor output */
/* Note: debug_file is defined in test_runner.c as a FILE* */
extern FILE* debug_file;

#ifndef DEBUG_PRINTF
#define DEBUG_PRINTF(fmt, ...) do { \\
    if (debug_file) { \\
        fprintf(debug_file, fmt, ##__VA_ARGS__); \\
    } else { \\
        printf(fmt, ##__VA_ARGS__); \\
    } \\
} while(0)
#endif

#define DEBUG_PRINT_TENSOR_START(name, idx) DEBUG_PRINTF("DEBUG_TENSOR_START %s %d\\n", name, idx)
#define DEBUG_PRINT_SHAPE(ndim) DEBUG_PRINTF("SHAPE %d\\n", ndim)
#define DEBUG_PRINT_DIM(i, val) DEBUG_PRINTF("DIM %d %d\\n", i, val)
#define DEBUG_PRINT_DATA_START() DEBUG_PRINTF("DATA_START\\n")
#define DEBUG_PRINT_VALUE(val) DEBUG_PRINTF("%f\\n", val)
#define DEBUG_PRINT_DATA_END() DEBUG_PRINTF("DATA_END\\n")
#define DEBUG_PRINT_TENSOR_END(name) DEBUG_PRINTF("DEBUG_TENSOR_END %s\\n\\n", name)

"""

        lines = source_code.split("\n")
        insert_at = None
        in_block_comment = False

        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("/*"):
                in_block_comment = True
            if not in_block_comment and line.startswith("#include"):
                insert_at = index + 1
            if in_block_comment and "*/" in stripped:
                in_block_comment = False

        if insert_at is None:
            insert_at = 0
            for index, line in enumerate(lines):
                stripped = line.strip()
                if not stripped or stripped.startswith("/*"):
                    continue
                insert_at = index
                break

        output = lines[:insert_at] + [debug_macros] + lines[insert_at:]
        return "\n".join(output)

    def _generate_source_with_spill(
        self,
        ctx: CompileContext,
        spill_plan: "SpillPlan",
        pipeline_codegen_metadata: dict[str, Any],
    ) -> str:
        """Generate source file with spill/reload wrapper functions."""
        from nnc_py.codegen.c_emitter import CEmitter

        pipeline_codegen_metadata = self._augment_parallel_runtime_for_legacy_spill(
            ctx,
            spill_plan,
            pipeline_codegen_metadata,
        )

        lines = [
            "/* Auto-generated by NNC - DO NOT EDIT */",
            "",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            '#include "model.h"',
            '#include "nnc_ops.h"',
            "",
            "/* External memory pools (defined in tensors.c) */",
            "extern uint8_t _nnc_fast_pool[];",
            "extern uint8_t _nnc_slow_pool[];",
            "",
            "/* Forward declarations for spill/reload functions */",
        ]
        self._append_parallel_runtime_includes(lines, pipeline_codegen_metadata)
        self._append_pipeline_schedule_summary_block(lines, pipeline_codegen_metadata)

        # Generate forward declarations for node functions
        nodes = ctx.graph.topological_sort()
        for node in nodes:
            if node.op_type == OpType.CONSTANT:
                continue
            func_name = ctx.node_symbols.get(node.name, node.name)
            lines.append(f"static void {func_name}_body(void);")

        lines.append("")
        lines.append("/* Spill/Reload wrapper functions */")

        # Generate wrapper functions with spill/reload
        spill_plan_dict = {
            (p.tensor_name, p.after_node): p
            for p in spill_plan.spill_points
        }
        reload_plan_dict = {
            (p.tensor_name, p.before_node): p
            for p in spill_plan.reload_points
        }

        for node in nodes:
            if node.op_type == OpType.CONSTANT:
                continue

            func_name = ctx.node_symbols.get(node.name, node.name)

            # Collect spills after this node
            spills_after = [
                p for p in spill_plan.spill_points if p.after_node == node.name
            ]

            # Collect reloads before this node
            reloads_before = [
                p for p in spill_plan.reload_points if p.before_node == node.name
            ]

            lines.append(f"/* {func_name}: {node.op_type.value} */")
            lines.append(f"static void {func_name}(void) {{")
            self._append_pipeline_step_comment_lines(
                lines,
                pipeline_codegen_metadata,
                node.name,
                indent="    ",
            )

            # Reload before node
            for reload in reloads_before:
                lines.extend([
                    f"    /* Reload {reload.tensor_name} from slow memory */",
                    f"    memcpy(",
                    f"        _nnc_fast_pool + {reload.to_fast_offset},",
                    f"        _nnc_slow_pool + {reload.from_slow_offset},",
                    f"        {reload.size}",
                    f"    );",
                    "",
                ])

            # Call the body
            lines.append(f"    {func_name}_body();")

            # Spill after node
            for spill in spills_after:
                lines.extend([
                    "",
                    f"    /* Spill {spill.tensor_name} to slow memory */",
                    f"    memcpy(",
                    f"        _nnc_slow_pool + {spill.to_slow_offset},",
                    f"        _nnc_fast_pool + {spill.from_fast_offset},",
                    f"        {spill.size}",
                    f"    );",
                ])

            lines.append("}")
            lines.append("")

        lines.append("/* Node body functions */")

        # Generate body functions (using standard emitter)
        emitter = CEmitter(pipeline_schedule_metadata=pipeline_codegen_metadata)
        body_code = emitter.emit(ctx)

        # We need to modify the body_code to rename functions to _body
        # and remove the main entry point
        lines.append(self._process_body_code(body_code, ctx))
        helper_block = self._render_parallel_step_helper_block(pipeline_codegen_metadata)
        runtime_block = self._render_parallel_runtime_block(pipeline_codegen_metadata)
        if helper_block:
            lines.append("")
            lines.extend(helper_block)
        if runtime_block:
            lines.append("")
            lines.extend(runtime_block)

        # Generate main entry point
        lines.append("")
        lines.append("/* Main inference entry point */")
        lines.append("void nnc_run(void) {")

        if self._has_parallel_runtime(pipeline_codegen_metadata):
            lines.append("    nnc_pipeline_run_parallel();")
        else:
            for node in nodes:
                if node.op_type == OpType.CONSTANT:
                    continue
                func_name = ctx.node_symbols.get(node.name, node.name)
                self._append_pipeline_step_comment_lines(
                    lines,
                    pipeline_codegen_metadata,
                    node.name,
                    indent="    ",
                )
                lines.append(f"    {func_name}();")

        lines.append("}")

        return "\n".join(lines)

    def _generate_source_with_scheduled_spill(
        self,
        ctx: CompileContext,
        scheduled_plan: Any,
        pipeline_codegen_metadata: dict[str, Any],
    ) -> str:
        """Generate serial source with scheduled spill/reload wrappers."""
        from nnc_py.codegen.c_emitter import CEmitter

        lines = [
            "/* Auto-generated by NNC - DO NOT EDIT */",
            "",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            '#include "model.h"',
            '#include "nnc_ops.h"',
            "",
            "/* External memory pools (defined in tensors.c) */",
            "extern uint8_t _nnc_fast_pool[];",
            "extern uint8_t _nnc_slow_pool[];",
            "",
        ]
        self._append_pipeline_schedule_summary_block(lines, pipeline_codegen_metadata)

        nodes = ctx.graph.topological_sort()
        for node in nodes:
            if node.op_type == OpType.CONSTANT:
                continue
            func_name = ctx.node_symbols.get(node.name, node.name)
            lines.append(f"static void {func_name}_body(void);")

        lines.append("")
        lines.append("/* Scheduled spill/reload wrapper functions */")

        for node in nodes:
            if node.op_type == OpType.CONSTANT:
                continue

            func_name = ctx.node_symbols.get(node.name, node.name)
            reload_points = self._get_scheduled_transfer_points_for_node(
                scheduled_plan,
                before_node_name=node.name,
                transfer_kind="reload_dma",
            )
            spill_points = self._get_scheduled_transfer_points_for_node(
                scheduled_plan,
                after_node_name=node.name,
                transfer_kind="spill_dma",
            )

            lines.append(f"/* {func_name}: {node.op_type.value} */")
            lines.append(f"static void {func_name}(void) {{")
            self._append_pipeline_step_comment_lines(
                lines,
                pipeline_codegen_metadata,
                node.name,
                indent="    ",
            )

            for transfer_point in reload_points:
                for body_line in self._build_scheduled_transfer_body_lines(ctx, transfer_point):
                    lines.append(f"    {body_line}")

            lines.append(f"    {func_name}_body();")

            for transfer_point in spill_points:
                for body_line in self._build_scheduled_transfer_body_lines(ctx, transfer_point):
                    lines.append(f"    {body_line}")

            lines.append("}")
            lines.append("")

        lines.append("/* Node body functions */")
        emitter = CEmitter(pipeline_schedule_metadata=pipeline_codegen_metadata)
        body_code = emitter.emit(ctx)
        lines.append(self._process_body_code(body_code, ctx))

        lines.append("")
        lines.append("/* Main inference entry point */")
        lines.append("void nnc_run(void) {")
        for node in nodes:
            if node.op_type == OpType.CONSTANT:
                continue
            func_name = ctx.node_symbols.get(node.name, node.name)
            self._append_pipeline_step_comment_lines(
                lines,
                pipeline_codegen_metadata,
                node.name,
                indent="    ",
            )
            lines.append(f"    {func_name}();")
        lines.append("}")

        return "\n".join(lines)

    def _process_body_code(self, body_code: str, ctx: CompileContext) -> str:
        """Process the body code from CEmitter."""
        lines = body_code.split("\n")
        output = []
        skip_main = False

        for line in lines:
            # Skip the includes (we already added them)
            if line.startswith("#include") or line.startswith("/* Auto-generated"):
                continue

            # Skip the main nnc_run function (we'll generate our own)
            if "void nnc_run(void)" in line:
                skip_main = True
                continue

            if skip_main:
                if line.startswith("}"):
                    skip_main = False
                continue

            # Rename functions to _body
            for node_name, func_name in ctx.node_symbols.items():
                if f"void {func_name}(void)" in line:
                    line = line.replace(f"void {func_name}(void)", f"static void {func_name}_body(void)")

            output.append(line)

        return "\n".join(output)

    def _generate_source_with_unified_spill(
        self,
        ctx: CompileContext,
        plan: "MemoryAllocationPlan",
        pipeline_codegen_metadata: dict[str, Any],
    ) -> str:
        """Generate source file with spill/reload wrapper functions using MemoryAllocationPlan.

        The key requirement: ALL operators must execute with inputs/outputs in fast memory.
        When tensors are spilled to slow memory, they must be:
        1. Reloaded to fast memory before operator execution
        2. Used via temp Tensor structs pointing to fast memory
        3. Spilled back to slow memory after operator execution
        """
        from nnc_py.codegen.c_emitter import CEmitter

        pipeline_codegen_metadata = self._augment_parallel_runtime_for_unified_spill(
            ctx,
            plan,
            pipeline_codegen_metadata,
        )

        lines = [
            "/* Auto-generated by NNC - DO NOT EDIT */",
            "",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            '#include "model.h"',
            '#include "nnc_ops.h"',
            "",
            "/* External memory pools (defined in tensors.c) */",
            "extern uint8_t _nnc_fast_pool[];",
            "extern uint8_t _nnc_slow_pool[];",
            "",
        ]
        self._append_parallel_runtime_includes(lines, pipeline_codegen_metadata)
        self._append_pipeline_schedule_summary_block(lines, pipeline_codegen_metadata)

        def get_output_spill_points(node_idx: int) -> dict[str, SpillPoint]:
            return {
                point.tensor_name: point
                for point in plan.get_spill_points_after(node_idx)
            }

        def get_move_points_at(node_idx: int) -> list[MovePoint]:
            return plan.get_move_points_at(node_idx)

        def get_internal_storage_expr(tensor_name: str) -> str | None:
            alloc = plan.tensor_allocations.get(tensor_name)
            if alloc is None:
                return None
            pool_name = "_nnc_slow_pool" if alloc.is_spilled else "_nnc_fast_pool"
            return f"{pool_name} + {alloc.offset}"

        def get_saved_input_binding_name(tensor_name: str) -> str:
            symbol = ctx.tensor_symbols.get(tensor_name, tensor_name)
            return f"_nnc_bound_input_{symbol}"

        input_first_use: dict[str, int] = {}
        liveness_map = ctx.metadata.get("tensor_liveness", {})
        for tensor_name in ctx.graph.inputs:
            liveness = liveness_map.get(tensor_name)
            if liveness is None or not liveness.use_positions:
                continue
            input_first_use[tensor_name] = liveness.use_positions[0]

        for tensor_name in ctx.graph.inputs:
            if tensor_name not in plan.tensor_allocations:
                continue
            lines.append(f"static uint8_t* {get_saved_input_binding_name(tensor_name)} = NULL;")
        if any(tensor_name in plan.tensor_allocations for tensor_name in ctx.graph.inputs):
            lines.append("")

        # Calculate reload buffer requirements
        # Need buffers for:
        # 1. Inputs to reload
        # 2. Outputs that are spilled (need temp buffers in fast memory)
        max_reload_slots = plan.get_max_reload_slots()

        # Count max total buffers needed at any node (inputs + spilled outputs)
        nodes = ctx.graph.topological_sort()
        max_total_slots = 0
        for node_idx, node in enumerate(nodes):
            if node.op_type == OpType.CONSTANT:
                continue

            # Count inputs that need reload
            reloads_before = plan.get_reload_points_before(node_idx)
            input_slots = len(reloads_before)

            # Count outputs that spill after this node, even if they are later reloaded
            # and do not remain slow-backed in the final tensor allocation map.
            spill_outputs_after = get_output_spill_points(node_idx)
            output_slots = len(spill_outputs_after)

            max_total_slots = max(max_total_slots, input_slots + output_slots)

        # Use the larger of the two calculations
        max_reload_slots = max(max_reload_slots, max_total_slots)

        # Get max tensor size for reload buffer sizing
        max_tensor_size = 0
        transferred_tensors = {
            point.tensor_name for point in plan.spill_points
        } | {
            point.tensor_name for point in plan.reload_points
        } | plan.spilled_tensors
        for tensor_name in transferred_tensors:
            alloc = plan.tensor_allocations.get(tensor_name)
            if alloc is not None:
                max_tensor_size = max(max_tensor_size, alloc.size)
        for point in plan.spill_points:
            max_tensor_size = max(max_tensor_size, point.size)
        for point in plan.reload_points:
            max_tensor_size = max(max_tensor_size, point.size)

        if not plan.has_spill and not plan.move_points:
            emitter = CEmitter(pipeline_schedule_metadata=pipeline_codegen_metadata)
            return emitter.emit(ctx)

        # Generate reload buffers in fast memory
        if max_reload_slots > 0 and max_tensor_size > 0:
            lines.append("/* Reload buffers in fast memory for spilled tensors */")
            lines.append(f"/* Max reload slots: {max_reload_slots}, Slot size: {max_tensor_size} */")
            for i in range(max_reload_slots):
                lines.append(f"static uint8_t _nnc_reload_buffer_{i}[{max_tensor_size}];")
            lines.append("")

        # Generate forward declarations for node functions (only for those without spill/reload)
        nodes = ctx.graph.topological_sort()
        parallel_runtime_enabled = self._has_parallel_runtime(pipeline_codegen_metadata)
        for node_idx, node in enumerate(nodes):
            if node.op_type == OpType.CONSTANT:
                continue

            func_name = ctx.node_symbols.get(node.name, node.name)
            spill_outputs_after = get_output_spill_points(node_idx)
            move_points_at = sorted(
                get_move_points_at(node_idx),
                key=lambda move: move.from_offset,
            )

            # Check if this node needs spill/reload
            has_spill = len(spill_outputs_after) > 0
            has_move = len(move_points_at) > 0
            has_reload = len(plan.get_reload_points_before(node_idx)) > 0
            has_spilled_output = any(
                (
                    o in spill_outputs_after
                    or (
                        plan.tensor_allocations.get(o) is not None
                        and plan.tensor_allocations.get(o).is_spilled  # type: ignore[union-attr]
                    )
                )
                for o in node.outputs
            )

            # Only declare _body if no spill/reload needed
            if not has_spill and not has_move and not has_reload and not has_spilled_output:
                lines.append(f"static void {func_name}_body(void);")

        lines.append("")
        lines.append("/* Spill/Reload wrapper functions */")

        # Generate wrapper functions with spill/reload
        for node_idx, node in enumerate(nodes):
            if node.op_type == OpType.CONSTANT:
                continue

            func_name = ctx.node_symbols.get(node.name, node.name)

            # Collect spills after this node
            spill_points_after = get_output_spill_points(node_idx)
            move_points_at = sorted(
                get_move_points_at(node_idx),
                key=lambda move: move.from_offset,
            )
            move_points_by_tensor = {
                move.tensor_name: move for move in move_points_at
            }

            # Collect reloads before this node
            reloads_before = plan.get_reload_points_before(node_idx)

            # Determine which inputs need reload (have reload points before this node)
            inputs_need_reload: dict[str, Union["ReloadPoint", "SpillPoint"]] = {
                rp.tensor_name: rp for rp in reloads_before
            }

            # Also check if any input tensors are spilled (allocated in slow memory)
            # This handles model inputs that happen to be in slow memory
            additional_inputs: list[tuple[str, "TensorAllocation"]] = []
            for input_name in node.inputs:
                if input_name in inputs_need_reload:
                    continue  # Already has a reload point
                alloc = plan.tensor_allocations.get(input_name)
                if alloc is not None and alloc.is_spilled:
                    # Input is spilled but no reload point - create a synthetic one
                    # Look up the correct slow offset from existing reload points
                    existing_reload = next((rp for rp in plan.reload_points if rp.tensor_name == input_name), None)
                    slow_offset = existing_reload.from_slow_offset if existing_reload else 0
                    additional_inputs.append((input_name, alloc))
                    new_reload_point = ReloadPoint(
                        tensor_name=input_name,
                        before_node=node.name,
                        before_node_idx=node_idx,
                        from_slow_offset=slow_offset,
                        to_buffer_id=alloc.buffer_id,
                        to_fast_offset=0,
                        size=alloc.size,
                        reload_slot_id=-1,  # Will assign below
                    )
                    inputs_need_reload[input_name] = new_reload_point

            # Assign slot IDs to synthetic reload points
            current_slot = max(
                [rp.reload_slot_id for rp in inputs_need_reload.values() if isinstance(rp, ReloadPoint)],
                default=-1,
            ) + 1
            for input_name, _ in additional_inputs:
                rp = inputs_need_reload[input_name]
                if isinstance(rp, ReloadPoint):
                    rp.reload_slot_id = current_slot
                current_slot += 1

            # Determine which outputs need temp tensors
            # An output needs a temp tensor if:
            # 1. It's spilled (allocated in slow memory), OR
            # 2. There's a spill point after this node for it
            outputs_need_temp: dict[str, Union["ReloadPoint", "SpillPoint"]] = {}
            for output_name in node.outputs:
                spill_point = spill_points_after.get(output_name)
                if spill_point is not None:
                    outputs_need_temp[output_name] = spill_point
                    continue

                alloc = plan.tensor_allocations.get(output_name)
                if alloc is not None and alloc.is_spilled:
                    # Output lives in slow memory - need temp tensor
                    # Find the spill point for this output
                    # Create a synthetic spill point for temp allocation.
                    existing_spill = next((sp for sp in plan.spill_points if sp.tensor_name == output_name), None)
                    slow_offset = existing_spill.to_slow_offset if existing_spill else 0
                    outputs_need_temp[output_name] = SpillPoint(
                        tensor_name=output_name,
                        after_node=node.name,
                        after_node_idx=node_idx,
                        from_buffer_id=alloc.buffer_id,
                        from_fast_offset=0,
                        to_slow_offset=slow_offset,
                        size=alloc.size,
                    )

            lines.append(f"/* {func_name}: {node.op_type.value} */")
            lines.append(f"static void {func_name}(void) {{")
            self._append_pipeline_step_comment_lines(
                lines,
                pipeline_codegen_metadata,
                node.name,
                indent="    ",
            )

            moved_first_use_inputs = [
                input_name
                for input_name in node.inputs
                if input_first_use.get(input_name) == node_idx
                and input_name in move_points_by_tensor
            ]
            for input_name in moved_first_use_inputs:
                alloc = plan.tensor_allocations.get(input_name)
                move = move_points_by_tensor[input_name]
                var_name = ctx.tensor_symbols.get(input_name, input_name)
                from_expr = f"_nnc_fast_pool + {move.from_offset}"
                if alloc is None:
                    continue
                lines.extend([
                    f"    /* Stage graph input {input_name} into pre-move fast memory */",
                    f"    memcpy({from_expr}, {get_saved_input_binding_name(input_name)}, {alloc.size});",
                    f"    {var_name}.data = {from_expr};",
                    "",
                ])

            for move in move_points_at:
                var_name = ctx.tensor_symbols.get(move.tensor_name, move.tensor_name)
                to_expr = f"_nnc_fast_pool + {move.to_offset}"
                from_expr = f"_nnc_fast_pool + {move.from_offset}"
                lines.extend([
                    f"    /* Move {move.tensor_name} within fast memory */",
                    f"    memmove({to_expr}, {from_expr}, {move.size});",
                    f"    {var_name}.data = {to_expr};",
                    "",
                ])

            for input_name in node.inputs:
                if input_first_use.get(input_name) != node_idx:
                    continue
                if input_name in move_points_by_tensor:
                    continue

                alloc = plan.tensor_allocations.get(input_name)
                target_expr = get_internal_storage_expr(input_name)
                if alloc is None or target_expr is None:
                    continue

                var_name = ctx.tensor_symbols.get(input_name, input_name)
                lines.extend([
                    f"    /* Stage graph input {input_name} into planned memory */",
                    f"    memcpy({target_expr}, {get_saved_input_binding_name(input_name)}, {alloc.size});",
                    f"    {var_name}.data = {target_expr};",
                    "",
                ])

            # Declare temp tensors for spilled inputs
            temp_tensor_decls: list[tuple[str, int, str, int, str]] = []
            for input_name in node.inputs:
                if input_name in inputs_need_reload:
                    rp = inputs_need_reload[input_name]
                    temp_name = f"temp_{ctx.tensor_symbols.get(input_name, input_name)}"
                    slot_id = rp.reload_slot_id if isinstance(rp, ReloadPoint) else -1
                    temp_tensor_decls.append((temp_name, slot_id, input_name, rp.size, 'input'))

            # Declare temp tensors for spilled outputs
            output_slot_idx = len(inputs_need_reload)
            for output_name in node.outputs:
                if output_name in outputs_need_temp:
                    sp_val = outputs_need_temp[output_name]
                    temp_name = f"temp_{ctx.tensor_symbols.get(output_name, output_name)}"
                    slot_id = output_slot_idx
                    temp_tensor_decls.append((temp_name, slot_id, output_name, sp_val.size, 'output'))
                    output_slot_idx += 1

            # Generate temp tensor declarations
            if temp_tensor_decls:
                lines.append("    /* Temp tensors pointing to fast memory */")
                for temp_name, slot_id, orig_name, size, _ in temp_tensor_decls:
                    lines.append(f"    static Tensor {temp_name};")

            # Reload spilled inputs to fast memory
            for input_name in node.inputs:
                if input_name in inputs_need_reload:
                    rp = inputs_need_reload[input_name]
                    if not isinstance(rp, ReloadPoint):
                        continue
                    temp_name = f"temp_{ctx.tensor_symbols.get(input_name, input_name)}"
                    var_name = ctx.tensor_symbols.get(input_name, input_name)
                    lines.extend([
                        f"    /* Reload {input_name} from slow to fast memory */",
                        f"    memcpy(_nnc_reload_buffer_{rp.reload_slot_id},",
                        f"           _nnc_slow_pool + {rp.from_slow_offset}, {rp.size});",
                        f"    {temp_name}.data = _nnc_reload_buffer_{rp.reload_slot_id};",
                        f"    {temp_name}.dtype = {var_name}.dtype;",
                        f"    {temp_name}.shape = {var_name}.shape;",
                        f"    {temp_name}.ndim = {var_name}.ndim;",
                        f"    {temp_name}.nbytes = {rp.size};",
                        "",
                    ])

            # Setup temp outputs
            output_slot_idx = len(inputs_need_reload)
            for output_name in node.outputs:
                if output_name in outputs_need_temp:
                    sp_val = outputs_need_temp[output_name]
                    temp_name = f"temp_{ctx.tensor_symbols.get(output_name, output_name)}"
                    var_name = ctx.tensor_symbols.get(output_name, output_name)
                    # Get the slot_id from temp_tensor_decls
                    slot_id = next((s for tn, s, _, _, _ in temp_tensor_decls if tn == temp_name), output_slot_idx)
                    # Both ReloadPoint and SpillPoint have 'size' attribute
                    sp_size = sp_val.size
                    lines.extend([
                        f"    /* Setup temp output for {output_name} */",
                        f"    {temp_name}.data = _nnc_reload_buffer_{slot_id};",
                        f"    {temp_name}.dtype = {var_name}.dtype;",
                        f"    {temp_name}.shape = {var_name}.shape;",
                        f"    {temp_name}.ndim = {var_name}.ndim;",
                        f"    {temp_name}.nbytes = {sp_size};",
                        "",
                    ])

            # Generate operator call with potentially swapped arguments
            lines.append("    /* Execute operator in fast memory */")
            op_call = self._generate_operator_call_with_temps(
                ctx, node, inputs_need_reload, outputs_need_temp
            )
            lines.append(f"    {op_call}")

            # Spill outputs back to slow memory
            output_slot_idx = len(inputs_need_reload)
            for output_name in node.outputs:
                if output_name in outputs_need_temp:
                    sp_val = outputs_need_temp[output_name]
                    temp_name = f"temp_{ctx.tensor_symbols.get(output_name, output_name)}"
                    # Get the slot_id from temp_tensor_decls
                    slot_id = next((s for tn, s, _, _, _ in temp_tensor_decls if tn == temp_name), output_slot_idx)
                    # Both ReloadPoint and SpillPoint have these attributes (or should)
                    # SpillPoint has to_slow_offset, ReloadPoint doesn't (but we shouldn't spill to ReloadPoint)
                    if isinstance(sp_val, SpillPoint):
                        lines.extend([
                            "",
                            f"    /* Spill {output_name} from fast to slow memory */",
                            f"    memcpy(_nnc_slow_pool + {sp_val.to_slow_offset},",
                            f"           _nnc_reload_buffer_{slot_id}, {sp_val.size});",
                        ])

            lines.append("}")
            lines.append("")

        # Generate body functions for nodes without spill/reload
        lines.append("/* Node body functions (no spill/reload needed) */")

        # Generate operator calls for all nodes (for reference)
        # We generate simplified versions that don't need spill/reload
        for node_idx, node in enumerate(nodes):
            if node.op_type == OpType.CONSTANT:
                continue

            func_name = ctx.node_symbols.get(node.name, node.name)
            spill_outputs_after = get_output_spill_points(node_idx)
            move_points_at = get_move_points_at(node_idx)

            # Check if this node needs spill/reload
            # A node needs spill/reload wrapper if:
            # 1. It has reload points before it (inputs to reload), OR
            # 2. Any of its outputs are spilled (live in slow memory)
            has_spill = len(spill_outputs_after) > 0
            has_move = len(move_points_at) > 0
            has_reload = len(plan.get_reload_points_before(node_idx)) > 0
            has_spilled_output = any(
                (
                    o in spill_outputs_after
                    or (
                        plan.tensor_allocations.get(o) is not None
                        and plan.tensor_allocations.get(o).is_spilled  # type: ignore[union-attr]
                    )
                )
                for o in node.outputs
            )

            if parallel_runtime_enabled:
                lines.append(f"/* {func_name}: {node.op_type.value} */")
                lines.append(f"static void {func_name}_body(void) {{")
                op_call = self._generate_operator_call(ctx, node, use_temps=False)
                lines.append(f"    {op_call}")
                lines.append("}")
                lines.append("")
            elif not has_spill and not has_move and not has_reload and not has_spilled_output:
                # No spill/reload needed, generate standard operator call
                lines.append(f"/* {func_name}: {node.op_type.value} */")
                lines.append(f"static void {func_name}_body(void) {{")
                op_call = self._generate_operator_call(ctx, node, use_temps=False)
                lines.append(f"    {op_call}")
                lines.append("}")
                lines.append("")
            else:
                # Has spill/reload, don't generate _body function
                # The wrapper handles everything
                pass

        helper_block = self._render_parallel_step_helper_block(pipeline_codegen_metadata)
        runtime_block = self._render_parallel_runtime_block(pipeline_codegen_metadata)
        if helper_block:
            lines.append("")
            lines.extend(helper_block)
        if runtime_block:
            lines.append("")
            lines.extend(runtime_block)

        # Generate main entry point
        lines.append("")
        lines.append("/* Main inference entry point */")
        lines.append("void nnc_run(void) {")

        for tensor_name in ctx.graph.inputs:
            if tensor_name not in plan.tensor_allocations:
                continue
            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            lines.append(f"    {get_saved_input_binding_name(tensor_name)} = {var_name}.data;")

        if self._has_parallel_runtime(pipeline_codegen_metadata):
            lines.append("    nnc_pipeline_run_parallel();")
        else:
            for node in nodes:
                if node.op_type == OpType.CONSTANT:
                    continue
                func_name = ctx.node_symbols.get(node.name, node.name)
                self._append_pipeline_step_comment_lines(
                    lines,
                    pipeline_codegen_metadata,
                    node.name,
                    indent="    ",
                )
                lines.append(f"    {func_name}();")

        for tensor_name in ctx.graph.inputs:
            if tensor_name not in plan.tensor_allocations:
                continue
            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            lines.append(f"    {var_name}.data = {get_saved_input_binding_name(tensor_name)};")

        lines.append("}")

        return "\n".join(lines)

    def _generate_operator_call_with_temps(
        self,
        ctx: CompileContext,
        node: Node,
        inputs_need_reload: dict[str, Union["ReloadPoint", "SpillPoint"]],
        outputs_need_temp: dict[str, Union["ReloadPoint", "SpillPoint"]],
    ) -> str:
        """Generate operator call using temp tensors for spilled inputs/outputs.

        Args:
            ctx: Compilation context
            node: Node to generate call for
            inputs_need_reload: Map of input_name -> ReloadPoint
            outputs_need_temp: Map of output_name -> SpillPoint (for spilled outputs)

        Returns:
            C code line for the operator call
        """
        # Special handling for operators with attribute-as-input (after onnxsim optimization)
        # These operators need special handling because some attributes became inputs
        if node.op_type == OpType.CONCAT:
            return self._generate_concat_call(ctx, node, inputs_need_reload, outputs_need_temp)
        if node.op_type == OpType.SPLIT:
            return self._generate_split_call(ctx, node, inputs_need_reload, outputs_need_temp)

        # Operators that have extra "attribute" inputs after onnxsim
        if node.op_type in (OpType.REDUCE_SUM, OpType.REDUCE_MEAN, OpType.UNSQUEEZE, OpType.TRANSPOSE, OpType.TILE, OpType.RESHAPE, OpType.POW, OpType.CLIP):
            return self._generate_attr_as_input_op_call(ctx, node, inputs_need_reload, outputs_need_temp)

        op_name = f"nnc_{node.op_type.value.lower()}"
        args = []

        # Input tensors
        for input_name in node.inputs:
            var_name = ctx.tensor_symbols.get(input_name, input_name)
            if input_name in inputs_need_reload:
                # Use temp tensor
                temp_name = f"temp_{var_name}"
                args.append(f"&{temp_name}")
            else:
                # Use original tensor
                args.append(f"&{var_name}")

        # Output tensors
        for output_name in node.outputs:
            var_name = ctx.tensor_symbols.get(output_name, output_name)
            if output_name in outputs_need_temp:
                # Use temp tensor
                temp_name = f"temp_{var_name}"
                args.append(f"&{temp_name}")
            else:
                # Use original tensor
                args.append(f"&{var_name}")

        # Add operation-specific attributes
        if node.op_type == OpType.CONV2D:
            kernel_shape = node.attrs.get("kernel_shape", [1, 1])
            strides = node.attrs.get("strides", [1, 1])
            pads = node.attrs.get("pads", [0, 0])
            args.extend([str(kernel_shape[0]), str(kernel_shape[1])])
            args.extend([str(strides[0]), str(strides[1])])
            if len(pads) == 4:
                args.append(str(pads[0]))
                args.append(str(pads[1]))
            elif len(pads) == 2:
                args.extend([str(pads[0]), str(pads[1])])
            else:
                args.extend(["0", "0"])
        elif node.op_type == OpType.LAYER_NORM:
            axis = node.attrs.get("axis", -1)
            epsilon = node.attrs.get("epsilon", 1e-5)
            args.append(f"{axis}")
            args.append(f"{epsilon}f")
        elif node.op_type == OpType.CAST:
            # Cast: input, output, to_dtype
            to_dtype = node.attrs.get("to")
            if to_dtype and isinstance(to_dtype, DataType):
                dtype_const = self.DTYPE_MAP.get(to_dtype, "NNC_DTYPE_FLOAT32")
                args.append(dtype_const)
            else:
                args.append("NNC_DTYPE_FLOAT32")
        elif node.op_type == OpType.SPLIT:
            # Split has signature: nnc_split(Tensor* input, Tensor** outputs, int num_outputs, int axis)
            # We need to completely rewrite the call
            return self._generate_split_call(ctx, node, inputs_need_reload, outputs_need_temp)
        elif node.op_type == OpType.GATHER:
            # Gather: data, indices, output, axis, data_dtype
            axis = node.attrs.get("axis", 0)
            # Determine data type: 1 for int64 (e.g., from Shape operator), 0 for float
            data_tensor = ctx.graph.get_tensor(node.inputs[0])
            data_dtype = 1 if data_tensor and data_tensor.dtype == DataType.INT64 else 0
            args.append(str(axis))
            args.append(str(data_dtype))
        elif node.op_type == OpType.REDUCE_MEAN:
            # ONNX uses "axes" (plural) as a list, or "axis" (singular) as int
            axes = node.attrs.get("axes", None)
            axis = node.attrs.get("axis", None)
            keepdims = node.attrs.get("keepdims", 1)
            # Determine axis value (handle both "axes" list and "axis" int)
            if axes is not None and isinstance(axes, list) and len(axes) > 0:
                axis_val = axes[0]
            elif axis is not None:
                axis_val = axis
            else:
                axis_val = -1  # Default to last axis
            args.append(str(axis_val))
            args.append(str(keepdims))
        elif node.op_type == OpType.REDUCE_SUM:
            # ONNX uses "axes" (plural) as a list, or "axis" (singular) as int
            axes = node.attrs.get("axes", None)
            axis = node.attrs.get("axis", None)
            keepdims = node.attrs.get("keepdims", 1)
            # Determine axis value (handle both "axes" list and "axis" int)
            if axes is not None and isinstance(axes, list) and len(axes) > 0:
                axis_val = axes[0]
            elif axis is not None:
                axis_val = axis
            else:
                axis_val = -1  # Default to last axis
            args.append(str(axis_val))
            args.append(str(keepdims))
        elif node.op_type == OpType.TRANSPOSE:
            # Transpose: input, output, perm array, ndim
            # perm can be in attributes or as an input (after onnxsim optimization)
            perm = node.attrs.get("perm", None)
            if perm is None and len(node.inputs) > 1:
                # perm might be the second input (constant)
                perm_input = node.inputs[1]
                if perm_input in ctx.graph.constants:
                    perm = ctx.graph.constants[perm_input].tolist()
            if perm is None:
                perm = []  # Will use default
            ndim = len(perm) if perm else 0
            # Generate static perm array
            perm_name = f"{ctx.node_symbols.get(node.name, node.name)}_perm"
            perm_decl = f"static const int64_t {perm_name}[] = {{{', '.join(map(str, perm))}}};"
            # Need to return multi-line string
            output_var = ctx.tensor_symbols.get(node.outputs[0], node.outputs[0])
            return f"{perm_decl}\n    nnc_transpose(&{ctx.tensor_symbols.get(node.inputs[0], node.inputs[0])}, &{output_var}, (int64_t*){perm_name}, {ndim});"
        elif node.op_type == OpType.UNSQUEEZE:
            # Unsqueeze: input, output, axis
            # In ONNX opset 13+, axes can be an input instead of attribute
            axes = node.attrs.get("axes", None)
            if axes is None and len(node.inputs) > 1:
                # axes might be the second input (constant)
                axes_input = node.inputs[1]
                if axes_input in ctx.graph.constants:
                    axes_val = ctx.graph.constants[axes_input]
                    if isinstance(axes_val, list):
                        axes = axes_val
                    elif hasattr(axes_val, 'tolist'):
                        axes = axes_val.tolist()
                    else:
                        axes = [axes_val]
            if axes is None:
                axes = [-1]  # Default to last axis
            axis_val = axes[0] if isinstance(axes, list) else axes
            args.append(str(axis_val))
        elif node.op_type == OpType.TILE:
            # Tile: input, output, repeats array, ndim
            # repeats is the second input (constant)
            if len(node.inputs) > 1:
                repeats_input = node.inputs[1]
                if repeats_input in ctx.graph.constants:
                    repeats = ctx.graph.constants[repeats_input]
                    if isinstance(repeats, list):
                        repeats_list = repeats
                    elif hasattr(repeats, 'tolist'):
                        repeats_list = repeats.tolist()
                    else:
                        repeats_list = [repeats]
                    ndim = len(repeats_list)
                    # Generate static repeats array
                    repeats_name = f"{ctx.node_symbols.get(node.name, node.name)}_repeats"
                    repeats_decl = f"static const int64_t {repeats_name}[] = {{{', '.join(map(str, repeats_list))}}};"
                    # Need to return multi-line string
                    output_var = ctx.tensor_symbols.get(node.outputs[0], node.outputs[0])
                    return f"{repeats_decl}\n    nnc_tile(&{ctx.tensor_symbols.get(node.inputs[0], node.inputs[0])}, &{output_var}, (int64_t*){repeats_name}, {ndim});"
            # Fallback: pass NULL for repeats
            args.append("NULL")
            args.append("0")
        elif node.op_type == OpType.GEMM:
            # Gemm has optional bias (3rd input), need to add attributes
            # nnc_gemm(Tensor* a, Tensor* b, Tensor* c, Tensor* output,
            #          float alpha, float beta, int trans_a, int trans_b)
            # Check if bias (C) is present
            if len(node.inputs) < 3:
                # No bias tensor, insert NULL
                args.insert(2, "NULL")
            # Add attributes
            alpha = node.attrs.get("alpha", 1.0)
            beta = node.attrs.get("beta", 1.0)
            trans_a = node.attrs.get("transA", 0)
            trans_b = node.attrs.get("transB", 0)
            args.append(f"{alpha}f")
            args.append(f"{beta}f")
            args.append(str(trans_a))
            args.append(str(trans_b))
        elif node.op_type == OpType.GATHER:
            # Gather: data, indices, output, axis, data_dtype
            axis = node.attrs.get("axis", 0)
            # Determine data type: 1 for int64 (e.g., from Shape operator), 0 for float
            data_tensor = ctx.graph.get_tensor(node.inputs[0])
            data_dtype = 1 if data_tensor and data_tensor.dtype == DataType.INT64 else 0
            args.append(str(axis))
            args.append(str(data_dtype))
        elif node.op_type == OpType.LSTM:
            # LSTM has special handling
            # X, W, R, B inputs
            # Y, Y_h, Y_c outputs
            # direction, hidden_size attributes
            direction = node.attrs.get("direction", "forward")
            hidden_size = node.attrs.get("hidden_size", 0)
            direction_map = {"forward": 0, "reverse": 1, "bidirectional": 2}
            direction_val = direction_map.get(direction, 0)
            args.append(str(direction_val))
            args.append(str(hidden_size))

        return f"{op_name}({', '.join(args)});"

    def _generate_operator_call(self, ctx: CompileContext, node: Node, use_temps: bool = False) -> str:
        """Generate operator call (for non-spill nodes).

        Args:
            ctx: Compilation context
            node: Node to generate call for
            use_temps: Whether to use temp tensors (not used here, for compatibility)

        Returns:
            C code line for the operator call
        """
        # Special handling for operators with attribute-as-input (after onnxsim optimization)
        if node.op_type == OpType.CONCAT:
            return self._generate_concat_call(ctx, node, inputs_need_reload={}, outputs_need_temp={})
        if node.op_type == OpType.SPLIT:
            return self._generate_split_call(ctx, node, inputs_need_reload={}, outputs_need_temp={})
        if node.op_type in (OpType.REDUCE_SUM, OpType.REDUCE_MEAN, OpType.UNSQUEEZE, OpType.TRANSPOSE, OpType.TILE, OpType.RESHAPE, OpType.POW, OpType.CLIP):
            return self._generate_attr_as_input_op_call(ctx, node, inputs_need_reload={}, outputs_need_temp={})

        op_name = f"nnc_{node.op_type.value.lower()}"
        args = []

        # Input tensors
        for input_name in node.inputs:
            var_name = ctx.tensor_symbols.get(input_name, input_name)
            args.append(f"&{var_name}")

        # Output tensors
        for output_name in node.outputs:
            var_name = ctx.tensor_symbols.get(output_name, output_name)
            args.append(f"&{var_name}")

        # Add operation-specific attributes
        if node.op_type == OpType.CONV2D:
            kernel_shape = node.attrs.get("kernel_shape", [1, 1])
            strides = node.attrs.get("strides", [1, 1])
            pads = node.attrs.get("pads", [0, 0])
            args.extend([str(kernel_shape[0]), str(kernel_shape[1])])
            args.extend([str(strides[0]), str(strides[1])])
            if len(pads) == 4:
                args.append(str(pads[0]))
                args.append(str(pads[1]))
            elif len(pads) == 2:
                args.extend([str(pads[0]), str(pads[1])])
            else:
                args.extend(["0", "0"])
        elif node.op_type == OpType.LAYER_NORM:
            axis = node.attrs.get("axis", -1)
            epsilon = node.attrs.get("epsilon", 1e-5)
            args.append(f"{axis}")
            args.append(f"{epsilon}f")
        elif node.op_type == OpType.CAST:
            # Cast: input, output, to_dtype
            to_dtype = node.attrs.get("to")
            if to_dtype and isinstance(to_dtype, DataType):
                dtype_const = self.DTYPE_MAP.get(to_dtype, "NNC_DTYPE_FLOAT32")
                args.append(dtype_const)
            else:
                args.append("NNC_DTYPE_FLOAT32")
        elif node.op_type == OpType.SPLIT:
            # Split has signature: nnc_split(Tensor* input, Tensor** outputs, int num_outputs, int axis)
            # We need to completely rewrite the call
            return self._generate_split_call(ctx, node, inputs_need_reload, outputs_need_temp)
        elif node.op_type == OpType.GATHER:
            # Gather: data, indices, output, axis, data_dtype
            axis = node.attrs.get("axis", 0)
            # Determine data type: 1 for int64 (e.g., from Shape operator), 0 for float
            data_tensor = ctx.graph.get_tensor(node.inputs[0])
            data_dtype = 1 if data_tensor and data_tensor.dtype == DataType.INT64 else 0
            args.append(str(axis))
            args.append(str(data_dtype))
        elif node.op_type == OpType.REDUCE_MEAN:
            # ONNX uses "axes" (plural) as a list, or "axis" (singular) as int
            axes = node.attrs.get("axes", None)
            axis = node.attrs.get("axis", None)
            keepdims = node.attrs.get("keepdims", 1)
            # Determine axis value (handle both "axes" list and "axis" int)
            if axes is not None and isinstance(axes, list) and len(axes) > 0:
                axis_val = axes[0]
            elif axis is not None:
                axis_val = axis
            else:
                axis_val = -1  # Default to last axis
            args.append(str(axis_val))
            args.append(str(keepdims))
        elif node.op_type == OpType.REDUCE_SUM:
            # ONNX uses "axes" (plural) as a list, or "axis" (singular) as int
            axes = node.attrs.get("axes", None)
            axis = node.attrs.get("axis", None)
            keepdims = node.attrs.get("keepdims", 1)
            # Determine axis value (handle both "axes" list and "axis" int)
            if axes is not None and isinstance(axes, list) and len(axes) > 0:
                axis_val = axes[0]
            elif axis is not None:
                axis_val = axis
            else:
                axis_val = -1  # Default to last axis
            args.append(str(axis_val))
            args.append(str(keepdims))
        elif node.op_type == OpType.TRANSPOSE:
            # Transpose: input, output, perm array, ndim
            # perm can be in attributes or as an input (after onnxsim optimization)
            perm = node.attrs.get("perm", None)
            if perm is None and len(node.inputs) > 1:
                # perm might be the second input (constant)
                perm_input = node.inputs[1]
                if perm_input in ctx.graph.constants:
                    perm = ctx.graph.constants[perm_input].tolist()
            if perm is None:
                perm = []  # Will use default
            ndim = len(perm) if perm else 0
            # Generate static perm array
            perm_name = f"{ctx.node_symbols.get(node.name, node.name)}_perm"
            perm_decl = f"static const int64_t {perm_name}[] = {{{', '.join(map(str, perm))}}};"
            # Need to return multi-line string
            output_var = ctx.tensor_symbols.get(node.outputs[0], node.outputs[0])
            return f"{perm_decl}\n    nnc_transpose(&{ctx.tensor_symbols.get(node.inputs[0], node.inputs[0])}, &{output_var}, (int64_t*){perm_name}, {ndim});"
        elif node.op_type == OpType.UNSQUEEZE:
            # Unsqueeze: input, output, axis
            # In ONNX opset 13+, axes can be an input instead of attribute
            axes = node.attrs.get("axes", None)
            if axes is None and len(node.inputs) > 1:
                # axes might be the second input (constant)
                axes_input = node.inputs[1]
                if axes_input in ctx.graph.constants:
                    axes_val = ctx.graph.constants[axes_input]
                    if isinstance(axes_val, list):
                        axes = axes_val
                    elif hasattr(axes_val, 'tolist'):
                        axes = axes_val.tolist()
                    else:
                        axes = [axes_val]
            if axes is None:
                axes = [-1]  # Default to last axis
            axis_val = axes[0] if isinstance(axes, list) else axes
            args.append(str(axis_val))
        elif node.op_type == OpType.TILE:
            # Tile: input, output, repeats array, ndim
            # repeats is the second input (constant)
            if len(node.inputs) > 1:
                repeats_input = node.inputs[1]
                if repeats_input in ctx.graph.constants:
                    repeats = ctx.graph.constants[repeats_input]
                    if isinstance(repeats, list):
                        repeats_list = repeats
                    elif hasattr(repeats, 'tolist'):
                        repeats_list = repeats.tolist()
                    else:
                        repeats_list = [repeats]
                    ndim = len(repeats_list)
                    # Generate static repeats array
                    repeats_name = f"{ctx.node_symbols.get(node.name, node.name)}_repeats"
                    repeats_decl = f"static const int64_t {repeats_name}[] = {{{', '.join(map(str, repeats_list))}}};"
                    # Need to return multi-line string
                    output_var = ctx.tensor_symbols.get(node.outputs[0], node.outputs[0])
                    return f"{repeats_decl}\n    nnc_tile(&{ctx.tensor_symbols.get(node.inputs[0], node.inputs[0])}, &{output_var}, (int64_t*){repeats_name}, {ndim});"
            # Fallback: pass NULL for repeats
            args.append("NULL")
            args.append("0")
        elif node.op_type == OpType.GEMM:
            # Gemm has optional bias (3rd input), need to add attributes
            # nnc_gemm(Tensor* a, Tensor* b, Tensor* c, Tensor* output,
            #          float alpha, float beta, int trans_a, int trans_b)
            # Check if bias (C) is present
            if len(node.inputs) < 3:
                # No bias tensor, insert NULL
                args.insert(2, "NULL")
            # Add attributes
            alpha = node.attrs.get("alpha", 1.0)
            beta = node.attrs.get("beta", 1.0)
            trans_a = node.attrs.get("transA", 0)
            trans_b = node.attrs.get("transB", 0)
            args.append(f"{alpha}f")
            args.append(f"{beta}f")
            args.append(str(trans_a))
            args.append(str(trans_b))
        elif node.op_type == OpType.GATHER:
            # Gather: data, indices, output, axis, data_dtype
            axis = node.attrs.get("axis", 0)
            # Determine data type: 1 for int64 (e.g., from Shape operator), 0 for float
            data_tensor = ctx.graph.get_tensor(node.inputs[0])
            data_dtype = 1 if data_tensor and data_tensor.dtype == DataType.INT64 else 0
            args.append(str(axis))
            args.append(str(data_dtype))
        elif node.op_type == OpType.LSTM:
            # LSTM has special handling
            # X, W, R, B inputs
            # Y, Y_h, Y_c outputs
            # direction, hidden_size attributes
            direction = node.attrs.get("direction", "forward")
            hidden_size = node.attrs.get("hidden_size", 0)
            direction_map = {"forward": 0, "reverse": 1, "bidirectional": 2}
            direction_val = direction_map.get(direction, 0)
            args.append(str(direction_val))
            args.append(str(hidden_size))

        return f"{op_name}({', '.join(args)});"

    def _generate_concat_call(
        self,
        ctx: CompileContext,
        node: Node,
        inputs_need_reload: dict[str, Union["ReloadPoint", "SpillPoint"]],
        outputs_need_temp: dict[str, Union["ReloadPoint", "SpillPoint"]],
    ) -> str:
        """Generate concat operation call.

        Concat has signature: void nnc_concat(Tensor** inputs, Tensor* output, int num_inputs, int axis);

        Args:
            ctx: Compilation context
            node: Node to generate call for
            inputs_need_reload: Map of input_name -> ReloadPoint
            outputs_need_temp: Map of output_name -> SpillPoint

        Returns:
            C code for the concat call (may be multi-line)
        """
        num_inputs = len(node.inputs)
        axis = node.attrs.get("axis", 0)
        output_var = ctx.tensor_symbols.get(node.outputs[0], node.outputs[0])

        # Create static array of input pointers
        array_name = f"{ctx.node_symbols.get(node.name, node.name)}_inputs"
        input_ptrs = []

        for input_name in node.inputs:
            var_name = ctx.tensor_symbols.get(input_name, input_name)
            if input_name in inputs_need_reload:
                # Use temp tensor
                temp_name = f"temp_{var_name}"
                input_ptrs.append(f"&{temp_name}")
            else:
                # Use original tensor
                input_ptrs.append(f"&{var_name}")

        # Generate the array declaration and call
        array_decl = f"static Tensor* {array_name}[{num_inputs}] = {{{', '.join(input_ptrs)}}};"
        output_arg = f"&{output_var}"

        if node.outputs[0] in outputs_need_temp:
            temp_name = f"temp_{output_var}"
            output_arg = f"&{temp_name}"

        call = f"nnc_concat({array_name}, {output_arg}, {num_inputs}, {axis});"

        # Return multi-line string with array decl and call
        return f"{array_decl}\n    {call}"

    def _generate_split_call(
        self,
        ctx: CompileContext,
        node: Node,
        inputs_need_reload: dict[str, Union["ReloadPoint", "SpillPoint"]],
        outputs_need_temp: dict[str, Union["ReloadPoint", "SpillPoint"]],
    ) -> str:
        """Generate split operation call.

        Split has signature: void nnc_split(Tensor* input, Tensor** outputs, int num_outputs, int axis);

        Args:
            ctx: Compilation context
            node: Node to generate call for
            inputs_need_reload: Map of input_name -> ReloadPoint
            outputs_need_temp: Map of output_name -> SpillPoint

        Returns:
            C code for the split call (may be multi-line)
        """
        num_outputs = len(node.outputs)
        axis = node.attrs.get("axis", 0)
        input_var = ctx.tensor_symbols.get(node.inputs[0], node.inputs[0])

        # Create static array of output pointers
        array_name = f"{ctx.node_symbols.get(node.name, node.name)}_outputs"
        output_ptrs = []

        for output_name in node.outputs:
            var_name = ctx.tensor_symbols.get(output_name, output_name)
            if output_name in outputs_need_temp:
                # Use temp tensor
                temp_name = f"temp_{var_name}"
                output_ptrs.append(f"&{temp_name}")
            else:
                # Use original tensor
                output_ptrs.append(f"&{var_name}")

        # Generate the array declaration and call
        array_decl = f"static Tensor* {array_name}[{num_outputs}] = {{{', '.join(output_ptrs)}}};"
        input_arg = f"&{input_var}"

        if node.inputs[0] in inputs_need_reload:
            temp_name = f"temp_{input_var}"
            input_arg = f"&{temp_name}"

        call = f"nnc_split({input_arg}, {array_name}, {num_outputs}, {axis});"

        # Return multi-line string with array decl and call
        return f"{array_decl}\n    {call}"

    def _generate_attr_as_input_op_call(
        self,
        ctx: CompileContext,
        node: Node,
        inputs_need_reload: dict[str, Union["ReloadPoint", "SpillPoint"]],
        outputs_need_temp: dict[str, Union["ReloadPoint", "SpillPoint"]],
    ) -> str:
        """Generate operator call for ops where attributes became inputs (after onnxsim).

        This handles REDUCE_SUM, REDUCE_MEAN, UNSQUEEZE, TRANSPOSE, TILE
        where ONNX opset 13+ converted some attributes to inputs.

        Args:
            ctx: Compilation context
            node: Node to generate call for
            inputs_need_reload: Map of input_name -> ReloadPoint
            outputs_need_temp: Map of output_name -> SpillPoint

        Returns:
            C code for the operator call (may be multi-line)
        """
        op_type = node.op_type

        # Get input tensor (first input is always the data input)
        input_name = node.inputs[0]
        input_var = ctx.tensor_symbols.get(input_name, input_name)
        if input_name in inputs_need_reload:
            input_arg = f"&temp_{input_var}"
        else:
            input_arg = f"&{input_var}"

        # Get output tensor
        output_var = ctx.tensor_symbols.get(node.outputs[0], node.outputs[0])
        if node.outputs[0] in outputs_need_temp:
            output_arg = f"&temp_{output_var}"
        else:
            output_arg = f"&{output_var}"

        if op_type == OpType.REDUCE_SUM:
            # nnc_reducesum(Tensor* input, Tensor* output, int axis, int keepdims)
            # axis may be from attributes or from second input
            axis_val = node.attrs.get("axes", node.attrs.get("axis", None))
            if axis_val is None and len(node.inputs) > 1:
                # axis is from second input (constant)
                axis_input = node.inputs[1]
                if axis_input in ctx.graph.constants:
                    axis_val = ctx.graph.constants[axis_input]
                    if isinstance(axis_val, list):
                        axis_val = axis_val[0] if len(axis_val) > 0 else -1
                    elif hasattr(axis_val, 'item'):
                        axis_val = axis_val.item()
            if axis_val is None:
                axis_val = -1
            if isinstance(axis_val, list):
                axis_val = axis_val[0] if len(axis_val) > 0 else -1
            keepdims = node.attrs.get("keepdims", 1)
            return f"nnc_reducesum({input_arg}, {output_arg}, {axis_val}, {keepdims});"

        elif op_type == OpType.REDUCE_MEAN:
            # nnc_reducemean(Tensor* input, Tensor* output, int axis, int keepdims)
            axis_val = node.attrs.get("axes", node.attrs.get("axis", None))
            if axis_val is None and len(node.inputs) > 1:
                # axis is from second input (constant)
                axis_input = node.inputs[1]
                if axis_input in ctx.graph.constants:
                    axis_val = ctx.graph.constants[axis_input]
                    if isinstance(axis_val, list):
                        axis_val = axis_val[0] if len(axis_val) > 0 else -1
                    elif hasattr(axis_val, 'item'):
                        axis_val = axis_val.item()
            if axis_val is None:
                axis_val = -1
            if isinstance(axis_val, list):
                axis_val = axis_val[0] if len(axis_val) > 0 else -1
            keepdims = node.attrs.get("keepdims", 1)
            return f"nnc_reducemean({input_arg}, {output_arg}, {axis_val}, {keepdims});"

        elif op_type == OpType.UNSQUEEZE:
            # nnc_unsqueeze(Tensor* input, Tensor* output, int axis)
            # axes may be from attributes or from second input
            axis_val = node.attrs.get("axes", None)
            if axis_val is None and len(node.inputs) > 1:
                # axes is from second input (constant)
                axes_input = node.inputs[1]
                if axes_input in ctx.graph.constants:
                    axes_val = ctx.graph.constants[axes_input]
                    if isinstance(axes_val, list):
                        axis_val = axes_val[0] if len(axes_val) > 0 else -1
                    elif hasattr(axes_val, 'item'):
                        axis_val = axes_val.item()
                    else:
                        axis_val = axes_val
            if axis_val is None:
                axis_val = -1
            if isinstance(axis_val, list):
                axis_val = axis_val[0] if len(axis_val) > 0 else -1
            return f"nnc_unsqueeze({input_arg}, {output_arg}, {axis_val});"

        elif op_type == OpType.RESHAPE:
            # nnc_reshape(Tensor* input, Tensor* output, int64_t* shape, int ndim)
            # shape may be from attributes or from second input
            shape = node.attrs.get("shape", None)
            if shape is None and len(node.inputs) > 1:
                # shape is from second input (constant)
                shape_input = node.inputs[1]
                if shape_input in ctx.graph.constants:
                    shape = ctx.graph.constants[shape_input]
                    if hasattr(shape, 'tolist'):
                        shape = shape.tolist()
                    elif not isinstance(shape, list):
                        shape = [shape]
            if shape is None:
                shape = []
            ndim = len(shape)
            shape_name = f"{ctx.node_symbols.get(node.name, node.name)}_shape"
            shape_decl = f"static const int64_t {shape_name}[] = {{{', '.join(map(str, shape))}}};"
            return f"{shape_decl}\n    nnc_reshape({input_arg}, {output_arg}, (int64_t*){shape_name}, {ndim});"

        elif op_type == OpType.TRANSPOSE:
            # nnc_transpose(Tensor* input, Tensor* output, int64_t* perm, int ndim)
            # perm may be from attributes or from second input
            perm = node.attrs.get("perm", None)
            if perm is None and len(node.inputs) > 1:
                # perm is from second input (constant)
                perm_input = node.inputs[1]
                if perm_input in ctx.graph.constants:
                    perm = ctx.graph.constants[perm_input].tolist()
            if perm is None:
                perm = []
            ndim = len(perm)
            perm_name = f"{ctx.node_symbols.get(node.name, node.name)}_perm"
            perm_decl = f"static const int64_t {perm_name}[] = {{{', '.join(map(str, perm))}}};"
            return f"{perm_decl}\n    nnc_transpose({input_arg}, {output_arg}, (int64_t*){perm_name}, {ndim});"

        elif op_type == OpType.TILE:
            # nnc_tile(Tensor* input, Tensor* output, int64_t* repeats, int ndim)
            # repeats is from second input (constant)
            if len(node.inputs) > 1:
                repeats_input = node.inputs[1]
                if repeats_input in ctx.graph.constants:
                    repeats = ctx.graph.constants[repeats_input]
                    if isinstance(repeats, list):
                        repeats_list = repeats
                    elif hasattr(repeats, 'tolist'):
                        repeats_list = repeats.tolist()
                    else:
                        repeats_list = [repeats]
                    ndim = len(repeats_list)
                    repeats_name = f"{ctx.node_symbols.get(node.name, node.name)}_repeats"
                    repeats_decl = f"static const int64_t {repeats_name}[] = {{{', '.join(map(str, repeats_list))}}};"
                    return f"{repeats_decl}\n    nnc_tile({input_arg}, {output_arg}, (int64_t*){repeats_name}, {ndim});"
            # Fallback
            return f"nnc_tile({input_arg}, {output_arg}, NULL, 0);"

        elif op_type == OpType.POW:
            # nnc_pow(Tensor* input, Tensor* output) - computes x^2
            # ONNX Pow has two inputs (base, exponent), but nnc_pow only has input (implicitly x^2)
            # We only use the first input (base) and ignore the exponent
            return f"nnc_pow({input_arg}, {output_arg});"

        elif op_type == OpType.CLIP:
            # nnc_clip(Tensor* input, Tensor* output, float min_val, float max_val)
            # Clip may have min/max as inputs (after onnx opset 11) or as attributes
            min_val = node.attrs.get("min", None)
            max_val = node.attrs.get("max", None)

            # Try to get min from inputs (3rd input in ONNX opset 11+)
            if min_val is None and len(node.inputs) >= 2:
                min_input = node.inputs[1]
                if min_input in ctx.graph.constants:
                    min_val = ctx.graph.constants[min_input]
                    if hasattr(min_val, 'item'):
                        min_val = min_val.item()

            # Try to get max from inputs (4th input in ONNX opset 11+)
            if max_val is None and len(node.inputs) >= 3:
                max_input = node.inputs[2] if len(node.inputs) > 2 else node.inputs[-1]
                if max_input in ctx.graph.constants:
                    max_val = ctx.graph.constants[max_input]
                    if hasattr(max_val, 'item'):
                        max_val = max_val.item()

            if min_val is None:
                min_str = "-1e30f"
            else:
                min_str = f"{float(min_val)}f"

            if max_val is None:
                max_str = "1e30f"
            else:
                max_str = f"{float(max_val)}f"

            return f"nnc_clip({input_arg}, {output_arg}, {min_str}, {max_str});"

        return f"/* Unknown operator {op_type} */"

    def _generate_header(self, ctx: CompileContext) -> str:
        """Generate header file."""
        lines = [
            "/* Auto-generated by NTC - DO NOT EDIT */",
            "#ifndef MODEL_H",
            "#define MODEL_H",
            "",
            "#include <stdint.h>",
            '#include "nnc_types.h"',
            "",
            "/* Tensor declarations */",
        ]

        # Declare all non-constant tensors
        for tensor_name in ctx.graph.inputs:
            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            lines.append(f"extern Tensor {var_name};")

        for tensor_name, tensor in ctx.graph.tensors.items():
            if tensor_name in ctx.graph.inputs or tensor_name in ctx.graph.constants:
                continue
            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            lines.append(f"extern Tensor {var_name};")

        for tensor_name in ctx.graph.outputs:
            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            if f"extern Tensor {var_name};" not in lines:
                lines.append(f"extern Tensor {var_name};")

        # Declare constant tensors (defined in constants_loader.c)
        for tensor_name in ctx.graph.constants:
            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            if f"extern Tensor {var_name};" not in lines:
                lines.append(f"extern Tensor {var_name};")

        # Declare constant data arrays for use in shape operations
        import numpy as np
        for tensor_name, arr in ctx.graph.constants.items():
            # Only declare INT64 arrays (used for shapes)
            if arr.dtype == np.int64 or arr.dtype == np.int32:
                var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
                dtype = "int64_t" if arr.dtype == np.int64 else "int32_t"
                size = arr.size
                lines.append(f"extern const {dtype} {var_name}_data[{size}];")

        lines.extend([
            "",
            "/* Constants loading function */",
            "int nnc_load_constants(const char* path);",
            "",
            "/* Main inference function */",
            "void nnc_run(void);",
        ])

        entry_point = self._get_public_entry_point(ctx)
        if entry_point != "nnc_run":
            lines.extend([
                "",
                "/* Public entry point alias */",
                f"void {entry_point}(void);",
            ])

        lines.extend([
            "",
            "#endif  // MODEL_H",
        ])

        return "\n".join(lines)

    def _generate_constants_binary(self, ctx: CompileContext) -> tuple[bytes, dict[str, Any]]:
        """Generate constants as binary file with metadata.

        Binary format:
        - Header: magic number (4 bytes), version (4 bytes), num_constants (4 bytes)
        - For each constant:
            - name_len (4 bytes), name (null-terminated string)
            - dtype (4 bytes: 0=float32, 1=float16, 2=int32, 3=int8, 4=uint8, 5=bool)
            - ndim (4 bytes)
            - shape (ndim * 8 bytes, int64_t each)
            - nbytes (8 bytes)
            - data (nbytes)

        Returns:
            Tuple of (binary_data, metadata_dict)
        """
        import struct
        import numpy as np

        MAGIC = b'NNCB'
        VERSION = 1

        # Build metadata dict for C code generation
        metadata: dict[str, Any] = {}

        # First pass: calculate offsets
        offset = 12  # header: magic (4) + version (4) + num_constants (4)

        constant_entries: list[dict[str, Any]] = []
        for name, arr in ctx.graph.constants.items():
            var_name = ctx.tensor_symbols.get(name, name)
            dtype, element_size, nnc_dtype = self._get_constant_type_info(arr)
            shape = list(arr.shape)
            ndim = len(shape)
            nbytes = arr.nbytes

            # Map dtype to enum
            dtype_enum = {
                "float": 0,
                "double": 0,
                "uint16_t": 1,
                "int64_t": 2,
                "int32_t": 3,
                "int8_t": 4,
                "uint8_t": 5,
            }.get(dtype, 0)

            # Calculate entry size
            name_bytes = var_name.encode('utf-8')
            name_len = len(name_bytes) + 1  # null-terminated

            entry_header_size = 4 + name_len + 4 + 4 + (ndim * 8) + 8
            # name_len + name + dtype + ndim + shape + nbytes

            data_offset = offset + entry_header_size

            constant_entries.append({
                'name': var_name,
                'original_name': name,
                'dtype': dtype,
                'dtype_enum': dtype_enum,
                'nnc_dtype': nnc_dtype,
                'shape': shape,
                'ndim': ndim,
                'nbytes': nbytes,
                'element_size': element_size,
                'data': arr.tobytes(),
                'offset': data_offset,
            })

            metadata[var_name] = {
                'dtype': dtype,
                'nnc_dtype': nnc_dtype,
                'shape': shape,
                'ndim': ndim,
                'nbytes': nbytes,
                'offset': data_offset,
            }

            offset = data_offset + nbytes

        # Build binary data with interleaved headers and data
        parts: list[bytes] = []
        parts.append(MAGIC)
        parts.append(struct.pack('<I', VERSION))
        parts.append(struct.pack('<I', len(constant_entries)))

        for entry in constant_entries:
            name = str(entry['name'])
            name_bytes = name.encode('utf-8')
            parts.append(struct.pack('<I', len(name_bytes) + 1))  # name_len
            parts.append(name_bytes + b'\x00')  # name (null-terminated)
            parts.append(struct.pack('<I', int(entry['dtype_enum'])))  # dtype
            parts.append(struct.pack('<I', int(entry['ndim'])))  # ndim
            # shape
            shape = entry['shape']
            if isinstance(shape, list):
                for dim in shape:
                    parts.append(struct.pack('<q', int(dim)))
            parts.append(struct.pack('<Q', int(entry['nbytes'])))  # nbytes
            # Data immediately follows header
            data = entry['data']
            if isinstance(data, bytes):
                parts.append(data)
            else:
                parts.append(bytes(data))

        return b''.join(parts), metadata

    def _generate_constants_loader(self, ctx: CompileContext, metadata: dict[str, Any]) -> str:
        """Generate C code to load constants from binary file."""
        lines = [
            "/* Auto-generated by NNC - DO NOT EDIT */",
            '#include "nnc_types.h"',
            '#include <stdio.h>',
            '#include <stdlib.h>',
            '#include <string.h>',
            "",
            "/* Constants data loaded from binary file */",
            "",
        ]

        import numpy as np

        # Generate static data buffers for each constant
        for name, arr in ctx.graph.constants.items():
            var_name = ctx.tensor_symbols.get(name, name)
            dtype, element_size, nnc_dtype = self._get_constant_type_info(arr)
            size = arr.size

            # For INT64/INT32 constants (used as shapes), make data array non-static
            is_shape_constant = (arr.dtype == np.int64 or arr.dtype == np.int32)
            storage_class = "" if is_shape_constant else "static "

            lines.append(f"/* Constant: {name} */")
            lines.append(f"{storage_class}{dtype} {var_name}_data[{size}];")
            lines.append("")

        lines.append("/* Tensor structures */")
        for name, arr in ctx.graph.constants.items():
            var_name = ctx.tensor_symbols.get(name, name)
            shape = list(arr.shape)
            dtype, element_size, nnc_dtype = self._get_constant_type_info(arr)
            # Handle symbolic dimensions (strings) by converting to -1
            shape_str = ", ".join(str(s) if isinstance(s, (int, float)) else "-1" for s in shape)

            lines.append(f"static const int64_t {var_name}_shape[] = {{{shape_str}}};")
            lines.append(f"Tensor {var_name} = {{")
            lines.append(f"    .data = (void*){var_name}_data,")
            lines.append(f"    .dtype = {nnc_dtype},")
            lines.append(f"    .shape = (int64_t*){var_name}_shape,")
            lines.append(f"    .ndim = {len(shape)},")
            lines.append(f"    .nbytes = {arr.nbytes},")
            lines.append("};")
            lines.append("")

        # Generate loading code - one section per constant in order
        lines.extend([
            "/* Load constants from binary file */",
            "int nnc_load_constants(const char* path) {",
            "    FILE* f = fopen(path, \"rb\");",
            "    if (!f) {",
            "        fprintf(stderr, \"Failed to open constants file: %s\\n\", path);",
            "        return -1;",
            "    }",
            "",
            "    /* Check magic number */",
            "    char magic[4];",
            "    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, \"NNCB\", 4) != 0) {",
            "        fprintf(stderr, \"Invalid constants file format\\n\");",
            "        fclose(f);",
            "        return -1;",
            "    }",
            "",
            "    /* Check version */",
            "    uint32_t version;",
            "    if (fread(&version, sizeof(uint32_t), 1, f) != 1) {",
            "        fprintf(stderr, \"Failed to read version\\n\");",
            "        fclose(f);",
            "        return -1;",
            "    }",
            "    if (version != 1) {",
            "        fprintf(stderr, \"Unsupported version: %u\\n\", version);",
            "        fclose(f);",
            "        return -1;",
            "    }",
            "",
            "    /* Get number of constants */",
            "    uint32_t num_constants;",
            "    if (fread(&num_constants, sizeof(uint32_t), 1, f) != 1) {",
            "        fprintf(stderr, \"Failed to read num_constants\\n\");",
            "        fclose(f);",
            "        return -1;",
            "    }",
            "",
            f"    (void)num_constants;  /* Will be implicitly checked by loading each constant */",
            "",
        ])

        # Generate a loading block for each constant - they must match the file order
        for idx, (name, arr) in enumerate(ctx.graph.constants.items()):
            var_name = ctx.tensor_symbols.get(name, name)
            dtype, element_size, nnc_dtype = self._get_constant_type_info(arr)
            ndim = len(arr.shape)
            # Handle symbolic dimensions (strings) by converting to -1
            shape_str = ", ".join(str(s) if isinstance(s, (int, float)) else "-1" for s in arr.shape)
            data_size = arr.nbytes

            lines.extend([
                f"    /* Load constant {idx}: {name} */",
                f"    {{",
                f"        /* Read name */",
                f"        uint32_t name_len;",
                f"        if (fread(&name_len, sizeof(uint32_t), 1, f) != 1) {{",
                f"            fprintf(stderr, \"Failed to read name_len for constant {idx}\\\\n\");",
                f"            fclose(f);",
                f"            return -1;",
                f"        }}",
                f"        char name[256];",
                f"        if (name_len >= sizeof(name)) {{",
                f"            fprintf(stderr, \"Name too long for constant {idx}: %u\\\\n\", name_len);",
                f"            fclose(f);",
                f"            return -1;",
                f"        }}",
                f"        if (fread(name, 1, name_len, f) != name_len) {{",
                f"            fprintf(stderr, \"Failed to read name for constant {idx}\\\\n\");",
                f"            fclose(f);",
                f"            return -1;",
                f"        }}",
                f"        if (strcmp(name, \"{var_name}\") != 0) {{",
                f"            fprintf(stderr, \"Name mismatch for constant {idx}: expected '{var_name}', got '%s'\\\\n\", name);",
                f"            fclose(f);",
                f"            return -1;",
                f"        }}",
                f"        /* Skip dtype, ndim, shape - we know them from codegen */",
                f"        fseek(f, 4, SEEK_CUR);  /* dtype */",
                f"        fseek(f, 4, SEEK_CUR);  /* ndim */",
                f"        fseek(f, {ndim * 8}, SEEK_CUR);  /* shape ({ndim} dims) */",
                f"        fseek(f, 8, SEEK_CUR);  /* nbytes */",
                f"        /* Read data */",
                f"        if (fread({var_name}_data, 1, {data_size}, f) != {data_size}) {{",
                f"            fprintf(stderr, \"Failed to read data for {var_name}\\\\n\");",
                f"            fclose(f);",
                f"            return -1;",
                f"        }}",
                f"    }}",
                f"",
            ])

        lines.extend([
            "    fclose(f);",
            "    return 0;",
            "}",
        ])

        return "\n".join(lines)

    def _get_constant_type_info(self, arr: np.ndarray) -> tuple[str, int, str]:
        """Get C type, element size, and NNC dtype enum for a constant array.

        Returns:
            Tuple of (c_dtype, element_size, nnc_dtype_enum)
        """
        import numpy as np

        if arr.dtype == np.float32:
            return "float", 4, "NNC_DTYPE_FLOAT32"
        elif arr.dtype == np.float64:
            return "double", 8, "NNC_DTYPE_FLOAT32"
        elif arr.dtype == np.float16:
            return "uint16_t", 2, "NNC_DTYPE_FLOAT16"
        elif arr.dtype == np.int64:
            return "int64_t", 8, "NNC_DTYPE_INT64"
        elif arr.dtype == np.int32:
            return "int32_t", 4, "NNC_DTYPE_INT32"
        elif arr.dtype == np.int8:
            return "int8_t", 1, "NNC_DTYPE_INT8"
        elif arr.dtype == np.uint8:
            return "uint8_t", 1, "NNC_DTYPE_UINT8"
        elif arr.dtype == np.bool_:
            return "uint8_t", 1, "NNC_DTYPE_BOOL"
        else:
            # Default to float32
            return "float", 4, "NNC_DTYPE_FLOAT32"

    def _generate_makefile(self, ctx: CompileContext) -> str:
        """Generate Makefile."""
        has_constants = bool(ctx.graph.constants)
        objs = "model.o tensors.o"
        if has_constants:
            objs += " constants_loader.o"
        objs += " test_runner.o ops.o"
        runtime_dir = self._default_runtime_dir()

        # Different makefile content based on whether we have constants
        if has_constants:
            makefile_body = f"""# Auto-generated by NNC - DO NOT EDIT
# Set NNC_RUNTIME_PATH to point to the runtime directory if not in dev tree
CC = gcc
CFLAGS = -D_GNU_SOURCE -std=c11 -O2 -Wall -Wextra -pthread
LDFLAGS = -lm -pthread

# Runtime include path - can be overridden by environment
NNC_RUNTIME ?= {runtime_dir}
CFLAGS += -I$(NNC_RUNTIME)/include

.PHONY: all clean run

all: model

model: {objs}
\t$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ops.o:
\t$(CC) $(CFLAGS) -c $(NNC_RUNTIME)/x86/ops.c

%.o: %.c
\t$(CC) $(CFLAGS) -c $<

clean:
\trm -f *.o model

run: model
\t./model
"""
        else:
            makefile_body = f"""# Auto-generated by NNC - DO NOT EDIT
# Set NNC_RUNTIME_PATH to point to the runtime directory if not in dev tree
CC = gcc
CFLAGS = -D_GNU_SOURCE -std=c11 -O2 -Wall -Wextra -pthread
LDFLAGS = -lm -pthread

# Runtime include path - can be overridden by environment
NNC_RUNTIME ?= {runtime_dir}
CFLAGS += -I$(NNC_RUNTIME)/include

.PHONY: all clean run

all: model

model: {objs}
\t$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ops.o:
\t$(CC) $(CFLAGS) -c $(NNC_RUNTIME)/x86/ops.c

%.o: %.c
\t$(CC) $(CFLAGS) -c $<

clean:
\trm -f *.o model

run: model
\t./model
"""

        return makefile_body

    def _default_runtime_dir(self) -> str:
        """Return the default runtime directory to embed into generated builds."""
        runtime_dir = Path(__file__).resolve().parents[3] / "runtime"
        return str(runtime_dir)

    def _prune_unused_tensor_defs(self, ctx: CompileContext) -> None:
        """Drop dead tensor definitions left behind by earlier graph rewrites."""
        used_tensor_names = set(ctx.graph.inputs) | set(ctx.graph.outputs) | set(ctx.graph.constants)
        for node in ctx.graph.nodes.values():
            used_tensor_names.update(node.inputs)
            used_tensor_names.update(node.outputs)

        dead_tensor_names = [
            tensor_name
            for tensor_name in ctx.graph.tensors
            if tensor_name not in used_tensor_names
        ]
        for tensor_name in dead_tensor_names:
            ctx.graph.tensors.pop(tensor_name, None)

    def _build_tensor_offsets_from_scheduled_plan(
        self,
        ctx: CompileContext,
        scheduled_plan: Any,
    ) -> dict[str, tuple[str, int]]:
        tensor_offsets: dict[str, tuple[str, int]] = {}
        best_fast_allocations = self._best_scheduled_fast_allocations(ctx, scheduled_plan)

        for graph_tensor_name, (_, offset, _) in best_fast_allocations.items():
            tensor_offsets[graph_tensor_name] = ("fast", offset)

        for value_name, allocation in getattr(scheduled_plan, "slow_allocations", {}).items():
            graph_tensor_name = self._resolve_schedule_value_graph_tensor_name(
                ctx,
                str(value_name),
            )
            if graph_tensor_name is None or graph_tensor_name in tensor_offsets:
                continue
            tensor_offsets[graph_tensor_name] = ("slow", int(allocation.offset))

        return tensor_offsets

    def _best_scheduled_fast_allocations(
        self,
        ctx: CompileContext,
        scheduled_plan: Any,
    ) -> dict[str, tuple[tuple[int, int, str], int, int]]:
        best_fast_allocations: dict[str, tuple[tuple[int, int, str], int, int]] = {}
        schedule_value_map = self._build_schedule_value_map(ctx)

        for allocation in getattr(scheduled_plan, "fast_allocations", {}).values():
            value_name = str(allocation.value_name)
            graph_tensor_name = self._resolve_schedule_value_graph_tensor_name(
                ctx,
                value_name,
            )
            if graph_tensor_name is None:
                continue
            sort_key = (
                int(getattr(allocation, "start_time", 0)),
                int(getattr(allocation, "end_time", 0)),
                str(getattr(allocation, "residency_id", "")),
            )
            scheduled_value = schedule_value_map.get(value_name)
            size_bytes = int(getattr(scheduled_value, "size_bytes", 0))
            if size_bytes <= 0:
                tensor = ctx.graph.tensors.get(graph_tensor_name)
                size_bytes = int(tensor.byte_size()) if tensor is not None else 0
            current = best_fast_allocations.get(graph_tensor_name)
            if current is None or sort_key < current[0]:
                best_fast_allocations[graph_tensor_name] = (
                    sort_key,
                    int(allocation.offset),
                    max(size_bytes, 0),
                )

        return best_fast_allocations

    def _scheduled_native_direct_fast_tensor_names(
        self,
        ctx: CompileContext,
        scheduled_plan: Any | None,
    ) -> set[str]:
        if scheduled_plan is None:
            return set()

        execution_plans = ctx.metadata.get("node_execution_plans")
        if not isinstance(execution_plans, dict):
            execution_plans = {}

        streaming_metadata = self._scheduled_tile_streaming_metadata(ctx)
        streamed_node_names = set(streaming_metadata["streamed_node_names"])
        best_fast_allocations = self._best_scheduled_fast_allocations(ctx, scheduled_plan)
        safe_tensors: set[str] = set()

        for tensor_name, tensor in ctx.graph.tensors.items():
            if tensor_name in ctx.graph.inputs or tensor_name in ctx.graph.constants:
                continue

            producers = [
                producer
                for producer in ctx.graph.get_producers(tensor_name)
                if producer.op_type != OpType.CONSTANT
            ]
            if len(producers) != 1:
                continue

            producer = producers[0]
            plan = execution_plans.get(producer.name)
            if plan is not None:
                op_family = getattr(plan, "op_family", None)
                if op_family not in {"gemm", "average_pool", "global_average_pool"}:
                    if self._node_execution_plan_uses_tiled_storage(plan):
                        continue
                elif op_family in {"average_pool", "global_average_pool"} and producer.name not in streamed_node_names:
                    continue

            selected_fast = best_fast_allocations.get(tensor_name)
            if selected_fast is None:
                continue
            if int(selected_fast[2]) < int(tensor.byte_size()):
                continue

            safe_tensors.add(tensor_name)

        return safe_tensors

    def _generate_tensors(self, ctx: CompileContext) -> str:
        """Generate tensors definition file."""
        lines = [
            "/* Auto-generated by NNC - DO NOT EDIT */",
            '#include "nnc_types.h"',
            "",
            "#ifndef NNC_MEMORY_ALIGNMENT",
            "#define NNC_MEMORY_ALIGNMENT 16",
            "#endif",
            "",
        ]

        # Check for new memory allocation plan
        from nnc_py.passes.memory_planning import get_memory_allocation_plan
        alloc_plan = get_memory_allocation_plan(ctx)
        scheduled_plan = self._get_scheduled_memory_plan(ctx)
        prefer_scheduled_plan = self._prefer_scheduled_memory_plan(ctx, scheduled_plan)
        tile_aware_runtime_plan = self._get_tile_aware_runtime_plan(ctx, alloc_plan)
        tile_aware_tensor_bindings = tile_aware_runtime_plan.get("tensor_bindings", {})
        scheduled_tile_streaming_internal_tensors = self._scheduled_tile_streaming_internal_tensor_names(
            ctx
        )
        scheduled_direct_fast_tensors = self._scheduled_native_direct_fast_tensor_names(
            ctx,
            scheduled_plan,
        )
        fallback_to_linear_storage = (
            not prefer_scheduled_plan
            and alloc_plan is not None
            and alloc_plan.strategy_name == "tile_regions_v3"
            and not tile_aware_runtime_plan
        )

        has_slow_memory_tensors = False
        has_logical_regions = False
        if prefer_scheduled_plan and scheduled_plan is not None:
            has_slow_memory_tensors = bool(scheduled_plan.slow_allocations)
        elif alloc_plan is not None:
            has_slow_memory_tensors = any(
                alloc.is_spilled for alloc in alloc_plan.tensor_allocations.values()
            )
            has_logical_regions = bool(alloc_plan.logical_regions) and bool(tile_aware_runtime_plan)
        has_spill = (
            bool(getattr(scheduled_plan, "transfer_points", ()))
            if prefer_scheduled_plan and scheduled_plan is not None
            else alloc_plan is not None and alloc_plan.has_spill
        )
        has_moves = False if prefer_scheduled_plan else alloc_plan is not None and bool(alloc_plan.move_points)
        uses_unified_runtime = (
            bool(scheduled_plan.fast_allocations)
            or has_slow_memory_tensors
            or has_spill
        ) if prefer_scheduled_plan and scheduled_plan is not None else (
            alloc_plan is not None and (has_spill or has_slow_memory_tensors or has_moves)
        )
        uses_fast_pool_symbol = (
            bool(scheduled_plan.fast_allocations)
            or has_slow_memory_tensors
            or has_spill
        ) if prefer_scheduled_plan and scheduled_plan is not None else (
            alloc_plan is not None and (
                has_spill or has_slow_memory_tensors or has_moves or has_logical_regions
            )
        )

        # Generate slow pool if we have spill points OR slow memory tensors
        needs_slow_pool = has_spill or has_slow_memory_tensors

        # Check if memory planning was performed
        has_memory_plan = (
            prefer_scheduled_plan
            or alloc_plan is not None
            or "memory_plan" in ctx.metadata
        )

        if has_memory_plan:
            # Generate static memory pool(s)
            lines.extend(self._generate_memory_pool(ctx))
            lines.append("")

        # Determine which pool names to use
        if uses_fast_pool_symbol:
            fast_pool_name = "_nnc_fast_pool"
            slow_pool_name = "_nnc_slow_pool" if needs_slow_pool else None
        else:
            fast_pool_name = "_nnc_memory_pool"
            slow_pool_name = None

        # Get tensor offsets from allocation plan
        tensor_offsets = {}

        if prefer_scheduled_plan and scheduled_plan is not None:
            tensor_offsets = self._build_tensor_offsets_from_scheduled_plan(
                ctx,
                scheduled_plan,
            )
        elif fallback_to_linear_storage:
            tensor_offsets, _ = self._build_linear_tensor_fallback(ctx)
        elif alloc_plan is not None:
            # Use new MemoryAllocationPlan
            for tensor_name, alloc in alloc_plan.tensor_allocations.items():
                if alloc.is_spilled:
                    # Spilled tensors go to slow memory
                    tensor_offsets[tensor_name] = ("slow", alloc.offset)
                else:
                    # Non-spilled tensors go to fast memory
                    # Use alloc.offset which is the tensor's offset within the buffer
                    tensor_offsets[tensor_name] = ("fast", alloc.offset)
        elif "memory_plan" in ctx.metadata:
            # Use legacy memory plan
            from nnc_py.passes.memory_plan import get_memory_plan
            from nnc_py.passes.spill import get_spill_plan
            plan = get_memory_plan(ctx)
            spill_plan = get_spill_plan(ctx)

            spilled_tensors: set[str] = set()
            if spill_plan is not None and spill_plan.has_overflow:
                # Has spill from legacy plan
                spilled_tensors = set(spill_plan.spilled_tensors)

            for tensor_name, mem_info in plan.tensor_info.items():
                pool_type = "slow" if tensor_name in spilled_tensors else "fast"
                tensor_offsets[tensor_name] = (pool_type, mem_info.pool_offset)

        # Define all non-constant tensors
        emitted_tile_aware_buffers: set[str] = set()
        for tensor_name, tensor in ctx.graph.tensors.items():
            if tensor_name in ctx.graph.constants:
                continue

            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            shape_list = tensor.shape.dims
            # Handle symbolic dimensions (strings) by converting to -1
            shape_init = ", ".join(str(d) if isinstance(d, (int, float)) else "-1" for d in shape_list)

            data_init = "NULL"
            force_detached_home_tensor = (
                prefer_scheduled_plan
                and self._should_use_scheduled_home_execution(ctx)
                and tensor_name not in ctx.graph.inputs
                and tensor_name not in scheduled_tile_streaming_internal_tensors
                and tensor_name not in scheduled_direct_fast_tensors
            )
            if not force_detached_home_tensor and tensor_name in tile_aware_tensor_bindings:
                binding = tile_aware_tensor_bindings[tensor_name]
                if binding["kind"] in {"input_staging", "static_buffer"}:
                    buffer_name = binding["symbol"]
                    if buffer_name not in emitted_tile_aware_buffers:
                        buffer_label = (
                            "Input staging buffer"
                            if binding["kind"] == "input_staging"
                            else "Tile-aware tensor buffer"
                        )
                        lines.append(f"/* {buffer_label}: {tensor_name} */")
                        lines.append(
                            f"uint8_t {buffer_name}[{tensor.byte_size()}] "
                            f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};"
                        )
                        lines.append("")
                        emitted_tile_aware_buffers.add(buffer_name)
                    data_init = buffer_name
                elif binding["kind"] == "fast_pool":
                    data_init = f"{fast_pool_name} + {binding['offset']}"
            elif not force_detached_home_tensor and tensor_name in tensor_offsets:
                pool_type, offset = tensor_offsets[tensor_name]
                if uses_unified_runtime and tensor_name in ctx.graph.inputs:
                    input_buffer_name = f"_nnc_input_buffer_{var_name}"
                    lines.append(f"/* Input staging buffer: {tensor_name} */")
                    lines.append(
                        f"uint8_t {input_buffer_name}[{tensor.byte_size()}] "
                        f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};"
                    )
                    lines.append("")
                    data_init = input_buffer_name
                else:
                    if pool_type == "slow":
                        pool_to_use = slow_pool_name
                    else:
                        pool_to_use = fast_pool_name
                    data_init = f"{pool_to_use} + {offset}" if pool_to_use else "NULL"

            if data_init == "NULL":
                detached_buffer_name = f"_nnc_tensor_buffer_{var_name}"
                if detached_buffer_name not in emitted_tile_aware_buffers:
                    lines.append(f"/* Detached tensor buffer: {tensor_name} */")
                    lines.append(
                        f"uint8_t {detached_buffer_name}[{tensor.byte_size()}] "
                        f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};"
                    )
                    lines.append("")
                    emitted_tile_aware_buffers.add(detached_buffer_name)
                data_init = detached_buffer_name

            lines.append(f"/* Tensor: {tensor_name} */")
            lines.append(f"static int64_t {var_name}_shape[] = {{{shape_init}}};")

            # Map dtype
            dtype_enum = self._map_dtype_to_enum(tensor.dtype)

            lines.append(f"Tensor {var_name} = {{")
            lines.append(f"    .data = {data_init},")
            lines.append(f"    .dtype = {dtype_enum},")
            lines.append(f"    .shape = {var_name}_shape,")
            lines.append(f"    .ndim = {len(shape_list)},")
            lines.append(f"    .nbytes = {tensor.byte_size()},")
            lines.append("};")
            lines.append("")

        return "\n".join(lines)

    def _generate_memory_pool(self, ctx: CompileContext) -> List[str]:
        """Generate static memory pool declaration."""
        from nnc_py.passes.memory_planning import get_memory_allocation_plan
        from nnc_py.passes.memory_plan import get_memory_plan
        from nnc_py.passes.spill import get_spill_plan

        # Check for new memory allocation plan first
        alloc_plan = get_memory_allocation_plan(ctx)
        scheduled_plan = self._get_scheduled_memory_plan(ctx)
        prefer_scheduled_plan = self._prefer_scheduled_memory_plan(ctx, scheduled_plan)
        tile_aware_runtime_plan = self._get_tile_aware_runtime_plan(ctx, alloc_plan)

        if prefer_scheduled_plan and scheduled_plan is not None:
            fast_memory_size = max(int(scheduled_plan.total_fast_memory), 1)
            fast_memory_size = max(
                fast_memory_size,
                self._scheduled_tile_streaming_required_fast_memory(ctx),
            )
            if self._should_use_scheduled_home_execution(ctx):
                requested_max_memory = ctx.metadata.get("max_memory")
                if isinstance(requested_max_memory, int) and requested_max_memory > 0:
                    fast_memory_size = min(fast_memory_size, requested_max_memory)
            transfer_count = len(getattr(scheduled_plan, "transfer_points", ()))
            lines = [
                "/* Scheduled Native Memory Pools */",
                f"/* Fast memory: {fast_memory_size} bytes ({fast_memory_size / 1024:.2f} KB) */",
                f"/* Slow memory: {scheduled_plan.total_slow_memory} bytes ({scheduled_plan.total_slow_memory / 1024:.2f} KB) */",
                f"/* Fast allocations: {len(scheduled_plan.fast_allocations)}, Transfer points: {transfer_count} */",
                "",
                "/* Fast Memory Pool (SRAM/On-chip) */",
                f"#define NNC_FAST_MEMORY_SIZE {fast_memory_size}",
                "#define NNC_MEMORY_ALIGNMENT 16",
                f"uint8_t _nnc_fast_pool[NNC_FAST_MEMORY_SIZE] "
                f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};",
                "",
            ]
            if scheduled_plan.total_slow_memory > 0:
                slow_memory_size = max(int(scheduled_plan.total_slow_memory), 1)
                lines.extend([
                    "/* Slow Memory Pool (DRAM/External) */",
                    f"#define NNC_SLOW_MEMORY_SIZE {slow_memory_size}",
                    f"uint8_t _nnc_slow_pool[NNC_SLOW_MEMORY_SIZE] "
                    f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};",
                    "",
                ])
            return lines

        if alloc_plan is not None:
            if alloc_plan.strategy_name == "tile_regions_v3" and not tile_aware_runtime_plan:
                _, fallback_total_size = self._build_linear_tensor_fallback(ctx)
                lines = [
                    "/* Static Memory Pool */",
                    f"/* Fallback size: {fallback_total_size} bytes ({fallback_total_size / 1024:.2f} KB) */",
                    "#define NNC_MEMORY_ALIGNMENT 16",
                    f"#define NNC_MEMORY_SIZE {fallback_total_size}",
                    "static uint8_t _nnc_memory_pool[NNC_MEMORY_SIZE] "
                    "__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {0};",
                    "",
                ]
                return lines

            has_slow_memory_tensors = any(
                alloc.is_spilled for alloc in alloc_plan.tensor_allocations.values()
            )
            has_logical_regions = bool(alloc_plan.logical_regions) and bool(tile_aware_runtime_plan)
            move_count = len(alloc_plan.move_points)
            spill_count = alloc_plan.spill_count
            reload_count = alloc_plan.reload_count
            needs_slow_pool = alloc_plan.has_spill or has_slow_memory_tensors
            uses_unified_runtime = (
                alloc_plan.has_spill
                or has_slow_memory_tensors
                or bool(alloc_plan.move_points)
                or has_logical_regions
            )

            if uses_unified_runtime:
                region_lines = self._generate_logical_region_lines(alloc_plan)
                if needs_slow_pool:
                    fast_memory_size = alloc_plan.total_fast_memory
                    lines = [
                        "/* Dual Memory Pools (Fast + Slow for spilled tensors) */",
                        f"/* Fast memory: {fast_memory_size} bytes ({fast_memory_size / 1024:.2f} KB) */",
                        f"/* Slow memory: {alloc_plan.total_slow_memory} bytes ({alloc_plan.total_slow_memory / 1024:.2f} KB) */",
                        f"/* Buffers: {alloc_plan.num_buffers}, Spill points: {spill_count}, Reload points: {reload_count} */",
                        "",
                        "/* Fast Memory Pool (SRAM/On-chip) */",
                        f"#define NNC_FAST_MEMORY_SIZE {fast_memory_size}",
                        "#define NNC_MEMORY_ALIGNMENT 16",
                    ]
                    lines.extend(region_lines)
                    lines.extend([
                        f"uint8_t _nnc_fast_pool[NNC_FAST_MEMORY_SIZE] "
                        f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};",
                        "",
                        "/* Slow Memory Pool (DRAM/External) */",
                        f"#define NNC_SLOW_MEMORY_SIZE {alloc_plan.total_slow_memory}",
                        f"uint8_t _nnc_slow_pool[NNC_SLOW_MEMORY_SIZE] "
                        f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};",
                        "",
                    ])
                    return lines

                # Move-only unified plans still need the unified fast pool symbol and
                # enough headroom for transient pre-move source offsets.
                max_memory = ctx.metadata.get('max_memory', alloc_plan.total_fast_memory)
                fast_memory_size = max(alloc_plan.total_fast_memory, max_memory)
                lines = [
                    "/* Unified Memory Pools */",
                    f"/* Fast memory: {fast_memory_size} bytes ({fast_memory_size / 1024:.2f} KB) */",
                    f"/* Buffers: {alloc_plan.num_buffers}, Spill points: {spill_count}, Reload points: {reload_count}, Move points: {move_count} */",
                    "",
                    "/* Fast Memory Pool (SRAM/On-chip) */",
                    f"#define NNC_FAST_MEMORY_SIZE {fast_memory_size}",
                    "#define NNC_MEMORY_ALIGNMENT 16",
                    "",
                ]
                lines.extend(region_lines)
                lines.append(
                    f"uint8_t _nnc_fast_pool[NNC_FAST_MEMORY_SIZE] "
                    f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};"
                )
                lines.append("")
                return lines

            lines = [
                "/* Static Memory Pool */",
                f"/* Strategy: {alloc_plan.strategy_name} */",
                f"/* Total size: {alloc_plan.total_fast_memory} bytes ({alloc_plan.total_fast_memory / 1024:.2f} KB) */",
                f"/* Buffers: {alloc_plan.num_buffers}, Logical regions: {len(alloc_plan.logical_regions)} */",
                f"#define NNC_MEMORY_SIZE {alloc_plan.total_fast_memory}",
                "#define NNC_MEMORY_ALIGNMENT 16",
                "static uint8_t _nnc_memory_pool[NNC_MEMORY_SIZE] "
                "__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {0};",
                "",
            ]
            return lines

        # Fall back to legacy implementation
        plan = get_memory_plan(ctx)
        spill_plan = get_spill_plan(ctx)

        # Check if we have spill (overflow)
        has_spill = spill_plan is not None and spill_plan.has_overflow

        if has_spill and spill_plan is not None:
            # Generate dual memory pools
            lines = [
                "/* Dual Memory Pools (Fast + Slow for overflow) */",
                f"/* Fast memory limit: {spill_plan.max_memory} bytes ({spill_plan.max_memory / 1024:.2f} KB) */",
                f"/* Original requirement: {plan.total_size} bytes ({plan.total_size / 1024:.2f} KB) */",
                f"/* Slow memory used: {spill_plan.slow_memory_size} bytes ({spill_plan.slow_memory_size / 1024:.2f} KB) */",
                f"/* Spilled tensors: {len(spill_plan.spilled_tensors)} */",
                "",
                f"/* Fast Memory Pool (SRAM/On-chip) */",
                f"#define NNC_FAST_MEMORY_SIZE {spill_plan.max_memory}",
                f"#define NNC_MEMORY_ALIGNMENT {plan.alignment}",
                f"uint8_t _nnc_fast_pool[NNC_FAST_MEMORY_SIZE] "
                f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};",
                "",
                f"/* Slow Memory Pool (DRAM/External) */",
                f"#define NNC_SLOW_MEMORY_SIZE {spill_plan.slow_memory_size}",
                f"uint8_t _nnc_slow_pool[NNC_SLOW_MEMORY_SIZE] "
                f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};",
                "",
            ]
            return lines

        else:
            # Single memory pool (no overflow)
            if plan is None:
                # Fallback if no plan available
                lines = [
                    "/* Static Memory Pool */",
                    "#define NNC_MEMORY_SIZE 4096",
                    "#define NNC_MEMORY_ALIGNMENT 16",
                    "static uint8_t _nnc_memory_pool[NNC_MEMORY_SIZE] "
                    "__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {0};",
                    "",
                ]
                return lines
            lines = [
                "/* Static Memory Pool */",
                f"/* Total size: {plan.total_size} bytes ({plan.total_size / 1024:.2f} KB) */",
                f"/* Buffers: {plan.num_buffers}, Tensors: {plan.num_tensors} */",
                f"#define NNC_MEMORY_SIZE {plan.total_size}",
                f"#define NNC_MEMORY_ALIGNMENT {plan.alignment}",
                f"static uint8_t _nnc_memory_pool[NNC_MEMORY_SIZE] "
                f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};",
                "",
            ]
            return lines

    def _generate_logical_region_lines(
        self,
        alloc_plan: "MemoryAllocationPlan",
    ) -> List[str]:
        """Emit logical region metadata for tile-aware fast-memory layouts."""
        if not alloc_plan.logical_regions:
            return []

        lines = [
            "/* Tile-aware fast-memory regions (phase 1 metadata only) */",
        ]
        for region in sorted(
            alloc_plan.logical_regions.values(),
            key=lambda logical_region: (logical_region.offset, logical_region.name),
        ):
            macro_name = region.name.upper()
            lines.append(
                f"/* Region {region.name}: offset {region.offset} bytes, size {region.size_bytes} bytes */"
            )
            lines.append(f"#define NNC_{macro_name}_MEMORY_SIZE {region.size_bytes}")
        lines.append("")
        return lines

    def _map_dtype_to_enum(self, dtype: "DataType") -> str:
        """Map IR dtype to NNC dtype enum."""
        from nnc_py.ir.types import DataType

        mapping = {
            DataType.FLOAT32: "NNC_DTYPE_FLOAT32",
            DataType.FLOAT16: "NNC_DTYPE_FLOAT16",
            DataType.INT32: "NNC_DTYPE_INT32",
            DataType.INT64: "NNC_DTYPE_INT64",
            DataType.INT8: "NNC_DTYPE_INT8",
            DataType.UINT8: "NNC_DTYPE_UINT8",
            DataType.BOOL: "NNC_DTYPE_BOOL",
        }
        return mapping.get(dtype, "NNC_DTYPE_FLOAT32")

    def _generate_test_runner(self, ctx: CompileContext) -> str:
        """Generate test runner."""
        scheduled_plan = self._get_scheduled_memory_plan(ctx)
        static_storage_mode = (
            self._prefer_scheduled_memory_plan(ctx, scheduled_plan)
            or ctx.metadata.get("memory_allocation_plan") is not None
            or "memory_plan" in ctx.metadata
        )

        # Debug mode: add file setup for debug output
        debug_decl = ""
        debug_setup = ""
        debug_cleanup = ""
        debug_macros = ""
        if self.debug_mode:
            # Declare debug_file as a global variable (extern declaration in model.c)
            debug_decl = """
/* Global debug file pointer (also declared as extern in model.c) */
FILE* debug_file = NULL;
"""
            debug_setup = """
    /* Open debug output file */
    debug_file = fopen("nnc_debug_output.txt", "w");
    if (!debug_file) {
        fprintf(stderr, "Failed to open debug output file\\n");
        return 1;
    }
"""
            debug_cleanup = """
    /* Close debug output file */
    if (debug_file) {
        fclose(debug_file);
        debug_file = NULL;
    }
    printf("Debug output written to nnc_debug_output.txt\\n");"""
            # Define macros to redirect debug printf to file
            debug_macros = """
/* Debug mode: redirect debug output to file */
#define DEBUG_PRINTF(fmt, ...) fprintf(debug_file, fmt, ##__VA_ARGS__)
#define DEBUG_PRINT_TENSOR_START(name, idx) DEBUG_PRINTF("DEBUG_TENSOR_START %s %d\\n", name, idx)
#define DEBUG_PRINT_SHAPE(ndim) DEBUG_PRINTF("SHAPE %d\\n", ndim)
#define DEBUG_PRINT_DIM(i, val) DEBUG_PRINTF("DIM %d %d\\n", i, val)
#define DEBUG_PRINT_DATA_START() DEBUG_PRINTF("DATA_START\\n")
#define DEBUG_PRINT_VALUE(val) DEBUG_PRINTF("%f\\n", val)
#define DEBUG_PRINT_DATA_END() DEBUG_PRINTF("DATA_END\\n")
#define DEBUG_PRINT_TENSOR_END(name) DEBUG_PRINTF("DEBUG_TENSOR_END %s\\n\\n", name)
"""

        # Generate tensor setup code
        tensor_setups = []

        if static_storage_mode:
            # Static allocation - memory is pre-allocated in memory pool
            # Only need to initialize input tensors with test data
            static_bytes = 0
            if self._prefer_scheduled_memory_plan(ctx, scheduled_plan) and scheduled_plan is not None:
                static_bytes = int(getattr(scheduled_plan, "total_fast_memory", 0))
            elif ctx.metadata.get("memory_allocation_plan") is not None:
                alloc_plan = ctx.metadata["memory_allocation_plan"]
                static_bytes = int(getattr(alloc_plan, "total_fast_memory", 0))
            else:
                from nnc_py.passes.memory_plan import get_memory_plan
                plan = get_memory_plan(ctx)
                static_bytes = int(getattr(plan, "total_size", 0))

            tensor_setups.append(f"    /* Using static memory pool: {static_bytes} bytes */")
            tensor_setups.append("")

            # Setup input tensors - only initialize with test data
            for tensor_name in ctx.graph.inputs:
                tensor = ctx.graph.get_tensor(tensor_name)
                var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
                size = tensor.byte_size()

                tensor_setups.append(f"    /* Initialize {var_name} */")

                # Initialize with test pattern
                num_elements = size // 4
                tensor_setups.append(f"    for (int i = 0; i < {num_elements}; i++) {{")
                tensor_setups.append(f"        ((float*){var_name}.data)[i] = (float)i * 0.01f;  /* Test pattern */")
                tensor_setups.append(f"    }}")
                tensor_setups.append("")
        else:
            # Dynamic allocation - use malloc/calloc
            # Setup input tensors
            for tensor_name in ctx.graph.inputs:
                tensor = ctx.graph.get_tensor(tensor_name)
                var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
                size = tensor.byte_size()

                # Generate shape array
                shape_list = tensor.shape.dims
                # Handle symbolic dimensions (strings) by converting to -1
                shape_init = ", ".join(str(d) if isinstance(d, (int, float)) else "-1" for d in shape_list)

                # Setup code
                tensor_setups.append(f"    /* Setup {var_name} */")
                tensor_setups.append(f"    static int64_t {var_name}_shape[] = {{{shape_init}}};")
                tensor_setups.append(f"    {var_name}.shape = {var_name}_shape;")
                tensor_setups.append(f"    {var_name}.ndim = {len(shape_list)};")
                tensor_setups.append(f"    {var_name}.dtype = NNC_DTYPE_FLOAT32;")
                tensor_setups.append(f"    {var_name}.nbytes = {size};")
                tensor_setups.append(f"    {var_name}.data = malloc({size});")
                tensor_setups.append(f"    if (!{var_name}.data) {{ fprintf(stderr, \"Failed to allocate {var_name}\\\\n\"); return 1; }}")

                # Initialize with test pattern
                num_elements = size // 4
                tensor_setups.append(f"    for (int i = 0; i < {num_elements}; i++) {{")
                tensor_setups.append(f"        ((float*){var_name}.data)[i] = (float)i * 0.01f;  /* Test pattern */")
                tensor_setups.append(f"    }}")
                tensor_setups.append("")

            # Setup intermediate and output tensors
            for tensor_name, tensor in ctx.graph.tensors.items():
                if tensor_name in ctx.graph.inputs or tensor_name in ctx.graph.constants:
                    continue

                var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
                size = tensor.byte_size()
                shape_list = tensor.shape.dims
                # Handle symbolic dimensions (strings) by converting to -1
                shape_init = ", ".join(str(d) if isinstance(d, (int, float)) else "-1" for d in shape_list)

                tensor_setups.append(f"    /* Setup {var_name} */")
                tensor_setups.append(f"    static int64_t {var_name}_shape[] = {{{shape_init}}};")
                tensor_setups.append(f"    {var_name}.shape = {var_name}_shape;")
                tensor_setups.append(f"    {var_name}.ndim = {len(shape_list)};")
                tensor_setups.append(f"    {var_name}.dtype = NNC_DTYPE_FLOAT32;")
                tensor_setups.append(f"    {var_name}.nbytes = {size};")
                tensor_setups.append(f"    {var_name}.data = calloc({size // 4}, sizeof(float));  /* Initialize to zero */")
                tensor_setups.append(f"    if (!{var_name}.data) {{ fprintf(stderr, \"Failed to allocate {var_name}\\\\n\"); return 1; }}")
                tensor_setups.append("")

        setup_code = "\n".join(tensor_setups)

        # Free memory - only for dynamic allocation
        if static_storage_mode:
            frees_code = "    /* Static allocation - no free needed */"
        else:
            tensor_frees = []
            for tensor_name in ctx.graph.tensors:
                if tensor_name in ctx.graph.constants:
                    continue
                var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
                tensor_frees.append(f"    free({var_name}.data);")
            frees_code = "\n".join(tensor_frees)

        # Get input and output tensor variable names
        input_var = None
        if ctx.graph.inputs:
            input_name = ctx.graph.inputs[0]
            input_var = ctx.tensor_symbols.get(input_name, input_name)

        output_var = None
        if ctx.graph.outputs:
            output_name = ctx.graph.outputs[0]
            output_var = ctx.tensor_symbols.get(output_name, output_name)

        # Generate print code only if we have input/output tensors
        print_code = ""

        # Add constants loading code if there are constants
        has_constants = bool(ctx.graph.constants)
        if has_constants:
            constants_load_code = """
    /* Load constants from binary file */
    printf("Loading constants from constants.bin...\\n");
    if (nnc_load_constants("constants.bin") != 0) {
        fprintf(stderr, "Failed to load constants\\n");
        return 1;
    }
    printf("Constants loaded successfully.\\n");"""
        else:
            constants_load_code = ""

        if input_var and output_var:
            print_code = f"""
{constants_load_code}
    /* Print input info */
    printf("Input data (first 10):\\n");
    int64_t input_size = 1;
    for (int i = 0; i < {input_var}.ndim; i++) {{
        input_size *= {input_var}.shape[i];
    }}
    for (int i = 0; i < 10 && i < input_size; i++) {{
        printf("  input[%d] = %f\\n", i, ((float*){input_var}.data)[i]);
    }}

    /* Run inference */
    printf("\\nRunning inference...\\n");
    nnc_run();

    /* Print output results */
    printf("\\nOutput results (first 10):\\n");
    int64_t output_size = 1;
    for (int i = 0; i < {output_var}.ndim; i++) {{
        output_size *= {output_var}.shape[i];
    }}
    for (int i = 0; i < 10 && i < output_size; i++) {{
        printf("  output[%d] = %f\\n", i, ((float*){output_var}.data)[i]);
    }}"""
        else:
            if constants_load_code:
                print_code = f"""
{constants_load_code}

    /* Run inference */
    printf("Running inference...\\n");
    nnc_run();"""
            else:
                print_code = """
    /* Run inference */
    printf("Running inference...\\n");
    nnc_run();"""

        return f"""/* Auto-generated by NNC - DO NOT EDIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nnc_types.h"
#include "model.h"
{debug_macros}
{debug_decl}
int main(void) {{
    printf("NNC Model Runner\\n");
    printf("================\\n");
{debug_setup}
    /* Setup all tensors */
{setup_code}
{print_code}

    /* Free memory */
{frees_code}
{debug_cleanup}
    printf("\\nInference complete.\\n");
    return 0;
}}
"""

    def _assign_symbols(self, ctx: CompileContext) -> None:
        """Assign C symbol names using NameManager."""
        name_manager = NameManager()

        # Assign names for tensors
        for tensor_name, tensor in ctx.graph.tensors.items():
            # Use NameManager to sanitize the name
            c_name = name_manager.get_symbol(tensor_name, prefix="tensor_")
            ctx.tensor_symbols[tensor_name] = c_name

        # Assign names for nodes
        for node_name in ctx.graph.nodes:
            c_name = name_manager.get_symbol(node_name, prefix="node_")
            ctx.node_symbols[node_name] = c_name

    def _map_dtype(self, dtype: DataType) -> str:
        """Map IR dtype to C dtype."""
        mapping = {
            DataType.FLOAT32: "float",
            DataType.FLOAT16: "uint16_t",
            DataType.INT32: "int32_t",
            DataType.INT8: "int8_t",
            DataType.UINT8: "uint8_t",
            DataType.BOOL: "uint8_t",
        }
        return mapping.get(dtype, "float")

    def _map_numpy_dtype(self, np_dtype: Any) -> str:
        """Map numpy dtype to C dtype."""
        import numpy as np

        if np_dtype == np.float32:
            return "float"
        elif np_dtype == np.float16:
            return "uint16_t"
        elif np_dtype == np.int32:
            return "int32_t"
        elif np_dtype == np.int8:
            return "int8_t"
        elif np_dtype == np.uint8:
            return "uint8_t"
        else:
            return "float"

    def _generate_debug_dump_code(self, ctx: CompileContext, tensor_name: str, node_idx: int, node_name: str) -> str:
        """Generate C code to dump tensor values for debug comparison.

        Args:
            ctx: Compilation context.
            tensor_name: Name of the tensor to dump.
            node_idx: Index of the node that produced this tensor.
            node_name: Name of the node that produced this tensor.

        Returns:
            C code string that dumps tensor values.
        """
        tensor = ctx.graph.get_tensor(tensor_name)
        if tensor is None:
            return ""

        var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)

        # Calculate element size from dtype
        elem_size_map = {
            DataType.FLOAT32: 4,
            DataType.FLOAT16: 2,
            DataType.INT32: 4,
            DataType.INT64: 8,
            DataType.INT8: 1,
            DataType.UINT8: 1,
            DataType.BOOL: 1,
        }
        elem_size = elem_size_map.get(tensor.dtype, 4)

        # Calculate number of elements from byte size
        byte_size = tensor.byte_size()
        if byte_size < 0:
            # Unknown size due to symbolic dimensions
            num_elements = -1
        else:
            num_elements = byte_size // elem_size

        # Get number of dimensions
        ndim = len(tensor.shape.dims)

        # Determine how to read the tensor data based on dtype
        # BOOL tensors are stored as uint8_t, need conversion to float for output
        # INT64 tensors (e.g., from Shape operator) are stored as int64_t
        is_bool = tensor.dtype == DataType.BOOL
        is_int64 = tensor.dtype == DataType.INT64
        if is_bool:
            # For BOOL, read as uint8_t and convert to float
            data_read_expr = f"((uint8_t*){var_name}.data)[i]"
        elif is_int64:
            # For INT64, read as int64_t and convert to float
            data_read_expr = f"(float)((int64_t*){var_name}.data)[i]"
        else:
            # For other types, read as float
            data_read_expr = f"((float*){var_name}.data)[i]"

        if self.debug_mode:
            # Use debug macros for file output
            code = f"""
    /* Debug dump: {tensor_name} after node {node_idx} ({node_name}) */
    DEBUG_PRINT_TENSOR_START("{tensor_name}", {node_idx});
    DEBUG_PRINT_SHAPE({ndim});
"""
            for i, dim in enumerate(tensor.shape.dims):
                if isinstance(dim, int):
                    code += f'    DEBUG_PRINT_DIM({i}, {dim});\n'
                else:
                    # For symbolic dimensions, use the actual index value (not Python variable i)
                    code += f'    DEBUG_PRINT_DIM({i}, (int){var_name}.shape[{i}]);\n'

            code += f"""
    DEBUG_PRINT_DATA_START();
    for (int i = 0; i < {num_elements}; i++) {{
        DEBUG_PRINT_VALUE((float){data_read_expr});
    }}
    DEBUG_PRINT_DATA_END();
    DEBUG_PRINT_TENSOR_END("{tensor_name}");
"""
        else:
            # Standard printf output (for debugging compiler itself)
            code = f"""
    /* Debug dump: {tensor_name} after node {node_idx} ({node_name}) */
    printf("DEBUG_TENSOR_START %s %d\\n", "{tensor_name}", {node_idx});
    printf("SHAPE %d\\n", {ndim});
"""
            for i, dim in enumerate(tensor.shape.dims):
                if isinstance(dim, int):
                    code += f'    printf("DIM {i} %d\\\\n", {dim});\n'
                else:
                    # For symbolic dimensions, use the actual index value (not Python variable i)
                    code += f'    printf("DIM {i} %d\\\\n", (int){var_name}.shape[{i}]);\n'

            code += f"""
    printf("DATA_START\\\\n");
    for (int i = 0; i < {num_elements}; i++) {{
        printf("%f\\\\n", (float){data_read_expr});
    }}
    printf("DATA_END\\\\n");
    printf("DEBUG_TENSOR_END %s\\\\n\\n", "{tensor_name}");
"""
        return code

    def _inject_debug_into_nnc_run(self, source_code: str, ctx: CompileContext) -> str:
        """Inject debug dump code into nnc_run function.

        Args:
            source_code: Generated C source code.
            ctx: Compilation context.

        Returns:
            Modified source code with debug dumps.
        """
        if not self.debug_mode:
            return source_code

        lines = source_code.split("\n")
        output = []
        in_nnc_run = False
        brace_count = 0
        node_idx = 0
        nodes = ctx.graph.topological_sort()

        for i, line in enumerate(lines):
            output.append(line)

            # Detect nnc_run function start
            if "void nnc_run(void)" in line:
                in_nnc_run = True
                brace_count = 0
                continue

            if in_nnc_run:
                # Count braces to track function scope
                brace_count += line.count("{") - line.count("}")

                # After each function call, add debug dump for outputs
                for node in nodes:
                    if node.op_type == OpType.CONSTANT:
                        continue
                    func_name = ctx.node_symbols.get(node.name, node.name)
                    if f"{func_name}();" in line:
                        # Add debug dump for each output tensor of this node
                        for out_name in node.outputs:
                            tensor = ctx.graph.get_tensor(out_name)
                            if tensor is not None and out_name not in ctx.graph.constants:
                                debug_code = self._generate_debug_dump_code(
                                    ctx, out_name, node_idx, node.name
                                )
                                # Add indentation and lines
                                for debug_line in debug_code.strip().split("\n"):
                                    output.append(debug_line)
                        node_idx += 1
                        break

                # End of function
                if brace_count == 0 and "}" in line:
                    in_nnc_run = False

        return "\n".join(output)
