"""Tensor emitter for x86 codegen packages."""

from __future__ import annotations

from typing import Any

from nnc_py.codegen.x86_ir import X86CodegenPackage


# ---------------------------------------------------------------------------
# Shared helpers (also used by the backend or other modules)
# ---------------------------------------------------------------------------

def _get_scheduled_memory_plan(ctx: Any) -> Any | None:
    plan = ctx.metadata.get("scheduled_memory_plan")
    if plan is None:
        return None
    if not hasattr(plan, "fast_allocations") or not hasattr(plan, "transfer_points"):
        return None
    return plan


def _prefer_scheduled_memory_plan(
    ctx: Any,
    scheduled_plan: Any | None = None,
) -> bool:
    if scheduled_plan is None:
        scheduled_plan = _get_scheduled_memory_plan(ctx)
    return (
        scheduled_plan is not None
        and ctx.optimization_level >= 3
        and bool(ctx.metadata.get("pipeline_scheduler_enabled"))
    )


def _should_use_scheduled_home_execution(ctx: Any) -> bool:
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


def _node_execution_plan_uses_tiled_storage(plan: Any) -> bool:
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


def _build_schedule_value_map(ctx: Any) -> dict[str, Any]:
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


def _build_schedule_value_graph_tensor_map(ctx: Any) -> dict[str, str]:
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


def _infer_schedule_value_graph_tensor_name(value_name: str) -> str | None:
    if not value_name:
        return None
    for marker in (".reload", ".spill"):
        if marker in value_name:
            candidate = value_name.split(marker, 1)[0]
            if candidate:
                return candidate
    return None


def _decode_schedule_value_graph_tensor_name(value_name: str) -> str | None:
    if value_name.startswith("sram|node|") and "|tensor|" in value_name:
        encoded_tensor = value_name.split("|tensor|", 1)[1]
        name_parts = encoded_tensor.split(":", 1)
        if len(name_parts) == 2:
            return name_parts[1]
        return encoded_tensor
    if value_name.startswith("sram|"):
        return None
    return value_name


def _resolve_schedule_value_graph_tensor_name(
    ctx: Any,
    value_name: str,
) -> str | None:
    candidates: list[str] = []
    graph_tensor_name = _build_schedule_value_graph_tensor_map(ctx).get(value_name)
    if isinstance(graph_tensor_name, str) and graph_tensor_name:
        candidates.append(graph_tensor_name)

    decoded_name = _decode_schedule_value_graph_tensor_name(value_name)
    if isinstance(decoded_name, str) and decoded_name:
        candidates.append(decoded_name)

    inferred_name = _infer_schedule_value_graph_tensor_name(value_name)
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


# ---------------------------------------------------------------------------
# Exclusive helpers (only used by tensors generation)
# ---------------------------------------------------------------------------

def _map_dtype_to_enum(dtype: Any) -> str:
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


def _generate_logical_region_lines(alloc_plan: Any) -> list[str]:
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


def _best_scheduled_fast_allocations(
    ctx: Any,
    scheduled_plan: Any,
) -> dict[str, tuple[tuple[int, int, str], int, int]]:
    best_fast_allocations: dict[str, tuple[tuple[int, int, str], int, int]] = {}
    schedule_value_map = _build_schedule_value_map(ctx)

    for allocation in getattr(scheduled_plan, "fast_allocations", {}).values():
        value_name = str(allocation.value_name)
        graph_tensor_name = _resolve_schedule_value_graph_tensor_name(
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


def _build_tensor_offsets_from_scheduled_plan(
    ctx: Any,
    scheduled_plan: Any,
) -> dict[str, tuple[str, int]]:
    tensor_offsets: dict[str, tuple[str, int]] = {}
    best_fast_allocations = _best_scheduled_fast_allocations(ctx, scheduled_plan)

    for graph_tensor_name, (_, offset, _) in best_fast_allocations.items():
        tensor_offsets[graph_tensor_name] = ("fast", offset)

    for value_name, allocation in getattr(scheduled_plan, "slow_allocations", {}).items():
        graph_tensor_name = _resolve_schedule_value_graph_tensor_name(
            ctx,
            str(value_name),
        )
        if graph_tensor_name is None or graph_tensor_name in tensor_offsets:
            continue
        tensor_offsets[graph_tensor_name] = ("slow", int(allocation.offset))

    return tensor_offsets


def _build_linear_tensor_fallback(
    ctx: Any,
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


def _scheduled_tile_streaming_internal_tensor_names(ctx: Any) -> set[str]:
    metadata = _scheduled_tile_streaming_metadata(ctx)
    return set(metadata["internal_tensor_names"])


def _scheduled_tile_streaming_metadata(ctx: Any) -> dict[str, Any]:
    schedule_problem = ctx.pipeline_schedule_problem
    schedule_result = ctx.pipeline_schedule_result
    if schedule_problem is None or schedule_result is None or not schedule_result.feasible:
        return {"internal_tensor_names": set(), "streamed_node_names": set()}

    # Use backend for deeply shared methods
    from nnc_py.codegen.x86_backend import X86Backend
    backend = X86Backend()

    runtime = backend._build_pipeline_parallel_runtime_metadata(
        ctx,
        schedule_problem=schedule_problem,
        schedule_result=schedule_result,
        scheduled_plan=_get_scheduled_memory_plan(ctx),
    )
    if runtime is None:
        return {"internal_tensor_names": set(), "streamed_node_names": set()}
    plan = backend._build_scheduled_tile_streaming_plan(
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


def _scheduled_native_direct_fast_tensor_names(
    ctx: Any,
    scheduled_plan: Any | None,
) -> set[str]:
    from nnc_py.ir.node import OpType

    if scheduled_plan is None:
        return set()

    execution_plans = ctx.metadata.get("node_execution_plans")
    if not isinstance(execution_plans, dict):
        execution_plans = {}

    streaming_metadata = _scheduled_tile_streaming_metadata(ctx)
    streamed_node_names = set(streaming_metadata["streamed_node_names"])
    best_fast_allocations = _best_scheduled_fast_allocations(ctx, scheduled_plan)
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
                if _node_execution_plan_uses_tiled_storage(plan):
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


def _scheduled_tile_streaming_required_fast_memory(ctx: Any) -> int:
    schedule_problem = ctx.pipeline_schedule_problem
    schedule_result = ctx.pipeline_schedule_result
    if schedule_problem is None or schedule_result is None or not schedule_result.feasible:
        return 0

    from nnc_py.codegen.x86_backend import X86Backend
    backend = X86Backend()

    runtime = backend._build_pipeline_parallel_runtime_metadata(
        ctx,
        schedule_problem=schedule_problem,
        schedule_result=schedule_result,
        scheduled_plan=_get_scheduled_memory_plan(ctx),
    )
    if runtime is None:
        return 0
    plan = backend._build_scheduled_tile_streaming_plan(
        ctx,
        {"parallel_runtime": runtime},
    )
    return int(plan.get("required_fast_memory", 0) or 0)


def _get_tile_aware_runtime_plan(
    ctx: Any,
    alloc_plan: Any,
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

    from nnc_py.codegen.x86_backend import X86Backend
    backend = X86Backend()

    execution_groups = backend._collect_tile_aware_execution_groups(ctx, execution_plans)
    if not execution_groups:
        return {}

    tensor_bindings: dict[str, dict[str, Any]] = {}
    wrapper_nodes: dict[str, dict[str, Any]] = {}
    logical_region_names = ", ".join(sorted(alloc_plan.logical_regions))

    for execution_group in execution_groups:
        required_fast_bytes = max(
            (
                backend._tile_aware_tensor_size_bytes(ctx, tensor_name)
                for tensor_name in execution_group["fast_tensors"]
            ),
            default=0,
        )
        selected_region = backend._select_tile_aware_region(alloc_plan, required_fast_bytes)
        if selected_region is None:
            return {}

        group_label = " -> ".join(execution_group["node_names"])
        for node_name in execution_group["node_names"]:
            plan = execution_plans.get(node_name)
            wrapper_nodes[node_name] = {
                "comment": backend._build_tile_aware_wrapper_comment(
                    group_label=group_label,
                    logical_region_names=logical_region_names,
                    plan=plan,
                )
            }

        for tensor_name in execution_group["external_inputs"]:
            tensor_bindings.setdefault(
                tensor_name,
                backend._make_tile_aware_external_binding(ctx, tensor_name),
            )
        for tensor_name in execution_group["static_tensors"]:
            tensor_bindings[tensor_name] = backend._make_tile_aware_static_binding(ctx, tensor_name)
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
            backend._make_tile_aware_external_binding(ctx, tensor_name),
        )

    return {
        "wrapper_nodes": wrapper_nodes,
        "tensor_bindings": tensor_bindings,
    }


# ---------------------------------------------------------------------------
# Memory pool generation
# ---------------------------------------------------------------------------

def _generate_memory_pool(ctx: Any) -> list[str]:
    """Generate static memory pool declaration."""
    from nnc_py.passes.memory_planning import get_memory_allocation_plan
    from nnc_py.passes.memory_plan import get_memory_plan
    from nnc_py.passes.spill import get_spill_plan

    # Check for new memory allocation plan first
    alloc_plan = get_memory_allocation_plan(ctx)
    scheduled_plan = _get_scheduled_memory_plan(ctx)
    prefer_scheduled_plan = _prefer_scheduled_memory_plan(ctx, scheduled_plan)
    tile_aware_runtime_plan = _get_tile_aware_runtime_plan(ctx, alloc_plan)

    if prefer_scheduled_plan and scheduled_plan is not None:
        fast_memory_size = max(int(scheduled_plan.total_fast_memory), 1)
        fast_memory_size = max(
            fast_memory_size,
            _scheduled_tile_streaming_required_fast_memory(ctx),
        )
        if _should_use_scheduled_home_execution(ctx):
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
            _, fallback_total_size = _build_linear_tensor_fallback(ctx)
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
            region_lines = _generate_logical_region_lines(alloc_plan)
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


# ---------------------------------------------------------------------------
# Main tensor emitter
# ---------------------------------------------------------------------------

def emit_tensors(package: X86CodegenPackage) -> str:
    """Emit tensor definitions from a lowered package."""
    ctx = package.ctx
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
    scheduled_plan = _get_scheduled_memory_plan(ctx)
    prefer_scheduled_plan = _prefer_scheduled_memory_plan(ctx, scheduled_plan)
    tile_aware_runtime_plan = _get_tile_aware_runtime_plan(ctx, alloc_plan)
    tile_aware_tensor_bindings = tile_aware_runtime_plan.get("tensor_bindings", {})
    scheduled_tile_streaming_internal_tensors = _scheduled_tile_streaming_internal_tensor_names(
        ctx
    )
    scheduled_direct_fast_tensors = _scheduled_native_direct_fast_tensor_names(
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
        lines.extend(_generate_memory_pool(ctx))
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
        tensor_offsets = _build_tensor_offsets_from_scheduled_plan(
            ctx,
            scheduled_plan,
        )
    elif fallback_to_linear_storage:
        tensor_offsets, _ = _build_linear_tensor_fallback(ctx)
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
            and _should_use_scheduled_home_execution(ctx)
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
        dtype_enum = _map_dtype_to_enum(tensor.dtype)

        lines.append(f"Tensor {var_name} = {{")
        lines.append(f"    .data = {data_init},")
        lines.append(f"    .dtype = {dtype_enum},")
        lines.append(f"    .shape = {var_name}_shape,")
        lines.append(f"    .ndim = {len(shape_list)},")
        lines.append(f"    .nbytes = {tensor.byte_size()},")
        lines.append("};")
        lines.append("")

    return "\n".join(lines)
