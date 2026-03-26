"""Phase-1 tiled lowering for convolution and pooling nodes."""

from __future__ import annotations

from math import sqrt

from nnc_py.ir.context import CompileContext
from nnc_py.ir.execution_plan import (
    LayoutClass,
    MemoryRegionKind,
    NodeExecutionPlan,
    TensorAccessPlan,
    TileRegion,
)
from nnc_py.ir.node import OpType
from nnc_py.ir.tensor import TensorType
from nnc_py.ir.types import DataType
from nnc_py.ir.types import GenericBlockedLayoutKind
from nnc_py.passes.base import PassBase
from nnc_py.passes.layout_planning import LAYOUT_PLANS_METADATA_KEY, LayoutPlan
from nnc_py.passes.schedule_analysis import FAST_MEMORY_BUDGET_BYTES, ScheduleCandidate

_SUPPORTED_OP_FAMILIES = {
    "conv2d",
    "maxpool",
    "average_pool",
    "global_average_pool",
    "gemm",
}
_REGION_SIZE_HINTS_METADATA_KEY = "node_execution_plan_region_sizes"
_DEFAULT_CHANNEL_BLOCK = 16
_MIN_TILE_EXTENTS = (1, 1)
_DTYPE_SIZES = {
    DataType.FLOAT32: 4,
    DataType.FLOAT16: 2,
    DataType.INT32: 4,
    DataType.INT64: 8,
    DataType.INT8: 1,
    DataType.UINT8: 1,
    DataType.BOOL: 1,
}


class TiledLoweringPass(PassBase):
    """Lower phase-1 scheduled nodes into execution-plan IR."""

    @property
    def name(self) -> str:
        return "TiledLowering"

    def _execute(self, ctx: CompileContext) -> None:
        execution_plans, region_sizes = lower_phase1_nodes(ctx)
        ctx.metadata["node_execution_plans"] = execution_plans
        ctx.metadata[_REGION_SIZE_HINTS_METADATA_KEY] = region_sizes


def lower_phase1_nodes(
    ctx: CompileContext,
) -> tuple[dict[str, NodeExecutionPlan], dict[str, dict[str, dict[str, int]]]]:
    """Build execution plans from schedule and generic layout metadata."""

    schedule_candidates: dict[str, ScheduleCandidate] = ctx.metadata.get(
        "schedule_candidates", {}
    )
    layout_plans: dict[str, LayoutPlan] = ctx.metadata.get(LAYOUT_PLANS_METADATA_KEY, {})
    execution_plans: dict[str, NodeExecutionPlan] = {}
    region_sizes: dict[str, dict[str, dict[str, int]]] = {}

    for node in ctx.graph.topological_sort():
        candidate = schedule_candidates.get(node.name)
        if candidate is None or candidate.op_family not in _SUPPORTED_OP_FAMILIES:
            continue

        layout_plan = _require_layout_plan(node.name, candidate, layout_plans)
        input_layout = _layout_class_for(layout_plan.input_layout)
        weight_layout = _layout_class_for(layout_plan.weight_layout)
        if candidate.op_family == "gemm":
            execution_plan = _build_safe_gemm_execution_plan(
                ctx,
                node,
                input_layout=input_layout,
                weight_layout=weight_layout,
            )
            if execution_plan is None:
                continue
            execution_plans[node.name] = execution_plan
            region_sizes[node.name] = _estimate_region_size_hints(
                ctx,
                node.name,
                execution_plan,
            )
            continue

        input_halo = _input_halo_for(node.attrs, candidate.op_family)
        output_tile_extents = _choose_output_tile_extents(ctx, node)
        input_tile_extents = _input_tile_extents_for(node.attrs, output_tile_extents)

        input_accesses = [
            TensorAccessPlan(
                tensor_name=node.inputs[0],
                layout_class=input_layout,
                tile_region=TileRegion(
                    logical_extents=input_tile_extents,
                    halo_extents=input_halo,
                ),
                memory_region=MemoryRegionKind.TILE,
            )
        ]
        if candidate.op_family == "conv2d" and len(node.inputs) >= 2:
            input_accesses.append(
                TensorAccessPlan(
                    tensor_name=node.inputs[1],
                    layout_class=weight_layout,
                    memory_region=MemoryRegionKind.PERSISTENT,
                )
            )

        output_accesses = tuple(
            TensorAccessPlan(
                tensor_name=tensor_name,
                layout_class=input_layout,
                tile_region=TileRegion(logical_extents=output_tile_extents),
                memory_region=MemoryRegionKind.TILE,
            )
            for tensor_name in node.outputs
        )

        memory_regions = _memory_regions_for(
            tuple(input_accesses),
            output_accesses,
            include_scratch=candidate.op_family == "conv2d",
        )

        execution_plans[node.name] = NodeExecutionPlan(
            node_name=node.name,
            op_family=candidate.op_family,
            tile_axes=("h", "w"),
            layout_class=input_layout,
            memory_regions=memory_regions,
            input_accesses=tuple(input_accesses),
            output_accesses=output_accesses,
        )
        region_sizes[node.name] = _estimate_region_size_hints(
            ctx,
            node.name,
            execution_plans[node.name],
        )

    _propagate_tile_execution_groups(ctx, execution_plans, region_sizes)
    return execution_plans, region_sizes


def _propagate_tile_execution_groups(
    ctx: CompileContext,
    execution_plans: dict[str, NodeExecutionPlan],
    region_sizes: dict[str, dict[str, dict[str, int]]],
) -> None:
    for node in ctx.graph.topological_sort():
        if node.name in execution_plans:
            continue
        if node.op_type not in {OpType.RELU, OpType.ADD, OpType.FUSED_ADD_RELU}:
            continue
        if len(node.outputs) != 1:
            continue

        producer_plan, flow_tensor_name = _find_tiled_flow_producer_plan(ctx, execution_plans, node)
        if producer_plan is None or flow_tensor_name is None:
            continue

        producer_output_access = next(
            (
                access
                for access in producer_plan.output_accesses
                if access.tensor_name == flow_tensor_name
            ),
            None,
        )
        if producer_output_access is None:
            continue

        tile_region = producer_output_access.tile_region
        layout_class = producer_output_access.layout_class
        if not tile_region.logical_extents:
            continue

        input_accesses: list[TensorAccessPlan] = []
        for input_name in node.inputs:
            tensor = _get_tensor(ctx, input_name)
            output_tensor = _get_tensor(ctx, node.outputs[0])
            if tensor is None or output_tensor is None:
                input_accesses = []
                break
            if tensor.dtype != output_tensor.dtype or tensor.byte_size() != output_tensor.byte_size():
                input_accesses = []
                break
            input_accesses.append(
                TensorAccessPlan(
                    tensor_name=input_name,
                    layout_class=layout_class,
                    tile_region=tile_region,
                    memory_region=MemoryRegionKind.TILE,
                )
            )
        if not input_accesses:
            continue

        output_accesses = (
            TensorAccessPlan(
                tensor_name=node.outputs[0],
                layout_class=layout_class,
                tile_region=tile_region,
                memory_region=MemoryRegionKind.TILE,
            ),
        )
        execution_plans[node.name] = NodeExecutionPlan(
            node_name=node.name,
            op_family=node.op_type.name.lower(),
            tile_axes=producer_plan.tile_axes,
            layout_class=layout_class,
            memory_regions=(MemoryRegionKind.TILE,),
            input_accesses=tuple(input_accesses),
            output_accesses=output_accesses,
        )
        region_sizes[node.name] = {
            "tensor_bytes": {
                access.tensor_name: _estimate_tile_tensor_bytes(
                    ctx,
                    access.tensor_name,
                    tile_region.logical_extents,
                )
                or 0
                for access in (*input_accesses, *output_accesses)
            },
            "region_bytes": {},
        }


def _find_tiled_flow_producer_plan(
    ctx: CompileContext,
    execution_plans: dict[str, NodeExecutionPlan],
    node,
) -> tuple[NodeExecutionPlan | None, str | None]:
    for input_name in node.inputs:
        producers = ctx.graph.get_producers(input_name)
        if len(producers) != 1:
            continue
        producer_plan = execution_plans.get(producers[0].name)
        if producer_plan is None:
            continue
        if any(access.tensor_name == input_name for access in producer_plan.output_accesses):
            return producer_plan, input_name
    return None, None


def _layout_class_for(layout: object | None) -> LayoutClass:
    if layout is None:
        return LayoutClass.PLAIN

    kind = getattr(layout, "kind", None)
    if kind is GenericBlockedLayoutKind.BLOCKED_ACTIVATION:
        return LayoutClass.BLOCKED_ACTIVATION
    if kind is GenericBlockedLayoutKind.BLOCKED_WEIGHT:
        return LayoutClass.BLOCKED_WEIGHT
    return LayoutClass.PLAIN


def _require_layout_plan(
    node_name: str,
    candidate: ScheduleCandidate,
    layout_plans: dict[str, LayoutPlan],
) -> LayoutPlan:
    layout_plan = layout_plans.get(node_name)
    if layout_plan is None:
        raise ValueError(f"Missing layout plan for scheduled node '{node_name}'")
    if getattr(layout_plan, "input_layout", None) is None:
        raise ValueError(f"Missing input layout plan for scheduled node '{node_name}'")
    if candidate.op_family == "conv2d" and getattr(layout_plan, "weight_layout", None) is None:
        raise ValueError(f"Missing weight layout plan for scheduled node '{node_name}'")
    return layout_plan


def _input_halo_for(attrs: dict[str, object], op_family: str) -> tuple[int, int]:
    if op_family == "global_average_pool":
        return (0, 0)

    kernel_h, kernel_w = _normalize_hw(attrs.get("kernel_shape", [1, 1]), fill=1)
    stride_h, stride_w = _normalize_hw(attrs.get("strides", [1, 1]), fill=1)
    return (
        _halo_extent_for(kernel_h, stride_h),
        _halo_extent_for(kernel_w, stride_w),
    )


def _build_safe_gemm_execution_plan(
    ctx: CompileContext,
    node,
    *,
    input_layout: LayoutClass,
    weight_layout: LayoutClass,
) -> NodeExecutionPlan | None:
    if not _supports_minimal_phase1_gemm(ctx, node):
        return None

    input_accesses = [
        TensorAccessPlan(
            tensor_name=node.inputs[0],
            layout_class=input_layout,
            memory_region=MemoryRegionKind.PERSISTENT,
        ),
        TensorAccessPlan(
            tensor_name=node.inputs[1],
            layout_class=weight_layout,
            memory_region=MemoryRegionKind.PERSISTENT,
        ),
    ]
    if len(node.inputs) >= 3:
        input_accesses.append(
            TensorAccessPlan(
                tensor_name=node.inputs[2],
                layout_class=LayoutClass.PLAIN,
                memory_region=MemoryRegionKind.PERSISTENT,
            )
        )

    output_accesses = tuple(
        TensorAccessPlan(
            tensor_name=tensor_name,
            layout_class=input_layout,
            memory_region=MemoryRegionKind.PERSISTENT,
        )
        for tensor_name in node.outputs
    )

    return NodeExecutionPlan(
        node_name=node.name,
        op_family="gemm",
        tile_axes=("m", "n"),
        layout_class=input_layout,
        memory_regions=_memory_regions_for(
            tuple(input_accesses),
            output_accesses,
            include_scratch=False,
        ),
        input_accesses=tuple(input_accesses),
        output_accesses=output_accesses,
    )


def _supports_minimal_phase1_gemm(ctx: CompileContext, node) -> bool:
    if len(node.outputs) != 1 or len(node.inputs) < 2:
        return False

    lhs = _get_tensor(ctx, node.inputs[0])
    rhs = _get_tensor(ctx, node.inputs[1])
    output = _get_tensor(ctx, node.outputs[0])
    if lhs is None or rhs is None or output is None:
        return False

    if node.inputs[1] not in ctx.graph.constants:
        return False

    trans_a = int(node.attrs.get("transA", 0))
    trans_b = int(node.attrs.get("transB", 0))
    if trans_a != 0 or trans_b not in {0, 1}:
        return False

    lhs_dims = _static_matrix_dims(lhs)
    rhs_dims = _static_matrix_dims(rhs)
    output_dims = _static_matrix_dims(output)
    if lhs_dims is None or rhs_dims is None or output_dims is None:
        return False

    lhs_rows, lhs_k = lhs_dims
    if lhs_rows != 1:
        return False

    if trans_b == 0:
        rhs_k, output_cols = rhs_dims
    else:
        output_cols, rhs_k = rhs_dims
    if lhs_k != rhs_k or output_dims != (1, output_cols):
        return False

    if len(node.inputs) >= 3 and not _supports_minimal_gemm_bias(ctx, node.inputs[2], output_cols):
        return False

    lowering = node.metadata.get("lowering")
    if isinstance(lowering, dict) and lowering.get("weight_pack") == "rhs_runtime":
        return False

    return True


def _supports_minimal_gemm_bias(
    ctx: CompileContext,
    bias_name: str,
    output_cols: int,
) -> bool:
    bias = _get_tensor(ctx, bias_name)
    if bias is None or bias_name not in ctx.graph.constants:
        return False

    dims = bias.shape.dims
    if len(dims) == 1:
        return isinstance(dims[0], int) and int(dims[0]) == output_cols
    if len(dims) == 2:
        return (
            isinstance(dims[0], int)
            and isinstance(dims[1], int)
            and int(dims[0]) == 1
            and int(dims[1]) == output_cols
        )
    return False


def _normalize_hw(value: object, *, fill: int) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if isinstance(value, (list, tuple)) and len(value) == 1:
        dim = int(value[0])
        return dim, dim
    return fill, fill


def _halo_extent_for(kernel: int, stride: int) -> int:
    return max(0, (int(kernel) - int(stride) + 1) // 2)


def _memory_regions_for(
    input_accesses: tuple[TensorAccessPlan, ...],
    output_accesses: tuple[TensorAccessPlan, ...],
    *,
    include_scratch: bool,
) -> tuple[MemoryRegionKind, ...]:
    required_regions = {access.memory_region for access in (*input_accesses, *output_accesses)}
    if include_scratch:
        required_regions.add(MemoryRegionKind.SCRATCH)

    region_order = (
        MemoryRegionKind.PERSISTENT,
        MemoryRegionKind.TILE,
        MemoryRegionKind.SCRATCH,
        MemoryRegionKind.PACK,
        MemoryRegionKind.STAGE,
    )
    return tuple(region for region in region_order if region in required_regions)


def _estimate_region_size_hints(
    ctx: CompileContext,
    node_name: str,
    plan: NodeExecutionPlan,
) -> dict[str, dict[str, int]]:
    tensor_bytes: dict[str, int] = {}
    region_bytes: dict[str, int] = {}
    node = ctx.graph.get_node(node_name)

    for access in (*plan.input_accesses, *plan.output_accesses):
        if access.memory_region not in {
            MemoryRegionKind.TILE,
            MemoryRegionKind.PACK,
            MemoryRegionKind.STAGE,
        }:
            continue
        size_bytes = _estimate_tile_tensor_bytes(
            ctx,
            access.tensor_name,
            access.tile_region.logical_extents,
        )
        if size_bytes is not None:
            tensor_bytes[access.tensor_name] = size_bytes

    if MemoryRegionKind.SCRATCH in plan.memory_regions:
        region_bytes[MemoryRegionKind.SCRATCH.value] = _estimate_scratch_bytes(ctx, node)

    return {
        "tensor_bytes": tensor_bytes,
        "region_bytes": region_bytes,
    }


def _choose_output_tile_extents(ctx: CompileContext, node) -> tuple[int, int]:
    output_tensor = _get_tensor(ctx, node.outputs[0])
    if output_tensor is None:
        return ()

    output_dims = _static_spatial_dims(output_tensor)
    if output_dims is None:
        return ()

    output_h, output_w = output_dims
    if output_h <= 0 or output_w <= 0:
        return ()

    scratch_bytes = _estimate_scratch_bytes(ctx, node)
    available_budget = max(_dtype_size(output_tensor.dtype), FAST_MEMORY_BUDGET_BYTES - scratch_bytes)

    max_tile_h = output_h
    max_tile_w = output_w
    width_guess = max(1, min(output_w, int(sqrt(output_w * output_h)) or 1))
    tile_width_candidates = []
    for width in (output_w, width_guess, min(output_w, 64), min(output_w, 32), _MIN_TILE_EXTENTS[1]):
        if width not in tile_width_candidates:
            tile_width_candidates.append(width)

    for tile_w in tile_width_candidates:
        for tile_h in range(max_tile_h, 0, -1):
            total_bytes = _estimate_node_working_set_bytes(
                ctx,
                node,
                output_tile_extents=(tile_h, tile_w),
                scratch_bytes=scratch_bytes,
            )
            if total_bytes is None:
                return _MIN_TILE_EXTENTS
            if total_bytes <= available_budget:
                return (tile_h, tile_w)

    for tile_w in range(max_tile_w, 0, -1):
        total_bytes = _estimate_node_working_set_bytes(
            ctx,
            node,
            output_tile_extents=(_MIN_TILE_EXTENTS[0], tile_w),
            scratch_bytes=scratch_bytes,
        )
        if total_bytes is None:
            return _MIN_TILE_EXTENTS
        if total_bytes <= available_budget:
            return (_MIN_TILE_EXTENTS[0], tile_w)

    return _MIN_TILE_EXTENTS


def _estimate_node_working_set_bytes(
    ctx: CompileContext,
    node,
    *,
    output_tile_extents: tuple[int, int],
    scratch_bytes: int,
) -> int | None:
    total_bytes = scratch_bytes + _estimate_persistent_tensor_bytes(ctx, node)
    input_tile_extents = _input_tile_extents_for(node.attrs, output_tile_extents)

    activation_input = _get_tensor(ctx, node.inputs[0]) if node.inputs else None
    if activation_input is not None:
        input_tile_bytes = _estimate_tensor_tile_bytes(activation_input, input_tile_extents)
        if input_tile_bytes is None:
            return None
        total_bytes += input_tile_bytes

    for tensor_name in node.outputs:
        tensor = _get_tensor(ctx, tensor_name)
        if tensor is not None:
            output_tile_bytes = _estimate_tensor_tile_bytes(tensor, output_tile_extents)
            if output_tile_bytes is None:
                return None
            total_bytes += output_tile_bytes

    return total_bytes


def _estimate_persistent_tensor_bytes(ctx: CompileContext, node) -> int:
    total_bytes = 0
    for input_name in node.inputs[1:]:
        tensor = _get_tensor(ctx, input_name)
        if tensor is None:
            continue
        tensor_bytes = tensor.byte_size()
        if tensor_bytes > 0:
            total_bytes += tensor_bytes
    return total_bytes


def _input_tile_extents_for(
    attrs: dict[str, object],
    output_tile_extents: tuple[int, int],
) -> tuple[int, int]:
    if len(output_tile_extents) != 2:
        return ()

    output_h, output_w = output_tile_extents
    kernel_h, kernel_w = _normalize_hw(attrs.get("kernel_shape", [1, 1]), fill=1)
    stride_h, stride_w = _normalize_hw(attrs.get("strides", [1, 1]), fill=1)
    return (
        max(1, (output_h - 1) * stride_h + kernel_h),
        max(1, (output_w - 1) * stride_w + kernel_w),
    )


def _estimate_tile_tensor_bytes(
    ctx: CompileContext,
    tensor_name: str,
    logical_extents: tuple[int, ...],
) -> int | None:
    tensor = _get_tensor(ctx, tensor_name)
    if tensor is None:
        return None
    return _estimate_tensor_tile_bytes(tensor, logical_extents)


def _estimate_tensor_tile_bytes(
    tensor: TensorType,
    logical_extents: tuple[int, ...],
) -> int | None:
    if len(logical_extents) != 2:
        return None

    batch_channels = _static_batch_channel_dims(tensor)
    if batch_channels is None:
        return None

    batch, channels = batch_channels
    tile_h, tile_w = logical_extents
    if tile_h <= 0 or tile_w <= 0:
        return None
    return batch * channels * tile_h * tile_w * _dtype_size(tensor.dtype)


def _estimate_scratch_bytes(ctx: CompileContext, node) -> int:
    if node.op_type.value not in {"Conv", "FusedConvRelu", "FusedConvBiasRelu", "FusedConvSigmoid"}:
        return 0

    input_tensor = _get_tensor(ctx, node.inputs[0]) if node.inputs else None
    output_tensor = _get_tensor(ctx, node.outputs[0]) if node.outputs else None
    if input_tensor is None or output_tensor is None:
        return 0

    batch_channels_in = _static_batch_channel_dims(input_tensor)
    batch_channels_out = _static_batch_channel_dims(output_tensor)
    if batch_channels_in is None or batch_channels_out is None:
        return 0

    _, channels_in = batch_channels_in
    _, channels_out = batch_channels_out
    kernel_h, kernel_w = _normalize_hw(node.attrs.get("kernel_shape", [1, 1]), fill=1)
    channel_block = min(channels_out, _DEFAULT_CHANNEL_BLOCK)
    return kernel_h * kernel_w * channels_in * channel_block * _dtype_size(output_tensor.dtype)


def _get_tensor(ctx: CompileContext, tensor_name: str) -> TensorType | None:
    return ctx.graph.tensors.get(tensor_name)


def _static_batch_channel_dims(tensor: TensorType) -> tuple[int, int] | None:
    dims = tensor.shape.dims
    if len(dims) < 4 or not isinstance(dims[0], int) or not isinstance(dims[1], int):
        return None
    return int(dims[0]), int(dims[1])


def _static_spatial_dims(tensor: TensorType) -> tuple[int, int] | None:
    dims = tensor.shape.dims
    if len(dims) < 4 or not isinstance(dims[-2], int) or not isinstance(dims[-1], int):
        return None
    return int(dims[-2]), int(dims[-1])


def _static_matrix_dims(tensor: TensorType) -> tuple[int, int] | None:
    dims = tensor.shape.dims
    if len(dims) != 2 or not isinstance(dims[0], int) or not isinstance(dims[1], int):
        return None
    return int(dims[0]), int(dims[1])


def _dtype_size(dtype: DataType) -> int:
    return _DTYPE_SIZES[dtype]


__all__ = ["TiledLoweringPass", "lower_phase1_nodes"]
