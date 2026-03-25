"""Lower execution plans into a baseline pipeline scheduling problem."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import shlex

from nnc_py.cost_model import CliCostModelProvider, CostModelProvider, SimpleCostModelProvider
from nnc_py.ir.context import CompileContext
from nnc_py.ir.execution_plan import LayoutClass, NodeExecutionPlan, TensorAccessPlan
from nnc_py.ir.node import Node
from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    PipelineScheduleProblem,
    ScheduleDependencyKind,
    ScheduleEdge,
    ScheduleStep,
    ScheduleStepKind,
    SramValue,
    set_pipeline_schedule_problem,
)
from nnc_py.ir.types import DataType
from nnc_py.passes.base import PassBase


_LARGE_OP_FAMILIES = {"conv2d", "gemm", "matmul"}
_SHAPE_OP_FAMILIES = {
    "expand",
    "flatten",
    "reshape",
    "shape",
    "split",
    "squeeze",
    "tile",
    "transpose",
    "unsqueeze",
}
_RESOURCE_ORDER = (
    PipelineResourceKind.DMA,
    PipelineResourceKind.SHAPE,
    PipelineResourceKind.MATMUL,
    PipelineResourceKind.OTHER,
)
_DTYPE_SIZES = {
    DataType.FLOAT32: 4,
    DataType.FLOAT16: 2,
    DataType.INT32: 4,
    DataType.INT64: 8,
    DataType.INT8: 1,
    DataType.UINT8: 1,
    DataType.BOOL: 1,
}


@dataclass(frozen=True)
class _ValueSpec:
    name: str
    size_bytes: int
    shape: tuple[int, ...]
    dtype: str
    graph_tensor_name: str | None = None
    producer_step_id: str | None = None


@dataclass(frozen=True)
class _NodeLowering:
    steps: tuple[ScheduleStep, ...]
    produced_values: tuple[_ValueSpec, ...]


class PipelineStepLoweringPass(PassBase):
    """Build a baseline pipeline schedule problem from node execution plans."""

    @property
    def name(self) -> str:
        return "PipelineStepLowering"

    def _execute(self, ctx: CompileContext) -> None:
        provider = _build_cost_model_provider(ctx)
        problem = lower_execution_plans_to_schedule_problem(ctx, provider=provider)
        set_pipeline_schedule_problem(ctx, problem)


def lower_execution_plans_to_schedule_problem(
    ctx: CompileContext,
    *,
    provider: CostModelProvider | None = None,
) -> PipelineScheduleProblem:
    """Lower node execution plans into schedule steps and conservative dependencies."""

    provider = provider or _build_cost_model_provider(ctx)
    steps: list[ScheduleStep] = []
    edges: list[ScheduleEdge] = []
    edge_keys: set[tuple[str, str, ScheduleDependencyKind]] = set()
    step_order_by_node: dict[str, list[str]] = {}
    producer_last_step_by_tensor: dict[str, str] = {}
    produced_values_by_name: dict[str, _ValueSpec] = {}
    external_values_by_name: dict[str, _ValueSpec] = {}

    for node in ctx.graph.topological_sort():
        plan = ctx.get_node_execution_plan(node.name) or _synthesize_execution_plan(ctx, node)
        for input_access in plan.input_accesses:
            external_spec = _external_value_spec_for_access(ctx, input_access)
            existing_spec = external_values_by_name.get(external_spec.name)
            if existing_spec is None or external_spec.size_bytes > existing_spec.size_bytes:
                external_values_by_name[external_spec.name] = external_spec
        lowering = _build_node_steps(
            ctx,
            node,
            plan,
            provider=provider,
        )
        if not lowering.steps:
            continue

        steps.extend(lowering.steps)
        step_order_by_node[node.name] = [step.id for step in lowering.steps]
        for src_step, dst_step in zip(lowering.steps, lowering.steps[1:]):
            _append_edge(
                edges,
                edge_keys,
                src_step_id=src_step.id,
                dst_step_id=dst_step.id,
                kind=ScheduleDependencyKind.SAME_NODE_SEQUENCE,
            )

        last_step_id = lowering.steps[-1].id
        for value_spec in lowering.produced_values:
            produced_values_by_name[value_spec.name] = value_spec
        for output_access in plan.output_accesses:
            producer_last_step_by_tensor[output_access.tensor_name] = last_step_id

    for node in ctx.graph.topological_sort():
        node_step_ids = step_order_by_node.get(node.name)
        if not node_step_ids:
            continue
        first_step_id = node_step_ids[0]
        for input_name in node.inputs:
            producer_step_id = producer_last_step_by_tensor.get(input_name)
            if producer_step_id is None:
                continue
            _append_edge(
                edges,
                edge_keys,
                src_step_id=producer_step_id,
                dst_step_id=first_step_id,
                kind=ScheduleDependencyKind.DATA,
            )

    return PipelineScheduleProblem(
        steps=tuple(steps),
        edges=tuple(edges),
        sram_values=_build_sram_values(
            steps,
            produced_values_by_name,
            external_values_by_name=external_values_by_name,
        ),
        resources=_RESOURCE_ORDER,
        sram_capacity_bytes=int(ctx.metadata.get("pipeline_sram_capacity_bytes", 0) or 0),
        metadata={
            "origin": "pipeline_step_lowering",
            "node_count": len(step_order_by_node),
            "step_count": len(steps),
        },
    )


def _build_node_steps(
    ctx: CompileContext,
    node: Node,
    plan: NodeExecutionPlan,
    *,
    provider: CostModelProvider,
) -> _NodeLowering:
    if _is_large_tiled_op(plan):
        return _build_large_op_steps(
            ctx,
            node,
            plan,
            provider=provider,
        )
    if _is_shape_family_op(plan):
        return _build_shape_step(
            ctx,
            plan,
            provider=provider,
        )
    return _build_single_compute_step(
        ctx,
        plan,
        provider=provider,
    )


def _build_large_op_steps(
    ctx: CompileContext,
    node: Node,
    plan: NodeExecutionPlan,
    *,
    provider: CostModelProvider,
) -> _NodeLowering:
    node_name = plan.node_name
    dma_in_step_id = f"{node_name}.dma_in"
    dma_input_specs = tuple(
        _external_value_spec_for_access(ctx, access) for access in plan.input_accesses
    )
    staged_inputs = tuple(
        _value_spec_for_access(
            ctx,
            access,
            staged_name=_staged_value_name(node_name, access.tensor_name),
            producer_step_id=dma_in_step_id,
            include_halo=True,
        )
        for access in plan.input_accesses
    )
    steps = [
        _make_step(
            plan,
            provider=provider,
            step_id=dma_in_step_id,
            step_kind=ScheduleStepKind.DMA_IN,
            resource_kind=PipelineResourceKind.DMA,
            input_specs=dma_input_specs,
            output_specs=staged_inputs,
            sram_temp_bytes=0,
            attrs={"phase": "ingress"},
        )
    ]

    compute_input_specs = list(staged_inputs)
    produced_values: list[_ValueSpec] = list(staged_inputs)
    if _needs_shape_prep(plan):
        shape_step_id = f"{node_name}.shape_prep"
        shape_token = _shape_token_spec(plan, producer_step_id=shape_step_id)
        steps.append(
            _make_step(
                plan,
                provider=provider,
                step_id=shape_step_id,
                step_kind=ScheduleStepKind.SHAPE_PREP,
                resource_kind=PipelineResourceKind.SHAPE,
                input_specs=tuple(staged_inputs),
                output_specs=(shape_token,),
                sram_temp_bytes=shape_token.size_bytes,
                attrs={"phase": "shape_prep"},
            )
        )
        compute_input_specs.append(shape_token)
        produced_values.append(shape_token)

    compute_step_id = f"{node_name}.compute"
    compute_outputs = tuple(
        _value_spec_for_access(
            ctx,
            access,
            staged_name=_staged_value_name(node_name, access.tensor_name),
            producer_step_id=compute_step_id,
        )
        for access in plan.output_accesses
    )
    steps.append(
        _make_step(
            plan,
            provider=provider,
            step_id=compute_step_id,
            step_kind=ScheduleStepKind.COMPUTE,
            resource_kind=PipelineResourceKind.MATMUL,
            input_specs=tuple(compute_input_specs),
            output_specs=compute_outputs,
            sram_temp_bytes=max(_scratch_bytes(ctx, plan), 1),
            attrs={"phase": "compute", "macs": _estimate_macs(ctx, node, plan)},
        )
    )
    steps.append(
        _make_step(
            plan,
            provider=provider,
            step_id=f"{node_name}.dma_out",
            step_kind=ScheduleStepKind.DMA_OUT,
            resource_kind=PipelineResourceKind.DMA,
            input_specs=compute_outputs,
            output_specs=(),
            sram_temp_bytes=0,
            attrs={"phase": "egress"},
        )
    )
    produced_values.extend(compute_outputs)
    return _NodeLowering(
        steps=tuple(steps),
        produced_values=tuple(produced_values),
    )


def _build_shape_step(
    ctx: CompileContext,
    plan: NodeExecutionPlan,
    *,
    provider: CostModelProvider,
) -> _NodeLowering:
    step_id = f"{plan.node_name}.shape"
    input_specs = tuple(
        _external_value_spec_for_access(ctx, access) for access in plan.input_accesses
    )
    output_specs = tuple(
        _value_spec_for_access(
            ctx,
            access,
            staged_name=_staged_value_name(plan.node_name, access.tensor_name),
            producer_step_id=step_id,
        )
        for access in plan.output_accesses
    )
    return _NodeLowering(
        steps=(
            _make_step(
                plan,
                provider=provider,
                step_id=step_id,
                step_kind=ScheduleStepKind.SHAPE_PREP,
                resource_kind=PipelineResourceKind.SHAPE,
                input_specs=input_specs,
                output_specs=output_specs,
                sram_temp_bytes=max(_shape_work_bytes(plan), 1),
                attrs={"phase": "shape"},
            ),
        ),
        produced_values=output_specs,
    )


def _build_single_compute_step(
    ctx: CompileContext,
    plan: NodeExecutionPlan,
    *,
    provider: CostModelProvider,
) -> _NodeLowering:
    step_id = f"{plan.node_name}.compute"
    input_specs = tuple(
        _external_value_spec_for_access(ctx, access) for access in plan.input_accesses
    )
    output_specs = tuple(
        _value_spec_for_access(
            ctx,
            access,
            staged_name=_staged_value_name(plan.node_name, access.tensor_name),
            producer_step_id=step_id,
        )
        for access in plan.output_accesses
    )
    return _NodeLowering(
        steps=(
            _make_step(
                plan,
                provider=provider,
                step_id=step_id,
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                input_specs=input_specs,
                output_specs=output_specs,
                sram_temp_bytes=0,
                attrs={"phase": "compute", "work": _estimate_work(ctx, plan)},
            ),
        ),
        produced_values=output_specs,
    )


def _make_step(
    plan: NodeExecutionPlan,
    *,
    provider: CostModelProvider,
    step_id: str,
    step_kind: ScheduleStepKind,
    resource_kind: PipelineResourceKind,
    input_specs: Sequence[_ValueSpec],
    output_specs: Sequence[_ValueSpec],
    sram_temp_bytes: int,
    attrs: dict[str, object],
) -> ScheduleStep:
    estimate = provider.estimate_step(
        op_type=plan.op_family,
        step_kind=step_kind,
        resource_kind=resource_kind,
        input_shapes=tuple(spec.shape for spec in input_specs),
        output_shapes=tuple(spec.shape for spec in output_specs),
        dtypes=tuple(spec.dtype for spec in (*input_specs, *output_specs)),
        tensor_bytes=_step_tensor_bytes(step_kind, input_specs, output_specs),
        attrs=attrs,
    )
    return ScheduleStep(
        id=step_id,
        node_name=plan.node_name,
        step_kind=step_kind,
        resource_kind=resource_kind,
        duration=max(int(estimate.latency), 1),
        launch_overhead=max(int(estimate.launch_overhead), 1),
        sram_input_names=tuple(spec.name for spec in input_specs),
        sram_output_names=tuple(spec.name for spec in output_specs),
        sram_temp_bytes=max(int(sram_temp_bytes), 0),
        attrs={
            "op_family": plan.op_family,
            "cost_model": estimate.source,
            **attrs,
        },
    )


def _build_sram_values(
    steps: Sequence[ScheduleStep],
    produced_values_by_name: dict[str, _ValueSpec],
    *,
    external_values_by_name: dict[str, _ValueSpec],
) -> tuple[SramValue, ...]:
    consumers_by_name: dict[str, list[str]] = {}
    for step in steps:
        for name in step.sram_input_names:
            consumers_by_name.setdefault(name, []).append(step.id)

    ordered_value_specs: list[_ValueSpec] = list(produced_values_by_name.values())
    for value_name in sorted(external_values_by_name):
        if value_name in produced_values_by_name:
            continue
        ordered_value_specs.append(external_values_by_name[value_name])

    return tuple(
        SramValue(
            name=value_spec.name,
            size_bytes=(
                0
                if value_spec.producer_step_id is None
                else max(value_spec.size_bytes, 1)
            ),
            producer_step_id=value_spec.producer_step_id,
            consumer_step_ids=tuple(consumers_by_name.get(value_spec.name, ())),
            must_reside_in_sram=False,
            can_alias=True,
        )
        for value_spec in ordered_value_specs
    )


def _append_edge(
    edges: list[ScheduleEdge],
    edge_keys: set[tuple[str, str, ScheduleDependencyKind]],
    *,
    src_step_id: str,
    dst_step_id: str,
    kind: ScheduleDependencyKind,
) -> None:
    edge_key = (src_step_id, dst_step_id, kind)
    if edge_key in edge_keys:
        return
    edge_keys.add(edge_key)
    edges.append(
        ScheduleEdge(
            src_step_id=src_step_id,
            dst_step_id=dst_step_id,
            kind=kind,
        )
    )


def _build_cost_model_provider(ctx: CompileContext) -> CostModelProvider:
    command = ctx.metadata.get("cost_model_cli_command")
    normalized = _normalize_command(command)
    if normalized:
        return CliCostModelProvider(command=normalized)
    return SimpleCostModelProvider()


def _normalize_command(command: object) -> list[str] | None:
    if command is None:
        return None
    if isinstance(command, str):
        return shlex.split(command) or None
    if isinstance(command, Iterable):
        normalized = [str(part) for part in command]
        return normalized or None
    return None


def _is_large_tiled_op(plan: NodeExecutionPlan) -> bool:
    return plan.op_family.lower() in _LARGE_OP_FAMILIES and (
        bool(plan.tile_axes)
        or any(bool(access.tile_region.logical_extents) for access in plan.input_accesses)
        or any(bool(access.tile_region.logical_extents) for access in plan.output_accesses)
    )


def _is_shape_family_op(plan: NodeExecutionPlan) -> bool:
    return plan.op_family.lower() in _SHAPE_OP_FAMILIES


def _needs_shape_prep(plan: NodeExecutionPlan) -> bool:
    if plan.layout_class is not LayoutClass.PLAIN:
        return True
    if plan.target_physical_layout:
        return True
    return any(
        access.layout_class is not LayoutClass.PLAIN
        or bool(access.tile_region.halo_extents)
        or bool(access.tile_region.block_alignment)
        for access in (*plan.input_accesses, *plan.output_accesses)
    )


def _synthesize_execution_plan(ctx: CompileContext, node: Node) -> NodeExecutionPlan:
    return NodeExecutionPlan(
        node_name=node.name,
        op_family=_normalize_op_family(node),
        input_accesses=tuple(
            TensorAccessPlan(tensor_name=tensor_name)
            for tensor_name in node.inputs
            if tensor_name in ctx.graph.tensors
        ),
        output_accesses=tuple(
            TensorAccessPlan(tensor_name=tensor_name)
            for tensor_name in node.outputs
            if tensor_name in ctx.graph.tensors
        ),
    )


def _normalize_op_family(node: Node) -> str:
    return node.op_type.name.lower()


def _external_value_spec_for_access(
    ctx: CompileContext, access: TensorAccessPlan
) -> _ValueSpec:
    tensor = ctx.graph.get_tensor(access.tensor_name)
    return _ValueSpec(
        name=access.tensor_name,
        size_bytes=_access_size_bytes(ctx, access),
        shape=_access_shape(ctx, access),
        dtype=tensor.dtype.value,
        graph_tensor_name=access.tensor_name,
        producer_step_id=None,
    )


def _value_spec_for_access(
    ctx: CompileContext,
    access: TensorAccessPlan,
    *,
    staged_name: str,
    producer_step_id: str,
    include_halo: bool = False,
) -> _ValueSpec:
    tensor = ctx.graph.get_tensor(access.tensor_name)
    return _ValueSpec(
        name=staged_name,
        size_bytes=_access_size_bytes(ctx, access, include_halo=include_halo),
        shape=_access_shape(ctx, access, include_halo=include_halo),
        dtype=tensor.dtype.value,
        graph_tensor_name=access.tensor_name,
        producer_step_id=producer_step_id,
    )


def _shape_token_spec(plan: NodeExecutionPlan, *, producer_step_id: str) -> _ValueSpec:
    rank_sum = sum(_access_rank(access) for access in plan.input_accesses)
    rank_sum += sum(_access_rank(access) for access in plan.output_accesses)
    return _ValueSpec(
        name=f"sram|node|{_encode_name_part(plan.node_name)}|shape",
        size_bytes=max(_shape_work_bytes(plan), 1),
        shape=(max(rank_sum, 1),),
        dtype=DataType.INT32.value,
        graph_tensor_name=None,
        producer_step_id=producer_step_id,
    )


def _staged_value_name(node_name: str, tensor_name: str) -> str:
    return (
        f"sram|node|{_encode_name_part(node_name)}"
        f"|tensor|{_encode_name_part(tensor_name)}"
    )


def _encode_name_part(value: str) -> str:
    return f"{len(value)}:{value}"


def _access_shape(
    ctx: CompileContext,
    access: TensorAccessPlan,
    *,
    include_halo: bool = False,
) -> tuple[int, ...]:
    if access.tile_region.logical_extents:
        extents = [int(max(dim, 0)) for dim in access.tile_region.logical_extents]
        if include_halo and len(access.tile_region.halo_extents) == len(extents):
            extents = [
                extent + (2 * max(int(halo), 0))
                for extent, halo in zip(extents, access.tile_region.halo_extents)
            ]
        return tuple(extents)
    tensor = ctx.graph.get_tensor(access.tensor_name)
    dims: list[int] = []
    for dim in tensor.shape.dims:
        dims.append(int(dim) if isinstance(dim, int) else 1)
    return tuple(dims)


def _access_size_bytes(
    ctx: CompileContext,
    access: TensorAccessPlan,
    *,
    include_halo: bool = False,
) -> int:
    tensor = ctx.graph.get_tensor(access.tensor_name)
    elem_size = _DTYPE_SIZES[tensor.dtype]
    shape = _access_shape(ctx, access, include_halo=include_halo)
    if access.tile_region.logical_extents:
        elements = _shape_elements(shape)
        return max(elements * elem_size, elem_size)

    tensor_bytes = tensor.byte_size()
    if tensor_bytes > 0:
        return tensor_bytes
    return elem_size
def _scratch_bytes(ctx: CompileContext, plan: NodeExecutionPlan) -> int:
    input_bytes = sum(_access_size_bytes(ctx, access) for access in plan.input_accesses)
    output_bytes = sum(_access_size_bytes(ctx, access) for access in plan.output_accesses)
    return max(output_bytes, input_bytes // 2, 1)


def _shape_work_bytes(plan: NodeExecutionPlan) -> int:
    rank_sum = sum(_access_rank(access) for access in plan.input_accesses)
    rank_sum += sum(_access_rank(access) for access in plan.output_accesses)
    return max(rank_sum * 16, 16)


def _estimate_work(ctx: CompileContext, plan: NodeExecutionPlan) -> int:
    output_elements = sum(
        _shape_elements(_access_shape(ctx, access)) for access in plan.output_accesses
    )
    input_elements = sum(
        _shape_elements(_access_shape(ctx, access)) for access in plan.input_accesses
    )
    return max(output_elements or input_elements, 1)


def _estimate_macs(ctx: CompileContext, node: Node, plan: NodeExecutionPlan) -> int:
    if plan.op_family.lower() in {"gemm", "matmul"} and len(plan.input_accesses) >= 2:
        lhs_shape = _access_shape(ctx, plan.input_accesses[0])
        rhs_shape = _access_shape(ctx, plan.input_accesses[1])
        if len(lhs_shape) >= 2 and len(rhs_shape) >= 2:
            m = lhs_shape[-2]
            k = lhs_shape[-1]
            n = rhs_shape[-1]
            return max(m * k * n, 1)

    output_elements = sum(
        _shape_elements(_access_shape(ctx, access)) for access in plan.output_accesses
    )
    kernel_work = 1
    kernel_shape = node.attrs.get("kernel_shape")
    if isinstance(kernel_shape, Sequence) and not isinstance(kernel_shape, str | bytes):
        kernel_work = _shape_elements(
            tuple(int(dim) if isinstance(dim, int) else 1 for dim in kernel_shape)
        )
    elif len(plan.input_accesses) >= 2:
        kernel_shape = _access_shape(ctx, plan.input_accesses[1])
        kernel_work = _shape_elements(kernel_shape)
    return max(output_elements * kernel_work, 1)


def _step_tensor_bytes(
    step_kind: ScheduleStepKind,
    input_specs: Sequence[_ValueSpec],
    output_specs: Sequence[_ValueSpec],
) -> int:
    if step_kind is ScheduleStepKind.DMA_IN:
        return max(sum(spec.size_bytes for spec in output_specs), 0)
    if step_kind is ScheduleStepKind.DMA_OUT:
        return max(sum(spec.size_bytes for spec in input_specs), 0)
    return max(
        sum(spec.size_bytes for spec in input_specs)
        + sum(spec.size_bytes for spec in output_specs),
        0,
    )


def _access_rank(access: TensorAccessPlan) -> int:
    return len(access.tile_region.logical_extents) or 1


def _shape_elements(shape: Sequence[int]) -> int:
    if not shape:
        return 1
    total = 1
    for dim in shape:
        total *= max(int(dim), 1)
    return max(total, 1)
