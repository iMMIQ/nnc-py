"""Recipe and problem builders for the external joint tiling/schedule problem."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import replace
import math
import sys

from nnc_py.ir.context import CompileContext
from nnc_py.ir.execution_plan import NodeExecutionPlan, TensorAccessPlan
from nnc_py.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_OBJECTIVE,
    JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
    JointAction,
    JointActionKind,
    JointBoundaryConstraint,
    JointCompatibleRecipePair,
    JointCostParameters,
    JointDependencyEdge,
    JointDependencyEdgeKind,
    JointLayoutSpec,
    JointProblem,
    JointRecipe,
    JointRegion,
    JointResource,
    JointResourceKind,
    JointTileSpec,
    JointSramItem,
    JointSramItemKind,
    JointValue,
    JointValueConsumer,
    JointValueFootprint,
    JointValueProducer,
    JointValueTier,
)
from nnc_py.ir.pipeline_schedule import PipelineResourceKind, ScheduleStepKind
from nnc_py.ir.types import MemoryLayout
from nnc_py.passes.pipeline_step_lowering import (
    _DTYPE_SIZES,
    _access_shape,
    _access_size_bytes,
    _build_cost_model_provider,
    _estimate_macs,
    _estimate_work,
    _is_large_tiled_op,
    _is_shape_family_op,
    _shape_elements,
    _scratch_bytes,
)

from .regions import JointProblemBuilderError, build_joint_regions, get_joint_problem_plans

_DEFAULT_ALIGNMENT_BYTES = 16


@dataclass(frozen=True)
class _RegionAssembly:
    region: JointRegion
    plan: NodeExecutionPlan
    recipe: JointRecipe
    actions: tuple[JointAction, ...]
    compute_action: JointAction
    spillable_output_value_ids: tuple[str, ...]


def build_joint_problem(ctx: CompileContext) -> JointProblem:
    """Build the external joint tiling/schedule problem from current metadata."""

    regions = build_joint_regions(ctx)
    plan_by_region_id = _plan_by_region_id(ctx, regions)
    provider = _build_cost_model_provider(ctx)
    producer_region_by_value = _producer_region_by_value(regions)
    plan_variants_by_region_id = {
        region.region_id: _plan_variants_for_region(
            ctx,
            region=region,
            base_plan=plan_by_region_id[region.region_id],
        )
        for region in regions
    }
    value_sizes = _value_sizes(ctx, regions, plan_variants_by_region_id)

    assemblies = tuple(
        assembly
        for region in regions
        for assembly in _build_region_assemblies(
            ctx,
            region=region,
            plans=plan_variants_by_region_id[region.region_id],
            provider=provider,
            producer_region_by_value=producer_region_by_value,
            value_sizes=value_sizes,
        )
    )
    actions = [action for assembly in assemblies for action in assembly.actions]
    actions_by_id = {action.action_id: action for action in actions}

    optional_actions = _build_optional_actions(
        ctx,
        assemblies=assemblies,
        value_sizes=value_sizes,
        actions_by_id=actions_by_id,
        provider=provider,
    )
    actions.extend(optional_actions)
    actions_by_id.update({action.action_id: action for action in optional_actions})

    values = _build_values(
        ctx,
        assemblies=assemblies,
        actions=tuple(actions),
        producer_region_by_value=producer_region_by_value,
        value_sizes=value_sizes,
    )
    boundary_constraints = _build_boundary_constraints(assemblies)
    dependency_edges = _build_dependency_edges(
        assemblies=assemblies,
        values=values,
        optional_actions=optional_actions,
    )
    sram_items = _build_problem_sram_items(ctx, assemblies)
    problem = JointProblem(
        schema_version=JOINT_TILING_SCHEDULE_PROBLEM_SCHEMA_VERSION,
        regions=regions,
        recipes=tuple(assembly.recipe for assembly in assemblies),
        values=values,
        actions=tuple(actions),
        boundary_constraints=boundary_constraints,
        dependency_edges=dependency_edges,
        resources=(
            JointResource(resource_kind=JointResourceKind.DMA, slot_count=1),
            JointResource(resource_kind=JointResourceKind.MATMUL, slot_count=1),
            JointResource(resource_kind=JointResourceKind.SHAPE, slot_count=1),
            JointResource(resource_kind=JointResourceKind.OTHER, slot_count=1),
        ),
        sram_capacity_bytes=_sram_capacity_bytes(ctx),
        sram_items=sram_items,
        default_alignment_bytes=_DEFAULT_ALIGNMENT_BYTES,
        objective=JOINT_TILING_SCHEDULE_OBJECTIVE,
    )
    _validate_problem(problem)
    return problem


def _build_region_assemblies(
    ctx: CompileContext,
    *,
    region: JointRegion,
    plans: tuple[NodeExecutionPlan, ...],
    provider,
    producer_region_by_value: dict[str, str],
    value_sizes: dict[str, int],
) -> tuple[_RegionAssembly, ...]:
    base_plan = plans[0]
    return tuple(
        _build_region_assembly(
            ctx,
            region=region,
            plan=plan,
            base_plan=base_plan,
            recipe_index=recipe_index,
            provider=provider,
            producer_region_by_value=producer_region_by_value,
            value_sizes=value_sizes,
        )
        for recipe_index, plan in enumerate(plans)
    )


def _build_region_assembly(
    ctx: CompileContext,
    *,
    region: JointRegion,
    plan: NodeExecutionPlan,
    base_plan: NodeExecutionPlan,
    recipe_index: int,
    provider,
    producer_region_by_value: dict[str, str],
    value_sizes: dict[str, int],
) -> _RegionAssembly:
    recipe_id = f"{region.region_id}.recipe{recipe_index}"
    recipe_value_sizes = dict(value_sizes)
    recipe_value_sizes.update(_recipe_value_sizes(ctx, plan, base_plan=base_plan))
    external_input_value_ids = tuple(
        value_id
        for value_id in region.input_value_ids
        if value_id not in producer_region_by_value
        and _classify_external_value(ctx, value_id) in (JointValueTier.INPUT, JointValueTier.CONST)
    )
    dma_in_actions = tuple(
        _build_transfer_action(
            ctx,
            provider=provider,
            plan=plan,
            action_id=f"{recipe_id}.dma_in.{value_id}",
            action_kind=JointActionKind.DMA_IN,
            schedule_step_kind=ScheduleStepKind.DMA_IN,
            value_id=value_id,
            size_bytes=recipe_value_sizes[value_id],
            region_id=region.region_id,
            recipe_id=recipe_id,
            is_optional=False,
            optional_value_id=None,
        )
        for value_id in external_input_value_ids
    )
    compute_reads = tuple(access.tensor_name for access in plan.input_accesses)
    compute_writes = tuple(access.tensor_name for access in plan.output_accesses)
    compute_action = _build_compute_action(
        ctx,
        provider=provider,
        region=region,
        plan=plan,
        reads=compute_reads,
        writes=compute_writes,
        action_id=f"{recipe_id}.compute",
        recipe_id=recipe_id,
        value_sizes=recipe_value_sizes,
    )
    dma_out_actions = tuple(
        _build_transfer_action(
            ctx,
            provider=provider,
            plan=plan,
            action_id=f"{recipe_id}.dma_out.{value_id}",
            action_kind=JointActionKind.DMA_OUT,
            schedule_step_kind=ScheduleStepKind.DMA_OUT,
            value_id=value_id,
            size_bytes=recipe_value_sizes[value_id],
            region_id=region.region_id,
            recipe_id=recipe_id,
            is_optional=False,
            optional_value_id=None,
        )
        for value_id in region.output_value_ids
        if value_id in ctx.graph.outputs
    )
    mandatory_actions = dma_in_actions + (compute_action,) + dma_out_actions
    spillable_output_value_ids = tuple(
        value_id
        for value_id in region.output_value_ids
        if value_id not in ctx.graph.outputs and any(
            successor.region_id != region.region_id and value_id in successor.input_value_ids
            for successor in build_joint_regions(ctx)
        )
    )
    recipe = JointRecipe(
        recipe_id=recipe_id,
        region_id=region.region_id,
        tile_spec=_tile_spec_for(ctx, plan),
        layout_spec=_layout_spec_for(plan),
        activates_action_ids=tuple(action.action_id for action in mandatory_actions),
        value_footprint=_recipe_footprint(
            region=region,
            mandatory_actions=mandatory_actions,
            value_sizes=recipe_value_sizes,
        ),
        cost_parameters=_recipe_cost_parameters(mandatory_actions),
    )
    return _RegionAssembly(
        region=region,
        plan=plan,
        recipe=recipe,
        actions=mandatory_actions,
        compute_action=compute_action,
        spillable_output_value_ids=spillable_output_value_ids,
    )


def _plan_by_region_id(
    ctx: CompileContext, regions: tuple[JointRegion, ...]
) -> dict[str, NodeExecutionPlan]:
    available_plans = get_joint_problem_plans(ctx)
    plan_by_region_id: dict[str, NodeExecutionPlan] = {}
    for region in regions:
        plan = available_plans.get(region.region_id)
        if plan is None:
            raise JointProblemBuilderError(
                f"missing node execution plan for region {region.region_id!r}"
            )
        plan_by_region_id[region.region_id] = plan
    return plan_by_region_id


def _producer_region_by_value(
    regions: tuple[JointRegion, ...]
) -> dict[str, str]:
    return {
        value_id: region.region_id
        for region in regions
        for value_id in region.output_value_ids
    }


def _value_sizes(
    ctx: CompileContext,
    regions: tuple[JointRegion, ...],
    plan_variants_by_region_id: dict[str, tuple[NodeExecutionPlan, ...]],
) -> dict[str, int]:
    size_by_value: dict[str, int] = {}
    for region in regions:
        plans = plan_variants_by_region_id[region.region_id]
        base_plan = plans[0]
        for plan in plans:
            for value_id, access_size in _recipe_value_sizes(
                ctx,
                plan,
                base_plan=base_plan,
            ).items():
                size_by_value[value_id] = max(size_by_value.get(value_id, 0), access_size)
    for region in regions:
        for value_id in (*region.input_value_ids, *region.output_value_ids):
            size_by_value.setdefault(value_id, _fallback_tensor_size(ctx, value_id))
    return size_by_value


def _recipe_value_sizes(
    ctx: CompileContext,
    plan: NodeExecutionPlan,
    *,
    base_plan: NodeExecutionPlan,
) -> dict[str, int]:
    size_by_value: dict[str, int] = {}

    for access, base_access in zip(plan.input_accesses, base_plan.input_accesses):
        if access.tensor_name not in ctx.graph.tensors:
            raise JointProblemBuilderError(
                f"node execution plan for {plan.node_name!r} references unknown tensor "
                f"{access.tensor_name!r}"
            )
        size_by_value[access.tensor_name] = max(
            size_by_value.get(access.tensor_name, 0),
            _recipe_access_size_bytes(
                ctx,
                access,
                base_access=base_access,
                node_name=base_plan.node_name,
                include_halo=True,
            ),
        )

    for access, base_access in zip(plan.output_accesses, base_plan.output_accesses):
        if access.tensor_name not in ctx.graph.tensors:
            raise JointProblemBuilderError(
                f"node execution plan for {plan.node_name!r} references unknown tensor "
                f"{access.tensor_name!r}"
            )
        size_by_value[access.tensor_name] = max(
            size_by_value.get(access.tensor_name, 0),
            _recipe_access_size_bytes(
                ctx,
                access,
                base_access=base_access,
                node_name=base_plan.node_name,
                include_halo=False,
            ),
        )

    return size_by_value


def _recipe_access_size_bytes(
    ctx: CompileContext,
    access: TensorAccessPlan,
    *,
    base_access: TensorAccessPlan,
    node_name: str,
    include_halo: bool,
) -> int:
    if access.tensor_name != base_access.tensor_name:
        raise JointProblemBuilderError(
            f"plan variant for {node_name!r} changed access tensor ordering "
            f"from {base_access.tensor_name!r} to {access.tensor_name!r}"
        )
    tensor = ctx.graph.get_tensor(access.tensor_name)
    elem_size = _DTYPE_SIZES[tensor.dtype]
    base_bytes = _access_size_bytes(
        ctx,
        base_access,
        node_name=node_name,
        include_halo=include_halo,
    )
    if access.tile_region.logical_extents and base_access.tile_region.logical_extents:
        current_elements = _shape_elements(
            _access_shape(ctx, access, include_halo=include_halo)
        )
        base_elements = _shape_elements(
            _access_shape(ctx, base_access, include_halo=include_halo)
        )
        if current_elements > 0 and base_elements > 0:
            scaled_bytes = math.ceil(base_bytes * current_elements / base_elements)
            return max(scaled_bytes, elem_size)
    return max(base_bytes, elem_size)


def _fallback_tensor_size(ctx: CompileContext, value_id: str) -> int:
    tensor = ctx.graph.get_tensor(value_id)
    tensor_bytes = tensor.byte_size()
    return max(tensor_bytes, 1)


def _build_compute_action(
    ctx: CompileContext,
    *,
    provider,
    region: JointRegion,
    plan: NodeExecutionPlan,
    reads: tuple[str, ...],
    writes: tuple[str, ...],
    action_id: str,
    recipe_id: str,
    value_sizes: dict[str, int],
) -> JointAction:
    pipeline_resource_kind = _pipeline_resource_kind_for(plan)
    if pipeline_resource_kind is PipelineResourceKind.MATMUL:
        attrs = {
            "phase": "compute",
            "macs": _estimate_macs(ctx, ctx.graph.get_node(plan.node_name), plan),
        }
    else:
        attrs = {
            "phase": "compute",
            "work": _estimate_work(ctx, plan),
        }
    estimate = provider.estimate_step(
        op_type=plan.op_family,
        step_kind=ScheduleStepKind.COMPUTE,
        resource_kind=pipeline_resource_kind,
        input_shapes=tuple(
            _access_shape(ctx, access, node_name=plan.node_name)
            for access in plan.input_accesses
        ),
        output_shapes=tuple(
            _access_shape(ctx, access, node_name=plan.node_name)
            for access in plan.output_accesses
        ),
        dtypes=tuple(
            ctx.graph.get_tensor(access.tensor_name).dtype.value
            for access in (*plan.input_accesses, *plan.output_accesses)
        ),
        tensor_bytes=sum(value_sizes.get(value_id, 0) for value_id in (*reads, *writes)),
        attrs=attrs,
    )
    return JointAction(
        action_id=action_id,
        kind=JointActionKind.COMPUTE,
        resource_kind=_joint_resource_kind_for(pipeline_resource_kind),
        duration=max(int(estimate.latency), 1),
        launch_overhead=max(int(estimate.launch_overhead), 1),
        reads=reads,
        writes=writes,
        temp_bytes=_compute_temp_bytes(ctx, plan),
        is_optional=False,
        region_id=region.region_id,
        recipe_id=recipe_id,
        optional_value_id=None,
    )


def _build_transfer_action(
    ctx: CompileContext,
    *,
    provider,
    plan: NodeExecutionPlan,
    action_id: str,
    action_kind: JointActionKind,
    schedule_step_kind: ScheduleStepKind,
    value_id: str,
    size_bytes: int,
    region_id: str | None,
    recipe_id: str | None,
    is_optional: bool,
    optional_value_id: str | None,
) -> JointAction:
    tensor = ctx.graph.get_tensor(value_id)
    shape = tuple(int(dim) if isinstance(dim, int) else 1 for dim in tensor.shape.dims)
    estimate = provider.estimate_step(
        op_type=plan.op_family,
        step_kind=schedule_step_kind,
        resource_kind=PipelineResourceKind.DMA,
        input_shapes=(shape,),
        output_shapes=(shape,),
        dtypes=(tensor.dtype.value, tensor.dtype.value),
        tensor_bytes=size_bytes,
        attrs={"phase": action_kind.value},
    )
    return JointAction(
        action_id=action_id,
        kind=action_kind,
        resource_kind=JointResourceKind.DMA,
        duration=max(int(estimate.latency), 1),
        launch_overhead=max(int(estimate.launch_overhead), 1),
        reads=(value_id,),
        writes=(value_id,),
        temp_bytes=0,
        is_optional=is_optional,
        region_id=region_id,
        recipe_id=recipe_id,
        optional_value_id=optional_value_id,
    )


def _build_optional_actions(
    ctx: CompileContext,
    *,
    assemblies: tuple[_RegionAssembly, ...],
    value_sizes: dict[str, int],
    actions_by_id: dict[str, JointAction],
    provider,
) -> tuple[JointAction, ...]:
    producer_action_by_value = {
        value_id: assembly.compute_action.action_id
        for assembly in assemblies
        for value_id in assembly.spillable_output_value_ids
    }
    optional_actions: list[JointAction] = []
    seen_value_ids: set[str] = set()
    for assembly in assemblies:
        for value_id in assembly.spillable_output_value_ids:
            if value_id not in producer_action_by_value or value_id in seen_value_ids:
                continue
            seen_value_ids.add(value_id)
            optional_actions.append(
                _build_transfer_action(
                    ctx,
                    provider=provider,
                    plan=assembly.plan,
                    action_id=f"{value_id}.spill",
                    action_kind=JointActionKind.SPILL,
                    schedule_step_kind=ScheduleStepKind.SPILL_DMA,
                    value_id=value_id,
                    size_bytes=value_sizes[value_id],
                    region_id=None,
                    recipe_id=None,
                    is_optional=True,
                    optional_value_id=value_id,
                )
            )
            optional_actions.append(
                _build_transfer_action(
                    ctx,
                    provider=provider,
                    plan=assembly.plan,
                    action_id=f"{value_id}.reload",
                    action_kind=JointActionKind.RELOAD,
                    schedule_step_kind=ScheduleStepKind.RELOAD_DMA,
                    value_id=value_id,
                    size_bytes=value_sizes[value_id],
                    region_id=None,
                    recipe_id=None,
                    is_optional=True,
                    optional_value_id=value_id,
                )
            )
    return tuple(optional_actions)


def _build_values(
    ctx: CompileContext,
    *,
    assemblies: tuple[_RegionAssembly, ...],
    actions: tuple[JointAction, ...],
    producer_region_by_value: dict[str, str],
    value_sizes: dict[str, int],
) -> tuple[JointValue, ...]:
    action_ids_reading_value: dict[str, list[str]] = defaultdict(list)
    action_ids_writing_value: dict[str, list[str]] = defaultdict(list)
    for action in actions:
        for value_id in action.reads:
            action_ids_reading_value[value_id].append(action.action_id)
        for value_id in action.writes:
            action_ids_writing_value[value_id].append(action.action_id)

    ordered_value_ids: list[str] = []
    for assembly in assemblies:
        for value_id in (*assembly.region.input_value_ids, *assembly.region.output_value_ids):
            if value_id not in ordered_value_ids:
                ordered_value_ids.append(value_id)

    values: list[JointValue] = []
    for value_id in ordered_value_ids:
        producing_region_id = producer_region_by_value.get(value_id)
        initial_tier = (
            JointValueTier.UNMATERIALIZED
            if producing_region_id is not None
            else _classify_external_value(ctx, value_id)
        )
        required_final_tier = _required_final_tier(
            ctx,
            value_id=value_id,
            producing_region_id=producing_region_id,
        )
        producer_action_id = None
        for assembly in assemblies:
            if value_id in assembly.compute_action.writes:
                producer_action_id = assembly.compute_action.action_id
                break
        spillable = bool(
            producing_region_id is not None
            and value_id not in ctx.graph.outputs
            and any(
                value_id in other.region.input_value_ids
                and other.region.region_id != producing_region_id
                for other in assemblies
            )
        )
        values.append(
            JointValue(
                value_id=value_id,
                size_bytes=max(value_sizes.get(value_id, 0), 1),
                initial_tier=initial_tier,
                required_final_tier=required_final_tier,
                must_keep=False,
                spillable=spillable,
                allows_multiple_sram_windows=spillable,
                producer=(
                    None
                    if producer_action_id is None
                    else JointValueProducer(action_id=producer_action_id)
                ),
                consumers=tuple(
                    JointValueConsumer(action_id=action_id)
                    for action_id in action_ids_reading_value.get(value_id, ())
                ),
            )
        )
    return tuple(values)


def _build_boundary_constraints(
    assemblies: tuple[_RegionAssembly, ...]
) -> tuple[JointBoundaryConstraint, ...]:
    recipes_by_region: dict[str, tuple[JointRecipe, ...]] = {
        region_id: tuple(
            assembly.recipe
            for assembly in assemblies
            if assembly.region.region_id == region_id
        )
        for region_id in {assembly.region.region_id for assembly in assemblies}
    }
    primary_assemblies = {
        region_id: next(
            assembly
            for assembly in assemblies
            if assembly.region.region_id == region_id
        )
        for region_id in recipes_by_region
    }
    boundaries: list[JointBoundaryConstraint] = []
    for src in primary_assemblies.values():
        src_outputs = set(src.region.output_value_ids)
        for dst in primary_assemblies.values():
            if src.region.region_id == dst.region.region_id:
                continue
            if not src_outputs.intersection(dst.region.input_value_ids):
                continue
            boundaries.append(
                JointBoundaryConstraint(
                    boundary_id=f"{src.region.region_id}->{dst.region.region_id}",
                    src_region_id=src.region.region_id,
                    dst_region_id=dst.region.region_id,
                    compatible_recipe_pairs=tuple(
                        JointCompatibleRecipePair(
                            src_recipe_id=src_recipe.recipe_id,
                            dst_recipe_id=dst_recipe.recipe_id,
                        )
                        for src_recipe in recipes_by_region[src.region.region_id]
                        for dst_recipe in recipes_by_region[dst.region.region_id]
                    ),
                )
            )
    return tuple(boundaries)


def _build_problem_sram_items(
    ctx: CompileContext,
    assemblies: tuple[_RegionAssembly, ...],
) -> tuple[JointSramItem, ...]:
    items: list[JointSramItem] = []
    for assembly in assemblies:
        compute_action = assembly.compute_action
        if compute_action.temp_bytes > 0:
            items.append(
                JointSramItem(
                    item_id=f"{compute_action.action_id}.temp",
                    kind=JointSramItemKind.TEMP_INTERVAL,
                    size_bytes=compute_action.temp_bytes,
                    alignment_bytes=_DEFAULT_ALIGNMENT_BYTES,
                    is_optional=False,
                    owner_action_id=compute_action.action_id,
                    owner_value_id=None,
                    owner_residency_id=None,
                )
            )
    return tuple(items)


def _build_dependency_edges(
    *,
    assemblies: tuple[_RegionAssembly, ...],
    values: tuple[JointValue, ...],
    optional_actions: tuple[JointAction, ...],
) -> tuple[JointDependencyEdge, ...]:
    edges: list[JointDependencyEdge] = []
    edge_keys: set[tuple[str, str, JointDependencyEdgeKind]] = set()

    def append_edge(src_action_id: str, dst_action_id: str, kind: JointDependencyEdgeKind) -> None:
        edge_key = (src_action_id, dst_action_id, kind)
        if src_action_id == dst_action_id or edge_key in edge_keys:
            return
        edge_keys.add(edge_key)
        edges.append(
            JointDependencyEdge(
                src_action_id=src_action_id,
                dst_action_id=dst_action_id,
                kind=kind,
            )
        )

    action_by_id = {
        action.action_id: action
        for assembly in assemblies
        for action in assembly.actions
    }
    action_by_id.update({action.action_id: action for action in optional_actions})

    for assembly in assemblies:
        for action in assembly.actions:
            if action.kind is JointActionKind.DMA_IN:
                append_edge(action.action_id, assembly.compute_action.action_id, JointDependencyEdgeKind.DATA)
            if action.kind is JointActionKind.DMA_OUT:
                append_edge(assembly.compute_action.action_id, action.action_id, JointDependencyEdgeKind.DATA)

    for value in values:
        writer_action_ids = sorted(
            action.action_id
            for action in action_by_id.values()
            if value.value_id in action.writes
            and action.kind in (
                JointActionKind.COMPUTE,
                JointActionKind.DMA_IN,
                JointActionKind.RELOAD,
            )
        )
        for writer_action_id in writer_action_ids:
            for consumer in value.consumers:
                append_edge(
                    writer_action_id,
                    consumer.action_id,
                    JointDependencyEdgeKind.DATA,
                )

    consumers_by_value: dict[str, list[str]] = defaultdict(list)
    for value in values:
        consumers_by_value[value.value_id].extend(
            consumer.action_id for consumer in value.consumers
        )
    for action in optional_actions:
        value_id = action.optional_value_id
        if value_id is None:
            continue
        producer_action_ids = sorted(
            action_id
            for action_id, candidate_action in action_by_id.items()
            if value_id in candidate_action.writes
            and candidate_action.kind in (
                JointActionKind.COMPUTE,
                JointActionKind.DMA_IN,
                JointActionKind.RELOAD,
            )
        )
        if not producer_action_ids:
            continue
        if action.kind is JointActionKind.SPILL:
            for producer_action_id in producer_action_ids:
                append_edge(producer_action_id, action.action_id, JointDependencyEdgeKind.ORDER)
            reload_action_id = f"{value_id}.reload"
            if reload_action_id in action_by_id:
                append_edge(action.action_id, reload_action_id, JointDependencyEdgeKind.ORDER)
        if action.kind is JointActionKind.RELOAD:
            for consumer_action_id in consumers_by_value.get(value_id, ()):
                append_edge(action.action_id, consumer_action_id, JointDependencyEdgeKind.ORDER)

    return tuple(edges)


def _tile_spec_for(ctx: CompileContext, plan: NodeExecutionPlan) -> JointTileSpec:
    axes = tuple(plan.tile_axes)
    shape = ()
    if axes:
        accesses = tuple(plan.output_accesses) + tuple(plan.input_accesses)
        for access in accesses:
            extents = tuple(int(max(dim, 0)) for dim in access.tile_region.logical_extents)
            if not extents:
                continue
            shape = _tile_shape_for_axes(
                tensor=ctx.graph.get_tensor(access.tensor_name),
                axes=axes,
                extents=extents,
            )
            break
    return JointTileSpec(axes=axes, shape=shape)


def _tile_shape_for_axes(
    *,
    tensor,
    axes: tuple[str, ...],
    extents: tuple[int, ...],
) -> tuple[int, ...]:
    if len(extents) == len(axes):
        return extents
    axis_indices = tuple(
        _axis_index_for_tensor(axis_name, tensor.shape.layout, len(extents))
        for axis_name in axes
    )
    return tuple(extents[index] for index in axis_indices)


def _axis_index_for_tensor(
    axis_name: str,
    layout: MemoryLayout,
    rank: int,
) -> int:
    lowered = axis_name.lower()
    if layout is MemoryLayout.NCHW:
        mapping = {"n": 0, "c": 1, "h": 2, "w": 3}
    elif layout is MemoryLayout.NHWC:
        mapping = {"n": 0, "h": 1, "w": 2, "c": 3}
    elif layout is MemoryLayout.OIHW:
        mapping = {"o": 0, "i": 1, "h": 2, "w": 3}
    else:
        mapping = {}
    if lowered in mapping and mapping[lowered] < rank:
        return mapping[lowered]
    if lowered == "m" and rank >= 2:
        return rank - 2
    if lowered == "n" and rank >= 1:
        return rank - 1
    if lowered == "k" and rank >= 1:
        return rank - 1
    raise JointProblemBuilderError(
        f"cannot map tile axis {axis_name!r} onto rank-{rank} tensor"
    )


def _layout_spec_for(plan: NodeExecutionPlan) -> JointLayoutSpec:
    layout_tags = [plan.layout_class.value]
    if plan.target_physical_layout:
        layout_tags.append(plan.target_physical_layout)
    for access in (*plan.input_accesses, *plan.output_accesses):
        layout_tag = access.layout_class.value
        if layout_tag not in layout_tags:
            layout_tags.append(layout_tag)
    return JointLayoutSpec(layout_tags=tuple(layout_tags))


def _recipe_footprint(
    *,
    region: JointRegion,
    mandatory_actions: tuple[JointAction, ...],
    value_sizes: dict[str, int],
) -> JointValueFootprint:
    resident_bytes = sum(
        value_sizes.get(value_id, 0)
        for value_id in (*region.input_value_ids, *region.output_value_ids)
    )
    scratch_bytes = max(
        (action.temp_bytes for action in mandatory_actions if action.kind is JointActionKind.COMPUTE),
        default=0,
    )
    transfer_bytes = sum(
        value_sizes.get(action.reads[0], 0)
        for action in mandatory_actions
        if action.kind in (JointActionKind.DMA_IN, JointActionKind.DMA_OUT)
        and action.reads
    )
    return JointValueFootprint(
        resident_bytes=max(resident_bytes, 0),
        scratch_bytes=max(scratch_bytes, 0),
        transfer_bytes=max(transfer_bytes, 0),
    )


def _recipe_cost_parameters(
    mandatory_actions: tuple[JointAction, ...]
) -> JointCostParameters:
    return JointCostParameters(
        latency=sum(action.duration for action in mandatory_actions),
        launch_overhead=sum(action.launch_overhead for action in mandatory_actions),
    )


def _pipeline_resource_kind_for(plan: NodeExecutionPlan) -> PipelineResourceKind:
    if _is_large_tiled_op(plan):
        return PipelineResourceKind.MATMUL
    if _is_shape_family_op(plan):
        return PipelineResourceKind.SHAPE
    return PipelineResourceKind.OTHER


def _joint_resource_kind_for(
    pipeline_resource_kind: PipelineResourceKind,
) -> JointResourceKind:
    if pipeline_resource_kind is PipelineResourceKind.DMA:
        return JointResourceKind.DMA
    if pipeline_resource_kind is PipelineResourceKind.MATMUL:
        return JointResourceKind.MATMUL
    if pipeline_resource_kind is PipelineResourceKind.SHAPE:
        return JointResourceKind.SHAPE
    return JointResourceKind.OTHER


def _compute_temp_bytes(ctx: CompileContext, plan: NodeExecutionPlan) -> int:
    if _is_large_tiled_op(plan):
        return _scratch_bytes(ctx, plan)
    return 0


def _classify_external_value(ctx: CompileContext, value_id: str) -> JointValueTier:
    if value_id in ctx.graph.inputs:
        return JointValueTier.INPUT
    return JointValueTier.CONST


def _required_final_tier(
    ctx: CompileContext,
    *,
    value_id: str,
    producing_region_id: str | None,
) -> JointValueTier:
    if producing_region_id is not None:
        return JointValueTier.SLOW
    if value_id in ctx.graph.inputs:
        return JointValueTier.INPUT
    return JointValueTier.CONST


def _sram_capacity_bytes(ctx: CompileContext) -> int:
    for key in ("pipeline_sram_capacity_bytes", "max_memory"):
        value = ctx.metadata.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return sys.maxsize


def _validate_problem(problem: JointProblem) -> None:
    actions_by_id = {action.action_id: action for action in problem.actions}
    recipes_by_id = {recipe.recipe_id: recipe for recipe in problem.recipes}
    values_by_id = {value.value_id: value for value in problem.values}

    recipe_ids_by_action_id: dict[str, list[str]] = defaultdict(list)
    for recipe in problem.recipes:
        for action_id in recipe.activates_action_ids:
            recipe_ids_by_action_id[action_id].append(recipe.recipe_id)

    for action in problem.actions:
        if not action.is_optional:
            if action.recipe_id is None:
                raise JointProblemBuilderError(
                    f"mandatory action {action.action_id!r} must have recipe_id"
                )
            owning_recipe_ids = recipe_ids_by_action_id.get(action.action_id, [])
            if owning_recipe_ids != [action.recipe_id]:
                raise JointProblemBuilderError(
                    f"mandatory action {action.action_id!r} must belong to exactly one recipe"
                )
        elif action.optional_value_id is None:
            raise JointProblemBuilderError(
                f"optional action {action.action_id!r} must declare optional_value_id"
            )

    for value in problem.values:
        if value.value_id not in values_by_id:
            raise JointProblemBuilderError(f"unknown value {value.value_id!r}")
        if value.producer is not None:
            producer_action = actions_by_id.get(value.producer.action_id)
            if producer_action is None or value.value_id not in producer_action.writes:
                raise JointProblemBuilderError(
                    f"value {value.value_id!r} producer does not match action writes"
                )
        for consumer in value.consumers:
            consumer_action = actions_by_id.get(consumer.action_id)
            if consumer_action is None or value.value_id not in consumer_action.reads:
                raise JointProblemBuilderError(
                    f"value {value.value_id!r} consumer does not match action reads"
                )

    for action in problem.actions:
        for value_id in (*action.reads, *action.writes):
            if value_id not in values_by_id:
                raise JointProblemBuilderError(
                    f"action {action.action_id!r} references unknown value {value_id!r}"
                )

    for region in problem.regions:
        region_recipes = tuple(
            recipe for recipe in problem.recipes if recipe.region_id == region.region_id
        )
        for value_id in (*region.input_value_ids, *region.output_value_ids):
            if value_id not in values_by_id:
                raise JointProblemBuilderError(
                    f"region {region.region_id!r} references unknown value {value_id!r}"
                )
        region_actions = tuple(
            action for action in problem.actions if action.region_id == region.region_id
        )
        if any(
            not any(value_id in action.reads for action in region_actions)
            for value_id in region.input_value_ids
        ):
            raise JointProblemBuilderError(
                f"region {region.region_id!r} input interface does not match action reads"
            )
        if any(
            not any(value_id in action.writes for action in region_actions)
            for value_id in region.output_value_ids
        ):
            raise JointProblemBuilderError(
                f"region {region.region_id!r} output interface does not match action writes"
            )
        for recipe in region_recipes:
            recipe_compute_actions = tuple(
                action
                for action in region_actions
                if action.recipe_id == recipe.recipe_id and action.kind is JointActionKind.COMPUTE
            )
            if len(recipe_compute_actions) != 1:
                raise JointProblemBuilderError(
                    f"recipe {recipe.recipe_id!r} must have exactly one compute action"
                )

    adjacent_pairs = {
        (src.region_id, dst.region_id)
        for src in problem.regions
        for dst in problem.regions
        if src.region_id != dst.region_id
        and set(src.output_value_ids).intersection(dst.input_value_ids)
    }
    boundary_pairs = {
        (boundary.src_region_id, boundary.dst_region_id)
        for boundary in problem.boundary_constraints
    }
    if adjacent_pairs != boundary_pairs:
        raise JointProblemBuilderError("boundary constraints must cover every adjacent region pair exactly once")

    for boundary in problem.boundary_constraints:
        if boundary.src_region_id not in {region.region_id for region in problem.regions}:
            raise JointProblemBuilderError(
                f"boundary {boundary.boundary_id!r} references unknown src region"
            )
        if boundary.dst_region_id not in {region.region_id for region in problem.regions}:
            raise JointProblemBuilderError(
                f"boundary {boundary.boundary_id!r} references unknown dst region"
            )
        for pair in boundary.compatible_recipe_pairs:
            if pair.src_recipe_id not in recipes_by_id or pair.dst_recipe_id not in recipes_by_id:
                raise JointProblemBuilderError(
                    f"boundary {boundary.boundary_id!r} references unknown recipes"
                )


__all__ = ["build_joint_problem"]


def _plan_variants_for_region(
    ctx: CompileContext,
    *,
    region: JointRegion,
    base_plan: NodeExecutionPlan,
) -> tuple[NodeExecutionPlan, ...]:
    del region
    variants = [base_plan]
    seen = {_plan_variant_key(base_plan)}
    for candidate in _candidate_plan_variants(ctx, base_plan=base_plan):
        key = _plan_variant_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        variants.append(candidate)
    return tuple(variants)


def _candidate_plan_variants(
    ctx: CompileContext,
    *,
    base_plan: NodeExecutionPlan,
) -> tuple[NodeExecutionPlan, ...]:
    if tuple(base_plan.tile_axes) != ("h", "w"):
        return ()
    output_access = next(
        (access for access in base_plan.output_accesses if access.tile_region.logical_extents),
        None,
    )
    if output_access is None:
        return ()
    output_extents = tuple(int(max(dim, 0)) for dim in output_access.tile_region.logical_extents)
    if len(output_extents) < 2:
        return ()
    output_h = output_extents[-2]
    output_w = output_extents[-1]
    if output_h <= 1 and output_w <= 1:
        return ()

    candidate_hw: list[tuple[int, int]] = []
    for tile_h, tile_w in (
        (max(1, output_h // 2), output_w),
        (output_h, max(1, output_w // 2)),
        (max(1, output_h // 2), max(1, output_w // 2)),
        (1, output_w),
        (output_h, 1),
    ):
        if (tile_h, tile_w) != (output_h, output_w) and (tile_h, tile_w) not in candidate_hw:
            candidate_hw.append((tile_h, tile_w))

    variants: list[NodeExecutionPlan] = []
    for tile_h, tile_w in candidate_hw:
        output_accesses = tuple(
            _replace_access_hw(access, tile_h=tile_h, tile_w=tile_w)
            for access in base_plan.output_accesses
        )
        input_hw = _input_hw_for_variant(base_plan, tile_h=tile_h, tile_w=tile_w)
        input_accesses = tuple(
            _replace_access_hw(access, tile_h=input_hw[0], tile_w=input_hw[1])
            if access.tile_region.logical_extents
            else access
            for access in base_plan.input_accesses
        )
        variants.append(
            replace(
                base_plan,
                input_accesses=input_accesses,
                output_accesses=output_accesses,
            )
        )
    return tuple(variants)


def _replace_access_hw(
    access: TensorAccessPlan,
    *,
    tile_h: int,
    tile_w: int,
) -> TensorAccessPlan:
    if not access.tile_region.logical_extents:
        return access
    extents = access.tile_region.logical_extents
    if len(extents) < 2:
        return access
    tile_region = replace(
        access.tile_region,
        logical_extents=(*extents[:-2], tile_h, tile_w),
    )
    return replace(access, tile_region=tile_region)


def _input_hw_for_variant(
    plan: NodeExecutionPlan,
    *,
    tile_h: int,
    tile_w: int,
) -> tuple[int, int]:
    input_access = next(
        (access for access in plan.input_accesses if access.tile_region.logical_extents),
        None,
    )
    output_access = next(
        (access for access in plan.output_accesses if access.tile_region.logical_extents),
        None,
    )
    if input_access is None or output_access is None:
        return tile_h, tile_w
    input_extents = input_access.tile_region.logical_extents
    output_extents = output_access.tile_region.logical_extents
    if len(input_extents) < 2 or len(output_extents) < 2:
        return tile_h, tile_w
    halo_h = max(input_extents[-2] - output_extents[-2], 0)
    halo_w = max(input_extents[-1] - output_extents[-1], 0)
    return max(1, tile_h + halo_h), max(1, tile_w + halo_w)


def _plan_variant_key(plan: NodeExecutionPlan) -> tuple[object, ...]:
    return (
        plan.node_name,
        plan.tile_axes,
        tuple(access.tile_region.logical_extents for access in plan.input_accesses),
        tuple(access.tile_region.logical_extents for access in plan.output_accesses),
    )
