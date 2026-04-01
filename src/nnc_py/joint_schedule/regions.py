"""Region builders for the external joint tiling/schedule problem."""

from __future__ import annotations

from nnc_py.ir.context import CompileContext
from nnc_py.ir.execution_plan import NodeExecutionPlan
from nnc_py.ir.joint_tiling_schedule import JointRegion, JointRegionKind
from nnc_py.passes.pipeline_step_lowering import _synthesize_execution_plan


class JointProblemBuilderError(ValueError):
    """Raised when the compiler cannot emit a consistent joint problem."""


def build_joint_regions(ctx: CompileContext) -> tuple[JointRegion, ...]:
    """Emit one region per current execution-plan decision unit."""

    plans = get_joint_problem_plans(ctx)
    if not plans:
        return ()

    ordered_nodes = []
    seen_node_names: set[str] = set()
    for node in ctx.graph.topological_sort():
        if node.name in plans:
            ordered_nodes.append(node)
            seen_node_names.add(node.name)

    unknown_nodes = sorted(set(plans) - seen_node_names)
    if unknown_nodes:
        raise JointProblemBuilderError(
            f"node_execution_plans reference unknown graph nodes: {unknown_nodes}"
        )

    producer_region_by_value: dict[str, str] = {}
    region_specs: list[tuple[str, JointRegionKind, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    for node in ordered_nodes:
        plan = plans[node.name]
        member_nodes = _member_nodes_for(node)
        kind = (
            JointRegionKind.FUSED_GROUP
            if "fused_from" in node.metadata
            else JointRegionKind.SINGLE_OP
        )
        input_value_ids = tuple(
            _validate_region_tensor_access(
                ctx,
                plans,
                region_id=node.name,
                tensor_name=access.tensor_name,
                direction="input",
            )
            for access in plan.input_accesses
        )
        output_value_ids = tuple(
            _validate_region_tensor_access(
                ctx,
                plans,
                region_id=node.name,
                tensor_name=access.tensor_name,
                direction="output",
            )
            for access in plan.output_accesses
        )
        region_specs.append(
            (node.name, kind, member_nodes, input_value_ids, output_value_ids)
        )
        for value_id in output_value_ids:
            existing_region_id = producer_region_by_value.get(value_id)
            if existing_region_id is not None and existing_region_id != node.name:
                raise JointProblemBuilderError(
                    f"value {value_id!r} is produced by multiple regions: "
                    f"{existing_region_id!r} and {node.name!r}"
                )
            producer_region_by_value[value_id] = node.name

    predecessor_ids_by_region: dict[str, list[str]] = {
        region_id: [] for region_id, *_ in region_specs
    }
    successor_ids_by_region: dict[str, list[str]] = {
        region_id: [] for region_id, *_ in region_specs
    }
    for region_id, _kind, _members, input_value_ids, _output_value_ids in region_specs:
        for value_id in input_value_ids:
            predecessor_region_id = producer_region_by_value.get(value_id)
            if predecessor_region_id is None or predecessor_region_id == region_id:
                continue
            if predecessor_region_id not in predecessor_ids_by_region[region_id]:
                predecessor_ids_by_region[region_id].append(predecessor_region_id)
            if region_id not in successor_ids_by_region[predecessor_region_id]:
                successor_ids_by_region[predecessor_region_id].append(region_id)

    return tuple(
        JointRegion(
            region_id=region_id,
            kind=kind,
            member_nodes=member_nodes,
            input_value_ids=input_value_ids,
            output_value_ids=output_value_ids,
            predecessor_region_ids=tuple(predecessor_ids_by_region[region_id]),
            successor_region_ids=tuple(successor_ids_by_region[region_id]),
        )
        for region_id, kind, member_nodes, input_value_ids, output_value_ids in region_specs
    )


def get_joint_problem_plans(ctx: CompileContext) -> dict[str, NodeExecutionPlan]:
    """Return complete execution plans for the joint problem.

    The joint path needs full graph coverage, including shape/plain ops that do
    not participate in tiled lowering. Reuse synthesized plans for uncovered
    nodes instead of requiring every producer to be present in
    ``ctx.node_execution_plans``.
    """

    plans = dict(ctx.node_execution_plans)
    for node in ctx.graph.topological_sort():
        if node.name in plans:
            continue
        plans[node.name] = _synthesize_execution_plan(ctx, node)
    return plans


def _member_nodes_for(node) -> tuple[str, ...]:
    fused_from = node.metadata.get("fused_from")
    if fused_from is None:
        return (node.name,)
    if isinstance(fused_from, str):
        raise JointProblemBuilderError(
            f"node.metadata['fused_from'] for {node.name!r} must be a sequence of node names"
        )
    try:
        member_nodes = tuple(fused_from)
    except TypeError as exc:
        raise JointProblemBuilderError(
            f"node.metadata['fused_from'] for {node.name!r} must be iterable"
        ) from exc
    if not member_nodes or not all(isinstance(member, str) for member in member_nodes):
        raise JointProblemBuilderError(
            f"node.metadata['fused_from'] for {node.name!r} must contain only strings"
        )
    return member_nodes


def _validate_region_tensor_access(
    ctx: CompileContext,
    plans: dict[str, object],
    *,
    region_id: str,
    tensor_name: str,
    direction: str,
) -> str:
    if tensor_name not in ctx.graph.tensors:
        raise JointProblemBuilderError(
            f"region {region_id!r} {direction} access references unknown tensor {tensor_name!r}"
        )
    if direction == "input":
        producer_nodes = ctx.graph.get_producers(tensor_name)
        if producer_nodes and not any(producer.name in plans for producer in producer_nodes):
            raise JointProblemBuilderError(
                f"region {region_id!r} consumes tensor {tensor_name!r} from a producer "
                "that has no execution plan"
            )
    return tensor_name


__all__ = [
    "JointProblemBuilderError",
    "build_joint_regions",
    "get_joint_problem_plans",
]
