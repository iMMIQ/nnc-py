from nnc_py.ir.context import CompileContext
from nnc_py.ir.execution_plan import (
    LayoutClass,
    MemoryRegionKind,
    NodeExecutionPlan,
    get_node_execution_plan,
    get_node_execution_plans,
    set_node_execution_plan,
)
from nnc_py.ir.graph import Graph


def test_execution_plan_records_tile_and_layout_metadata():
    plan = NodeExecutionPlan(
        node_name="conv0",
        op_family="conv2d",
        tile_axes=("h", "w"),
        layout_class=LayoutClass.BLOCKED_ACTIVATION,
        memory_regions=(MemoryRegionKind.TILE, MemoryRegionKind.SCRATCH),
    )

    assert plan.node_name == "conv0"
    assert plan.op_family == "conv2d"
    assert plan.tile_axes == ("h", "w")
    assert plan.layout_class is LayoutClass.BLOCKED_ACTIVATION
    assert plan.memory_regions == (MemoryRegionKind.TILE, MemoryRegionKind.SCRATCH)


def test_execution_plan_accessors_store_and_load_typed_metadata():
    ctx = CompileContext(graph=Graph("plan_ctx"), target="x86")
    plan = NodeExecutionPlan(
        node_name="conv0",
        op_family="conv2d",
        tile_axes=("h", "w"),
        layout_class=LayoutClass.BLOCKED_ACTIVATION,
        memory_regions=(MemoryRegionKind.TILE, MemoryRegionKind.SCRATCH),
    )

    set_node_execution_plan(ctx, plan)

    assert ctx.metadata["node_execution_plans"] == {"conv0": plan}
    assert get_node_execution_plans(ctx) == {"conv0": plan}
    assert get_node_execution_plan(ctx, "conv0") is plan
    assert ctx.node_execution_plans == {"conv0": plan}
    assert ctx.get_node_execution_plan("conv0") is plan


def test_execution_plan_accessors_do_not_mutate_missing_metadata_on_read():
    ctx = CompileContext(graph=Graph("empty_ctx"), target="x86")

    assert "node_execution_plans" not in ctx.metadata
    assert get_node_execution_plans(ctx) == {}
    assert get_node_execution_plan(ctx, "missing") is None
    assert ctx.node_execution_plans == {}
    assert ctx.get_node_execution_plan("missing") is None
    assert "node_execution_plans" not in ctx.metadata
