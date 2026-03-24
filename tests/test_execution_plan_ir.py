from nnc_py.ir.execution_plan import LayoutClass, MemoryRegionKind, NodeExecutionPlan


def test_execution_plan_records_tile_and_layout_metadata():
    plan = NodeExecutionPlan(
        node_name="conv0",
        op_family="conv2d",
        tile_axes=("h", "w"),
        layout_class=LayoutClass.BLOCKED_ACTIVATION,
        memory_regions=(MemoryRegionKind.TILE, MemoryRegionKind.SCRATCH),
    )

    assert plan.node_name == "conv0"
    assert plan.layout_class is LayoutClass.BLOCKED_ACTIVATION
    assert MemoryRegionKind.SCRATCH in plan.memory_regions
