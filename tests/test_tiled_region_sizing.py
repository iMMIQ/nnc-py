from nnc_py.passes.tiled_lowering import TiledLoweringPass

from tests.test_tiled_lowering_pass import (
    make_dynamic_spatial_conv_context,
    make_maxpool_context,
    seed_schedule_and_layout,
)


def test_tiled_lowering_records_region_size_hints_for_maxpool_plan():
    ctx = make_maxpool_context()
    seed_schedule_and_layout(ctx, node_name="pool0", op_family="maxpool")

    TiledLoweringPass().run(ctx)

    region_sizes = ctx.metadata["node_execution_plan_region_sizes"]["pool0"]

    assert region_sizes["tensor_bytes"]["input"] > 0
    assert region_sizes["tensor_bytes"]["output"] > 0
    assert region_sizes["region_bytes"].get("scratch", 0) == 0


def test_tiled_lowering_omits_unresolved_tensor_hints_for_dynamic_spatial_conv():
    ctx = make_dynamic_spatial_conv_context()
    seed_schedule_and_layout(ctx, node_name="conv0", op_family="conv2d")

    TiledLoweringPass().run(ctx)

    plan = ctx.metadata["node_execution_plans"]["conv0"]
    region_sizes = ctx.metadata["node_execution_plan_region_sizes"]["conv0"]

    assert plan.output_accesses[0].tile_region.logical_extents == ()
    assert region_sizes["tensor_bytes"] == {}
    assert region_sizes["region_bytes"]["scratch"] > 0
