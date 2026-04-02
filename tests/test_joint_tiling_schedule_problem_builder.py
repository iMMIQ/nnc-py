import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.execution_plan import (
    LayoutClass,
    MemoryRegionKind,
    NodeExecutionPlan,
    TensorAccessPlan,
    TileRegion,
    set_node_execution_plan,
)
from nnc_py.ir.graph import Graph
from nnc_py.ir.joint_tiling_schedule import (
    JointActionKind,
    JointRegionKind,
    JointResourceKind,
    JointSramItemKind,
    JointValueTier,
)
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType, MemoryLayout
from nnc_py.joint_schedule import JointProblemBuilderError
from nnc_py.joint_schedule.recipes import build_joint_problem
from nnc_py.joint_schedule.regions import build_joint_regions


def _make_fused_region_context() -> CompileContext:
    graph = Graph("fused_region")
    graph.inputs = ["input"]
    graph.outputs = ["relu_out"]

    graph.add_tensor(
        TensorType(
            name="input",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 16, 32, 32], layout=MemoryLayout.NCHW),
        )
    )
    graph.add_tensor(
        TensorType(
            name="weight",
            dtype=DataType.FLOAT32,
            shape=TensorShape([16, 16, 3, 3], layout=MemoryLayout.OIHW),
        )
    )
    graph.add_tensor(
        TensorType(
            name="relu_out",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 16, 32, 32], layout=MemoryLayout.NCHW),
        )
    )
    graph.constants["weight"] = [1.0]
    graph.add_node(
        Node(
            op_type=OpType.FUSED_CONV_RELU,
            name="conv0_group",
            inputs=["input", "weight"],
            outputs=["relu_out"],
            metadata={"fused_from": ["conv0", "relu0"]},
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    set_node_execution_plan(
        ctx,
        NodeExecutionPlan(
            node_name="conv0_group",
            op_family="conv2d",
            tile_axes=("h", "w"),
            layout_class=LayoutClass.BLOCKED_ACTIVATION,
            memory_regions=(MemoryRegionKind.TILE, MemoryRegionKind.SCRATCH),
            input_accesses=(
                TensorAccessPlan(
                    tensor_name="input",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    tile_region=TileRegion(
                        logical_extents=(1, 16, 16, 16),
                        halo_extents=(0, 0, 1, 1),
                    ),
                    memory_region=MemoryRegionKind.TILE,
                ),
                TensorAccessPlan(
                    tensor_name="weight",
                    layout_class=LayoutClass.BLOCKED_WEIGHT,
                    memory_region=MemoryRegionKind.PERSISTENT,
                ),
            ),
            output_accesses=(
                TensorAccessPlan(
                    tensor_name="relu_out",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    tile_region=TileRegion(logical_extents=(1, 16, 16, 16)),
                    memory_region=MemoryRegionKind.TILE,
                ),
            ),
        ),
    )
    ctx.metadata["node_execution_plan_region_sizes"] = {
        "conv0_group": {
            "tensor_bytes": {
                "input": 1 * 16 * 18 * 18 * 4,
                "relu_out": 1 * 16 * 16 * 16 * 4,
            },
            "region_bytes": {
                "scratch": 2048,
            },
        }
    }
    return ctx


def _make_two_region_context() -> CompileContext:
    graph = Graph("two_region_problem")
    graph.inputs = ["input"]
    graph.outputs = ["out"]

    tensor_specs = (
        ("input", [1, 16, 32, 32], MemoryLayout.NCHW),
        ("weight0", [16, 16, 3, 3], MemoryLayout.OIHW),
        ("weight1", [16, 16, 3, 3], MemoryLayout.OIHW),
        ("mid", [1, 16, 32, 32], MemoryLayout.NCHW),
        ("out", [1, 16, 32, 32], MemoryLayout.NCHW),
    )
    for name, dims, layout in tensor_specs:
        graph.add_tensor(
            TensorType(
                name=name,
                dtype=DataType.FLOAT32,
                shape=TensorShape(dims, layout=layout),
            )
        )
    graph.constants["weight0"] = [1.0]
    graph.constants["weight1"] = [1.0]
    graph.add_node(
        Node(
            op_type=OpType.CONV2D,
            name="conv0",
            inputs=["input", "weight0"],
            outputs=["mid"],
            attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["mid", "weight1"],
            outputs=["out"],
            attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    ctx.metadata["pipeline_sram_capacity_bytes"] = 32768
    for node_name, input_name, weight_name, output_name in (
        ("conv0", "input", "weight0", "mid"),
        ("conv1", "mid", "weight1", "out"),
    ):
        set_node_execution_plan(
            ctx,
            NodeExecutionPlan(
                node_name=node_name,
                op_family="conv2d",
                tile_axes=("h", "w"),
                layout_class=LayoutClass.BLOCKED_ACTIVATION,
                memory_regions=(
                    MemoryRegionKind.TILE,
                    MemoryRegionKind.SCRATCH,
                ),
                input_accesses=(
                    TensorAccessPlan(
                        tensor_name=input_name,
                        layout_class=LayoutClass.BLOCKED_ACTIVATION,
                        tile_region=TileRegion(
                            logical_extents=(1, 16, 16, 16),
                            halo_extents=(0, 0, 1, 1),
                        ),
                        memory_region=MemoryRegionKind.TILE,
                    ),
                    TensorAccessPlan(
                        tensor_name=weight_name,
                        layout_class=LayoutClass.BLOCKED_WEIGHT,
                        memory_region=MemoryRegionKind.PERSISTENT,
                    ),
                ),
                output_accesses=(
                    TensorAccessPlan(
                        tensor_name=output_name,
                        layout_class=LayoutClass.BLOCKED_ACTIVATION,
                        tile_region=TileRegion(logical_extents=(1, 16, 16, 16)),
                        memory_region=MemoryRegionKind.TILE,
                    ),
                ),
            ),
        )
    ctx.metadata["node_execution_plan_region_sizes"] = {
        "conv0": {
            "tensor_bytes": {
                "input": 1 * 16 * 18 * 18 * 4,
                "mid": 1 * 16 * 16 * 16 * 4,
            },
            "region_bytes": {
                "scratch": 1024,
            },
        },
        "conv1": {
            "tensor_bytes": {
                "mid": 1 * 16 * 18 * 18 * 4,
                "out": 1 * 16 * 16 * 16 * 4,
            },
            "region_bytes": {
                "scratch": 1536,
            },
        },
    }
    return ctx


def _make_pack_region_context() -> CompileContext:
    graph = Graph("pack_region_problem")
    graph.inputs = ["lhs"]
    graph.outputs = ["out"]

    for name, dims in (
        ("lhs", [1, 32]),
        ("rhs", [32, 32]),
        ("out", [1, 32]),
    ):
        graph.add_tensor(
            TensorType(
                name=name,
                dtype=DataType.FLOAT32,
                shape=TensorShape(dims),
            )
        )
    graph.constants["rhs"] = [1.0]
    graph.add_node(
        Node(
            op_type=OpType.GEMM,
            name="gemm0",
            inputs=["lhs", "rhs"],
            outputs=["out"],
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    ctx.metadata["pipeline_sram_capacity_bytes"] = 4096
    set_node_execution_plan(
        ctx,
        NodeExecutionPlan(
            node_name="gemm0",
            op_family="gemm",
            tile_axes=("m", "n"),
            layout_class=LayoutClass.BLOCKED_ACTIVATION,
            memory_regions=(MemoryRegionKind.TILE, MemoryRegionKind.PACK),
            input_accesses=(
                TensorAccessPlan(
                    tensor_name="lhs",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    tile_region=TileRegion(logical_extents=(1, 32)),
                    memory_region=MemoryRegionKind.TILE,
                ),
                TensorAccessPlan(
                    tensor_name="rhs",
                    layout_class=LayoutClass.BLOCKED_WEIGHT,
                    memory_region=MemoryRegionKind.PERSISTENT,
                ),
            ),
            output_accesses=(
                TensorAccessPlan(
                    tensor_name="out",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    tile_region=TileRegion(logical_extents=(1, 32)),
                    memory_region=MemoryRegionKind.TILE,
                ),
            ),
        ),
    )
    ctx.metadata["node_execution_plan_region_sizes"] = {
        "gemm0": {
            "tensor_bytes": {
                "lhs": 128,
                "out": 128,
            },
            "region_bytes": {
                "pack": 80,
            },
        }
    }
    return ctx


def _make_inconsistent_context() -> CompileContext:
    graph = Graph("inconsistent_problem")
    graph.inputs = ["input"]
    graph.outputs = ["out"]

    graph.add_tensor(
        TensorType(
            name="input",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 8]),
        )
    )
    graph.add_tensor(
        TensorType(
            name="out",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 8]),
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.RELU,
            name="relu0",
            inputs=["input"],
            outputs=["out"],
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    set_node_execution_plan(
        ctx,
        NodeExecutionPlan(
            node_name="relu0",
            op_family="relu",
            input_accesses=(
                TensorAccessPlan(
                    tensor_name="ghost",
                    memory_region=MemoryRegionKind.PERSISTENT,
                ),
            ),
            output_accesses=(
                TensorAccessPlan(
                    tensor_name="out",
                    memory_region=MemoryRegionKind.PERSISTENT,
                ),
            ),
        ),
    )
    return ctx


def _make_extra_plan_access_context() -> CompileContext:
    graph = Graph("extra_plan_access")
    graph.inputs = ["input"]
    graph.outputs = ["out"]

    graph.add_tensor(
        TensorType(
            name="input",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 8]),
        )
    )
    graph.add_tensor(
        TensorType(
            name="weight",
            dtype=DataType.FLOAT32,
            shape=TensorShape([8, 8]),
        )
    )
    graph.add_tensor(
        TensorType(
            name="out",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 8]),
        )
    )
    graph.constants["weight"] = [1.0]
    graph.add_node(
        Node(
            op_type=OpType.RELU,
            name="relu0",
            inputs=["input"],
            outputs=["out"],
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    set_node_execution_plan(
        ctx,
        NodeExecutionPlan(
            node_name="relu0",
            op_family="relu",
            input_accesses=(
                TensorAccessPlan(
                    tensor_name="input",
                    memory_region=MemoryRegionKind.PERSISTENT,
                ),
                TensorAccessPlan(
                    tensor_name="weight",
                    memory_region=MemoryRegionKind.PERSISTENT,
                ),
            ),
            output_accesses=(
                TensorAccessPlan(
                    tensor_name="out",
                    memory_region=MemoryRegionKind.PERSISTENT,
                ),
            ),
        ),
    )
    return ctx


def _make_partial_plan_coverage_context() -> CompileContext:
    graph = Graph("partial_plan_coverage")
    graph.inputs = ["input"]
    graph.outputs = ["out"]

    for name, dims, layout in (
        ("input", [1, 8, 8, 8], MemoryLayout.NCHW),
        ("mid", [1, 8, 8, 8], MemoryLayout.NCHW),
        ("weight", [8, 8, 3, 3], MemoryLayout.OIHW),
        ("out", [1, 8, 8, 8], MemoryLayout.NCHW),
    ):
        graph.add_tensor(
            TensorType(
                name=name,
                dtype=DataType.FLOAT32,
                shape=TensorShape(dims, layout=layout),
            )
        )
    graph.constants["weight"] = [1.0]
    graph.add_node(
        Node(
            op_type=OpType.RELU,
            name="relu0",
            inputs=["input"],
            outputs=["mid"],
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.CONV2D,
            name="conv0",
            inputs=["mid", "weight"],
            outputs=["out"],
            attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    set_node_execution_plan(
        ctx,
        NodeExecutionPlan(
            node_name="conv0",
            op_family="conv2d",
            tile_axes=("h", "w"),
            layout_class=LayoutClass.BLOCKED_ACTIVATION,
            memory_regions=(MemoryRegionKind.TILE, MemoryRegionKind.SCRATCH),
            input_accesses=(
                TensorAccessPlan(
                    tensor_name="mid",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    tile_region=TileRegion(logical_extents=(1, 8, 6, 6)),
                    memory_region=MemoryRegionKind.TILE,
                ),
                TensorAccessPlan(
                    tensor_name="weight",
                    layout_class=LayoutClass.BLOCKED_WEIGHT,
                    memory_region=MemoryRegionKind.PERSISTENT,
                ),
            ),
            output_accesses=(
                TensorAccessPlan(
                    tensor_name="out",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    tile_region=TileRegion(logical_extents=(1, 8, 4, 4)),
                    memory_region=MemoryRegionKind.TILE,
                ),
            ),
        ),
    )
    return ctx


def _make_non_trailing_axis_context() -> CompileContext:
    ctx = _make_fused_region_context()
    set_node_execution_plan(
        ctx,
        NodeExecutionPlan(
            node_name="conv0_group",
            op_family="conv2d",
            tile_axes=("n", "h"),
            layout_class=LayoutClass.BLOCKED_ACTIVATION,
            memory_regions=(MemoryRegionKind.TILE, MemoryRegionKind.SCRATCH),
            input_accesses=(
                TensorAccessPlan(
                    tensor_name="input",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    tile_region=TileRegion(
                        logical_extents=(1, 16, 11, 13),
                        halo_extents=(0, 0, 1, 1),
                    ),
                    memory_region=MemoryRegionKind.TILE,
                ),
                TensorAccessPlan(
                    tensor_name="weight",
                    layout_class=LayoutClass.BLOCKED_WEIGHT,
                    memory_region=MemoryRegionKind.PERSISTENT,
                ),
            ),
            output_accesses=(
                TensorAccessPlan(
                    tensor_name="relu_out",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    tile_region=TileRegion(logical_extents=(1, 16, 9, 11)),
                    memory_region=MemoryRegionKind.TILE,
                ),
            ),
        ),
    )
    return ctx


def _make_reduced_rank_tile_axis_context() -> CompileContext:
    graph = Graph("reduced_rank_tile_axes")
    graph.inputs = ["input"]
    graph.outputs = ["out"]

    for name, dims, layout in (
        ("input", [1, 8, 16, 16], MemoryLayout.NCHW),
        ("weight", [8, 8, 3, 3], MemoryLayout.OIHW),
        ("out", [1, 8, 16, 16], MemoryLayout.NCHW),
    ):
        graph.add_tensor(
            TensorType(
                name=name,
                dtype=DataType.FLOAT32,
                shape=TensorShape(dims, layout=layout),
            )
        )
    graph.constants["weight"] = [1.0]
    graph.add_node(
        Node(
            op_type=OpType.CONV2D,
            name="conv0",
            inputs=["input", "weight"],
            outputs=["out"],
            attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    set_node_execution_plan(
        ctx,
        NodeExecutionPlan(
            node_name="conv0",
            op_family="conv2d",
            tile_axes=("h", "w"),
            layout_class=LayoutClass.BLOCKED_ACTIVATION,
            memory_regions=(MemoryRegionKind.TILE, MemoryRegionKind.SCRATCH),
            input_accesses=(
                TensorAccessPlan(
                    tensor_name="input",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    tile_region=TileRegion(logical_extents=(17, 19)),
                    memory_region=MemoryRegionKind.TILE,
                ),
                TensorAccessPlan(
                    tensor_name="weight",
                    layout_class=LayoutClass.BLOCKED_WEIGHT,
                    memory_region=MemoryRegionKind.PERSISTENT,
                ),
            ),
            output_accesses=(
                TensorAccessPlan(
                    tensor_name="out",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    tile_region=TileRegion(logical_extents=(16, 16)),
                    memory_region=MemoryRegionKind.TILE,
                ),
            ),
        ),
    )
    return ctx


def test_build_joint_regions_uses_fused_metadata_for_region_interfaces():
    ctx = _make_fused_region_context()

    regions = build_joint_regions(ctx)

    assert [region.region_id for region in regions] == ["conv0_group"]
    assert regions[0].kind is JointRegionKind.FUSED_GROUP
    assert regions[0].member_nodes == ("conv0", "relu0")
    assert regions[0].input_value_ids == ("input", "weight")
    assert regions[0].output_value_ids == ("relu_out",)
    assert regions[0].predecessor_region_ids == ()
    assert regions[0].successor_region_ids == ()


def test_build_joint_problem_emits_boundaries_actions_and_logical_values():
    ctx = _make_two_region_context()

    problem = build_joint_problem(ctx)

    assert [region.region_id for region in problem.regions] == ["conv0", "conv1"]
    assert {recipe.region_id for recipe in problem.recipes} == {"conv0", "conv1"}
    assert len([recipe for recipe in problem.recipes if recipe.region_id == "conv0"]) >= 2
    assert len([recipe for recipe in problem.recipes if recipe.region_id == "conv1"]) >= 2
    assert problem.values
    assert problem.actions
    assert problem.boundary_constraints
    assert tuple(resource.resource_kind for resource in problem.resources) == (
        JointResourceKind.DMA,
        JointResourceKind.MATMUL,
        JointResourceKind.SHAPE,
        JointResourceKind.OTHER,
    )
    assert problem.sram_capacity_bytes == 32768

    boundary = problem.boundary_constraints[0]
    assert boundary.src_region_id == "conv0"
    assert boundary.dst_region_id == "conv1"
    compatible_pairs = {
        (pair.src_recipe_id, pair.dst_recipe_id) for pair in boundary.compatible_recipe_pairs
    }
    assert ("conv0.recipe0", "conv1.recipe0") in compatible_pairs
    assert ("conv0.recipe1", "conv1.recipe1") in compatible_pairs

    values = {value.value_id: value for value in problem.values}
    assert values["input"].initial_tier is JointValueTier.INPUT
    assert values["weight0"].initial_tier is JointValueTier.CONST
    assert values["mid"].initial_tier is JointValueTier.UNMATERIALIZED
    assert values["mid"].required_final_tier is JointValueTier.SLOW
    assert values["mid"].spillable is True
    assert values["mid"].allows_multiple_sram_windows is True
    assert values["out"].required_final_tier is JointValueTier.SLOW

    actions = {action.action_id: action for action in problem.actions}
    assert actions["conv0.recipe0.compute"].kind is JointActionKind.COMPUTE
    assert actions["conv1.recipe0.compute"].kind is JointActionKind.COMPUTE
    assert actions["conv0.recipe1.compute"].kind is JointActionKind.COMPUTE
    assert actions["conv1.recipe1.compute"].kind is JointActionKind.COMPUTE
    assert actions["conv0.recipe0.dma_in.input"].kind is JointActionKind.DMA_IN
    assert actions["conv0.recipe0.dma_in.weight0"].kind is JointActionKind.DMA_IN
    assert actions["conv1.recipe0.dma_in.weight1"].kind is JointActionKind.DMA_IN
    assert actions["conv1.recipe0.dma_out.out"].kind is JointActionKind.DMA_OUT
    assert actions["mid.spill"].kind is JointActionKind.SPILL
    assert actions["mid.reload"].kind is JointActionKind.RELOAD
    assert all(
        not any(value_id.startswith("sram|") for value_id in (*action.reads, *action.writes))
        for action in problem.actions
    )
    assert any(
        edge.src_action_id == "conv0.recipe0.compute"
        and edge.dst_action_id == "conv1.recipe0.compute"
        for edge in problem.dependency_edges
    )
    assert any(
        edge.src_action_id == "conv0.recipe1.compute"
        and edge.dst_action_id == "conv1.recipe1.compute"
        for edge in problem.dependency_edges
    )


def test_build_joint_problem_emits_fixed_sram_items_for_compute_actions():
    ctx = _make_two_region_context()

    problem = build_joint_problem(ctx)

    sram_items = {item.item_id: item for item in problem.sram_items}

    assert problem.default_alignment_bytes == 16
    assert problem.sram_items
    assert all(
        item.kind is not JointSramItemKind.RESIDENT_WINDOW for item in problem.sram_items
    )
    assert sram_items["conv0.recipe0.compute.temp"].kind is JointSramItemKind.TEMP_INTERVAL
    assert sram_items["conv0.recipe0.compute.temp"].size_bytes == 1024
    assert sram_items["conv0.recipe0.compute.temp"].alignment_bytes == 16
    assert sram_items["conv0.recipe0.compute.temp"].owner_action_id == "conv0.recipe0.compute"
    assert "conv0.recipe1.compute.temp" in sram_items
    assert sram_items["conv1.recipe0.compute.temp"].kind is JointSramItemKind.TEMP_INTERVAL
    assert sram_items["conv1.recipe0.compute.temp"].size_bytes == 1536
    assert sram_items["conv1.recipe0.compute.temp"].alignment_bytes == 16
    assert sram_items["conv1.recipe0.compute.temp"].owner_action_id == "conv1.recipe0.compute"
    assert "conv1.recipe1.compute.temp" in sram_items


def test_build_joint_problem_uses_recipe_specific_dma_sizes_and_footprints():
    ctx = _make_two_region_context()

    problem = build_joint_problem(ctx)

    actions = {action.action_id: action for action in problem.actions}
    recipes = {recipe.recipe_id: recipe for recipe in problem.recipes}
    values = {value.value_id: value for value in problem.values}

    assert (
        actions["conv0.recipe0.dma_in.input"].duration
        > actions["conv0.recipe4.dma_in.input"].duration
    )
    assert (
        actions["conv1.recipe0.dma_out.out"].duration
        > actions["conv1.recipe4.dma_out.out"].duration
    )
    assert (
        recipes["conv0.recipe0"].value_footprint.resident_bytes
        > recipes["conv0.recipe4"].value_footprint.resident_bytes
    )
    assert (
        recipes["conv0.recipe0"].value_footprint.transfer_bytes
        > recipes["conv0.recipe4"].value_footprint.transfer_bytes
    )
    assert values["mid"].size_bytes == 20736
    assert values["out"].size_bytes == 16384


def test_build_joint_problem_does_not_invent_transfer_buffer_items_from_region_hints():
    ctx = _make_pack_region_context()

    problem = build_joint_problem(ctx)

    assert all(
        item.kind is not JointSramItemKind.TRANSFER_BUFFER for item in problem.sram_items
    )


def test_build_joint_problem_raises_typed_error_for_inconsistent_interfaces():
    ctx = _make_inconsistent_context()

    with pytest.raises(JointProblemBuilderError):
        build_joint_problem(ctx)


def test_build_joint_problem_uses_plan_accesses_for_region_values():
    ctx = _make_extra_plan_access_context()

    problem = build_joint_problem(ctx)

    assert problem.regions[0].input_value_ids == ("input", "weight")
    assert {value.value_id for value in problem.values} == {"input", "weight", "out"}


def test_build_joint_regions_rejects_unknown_tensor_accesses():
    ctx = _make_inconsistent_context()

    with pytest.raises(JointProblemBuilderError):
        build_joint_regions(ctx)


def test_build_joint_problem_synthesizes_missing_plain_execution_plan_coverage():
    ctx = _make_partial_plan_coverage_context()

    problem = build_joint_problem(ctx)

    assert {region.region_id for region in problem.regions} == {"relu0", "conv0"}
    relu_recipes = [recipe for recipe in problem.recipes if recipe.region_id == "relu0"]
    conv_recipes = [recipe for recipe in problem.recipes if recipe.region_id == "conv0"]
    assert len(relu_recipes) == 1
    assert relu_recipes[0].tile_spec.axes == ()
    assert relu_recipes[0].tile_spec.shape == ()
    assert conv_recipes[0].tile_spec.axes == ("h", "w")


def test_build_joint_problem_maps_tile_shape_by_named_axes():
    ctx = _make_non_trailing_axis_context()

    problem = build_joint_problem(ctx)

    assert problem.recipes[0].tile_spec.axes == ("n", "h")
    assert problem.recipes[0].tile_spec.shape == (1, 9)


def test_build_joint_problem_keeps_reduced_rank_tile_extents_in_axis_order():
    ctx = _make_reduced_rank_tile_axis_context()

    problem = build_joint_problem(ctx)

    assert problem.recipes[0].tile_spec.axes == ("h", "w")
    assert problem.recipes[0].tile_spec.shape == (16, 16)
