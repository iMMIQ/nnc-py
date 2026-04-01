import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.execution_plan import (
    LayoutClass,
    MemoryRegionKind,
    TensorAccessPlan,
)
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType, GenericBlockedLayoutKind, MemoryLayout
from nnc_py.passes.layout_planning import GenericBlockedLayout, LayoutPlan
from nnc_py.passes.schedule_analysis import FAST_MEMORY_BUDGET_BYTES, ScheduleCandidate
from nnc_py.passes.tiled_lowering import TiledLoweringPass


def make_conv_context() -> CompileContext:
    graph = Graph("conv_tiled")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 56, 56], layout=MemoryLayout.NCHW),
            name="input",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([64, 64, 3, 3], layout=MemoryLayout.OIHW),
            name="weight",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 56, 56], layout=MemoryLayout.NCHW),
            name="output",
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.CONV2D,
            name="conv0",
            inputs=["input", "weight"],
            outputs=["output"],
            attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )

    return CompileContext(graph=graph, target="x86", optimization_level=3)


def make_maxpool_context() -> CompileContext:
    graph = Graph("pool_tiled")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 16, 16, 16], layout=MemoryLayout.NCHW),
            name="input",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 16, 8, 8], layout=MemoryLayout.NCHW),
            name="output",
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.MAXPOOL,
            name="pool0",
            inputs=["input"],
            outputs=["output"],
            attrs={"kernel_shape": [2, 2], "strides": [2, 2]},
        )
    )

    return CompileContext(graph=graph, target="x86", optimization_level=3)


def make_safe_gemm_context() -> CompileContext:
    graph = Graph("gemm_tiled")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 512]),
            name="input",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([512, 1000]),
            name="weight",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1000]),
            name="bias",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 1000]),
            name="output",
        )
    )
    graph.constants["weight"] = [1.0]
    graph.constants["bias"] = [0.0]
    graph.add_node(
        Node(
            op_type=OpType.GEMM,
            name="fc",
            inputs=["input", "weight", "bias"],
            outputs=["output"],
            attrs={"transB": 0},
        )
    )

    return CompileContext(graph=graph, target="x86", optimization_level=3)


def make_valid_conv_context() -> CompileContext:
    graph = Graph("conv_valid_tiled")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 56, 56], layout=MemoryLayout.NCHW),
            name="input",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([64, 64, 3, 3], layout=MemoryLayout.OIHW),
            name="weight",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 54, 54], layout=MemoryLayout.NCHW),
            name="output",
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.CONV2D,
            name="conv0",
            inputs=["input", "weight"],
            outputs=["output"],
            attrs={"kernel_shape": [3, 3], "pads": [0, 0, 0, 0], "strides": [1, 1]},
        )
    )

    return CompileContext(graph=graph, target="x86", optimization_level=3)


def make_overlapping_maxpool_context() -> CompileContext:
    graph = Graph("pool_overlap_tiled")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 16, 16, 16], layout=MemoryLayout.NCHW),
            name="input",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 16, 14, 14], layout=MemoryLayout.NCHW),
            name="output",
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.MAXPOOL,
            name="pool0",
            inputs=["input"],
            outputs=["output"],
            attrs={"kernel_shape": [3, 3], "strides": [1, 1]},
        )
    )

    return CompileContext(graph=graph, target="x86", optimization_level=3)


def make_dynamic_spatial_conv_context() -> CompileContext:
    graph = Graph("conv_dynamic_tiled")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, "H", "W"], layout=MemoryLayout.NCHW),
            name="input",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([64, 64, 3, 3], layout=MemoryLayout.OIHW),
            name="weight",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, "H", "W"], layout=MemoryLayout.NCHW),
            name="output",
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.CONV2D,
            name="conv0",
            inputs=["input", "weight"],
            outputs=["output"],
            attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )

    return CompileContext(graph=graph, target="x86", optimization_level=3)


def make_oversubscribed_conv_context() -> CompileContext:
    graph = Graph("conv_oversubscribed_tiled")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 4096, 64, 64], layout=MemoryLayout.NCHW),
            name="input",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([4096, 4096, 3, 3], layout=MemoryLayout.OIHW),
            name="weight",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 4096, 64, 64], layout=MemoryLayout.NCHW),
            name="output",
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.CONV2D,
            name="conv0",
            inputs=["input", "weight"],
            outputs=["output"],
            attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )

    return CompileContext(graph=graph, target="x86", optimization_level=3)


def make_partial_dynamic_conv_context() -> CompileContext:
    graph = Graph("conv_partial_dynamic_tiled")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, "C", 56, 56], layout=MemoryLayout.NCHW),
            name="input",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([64, "C", 3, 3], layout=MemoryLayout.OIHW),
            name="weight",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 56, 56], layout=MemoryLayout.NCHW),
            name="output",
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.CONV2D,
            name="conv0",
            inputs=["input", "weight"],
            outputs=["output"],
            attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )

    return CompileContext(graph=graph, target="x86", optimization_level=3)


def make_resnet_stem_conv_context() -> CompileContext:
    graph = Graph("conv_resnet_stem_tiled")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 3, 224, 224], layout=MemoryLayout.NCHW),
            name="input",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([64, 3, 7, 7], layout=MemoryLayout.OIHW),
            name="weight",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 112, 112], layout=MemoryLayout.NCHW),
            name="output",
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.CONV2D,
            name="conv0",
            inputs=["input", "weight"],
            outputs=["output"],
            attrs={"kernel_shape": [7, 7], "pads": [3, 3, 3, 3], "strides": [2, 2]},
        )
    )

    return CompileContext(graph=graph, target="x86", optimization_level=3)


def seed_schedule_and_layout(ctx: CompileContext, node_name: str, op_family: str) -> None:
    ctx.metadata["schedule_candidates"] = {
        node_name: ScheduleCandidate(
            node_name=node_name,
            op_family=op_family,
            tensor_footprint_bytes=1,
            must_tile=op_family == "conv2d",
            reason="peak_working_set" if op_family == "conv2d" else "fits_working_set",
        )
    }
    ctx.metadata["layout_plans"] = {
        node_name: LayoutPlan(
            node_name=node_name,
            op_family=op_family,
            input_layout=GenericBlockedLayout(
                kind=GenericBlockedLayoutKind.BLOCKED_ACTIVATION,
                blocked_axes=("C",),
            ),
            weight_layout=(
                GenericBlockedLayout(
                    kind=GenericBlockedLayoutKind.BLOCKED_WEIGHT,
                    blocked_axes=("K", "C"),
                )
                if op_family in {"conv2d", "gemm"}
                else None
            ),
        )
    }


def find_access(accesses: tuple[TensorAccessPlan, ...], tensor_name: str) -> TensorAccessPlan:
    return next(access for access in accesses if access.tensor_name == tensor_name)


def test_tiled_lowering_emits_conv_plan_with_halo_and_scratch():
    ctx = make_conv_context()
    seed_schedule_and_layout(ctx, node_name="conv0", op_family="conv2d")

    TiledLoweringPass().run(ctx)

    plan = ctx.metadata["node_execution_plans"]["conv0"]
    input_access = find_access(plan.input_accesses, "input")
    weight_access = find_access(plan.input_accesses, "weight")

    assert plan.op_family == "conv2d"
    assert plan.tile_axes == ("h", "w")
    assert plan.layout_class is LayoutClass.BLOCKED_ACTIVATION
    assert input_access.layout_class is LayoutClass.BLOCKED_ACTIVATION
    assert input_access.tile_region.halo_extents == (1, 1)
    assert MemoryRegionKind.SCRATCH in plan.memory_regions
    assert weight_access.layout_class is LayoutClass.BLOCKED_WEIGHT


def test_tiled_lowering_emits_pool_plan_without_weight_access():
    ctx = make_maxpool_context()
    seed_schedule_and_layout(ctx, node_name="pool0", op_family="maxpool")

    TiledLoweringPass().run(ctx)

    plan = ctx.metadata["node_execution_plans"]["pool0"]

    assert plan.op_family == "maxpool"
    assert plan.tile_axes == ("h", "w")
    assert len(plan.input_accesses) == 1
    assert plan.input_accesses[0].tensor_name == "input"
    assert plan.input_accesses[0].tile_region.halo_extents == (0, 0)
    assert all(access.layout_class is not LayoutClass.BLOCKED_WEIGHT for access in plan.input_accesses)


def test_tiled_lowering_computes_halo_for_valid_conv_without_padding():
    ctx = make_valid_conv_context()
    seed_schedule_and_layout(ctx, node_name="conv0", op_family="conv2d")

    TiledLoweringPass().run(ctx)

    plan = ctx.metadata["node_execution_plans"]["conv0"]
    input_access = find_access(plan.input_accesses, "input")

    assert input_access.tile_region.halo_extents == (1, 1)


def test_tiled_lowering_computes_halo_for_overlapping_pool():
    ctx = make_overlapping_maxpool_context()
    seed_schedule_and_layout(ctx, node_name="pool0", op_family="maxpool")

    TiledLoweringPass().run(ctx)

    plan = ctx.metadata["node_execution_plans"]["pool0"]

    assert plan.input_accesses[0].tile_region.halo_extents == (1, 1)


def test_tiled_lowering_requires_layout_plan_for_scheduled_node():
    ctx = make_conv_context()
    ctx.metadata["schedule_candidates"] = {
        "conv0": ScheduleCandidate(
            node_name="conv0",
            op_family="conv2d",
            tensor_footprint_bytes=1,
            must_tile=True,
            reason="peak_working_set",
        )
    }

    with pytest.raises(ValueError, match="layout plan.*conv0"):
        TiledLoweringPass().run(ctx)


def test_tiled_lowering_conv_plan_memory_regions_match_accesses_and_scratch():
    ctx = make_conv_context()
    seed_schedule_and_layout(ctx, node_name="conv0", op_family="conv2d")

    TiledLoweringPass().run(ctx)

    plan = ctx.metadata["node_execution_plans"]["conv0"]

    assert plan.memory_regions == (
        MemoryRegionKind.PERSISTENT,
        MemoryRegionKind.TILE,
        MemoryRegionKind.SCRATCH,
    )


def test_tiled_lowering_records_region_size_hints_for_conv_plan():
    ctx = make_conv_context()
    seed_schedule_and_layout(ctx, node_name="conv0", op_family="conv2d")

    TiledLoweringPass().run(ctx)

    region_sizes = ctx.metadata["node_execution_plan_region_sizes"]["conv0"]

    assert set(region_sizes["tensor_bytes"]) == {"input", "weight", "output"}
    assert region_sizes["tensor_bytes"]["input"] > 0
    assert region_sizes["tensor_bytes"]["weight"] > 0
    assert region_sizes["tensor_bytes"]["output"] > 0
    assert region_sizes["tensor_bytes"]["weight"] < ctx.graph.tensors["weight"].byte_size()
    assert "persistent" not in region_sizes["region_bytes"]
    assert region_sizes["region_bytes"]["scratch"] >= 0


def test_tiled_lowering_records_tile_sized_weight_hints_for_safe_gemm():
    ctx = make_safe_gemm_context()
    seed_schedule_and_layout(ctx, node_name="fc", op_family="gemm")

    TiledLoweringPass().run(ctx)

    plan = ctx.metadata["node_execution_plans"]["fc"]
    region_sizes = ctx.metadata["node_execution_plan_region_sizes"]["fc"]

    assert plan.tile_axes == ("m", "n")
    assert region_sizes["tensor_bytes"]["weight"] > 0
    assert region_sizes["tensor_bytes"]["weight"] < ctx.graph.tensors["weight"].byte_size()
    assert region_sizes["tensor_bytes"]["output"] > 0
    assert region_sizes["tensor_bytes"]["output"] < ctx.graph.tensors["output"].byte_size()


def test_tiled_lowering_uses_minimal_tile_when_conv_scratch_exceeds_budget():
    ctx = make_oversubscribed_conv_context()
    seed_schedule_and_layout(ctx, node_name="conv0", op_family="conv2d")

    TiledLoweringPass().run(ctx)

    plan = ctx.metadata["node_execution_plans"]["conv0"]
    region_sizes = ctx.metadata["node_execution_plan_region_sizes"]["conv0"]

    assert find_access(plan.output_accesses, "output").tile_region.logical_extents == (1, 1)
    assert find_access(plan.input_accesses, "input").tile_region.logical_extents == (3, 3)
    assert region_sizes["tensor_bytes"]["output"] < ctx.graph.tensors["output"].byte_size()


def test_tiled_lowering_skips_unresolved_tensor_hints_for_partial_dynamic_conv():
    ctx = make_partial_dynamic_conv_context()
    seed_schedule_and_layout(ctx, node_name="conv0", op_family="conv2d")

    TiledLoweringPass().run(ctx)

    plan = ctx.metadata["node_execution_plans"]["conv0"]
    region_sizes = ctx.metadata["node_execution_plan_region_sizes"]["conv0"]

    assert find_access(plan.output_accesses, "output").tile_region.logical_extents == (1, 1)
    assert region_sizes["tensor_bytes"]["output"] > 0
    assert "input" not in region_sizes["tensor_bytes"]
    assert region_sizes["region_bytes"]["scratch"] >= 0


def test_tiled_lowering_accounts_for_persistent_weights_in_stem_working_set():
    ctx = make_resnet_stem_conv_context()
    seed_schedule_and_layout(ctx, node_name="conv0", op_family="conv2d")

    TiledLoweringPass().run(ctx)

    region_sizes = ctx.metadata["node_execution_plan_region_sizes"]["conv0"]
    weight_bytes = ctx.graph.tensors["weight"].byte_size()
    total_bytes = (
        region_sizes["tensor_bytes"]["input"]
        + region_sizes["tensor_bytes"]["output"]
        + region_sizes["region_bytes"]["scratch"]
        + weight_bytes
    )

    assert total_bytes <= FAST_MEMORY_BUDGET_BYTES
