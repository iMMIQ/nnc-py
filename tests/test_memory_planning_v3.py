"""Tests for tile-aware memory planning in MemoryPlanningPassV3."""

from pathlib import Path

import numpy as np
import pytest

from nnc_py.frontend.onnx_loader import ONNXFrontend
from nnc_py.ir.context import CompileContext
from nnc_py.ir.execution_plan import (
    LayoutClass,
    MemoryRegionKind,
    NodeExecutionPlan,
    TensorAccessPlan,
)
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType, GenericBlockedLayoutKind, MemoryLayout
from nnc_py.passes.liveness import LivenessAnalysisPass
from nnc_py.passes.layout_planning import GenericBlockedLayout, LayoutPlan
from nnc_py.passes.memory_planning import MemoryPlanningPassV3, allocate_tile_regions
from nnc_py.passes.base import PassManager
from nnc_py.passes.prepack_lowering import PrepackLoweringPass
from nnc_py.passes.schedule_analysis import ScheduleCandidate
from nnc_py.passes.tiled_lowering import TiledLoweringPass


def assert_tile_regions_v3_pool_upper_bound(plan) -> None:
    if plan.strategy_name != "tile_regions_v3":
        return
    buffer_end = max((buffer.offset + buffer.size for buffer in plan.buffers), default=0)
    assert plan.total_fast_memory >= buffer_end


def make_tiled_conv_context() -> CompileContext:
    graph = Graph("conv_tiled_v3")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 112, 112], layout=MemoryLayout.NCHW),
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
            attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )

    return CompileContext(graph=graph, target="x86", optimization_level=3)


def make_mixed_conv_relu_context() -> CompileContext:
    graph = Graph("conv_relu_mixed_v3")
    graph.inputs = ["input"]
    graph.outputs = ["relu_out"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 112, 112], layout=MemoryLayout.NCHW),
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
            shape=TensorShape([1, 64, 112, 112], layout=MemoryLayout.NCHW),
            name="conv_out",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 112, 112], layout=MemoryLayout.NCHW),
            name="relu_out",
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.CONV2D,
            name="conv0",
            inputs=["input", "weight"],
            outputs=["conv_out"],
            attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.RELU,
            name="relu0",
            inputs=["conv_out"],
            outputs=["relu_out"],
        )
    )

    return CompileContext(graph=graph, target="x86", optimization_level=3)


def make_safe_gemm_context() -> CompileContext:
    graph = Graph("gemm_tiled_v3")
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
    graph.constants["weight"] = np.arange(512 * 1000, dtype=np.float32).reshape(512, 1000)
    graph.constants["bias"] = np.zeros(1000, dtype=np.float32)
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


def make_two_stage_tiled_context() -> CompileContext:
    graph = Graph("two_stage_tiled_v3")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 112, 112], layout=MemoryLayout.NCHW),
            name="input",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([64, 64, 3, 3], layout=MemoryLayout.OIHW),
            name="weight0",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 112, 112], layout=MemoryLayout.NCHW),
            name="mid",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([64, 64, 3, 3], layout=MemoryLayout.OIHW),
            name="weight1",
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
            outputs=["output"],
            attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )

    return CompileContext(graph=graph, target="x86", optimization_level=3)


def _compile_resnet18_ctx() -> CompileContext | None:
    model_path = Path(__file__).resolve().parent.parent / "models" / "resnet18.onnx"
    if not model_path.exists():
        return None

    graph = ONNXFrontend().load(str(model_path))
    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    pass_manager = PassManager()
    for pass_obj in PassManager.get_default_passes(3):
        pass_manager.register(pass_obj)
    pass_manager.run(ctx)
    return ctx


def attach_phase1_execution_plan(
    ctx: CompileContext,
    *,
    input_tile_bytes: int,
    output_tile_bytes: int,
    scratch_bytes: int = 0,
) -> None:
    ctx.metadata["node_execution_plans"] = {
        "conv0": NodeExecutionPlan(
            node_name="conv0",
            op_family="conv2d",
            tile_axes=("h", "w"),
            layout_class=LayoutClass.BLOCKED_ACTIVATION,
            memory_regions=(
                MemoryRegionKind.PERSISTENT,
                MemoryRegionKind.TILE,
                MemoryRegionKind.SCRATCH,
            ),
            input_accesses=(
                TensorAccessPlan(
                    tensor_name="input",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
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
                    tensor_name="output",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    memory_region=MemoryRegionKind.TILE,
                ),
            ),
        )
    }
    ctx.metadata["node_execution_plan_region_sizes"] = {
        "conv0": {
            "tensor_bytes": {
                "input": input_tile_bytes,
                "output": output_tile_bytes,
            },
            "region_bytes": {
                "scratch": scratch_bytes,
            },
        }
    }


def attach_two_stage_execution_plan(
    ctx: CompileContext,
    *,
    conv0_input_tile_bytes: int,
    conv0_output_tile_bytes: int,
    conv0_scratch_bytes: int = 0,
    conv1_input_tile_bytes: int,
    conv1_output_tile_bytes: int,
    conv1_scratch_bytes: int = 0,
) -> None:
    ctx.metadata["node_execution_plans"] = {
        node_name: NodeExecutionPlan(
            node_name=node_name,
            op_family="conv2d",
            tile_axes=("h", "w"),
            layout_class=LayoutClass.BLOCKED_ACTIVATION,
            memory_regions=(
                MemoryRegionKind.PERSISTENT,
                MemoryRegionKind.TILE,
                MemoryRegionKind.SCRATCH,
            ),
            input_accesses=(
                TensorAccessPlan(
                    tensor_name=input_name,
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
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
                    memory_region=MemoryRegionKind.TILE,
                ),
            ),
        )
        for node_name, input_name, weight_name, output_name in (
            ("conv0", "input", "weight0", "mid"),
            ("conv1", "mid", "weight1", "output"),
        )
    }
    ctx.metadata["node_execution_plan_region_sizes"] = {
        "conv0": {
            "tensor_bytes": {
                "input": conv0_input_tile_bytes,
                "mid": conv0_output_tile_bytes,
            },
            "region_bytes": {
                "scratch": conv0_scratch_bytes,
            },
        },
        "conv1": {
            "tensor_bytes": {
                "mid": conv1_input_tile_bytes,
                "output": conv1_output_tile_bytes,
            },
            "region_bytes": {
                "scratch": conv1_scratch_bytes,
            },
        },
    }


def attach_three_region_conflict_execution_plan(ctx: CompileContext) -> None:
    ctx.metadata["node_execution_plans"] = {
        "conv0": NodeExecutionPlan(
            node_name="conv0",
            op_family="conv2d",
            tile_axes=("h", "w"),
            layout_class=LayoutClass.BLOCKED_ACTIVATION,
            memory_regions=(MemoryRegionKind.TILE, MemoryRegionKind.SCRATCH),
            input_accesses=(
                TensorAccessPlan(
                    tensor_name="input",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    memory_region=MemoryRegionKind.TILE,
                ),
            ),
            output_accesses=(),
        ),
        "pack0": NodeExecutionPlan(
            node_name="pack0",
            op_family="gemm",
            layout_class=LayoutClass.BLOCKED_ACTIVATION,
            memory_regions=(MemoryRegionKind.TILE, MemoryRegionKind.PACK),
            input_accesses=(
                TensorAccessPlan(
                    tensor_name="mid",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    memory_region=MemoryRegionKind.TILE,
                ),
            ),
            output_accesses=(),
        ),
        "pack1": NodeExecutionPlan(
            node_name="pack1",
            op_family="gemm",
            layout_class=LayoutClass.BLOCKED_ACTIVATION,
            memory_regions=(MemoryRegionKind.SCRATCH, MemoryRegionKind.PACK),
            input_accesses=(),
            output_accesses=(),
        ),
    }
    ctx.metadata["node_execution_plan_region_sizes"] = {
        "conv0": {
            "tensor_bytes": {
                "input": 128,
            },
            "region_bytes": {
                "scratch": 96,
            },
        },
        "pack0": {
            "tensor_bytes": {
                "mid": 128,
            },
            "region_bytes": {
                "pack": 80,
            },
        },
        "pack1": {
            "tensor_bytes": {},
            "region_bytes": {
                "scratch": 96,
                "pack": 80,
            },
        },
    }


def attach_scratch_only_execution_plan(ctx: CompileContext, *, scratch_bytes: int) -> None:
    ctx.metadata["node_execution_plans"] = {
        "conv0": NodeExecutionPlan(
            node_name="conv0",
            op_family="conv2d",
            tile_axes=("h", "w"),
            layout_class=LayoutClass.BLOCKED_ACTIVATION,
            memory_regions=(MemoryRegionKind.SCRATCH,),
            input_accesses=(),
            output_accesses=(),
        )
    }
    ctx.metadata["node_execution_plan_region_sizes"] = {
        "conv0": {
            "tensor_bytes": {},
            "region_bytes": {
                "scratch": scratch_bytes,
            },
        }
    }


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
                if op_family == "conv2d"
                else None
            ),
        )
    }


def test_memory_planning_v3_uses_tile_regions_not_full_tensor_peak():
    ctx = make_tiled_conv_context()
    attach_phase1_execution_plan(
        ctx,
        input_tile_bytes=262144,
        output_tile_bytes=131072,
    )

    MemoryPlanningPassV3().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.total_fast_memory <= 1024 * 1024
    assert plan.logical_regions["tile"].size_bytes >= 262144
    tile_buffer = next(buffer for buffer in plan.buffers if buffer.tensors == ["tile"])
    assert plan.logical_regions["tile"].offset == tile_buffer.offset


def test_memory_planning_v3_falls_back_when_tile_region_sizes_are_missing():
    ctx = make_tiled_conv_context()
    attach_phase1_execution_plan(
        ctx,
        input_tile_bytes=262144,
        output_tile_bytes=131072,
    )
    del ctx.metadata["node_execution_plan_region_sizes"]
    LivenessAnalysisPass().run(ctx)

    MemoryPlanningPassV3().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.strategy_name == "cost_aware"
    assert plan.logical_regions == {}


def test_memory_planning_v3_uses_real_region_hints_from_tiled_lowering():
    ctx = make_tiled_conv_context()
    seed_schedule_and_layout(ctx, node_name="conv0", op_family="conv2d")

    TiledLoweringPass().run(ctx)
    LivenessAnalysisPass().run(ctx)
    MemoryPlanningPassV3().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.strategy_name == "tile_regions_v3"
    assert "persistent" not in plan.logical_regions
    assert all(buffer.tensors != ["persistent"] for buffer in plan.buffers)


def test_memory_planning_v3_uses_per_node_peak_instead_of_summing_region_peaks():
    ctx = make_two_stage_tiled_context()
    attach_two_stage_execution_plan(
        ctx,
        conv0_input_tile_bytes=262144,
        conv0_output_tile_bytes=777472,
        conv1_input_tile_bytes=262144,
        conv1_output_tile_bytes=262144,
        conv1_scratch_bytes=294912,
    )

    MemoryPlanningPassV3().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.strategy_name == "tile_regions_v3"
    assert plan.logical_regions["tile"].size_bytes == 1039616
    assert plan.logical_regions["scratch"].size_bytes == 294912
    assert plan.total_fast_memory == 1039616
    assert plan.peak_memory == 1039616
    assert_tile_regions_v3_pool_upper_bound(plan)


def test_memory_planning_v3_three_region_conflicts_keep_pool_upper_bound():
    ctx = make_two_stage_tiled_context()
    attach_three_region_conflict_execution_plan(ctx)

    plan = allocate_tile_regions(ctx)

    assert plan.strategy_name == "tile_regions_v3"
    assert set(plan.logical_regions) == {"tile", "scratch", "pack"}
    assert_tile_regions_v3_pool_upper_bound(plan)


def test_memory_planning_v3_keeps_scratch_only_regions_as_metadata_without_buffers():
    ctx = make_tiled_conv_context()
    attach_scratch_only_execution_plan(ctx, scratch_bytes=256)

    plan = allocate_tile_regions(ctx)

    assert plan.strategy_name == "tile_regions_v3"
    assert set(plan.logical_regions) == {"scratch"}
    assert plan.buffers == []
    assert plan.total_fast_memory == 0
    assert plan.peak_memory == 256
    assert_tile_regions_v3_pool_upper_bound(plan)


def test_memory_planning_v3_uses_tile_regions_for_safe_gemm_lowering():
    ctx = make_safe_gemm_context()
    seed_schedule_and_layout(ctx, node_name="fc", op_family="gemm")

    PrepackLoweringPass().run(ctx)
    TiledLoweringPass().run(ctx)
    LivenessAnalysisPass().run(ctx)
    MemoryPlanningPassV3().run(ctx)

    execution_plan = ctx.metadata["node_execution_plans"]["fc"]
    plan = ctx.metadata["memory_allocation_plan"]

    assert execution_plan.op_family == "gemm"
    assert plan.strategy_name == "tile_regions_v3"
    assert_tile_regions_v3_pool_upper_bound(plan)


def test_memory_planning_v3_uses_tile_regions_for_resnet18_without_max_memory():
    ctx = _compile_resnet18_ctx()
    if ctx is None:
        pytest.skip("resnet18.onnx is not available")

    plan = ctx.metadata["memory_allocation_plan"]

    assert "/fc/Gemm" in ctx.metadata.get("node_execution_plans", {})
    assert plan.strategy_name == "tile_regions_v3"
    assert_tile_regions_v3_pool_upper_bound(plan)


def test_memory_planning_v3_falls_back_when_tile_region_hints_are_zero_sized():
    ctx = make_tiled_conv_context()
    attach_phase1_execution_plan(
        ctx,
        input_tile_bytes=0,
        output_tile_bytes=0,
    )
    LivenessAnalysisPass().run(ctx)

    MemoryPlanningPassV3().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.strategy_name == "cost_aware"
    assert plan.logical_regions == {}


def test_memory_planning_v3_falls_back_for_dynamic_spatial_lowering_hints():
    from tests.test_tiled_lowering_pass import make_dynamic_spatial_conv_context

    ctx = make_dynamic_spatial_conv_context()
    seed_schedule_and_layout(ctx, node_name="conv0", op_family="conv2d")

    TiledLoweringPass().run(ctx)
    LivenessAnalysisPass().run(ctx)
    MemoryPlanningPassV3().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.strategy_name == "cost_aware"
    assert plan.logical_regions == {}


def test_memory_planning_v3_falls_back_for_partial_dynamic_channel_lowering_hints():
    from tests.test_tiled_lowering_pass import make_partial_dynamic_conv_context

    ctx = make_partial_dynamic_conv_context()
    seed_schedule_and_layout(ctx, node_name="conv0", op_family="conv2d")

    TiledLoweringPass().run(ctx)
    LivenessAnalysisPass().run(ctx)
    MemoryPlanningPassV3().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.strategy_name == "cost_aware"
    assert plan.logical_regions == {}


def test_memory_planning_v3_tile_aware_path_does_not_write_legacy_memory_plan():
    ctx = make_tiled_conv_context()
    attach_phase1_execution_plan(
        ctx,
        input_tile_bytes=262144,
        output_tile_bytes=131072,
    )

    MemoryPlanningPassV3().run(ctx)

    assert "memory_plan" not in ctx.metadata


def test_memory_planning_v3_uses_tile_regions_for_mixed_graph_execution_group_coverage():
    ctx = make_mixed_conv_relu_context()
    attach_phase1_execution_plan(
        ctx,
        input_tile_bytes=262144,
        output_tile_bytes=131072,
    )
    LivenessAnalysisPass().run(ctx)

    MemoryPlanningPassV3().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.strategy_name == "tile_regions_v3"
    assert_tile_regions_v3_pool_upper_bound(plan)


def test_memory_planning_v3_raises_when_budget_is_smaller_than_v3_peak():
    ctx = make_tiled_conv_context()
    attach_phase1_execution_plan(
        ctx,
        input_tile_bytes=262144,
        output_tile_bytes=131072,
    )
    ctx.metadata["max_memory"] = 128 * 1024
    LivenessAnalysisPass().run(ctx)

    with pytest.raises(ValueError, match="max_memory"):
        MemoryPlanningPassV3().run(ctx)


def test_memory_planning_v3_keeps_tile_regions_when_budget_fits_plan():
    ctx = make_tiled_conv_context()
    attach_phase1_execution_plan(
        ctx,
        input_tile_bytes=262144,
        output_tile_bytes=131072,
        scratch_bytes=65536,
    )
    ctx.metadata["max_memory"] = 512 * 1024
    LivenessAnalysisPass().run(ctx)

    MemoryPlanningPassV3().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.strategy_name == "tile_regions_v3"
    assert plan.total_fast_memory == 393216
    assert plan.peak_memory == 458752
    assert "memory_plan" not in ctx.metadata
    assert "max_memory" not in ctx.metadata
    assert ctx.metadata["memory_budget_satisfied_by_v3"] == 512 * 1024
    assert_tile_regions_v3_pool_upper_bound(plan)
