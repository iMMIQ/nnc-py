import importlib

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
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    ScheduleDependencyKind,
    ScheduleStepKind,
    ScheduledValueHomeTier,
    get_pipeline_schedule_problem,
)
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType, MemoryLayout


def _get_pass_class():
    passes_module = importlib.import_module("nnc_py.passes")
    pass_class = getattr(passes_module, "PipelineStepLoweringPass", None)
    assert pass_class is not None
    return pass_class


def _run_pipeline_step_lowering(ctx: CompileContext) -> None:
    _get_pass_class()().run(ctx)


def _step_ids(problem, node_name: str) -> list[str]:
    return [step.id for step in problem.steps if step.node_name == node_name]


def _step_kinds(problem, node_name: str) -> list[ScheduleStepKind]:
    return [step.step_kind for step in problem.steps if step.node_name == node_name]


def _staged_value_name(node_name: str, tensor_name: str) -> str:
    return (
        f"sram|node|{len(node_name)}:{node_name}"
        f"|tensor|{len(tensor_name)}:{tensor_name}"
    )


def _shape_value_name(node_name: str) -> str:
    return f"sram|node|{len(node_name)}:{node_name}|shape"


def _make_conv_relu_context(*, include_relu_plan: bool = True) -> CompileContext:
    graph = Graph("conv_relu_pipeline")
    graph.inputs = ["input"]
    graph.outputs = ["relu_out"]

    graph.add_tensor(
        TensorType(
            name="input",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 32, 32], layout=MemoryLayout.NCHW),
        )
    )
    graph.add_tensor(
        TensorType(
            name="weight",
            dtype=DataType.FLOAT32,
            shape=TensorShape([64, 64, 3, 3], layout=MemoryLayout.OIHW),
        )
    )
    graph.add_tensor(
        TensorType(
            name="conv_out",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 32, 32], layout=MemoryLayout.NCHW),
        )
    )
    graph.add_tensor(
        TensorType(
            name="relu_out",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 32, 32], layout=MemoryLayout.NCHW),
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

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    set_node_execution_plan(
        ctx,
        NodeExecutionPlan(
            node_name="conv0",
            op_family="conv2d",
            tile_axes=("h", "w"),
            layout_class=LayoutClass.BLOCKED_ACTIVATION,
            target_physical_layout="nchwc16",
            memory_regions=(
                MemoryRegionKind.TILE,
                MemoryRegionKind.SCRATCH,
            ),
            input_accesses=(
                TensorAccessPlan(
                    tensor_name="input",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    tile_region=TileRegion(
                        logical_extents=(1, 64, 16, 16),
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
                    tensor_name="conv_out",
                    layout_class=LayoutClass.BLOCKED_ACTIVATION,
                    tile_region=TileRegion(logical_extents=(1, 64, 16, 16)),
                    memory_region=MemoryRegionKind.TILE,
                ),
            ),
        ),
    )
    if include_relu_plan:
        set_node_execution_plan(
            ctx,
            NodeExecutionPlan(
                node_name="relu0",
                op_family="relu",
                input_accesses=(
                    TensorAccessPlan(
                        tensor_name="conv_out",
                        layout_class=LayoutClass.PLAIN,
                        memory_region=MemoryRegionKind.PERSISTENT,
                    ),
                ),
                output_accesses=(
                    TensorAccessPlan(
                        tensor_name="relu_out",
                        layout_class=LayoutClass.PLAIN,
                        memory_region=MemoryRegionKind.PERSISTENT,
                    ),
                ),
            ),
        )
    return ctx


def _make_shape_context() -> CompileContext:
    graph = Graph("reshape_pipeline")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            name="input",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 16, 8, 8], layout=MemoryLayout.NCHW),
        )
    )
    graph.add_tensor(
        TensorType(
            name="output",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 1024]),
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.RESHAPE,
            name="reshape0",
            inputs=["input"],
            outputs=["output"],
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    set_node_execution_plan(
        ctx,
        NodeExecutionPlan(
            node_name="reshape0",
            op_family="reshape",
            input_accesses=(
                TensorAccessPlan(
                    tensor_name="input",
                    memory_region=MemoryRegionKind.PERSISTENT,
                ),
            ),
            output_accesses=(
                TensorAccessPlan(
                    tensor_name="output",
                    memory_region=MemoryRegionKind.PERSISTENT,
                ),
            ),
        ),
    )
    return ctx


def _make_conv_with_constant_weight_context() -> CompileContext:
    ctx = _make_conv_relu_context(include_relu_plan=False)
    ctx.graph.constants["weight"] = [1.0]
    return ctx


def _make_two_conv_shared_weight_context() -> CompileContext:
    graph = Graph("two_conv_shared_weight")
    graph.inputs = ["input"]
    graph.outputs = ["out0", "out1"]

    graph.add_tensor(
        TensorType(
            name="input",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 32, 32], layout=MemoryLayout.NCHW),
        )
    )
    graph.add_tensor(
        TensorType(
            name="weight",
            dtype=DataType.FLOAT32,
            shape=TensorShape([64, 64, 3, 3], layout=MemoryLayout.OIHW),
        )
    )
    graph.add_tensor(
        TensorType(
            name="out0",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 32, 32], layout=MemoryLayout.NCHW),
        )
    )
    graph.add_tensor(
        TensorType(
            name="out1",
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 64, 32, 32], layout=MemoryLayout.NCHW),
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.CONV2D,
            name="conv0",
            inputs=["input", "weight"],
            outputs=["out0"],
            attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )
    graph.add_node(
        Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "weight"],
            outputs=["out1"],
            attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)
    for node_name, output_name in (("conv0", "out0"), ("conv1", "out1")):
        set_node_execution_plan(
            ctx,
            NodeExecutionPlan(
                node_name=node_name,
                op_family="conv2d",
                tile_axes=("h", "w"),
                layout_class=LayoutClass.BLOCKED_ACTIVATION,
                target_physical_layout="nchwc16",
                memory_regions=(MemoryRegionKind.TILE, MemoryRegionKind.SCRATCH),
                input_accesses=(
                    TensorAccessPlan(
                        tensor_name="input",
                        layout_class=LayoutClass.BLOCKED_ACTIVATION,
                        tile_region=TileRegion(
                            logical_extents=(1, 64, 16, 16),
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
                        tensor_name=output_name,
                        layout_class=LayoutClass.BLOCKED_ACTIVATION,
                        tile_region=TileRegion(logical_extents=(1, 64, 16, 16)),
                        memory_region=MemoryRegionKind.TILE,
                    ),
                ),
            ),
        )
    return ctx


def test_tiled_conv_lowers_to_mixed_granularity_schedule_problem():
    ctx = _make_conv_relu_context()

    _run_pipeline_step_lowering(ctx)

    problem = get_pipeline_schedule_problem(ctx)
    assert problem is not None
    assert problem is ctx.pipeline_schedule_problem

    conv_steps = [step for step in problem.steps if step.node_name == "conv0"]
    assert [step.step_kind for step in conv_steps] == [
        ScheduleStepKind.DMA_IN,
        ScheduleStepKind.SHAPE_PREP,
        ScheduleStepKind.COMPUTE,
        ScheduleStepKind.DMA_OUT,
    ]
    assert [step.resource_kind for step in conv_steps] == [
        PipelineResourceKind.DMA,
        PipelineResourceKind.SHAPE,
        PipelineResourceKind.MATMUL,
        PipelineResourceKind.DMA,
    ]
    assert all(step.duration > 0 for step in conv_steps)
    assert all(step.launch_overhead > 0 for step in conv_steps)
    assert conv_steps[0].sram_output_names
    assert conv_steps[2].sram_temp_bytes > 0

    scheduled_values = {value.name: value for value in problem.scheduled_values}
    conv_input_name = _staged_value_name("conv0", "input")
    conv_weight_name = _staged_value_name("conv0", "weight")
    conv_shape_name = _shape_value_name("conv0")
    conv_output_name = _staged_value_name("conv0", "conv_out")
    assert scheduled_values[conv_input_name].size_bytes == 1 * 64 * 18 * 18 * 4
    assert scheduled_values[conv_input_name].producer_step_id == conv_steps[0].id
    assert set(scheduled_values[conv_input_name].consumer_step_ids) == {
        conv_steps[1].id,
        conv_steps[2].id,
    }
    assert scheduled_values[conv_input_name].home_tier is ScheduledValueHomeTier.SRAM
    assert scheduled_values[conv_input_name].graph_tensor_name == "input"
    assert scheduled_values[conv_weight_name].size_bytes == 64 * 64 * 3 * 3 * 4
    assert scheduled_values[conv_weight_name].producer_step_id == conv_steps[0].id
    assert set(scheduled_values[conv_weight_name].consumer_step_ids) == {
        conv_steps[1].id,
        conv_steps[2].id,
    }
    assert scheduled_values[conv_weight_name].home_tier is ScheduledValueHomeTier.SRAM
    assert scheduled_values[conv_weight_name].graph_tensor_name == "weight"
    assert scheduled_values[conv_shape_name].size_bytes == 9 * 16
    assert scheduled_values[conv_shape_name].producer_step_id == conv_steps[1].id
    assert scheduled_values[conv_shape_name].consumer_step_ids == (conv_steps[2].id,)
    assert scheduled_values[conv_shape_name].home_tier is ScheduledValueHomeTier.SRAM
    assert scheduled_values[conv_shape_name].graph_tensor_name is None
    assert scheduled_values[conv_output_name].size_bytes == 1 * 64 * 16 * 16 * 4
    assert scheduled_values[conv_output_name].producer_step_id == conv_steps[2].id
    assert scheduled_values[conv_output_name].consumer_step_ids == (conv_steps[3].id,)
    assert scheduled_values[conv_output_name].home_tier is ScheduledValueHomeTier.SRAM
    assert scheduled_values[conv_output_name].graph_tensor_name == "conv_out"
    assert scheduled_values["conv_out"].size_bytes == 1 * 64 * 32 * 32 * 4
    assert scheduled_values["conv_out"].producer_step_id is None
    assert scheduled_values["conv_out"].consumer_step_ids == ("relu0.compute",)
    assert scheduled_values["conv_out"].home_tier is ScheduledValueHomeTier.SLOW
    assert scheduled_values["conv_out"].graph_tensor_name == "conv_out"

    assert problem.resources == (
        PipelineResourceKind.DMA,
        PipelineResourceKind.SHAPE,
        PipelineResourceKind.MATMUL,
        PipelineResourceKind.OTHER,
    )

    edge_kinds = {
        (edge.src_step_id, edge.dst_step_id): edge.kind for edge in problem.edges
    }
    assert edge_kinds[(conv_steps[0].id, conv_steps[1].id)] is ScheduleDependencyKind.SAME_NODE_SEQUENCE
    assert edge_kinds[(conv_steps[1].id, conv_steps[2].id)] is ScheduleDependencyKind.SAME_NODE_SEQUENCE
    assert edge_kinds[(conv_steps[2].id, conv_steps[3].id)] is ScheduleDependencyKind.SAME_NODE_SEQUENCE


def test_no_plan_relu_still_lowers_to_single_compute_step():
    ctx = _make_conv_relu_context(include_relu_plan=False)

    _run_pipeline_step_lowering(ctx)

    problem = ctx.get_pipeline_schedule_problem()
    assert problem is not None

    relu_steps = [step for step in problem.steps if step.node_name == "relu0"]
    assert len(relu_steps) == 1
    assert relu_steps[0].step_kind is ScheduleStepKind.COMPUTE
    assert relu_steps[0].resource_kind is PipelineResourceKind.OTHER
    assert relu_steps[0].duration > 0
    assert relu_steps[0].launch_overhead > 0
    assert relu_steps[0].sram_input_names == ("conv_out",)
    assert relu_steps[0].sram_output_names == (
        _staged_value_name("relu0", "relu_out"),
    )

    conv_last_step_id = _step_ids(problem, "conv0")[-1]
    assert any(
        edge.src_step_id == conv_last_step_id
        and edge.dst_step_id == relu_steps[0].id
        and edge.kind is ScheduleDependencyKind.DATA
        for edge in problem.edges
    )

    scheduled_values = {value.name: value for value in problem.scheduled_values}
    relu_output_name = _staged_value_name("relu0", "relu_out")
    assert scheduled_values[relu_output_name].size_bytes == 1 * 64 * 32 * 32 * 4
    assert scheduled_values[relu_output_name].producer_step_id == relu_steps[0].id
    assert scheduled_values[relu_output_name].consumer_step_ids == ()
    assert scheduled_values[relu_output_name].home_tier is ScheduledValueHomeTier.SRAM
    conv_output_name = _staged_value_name("conv0", "conv_out")
    assert scheduled_values[conv_output_name].consumer_step_ids == ("conv0.dma_out",)


def test_shape_family_operator_uses_shape_pipeline_step():
    ctx = _make_shape_context()

    _run_pipeline_step_lowering(ctx)

    problem = ctx.get_pipeline_schedule_problem()
    assert problem is not None

    assert _step_kinds(problem, "reshape0") == [ScheduleStepKind.SHAPE_PREP]
    shape_step = next(step for step in problem.steps if step.node_name == "reshape0")
    assert shape_step.resource_kind is PipelineResourceKind.SHAPE
    assert shape_step.duration > 0
    assert shape_step.launch_overhead > 0


def test_pipeline_step_lowering_marks_external_values_with_home_tier():
    ctx = _make_conv_relu_context()

    _run_pipeline_step_lowering(ctx)

    problem = ctx.pipeline_schedule_problem
    input_value = next(value for value in problem.scheduled_values if value.name == "input")

    assert input_value.home_tier is ScheduledValueHomeTier.INPUT


def test_pipeline_step_lowering_preserves_external_sizes_in_scheduled_values_but_zero_sizes_legacy_externals():
    ctx = _make_conv_relu_context()

    _run_pipeline_step_lowering(ctx)

    problem = ctx.pipeline_schedule_problem
    scheduled_values = {value.name: value for value in problem.scheduled_values}
    sram_values = {value.name: value for value in problem.sram_values}

    assert scheduled_values["input"].size_bytes == 1 * 64 * 16 * 16 * 4
    assert scheduled_values["input"].home_tier is ScheduledValueHomeTier.INPUT
    assert scheduled_values["weight"].size_bytes == 64 * 64 * 3 * 3 * 4
    assert scheduled_values["weight"].home_tier is ScheduledValueHomeTier.SLOW
    assert sram_values["input"].size_bytes == 0
    assert sram_values["weight"].size_bytes == 0


def test_pipeline_step_lowering_keeps_staged_outputs_as_sram_values():
    ctx = _make_conv_relu_context()

    _run_pipeline_step_lowering(ctx)

    problem = ctx.pipeline_schedule_problem
    staged = next(
        value
        for value in problem.scheduled_values
        if value.name == _staged_value_name("conv0", "conv_out")
    )

    assert staged.home_tier is ScheduledValueHomeTier.SRAM
    assert staged.graph_tensor_name == "conv_out"


def test_pipeline_step_lowering_marks_constant_externals_with_const_home_tier():
    ctx = _make_conv_with_constant_weight_context()

    _run_pipeline_step_lowering(ctx)

    problem = ctx.pipeline_schedule_problem
    scheduled_values = {value.name: value for value in problem.scheduled_values}
    sram_values = {value.name: value for value in problem.sram_values}

    assert scheduled_values["weight"].home_tier is ScheduledValueHomeTier.CONST
    assert scheduled_values["weight"].size_bytes == 64 * 64 * 3 * 3 * 4
    assert sram_values["weight"].size_bytes == 0


def test_shared_graph_weight_stages_to_distinct_node_local_sram_values():
    ctx = _make_two_conv_shared_weight_context()

    _run_pipeline_step_lowering(ctx)

    problem = ctx.get_pipeline_schedule_problem()
    assert problem is not None

    scheduled_values = {value.name: value for value in problem.scheduled_values}
    conv0_weight_name = _staged_value_name("conv0", "weight")
    conv1_weight_name = _staged_value_name("conv1", "weight")
    assert conv0_weight_name in scheduled_values
    assert conv1_weight_name in scheduled_values
    assert scheduled_values[conv0_weight_name].name != scheduled_values[conv1_weight_name].name
    assert scheduled_values[conv0_weight_name].producer_step_id == "conv0.dma_in"
    assert scheduled_values[conv1_weight_name].producer_step_id == "conv1.dma_in"
    assert set(scheduled_values[conv0_weight_name].consumer_step_ids) == {
        "conv0.shape_prep",
        "conv0.compute",
    }
    assert set(scheduled_values[conv1_weight_name].consumer_step_ids) == {
        "conv1.shape_prep",
        "conv1.compute",
    }
    assert all(
        "conv1" not in consumer for consumer in scheduled_values[conv0_weight_name].consumer_step_ids
    )
    assert all(
        "conv0" not in consumer for consumer in scheduled_values[conv1_weight_name].consumer_step_ids
    )


def test_lowering_tracks_external_input_values_for_scheduler_validation():
    ctx = _make_conv_relu_context()

    _run_pipeline_step_lowering(ctx)

    problem = ctx.get_pipeline_schedule_problem()
    assert problem is not None

    scheduled_values = {value.name: value for value in problem.scheduled_values}
    assert "input" in scheduled_values
    assert "weight" in scheduled_values
    assert scheduled_values["input"].producer_step_id is None
    assert scheduled_values["weight"].producer_step_id is None
    assert scheduled_values["input"].home_tier is ScheduledValueHomeTier.INPUT
    assert scheduled_values["weight"].home_tier is ScheduledValueHomeTier.SLOW
    assert "conv0.dma_in" in scheduled_values["input"].consumer_step_ids
    assert "conv0.dma_in" in scheduled_values["weight"].consumer_step_ids
