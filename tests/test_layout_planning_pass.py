from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType, MemoryLayout
from nnc_py.passes import LayoutPlanningPass, ScheduleCandidate


def make_conv_context() -> CompileContext:
    graph = Graph("conv_layout")
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
    graph = Graph("pool_layout")
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


def candidate_for_conv(node_name: str) -> ScheduleCandidate:
    return ScheduleCandidate(
        node_name=node_name,
        op_family="conv2d",
        tensor_footprint_bytes=1_769_472,
        must_tile=True,
        reason="peak_working_set",
    )


def candidate_for_maxpool(node_name: str) -> ScheduleCandidate:
    return ScheduleCandidate(
        node_name=node_name,
        op_family="maxpool",
        tensor_footprint_bytes=20_480,
        must_tile=False,
        reason="fits_working_set",
    )


def test_layout_planning_assigns_generic_blocked_layout_to_conv_activation_and_weight():
    ctx = make_conv_context()
    ctx.metadata["schedule_candidates"] = {"conv0": candidate_for_conv("conv0")}

    LayoutPlanningPass().run(ctx)

    plan = ctx.metadata["layout_plans"]["conv0"]
    assert plan.input_layout.name == "blocked_activation"
    assert plan.weight_layout.name == "blocked_weight"
    assert plan.target_physical_layout is None


def test_layout_planning_uses_activation_blocking_for_supported_non_weight_ops():
    ctx = make_maxpool_context()
    ctx.metadata["schedule_candidates"] = {"pool0": candidate_for_maxpool("pool0")}

    LayoutPlanningPass().run(ctx)

    plan = ctx.metadata["layout_plans"]["pool0"]
    assert plan.input_layout.name == "blocked_activation"
    assert plan.weight_layout is None
    assert plan.target_physical_layout is None
