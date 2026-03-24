import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType, MemoryLayout
from nnc_py.passes import ScheduleAnalysisPass


def make_resnet_like_conv_context() -> CompileContext:
    graph = Graph("resnet_like_conv")
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


def test_schedule_analysis_marks_large_conv_as_tiled_candidate():
    ctx = make_resnet_like_conv_context()

    ScheduleAnalysisPass().run(ctx)

    schedule = ctx.metadata["schedule_candidates"]
    assert schedule["conv0"].must_tile is True
    assert schedule["conv0"].reason == "peak_working_set"


def test_schedule_analysis_leaves_small_supported_op_as_whole_tensor():
    graph = Graph("small_maxpool")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 8, 8, 8], layout=MemoryLayout.NCHW),
            name="input",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 8, 4, 4], layout=MemoryLayout.NCHW),
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

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)

    ScheduleAnalysisPass().run(ctx)

    schedule = ctx.metadata["schedule_candidates"]
    assert schedule["pool0"].must_tile is False
    assert schedule["pool0"].reason == "fits_working_set"


def test_schedule_analysis_skips_unsupported_operator_families():
    graph = Graph("unsupported_relu")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 16, 8, 8], layout=MemoryLayout.NCHW),
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
            op_type=OpType.RELU,
            name="relu0",
            inputs=["input"],
            outputs=["output"],
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)

    ScheduleAnalysisPass().run(ctx)

    schedule = ctx.metadata["schedule_candidates"]
    assert "relu0" not in schedule
