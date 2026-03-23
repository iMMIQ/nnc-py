import numpy as np

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType, MemoryLayout
from nnc_py.passes.prepack_lowering import PrepackLoweringPass


def test_prepack_lowering_pass_pretransposes_constant_gemm_weight():
    graph = Graph("gemm_prepack")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([2, 3], layout=MemoryLayout.NCHW),
            name="input",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([3, 4], layout=MemoryLayout.NCHW),
            name="weight",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([2, 4], layout=MemoryLayout.NCHW),
            name="output",
        )
    )
    graph.constants["weight"] = np.arange(12, dtype=np.float32).reshape(3, 4)

    graph.add_node(
        Node(
            op_type=OpType.GEMM,
            name="fc",
            inputs=["input", "weight"],
            outputs=["output"],
            attrs={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)

    PrepackLoweringPass().run(ctx)

    node = graph.get_node("fc")
    np.testing.assert_array_equal(
        graph.constants["weight"],
        np.arange(12, dtype=np.float32).reshape(3, 4).T,
    )
    assert node.attrs["transB"] == 1
    assert graph.get_tensor("weight").shape.dims == [4, 3]
    assert node.metadata["lowering"]["kernel_family"] == "gemm"
    assert node.metadata["lowering"]["weight_pack"] == "rhs_transposed_constant"


def test_prepack_lowering_pass_attaches_conv_kernel_hint():
    graph = Graph("conv_lowering")
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
            shape=TensorShape([32, 16, 3, 3], layout=MemoryLayout.OIHW),
            name="weight",
        )
    )
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape([1, 32, 8, 8], layout=MemoryLayout.NCHW),
            name="output",
        )
    )
    graph.constants["weight"] = np.ones((32, 16, 3, 3), dtype=np.float32)
    graph.add_node(
        Node(
            op_type=OpType.FUSED_CONV_RELU,
            name="conv",
            inputs=["input", "weight"],
            outputs=["output"],
            attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]},
        )
    )

    ctx = CompileContext(graph=graph, target="x86", optimization_level=3)

    PrepackLoweringPass().run(ctx)

    node = graph.get_node("conv")
    assert node.metadata["lowering"]["kernel_family"] == "conv2d"
    assert node.metadata["lowering"]["kernel_kind"] == "spatial_3x3"
    assert node.metadata["lowering"]["weight_pack"] == "oihw_constant"
