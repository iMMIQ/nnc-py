from nnc_py.codegen.c_emitter import CEmitter
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType


def _add_tensor(graph: Graph, name: str, shape: list[int]) -> None:
    graph.add_tensor(
        TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=shape),
            name=name,
        )
    )


def test_emit_conv_call_uses_specialized_entrypoint_for_3x3_stride1() -> None:
    graph = Graph("conv_specialized")
    _add_tensor(graph, "input", [1, 16, 8, 8])
    _add_tensor(graph, "weight", [32, 16, 3, 3])
    _add_tensor(graph, "output", [1, 32, 8, 8])

    node = Node(
        op_type=OpType.CONV2D,
        name="conv",
        inputs=["input", "weight"],
        outputs=["output"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]},
        metadata={
            "lowering": {
                "kernel_family": "conv2d",
                "kernel_kind": "spatial_3x3",
                "weight_pack": "oihw_constant",
            }
        },
    )
    graph.add_node(node)

    emitter = CEmitter()
    code = emitter.emit(CompileContext(graph=graph, target="x86", optimization_level=3))

    assert "nnc_conv3x3_s1(&input, &weight, NULL, &output);" in code


def test_emit_fused_conv_relu_uses_specialized_entrypoint_for_3x3_stride1() -> None:
    graph = Graph("conv_relu_specialized")
    _add_tensor(graph, "input", [1, 16, 8, 8])
    _add_tensor(graph, "weight", [32, 16, 3, 3])
    _add_tensor(graph, "output", [1, 32, 8, 8])

    node = Node(
        op_type=OpType.FUSED_CONV_RELU,
        name="conv_relu",
        inputs=["input", "weight"],
        outputs=["output"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]},
        metadata={
            "lowering": {
                "kernel_family": "conv2d",
                "kernel_kind": "spatial_3x3",
                "weight_pack": "oihw_constant",
            }
        },
    )
    graph.add_node(node)

    emitter = CEmitter()
    code = emitter.emit(CompileContext(graph=graph, target="x86", optimization_level=3))

    assert "nnc_conv_relu3x3_s1(&input, &weight, NULL, &output);" in code


def test_emit_gemm_call_uses_rhs_transposed_specialized_entrypoint() -> None:
    graph = Graph("gemm_specialized")
    _add_tensor(graph, "input", [2, 3])
    _add_tensor(graph, "weight", [4, 3])
    _add_tensor(graph, "bias", [4])
    _add_tensor(graph, "output", [2, 4])

    node = Node(
        op_type=OpType.GEMM,
        name="fc",
        inputs=["input", "weight", "bias"],
        outputs=["output"],
        attrs={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1},
        metadata={
            "lowering": {
                "kernel_family": "gemm",
                "weight_pack": "rhs_transposed_constant",
            }
        },
    )
    graph.add_node(node)

    emitter = CEmitter()
    code = emitter.emit(CompileContext(graph=graph, target="x86", optimization_level=3))

    assert "nnc_gemm_nt(&input, &weight, &bias, &output, 1.0f, 1.0f);" in code
