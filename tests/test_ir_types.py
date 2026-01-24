"""测试 IR 类型定义的完整性。"""

import pytest
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType


def test_graph_creation_typed():
    """Graph 创建应具有正确的类型。"""
    graph = Graph(name="test_graph")
    assert graph.name == "test_graph"
    assert isinstance(graph.nodes, dict)
    assert isinstance(graph.tensors, dict)


def test_node_creation_typed():
    """Node 创建应具有正确的类型。"""
    node = Node(
        op_type=OpType.CONV2D,
        name="conv1",
        inputs=["input"],
        outputs=["output"],
        attrs={"kernel_shape": [3, 3]},
    )
    assert node.op_type == OpType.CONV2D
    assert isinstance(node.inputs, list)
    assert isinstance(node.outputs, list)
    assert isinstance(node.attrs, dict)


def test_tensor_type_typed():
    """TensorType 创建应具有正确的类型。"""
    shape = TensorShape(dims=[1, 3, 224, 224], layout="NCHW")
    tensor = TensorType(
        dtype=DataType.FLOAT32,
        shape=shape,
        name="input",
    )
    assert tensor.dtype == DataType.FLOAT32
    assert isinstance(tensor.shape, TensorShape)
