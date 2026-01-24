"""测试代码生成器类型定义。"""

import pytest
from nnc_py.codegen.c_emitter import CEmitter
from nnc_py.codegen.x86_backend import X86Backend
from nnc_py.codegen.base import BackendBase, CodeGenResult, CodeArtifact
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType


def test_c_emitter_creation_typed():
    """CEmitter 创建应具有正确的类型。"""
    emitter = CEmitter()
    assert emitter.indent == 0
    assert isinstance(emitter.output, object)


def test_c_emitter_emit_returns_str():
    """CEmitter.emit 应返回字符串。"""
    graph = Graph(name="test_graph")
    ctx = CompileContext(graph=graph, target="x86")

    emitter = CEmitter()
    result = emitter.emit(ctx)
    assert isinstance(result, str)


def test_x86_backend_creation_typed():
    """X86Backend 创建应具有正确的类型。"""
    backend = X86Backend()
    assert isinstance(backend, BackendBase)


def test_x86_backend_generate_returns_codegen_result():
    """X86Backend.generate 应返回 CodeGenResult。"""
    graph = Graph(name="test_graph")
    ctx = CompileContext(graph=graph, target="x86")

    backend = X86Backend()
    result = backend.generate(ctx)
    assert isinstance(result, CodeGenResult)
    assert isinstance(result.files, list)


def test_codegen_result_artifact_types():
    """CodeGenResult 中的文件应为 CodeArtifact 类型。"""
    graph = Graph(name="test_graph")
    ctx = CompileContext(graph=graph, target="x86")

    backend = X86Backend()
    result = backend.generate(ctx)

    for artifact in result.files:
        assert isinstance(artifact, CodeArtifact)
        assert isinstance(artifact.filename, str)
        assert isinstance(artifact.file_type, str)


def test_codegen_result_add_file_typed():
    """CodeGenResult.add_file 应具有正确的类型签名。"""
    result = CodeGenResult()
    result.add_file("test.c", "int x = 42;", "source")
    assert len(result.files) == 1
    assert result.files[0].filename == "test.c"


def test_c_emitter_write_line_typed():
    """CEmitter.write_line 应正确处理字符串输入。"""
    emitter = CEmitter()
    emitter.write_line("test line")
    assert "test line" in emitter.output.getvalue()


def test_x86_backend_generate_with_simple_graph():
    """X86Backend 应能处理包含简单节点的图。"""
    graph = Graph(name="simple_graph")

    # Add input tensor
    input_shape = TensorShape(dims=[1, 3, 224, 224])
    input_tensor = TensorType(
        dtype=DataType.FLOAT32,
        shape=input_shape,
        name="input",
    )
    graph.add_tensor(input_tensor)
    graph.inputs.append("input")

    # Add output tensor
    output_shape = TensorShape(dims=[1, 64, 112, 112])
    output_tensor = TensorType(
        dtype=DataType.FLOAT32,
        shape=output_shape,
        name="output",
    )
    graph.add_tensor(output_tensor)
    graph.outputs.append("output")

    # Add a simple node
    node = Node(
        op_type=OpType.RELU,
        name="relu1",
        inputs=["input"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = CompileContext(graph=graph, target="x86")
    backend = X86Backend()
    result = backend.generate(ctx)

    assert isinstance(result, CodeGenResult)
    assert len(result.files) > 0

    # Check that expected files are generated
    filenames = [f.filename for f in result.files]
    assert "model.h" in filenames
    assert "model.c" in filenames


def test_backend_base_is_abstract():
    """BackendBase 应该是抽象类。"""
    with pytest.raises(TypeError):
        BackendBase()


def test_code_artifact_dataclass():
    """CodeArtifact 数据类应具有正确的字段类型。"""
    artifact = CodeArtifact(
        filename="test.c",
        content="int x = 42;",
        file_type="source",
    )
    assert artifact.filename == "test.c"
    assert artifact.content == "int x = 42;"
    assert artifact.file_type == "source"


def test_code_artifact_with_binary_content():
    """CodeArtifact 应支持二进制内容。"""
    binary_content = b"\x00\x01\x02\x03"
    artifact = CodeArtifact(
        filename="test.bin",
        content=binary_content,
        file_type="binary",
    )
    assert artifact.content == binary_content
    assert artifact.file_type == "binary"
