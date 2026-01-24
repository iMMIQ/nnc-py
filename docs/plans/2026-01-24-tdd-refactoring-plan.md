# TDD 重构实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 通过 TDD 驱动的重构，优化 NNC-Py 项目的代码结构并清理技术债，确保功能一致性。

**架构:** 分阶段重构，每个阶段通过完整的测试覆盖确保功能不退化。使用红-绿-重构循环，先编写失败测试，再重构代码。

**技术栈:** Python 3.10+, pytest, mypy, ruff, syrupy (snapshot testing)

---

## 重构原则

### TDD 核心原则
1. **无失败测试，不写生产代码** - 重构前必须先有测试覆盖
2. **先看测试失败** - 确保测试确实检测到问题
3. **最小化变更** - 每次只重构一个小的功能单元
4. **保持测试通过** - 每次重构后所有测试必须通过

### 重构安全网
- 现有的 45+ 测试文件作为回归防护
- 快照测试确保生成代码一致性
- 运行时测试验证 C 代码正确性

---

## Phase 1: 类型系统完善 (mypy strict)

### 目标
达到 `mypy --strict` 检查通过，消除所有类型警告。

### 文件范围
- `src/nnc_py/ir/*.py` - IR 类型定义
- `src/nnc_py/passes/*.py` - Pass 系统
- `src/nnc_py/codegen/*.py` - 代码生成
- `src/nnc_py/frontend/*.py` - 前端

### 步骤

#### Task 1.1: 完善 IR 层类型注解

**Files:**
- Modify: `src/nnc_py/ir/graph.py`
- Modify: `src/nnc_py/ir/node.py`
- Modify: `src/nnc_py/ir/tensor.py`
- Modify: `src/nnc_py/ir/context.py`
- Test: `tests/test_ir_types.py` (新文件)

**Step 1: 编写失败测试 - 类型覆盖测试**

创建 `tests/test_ir_types.py`:

```python
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
```

**Step 2: 运行测试**

```bash
pytest tests/test_ir_types.py -v
```

Expected: PASS (这些是简单的健全性检查)

**Step 3: 运行 mypy 检查**

```bash
mypy src/nnc_py/ir/ --strict
```

Expected: FAIL with type errors

**Step 4: 修复类型问题**

在 `src/nnc_py/ir/graph.py` 中添加缺失的类型注解:

```python
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from nnc_py.ir.node import Node
from nnc_py.ir.tensor import TensorType


class Graph:
    """Computation graph - core IR structure."""

    def __init__(self, name: str = "main") -> None:
        self.name: str = name
        self.nodes: Dict[str, Node] = {}
        self.tensors: Dict[str, TensorType] = {}
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.constants: Dict[str, np.ndarray] = {}
        self._nx_graph: Optional[nx.DiGraph] = None

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.name] = node
        self._nx_graph = None  # Invalidate cached graph

    # ... 其他方法类似地添加类型注解
```

**Step 5: 再次运行 mypy**

```bash
mypy src/nnc_py/ir/graph.py --strict
```

Expected: PASS (或继续修复直到通过)

**Step 6: 运行所有测试**

```bash
pytest tests/ -v
```

Expected: 所有测试通过

**Step 7: 提交**

```bash
git add src/nnc_py/ir/graph.py tests/test_ir_types.py
git commit -m "refactor(ir): add strict type annotations to Graph class"
```

---

#### Task 1.2: 完善 Pass 系统类型注解

**Files:**
- Modify: `src/nnc_py/passes/base.py`
- Modify: `src/nnc_py/passes/liveness.py`
- Modify: `src/nnc_py/passes/memory_planning.py`
- Test: `tests/test_pass_types.py` (新文件)

**Step 1: 编写失败测试 - Pass 类型测试**

```python
"""测试 Pass 系统类型定义。"""

from nnc_py.passes.base import PassBase, PassManager
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph


def test_pass_manager_typed():
    """PassManager 应具有正确的类型。"""
    manager = PassManager()
    assert isinstance(manager.passes, list)
    assert isinstance(manager.applied_passes, list)


def test_pass_base_abstract():
    """PassBase 应该是抽象类。"""
    # 尝试实例化抽象类应该失败
    with pytest.raises(TypeError):
        PassBase()
```

**Step 2-7:** 类似 Task 1.1 的流程

---

#### Task 1.3: 完善代码生成器类型注解

**Files:**
- Modify: `src/nnc_py/codegen/c_emitter.py`
- Modify: `src/nnc_py/codegen/x86_backend.py`
- Test: `tests/test_codegen_types.py`

**Step 1-7:** 类似流程

---

#### Task 1.4: 创建类型检查 CI 任务

**Files:**
- Create: `.github/workflows/typecheck.yml`

```yaml
name: Type Check

on: [push, pull_request]

jobs:
  mypy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -e ".[dev]"
      - run: mypy src/nnc_py --strict
```

---

## Phase 2: IR 层重构

### 目标
优化 IR 数据结构，提升可维护性和扩展性。

### 重构点
1. Graph 类：分离图操作和数据访问
2. TensorShape：增强形状操作
3. Node：统一属性处理

### 步骤

#### Task 2.1: 提取 GraphBuilder

**问题:** Graph 类职责过多，同时管理数据存储和图构建。

**Files:**
- Create: `src/nnc_py/ir/builder.py`
- Modify: `src/nnc_py/ir/graph.py`
- Modify: `src/nnc_py/frontend/onnx_loader.py` (更新导入)
- Test: `tests/test_graph_builder.py`

**Step 1: 编写失败测试 - GraphBuilder 接口**

```python
"""测试 GraphBuilder 分离后的功能。"""

import pytest
from nnc_py.ir.builder import GraphBuilder
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType


def test_graph_builder_creates_graph():
    """GraphBuilder 应该创建一个正确的 Graph。"""
    builder = GraphBuilder(name="test")
    builder.add_input("input", DataType.FLOAT32, [1, 3, 224, 224])
    builder.add_output("output", DataType.FLOAT32, [1, 10])

    graph = builder.build()

    assert graph.name == "test"
    assert "input" in graph.inputs
    assert "output" in graph.outputs


def test_graph_builder_add_node():
    """GraphBuilder 应该正确添加节点。"""
    builder = GraphBuilder(name="test")
    builder.add_input("x", DataType.FLOAT32, [1, 10])

    builder.add_node(
        op_type=OpType.RELU,
        name="relu1",
        inputs=["x"],
        outputs=["y"],
    )

    graph = builder.build()
    assert "relu1" in graph.nodes
    assert graph.nodes["relu1"].op_type == OpType.RELU
```

**Step 2: 运行测试**

```bash
pytest tests/test_graph_builder.py -v
```

Expected: FAIL - "GraphBuilder not found"

**Step 3: 实现 GraphBuilder**

创建 `src/nnc_py/ir/builder.py`:

```python
"""Builder for constructing IR Graph objects."""

from typing import Dict, List, Optional

from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType, MemoryLayout


class GraphBuilder:
    """Builder pattern for constructing Graph objects."""

    def __init__(self, name: str = "main") -> None:
        """Initialize builder with graph name."""
        self._name: str = name
        self._nodes: Dict[str, Node] = {}
        self._tensors: Dict[str, TensorType] = {}
        self._inputs: List[str] = []
        self._outputs: List[str] = []
        self._constants: Dict[str, "np.ndarray"] = {}

    def add_input(
        self,
        name: str,
        dtype: DataType,
        shape: List[int],
    ) -> "GraphBuilder":
        """Add an input tensor to the graph."""
        tensor = TensorType(
            dtype=dtype,
            shape=TensorShape(dims=shape, layout=MemoryLayout.NCHW),
            name=name,
        )
        self._tensors[name] = tensor
        self._inputs.append(name)
        return self

    def add_output(
        self,
        name: str,
        dtype: DataType,
        shape: List[int],
    ) -> "GraphBuilder":
        """Add an output tensor to the graph."""
        tensor = TensorType(
            dtype=dtype,
            shape=TensorShape(dims=shape, layout=MemoryLayout.NCHW),
            name=name,
        )
        self._tensors[name] = tensor
        self._outputs.append(name)
        return self

    def add_node(
        self,
        op_type: OpType,
        name: str,
        inputs: List[str],
        outputs: List[str],
        attrs: Optional[Dict[str, object]] = None,
    ) -> "GraphBuilder":
        """Add a computation node to the graph."""
        node = Node(
            op_type=op_type,
            name=name,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs or {},
        )
        self._nodes[name] = node

        # Auto-register output tensors if not exists
        for output_name in outputs:
            if output_name not in self._tensors:
                # Placeholder tensor - will be resolved during type inference
                self._tensors[output_name] = TensorType(
                    dtype=DataType.FLOAT32,
                    shape=TensorShape(dims=[], layout=MemoryLayout.NHWC),
                    name=output_name,
                )

        return self

    def build(self) -> Graph:
        """Build and return the Graph object."""
        graph = Graph(name=self._name)

        # Add all tensors
        for tensor in self._tensors.values():
            graph.add_tensor(tensor)

        # Add all nodes
        for node in self._nodes.values():
            graph.add_node(node)

        # Set inputs and outputs
        graph.inputs = self._inputs.copy()
        graph.outputs = self._outputs.copy()

        return graph
```

**Step 4: 运行测试**

```bash
pytest tests/test_graph_builder.py -v
```

Expected: PASS

**Step 5: 验证现有测试仍然通过**

```bash
pytest tests/ -v -k "snapshot or e2e or runtime"
```

Expected: 所有快照和端到端测试通过

**Step 6: 提交**

```bash
git add src/nnc_py/ir/builder.py tests/test_graph_builder.py
git commit -m "refactor(ir): extract GraphBuilder for cleaner graph construction"
```

---

#### Task 2.2: 增强 TensorShape 操作

**问题:** TensorShape 缺少常用操作方法，导致代码散落各处。

**Files:**
- Modify: `src/nnc_py/ir/tensor.py` (增强 TensorShape)
- Test: `tests/test_tensor_shape.py`

**Step 1: 编写失败测试**

```python
"""测试 TensorShape 增强功能。"""

import pytest
from nnc_py.ir.tensor import TensorShape
from nnc_py.ir.types import MemoryLayout


def test_tensor_shape_rank():
    """rank() 应返回维度数量。"""
    shape = TensorShape(dims=[1, 3, 224, 224], layout="NCHW")
    assert shape.rank() == 4


def test_tensor_shape_size():
    """size() 应返回元素总数。"""
    shape = TensorShape(dims=[2, 3, 4, 5], layout="NCHW")
    assert shape.size() == 120


def test_tensor_shape_with_unknown_dim():
    """size() 应处理未知维度。"""
    shape = TensorShape(dims=[2, -1, 4], layout="NCHW")
    assert shape.size() is None  # 有未知维度时返回 None


def test_tensor_shape_is_contiguous():
    """is_contiguous() 应检查形状是否连续。"""
    shape = TensorShape(dims=[1, 3, 224, 224], layout="NCHW")
    assert shape.is_contiguous()


def test_tensor_shape_eq():
    """两个形状应该可比较。"""
    shape1 = TensorShape(dims=[1, 3, 224, 224], layout="NCHW")
    shape2 = TensorShape(dims=[1, 3, 224, 224], layout="NCHW")
    assert shape1 == shape2


def test_tensor_shape_with_axis():
    """with_axis() 应返回修改指定轴后的新形状。"""
    shape = TensorShape(dims=[1, 3, 224, 224], layout="NCHW")
    new_shape = shape.with_axis(axis=1, value=64)
    assert new_shape.dims == [1, 64, 224, 224]
    # 原形状不变
    assert shape.dims == [1, 3, 224, 224]
```

**Step 2-7:** 类似流程

---

#### Task 2.3: 统一 Node 属性处理

**问题:** Node 属性散布在 attrs 字典中，类型不安全。

**Files:**
- Modify: `src/nnc_py/ir/node.py`
- Test: `tests/test_node_attrs.py`

**Step 1: 编写失败测试**

```python
"""测试 Node 属性访问器。"""

import pytest
from nnc_py.ir.node import Node, OpType


def test_node_get_attr_with_default():
    """get_attr 应支持默认值。"""
    node = Node(
        op_type=OpType.CONV2D,
        name="conv1",
        inputs=["x"],
        outputs=["y"],
        attrs={"kernel_shape": [3, 3]},
    )

    assert node.get_attr("kernel_shape") == [3, 3]
    assert node.get_attr("strides", [1, 1]) == [1, 1]
    assert node.get_attr("nonexistent", "default") == "default"


def test_node_get_attr_int():
    """get_attr_int 应返回整数。"""
    node = Node(
        op_type=OpType.CONV2D,
        name="conv1",
        inputs=["x"],
        outputs=["y"],
        attrs={"group": 2},
    )

    assert node.get_attr_int("group") == 2
    assert node.get_attr_int("nonexistent", 1) == 1


def test_node_get_attr_list():
    """get_attr_list 应返回列表。"""
    node = Node(
        op_type=OpType.CONV2D,
        name="conv1",
        inputs=["x"],
        outputs=["y"],
        attrs={"kernel_shape": [3, 3]},
    )

    assert node.get_attr_list("kernel_shape") == [3, 3]
```

---

## Phase 3: Pass 系统优化

### 目标
增强 Pass 系统的可扩展性和可组合性。

### 重构点
1. Pass 依赖管理
2. Pass 验证和恢复
3. Pass 调试支持

### 步骤

#### Task 3.1: 实现 Pass 依赖管理

**Files:**
- Modify: `src/nnc_py/passes/base.py`
- Test: `tests/test_pass_dependencies.py`

**Step 1: 编写失败测试**

```python
"""测试 Pass 依赖管理。"""

from nnc_py.passes.base import PassBase, PassManager
from nnc_py.passes.liveness import LivenessAnalysisPass
from nnc_py.passes.memory_planning import MemoryPlanningPassV2


class TestPass1(PassBase):
    @property
    def name(self) -> str:
        return "TestPass1"

    def _execute(self, ctx) -> None:
        pass

    @property
    def dependencies(self) -> list[str]:
        return []


class TestPass2(PassBase):
    @property
    def name(self) -> str:
        return "TestPass2"

    def _execute(self, ctx) -> None:
        pass

    @property
    def dependencies(self) -> list[str]:
        return ["TestPass1"]


def test_pass_dependency_ordering():
    """PassManager 应按依赖顺序运行 Pass。"""
    manager = PassManager()
    manager.register(TestPass2())
    manager.register(TestPass1())

    # 应该先运行 TestPass1 (依赖), 再运行 TestPass2
    order = []
    for p in manager.passes:
        order.append(p.name)

    # 确保依赖在前
    assert order.index("TestPass1") < order.index("TestPass2")
```

---

#### Task 3.2: 实现 Pass 验证

**Files:**
- Create: `src/nnc_py/passes/validation.py`
- Test: `tests/test_pass_validation.py`

**Step 1: 编写失败测试**

```python
"""测试 Pass 验证。"""

import pytest
from nnc_py.passes.validation import GraphValidator
from nnc_py.ir.builder import GraphBuilder
from nnc_py.ir.node import OpType
from nnc_py.ir.types import DataType


def test_validate_no_undefined_inputs():
    """验证器应检测未定义的输入。"""
    builder = GraphBuilder(name="test")
    builder.add_input("x", DataType.FLOAT32, [1, 10])
    # 添加引用未定义输入的节点
    builder.add_node(
        op_type=OpType.ADD,
        name="add1",
        inputs=["x", "undefined_tensor"],  # undefined_tensor 未定义!
        outputs=["y"],
    )

    graph = builder.build()
    validator = GraphValidator()

    errors = validator.validate(graph)
    assert len(errors) > 0
    assert any("undefined_tensor" in str(e) for e in errors)
```

---

## Phase 4: 代码生成器重构

### 目标
分离代码生成的关注点，提升可维护性。

### 重构点
1. 分离算子发射逻辑
2. 统一内存布局生成
3. 模块化 C 代码生成

### 步骤

#### Task 4.1: 提取 OperatorEmitter 基类

**Files:**
- Create: `src/nnc_py/codegen/operator_emitter.py`
- Modify: `src/nnc_py/codegen/c_emitter.py`
- Test: `tests/test_operator_emitter.py`

**Step 1: 编写失败测试**

```python
"""测试 OperatorEmitter 分离。"""

from nnc_py.codegen.operator_emitter import OperatorEmitter, EmitterContext
from nnc_py.ir.node import Node, OpType


def test_operator_emitter_registry():
    """OperatorEmitter 应该支持注册。"""
    # 测试发射器注册机制
    from nnc_py.codegen.operator_emitter import get_emitter

    emitter = get_emitter(OpType.CONV2D)
    assert emitter is not None
    assert emitter.op_type == OpType.CONV2D


def test_emitter_context():
    """EmitterContext 应包含必要的信息。"""
    from nnc_py.ir.builder import GraphBuilder
    from nnc_py.ir.types import DataType

    builder = GraphBuilder(name="test")
    builder.add_input("x", DataType.FLOAT32, [1, 10])
    builder.add_output("y", DataType.FLOAT32, [1, 10])
    graph = builder.build()

    ctx = EmitterContext(graph=graph)
    assert ctx.graph == graph
    assert ctx.tensor_symbols == {}
```

---

## Phase 5: 错误处理统一化

### 目标
建立统一的错误处理机制，提升调试体验。

### 重构点
1. 定义异常层次结构
2. 错误上下文收集
3. 友好的错误消息

### 步骤

#### Task 5.1: 定义异常类

**Files:**
- Create: `src/nnc_py/exceptions.py`
- Modify: `src/nnc_py/compiler.py`
- Test: `tests/test_exceptions.py`

**Step 1: 编写失败测试**

```python
"""测试自定义异常。"""

import pytest
from nnc_py.exceptions import (
    NNCError,
    CompilationError,
    ValidationError,
    TypeInferenceError,
)


def test_exception_hierarchy():
    """所有异常应继承自 NNCError。"""
    assert issubclass(CompilationError, NNCError)
    assert issubclass(ValidationError, NNCError)
    assert issubclass(TypeInferenceError, NNCError)


def test_exception_with_context():
    """异常应携带上下文信息。"""
    error = ValidationError(
        message="Undefined tensor 'input'",
        location="GraphConv1",
        details={"tensor_name": "input", "node": "conv1"},
    )

    assert "Undefined tensor 'input'" in str(error)
    assert error.location == "GraphConv1"
    assert error.details["node"] == "conv1"
```

---

## Phase 6: 文档和代码风格

### 目标
统一代码风格，完善文档。

### 重构点
1. 代码格式化（ruff）
2. 文档字符串标准化
3. 类型注解文档

### 步骤

#### Task 6.1: 配置 ruff 并修复代码风格

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/test_code_style.py` (空测试，仅作为钩子)

**Step 1: 更新 ruff 配置**

```toml
[tool.ruff]
line-length = 100
target-version = "py310"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
    "RUF", # ruff-specific rules
]
ignore = ["E501"]  # line too long (handled by formatter)

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

**Step 2: 运行 ruff 检查**

```bash
ruff check src/nnc_py
```

**Step 3: 自动修复**

```bash
ruff check --fix src/nnc_py
```

**Step 4: 格式化代码**

```bash
ruff format src/nnc_py
```

---

## 执行检查清单

每个 Phase 完成后：

- [ ] 所有现有测试通过
- [ ] mypy --strict 检查通过
- [ ] ruff check 通过
- [ ] 代码已提交
- [ ] 快照测试已更新（如有必要）

---

## 回滚计划

如果重构导致问题：

1. 使用 `git revert` 回滚有问题的提交
2. 重新审视测试，确保覆盖面
3. 更小的步骤进行重构
4. 使用特性分支隔离工作

---

## 时间线建议

- Phase 1: 2-3 天 (类型系统)
- Phase 2: 3-4 天 (IR 重构)
- Phase 3: 2-3 天 (Pass 系统)
- Phase 4: 3-4 天 (代码生成器)
- Phase 5: 1-2 天 (错误处理)
- Phase 6: 1-2 天 (文档风格)

总计: 约 2-3 周
