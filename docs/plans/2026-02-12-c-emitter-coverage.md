# C_Emitter Coverage Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve `c_emitter.py` test coverage from 37% to at least 70% by adding targeted unit tests for uncovered operator emit methods.

**Architecture:** The `CEmitter` class generates C code for each ONNX operator type. Each operator has a corresponding `_emit_<op>_call()` method. The strategy is to test each emit method by creating a minimal graph with that operator and verifying the generated C code contains expected patterns.

**Tech Stack:**
- pytest for testing
- ONNX `helper` for model creation
- `CEmitter` from `nnc_py.codegen.c_emitter`
- IR types: `Graph`, `Node`, `OpType`, `TensorType`, `CompileContext`

---

## Task 1: Create Test File and Helper Functions

**Files:**
- Create: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test (actually this will pass - we're adding infrastructure)**

```python
"""Test coverage for CEmitter operator emission methods.

This file targets increasing coverage of src/nnc_py/codegen/c_emitter.py
from 37% to 70%+ by testing individual operator emit methods.
"""

import pytest
from nnc_py.codegen.c_emitter import CEmitter
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType


def create_basic_graph(name: str = "test_graph") -> Graph:
    """Create a basic graph for testing."""
    graph = Graph(name=name)
    return graph


def create_compile_context(graph: Graph, target: str = "x86") -> CompileContext:
    """Create a compile context for testing."""
    return CompileContext(graph=graph, target=target)


def add_tensor_to_graph(graph: Graph, name: str, shape: list, dtype: DataType = DataType.FLOAT32) -> str:
    """Add a tensor to the graph and return its name."""
    tensor_shape = TensorShape(dims=shape)
    tensor = TensorType(dtype=dtype, shape=tensor_shape, name=name)
    graph.add_tensor(tensor)
    return name


def test_helper_functions_exist():
    """Test that helper functions work correctly."""
    graph = create_basic_graph()
    assert graph.name == "test_graph"

    ctx = create_compile_context(graph)
    assert ctx.target == "x86"

    add_tensor_to_graph(graph, "input", [1, 3, 224, 224])
    assert "input" in graph.tensors
```

**Step 2: Run test to verify it passes**

```bash
source .venv/bin/activate
pytest tests/test_c_emitter_coverage.py::test_helper_functions_exist -v
```

Expected: PASS

**Step 3: Nothing to implement - this is infrastructure**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test file structure and helper functions"
```

---

## Task 2: Test Concat Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_concat_call():
    """Test _emit_concat_call generates correct C code."""
    graph = create_basic_graph("concat_test")

    # Add input tensors
    add_tensor_to_graph(graph, "input1", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "input2", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "output", [1, 6, 224, 224])

    # Create concat node
    node = Node(
        op_type=OpType.CONCAT,
        name="concat1",
        inputs=["input1", "input2"],
        outputs=["output"],
        attrs={"axis": 1},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_concat_call(ctx, node)
    output = emitter.output.getvalue()

    # Verify generated code contains expected patterns
    assert "nnc_concat" in output
    assert "concat1_inputs" in output
    assert "&input1" in output or "&tensor_input1" in output
    assert "&input2" in output or "&tensor_input2" in output
    assert "&output" in output or "&tensor_output" in output
    assert "axis" in output or "1" in output
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_concat_call -v
```

Expected: PASS

**Step 3: Nothing to implement - CEmitter already has this method**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for concat operator emission"
```

---

## Task 3: Test Split Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_split_call():
    """Test _emit_split_call generates correct C code."""
    graph = create_basic_graph("split_test")

    # Add input and output tensors
    add_tensor_to_graph(graph, "input", [1, 4, 224, 224])
    add_tensor_to_graph(graph, "output1", [1, 2, 224, 224])
    add_tensor_to_graph(graph, "output2", [1, 2, 224, 224])

    # Create split node
    node = Node(
        op_type=OpType.SPLIT,
        name="split1",
        inputs=["input"],
        outputs=["output1", "output2"],
        attrs={"axis": 1},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_split_call(ctx, node)
    output = emitter.output.getvalue()

    # Verify generated code contains expected patterns
    assert "nnc_split" in output
    assert "split1_outputs" in output
    assert "&input" in output or "&tensor_input" in output
    assert "2" in output  # num_outputs
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_split_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for split operator emission"
```

---

## Task 4: Test Transpose Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_transpose_call_with_perm():
    """Test _emit_transpose_call with explicit permutation."""
    graph = create_basic_graph("transpose_test")

    add_tensor_to_graph(graph, "input", [1, 224, 224, 3])
    add_tensor_to_graph(graph, "output", [1, 3, 224, 224])

    node = Node(
        op_type=OpType.TRANSPOSE,
        name="transpose1",
        inputs=["input"],
        outputs=["output"],
        attrs={"perm": [0, 3, 1, 2]},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_transpose_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_transpose" in output
    assert "transpose1_perm" in output
    assert "0, 3, 1, 2" in output or "[0, 3, 1, 2]" in output


def test_emit_transpose_call_without_perm():
    """Test _emit_transpose_call without permutation (reverse)."""
    graph = create_basic_graph("transpose_test")

    add_tensor_to_graph(graph, "input", [1, 224, 224, 3])
    add_tensor_to_graph(graph, "output", [1, 3, 224, 224])

    node = Node(
        op_type=OpType.TRANSPOSE,
        name="transpose2",
        inputs=["input"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_transpose_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_transpose" in output
    assert "NULL" in output  # No permutation array
```

**Step 2: Run tests to verify they pass**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_transpose_call_with_perm -v
pytest tests/test_c_emitter_coverage.py::test_emit_transpose_call_without_perm -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run tests to verify they pass**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add tests for transpose operator emission"
```

---

## Task 5: Test Flatten Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_flatten_call():
    """Test _emit_flatten_call generates correct C code."""
    graph = create_basic_graph("flatten_test")

    add_tensor_to_graph(graph, "input", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "output", [1, 150528])

    node = Node(
        op_type=OpType.FLATTEN,
        name="flatten1",
        inputs=["input"],
        outputs=["output"],
        attrs={"axis": 1},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_flatten_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_flatten" in output
    assert "axis" in output or "1" in output
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_flatten_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for flatten operator emission"
```

---

## Task 6: Test Softmax Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_softmax_call():
    """Test _emit_softmax_call generates correct C code."""
    graph = create_basic_graph("softmax_test")

    add_tensor_to_graph(graph, "input", [1, 1000])
    add_tensor_to_graph(graph, "output", [1, 1000])

    node = Node(
        op_type=OpType.SOFTMAX,
        name="softmax1",
        inputs=["input"],
        outputs=["output"],
        attrs={"axis": -1},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_softmax_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_softmax" in output
    assert "-1" in output or "axis" in output
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_softmax_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for softmax operator emission"
```

---

## Task 7: Test Tile Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_tile_call():
    """Test _emit_tile_call generates correct C code."""
    graph = create_basic_graph("tile_test")

    add_tensor_to_graph(graph, "input", [1, 3])
    add_tensor_to_graph(graph, "repeats", [2])  # Tile 2 times
    add_tensor_to_graph(graph, "output", [1, 6])

    # Add repeats as constant
    graph.constants["repeats"] = [2, 2]

    node = Node(
        op_type=OpType.TILE,
        name="tile1",
        inputs=["input", "repeats"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_tile_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_tile" in output
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_tile_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for tile operator emission"
```

---

## Task 8: Test ReduceMean Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_reducemean_call():
    """Test _emit_reducemean_call generates correct C code."""
    graph = create_basic_graph("reducemean_test")

    add_tensor_to_graph(graph, "input", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "output", [1, 224, 224])

    node = Node(
        op_type=OpType.REDUCE_MEAN,
        name="reducemean1",
        inputs=["input"],
        outputs=["output"],
        attrs={"axes": [1], "keepdims": 1},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_reducemean_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_reducemean" in output
    assert "1" in output  # axis value
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_reducemean_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for reducemean operator emission"
```

---

## Task 9: Test ReduceSum Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_reducesum_call():
    """Test _emit_reducesum_call generates correct C code."""
    graph = create_basic_graph("reducesum_test")

    add_tensor_to_graph(graph, "input", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "output", [1, 224, 224])

    node = Node(
        op_type=OpType.REDUCE_SUM,
        name="reducesum1",
        inputs=["input"],
        outputs=["output"],
        attrs={"axes": [1], "keepdims": 1},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_reducesum_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_reducesum" in output
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_reducesum_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for reducesum operator emission"
```

---

## Task 10: Test Unsqueeze Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_unsqueeze_call():
    """Test _emit_unsqueeze_call generates correct C code."""
    graph = create_basic_graph("unsqueeze_test")

    add_tensor_to_graph(graph, "input", [1, 3])
    add_tensor_to_graph(graph, "output", [1, 1, 3])

    node = Node(
        op_type=OpType.UNSQUEEZE,
        name="unsqueeze1",
        inputs=["input"],
        outputs=["output"],
        attrs={"axis": 1},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_unsqueeze_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_unsqueeze" in output
    assert "1" in output  # axis value
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_unsqueeze_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for unsqueeze operator emission"
```

---

## Task 11: Test Clip Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_clip_call_with_attributes():
    """Test _emit_clip_call with min/max as attributes."""
    graph = create_basic_graph("clip_test")

    add_tensor_to_graph(graph, "input", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "output", [1, 3, 224, 224])

    node = Node(
        op_type=OpType.CLIP,
        name="clip1",
        inputs=["input"],
        outputs=["output"],
        attrs={"min": 0.0, "max": 6.0},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_clip_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_clip" in output
    assert "0.0" in output or "0" in output
    assert "6.0" in output or "6" in output


def test_emit_clip_call_with_inputs():
    """Test _emit_clip_call with min/max as constant inputs."""
    graph = create_basic_graph("clip_test2")

    add_tensor_to_graph(graph, "input", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "min_val", [1])
    add_tensor_to_graph(graph, "max_val", [1])
    add_tensor_to_graph(graph, "output", [1, 3, 224, 224])

    # Add constants
    graph.constants["min_val"] = [0.0]
    graph.constants["max_val"] = [6.0]

    node = Node(
        op_type=OpType.CLIP,
        name="clip2",
        inputs=["input", "min_val", "max_val"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_clip_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_clip" in output
```

**Step 2: Run tests to verify they pass**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_clip_call_with_attributes -v
pytest tests/test_c_emitter_coverage.py::test_emit_clip_call_with_inputs -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run tests to verify they pass**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add tests for clip operator emission"
```

---

## Task 12: Test Shape Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_shape_call():
    """Test _emit_shape_call generates correct C code."""
    graph = create_basic_graph("shape_test")

    add_tensor_to_graph(graph, "input", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "output", [4], dtype=DataType.INT64)

    node = Node(
        op_type=OpType.SHAPE,
        name="shape1",
        inputs=["input"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_shape_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_shape" in output
    assert "1, 3, 224, 224" in output or "[1, 3, 224, 224]" in output
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_shape_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for shape operator emission"
```

---

## Task 13: Test ConstantOfShape Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_constantofshape_call():
    """Test _emit_constantofshape_call generates correct C code."""
    graph = create_basic_graph("constantofshape_test")

    add_tensor_to_graph(graph, "shape_input", [3], dtype=DataType.INT64)
    add_tensor_to_graph(graph, "output", [2, 3, 4])

    # Add shape constant
    graph.constants["shape_input"] = [2, 3, 4]

    node = Node(
        op_type=OpType.CONSTANT_OF_SHAPE,
        name="constantofshape1",
        inputs=["shape_input"],
        outputs=["output"],
        attrs={"value": 1.0},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_constantofshape_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_constantofshape" in output
    assert "1.0" in output or "1f" in output
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_constantofshape_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for constantofshape operator emission"
```

---

## Task 14: Test Expand Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_expand_call():
    """Test _emit_expand_call generates correct C code."""
    graph = create_basic_graph("expand_test")

    add_tensor_to_graph(graph, "input", [1, 3])
    add_tensor_to_graph(graph, "shape_input", [3], dtype=DataType.INT64)
    add_tensor_to_graph(graph, "output", [2, 3])

    # Add shape constant
    graph.constants["shape_input"] = [2, 3]

    node = Node(
        op_type=OpType.EXPAND,
        name="expand1",
        inputs=["input", "shape_input"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_expand_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_expand" in output
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_expand_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for expand operator emission"
```

---

## Task 15: Test Greater/Or/Not Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_greater_call():
    """Test _emit_greater_call generates correct C code."""
    graph = create_basic_graph("greater_test")

    add_tensor_to_graph(graph, "input1", [1, 3])
    add_tensor_to_graph(graph, "input2", [1, 3])
    add_tensor_to_graph(graph, "output", [1, 3], dtype=DataType.BOOL)

    node = Node(
        op_type=OpType.GREATER,
        name="greater1",
        inputs=["input1", "input2"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_greater_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_greater" in output


def test_emit_or_call():
    """Test _emit_or_call generates correct C code."""
    graph = create_basic_graph("or_test")

    add_tensor_to_graph(graph, "input1", [1, 3], dtype=DataType.BOOL)
    add_tensor_to_graph(graph, "input2", [1, 3], dtype=DataType.BOOL)
    add_tensor_to_graph(graph, "output", [1, 3], dtype=DataType.BOOL)

    node = Node(
        op_type=OpType.OR,
        name="or1",
        inputs=["input1", "input2"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_or_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_or" in output


def test_emit_not_call():
    """Test _emit_not_call generates correct C code."""
    graph = create_basic_graph("not_test")

    add_tensor_to_graph(graph, "input", [1, 3], dtype=DataType.BOOL)
    add_tensor_to_graph(graph, "output", [1, 3], dtype=DataType.BOOL)

    node = Node(
        op_type=OpType.NOT,
        name="not1",
        inputs=["input"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_not_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_not" in output
```

**Step 2: Run tests to verify they pass**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_greater_call -v
pytest tests/test_c_emitter_coverage.py::test_emit_or_call -v
pytest tests/test_c_emitter_coverage.py::test_emit_not_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run tests to verify they pass**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add tests for comparison and logical operators"
```

---

## Task 16: Test Cast Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_cast_call():
    """Test _emit_cast_call generates correct C code."""
    graph = create_basic_graph("cast_test")

    add_tensor_to_graph(graph, "input", [1, 3], dtype=DataType.FLOAT32)
    add_tensor_to_graph(graph, "output", [1, 3], dtype=DataType.INT32)

    node = Node(
        op_type=OpType.CAST,
        name="cast1",
        inputs=["input"],
        outputs=["output"],
        attrs={"to": DataType.INT32},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_cast_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_cast" in output
    assert "NNC_DTYPE_INT32" in output
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_cast_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for cast operator emission"
```

---

## Task 17: Test Gather Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_gather_call():
    """Test _emit_gather_call generates correct C code."""
    graph = create_basic_graph("gather_test")

    add_tensor_to_graph(graph, "data", [2, 3], dtype=DataType.FLOAT32)
    add_tensor_to_graph(graph, "indices", [1], dtype=DataType.INT64)
    add_tensor_to_graph(graph, "output", [1, 3], dtype=DataType.FLOAT32)

    node = Node(
        op_type=OpType.GATHER,
        name="gather1",
        inputs=["data", "indices"],
        outputs=["output"],
        attrs={"axis": 0},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_gather_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_gather" in output
    assert "axis" in output or "0" in output
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_gather_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for gather operator emission"
```

---

## Task 18: Test LSTM Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_lstm_call():
    """Test _emit_lstm_call generates correct C code."""
    graph = create_basic_graph("lstm_test")

    add_tensor_to_graph(graph, "X", [1, 10, 64])
    add_tensor_to_graph(graph, "W", [256, 64])
    add_tensor_to_graph(graph, "R", [256, 64])
    add_tensor_to_graph(graph, "B", [512])
    add_tensor_to_graph(graph, "Y", [1, 10, 128])
    add_tensor_to_graph(graph, "Y_h", [1, 1, 128])
    add_tensor_to_graph(graph, "Y_c", [1, 1, 128])

    node = Node(
        op_type=OpType.LSTM,
        name="lstm1",
        inputs=["X", "W", "R", "B"],
        outputs=["Y", "Y_h", "Y_c"],
        attrs={"direction": "forward", "hidden_size": 128},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_lstm_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_lstm" in output
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_lstm_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run test to verify it passes**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add test for LSTM operator emission"
```

---

## Task 19: Test Fused Operator Emission

**Files:**
- Modify: `tests/test_c_emitter_coverage.py`

**Step 1: Write the failing test**

Add to `test_c_emitter_coverage.py`:

```python
def test_emit_fused_conv_relu_call():
    """Test _emit_fused_conv_relu_call generates correct C code."""
    graph = create_basic_graph("fused_conv_relu_test")

    add_tensor_to_graph(graph, "input", [1, 3, 224, 224])
    add_tensor_to_graph(graph, "weight", [64, 3, 7, 7])
    add_tensor_to_graph(graph, "output", [1, 64, 112, 112])

    node = Node(
        op_type=OpType.FUSED_CONV_RELU,
        name="conv_relu1",
        inputs=["input", "weight"],
        outputs=["output"],
        attrs={"kernel_shape": [7, 7], "strides": [2, 2], "pads": [3, 3]},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_fused_conv_relu_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_conv_relu" in output
    assert "NULL" in output  # No bias


def test_emit_fused_add_relu_call():
    """Test _emit_fused_add_relu_call generates correct C code."""
    graph = create_basic_graph("fused_add_relu_test")

    add_tensor_to_graph(graph, "input1", [1, 64, 112, 112])
    add_tensor_to_graph(graph, "input2", [1, 64, 112, 112])
    add_tensor_to_graph(graph, "output", [1, 64, 112, 112])

    node = Node(
        op_type=OpType.FUSED_ADD_RELU,
        name="add_relu1",
        inputs=["input1", "input2"],
        outputs=["output"],
        attrs={},
    )
    graph.add_node(node)

    ctx = create_compile_context(graph)

    emitter = CEmitter()
    emitter._emit_fused_add_relu_call(ctx, node)
    output = emitter.output.getvalue()

    assert "nnc_add_relu" in output
```

**Step 2: Run tests to verify they pass**

```bash
pytest tests/test_c_emitter_coverage.py::test_emit_fused_conv_relu_call -v
pytest tests/test_c_emitter_coverage.py::test_emit_fused_add_relu_call -v
```

Expected: PASS

**Step 3: Nothing to implement**

**Step 4: Run tests to verify they pass**

Same as Step 2.

**Step 5: Commit**

```bash
git add tests/test_c_emitter_coverage.py
git commit -m "test(c_emitter): add tests for fused operator emission"
```

---

## Task 20: Verify Coverage Improvement

**Files:**
- None (verification task)

**Step 1: Run coverage report**

```bash
source .venv/bin/activate
python -m coverage run -m pytest tests/test_c_emitter_coverage.py -v
python -m coverage report --include="src/nnc_py/codegen/c_emitter.py"
```

**Step 2: Verify coverage target met**

Expected: Coverage should be 60%+ (up from 37%)

**Step 3: If coverage target met, generate HTML report**

```bash
python -m coverage html --include="src/nnc_py/codegen/c_emitter.py"
```

**Step 4: Commit**

```bash
git add docs/plans/2026-02-12-c-emitter-coverage.md
git commit -m "docs: complete c_emitter coverage improvement plan"
```

---

## Summary of Tests

This plan adds ~30 new test cases covering:

1. **Concat** - Array of input pointers generation
2. **Split** - Array of output pointers generation
3. **Transpose** - With and without permutation
4. **Flatten** - Axis parameter handling
5. **Softmax** - Axis parameter handling
6. **Tile** - Repeats from constant input
7. **ReduceMean** - Axes and keepdims handling
8. **ReduceSum** - Axes and keepdims handling
9. **Unsqueeze** - Axis parameter handling
10. **Clip** - Min/max from attributes and inputs
11. **Shape** - Static shape array generation
12. **ConstantOfShape** - Value attribute handling
13. **Expand** - Broadcasting with shape constant
14. **Greater** - Comparison operator
15. **Or** - Logical OR operator
16. **Not** - Logical NOT operator
17. **Cast** - Type conversion with dtype mapping
18. **Gather** - Axis-aware indexing
19. **LSTM** - Complex multi-output RNN operator
20. **Fused Conv+ReLU** - Fusion operator without bias
21. **Fused Add+ReLU** - Binary fusion operator

**Expected Coverage Increase:** 37% → 60-70%
