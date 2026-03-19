# Transformer Fusion Patterns Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add operator fusion patterns for Transformer architecture models to NNC-Py, including Attention, Layer Normalization, GELU/SiLU activations, and residual connections.

**Architecture:** Extend the existing pattern-based fusion system by:
1. Adding new OpTypes for Transformer operators (Attention, GELU, SiLU, etc.)
2. Creating pattern helper functions for Transformer operations
3. Registering fusion patterns common in Transformer models
4. Writing tests for each pattern

**Tech Stack:** Python, pytest, existing NNC-Py pattern matching framework

---

## Background: Transformer Fusion Patterns Needed

Transformer models use these common fusion opportunities:

| Pattern | Description | Priority |
|---------|-------------|----------|
| MatMul + Bias | Basic linear layer | High |
| MatMul + Bias + GELU | FFN middle layer | High |
| MatMul + Bias + ReLU | FFN middle (alternative) | High |
| MatMul + Bias + SiLU | SwiGLU-style FFN | High |
| LayerNorm + ResidualAdd | Pre-norm block | High |
| Attention + Dropout | Post-attention | Medium |
| Softmax + Dropout | Attention inside | Medium |

---

## Task 1: Add Transformer OpTypes to OpType Enum

**Files:**
- Modify: `src/nnc_py/ir/node.py:8-93`

**Step 1: Write the failing test**

Create test file `tests/test_transformer_optypes.py`:

```python
# tests/test_transformer_optypes.py
import pytest
from nnc_py.ir.node import OpType


def test_transformer_optypes_exist():
    """Test that Transformer operator types are defined."""
    # Transformer-specific ops
    assert hasattr(OpType, 'GELU')
    assert hasattr(OpType, 'SILU')
    assert hasattr(OpType, 'ATTENTION')
    assert hasattr(OpType, 'DROPOUT')

    # Fused Transformer ops
    assert hasattr(OpType, 'FUSED_MATMUL_BIAS')
    assert hasattr(OpType, 'FUSED_MATMUL_BIAS_GELU')
    assert hasattr(OpType, 'FUSED_MATMUL_BIAS_RELU')
    assert hasattr(OpType, 'FUSED_MATMUL_BIAS_SILU')
    assert hasattr(OpType, 'FUSED_LAYER_NORM_ADD')
    assert hasattr(OpType, 'FUSED_SOFTMAX_DROPOUT')


def test_optype_values():
    """Test OpType enum values are correct."""
    assert OpType.GELU.value == "Gelu"
    assert OpType.SILU.value == "SiLU"
    assert OpType.ATTENTION.value == "Attention"
    assert OpType.DROPOUT.value == "Dropout"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_transformer_optypes.py -v
```

Expected: FAIL with `AttributeError: 'OpType' object has no attribute 'GELU'`

**Step 3: Write minimal implementation**

Edit `src/nnc_py/ir/node.py`, add to OpType enum (after line 75, before LAYER_NORM):

```python
# Transformer/Modern activation ops
GELU = "Gelu"
SILU = "SiLU"

# Attention ops
ATTENTION = "Attention"
DROPOUT = "Dropout"
```

Add fused OpTypes (after line 92):

```python
# Fused Transformer operators
FUSED_MATMUL_BIAS = "FusedMatMulBias"
FUSED_MATMUL_BIAS_GELU = "FusedMatMulBiasGelu"
FUSED_MATMUL_BIAS_RELU = "FusedMatMulBiasRelu"
FUSED_MATMUL_BIAS_SILU = "FusedMatMulBiasSiLU"
FUSED_LAYER_NORM_ADD = "FusedLayerNormAdd"
FUSED_SOFTMAX_DROPOUT = "FusedSoftmaxDropout"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_transformer_optypes.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/nnc_py/ir/node.py tests/test_transformer_optypes.py
git commit -m "feat(ir): add Transformer OpTypes (GELU, SiLU, Attention, Dropout, fused ops)"
```

---

## Task 2: Create Transformer Pattern Helper Functions

**Files:**
- Create: `src/nnc_py/pattern/transformer_patterns.py`

**Step 1: Write the failing test**

Create `tests/test_transformer_patterns.py`:

```python
# tests/test_transformer_patterns.py
import pytest
from nnc_py.pattern.transformer_patterns import (
    gelu, silu, attention, dropout, layer_norm, bias_add
)
from nnc_py.pattern.patterns import OpPattern
from nnc_py.ir.node import OpType


def test_pattern_helpers_return_op_patterns():
    """Test that pattern helpers return OpPattern instances."""
    assert isinstance(gelu(), OpPattern)
    assert isinstance(silu(), OpPattern)
    assert isinstance(attention(), OpPattern)
    assert isinstance(dropout(), OpPattern)
    assert isinstance(layer_norm(), OpPattern)
    assert isinstance(bias_add(), OpPattern)


def test_pattern_helpers_have_correct_op_types():
    """Test that pattern helpers match correct OpTypes."""
    assert gelu().op_type == OpType.GELU
    assert silu().op_type == OpType.SILU
    assert attention().op_type == OpType.ATTENTION
    assert dropout().op_type == OpType.DROPOUT
    assert layer_norm().op_type == OpType.LAYER_NORM
    assert bias_add().op_type == OpType.ADD
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_transformer_patterns.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nnc_py.pattern.transformer_patterns'`

**Step 3: Write minimal implementation**

Create `src/nnc_py/pattern/transformer_patterns.py`:

```python
# src/nnc_py/pattern/transformer_patterns.py
"""Transformer-specific pattern helpers for fusion."""

from nnc_py.pattern.patterns import OpPattern
from nnc_py.ir.node import OpType


def gelu(name: str = "gelu") -> OpPattern:
    """Create a GELU activation pattern."""
    return OpPattern(OpType.GELU, name)


def silu(name: str = "silu") -> OpPattern:
    """Create a SiLU (Swish) activation pattern."""
    return OpPattern(OpType.SILU, name)


def attention(name: str = "attention") -> OpPattern:
    """Create an Attention operator pattern."""
    return OpPattern(OpType.ATTENTION, name)


def dropout(name: str = "dropout") -> OpPattern:
    """Create a Dropout pattern."""
    return OpPattern(OpType.DROPOUT, name)


def layer_norm(name: str = "layer_norm") -> OpPattern:
    """Create a LayerNorm pattern."""
    return OpPattern(OpType.LAYER_NORM, name)


def bias_add(name: str = "bias") -> OpPattern:
    """Create a bias addition pattern (using Add op)."""
    return OpPattern(OpType.ADD, name)


def matmul(name: str = "matmul") -> OpPattern:
    """Create a MatMul pattern."""
    return OpPattern(OpType.MATMUL, name)


def softmax(name: str = "softmax") -> OpPattern:
    """Create a Softmax pattern."""
    return OpPattern(OpType.SOFTMAX, name)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_transformer_patterns.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/nnc_py/pattern/transformer_patterns.py tests/test_transformer_patterns.py
git commit -m "feat(patterns): add Transformer pattern helper functions"
```

---

## Task 3: Register MatMul + Bias Fusion Pattern

**Files:**
- Create: `src/nnc_py/pattern/transformer_fusion_patterns.py`

**Step 1: Write the failing test**

Create `tests/test_transformer_fusion.py`:

```python
# tests/test_transformer_fusion.py
import pytest
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.passes.pattern_fusion import PatternFusionPass
from nnc_py.pattern.registry import PatternRegistry


def test_matmul_bias_fusion():
    """Test MatMul + Bias fusion pattern."""
    # Clear registry for isolated test
    from nnc_py.pattern.registry import PatternRegistry
    PatternRegistry.clear()

    # Import to register patterns
    from nnc_py.pattern import transformer_fusion_patterns  # noqa: F401

    pattern = PatternRegistry.get("matmul_bias")
    assert pattern is not None
    assert pattern.name == "matmul_bias"

    # Create test graph: matmul -> add (bias)
    graph = Graph("test")
    matmul = Node(
        op_type=OpType.MATMUL,
        name="matmul1",
        inputs=["x", "w"],
        outputs=["matmul_out"],
        attrs={}
    )
    bias = Node(
        op_type=OpType.ADD,
        name="bias1",
        inputs=["matmul_out", "b"],
        outputs=["output"],
        attrs={}
    )
    graph.add_node(matmul)
    graph.add_node(bias)
    graph.outputs = ["output"]

    # Run pass
    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = PatternFusionPass()
    pass_obj.run(ctx)

    # Verify fusion
    assert "fused_matmul_bias_1" in graph.nodes
    fused_node = graph.nodes["fused_matmul_bias_1"]
    assert fused_node.op_type == OpType.FUSED_MATMUL_BIAS
    assert set(fused_node.inputs) == {"x", "w", "b"}
    assert fused_node.outputs == ["output"]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_transformer_fusion.py::test_matmul_bias_fusion -v
```

Expected: FAIL with `AssertionError: assert None is not None` (pattern not registered)

**Step 3: Write minimal implementation**

Create `src/nnc_py/pattern/transformer_fusion_patterns.py`:

```python
# src/nnc_py/pattern/transformer_fusion_patterns.py
"""Transformer fusion pattern definitions."""

from nnc_py.pattern.base import PatternMatch
from nnc_py.pattern.registry import register_pattern
from nnc_py.pattern.transformer_patterns import matmul, bias_add
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.graph import Graph


def _create_fused_matmul_bias(
    graph: Graph, match: PatternMatch, name: str
) -> Node:
    """Create fused MatMul+Bias node.

    Handles pattern: matmul(x, w) -> add(matmul_out, bias)
    """
    matmul_node = match.bindings["matmul"]
    bias_node = match.bindings["bias"]

    # Collect all inputs: matmul inputs + bias
    inputs = list(matmul_node.inputs)
    # The bias is the second input to Add (first is matmul output)
    if len(bias_node.inputs) == 2:
        bias_input = bias_node.inputs[1]
        inputs.append(bias_input)

    return Node(
        op_type=OpType.FUSED_MATMUL_BIAS,
        name=name,
        inputs=inputs,
        outputs=list(bias_node.outputs),
        attrs=matmul_node.attrs.copy(),
        metadata={"fused_from": [matmul_node.name, bias_node.name]}
    )


# Register MatMul + Bias pattern
register_pattern(
    name="matmul_bias",
    pattern=matmul().only_used_by(bias_add()),
    priority=250,  # Higher than basic patterns
    description="MatMul + Bias fusion (Linear layer)",
    fused_op_type=OpType.FUSED_MATMUL_BIAS,
    replace_func=_create_fused_matmul_bias,
)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_transformer_fusion.py::test_matmul_bias_fusion -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/nnc_py/pattern/transformer_fusion_patterns.py tests/test_transformer_fusion.py
git commit -m "feat(fusion): add MatMul+Bias fusion pattern"
```

---

## Task 4: Register MatMul + Bias + GELU Fusion Pattern

**Files:**
- Modify: `src/nnc_py/pattern/transformer_fusion_patterns.py`

**Step 1: Write the failing test**

Add to `tests/test_transformer_fusion.py`:

```python
def test_matmul_bias_gelu_fusion():
    """Test MatMul + Bias + GELU fusion pattern."""
    PatternRegistry.clear()
    from nnc_py.pattern import transformer_fusion_patterns  # noqa: F401

    pattern = PatternRegistry.get("matmul_bias_gelu")
    assert pattern is not None

    # Create test graph: matmul -> add -> gelu
    graph = Graph("test")
    matmul = Node(
        op_type=OpType.MATMUL,
        name="matmul1",
        inputs=["x", "w"],
        outputs=["matmul_out"],
        attrs={}
    )
    bias = Node(
        op_type=OpType.ADD,
        name="bias1",
        inputs=["matmul_out", "b"],
        outputs=["bias_out"],
        attrs={}
    )
    gelu_node = Node(
        op_type=OpType.GELU,
        name="gelu1",
        inputs=["bias_out"],
        outputs=["output"],
        attrs={}
    )
    graph.add_node(matmul)
    graph.add_node(bias)
    graph.add_node(gelu_node)
    graph.outputs = ["output"]

    # Run pass
    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = PatternFusionPass()
    pass_obj.run(ctx)

    # Verify fusion
    assert "fused_matmul_bias_gelu_1" in graph.nodes
    fused_node = graph.nodes["fused_matmul_bias_gelu_1"]
    assert fused_node.op_type == OpType.FUSED_MATMUL_BIAS_GELU
    assert set(fused_node.inputs) == {"x", "w", "b"}
    assert fused_node.outputs == ["output"]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_transformer_fusion.py::test_matmul_bias_gelu_fusion -v
```

Expected: FAIL with `AssertionError: assert None is not None`

**Step 3: Write minimal implementation**

Add to `src/nnc_py/pattern/transformer_fusion_patterns.py`:

```python
from nnc_py.pattern.transformer_patterns import gelu


def _create_fused_matmul_bias_gelu(
    graph: Graph, match: PatternMatch, name: str
) -> Node:
    """Create fused MatMul+Bias+GELU node (FFN middle layer)."""
    matmul_node = match.bindings["matmul"]
    bias_node = match.bindings["bias"]
    gelu_node = match.bindings["gelu"]

    inputs = list(matmul_node.inputs)
    if len(bias_node.inputs) == 2:
        inputs.append(bias_node.inputs[1])

    return Node(
        op_type=OpType.FUSED_MATMUL_BIAS_GELU,
        name=name,
        inputs=inputs,
        outputs=list(gelu_node.outputs),
        attrs=matmul_node.attrs.copy(),
        metadata={"fused_from": [matmul_node.name, bias_node.name, gelu_node.name]}
    )


# Register MatMul + Bias + GELU pattern
register_pattern(
    name="matmul_bias_gelu",
    pattern=matmul().only_used_by(bias_add()).only_used_by(gelu()),
    priority=260,  # Higher than matmul_bias
    description="MatMul + Bias + GELU fusion (FFN activation)",
    fused_op_type=OpType.FUSED_MATMUL_BIAS_GELU,
    replace_func=_create_fused_matmul_bias_gelu,
)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_transformer_fusion.py::test_matmul_bias_gelu_fusion -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/nnc_py/pattern/transformer_fusion_patterns.py tests/test_transformer_fusion.py
git commit -m "feat(fusion): add MatMul+Bias+GELU fusion pattern"
```

---

## Task 5: Register MatMul + Bias + ReLU Fusion Pattern

**Files:**
- Modify: `src/nnc_py/pattern/transformer_fusion_patterns.py`

**Step 1: Write the failing test**

Add to `tests/test_transformer_fusion.py`:

```python
def test_matmul_bias_relu_fusion():
    """Test MatMul + Bias + ReLU fusion pattern."""
    PatternRegistry.clear()
    from nnc_py.pattern import transformer_fusion_patterns  # noqa: F401

    pattern = PatternRegistry.get("matmul_bias_relu")
    assert pattern is not None

    graph = Graph("test")
    matmul = Node(
        op_type=OpType.MATMUL,
        name="matmul1",
        inputs=["x", "w"],
        outputs=["matmul_out"],
        attrs={}
    )
    bias = Node(
        op_type=OpType.ADD,
        name="bias1",
        inputs=["matmul_out", "b"],
        outputs=["bias_out"],
        attrs={}
    )
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu1",
        inputs=["bias_out"],
        outputs=["output"],
        attrs={}
    )
    graph.add_node(matmul)
    graph.add_node(bias)
    graph.add_node(relu_node)
    graph.outputs = ["output"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = PatternFusionPass()
    pass_obj.run(ctx)

    assert "fused_matmul_bias_relu_1" in graph.nodes
    fused_node = graph.nodes["fused_matmul_bias_relu_1"]
    assert fused_node.op_type == OpType.FUSED_MATMUL_BIAS_RELU
    assert set(fused_node.inputs) == {"x", "w", "b"}
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_transformer_fusion.py::test_matmul_bias_relu_fusion -v
```

Expected: FAIL with `AssertionError: assert None is not None`

**Step 3: Write minimal implementation**

Add to `src/nnc_py/pattern/transformer_fusion_patterns.py`:

```python
from nnc_py.pattern.patterns import OpPattern


def relu(name: str = "relu") -> OpPattern:
    """Create a ReLU pattern."""
    return OpPattern(OpType.RELU, name)


def _create_fused_matmul_bias_relu(
    graph: Graph, match: PatternMatch, name: str
) -> Node:
    """Create fused MatMul+Bias+ReLU node."""
    matmul_node = match.bindings["matmul"]
    bias_node = match.bindings["bias"]
    relu_node = match.bindings["relu"]

    inputs = list(matmul_node.inputs)
    if len(bias_node.inputs) == 2:
        inputs.append(bias_node.inputs[1])

    return Node(
        op_type=OpType.FUSED_MATMUL_BIAS_RELU,
        name=name,
        inputs=inputs,
        outputs=list(relu_node.outputs),
        attrs=matmul_node.attrs.copy(),
        metadata={"fused_from": [matmul_node.name, bias_node.name, relu_node.name]}
    )


register_pattern(
    name="matmul_bias_relu",
    pattern=matmul().only_used_by(bias_add()).only_used_by(relu()),
    priority=260,
    description="MatMul + Bias + ReLU fusion",
    fused_op_type=OpType.FUSED_MATMUL_BIAS_RELU,
    replace_func=_create_fused_matmul_bias_relu,
)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_transformer_fusion.py::test_matmul_bias_relu_fusion -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/nnc_py/pattern/transformer_fusion_patterns.py tests/test_transformer_fusion.py
git commit -m "feat(fusion): add MatMul+Bias+ReLU fusion pattern"
```

---

## Task 6: Register MatMul + Bias + SiLU Fusion Pattern

**Files:**
- Modify: `src/nnc_py/pattern/transformer_fusion_patterns.py`

**Step 1: Write the failing test**

Add to `tests/test_transformer_fusion.py`:

```python
def test_matmul_bias_silu_fusion():
    """Test MatMul + Bias + SiLU fusion pattern (SwiGLU-style)."""
    PatternRegistry.clear()
    from nnc_py.pattern import transformer_fusion_patterns  # noqa: F401

    pattern = PatternRegistry.get("matmul_bias_silu")
    assert pattern is not None

    graph = Graph("test")
    matmul = Node(
        op_type=OpType.MATMUL,
        name="matmul1",
        inputs=["x", "w"],
        outputs=["matmul_out"],
        attrs={}
    )
    bias = Node(
        op_type=OpType.ADD,
        name="bias1",
        inputs=["matmul_out", "b"],
        outputs=["bias_out"],
        attrs={}
    )
    silu_node = Node(
        op_type=OpType.SILU,
        name="silu1",
        inputs=["bias_out"],
        outputs=["output"],
        attrs={}
    )
    graph.add_node(matmul)
    graph.add_node(bias)
    graph.add_node(silu_node)
    graph.outputs = ["output"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = PatternFusionPass()
    pass_obj.run(ctx)

    assert "fused_matmul_bias_silu_1" in graph.nodes
    fused_node = graph.nodes["fused_matmul_bias_silu_1"]
    assert fused_node.op_type == OpType.FUSED_MATMUL_BIAS_SILU
    assert set(fused_node.inputs) == {"x", "w", "b"}
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_transformer_fusion.py::test_matmul_bias_silu_fusion -v
```

Expected: FAIL with `AssertionError: assert None is not None`

**Step 3: Write minimal implementation**

Add to `src/nnc_py/pattern/transformer_fusion_patterns.py`:

```python
from nnc_py.pattern.transformer_patterns import silu


def _create_fused_matmul_bias_silu(
    graph: Graph, match: PatternMatch, name: str
) -> Node:
    """Create fused MatMul+Bias+SiLU node (SwiGLU-style FFN)."""
    matmul_node = match.bindings["matmul"]
    bias_node = match.bindings["bias"]
    silu_node = match.bindings["silu"]

    inputs = list(matmul_node.inputs)
    if len(bias_node.inputs) == 2:
        inputs.append(bias_node.inputs[1])

    return Node(
        op_type=OpType.FUSED_MATMUL_BIAS_SILU,
        name=name,
        inputs=inputs,
        outputs=list(silu_node.outputs),
        attrs=matmul_node.attrs.copy(),
        metadata={"fused_from": [matmul_node.name, bias_node.name, silu_node.name]}
    )


register_pattern(
    name="matmul_bias_silu",
    pattern=matmul().only_used_by(bias_add()).only_used_by(silu()),
    priority=260,
    description="MatMul + Bias + SiLU fusion (SwiGLU-style FFN)",
    fused_op_type=OpType.FUSED_MATMUL_BIAS_SILU,
    replace_func=_create_fused_matmul_bias_silu,
)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_transformer_fusion.py::test_matmul_bias_silu_fusion -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/nnc_py/pattern/transformer_fusion_patterns.py tests/test_transformer_fusion.py
git commit -m "feat(fusion): add MatMul+Bias+SiLU fusion pattern (SwiGLU)"
```

---

## Task 7: Register LayerNorm + Add (Residual) Fusion Pattern

**Files:**
- Modify: `src/nnc_py/pattern/transformer_fusion_patterns.py`

**Step 1: Write the failing test**

Add to `tests/test_transformer_fusion.py`:

```python
def test_layer_norm_add_fusion():
    """Test LayerNorm + Add (residual) fusion pattern."""
    PatternRegistry.clear()
    from nnc_py.pattern import transformer_fusion_patterns  # noqa: F401

    pattern = PatternRegistry.get("layer_norm_add")
    assert pattern is not None

    graph = Graph("test")
    layer_norm = Node(
        op_type=OpType.LAYER_NORM,
        name="ln1",
        inputs=["x"],
        outputs=["ln_out"],
        attrs={"epsilon": 1e-5}
    )
    residual_add = Node(
        op_type=OpType.ADD,
        name="add1",
        inputs=["ln_out", "residual"],
        outputs=["output"],
        attrs={}
    )
    graph.add_node(layer_norm)
    graph.add_node(residual_add)
    graph.outputs = ["output"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = PatternFusionPass()
    pass_obj.run(ctx)

    assert "fused_layer_norm_add_1" in graph.nodes
    fused_node = graph.nodes["fused_layer_norm_add_1"]
    assert fused_node.op_type == OpType.FUSED_LAYER_NORM_ADD
    assert set(fused_node.inputs) == {"x", "residual"}
    assert fused_node.outputs == ["output"]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_transformer_fusion.py::test_layer_norm_add_fusion -v
```

Expected: FAIL with `AssertionError: assert None is not None`

**Step 3: Write minimal implementation**

Add to `src/nnc_py/pattern/transformer_fusion_patterns.py`:

```python
from nnc_py.pattern.transformer_patterns import layer_norm


def _create_fused_layer_norm_add(
    graph: Graph, match: PatternMatch, name: str
) -> Node:
    """Create fused LayerNorm+Add node (Pre-norm residual)."""
    ln_node = match.bindings["layer_norm"]
    add_node = match.bindings["add"]

    inputs = list(ln_node.inputs)
    # Add the residual input (second input to Add)
    if len(add_node.inputs) == 2:
        residual_input = add_node.inputs[1]
        inputs.append(residual_input)

    return Node(
        op_type=OpType.FUSED_LAYER_NORM_ADD,
        name=name,
        inputs=inputs,
        outputs=list(add_node.outputs),
        attrs=ln_node.attrs.copy(),
        metadata={"fused_from": [ln_node.name, add_node.name]}
    )


register_pattern(
    name="layer_norm_add",
    pattern=layer_norm().only_used_by(bias_add()),
    priority=240,
    description="LayerNorm + Add fusion (Pre-norm residual)",
    fused_op_type=OpType.FUSED_LAYER_NORM_ADD,
    replace_func=_create_fused_layer_norm_add,
)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_transformer_fusion.py::test_layer_norm_add_fusion -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/nnc_py/pattern/transformer_fusion_patterns.py tests/test_transformer_fusion.py
git commit -m "feat(fusion): add LayerNorm+Add residual fusion pattern"
```

---

## Task 8: Register Softmax + Dropout Fusion Pattern

**Files:**
- Modify: `src/nnc_py/pattern/transformer_fusion_patterns.py`

**Step 1: Write the failing test**

Add to `tests/test_transformer_fusion.py`:

```python
def test_softmax_dropout_fusion():
    """Test Softmax + Dropout fusion pattern (attention scores)."""
    PatternRegistry.clear()
    from nnc_py.pattern import transformer_fusion_patterns  # noqa: F401

    pattern = PatternRegistry.get("softmax_dropout")
    assert pattern is not None

    graph = Graph("test")
    softmax = Node(
        op_type=OpType.SOFTMAX,
        name="softmax1",
        inputs=["attn_scores"],
        outputs=["softmax_out"],
        attrs={"axis": -1}
    )
    dropout = Node(
        op_type=OpType.DROPOUT,
        name="dropout1",
        inputs=["softmax_out"],
        outputs=["output"],
        attrs={"rate": 0.1}
    )
    graph.add_node(softmax)
    graph.add_node(dropout)
    graph.outputs = ["output"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = PatternFusionPass()
    pass_obj.run(ctx)

    assert "fused_softmax_dropout_1" in graph.nodes
    fused_node = graph.nodes["fused_softmax_dropout_1"]
    assert fused_node.op_type == OpType.FUSED_SOFTMAX_DROPOUT
    assert fused_node.inputs == ["attn_scores"]
    assert fused_node.outputs == ["output"]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_transformer_fusion.py::test_softmax_dropout_fusion -v
```

Expected: FAIL with `AssertionError: assert None is not None`

**Step 3: Write minimal implementation**

Add to `src/nnc_py/pattern/transformer_fusion_patterns.py`:

```python
from nnc_py.pattern.transformer_patterns import softmax, dropout


def _create_fused_softmax_dropout(
    graph: Graph, match: PatternMatch, name: str
) -> Node:
    """Create fused Softmax+Dropout node (attention scores)."""
    softmax_node = match.bindings["softmax"]
    dropout_node = match.bindings["dropout"]

    # Merge attributes (keep softmax axis, add dropout rate)
    merged_attrs = softmax_node.attrs.copy()
    merged_attrs.update(dropout_node.attrs)

    return Node(
        op_type=OpType.FUSED_SOFTMAX_DROPOUT,
        name=name,
        inputs=list(softmax_node.inputs),
        outputs=list(dropout_node.outputs),
        attrs=merged_attrs,
        metadata={"fused_from": [softmax_node.name, dropout_node.name]}
    )


register_pattern(
    name="softmax_dropout",
    pattern=softmax().only_used_by(dropout()),
    priority=230,
    description="Softmax + Dropout fusion (attention scores)",
    fused_op_type=OpType.FUSED_SOFTMAX_DROPOUT,
    replace_func=_create_fused_softmax_dropout,
)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_transformer_fusion.py::test_softmax_dropout_fusion -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/nnc_py/pattern/transformer_fusion_patterns.py tests/test_transformer_fusion.py
git commit -m "feat(fusion): add Softmax+Dropout fusion pattern"
```

---

## Task 9: Import Transformer Fusion Patterns on Package Load

**Files:**
- Modify: `src/nnc_py/pattern/__init__.py`

**Step 1: Write the failing test**

Add to `tests/test_transformer_fusion.py`:

```python
def test_transformer_patterns_auto_registered():
    """Test that transformer patterns are registered on import."""
    # Clear and re-import
    PatternRegistry.clear()
    import nnc_py.pattern  # noqa: F401

    patterns = PatternRegistry.get_all()
    pattern_names = {p.name for p in patterns}

    # Check transformer patterns are registered
    assert "matmul_bias" in pattern_names
    assert "matmul_bias_gelu" in pattern_names
    assert "matmul_bias_relu" in pattern_names
    assert "matmul_bias_silu" in pattern_names
    assert "layer_norm_add" in pattern_names
    assert "softmax_dropout" in pattern_names
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_transformer_fusion.py::test_transformer_patterns_auto_registered -v
```

Expected: FAIL with `AssertionError: assert 'matmul_bias' in {...}` (patterns not auto-registered)

**Step 3: Write minimal implementation**

Edit `src/nnc_py/pattern/__init__.py`, add import after line 26:

```python
"""Dataflow Pattern Language for nnc-py.

This module provides TVM-style declarative pattern matching for operator fusion.
"""

from nnc_py.pattern.base import (
    DFPattern,
    PatternMatch,
    MatchContext,
)

from nnc_py.pattern.patterns import (
    WildcardPattern,
    OpPattern,
    OrPattern,
    AndPattern,
    AttrPattern,
    UsePattern,
    ExclusiveUsePattern,
)

from nnc_py.pattern.matcher import PatternMatcher
from nnc_py.pattern.registry import PatternRegistry, register_pattern, FusionPattern

# Import fusion_patterns to trigger pattern registration
from nnc_py.pattern import fusion_patterns  # noqa: F401
# Import transformer_fusion_patterns to trigger registration
from nnc_py.pattern import transformer_fusion_patterns  # noqa: F401

__all__ = [
    "DFPattern",
    "PatternMatch",
    "MatchContext",
    "WildcardPattern",
    "OpPattern",
    "OrPattern",
    "AndPattern",
    "AttrPattern",
    "UsePattern",
    "ExclusiveUsePattern",
    "PatternMatcher",
    "PatternRegistry",
    "register_pattern",
    "FusionPattern",
]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_transformer_fusion.py::test_transformer_patterns_auto_registered -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/nnc_py/pattern/__init__.py tests/test_transformer_fusion.py
git commit -m "feat(patterns): auto-register Transformer fusion patterns on import"
```

---

## Task 10: Update Built-in Patterns Test

**Files:**
- Modify: `tests/test_builtin_fusion_patterns.py`

**Step 1: Write the failing test**

Edit `tests/test_builtin_fusion_patterns.py`, add to `test_builtin_patterns_registered`:

```python
def test_builtin_patterns_registered():
    """Test that all built-in patterns are registered."""
    patterns = PatternRegistry.get_all()
    pattern_names = {p.name for p in patterns}

    # Original patterns
    assert "conv_relu" in pattern_names
    assert "conv_sigmoid" in pattern_names
    assert "add_relu" in pattern_names
    assert "add_sigmoid" in pattern_names
    assert "matmul_relu" in pattern_names

    # Transformer patterns
    assert "matmul_bias" in pattern_names
    assert "matmul_bias_gelu" in pattern_names
    assert "matmul_bias_relu" in pattern_names
    assert "matmul_bias_silu" in pattern_names
    assert "layer_norm_add" in pattern_names
    assert "softmax_dropout" in pattern_names
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_builtin_fusion_patterns.py::test_builtin_patterns_registered -v
```

Expected: PASS (if Task 9 is done)

**Step 3: Commit**

```bash
git add tests/test_builtin_fusion_patterns.py
git commit -m "test(patterns): add Transformer patterns to built-in patterns test"
```

---

## Task 11: Run All Tests

**Step 1: Run all pattern fusion tests**

```bash
pytest tests/test_transformer_fusion.py tests/test_transformer_patterns.py tests/test_transformer_optypes.py tests/test_builtin_fusion_patterns.py -v
```

Expected: All PASS

**Step 2: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All PASS

**Step 3: Commit**

```bash
git add .
git commit -m "test: all Transformer fusion pattern tests passing"
```

---

## Summary

After completing this plan, NNC-Py will support the following Transformer fusion patterns:

| Pattern | Fused OpType |
|---------|--------------|
| MatMul + Bias | FUSED_MATMUL_BIAS |
| MatMul + Bias + GELU | FUSED_MATMUL_BIAS_GELU |
| MatMul + Bias + ReLU | FUSED_MATMUL_BIAS_RELU |
| MatMul + Bias + SiLU | FUSED_MATMUL_BIAS_SILU |
| LayerNorm + Add | FUSED_LAYER_NORM_ADD |
| Softmax + Dropout | FUSED_SOFTMAX_DROPOUT |

These patterns cover the most common fusion opportunities in Transformer architectures like BERT, GPT, LLaMA, etc.
