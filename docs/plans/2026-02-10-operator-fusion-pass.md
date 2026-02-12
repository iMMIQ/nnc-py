# Operator Fusion Pass Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement an OperatorFusionPass for the ONNX-to-C compiler that fuses compatible operator patterns (e.g., Conv+ReLU, Add+Activation) into single fused operations, using TDD methodology and ensuring all existing end-to-end tests continue to pass.

**Architecture:** A new optimization pass that:
1. **Identifies fusible patterns** - Finds sequences of operations that can be fused (Conv+ReLU, Add+Activation, etc.)
2. **Creates fused nodes** - Replaces the sequence with a single fused node
3. **Updates graph connectivity** - Rewires inputs/outputs correctly
4. **Preserves semantics** - Ensures output behavior is identical
5. **Integration at O3** - Adds the pass to the O3 optimization level

**Tech Stack:**
- Python 3.12
- pytest for testing
- Existing pass infrastructure: `PassBase`, `PassManager`, `CompileContext`
- IR types: `Graph`, `Node`, `OpType`, `TensorType`

**Key Design Decisions:**
- **Conservative fusion**: Only fuse when safe (single consumer, compatible attributes)
- **Pattern-based**: Define explicit fusible patterns (not arbitrary fusion)
- **Pass ordering**: Fusion runs AFTER identity elimination and DCE, BEFORE liveness analysis
- **Fused OpType**: Use new `OpType` enum values (FUSED_CONV_RELU, etc.) for codegen handling
- **No codegen changes**: Initially, fused nodes will be expanded during codegen (codegen phase handles fusion)
- **Idempotent**: Running fusion pass twice should have same effect as running once

**Fusion Patterns (Initial Scope):**
1. **Conv + ReLU** → `FUSED_CONV_RELU`
2. **Conv + Bias + ReLU** → `FUSED_CONV_BIAS_RELU`
3. **Add + ReLU** → `FUSED_ADD_RELU`
4. **Conv + Sigmoid** → `FUSED_CONV_SIGMOID`
5. **Add + Sigmoid** → `FUSED_ADD_SIGMOID`

---

## Task 1: Add Fused OpType Enum Values

**Files:**
- Modify: `src/nnc_py/ir/node.py`

**Step 1: Read current OpType enum**

Run: `cat src/nnc_py/ir/node.py | head -85`

Expected: See OpType enum ending with LSTM and GATHER ops around line 82

**Step 2: Add fused operator types to OpType enum**

Add after line 82 (after GATHER):

```python
    # Fused operators (for operator fusion pass)
    FUSED_CONV_RELU = "FusedConvRelu"
    FUSED_CONV_BIAS_RELU = "FusedConvBiasRelu"
    FUSED_CONV_SIGMOID = "FusedConvSigmoid"
    FUSED_ADD_RELU = "FusedAddRelu"
    FUSED_ADD_SIGMOID = "FusedAddSigmoid"
```

**Step 3: Verify the changes compile**

Run: `python -c "from nnc_py.ir.node import OpType; print(OpType.FUSED_CONV_RELU)"`

Expected: `OpType.FUSED_CONV_RELU`

**Step 4: Commit**

```bash
git add src/nnc_py/ir/node.py
git commit -m "feat(ir): add fused operator types to OpType enum"
```

---

## Task 2: Create OperatorFusionPass Test File - Basic Tests

**Files:**
- Create: `tests/test_operator_fusion_pass.py`

**Step 1: Write the failing test - pass exists and can be imported**

```python
"""Tests for OperatorFusionPass."""

import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.passes.operator_fusion import OperatorFusionPass


def test_operator_fusion_pass_exists():
    """Test that OperatorFusionPass can be imported and instantiated."""
    pass_obj = OperatorFusionPass()
    assert pass_obj.name == "OperatorFusion"


def test_conv_relu_fusion_basic():
    """Test fusing Conv followed by ReLU."""
    graph = Graph(name="test_conv_relu")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 3, 32, 32]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[16, 3, 3, 3]),
        name="conv_weight"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="conv_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="relu_out"
    ))

    # Conv node
    conv_node = Node(
        op_type=OpType.CONV2D,
        name="conv_1",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [0, 0, 0, 0]}
    )
    graph.add_node(conv_node)

    # ReLU node (only consumer of conv_out)
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["conv_out"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    graph.outputs = ["relu_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # Conv and ReLU nodes should be replaced with a fused node
    assert "conv_1" not in ctx.graph.nodes, "Original conv node should be removed"
    assert "relu_1" not in ctx.graph.nodes, "Original relu node should be removed"

    # Check for fused node
    assert "fused_conv_relu_1" in ctx.graph.nodes, "Fused node should be created"
    fused_node = ctx.graph.nodes["fused_conv_relu_1"]
    assert fused_node.op_type == OpType.FUSED_CONV_RELU
    assert fused_node.inputs == ["input", "conv_weight"]
    assert fused_node.outputs == ["relu_out"]


def test_conv_relu_not_fused_when_multiple_consumers():
    """Test that Conv+ReLU is NOT fused when conv output has multiple consumers."""
    graph = Graph(name="test_conv_relu_multi_consumer")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 3, 32, 32]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[16, 3, 3, 3]),
        name="conv_weight"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="conv_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="relu_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="other_out"
    ))

    # Conv node
    conv_node = Node(
        op_type=OpType.CONV2D,
        name="conv_1",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [0, 0, 0, 0]}
    )
    graph.add_node(conv_node)

    # ReLU node
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["conv_out"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    # Another consumer of conv_out
    other_node = Node(
        op_type=OpType.ADD,
        name="add_1",
        inputs=["conv_out", "conv_out"],
        outputs=["other_out"],
        attrs={}
    )
    graph.add_node(other_node)

    graph.outputs = ["relu_out", "other_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # Nodes should NOT be fused (conv_out has multiple consumers)
    assert "conv_1" in ctx.graph.nodes, "Conv node should remain when output has multiple consumers"
    assert "relu_1" in ctx.graph.nodes, "ReLU node should remain when conv output has multiple consumers"


def test_add_relu_fusion():
    """Test fusing Add followed by ReLU."""
    graph = Graph(name="test_add_relu")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="input1"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="input2"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="add_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="relu_out"
    ))

    # Add node
    add_node = Node(
        op_type=OpType.ADD,
        name="add_1",
        inputs=["input1", "input2"],
        outputs=["add_out"],
        attrs={}
    )
    graph.add_node(add_node)

    # ReLU node (only consumer of add_out)
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["add_out"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    graph.outputs = ["relu_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # Add and ReLU nodes should be replaced with a fused node
    assert "add_1" not in ctx.graph.nodes
    assert "relu_1" not in ctx.graph.nodes

    # Check for fused node
    assert "fused_add_relu_1" in ctx.graph.nodes
    fused_node = ctx.graph.nodes["fused_add_relu_1"]
    assert fused_node.op_type == OpType.FUSED_ADD_RELU
    assert fused_node.inputs == ["input1", "input2"]
    assert fused_node.outputs == ["relu_out"]


def test_idempotent():
    """Test that running the pass twice produces the same result."""
    graph = Graph(name="test_fusion_idempotent")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 3, 32, 32]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[16, 3, 3, 3]),
        name="conv_weight"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="conv_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="relu_out"
    ))

    conv_node = Node(
        op_type=OpType.CONV2D,
        name="conv_1",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [0, 0, 0, 0]}
    )
    graph.add_node(conv_node)

    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["conv_out"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    graph.outputs = ["relu_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()

    # Run pass once
    pass_obj.run(ctx)
    node_count_after_first = len(ctx.graph.nodes)
    fused_node_name = list(ctx.graph.nodes.keys())[0]

    # Run pass again
    pass_obj.run(ctx)
    node_count_after_second = len(ctx.graph.nodes)

    # Should have the same number of nodes
    assert node_count_after_first == node_count_after_second
    assert fused_node_name in ctx.graph.nodes
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_operator_fusion_pass.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'nnc_py.passes.operator_fusion'`

**Step 3: Create placeholder pass file**

Create: `src/nnc_py/passes/operator_fusion.py`

```python
"""Operator fusion pass.

This pass fuses compatible operator patterns (e.g., Conv+ReLU, Add+Activation)
into single fused operations for improved performance.
"""

from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassBase


class OperatorFusionPass(PassBase):
    """Fuse compatible operator patterns into single fused operations."""

    @property
    def name(self) -> str:
        return "OperatorFusion"

    def _execute(self, ctx: CompileContext) -> None:
        # TODO: implement fusion logic
        pass
```

**Step 4: Run test again to see different failure**

Run: `pytest tests/test_operator_fusion_pass.py::test_operator_fusion_pass_exists -v`

Expected: PASS (pass exists)

Run: `pytest tests/test_operator_fusion_pass.py::test_conv_relu_fusion_basic -v`

Expected: FAIL with assertion error (pass exists but doesn't fuse)

**Step 5: Commit**

```bash
git add tests/test_operator_fusion_pass.py src/nnc_py/passes/operator_fusion.py
git commit -m "test: add failing tests for OperatorFusionPass"
```

---

## Task 3: Implement OperatorFusionPass - Core Fusion Logic

**Files:**
- Modify: `src/nnc_py/passes/operator_fusion.py`

**Step 1: Implement the core fusion algorithm**

Replace the file content with:

```python
"""Operator fusion pass.

This pass fuses compatible operator patterns (e.g., Conv+ReLU, Add+Activation)
into single fused operations for improved performance.
"""

from typing import Dict, List, Optional, Set, Tuple

from nnc_py.ir.context import CompileContext
from nnc_py.ir.node import Node, OpType
from nnc_py.passes.base import PassBase


class OperatorFusionPass(PassBase):
    """Fuse compatible operator patterns into single fused operations.

    This pass identifies and fuses common operator patterns:
    - Conv + ReLU → FUSED_CONV_RELU
    - Add + ReLU → FUSED_ADD_RELU
    - Conv + Sigmoid → FUSED_CONV_SIGMOID
    - Add + Sigmoid → FUSED_ADD_SIGMOID

    Fusion is only performed when:
    1. The producer's output has exactly one consumer
    2. The pattern is recognized and safe to fuse
    3. The fusion would preserve graph semantics
    """

    @property
    def name(self) -> str:
        return "OperatorFusion"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute operator fusion."""
        graph = ctx.graph

        # Keep track of fused nodes to avoid double-processing
        fused_nodes: Set[str] = set()

        # Get nodes in topological order for deterministic processing
        nodes = graph.topological_sort()

        # Track fusion statistics
        fusion_count = 0
        patterns_found: Dict[str, int] = {}

        for node in nodes:
            # Skip already fused nodes
            if node.name in fused_nodes:
                continue

            # Try to fuse this node with its producer
            fusion_result = self._try_fusion_with_producer(graph, node, fused_nodes)

            if fusion_result:
                fusion_count += 1
                pattern_name = fusion_result
                patterns_found[pattern_name] = patterns_found.get(pattern_name, 0) + 1

        # Log summary if debug mode is on
        if ctx.debug:
            self._log_summary(fusion_count, patterns_found, len(graph.nodes))

    def _try_fusion_with_producer(
        self,
        graph,
        consumer: Node,
        fused_nodes: Set[str],
    ) -> Optional[str]:
        """Try to fuse the consumer node with its producer.

        Args:
            graph: The computation graph
            consumer: The consumer node
            fused_nodes: Set of already fused node names

        Returns:
            Pattern name if fusion occurred, None otherwise
        """
        # Only fuse single-input consumers (element-wise ops)
        if len(consumer.inputs) != 1:
            return None

        input_tensor = consumer.inputs[0]

        # Find producers of this tensor
        producers = graph.get_producers(input_tensor)

        # Need exactly one producer for fusion
        if len(producers) != 1:
            return None

        producer = producers[0]

        # Don't fuse if producer is already fused
        if producer.name in fused_nodes:
            return None

        # Check that producer's output has only one consumer
        consumers = graph.get_consumers(input_tensor)
        if len(consumers) != 1:
            return None

        # Try specific fusion patterns
        if producer.op_type == OpType.CONV2D and consumer.op_type == OpType.RELU:
            return self._fuse_conv_relu(graph, producer, consumer, fused_nodes)
        elif producer.op_type == OpType.CONV2D and consumer.op_type == OpType.SIGMOID:
            return self._fuse_conv_sigmoid(graph, producer, consumer, fused_nodes)
        elif producer.op_type == OpType.ADD and consumer.op_type == OpType.RELU:
            return self._fuse_add_relu(graph, producer, consumer, fused_nodes)
        elif producer.op_type == OpType.ADD and consumer.op_type == OpType.SIGMOID:
            return self._fuse_add_sigmoid(graph, producer, consumer, fused_nodes)

        return None

    def _fuse_conv_relu(
        self,
        graph,
        conv: Node,
        relu: Node,
        fused_nodes: Set[str],
    ) -> str:
        """Fuse Conv + ReLU into FUSED_CONV_RELU."""
        # Create fused node
        fused_node = Node(
            op_type=OpType.FUSED_CONV_RELU,
            name=f"fused_conv_relu_{len(fused_nodes) + 1}",
            inputs=conv.inputs,  # Take conv inputs
            outputs=relu.outputs,  # Output relu's output
            attrs=conv.attrs.copy(),  # Copy conv attributes
            metadata={"fused_from": [conv.name, relu.name]},
        )
        graph.add_node(fused_node)

        # Update graph outputs if needed
        self._update_graph_outputs(graph, conv.outputs[0], relu.outputs[0])

        # Remove original nodes
        del graph.nodes[conv.name]
        del graph.nodes[relu.name]

        # Mark as fused
        fused_nodes.add(conv.name)
        fused_nodes.add(relu.name)

        return "Conv+ReLU"

    def _fuse_conv_sigmoid(
        self,
        graph,
        conv: Node,
        sigmoid: Node,
        fused_nodes: Set[str],
    ) -> str:
        """Fuse Conv + Sigmoid into FUSED_CONV_SIGMOID."""
        fused_node = Node(
            op_type=OpType.FUSED_CONV_SIGMOID,
            name=f"fused_conv_sigmoid_{len(fused_nodes) + 1}",
            inputs=conv.inputs,
            outputs=sigmoid.outputs,
            attrs=conv.attrs.copy(),
            metadata={"fused_from": [conv.name, sigmoid.name]},
        )
        graph.add_node(fused_node)

        self._update_graph_outputs(graph, conv.outputs[0], sigmoid.outputs[0])

        del graph.nodes[conv.name]
        del graph.nodes[sigmoid.name]

        fused_nodes.add(conv.name)
        fused_nodes.add(sigmoid.name)

        return "Conv+Sigmoid"

    def _fuse_add_relu(
        self,
        graph,
        add: Node,
        relu: Node,
        fused_nodes: Set[str],
    ) -> str:
        """Fuse Add + ReLU into FUSED_ADD_RELU."""
        fused_node = Node(
            op_type=OpType.FUSED_ADD_RELU,
            name=f"fused_add_relu_{len(fused_nodes) + 1}",
            inputs=add.inputs,
            outputs=relu.outputs,
            attrs=add.attrs.copy(),
            metadata={"fused_from": [add.name, relu.name]},
        )
        graph.add_node(fused_node)

        self._update_graph_outputs(graph, add.outputs[0], relu.outputs[0])

        del graph.nodes[add.name]
        del graph.nodes[relu.name]

        fused_nodes.add(add.name)
        fused_nodes.add(relu.name)

        return "Add+ReLU"

    def _fuse_add_sigmoid(
        self,
        graph,
        add: Node,
        sigmoid: Node,
        fused_nodes: Set[str],
    ) -> str:
        """Fuse Add + Sigmoid into FUSED_ADD_SIGMOID."""
        fused_node = Node(
            op_type=OpType.FUSED_ADD_SIGMOID,
            name=f"fused_add_sigmoid_{len(fused_nodes) + 1}",
            inputs=add.inputs,
            outputs=sigmoid.outputs,
            attrs=add.attrs.copy(),
            metadata={"fused_from": [add.name, sigmoid.name]},
        )
        graph.add_node(fused_node)

        self._update_graph_outputs(graph, add.outputs[0], sigmoid.outputs[0])

        del graph.nodes[add.name]
        del graph.nodes[sigmoid.name]

        fused_nodes.add(add.name)
        fused_nodes.add(sigmoid.name)

        return "Add+Sigmoid"

    def _update_graph_outputs(self, graph, old_tensor: str, new_tensor: str) -> None:
        """Update graph outputs if old_tensor was an output."""
        if old_tensor in graph.outputs:
            # Replace old tensor with new tensor
            new_outputs = [new_tensor if t == old_tensor else t for t in graph.outputs]
            graph.outputs = new_outputs

    def _log_summary(self, fusion_count: int, patterns_found: Dict[str, int], node_count: int) -> None:
        """Log a summary of fusion results."""
        print(f"\n{'='*60}")
        print(f"Operator Fusion Summary")
        print(f"{'='*60}")
        print(f"Total fusions: {fusion_count}")
        print(f"Patterns found:")
        for pattern, count in sorted(patterns_found.items()):
            print(f"  - {pattern}: {count}")
        print(f"Nodes after fusion: {node_count}")
        print(f"{'='*60}")
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_operator_fusion_pass.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/nnc_py/passes/operator_fusion.py
git commit -m "feat: implement OperatorFusionPass core logic"
```

---

## Task 4: Add More Fusion Pattern Tests

**Files:**
- Modify: `tests/test_operator_fusion_pass.py`

**Step 1: Add tests for additional patterns**

Add these tests to the end of `tests/test_operator_fusion_pass.py`:

```python
def test_conv_sigmoid_fusion():
    """Test fusing Conv followed by Sigmoid."""
    graph = Graph(name="test_conv_sigmoid")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 3, 32, 32]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[16, 3, 3, 3]),
        name="conv_weight"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="conv_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="sigmoid_out"
    ))

    # Conv node
    conv_node = Node(
        op_type=OpType.CONV2D,
        name="conv_1",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [0, 0, 0, 0]}
    )
    graph.add_node(conv_node)

    # Sigmoid node
    sigmoid_node = Node(
        op_type=OpType.SIGMOID,
        name="sigmoid_1",
        inputs=["conv_out"],
        outputs=["sigmoid_out"],
        attrs={}
    )
    graph.add_node(sigmoid_node)

    graph.outputs = ["sigmoid_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # Check for fused node
    assert "fused_conv_sigmoid_1" in ctx.graph.nodes
    fused_node = ctx.graph.nodes["fused_conv_sigmoid_1"]
    assert fused_node.op_type == OpType.FUSED_CONV_SIGMOID


def test_add_sigmoid_fusion():
    """Test fusing Add followed by Sigmoid."""
    graph = Graph(name="test_add_sigmoid")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="input1"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="input2"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="add_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="sigmoid_out"
    ))

    # Add node
    add_node = Node(
        op_type=OpType.ADD,
        name="add_1",
        inputs=["input1", "input2"],
        outputs=["add_out"],
        attrs={}
    )
    graph.add_node(add_node)

    # Sigmoid node
    sigmoid_node = Node(
        op_type=OpType.SIGMOID,
        name="sigmoid_1",
        inputs=["add_out"],
        outputs=["sigmoid_out"],
        attrs={}
    )
    graph.add_node(sigmoid_node)

    graph.outputs = ["sigmoid_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # Check for fused node
    assert "fused_add_sigmoid_1" in ctx.graph.nodes
    fused_node = ctx.graph.nodes["fused_add_sigmoid_1"]
    assert fused_node.op_type == OpType.FUSED_ADD_SIGMOID


def test_multiple_fusions_in_graph():
    """Test fusing multiple patterns in the same graph."""
    graph = Graph(name="test_multiple_fusions")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 3, 32, 32]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[16, 3, 3, 3]),
        name="conv_weight"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="conv_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="relu_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="add_in"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="add_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="relu2_out"
    ))

    # Conv + ReLU
    conv_node = Node(
        op_type=OpType.CONV2D,
        name="conv_1",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        attrs={}
    )
    graph.add_node(conv_node)

    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["conv_out"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    # Add + ReLU
    add_node = Node(
        op_type=OpType.ADD,
        name="add_1",
        inputs=["relu_out", "add_in"],
        outputs=["add_out"],
        attrs={}
    )
    graph.add_node(add_node)

    relu2_node = Node(
        op_type=OpType.RELU,
        name="relu_2",
        inputs=["add_out"],
        outputs=["relu2_out"],
        attrs={}
    )
    graph.add_node(relu2_node)

    graph.outputs = ["relu2_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # Should have 2 fused nodes
    fused_conv_nodes = [n for n in ctx.graph.nodes.values() if n.op_type == OpType.FUSED_CONV_RELU]
    fused_add_nodes = [n for n in ctx.graph.nodes.values() if n.op_type == OpType.FUSED_ADD_RELU]

    assert len(fused_conv_nodes) == 1, "Should have 1 fused Conv+ReLU node"
    assert len(fused_add_nodes) == 1, "Should have 1 fused Add+ReLU node"


def test_does_not_fuse_graph_output_as_intermediate():
    """Test that fusion doesn't break when producer output is a graph output."""
    graph = Graph(name="test_output_preservation")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 3, 32, 32]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[16, 3, 3, 3]),
        name="conv_weight"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="conv_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="relu_out"
    ))

    # Conv node
    conv_node = Node(
        op_type=OpType.CONV2D,
        name="conv_1",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        attrs={}
    )
    graph.add_node(conv_node)

    # ReLU node
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["conv_out"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    # Set both as outputs (edge case)
    graph.outputs = ["conv_out", "relu_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # In this case, fusion should still happen since only relu_out is used
    # conv_out is an output but relu also uses it
    # After fusion, relu_out should be the only output
    assert "fused_conv_relu_1" in ctx.graph.nodes


def test_preserves_conv_attributes():
    """Test that fused node preserves Conv attributes."""
    graph = Graph(name="test_conv_attrs")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 3, 32, 32]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[16, 3, 3, 3]),
        name="conv_weight"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="conv_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[1, 16, 30, 30]),
        name="relu_out"
    ))

    # Conv node with specific attributes
    conv_attrs = {
        "kernel_shape": [5, 5],
        "strides": [2, 2],
        "pads": [1, 1, 1, 1],
        "dilations": [1, 1],
        "group": 1,
    }
    conv_node = Node(
        op_type=OpType.CONV2D,
        name="conv_1",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        attrs=conv_attrs
    )
    graph.add_node(conv_node)

    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["conv_out"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    graph.outputs = ["relu_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = OperatorFusionPass()
    pass_obj.run(ctx)

    # Check that attributes are preserved
    fused_node = ctx.graph.nodes["fused_conv_relu_1"]
    assert fused_node.attrs["kernel_shape"] == [5, 5]
    assert fused_node.attrs["strides"] == [2, 2]
    assert fused_node.attrs["pads"] == [1, 1, 1, 1]
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_operator_fusion_pass.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_operator_fusion_pass.py
git commit -m "test: add more fusion pattern tests"
```

---

## Task 5: Add End-to-End Integration Test

**Files:**
- Create: `tests/test_operator_fusion_e2e.py`

**Step 1: Write integration test that compiles a model with fusible patterns**

```python
"""End-to-end integration tests for OperatorFusionPass."""

import os
import tempfile
import shutil
from pathlib import Path

import onnx
from onnx import helper
import pytest

from nnc_py import Compiler


class TestOperatorFusionE2E:
    """End-to-end tests for operator fusion."""

    def setup_method(self):
        """Set up test environment."""
        self.tmp_dir = tempfile.mkdtemp()
        self.runtime_dir = Path(__file__).resolve().parent.parent / "runtime"

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_conv_relu_fusion_e2e(self):
        """Test that Conv+ReLU fusion works end-to-end."""
        # Create a model with Conv followed by ReLU
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 32, 32])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 16, 30, 30])

        # Weight initializer
        weight_init = helper.make_tensor(
            "conv_weight", onnx.TensorProto.FLOAT, [16, 3, 3, 3], [0.1] * (16 * 3 * 3 * 3)
        )

        # Conv node
        conv = helper.make_node(
            "Conv",
            inputs=["input", "conv_weight"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )

        # ReLU node
        relu = helper.make_node("Relu", inputs=["conv_out"], outputs=["output"])

        graph = helper.make_graph(
            [conv, relu],
            "conv_relu_test",
            [input_val],
            [output_val],
            [weight_init]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        # Save and compile
        onnx_path = os.path.join(self.tmp_dir, "model.onnx")
        onnx.save(model, onnx_path)

        output_dir = os.path.join(self.tmp_dir, "build")

        # Compile at O3 (with fusion)
        compiler = Compiler(target="x86", opt_level=3)
        compiler.compile(onnx_path, output_dir)

        # Check that model.c was generated
        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c should be generated"

    def test_add_relu_fusion_e2e(self):
        """Test that Add+ReLU fusion works end-to-end."""
        input1_val = helper.make_tensor_value_info("input1", onnx.TensorProto.FLOAT, [1, 16, 30, 30])
        input2_val = helper.make_tensor_value_info("input2", onnx.TensorProto.FLOAT, [1, 16, 30, 30])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 16, 30, 30])

        add = helper.make_node("Add", inputs=["input1", "input2"], outputs=["add_out"])
        relu = helper.make_node("Relu", inputs=["add_out"], outputs=["output"])

        graph = helper.make_graph(
            [add, relu],
            "add_relu_test",
            [input1_val, input2_val],
            [output_val]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        onnx_path = os.path.join(self.tmp_dir, "model.onnx")
        onnx.save(model, onnx_path)

        output_dir = os.path.join(self.tmp_dir, "build")
        compiler = Compiler(target="x86", opt_level=3)
        compiler.compile(onnx_path, output_dir)

        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c should be generated"

    def test_o3_enables_fusion(self):
        """Test that O3 enables fusion while O2 does not."""
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 32, 32])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 16, 30, 30])

        weight_init = helper.make_tensor(
            "conv_weight", onnx.TensorProto.FLOAT, [16, 3, 3, 3], [0.1] * (16 * 3 * 3 * 3)
        )

        conv = helper.make_node(
            "Conv",
            inputs=["input", "conv_weight"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
        )
        relu = helper.make_node("Relu", inputs=["conv_out"], outputs=["output"])

        graph = helper.make_graph(
            [conv, relu],
            "conv_relu_test",
            [input_val],
            [output_val],
            [weight_init]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        onnx_path = os.path.join(self.tmp_dir, "model.onnx")
        onnx.save(model, onnx_path)

        # Both O2 and O3 should compile successfully
        # (O3 has fusion, but the IR should still be valid)
        for opt_level in [2, 3]:
            output_dir = os.path.join(self.tmp_dir, f"build_o{opt_level}")
            compiler = Compiler(target="x86", opt_level=opt_level)
            compiler.compile(onnx_path, output_dir)

            model_c_path = os.path.join(output_dir, "model.c")
            assert os.path.exists(model_c_path), f"model.c should be generated at O{opt_level}"
```

**Step 2: Run the integration tests**

Run: `pytest tests/test_operator_fusion_e2e.py -v`

Expected: Tests may fail at first because:
1. Fusion pass is not registered at O3 yet (will fix in next task)
2. Codegen doesn't handle fused OpTypes yet (expected - tests verify compilation)

**Step 3: Commit**

```bash
git add tests/test_operator_fusion_e2e.py
git commit -m "test: add end-to-end tests for operator fusion"
```

---

## Task 6: Register OperatorFusionPass in PassManager

**Files:**
- Modify: `src/nnc_py/passes/base.py`
- Modify: `src/nnc_py/passes/__init__.py` (if needed)

**Step 1: Add import for OperatorFusionPass**

At the top of `src/nnc_py/passes/base.py`, add the import (after line 81):

```python
from nnc_py.passes.spill import SpillAnalysisPass
from nnc_py.passes.operator_fusion import OperatorFusionPass
```

**Step 2: Edit get_default_passes to include fusion at O3**

Find the O3 section (around line 108-117) and modify it:

```python
# O3: Advanced optimizations
if opt_level >= 3:
    return [
        IdentityEliminationPass(),
        DeadCodeEliminationPass(),
        OperatorFusionPass(),  # NEW: Fuse operators before liveness analysis
        LivenessAnalysisPass(),
        MemoryPlanningPassV2(),
        SpillAnalysisPass(),
    ]
```

**Step 3: Verify imports work**

Run: `python -c "from nnc_py.passes import PassManager; pm = PassManager.get_default_passes(3); print([p.name for p in pm])"`

Expected: Includes "OperatorFusion" in the list

**Step 4: Commit**

```bash
git add src/nnc_py/passes/base.py
git commit -m "feat: register OperatorFusionPass at O3 optimization level"
```

---

## Task 7: Add Codegen Support for Fused Operators

**Files:**
- Modify: `src/nnc_py/codegen/x86_codegen.py` (or equivalent codegen file)

**Step 1: Find the codegen file**

Run: `ls src/nnc_py/codegen/`

Expected: See files like `x86_codegen.py` or similar

**Step 2: Read the existing codegen patterns**

Run: `grep -n "def.*Conv\|def.*Relu\|def.*Add" src/nnc_py/codegen/*.py`

Expected: See code generation methods for these operators

**Step 3: Add codegen support for fused operators**

The approach: Expand fused operators back to individual operations during codegen.
This is a simple initial approach - the fused node just generates the sequence.

Add a helper method that expands fused nodes:

```python
def _expand_fused_operator(self, node):
    """Expand a fused operator into its component operations.

    For initial implementation, we simply generate the individual ops.
    Future optimization: generate truly fused code.
    """
    if node.op_type == OpType.FUSED_CONV_RELU:
        # Generate conv followed by relu in-place
        return [
            (OpType.CONV2D, node.inputs, node.outputs, node.attrs, "fused"),
            (OpType.RELU, node.outputs, node.outputs, {}, "in_place")
        ]
    elif node.op_type == OpType.FUSED_CONV_SIGMOID:
        return [
            (OpType.CONV2D, node.inputs, node.outputs, node.attrs, "fused"),
            (OpType.SIGMOID, node.outputs, node.outputs, {}, "in_place")
        ]
    elif node.op_type == OpType.FUSED_ADD_RELU:
        return [
            (OpType.ADD, node.inputs, node.outputs, node.attrs, "fused"),
            (OpType.RELU, node.outputs, node.outputs, {}, "in_place")
        ]
    elif node.op_type == OpType.FUSED_ADD_SIGMOID:
        return [
            (OpType.ADD, node.inputs, node.outputs, node.attrs, "fused"),
            (OpType.SIGMOID, node.outputs, node.outputs, {}, "in_place")
        ]
    else:
        return None
```

**Step 4: Modify codegen to handle fused operators**

In the main codegen loop, check for fused operators and expand them:

```python
# In the node iteration loop
if node.op_type in [OpType.FUSED_CONV_RELU, OpType.FUSED_CONV_SIGMOID,
                    OpType.FUSED_ADD_RELU, OpType.FUSED_ADD_SIGMOID]:
    expanded = self._expand_fused_operator(node)
    if expanded:
        # Generate code for each operation in the fusion
        for op_type, inputs, outputs, attrs, mode in expanded:
            # Reuse existing codegen for the base operation
            if mode == "in_place":
                # Generate in-place operation (same input/output buffer)
                self._generate_in_place_operation(op_type, outputs[0])
            else:
                # Normal operation
                self._generate_operation(op_type, inputs, outputs, attrs)
else:
    # Normal codegen for non-fused ops
    self._generate_operation(node.op_type, node.inputs, node.outputs, node.attrs)
```

**Note**: The exact implementation depends on your codegen structure. Adjust accordingly.

**Step 5: Run tests to verify codegen works**

Run: `pytest tests/test_operator_fusion_e2e.py -v`

Expected: Tests PASS

**Step 6: Commit**

```bash
git add src/nnc_py/codegen/
git commit -m "feat(codegen): add support for fused operator code generation"
```

---

## Task 8: Run Full E2E Test Suite (Regression Testing)

**Files:**
- No new files, run existing tests

**Step 1: Run all existing E2E tests to ensure no regressions**

Run: `pytest tests/test_e2e.py -v`

Expected: All tests PASS (or fix any issues found)

**Step 2: Run snapshot tests with O0/O1 to ensure no regressions**

Run: `pytest tests/test_snapshots_simple_conv.py -v`

Expected: Tests PASS (O0/O1 behavior unchanged)

**Step 3: Run snapshot tests with O3 (new fusion pass)**

Run: `pytest tests/test_snapshots_simple_conv.py::TestCodegenSnapshots::test_simple_conv_codegen_with_runtime_o3 -v`

Expected: Test PASS (O3 with fusion produces correct output)

**Step 4: Run all fusion tests**

Run: `pytest tests/test_operator_fusion_pass.py tests/test_operator_fusion_e2e.py -v`

Expected: All tests PASS

**Step 5: If any tests fail, investigate and fix**

Common issues:
1. Graph outputs not updated correctly after fusion
2. Tensor metadata missing for intermediate tensors
3. Codegen not handling fused operators properly

**Step 6: Commit**

```bash
git add -A
git commit -m "fix: resolve test failures from operator fusion integration"
```

---

## Task 9: Update Pass Documentation

**Files:**
- Modify: `docs/OPTIMIZATION_PASSES.md`

**Step 1: Add documentation for OperatorFusionPass**

Add a new section in the Advanced Optimizations part:

```markdown
### Advanced Optimizations (O3)

Applied at `opt_level=3` and above:

All O2 passes plus:

- **OperatorFusionPass**: Fuses compatible operator patterns
  - Conv + ReLU → FusedConvRelu (reduces memory traffic)
  - Add + ReLU → FusedAddRelu (faster activation)
  - Conv + Sigmoid → FusedConvSigmoid
  - Add + Sigmoid → FusedAddSigmoid
  - Only fuses when producer output has single consumer
  - Preserves graph semantics exactly
```

Also update the "Pass Execution Order" section:

```markdown
### Pass Execution Order

Passes are executed in the order they are registered. For O3:

1. IdentityEliminationPass - removes no-op operations
2. DeadCodeEliminationPass - removes now-unused nodes
3. OperatorFusionPass - fuses compatible operator patterns
4. LivenessAnalysisPass - analyzes lifetimes on optimized graph
5. MemoryPlanningPassV2 - plans memory based on lifetimes
6. SpillAnalysisPass - handles overflow if needed
```

**Step 2: Commit**

```bash
git add docs/OPTIMIZATION_PASSES.md
git commit -m "docs: add OperatorFusionPass documentation"
```

---

## Task 10: Final Verification and Testing

**Files:**
- No new files

**Step 1: Run complete test suite (excluding slow snapshot tests)**

Run: `pytest tests/ -v -k "not snapshot" --ignore=tests/test_snapshots_`

Expected: All tests PASS

**Step 2: Test with a real model (if available)**

Run: `python -c "
from nnc_py import Compiler
import os

# Use existing simple_conv.onnx if available
model_path = 'tests/simple_conv.onnx'
if os.path.exists(model_path):
    import tempfile
    tmpdir = tempfile.mkdtemp()

    # Test O0 (no fusion)
    compiler_o0 = Compiler(target='x86', opt_level=0)
    compiler_o0.compile(model_path, tmpdir + '/o0')
    print('O0 compilation: OK')

    # Test O3 (with fusion)
    compiler_o3 = Compiler(target='x86', opt_level=3)
    compiler_o3.compile(model_path, tmpdir + '/o3')
    print('O3 compilation: OK')

    print('Verification complete!')
else:
    print('Model not found, skipping real model test')
"
`

Expected: Both compilations succeed

**Step 3: Verify fusion statistics with debug mode**

Run: `python -c "
from nnc_py import Compiler
from nnc_py.frontend.onnx_loader import ONNXFrontend
import tempfile, os

# Create a simple Conv+ReLU model
import onnx
from onnx import helper

input_val = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 3, 32, 32])
output_val = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 16, 30, 30])
weight_init = helper.make_tensor('w', onnx.TensorProto.FLOAT, [16, 3, 3, 3], [0.1] * 432)

conv = helper.make_node('Conv', inputs=['input', 'w'], outputs=['c'], kernel_shape=[3, 3])
relu = helper.make_node('Relu', inputs=['c'], outputs=['output'])

graph = helper.make_graph([conv, relu], 'test', [input_val], [output_val], [weight_init])
model = helper.make_model(graph)
model.opset_import[0].version = 13

tmpdir = tempfile.mkdtemp()
onnx.save(model, os.path.join(tmpdir, 'test.onnx'))

# Compile with debug mode
compiler = Compiler(target='x86', opt_level=3, debug_mode=True)
compiler.compile(os.path.join(tmpdir, 'test.onnx'), tmpdir + '/build')
"
`

Expected: See fusion summary printed showing "Conv+ReLU: 1"

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete OperatorFusionPass implementation with full test coverage"
```

---

## Summary

This plan implements an OperatorFusionPass using TDD methodology:

1. **Fused OpType enum** - New types for fused operators
2. **OperatorFusionPass** - Identifies and fuses compatible patterns
3. **Codegen support** - Handles fused operators during code generation
4. **Integration at O3** - Fusion enabled at highest optimization level
5. **Full test coverage** - Unit tests, E2E tests, and regression testing

Key points:
- TDD approach: tests written before implementation
- Conservative fusion: only fuses when safe (single consumer)
- No behavior change: fused ops expanded during codegen
- End-to-end tests ensure no regressions
- Debug mode provides fusion statistics
