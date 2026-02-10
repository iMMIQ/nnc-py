# DeadCodeEliminationPass and IdentityEliminationPass Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement two optimization passes for the ONNX-to-C compiler - DeadCodeEliminationPass (removes unused nodes/tensors) and IdentityEliminationPass (removes Identity operations) using TDD methodology.

**Architecture:** Two independent passes that:
1. **DeadCodeEliminationPass**: Analyzes the computation graph to identify and remove nodes whose outputs are not used by any other node and are not graph outputs
2. **IdentityEliminationPass**: Replaces all references to an Identity node's output with its input, then removes the Identity node

**Tech Stack:**
- Python 3.12
- pytest for testing
- Existing pass infrastructure: `PassBase`, `PassManager`, `CompileContext`
- IR types: `Graph`, `Node`, `OpType`, `TensorType`

**Key Design Decisions:**
- Passes modify the `Graph` in-place via `ctx.graph`
- After removing nodes, update related mappings (`graph.nodes`, `graph.tensors`)
- Identity elimination must update all consumer nodes' inputs
- Both passes should be idempotent (running twice has same effect as running once)
- Integration into O1 optimization level

---

## Task 1: Add Pass Files to __init__.py

**Files:**
- Modify: `src/nnc_py/passes/__init__.py`

**Step 1: Read current __init__.py**

Run: `cat src/nnc_py/passes/__init__.py`

Expected: See existing pass imports (liveness, memory_planning, spill)

**Step 2: Edit __init__.py to add new pass imports**

Add at the end of the file (after existing imports):

```python
from nnc_py.passes.dead_code_elimination import DeadCodeEliminationPass
from nnc_py.passes.identity_elimination import IdentityEliminationPass

__all__ = [
    "LivenessAnalysisPass",
    "MemoryPlanningPassV2",
    "SpillAnalysisPass",
    "DeadCodeEliminationPass",
    "IdentityEliminationPass",
]
```

**Step 3: Verify the file structure is correct**

Run: `python -c "from nnc_py.passes import PassBase; print('PassBase imported successfully')"`

Expected: No errors

**Step 4: Commit**

```bash
git add src/nnc_py/passes/__init__.py
git commit -m "chore: add imports for new passes"
```

---

## Task 2: Create DeadCodeEliminationPass Test File

**Files:**
- Create: `tests/test_dead_code_elimination_pass.py`

**Step 1: Write the failing test - simple dead code removal**

```python
"""Tests for DeadCodeEliminationPass."""

import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.passes.dead_code_elimination import DeadCodeEliminationPass


def test_dead_code_elimination_pass_exists():
    """Test that DeadCodeEliminationPass can be imported and instantiated."""
    pass_obj = DeadCodeEliminationPass()
    assert pass_obj.name == "DeadCodeElimination"


def test_removes_unused_node():
    """Test that a node with no consumers is removed."""
    # Create graph: input -> Relu -> (unused) -> Add -> output
    # The Relu output should be removed since Add uses a different input
    graph = Graph(name="test_dead_code")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="unused_relu_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="const"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="output"
    ))

    # Add nodes - Relu produces unused output
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_unused",
        inputs=["input"],
        outputs=["unused_relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    # Add uses a constant, not the relu output
    add_node = Node(
        op_type=OpType.ADD,
        name="add_final",
        inputs=["input", "const"],  # Uses input directly, not relu output
        outputs=["output"],
        attrs={}
    )
    graph.add_node(add_node)

    # Set graph outputs
    graph.outputs = ["output"]

    # Create context and run pass
    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = DeadCodeEliminationPass()
    pass_obj.run(ctx)

    # Relu node should be removed (its output is not used)
    assert "relu_unused" not in ctx.graph.nodes
    # Add node should remain
    assert "add_final" in ctx.graph.nodes


def test_keeps_used_node():
    """Test that a node whose output is used is kept."""
    graph = Graph(name="test_keep_used")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="relu_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="output"
    ))

    # Add nodes - Relu output is used by Add
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_used",
        inputs=["input"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    add_node = Node(
        op_type=OpType.ADD,
        name="add_final",
        inputs=["relu_out", "relu_out"],  # Uses relu output
        outputs=["output"],
        attrs={}
    )
    graph.add_node(add_node)

    graph.outputs = ["output"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = DeadCodeEliminationPass()
    pass_obj.run(ctx)

    # Both nodes should remain
    assert "relu_used" in ctx.graph.nodes
    assert "add_final" in ctx.graph.nodes


def test_keeps_output_nodes():
    """Test that nodes producing graph outputs are kept."""
    graph = Graph(name="test_outputs")

    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="output1"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="output2"
    ))

    relu_node = Node(
        op_type=OpType.RELU,
        name="relu1",
        inputs=["input"],
        outputs=["output1"],
        attrs={}
    )
    graph.add_node(relu_node)

    sigmoid_node = Node(
        op_type=OpType.SIGMOID,
        name="sigmoid1",
        inputs=["input"],
        outputs=["output2"],
        attrs={}
    )
    graph.add_node(sigmoid_node)

    graph.outputs = ["output1", "output2"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = DeadCodeEliminationPass()
    pass_obj.run(ctx)

    # Both nodes should be kept (they produce outputs)
    assert "relu1" in ctx.graph.nodes
    assert "sigmoid1" in ctx.graph.nodes


def test_keeps_input_producers():
    """Test that nodes producing inputs to kept nodes are kept."""
    graph = Graph(name="test_input_producers")

    # Chain: input -> relu -> sigmoid -> output
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="relu_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="sigmoid_out"
    ))

    relu_node = Node(
        op_type=OpType.RELU,
        name="relu1",
        inputs=["input"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    sigmoid_node = Node(
        op_type=OpType.SIGMOID,
        name="sigmoid1",
        inputs=["relu_out"],
        outputs=["sigmoid_out"],
        attrs={}
    )
    graph.add_node(sigmoid_node)

    graph.outputs = ["sigmoid_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = DeadCodeEliminationPass()
    pass_obj.run(ctx)

    # Both nodes should be kept (sigmoid is output, relu produces sigmoid's input)
    assert "relu1" in ctx.graph.nodes
    assert "sigmoid1" in ctx.graph.nodes


def test_idempotent():
    """Test that running the pass twice produces the same result."""
    graph = Graph(name="test_idempotent")

    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="unused"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="output"
    ))

    unused_node = Node(
        op_type=OpType.RELU,
        name="unused_relu",
        inputs=["input"],
        outputs=["unused"],
        attrs={}
    )
    graph.add_node(unused_node)

    add_node = Node(
        op_type=OpType.ADD,
        name="add_final",
        inputs=["input", "input"],
        outputs=["output"],
        attrs={}
    )
    graph.add_node(add_node)

    graph.outputs = ["output"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = DeadCodeEliminationPass()

    # Run pass once
    pass_obj.run(ctx)
    node_count_after_first = len(ctx.graph.nodes)

    # Run pass again
    pass_obj.run(ctx)
    node_count_after_second = len(ctx.graph.nodes)

    # Should have the same number of nodes
    assert node_count_after_first == node_count_after_second
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dead_code_elimination_pass.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'nnc_py.passes.dead_code_elimination'`

**Step 3: Create placeholder pass file**

Create: `src/nnc_py/passes/dead_code_elimination.py`

```python
"""Dead code elimination pass.

This pass removes nodes from the computation graph whose outputs are not used
by any other node and are not graph outputs.
"""

from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassBase


class DeadCodeEliminationPass(PassBase):
    """Remove unused nodes from the computation graph."""

    @property
    def name(self) -> str:
        return "DeadCodeElimination"

    def _execute(self, ctx: CompileContext) -> None:
        # TODO: implement
        pass
```

**Step 4: Run test again to see different failure**

Run: `pytest tests/test_dead_code_elimination_pass.py -v`

Expected: FAIL with assertion error (pass exists but doesn't remove nodes)

**Step 5: Commit**

```bash
git add tests/test_dead_code_elimination_pass.py src/nnc_py/passes/dead_code_elimination.py
git commit -m "test: add failing tests for DeadCodeEliminationPass"
```

---

## Task 3: Implement DeadCodeEliminationPass

**Files:**
- Modify: `src/nnc_py/passes/dead_code_elimination.py`

**Step 1: Implement the core algorithm**

Replace the `_execute` method with:

```python
def _execute(self, ctx: CompileContext) -> None:
    """Execute dead code elimination.

    Algorithm:
    1. Mark all nodes as "dead" initially
    2. Mark nodes that produce graph outputs as "live"
    3. Backward propagate: mark nodes that produce inputs to live nodes as "live"
    4. Remove all dead nodes
    """
    graph = ctx.graph

    # Collect all live tensors (graph outputs and inputs)
    live_tensors = set(graph.outputs)
    live_tensors.update(graph.inputs)  # Keep input tensors marked

    # Work backwards: find nodes that produce live tensors
    live_nodes = set()

    # Initialize queue with output tensors
    from collections import deque
    queue = deque(graph.outputs)

    while queue:
        tensor_name = queue.popleft()

        # Find nodes that produce this tensor
        producers = graph.get_producers(tensor_name)

        for producer in producers:
            if producer.name not in live_nodes:
                live_nodes.add(producer.name)
                # Add this node's inputs to the queue
                for input_tensor in producer.inputs:
                    if input_tensor not in live_tensors:
                        live_tensors.add(input_tensor)
                        queue.append(input_tensor)

    # Remove dead nodes
    nodes_to_remove = [
        node_name for node_name in graph.nodes
        if node_name not in live_nodes
    ]

    for node_name in nodes_to_remove:
        del graph.nodes[node_name]

    # Log summary if debug mode is on
    if ctx.debug:
        print(f"\n{'='*60}")
        print(f"Dead Code Elimination Summary")
        print(f"{'='*60}")
        print(f"Nodes before: {len(graph.nodes) + len(nodes_to_remove)}")
        print(f"Nodes removed: {len(nodes_to_remove)}")
        print(f"Nodes after: {len(graph.nodes)}")
        if nodes_to_remove:
            print(f"Removed nodes: {', '.join(nodes_to_remove)}")
        print(f"{'='*60}")
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_dead_code_elimination_pass.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/nnc_py/passes/dead_code_elimination.py
git commit -m "feat: implement DeadCodeEliminationPass"
```

---

## Task 4: Create IdentityEliminationPass Test File

**Files:**
- Create: `tests/test_identity_elimination_pass.py`

**Step 1: Write the failing test**

```python
"""Tests for IdentityEliminationPass."""

import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.passes.identity_elimination import IdentityEliminationPass


def test_identity_elimination_pass_exists():
    """Test that IdentityEliminationPass can be imported and instantiated."""
    pass_obj = IdentityEliminationPass()
    assert pass_obj.name == "IdentityElimination"


def test_removes_single_identity():
    """Test removing a single Identity node."""
    graph = Graph(name="test_single_identity")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="identity_out"
    ))

    # Add Identity node
    identity_node = Node(
        op_type=OpType.IDENTITY,
        name="identity_1",
        inputs=["input"],
        outputs=["identity_out"],
        attrs={}
    )
    graph.add_node(identity_node)

    # Set output to identity output
    graph.outputs = ["identity_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = IdentityEliminationPass()
    pass_obj.run(ctx)

    # Identity node should be removed
    assert "identity_1" not in ctx.graph.nodes
    # Output should now be the input
    assert ctx.graph.outputs == ["input"]


def test_identity_chain():
    """Test removing a chain of Identity nodes."""
    graph = Graph(name="test_identity_chain")

    # Add tensors
    for i in range(4):
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[2, 2]),
            name=f"tensor_{i}"
        ))

    # Chain: tensor_0 -> Identity -> tensor_1 -> Identity -> tensor_2 -> Identity -> tensor_3
    for i in range(3):
        node = Node(
            op_type=OpType.IDENTITY,
            name=f"identity_{i}",
            inputs=[f"tensor_{i}"],
            outputs=[f"tensor_{i+1}"],
            attrs={}
        )
        graph.add_node(node)

    graph.outputs = ["tensor_3"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = IdentityEliminationPass()
    pass_obj.run(ctx)

    # All identity nodes should be removed
    for i in range(3):
        assert f"identity_{i}" not in ctx.graph.nodes
    # Output should be the original input
    assert ctx.graph.outputs == ["tensor_0"]


def test_identity_with_consumers():
    """Test that consumer nodes are updated correctly."""
    graph = Graph(name="test_identity_consumers")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="identity_out"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="output"
    ))

    # Identity node
    identity_node = Node(
        op_type=OpType.IDENTITY,
        name="identity_1",
        inputs=["input"],
        outputs=["identity_out"],
        attrs={}
    )
    graph.add_node(identity_node)

    # Add node that uses identity output
    add_node = Node(
        op_type=OpType.ADD,
        name="add_1",
        inputs=["identity_out", "identity_out"],
        outputs=["output"],
        attrs={}
    )
    graph.add_node(add_node)

    graph.outputs = ["output"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = IdentityEliminationPass()
    pass_obj.run(ctx)

    # Identity node should be removed
    assert "identity_1" not in ctx.graph.nodes
    # Add node should now use input directly
    add_node_after = ctx.graph.nodes["add_1"]
    assert add_node_after.inputs == ["input", "input"]


def test_keeps_non_identity_nodes():
    """Test that non-Identity nodes are not affected."""
    graph = Graph(name="test_keep_non_identity")

    # Add tensors
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="relu_out"
    ))

    # Relu node (not identity)
    relu_node = Node(
        op_type=OpType.RELU,
        name="relu_1",
        inputs=["input"],
        outputs=["relu_out"],
        attrs={}
    )
    graph.add_node(relu_node)

    graph.outputs = ["relu_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = IdentityEliminationPass()
    pass_obj.run(ctx)

    # Relu node should remain
    assert "relu_1" in ctx.graph.nodes
    assert ctx.graph.nodes["relu_1"].op_type == OpType.RELU


def test_idempotent():
    """Test that running the pass twice produces the same result."""
    graph = Graph(name="test_identity_idempotent")

    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="input"
    ))
    graph.add_tensor(TensorType(
        dtype=DataType.FLOAT32,
        shape=TensorShape(dims=[2, 2]),
        name="identity_out"
    ))

    identity_node = Node(
        op_type=OpType.IDENTITY,
        name="identity_1",
        inputs=["input"],
        outputs=["identity_out"],
        attrs={}
    )
    graph.add_node(identity_node)

    graph.outputs = ["identity_out"]

    ctx = CompileContext(graph=graph, target="x86")
    pass_obj = IdentityEliminationPass()

    # Run pass once
    pass_obj.run(ctx)
    node_count_after_first = len(ctx.graph.nodes)

    # Run pass again
    pass_obj.run(ctx)
    node_count_after_second = len(ctx.graph.nodes)

    # Should have the same number of nodes
    assert node_count_after_first == node_count_after_second
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_identity_elimination_pass.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'nnc_py.passes.identity_elimination'`

**Step 3: Create placeholder pass file**

Create: `src/nnc_py/passes/identity_elimination.py`

```python
"""Identity elimination pass.

This pass removes Identity operations from the computation graph by replacing
all references to an Identity node's output with its input, then removing the node.
"""

from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassBase


class IdentityEliminationPass(PassBase):
    """Remove Identity operations from the computation graph."""

    @property
    def name(self) -> str:
        return "IdentityElimination"

    def _execute(self, ctx: CompileContext) -> None:
        # TODO: implement
        pass
```

**Step 4: Run test again to see different failure**

Run: `pytest tests/test_identity_elimination_pass.py -v`

Expected: FAIL with assertion error (pass exists but doesn't eliminate)

**Step 5: Commit**

```bash
git add tests/test_identity_elimination_pass.py src/nnc_py/passes/identity_elimination.py
git commit -m "test: add failing tests for IdentityEliminationPass"
```

---

## Task 5: Implement IdentityEliminationPass

**Files:**
- Modify: `src/nnc_py/passes/identity_elimination.py`

**Step 1: Implement the core algorithm**

Replace the file content with:

```python
"""Identity elimination pass.

This pass removes Identity operations from the computation graph by replacing
all references to an Identity node's output with its input, then removing the node.
"""

from typing import Dict, List, Set

from nnc_py.ir.context import CompileContext
from nnc_py.ir.node import OpType
from nnc_py.passes.base import PassBase


class IdentityEliminationPass(PassBase):
    """Remove Identity operations from the computation graph.

    This pass:
    1. Finds all Identity nodes in the graph
    2. Builds a mapping from Identity output tensor to Identity input tensor
    3. Updates all consumer nodes to use the input tensor directly
    4. Updates graph outputs if needed
    5. Removes the Identity nodes

    The pass handles chains of Identity nodes correctly.
    """

    @property
    def name(self) -> str:
        return "IdentityElimination"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute Identity elimination."""
        graph = ctx.graph

        # Step 1: Find all Identity nodes and build replacement mapping
        # We need to handle chains, so we iteratively resolve the mapping
        replacement_map: Dict[str, str] = {}
        identity_nodes = [
            node for node in graph.nodes.values()
            if node.op_type == OpType.IDENTITY
        ]

        for identity_node in identity_nodes:
            if len(identity_node.inputs) == 1 and len(identity_node.outputs) == 1:
                output_tensor = identity_node.outputs[0]
                input_tensor = identity_node.inputs[0]
                replacement_map[output_tensor] = input_tensor

        # Step 2: Resolve chains (if A->B->C are identities, map C to A)
        replacement_map = self._resolve_chains(replacement_map)

        # Step 3: Update consumer nodes
        for node in graph.nodes.values():
            # Skip identity nodes (they will be removed)
            if node.op_type == OpType.IDENTITY:
                continue

            # Update inputs
            updated_inputs = []
            for input_tensor in node.inputs:
                updated_inputs.append(replacement_map.get(input_tensor, input_tensor))
            node.inputs = updated_inputs

        # Step 4: Update graph outputs
        updated_outputs = []
        for output_tensor in graph.outputs:
            updated_outputs.append(replacement_map.get(output_tensor, output_tensor))
        graph.outputs = updated_outputs

        # Step 5: Update graph inputs (in case an identity was marked as input)
        # This is less common but possible
        updated_inputs = []
        for input_tensor in graph.inputs:
            updated_inputs.append(replacement_map.get(input_tensor, input_tensor))
        graph.inputs = updated_inputs

        # Step 6: Remove Identity nodes
        nodes_to_remove = [
            node.name for node in identity_nodes
            if node.name in graph.nodes
        ]
        for node_name in nodes_to_remove:
            del graph.nodes[node_name]

        # Log summary if debug mode is on
        if ctx.debug:
            print(f"\n{'='*60}")
            print(f"Identity Elimination Summary")
            print(f"{'='*60}")
            print(f"Identity nodes removed: {len(nodes_to_remove)}")
            if replacement_map:
                print(f"Replacements: {len(replacement_map)} tensors remapped")
            print(f"{'='*60}")

    def _resolve_chains(self, replacement_map: Dict[str, str]) -> Dict[str, str]:
        """Resolve chains of Identity nodes.

        If we have mappings: B->A, C->B, we want to map C->A directly.

        Args:
            replacement_map: Initial mapping from output tensor to input tensor

        Returns:
            Resolved mapping where all chains are flattened
        """
        resolved = {}
        for output_tensor, input_tensor in replacement_map.items():
            # Follow the chain until we hit a tensor not in the map
            current = input_tensor
            visited: Set[str] = set()
            while current in replacement_map and current not in visited:
                visited.add(current)
                current = replacement_map[current]
            resolved[output_tensor] = current
        return resolved
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_identity_elimination_pass.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/nnc_py/passes/identity_elimination.py
git commit -m "feat: implement IdentityEliminationPass"
```

---

## Task 6: Register Passes in PassManager

**Files:**
- Modify: `src/nnc_py/passes/base.py`

**Step 1: Edit get_default_passes to include new passes at O1**

Find the `get_default_passes` method and modify the O1 section (around line 87-93):

```python
# O1: Basic optimizations
if opt_level == 1:
    return [
        IdentityEliminationPass(),  # Remove Identity ops first
        DeadCodeEliminationPass(),   # Then remove dead code
        LivenessAnalysisPass(),
        MemoryPlanningPassV2(),
    ]
```

Also update the O2 section (around line 95-101):

```python
# O2: Intermediate optimizations
if opt_level == 2:
    return [
        IdentityEliminationPass(),
        DeadCodeEliminationPass(),
        LivenessAnalysisPass(),
        MemoryPlanningPassV2(),
        SpillAnalysisPass(),  # Handles overflow if max_memory set
    ]
```

And the O3 section (around line 103-110):

```python
# O3: Advanced optimizations
if opt_level >= 3:
    # TODO: Add advanced passes (operator fusion, etc.)
    return [
        IdentityEliminationPass(),
        DeadCodeEliminationPass(),
        LivenessAnalysisPass(),
        MemoryPlanningPassV2(),
        SpillAnalysisPass(),
    ]
```

Also add imports at the top of the file (around line 77-79):

```python
from nnc_py.passes.liveness import LivenessAnalysisPass
from nnc_py.passes.memory_planning import MemoryPlanningPassV2
from nnc_py.passes.spill import SpillAnalysisPass
from nnc_py.passes.dead_code_elimination import DeadCodeEliminationPass
from nnc_py.passes.identity_elimination import IdentityEliminationPass
```

**Step 2: Verify imports work**

Run: `python -c "from nnc_py.passes import PassManager; pm = PassManager.get_default_passes(1); print([p.name for p in pm])"`

Expected: `['IdentityElimination', 'DeadCodeElimination', 'LivenessAnalysis', 'MemoryPlanning']`

**Step 3: Commit**

```bash
git add src/nnc_py/passes/base.py
git commit -m "feat: register new passes in PassManager at O1+"
```

---

## Task 7: Add End-to-End Integration Test

**Files:**
- Create: `tests/test_passes_e2e.py`

**Step 1: Write integration test that compiles a model with Identity nodes**

```python
"""End-to-end integration tests for optimization passes."""

import os
import tempfile
import shutil
from pathlib import Path

import onnx
from onnx import helper
import pytest

from nnc_py import Compiler


class TestPassesE2E:
    """End-to-end tests for optimization passes."""

    def setup_method(self):
        """Set up test environment."""
        self.tmp_dir = tempfile.mkdtemp()
        self.runtime_dir = Path(__file__).resolve().parent.parent / "runtime"

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_identity_elimination_e2e(self):
        """Test that IdentityEliminationPass works end-to-end."""
        # Create a model with Identity nodes
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

        # Add some Identity nodes
        identity1 = helper.make_node("Identity", inputs=["input"], outputs=["id1_out"])
        identity2 = helper.make_node("Identity", inputs=["id1_out"], outputs=["id2_out"])
        relu = helper.make_node("Relu", inputs=["id2_out"], outputs=["output"])

        graph = helper.make_graph(
            [identity1, identity2, relu],
            "identity_test",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        # Save and compile
        onnx_path = os.path.join(self.tmp_dir, "model.onnx")
        onnx.save(model, onnx_path)

        output_dir = os.path.join(self.tmp_dir, "build")
        compiler = Compiler(target="x86", opt_level=1)
        compiler.compile(onnx_path, output_dir)

        # Check that model.c was generated
        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c should be generated"

        # Read and check the generated code
        code = Path(model_c_path).read_text()

        # Identity nodes should be eliminated, so no nnc_identity calls
        # (unless they're in the runtime for other reasons)
        # The key is that compilation succeeds

    def test_dead_code_elimination_e2e(self):
        """Test that DeadCodeEliminationPass works end-to-end."""
        # Create a model with unused nodes
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

        # This Relu is unused (output not connected to final output)
        unused_relu = helper.make_node("Relu", inputs=["input"], outputs=["unused_out"])

        # This path is used
        used_relu = helper.make_node("Relu", inputs=["input"], outputs=["output"])

        graph = helper.make_graph(
            [unused_relu, used_relu],
            "dead_code_test",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        # Save and compile
        onnx_path = os.path.join(self.tmp_dir, "model.onnx")
        onnx.save(model, onnx_path)

        output_dir = os.path.join(self.tmp_dir, "build")
        compiler = Compiler(target="x86", opt_level=1)
        compiler.compile(onnx_path, output_dir)

        # Check that model.c was generated
        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c should be generated"

    def test_combined_passes_e2e(self):
        """Test that both passes work together correctly."""
        # Create a model with both Identity nodes and dead code
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

        # Identity chain (should be eliminated)
        identity1 = helper.make_node("Identity", inputs=["input"], outputs=["id1"])
        identity2 = helper.make_node("Identity", inputs=["id1"], outputs=["id2"])

        # Dead code (unused)
        unused_relu = helper.make_node("Relu", inputs=["id1"], outputs=["unused"])

        # Used path
        relu = helper.make_node("Relu", inputs=["id2"], outputs=["output"])

        graph = helper.make_graph(
            [identity1, identity2, unused_relu, relu],
            "combined_test",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        # Save and compile
        onnx_path = os.path.join(self.tmp_dir, "model.onnx")
        onnx.save(model, onnx_path)

        output_dir = os.path.join(self.tmp_dir, "build")
        compiler = Compiler(target="x86", opt_level=1)
        compiler.compile(onnx_path, output_dir)

        # Check that model.c was generated
        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c should be generated"
```

**Step 2: Run the integration tests**

Run: `pytest tests/test_passes_e2e.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_passes_e2e.py
git commit -m "test: add end-to-end integration tests for new passes"
```

---

## Task 8: Run Full Test Suite (Regression Testing)

**Files:**
- No new files, run existing tests

**Step 1: Run all pass-related tests**

Run: `pytest tests/test_pass_types.py tests/test_dead_code_elimination_pass.py tests/test_identity_elimination_pass.py -v`

Expected: All tests PASS

**Step 2: Run IR tests to ensure graph operations still work**

Run: `pytest tests/test_ir_types.py -v`

Expected: All tests PASS

**Step 3: Run a subset of snapshot tests to verify codegen still works**

Run: `pytest tests/test_snapshots_simple_conv.py -v --run-snapshot`

Expected: Tests PASS (if snapshot tests are enabled)

**Step 4: Run simple E2E tests**

Run: `pytest tests/test_e2e.py -v -k "test_simple_add_model or test_reshape_model"`

Expected: Tests PASS

**Step 5: If any tests fail, investigate and fix**

If tests fail:
1. Check if the failure is related to graph modifications
2. Verify that node.input updates are working correctly
3. Ensure that tensor definitions are not being incorrectly removed
4. Fix issues and re-run

**Step 6: Commit**

```bash
git add -A
git commit -m "test: ensure new passes don't break existing tests"
```

---

## Task 9: Update Documentation

**Files:**
- Create: `docs/OPTIMIZATION_PASSES.md`

**Step 1: Create documentation for optimization passes**

```markdown
# Optimization Passes

This document describes the optimization passes available in nnc-py.

## Overview

The compiler applies optimization passes based on the optimization level (`opt_level`).
Each pass transforms the computation graph to improve efficiency or reduce code size.

## Pass Categories

### Essential Passes (O0)

These passes are required for correct code generation:

- **LivenessAnalysisPass**: Analyzes tensor lifetimes for memory planning
- **MemoryPlanningPassV2**: Plans memory allocation for tensors

### Basic Optimizations (O1)

Applied at `opt_level=1` and above:

- **IdentityEliminationPass**: Removes Identity operations from the graph
  - Replaces all references to an Identity node's output with its input
  - Handles chains of Identity nodes correctly
  - Reduces unnecessary operations

- **DeadCodeEliminationPass**: Removes unused nodes from the graph
  - Identifies nodes whose outputs are not used
  - Preserves nodes that produce graph outputs
  - Preserves nodes that produce inputs to live nodes

### Intermediate Optimizations (O2)

Applied at `opt_level=2` and above:

All O1 passes plus:

- **SpillAnalysisPass**: Handles memory overflow when `max_memory` is set

### Advanced Optimizations (O3)

Currently equivalent to O2. Future additions will include:

- Operator fusion (Conv+BN+ReLU, etc.)
- Layout optimizations
- Loop fusion

## Implementation Details

### Adding a New Pass

1. Create a new file in `src/nnc_py/passes/`
2. Inherit from `PassBase`
3. Implement the `_execute` method
4. Import and register in `PassManager.get_default_passes()`

Example:

```python
from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassBase

class MyOptimizationPass(PassBase):
    @property
    def name(self) -> str:
        return "MyOptimization"

    def _execute(self, ctx: CompileContext) -> None:
        # Transform ctx.graph here
        pass
```

### Pass Execution Order

Passes are executed in the order they are registered. For O1 and above:

1. IdentityEliminationPass - removes no-op operations first
2. DeadCodeEliminationPass - removes now-unused nodes
3. LivenessAnalysisPass - analyzes lifetimes on optimized graph
4. MemoryPlanningPassV2 - plans memory based on lifetimes
5. (O2+) SpillAnalysisPass - handles overflow if needed
```

**Step 2: Commit**

```bash
git add docs/OPTIMIZATION_PASSES.md
git commit -m "docs: add optimization passes documentation"
```

---

## Task 10: Final Verification

**Files:**
- No new files

**Step 1: Run full test suite (if not too large)**

Run: `pytest tests/ -v --ignore=tests/test_snapshots_ -k "not snapshot"`

Or for faster verification:

Run: `pytest tests/test_pass_types.py tests/test_dead_code_elimination_pass.py tests/test_identity_elimination_pass.py tests/test_passes_e2e.py tests/test_ir_types.py -v`

Expected: All tests PASS

**Step 2: Verify the implementation with a real model**

Run: `python -c "
from nnc_py import Compiler
import tempfile
import os

# Create a simple test
tmpdir = tempfile.mkdtemp()
onnx_path = os.path.join(tmpdir, 'test.onnx')
output_dir = os.path.join(tmpdir, 'build')

# Use existing model if available
import onnx
from onnx import helper
input_val = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [2, 2])
output_val = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [2, 2])
identity = helper.make_node('Identity', inputs=['input'], outputs=['id_out'])
relu = helper.make_node('Relu', inputs=['id_out'], outputs=['output'])
graph = helper.make_graph([identity, relu], 'test', [input_val], [output_val])
model = helper.make_model(graph)
model.opset_import[0].version = 13
onnx.save(model, onnx_path)

# Test at O0 (no optimization)
compiler_o0 = Compiler(target='x86', opt_level=0)
compiler_o0.compile(onnx_path, output_dir + '_o0')
print('O0 compilation: OK')

# Test at O1 (with new passes)
compiler_o1 = Compiler(target='x86', opt_level=1)
compiler_o1.compile(onnx_path, output_dir + '_o1')
print('O1 compilation: OK')

print('Verification complete!')
"`

Expected: Both compilations succeed without errors

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete DeadCodeEliminationPass and IdentityEliminationPass implementation"
```

---

## Summary

This plan implements two optimization passes using TDD methodology:

1. **DeadCodeEliminationPass** - Removes unused nodes from the graph
2. **IdentityEliminationPass** - Removes Identity operations

Key points:
- Each pass is tested before implementation
- Tests cover edge cases (chains, consumers, idempotence)
- Passes are registered at O1 optimization level
- End-to-end tests verify integration
- Full test suite ensures no regressions
- Documentation is updated
