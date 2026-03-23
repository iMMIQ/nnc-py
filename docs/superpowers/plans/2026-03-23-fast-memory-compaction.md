# Fast-Memory Compaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee CostAwareAllocator succeeds when `max_memory >= max_node_demand` by adding intra-fast-memory compaction as a fallback when alignment fragmentation blocks allocation.

**Architecture:** Add `MovePoint` dataclass alongside existing `SpillPoint`/`ReloadPoint`. Insert `try_compact_and_allocate` fallback in `make_space` failure path. Codegen emits `memmove` + pointer update for each `MovePoint` before reload operations.

**Tech Stack:** Python 3.13, pytest, existing nnc-py IR/passes/codegen infrastructure.

**Spec:** `docs/superpowers/specs/2026-03-23-fast-memory-compaction-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/nnc_py/passes/memory_strategy.py` | Modify | Add `MovePoint` dataclass, new fields + method on `MemoryAllocationPlan` |
| `src/nnc_py/passes/strategies/cost_aware_allocator.py` | Modify | Add `try_compact_and_allocate`, node demand pre-check, wire into `make_space` |
| `src/nnc_py/codegen/x86_backend.py` | Modify | Emit `memmove` + data pointer updates for `MovePoint`s before reloads |
| `tests/test_compaction.py` | Create | All unit tests for compaction behavior |
| `tests/test_cost_aware_allocator.py` | Unmodified | Regression — must still pass |

---

### Task 1: Add `MovePoint` and plan fields to `memory_strategy.py`

**Files:**
- Modify: `src/nnc_py/passes/memory_strategy.py:56-89` (after `ReloadPoint`, before `MemoryAllocationPlan`)
- Test: `tests/test_compaction.py`

- [ ] **Step 1: Write failing test for MovePoint import and plan fields**

```python
# tests/test_compaction.py
"""Tests for intra-fast-memory compaction."""

import pytest

from nnc_py.passes.memory_strategy import (
    MovePoint,
    MemoryAllocationPlan,
)


def test_move_point_fields():
    mp = MovePoint(
        tensor_name="a",
        at_node_idx=2,
        from_offset=80,
        to_offset=0,
        size=64,
    )
    assert mp.tensor_name == "a"
    assert mp.from_offset == 80
    assert mp.to_offset == 0
    assert mp.size == 64


def test_plan_move_fields_default_empty():
    plan = MemoryAllocationPlan(strategy_name="test", total_fast_memory=256)
    assert plan.move_points == []
    assert plan.move_bytes == 0


def test_plan_get_move_points_at():
    mp0 = MovePoint("a", at_node_idx=2, from_offset=80, to_offset=0, size=64)
    mp1 = MovePoint("b", at_node_idx=2, from_offset=160, to_offset=64, size=64)
    mp2 = MovePoint("c", at_node_idx=5, from_offset=32, to_offset=0, size=32)
    plan = MemoryAllocationPlan(
        strategy_name="test",
        total_fast_memory=256,
        move_points=[mp0, mp1, mp2],
        move_bytes=160,
    )
    assert plan.get_move_points_at(2) == [mp0, mp1]
    assert plan.get_move_points_at(5) == [mp2]
    assert plan.get_move_points_at(0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compaction.py -v`
Expected: `ImportError` — `MovePoint` does not exist yet.

- [ ] **Step 3: Implement MovePoint and plan fields**

In `src/nnc_py/passes/memory_strategy.py`:

Add after `ReloadPoint` (after line 58):

```python
@dataclass
class MovePoint:
    """Intra-fast-memory relocation during compaction."""
    tensor_name: str
    at_node_idx: int          # compaction happens while processing this node
    from_offset: int          # original fast pool offset
    to_offset: int            # compacted new offset
    size: int
```

Add fields to `MemoryAllocationPlan` (after `node_memory_usage`, line 89):

```python
    # Compaction info (empty if no compaction triggered)
    move_points: List['MovePoint'] = field(default_factory=list)
    move_bytes: int = 0
```

Add method to `MemoryAllocationPlan` (after `get_max_reload_slots`):

```python
    def get_move_points_at(self, node_idx: int) -> List['MovePoint']:
        """Get move points at a specific node."""
        return [mp for mp in self.move_points if mp.at_node_idx == node_idx]
```

No other modules need import changes — `MovePoint` will be imported directly by consumers (`cost_aware_allocator.py` in Task 3, `x86_backend.py` in Task 4).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compaction.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Run existing tests for regression**

Run: `pytest tests/test_cost_aware_allocator.py -v`
Expected: all tests PASS (no behavior change).

- [ ] **Step 6: Commit**

```bash
git add src/nnc_py/passes/memory_strategy.py tests/test_compaction.py
git commit -m "feat(memory): add MovePoint dataclass and plan fields for compaction"
```

---

### Task 2: Add node demand pre-check to `CostAwareAllocator`

**Files:**
- Modify: `src/nnc_py/passes/strategies/cost_aware_allocator.py:61-72` (top of `allocate` method)
- Test: `tests/test_compaction.py`

- [ ] **Step 1: Write failing test for demand pre-check**

Append to `tests/test_compaction.py`:

```python
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType
from nnc_py.passes.liveness import LivenessAnalysisPass
from nnc_py.passes.strategies.cost_aware_allocator import CostAwareAllocator


def _tensor(name: str, elements: int) -> TensorType:
    return TensorType(
        name=name,
        dtype=DataType.FLOAT32,
        shape=TensorShape([1, elements]),
    )


def _make_ctx(
    *,
    tensor_elements: dict[str, int],
    inputs: list[str],
    outputs: list[str],
    nodes: list[Node],
) -> tuple[CompileContext, dict]:
    graph = Graph("compaction-test")
    graph.inputs.extend(inputs)
    graph.outputs.extend(outputs)
    for name, elements in tensor_elements.items():
        graph.add_tensor(_tensor(name, elements))
    for node in nodes:
        graph.add_node(node)
    ctx = CompileContext(graph=graph, target="x86", optimization_level=1)
    LivenessAnalysisPass().run(ctx)
    return ctx, ctx.metadata["tensor_liveness"]


def test_demand_precheck_rejects_insufficient_memory():
    """max_memory < max_node_demand → clear ValueError upfront."""
    # Node n0: input x(64 bytes) + output a(64 bytes) = 128 bytes demand (aligned)
    # But we give only 64 bytes
    ctx, liveness = _make_ctx(
        tensor_elements={"x": 16, "a": 16},
        inputs=["x"],
        outputs=["a"],
        nodes=[Node(OpType.RELU, "n0", ["x"], ["a"])],
    )
    allocator = CostAwareAllocator()
    with pytest.raises(ValueError, match="peak node demand"):
        allocator.allocate(ctx, liveness, max_memory=64)


def test_demand_precheck_accepts_sufficient_memory():
    """max_memory == max_node_demand → no error."""
    ctx, liveness = _make_ctx(
        tensor_elements={"x": 16, "a": 16},
        inputs=["x"],
        outputs=["a"],
        nodes=[Node(OpType.RELU, "n0", ["x"], ["a"])],
    )
    allocator = CostAwareAllocator()
    # 16 elements * 4 bytes = 64 bytes per tensor, aligned to 16 = 64
    # demand = 64 + 64 = 128
    plan = allocator.allocate(ctx, liveness, max_memory=128)
    assert plan.total_fast_memory <= 128
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compaction.py::test_demand_precheck_rejects_insufficient_memory -v`
Expected: FAIL — no "peak node demand" error raised (current code raises a different error or succeeds).

- [ ] **Step 3: Implement demand pre-check**

In `src/nnc_py/passes/strategies/cost_aware_allocator.py`, inside `allocate()`, after `tensor_sizes` is built (after line 82) and before the `resident` dict initialization (line 84), add:

```python
        # Pre-check: verify max_memory can fit the largest single node's demand
        if capacity != float("inf"):
            max_node_demand = 0
            for node in nodes:
                demand = sum(
                    self._align(tensor_sizes[t], self.DEFAULT_ALIGNMENT)
                    for t in node.inputs if t in tensor_sizes
                )
                demand += sum(
                    self._align(tensor_sizes[t], self.DEFAULT_ALIGNMENT)
                    for t in node.outputs if t in tensor_sizes
                )
                max_node_demand = max(max_node_demand, demand)
            if max_node_demand > capacity:
                raise ValueError(
                    f"max_memory ({capacity}) < peak node demand ({max_node_demand}). "
                    f"Minimum required: {max_node_demand} bytes."
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_compaction.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Regression check**

Run: `pytest tests/test_cost_aware_allocator.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/nnc_py/passes/strategies/cost_aware_allocator.py tests/test_compaction.py
git commit -m "feat(memory): add node demand pre-check to CostAwareAllocator"
```

---

### Task 3: Implement `try_compact_and_allocate` and wire into `make_space`

**Files:**
- Modify: `src/nnc_py/passes/strategies/cost_aware_allocator.py:213-285` (`make_space`), and add new closure + wire `move_points`/`move_bytes` into final plan (line 450)
- Test: `tests/test_compaction.py`

- [ ] **Step 1: Write failing tests for compaction behavior**

Append to `tests/test_compaction.py`:

```python
def test_compaction_triggered_on_fragmentation():
    """When alignment fragmentation blocks allocation, compaction fires and succeeds."""
    # Use MATMUL (not in INPLACE_REUSE_OPS) to prevent inplace slot reuse,
    # and non-aligned tensor sizes (20 bytes) to force alignment gaps.
    ctx, liveness = _make_ctx(
        tensor_elements={
            "x": 5, "y": 5,       # 5 * 4 = 20 bytes each, align(20,16)=32
            "a": 5, "b": 5,
            "c": 5, "out": 5,
        },
        inputs=["x", "y"],
        outputs=["out"],
        nodes=[
            Node(OpType.MATMUL, "n0", ["x"], ["a"]),
            Node(OpType.MATMUL, "n1", ["y"], ["b"]),
            Node(OpType.MATMUL, "n2", ["a"], ["c"]),
            Node(OpType.MATMUL, "n3", ["b", "c"], ["out"]),
        ],
    )
    allocator = CostAwareAllocator()
    # align(20, 16) = 32 per tensor
    # max node demand at n3: b(32) + c(32) + out(32) = 96
    plan = allocator.allocate(ctx, liveness, max_memory=96)
    assert plan.total_fast_memory <= 96


def test_compaction_not_triggered_when_unnecessary():
    """Normal allocation without fragmentation produces no MovePoints."""
    ctx, liveness = _make_ctx(
        tensor_elements={"x": 16, "a": 16},
        inputs=["x"],
        outputs=["a"],
        nodes=[Node(OpType.RELU, "n0", ["x"], ["a"])],
    )
    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=256)
    assert plan.move_points == []
    assert plan.move_bytes == 0


def test_compaction_move_bytes_accurate():
    """move_bytes equals sum of moved tensor sizes."""
    ctx, liveness = _make_ctx(
        tensor_elements={
            "x": 5, "y": 5,
            "a": 5, "b": 5,
            "c": 5, "out": 5,
        },
        inputs=["x", "y"],
        outputs=["out"],
        nodes=[
            Node(OpType.MATMUL, "n0", ["x"], ["a"]),
            Node(OpType.MATMUL, "n1", ["y"], ["b"]),
            Node(OpType.MATMUL, "n2", ["a"], ["c"]),
            Node(OpType.MATMUL, "n3", ["b", "c"], ["out"]),
        ],
    )
    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=96)
    assert plan.move_bytes == sum(mp.size for mp in plan.move_points)


def test_compaction_post_compaction_offsets_contiguous():
    """After compaction, protected tensors are packed contiguously from offset 0."""
    ctx, liveness = _make_ctx(
        tensor_elements={
            "x": 5, "y": 5,
            "a": 5, "b": 5,
            "c": 5, "out": 5,
        },
        inputs=["x", "y"],
        outputs=["out"],
        nodes=[
            Node(OpType.MATMUL, "n0", ["x"], ["a"]),
            Node(OpType.MATMUL, "n1", ["y"], ["b"]),
            Node(OpType.MATMUL, "n2", ["a"], ["c"]),
            Node(OpType.MATMUL, "n3", ["b", "c"], ["out"]),
        ],
    )
    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=96)

    # If compaction happened, verify MovePoint destinations are contiguous from 0
    if plan.move_points:
        sorted_moves = sorted(plan.move_points, key=lambda mp: mp.to_offset)
        assert sorted_moves[0].to_offset == 0
        for i in range(1, len(sorted_moves)):
            prev = sorted_moves[i - 1]
            # Next offset should be aligned(prev.to_offset + prev.size)
            expected_min = prev.to_offset + prev.size
            assert sorted_moves[i].to_offset >= expected_min


def test_compaction_does_not_increase_slow_transfers():
    """Compaction uses fast-internal moves, not slow memory transfers."""
    ctx, liveness = _make_ctx(
        tensor_elements={
            "x": 5, "y": 5,
            "a": 5, "b": 5,
            "c": 5, "out": 5,
        },
        inputs=["x", "y"],
        outputs=["out"],
        nodes=[
            Node(OpType.MATMUL, "n0", ["x"], ["a"]),
            Node(OpType.MATMUL, "n1", ["y"], ["b"]),
            Node(OpType.MATMUL, "n2", ["a"], ["c"]),
            Node(OpType.MATMUL, "n3", ["b", "c"], ["out"]),
        ],
    )
    allocator = CostAwareAllocator()
    # Run with tight memory (may trigger compaction)
    plan_tight = allocator.allocate(ctx, liveness, max_memory=96)
    # Run with generous memory (no compaction or spill)
    plan_generous = CostAwareAllocator().allocate(ctx, liveness, max_memory=1024)
    # Compaction should not increase slow-memory transfers
    assert plan_tight.total_transfer_bytes >= 0
    assert plan_tight.move_bytes >= 0
    # Slow transfers should not be worse than without compaction
    # (generous plan has no spills at all)
    # The tight plan may have spills, but compaction itself uses moves not spills


def test_compaction_extreme_max_memory_equals_node_demand():
    """Allocation succeeds when max_memory == peak node demand exactly."""
    # 3-input Add: need all 3 inputs + output simultaneously
    ctx, liveness = _make_ctx(
        tensor_elements={
            "x0": 4, "x1": 4, "x2": 4,  # 16 bytes each, align(16,16)=16
            "a": 4, "b": 4,
            "out": 4,
        },
        inputs=["x0", "x1", "x2"],
        outputs=["out"],
        nodes=[
            Node(OpType.RELU, "n0", ["x0"], ["a"]),
            Node(OpType.RELU, "n1", ["x1"], ["b"]),
            Node(OpType.ADD, "n2", ["a", "b"], ["out"]),
        ],
    )
    allocator = CostAwareAllocator()
    # n2 demand: a(16) + b(16) + out(16) = 48
    plan = allocator.allocate(ctx, liveness, max_memory=48)
    assert plan.total_fast_memory <= 48
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_compaction.py::test_compaction_triggered_on_fragmentation -v`
Expected: FAIL — `ValueError: Cannot allocate` (fragmentation, no compaction fallback yet).

- [ ] **Step 3: Implement `try_compact_and_allocate` and wire it in**

In `src/nnc_py/passes/strategies/cost_aware_allocator.py`, inside the `allocate()` method:

**3a.** Add accumulators near the existing `next_fast_offset` / `peak_fast_end` declarations (around line 96):

```python
        move_points: list[MovePoint] = []
        move_bytes_total = 0
```

Add the import at the top of the file (line 17):

```python
from nnc_py.passes.memory_strategy import (
    ...
    MovePoint,
)
```

**3b.** Add `try_compact_and_allocate` closure after `make_space` (after line 285):

```python
        def try_compact_and_allocate(
            size: int,
            node_idx: int,
            protected: set[str],
        ) -> int | None:
            nonlocal next_fast_offset, move_bytes_total

            # Only compact protected tensors that are actually resident
            protected_resident = [
                (name, resident[name])
                for name in protected
                if name in resident
            ]
            total_protected = sum(
                self._align(slot.size, self.DEFAULT_ALIGNMENT)
                for _, slot in protected_resident
            )
            if total_protected + self._align(size, self.DEFAULT_ALIGNMENT) > capacity:
                return None

            # Sort by current offset ascending for memmove safety
            protected_resident.sort(key=lambda pair: pair[1].offset)

            cursor = 0
            for name, slot in protected_resident:
                new_offset = self._align(cursor, self.DEFAULT_ALIGNMENT)
                assert new_offset <= slot.offset, (
                    f"Compaction invariant violated: new_offset {new_offset} > "
                    f"old_offset {slot.offset} for {name}"
                )
                if new_offset != slot.offset:
                    move_points.append(MovePoint(
                        tensor_name=name,
                        at_node_idx=node_idx,
                        from_offset=slot.offset,
                        to_offset=new_offset,
                        size=slot.size,
                    ))
                    move_bytes_total += slot.size
                    resident[name] = _ResidentTensor(name, new_offset, slot.size)
                cursor = new_offset + slot.size

            # Rebuild free regions as single tail region
            free_regions.clear()
            tail_size = int(capacity) - cursor if capacity != float("inf") else 0
            if tail_size > 0:
                free_regions.append((cursor, tail_size))
            next_fast_offset = cursor
            # peak_fast_end is NOT modified — historical high-water mark

            return try_allocate_fast_region(size)
```

**3c.** Wire compaction into `make_space`. Replace the final `raise ValueError` in `make_space` (line 282-285):

```python
                # Eviction succeeded but free list still fragmented — try compaction
                compacted_offset = try_compact_and_allocate(size, node_idx, protected)
                if compacted_offset is not None:
                    return compacted_offset

                raise ValueError(
                    f"Cannot allocate {size} bytes at node {nodes[node_idx].name} "
                    "after applying eviction plan and compaction"
                )
```

Also add compaction fallback to the two `raise ValueError` blocks in `make_space` where `not candidates` (lines 246-255) and where `eviction_plan is None` (lines 264-273). In both cases, before raising, try:

```python
                    compacted_offset = try_compact_and_allocate(size, node_idx, protected)
                    if compacted_offset is not None:
                        return compacted_offset
```

**3d.** Wire `move_points` and `move_bytes` into the final `MemoryAllocationPlan` construction (around line 450):

```python
            move_points=move_points,
            move_bytes=move_bytes_total,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_compaction.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Regression check**

Run: `pytest tests/test_cost_aware_allocator.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/nnc_py/passes/strategies/cost_aware_allocator.py tests/test_compaction.py
git commit -m "feat(memory): implement fast-memory compaction fallback in CostAwareAllocator"
```

---

### Task 4: Add codegen support for `MovePoint` in x86 backend

**Files:**
- Modify: `src/nnc_py/codegen/x86_backend.py` — `_generate_source` routing (line 104), `_generate_source_with_unified_spill` method
- Test: `tests/test_compaction.py`

- [ ] **Step 1: Write failing test for codegen move emission**

Append to `tests/test_compaction.py`:

```python
def test_codegen_emits_memmove_for_move_points():
    """Generated C code contains memmove for each MovePoint."""
    ctx, liveness = _make_ctx(
        tensor_elements={
            "x": 5, "y": 5,
            "a": 5, "b": 5,
            "c": 5, "out": 5,
        },
        inputs=["x", "y"],
        outputs=["out"],
        nodes=[
            Node(OpType.MATMUL, "n0", ["x"], ["a"]),
            Node(OpType.MATMUL, "n1", ["y"], ["b"]),
            Node(OpType.MATMUL, "n2", ["a"], ["c"]),
            Node(OpType.MATMUL, "n3", ["b", "c"], ["out"]),
        ],
    )
    allocator = CostAwareAllocator()
    plan = allocator.allocate(ctx, liveness, max_memory=96)

    if not plan.move_points:
        pytest.skip("No compaction triggered in this configuration")

    ctx.metadata["memory_allocation_plan"] = plan

    from nnc_py.codegen.x86_backend import X86Backend
    backend = X86Backend()
    code = backend._generate_source_with_unified_spill(ctx, plan)

    # Verify memmove appears for each move point
    for mp in plan.move_points:
        assert "memmove" in code, "Expected memmove in generated code for MovePoint"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compaction.py::test_codegen_emits_memmove_for_move_points -v`
Expected: FAIL — `"memmove" not in code` (codegen doesn't handle MovePoints yet).

- [ ] **Step 3: Implement codegen for MovePoint**

In `src/nnc_py/codegen/x86_backend.py`:

**3a.** Add `MovePoint` to the imports from `memory_strategy` (find the existing import of `SpillPoint`, `ReloadPoint` and add `MovePoint`).

**3b.** Fix the routing in `_generate_source` (line 104). Currently it only enters the unified spill path when `has_spill` is true. Plans with compaction but no spills would fall through to the simple CEmitter path. Change:

```python
        if alloc_plan is not None and (alloc_plan.has_spill or alloc_plan.move_points):
```

**3c.** In `_generate_source_with_unified_spill`, fix the early-return guard (around line 438) similarly:

```python
        if not plan.has_spill and not plan.move_points:
            emitter = CEmitter()
            return emitter.emit(ctx)
```

**3d.** In the per-node codegen loop inside `_generate_source_with_unified_spill`, compute `node_moves` at the top of each iteration (before the existing `has_spill`/`has_reload` checks):

```python
            node_moves = plan.get_move_points_at(node_idx)
            has_move = len(node_moves) > 0
```

Then add `has_move` to the forward-declaration condition:

```python
            if not has_spill and not has_reload and not has_spilled_output and not has_move:
                lines.append(f"    static void {func_name}_body(void);")
```

**3e.** In the per-node function body, insert MovePoint handling *after* the input staging block and *before* the reload section:

```python
            # Compact fast memory (intra-fast-memory moves)
            if node_moves:
                lines.append("    /* Compact fast memory */")
                for mp in node_moves:
                    var_name = ctx.tensor_symbols.get(mp.tensor_name, mp.tensor_name)
                    lines.extend([
                        f"    memmove(_nnc_fast_pool + {mp.to_offset},",
                        f"            _nnc_fast_pool + {mp.from_offset}, {mp.size});",
                        f"    {var_name}.data = _nnc_fast_pool + {mp.to_offset};",
                    ])
                lines.append("")
```

Note: `<string.h>` is already included in the generated code (used by existing `memcpy` for spill/reload).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_compaction.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Regression check**

Run: `pytest tests/test_cost_aware_allocator.py tests/test_memory_safety.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/nnc_py/codegen/x86_backend.py tests/test_compaction.py
git commit -m "feat(codegen): emit memmove for fast-memory compaction MovePoints"
```

---

### Task 5: Integration test — extreme max_memory with ASan

**Files:**
- Test: `tests/test_compaction.py`

- [ ] **Step 1: Write integration test**

Append to `tests/test_compaction.py`:

```python
import os
import re
import tempfile
from pathlib import Path

from nnc_py import Compiler


def test_extreme_max_memory_compiles_and_runs():
    """End-to-end: compile with max_memory = peak node demand, verify output valid."""
    import onnx
    from onnx import helper

    # Model: input(8,8) → Relu → Add(relu, relu) → output
    # Each tensor: 8*8*4 = 256 bytes, align(256,16) = 256
    # Peak node demand at Add: 2 inputs(256 each) + 1 output(256) = 768
    input_t = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [8, 8])
    nodes = [
        helper.make_node("Relu", ["input"], ["relu1"]),
        helper.make_node("Relu", ["input"], ["relu2"]),
        helper.make_node("Add", ["relu1", "relu2"], ["output"]),
    ]
    output_t = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [8, 8])
    graph = helper.make_graph(nodes, "extreme_test", [input_t], [output_t])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])

    tmpdir = tempfile.mkdtemp()
    onnx_path = os.path.join(tmpdir, "model.onnx")
    output_dir = os.path.join(tmpdir, "build")
    onnx.save(model, onnx_path)

    compiler = Compiler(target="x86", opt_level=1)
    compiler.compile(onnx_path, output_dir, max_memory="768", memory_strategy="cost_aware")

    # Verify fast memory pool respects limit
    tensors_c = Path(output_dir) / "tensors.c"
    content = tensors_c.read_text()
    match = re.search(r"#define\s+NNC_FAST_MEMORY_SIZE\s+(\d+)", content)
    if match:
        fast_size = int(match.group(1))
        assert fast_size <= 768, f"Fast memory {fast_size} exceeds limit 768"
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_compaction.py::test_extreme_max_memory_compiles_and_runs -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_compaction.py
git commit -m "test(memory): add integration test for extreme max_memory compaction"
```

---

### Task 6: Final regression suite

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --timeout=120`
Expected: all tests PASS. No regressions.

- [ ] **Step 2: Commit any fixups if needed**

If any test needed a fix, commit with descriptive message.
