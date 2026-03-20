# Cost-Aware Memory Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the existing conservative memory allocator for `O0`, add a new cost-aware allocator for `O1/O2/O3`, and optimize memory planning around minimizing `spill_bytes + reload_bytes` under a user-provided fast-memory limit.

**Architecture:** Extend liveness data to expose use positions and next-use information, then add a new `cost_aware` memory allocation strategy that treats fast memory as a constrained cache with a reusable free-list. Keep the unified `MemoryAllocationPlan` contract, teach the planner/backend to consume allocator-produced spill/reload plans directly, and leave the legacy `basic` strategy intact for `O0`.

**Tech Stack:** Python 3.10+, existing pass framework, `dataclasses`, existing unified memory-planning interfaces, `pytest`, benchmark harness under `benchmarks/`

---

## File Structure

**Files to create**

- `src/nnc_py/passes/strategies/cost_aware_allocator.py`
  - Cost-aware allocator implementation
- `tests/test_liveness_analysis.py`
  - Tests for use-position and next-use metadata
- `tests/test_cost_aware_allocator.py`
  - Unit tests for eviction, reuse, and transfer accounting
- `tests/test_memory_strategy_selection.py`
  - Tests for `O0=basic`, `O1/O2/O3=cost_aware` default behavior

**Files to modify**

- `src/nnc_py/passes/liveness.py`
  - Extend liveness result structure with use-position metadata
- `src/nnc_py/passes/memory_strategy.py`
  - Add new strategy enum value and transfer accounting fields
- `src/nnc_py/passes/memory_planning.py`
  - Choose default strategy by optimization level and preserve unified plan shape
- `src/nnc_py/passes/base.py`
  - Keep pass order but ensure planner behavior is compatible with `O1/O2/O3`
- `src/nnc_py/passes/spill.py`
  - Short-circuit or adapt legacy spill analysis when unified allocator already emitted spill/reload
- `src/nnc_py/compiler.py`
  - Ensure explicit strategy override still works and default behavior remains deterministic
- `src/nnc_py/passes/__init__.py`
  - Re-export new strategy types if needed
- `tests/test_memory_limit_enforcement.py`
  - Adjust assertions to validate transfer-aware unified planning semantics
- `tests/test_spill_verification.py`
  - Cover `cost_aware` flow under constrained memory
- `tests/test_spill_correctness.py`
  - Verify correctness when unified allocator emits spill/reload directly

## Task 1: Extend Liveness With Use Metadata

**Files:**
- Create: `tests/test_liveness_analysis.py`
- Modify: `src/nnc_py/passes/liveness.py`

- [ ] **Step 1: Write the failing liveness tests**

```python
from onnx import TensorProto, helper

from nnc_py.ir.context import CompileContext
from nnc_py.frontend.onnx_loader import ONNXFrontend
from nnc_py.passes.liveness import LivenessAnalysisPass


def _run_liveness(model):
    ctx = CompileContext(graph=model, target="x86", optimization_level=0)
    LivenessAnalysisPass().run(ctx)
    return ctx.metadata["tensor_liveness"]


def test_tensor_liveness_tracks_use_positions_for_branching_graph(tmp_path):
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["input"], ["left"]),
            helper.make_node("Sigmoid", ["input"], ["right"]),
            helper.make_node("Add", ["left", "right"], ["output"]),
        ],
        "branch",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    path = tmp_path / "branch.onnx"
    import onnx
    onnx.save(model, path)

    frontend = ONNXFrontend()
    graph = frontend.load(str(path))
    liveness = _run_liveness(graph)

    assert liveness["left"].use_positions == [2]
    assert liveness["right"].use_positions == [2]
    assert liveness["input"].use_positions == [0, 1]


def test_tensor_liveness_reports_next_use_after():
    from nnc_py.passes.liveness import TensorLiveness

    info = TensorLiveness(
        tensor_name="x",
        live_start=0,
        live_end=5,
        use_positions=[1, 3, 5],
    )

    assert info.next_use_after(0) == 1
    assert info.next_use_after(1) == 3
    assert info.next_use_after(4) == 5
    assert info.next_use_after(5) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_liveness_analysis.py -v`
Expected: FAIL with missing `use_positions` or `next_use_after`

- [ ] **Step 3: Write minimal liveness implementation**

```python
@dataclass
class TensorLiveness:
    tensor_name: str
    live_start: int
    live_end: int
    use_positions: list[int] = field(default_factory=list)
    is_input: bool = False
    is_output: bool = False
    is_constant: bool = False

    def next_use_after(self, node_idx: int) -> int | None:
        for use_idx in self.use_positions:
            if use_idx > node_idx:
                return use_idx
        return None

    def remaining_uses_after(self, node_idx: int) -> int:
        return sum(1 for use_idx in self.use_positions if use_idx > node_idx)
```

Update `_analyze_tensor()` to derive `use_positions` from `graph.get_consumers(tensor_name)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_liveness_analysis.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_liveness_analysis.py src/nnc_py/passes/liveness.py
git commit -m "feat(liveness): track tensor use positions"
```

## Task 2: Extend Memory Strategy Schema For Cost-Aware Planning

**Files:**
- Create: `tests/test_memory_strategy_selection.py`
- Modify: `src/nnc_py/passes/memory_strategy.py`
- Modify: `src/nnc_py/passes/memory_planning.py`
- Modify: `src/nnc_py/compiler.py`
- Modify: `src/nnc_py/passes/__init__.py`

- [ ] **Step 1: Write the failing strategy-selection tests**

```python
from nnc_py.ir.context import CompileContext
from nnc_py.passes.memory_planning import MemoryPlanningPassV2
from nnc_py.passes.memory_strategy import AllocationStrategy


def test_default_strategy_is_basic_for_o0():
    ctx = CompileContext(graph=None, target="x86", optimization_level=0)
    planner = MemoryPlanningPassV2()
    strategy = planner._get_strategy_for_context(ctx, None)
    assert strategy.name == "basic"


def test_default_strategy_is_cost_aware_for_o1_and_above():
    for level in (1, 2, 3):
        ctx = CompileContext(graph=None, target="x86", optimization_level=level)
        planner = MemoryPlanningPassV2()
        strategy = planner._get_strategy_for_context(ctx, None)
        assert strategy.name == "cost_aware"


def test_explicit_memory_strategy_override_is_preserved():
    ctx = CompileContext(graph=None, target="x86", optimization_level=3)
    ctx.metadata["memory_strategy"] = "basic"
    planner = MemoryPlanningPassV2()
    strategy = planner._get_strategy_for_context(ctx, "basic")
    assert strategy.name == "basic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_strategy_selection.py -v`
Expected: FAIL with missing helper or wrong default strategy

- [ ] **Step 3: Extend the memory strategy schema**

Add a new strategy enum and new plan fields in `src/nnc_py/passes/memory_strategy.py`:

```python
class AllocationStrategy(Enum):
    BASIC = "basic"
    COST_AWARE = "cost_aware"


@dataclass
class MemoryAllocationPlan:
    strategy_name: str
    total_fast_memory: int
    total_slow_memory: int = 0
    peak_memory: int = 0
    num_buffers: int = 0
    spill_bytes: int = 0
    reload_bytes: int = 0
    total_transfer_bytes: int = 0
```

Add a planner helper in `src/nnc_py/passes/memory_planning.py`:

```python
def _get_strategy_for_context(self, ctx: CompileContext, strategy_config):
    if strategy_config is not None:
        return self._get_strategy(strategy_config)
    default_name = "basic" if ctx.optimization_level == 0 else "cost_aware"
    return self._get_strategy(default_name)
```

Use this helper inside `_execute()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_strategy_selection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_memory_strategy_selection.py src/nnc_py/passes/memory_strategy.py src/nnc_py/passes/memory_planning.py src/nnc_py/compiler.py src/nnc_py/passes/__init__.py
git commit -m "feat(memory): add cost-aware strategy selection"
```

## Task 3: Implement Cost-Aware Allocator Core

**Files:**
- Create: `src/nnc_py/passes/strategies/cost_aware_allocator.py`
- Create: `tests/test_cost_aware_allocator.py`
- Modify: `src/nnc_py/passes/memory_strategy.py`

- [ ] **Step 1: Write the failing allocator unit tests**

```python
from nnc_py.passes.strategies.cost_aware_allocator import CostAwareAllocator


def test_cost_aware_allocator_reuses_freed_fast_regions():
    allocator = CostAwareAllocator()
    free_regions = [(0, 1024)]
    offset = allocator._allocate_from_free_list(free_regions, 256, 16)
    assert offset == 0
    assert free_regions == [(256, 768)]


def test_cost_aware_allocator_evicts_farthest_small_tensor_first():
    allocator = CostAwareAllocator()
    candidates = [
        {"tensor_name": "near_big", "size": 1024, "next_use_distance": 1},
        {"tensor_name": "far_small", "size": 128, "next_use_distance": 50},
    ]
    ordered = allocator._sort_eviction_candidates(candidates)
    assert ordered[0]["tensor_name"] == "far_small"


def test_cost_aware_allocator_tracks_transfer_bytes_on_spill_and_reload():
    allocator = CostAwareAllocator()
    allocator._record_spill("x", size=256)
    allocator._record_reload("x", size=256)
    assert allocator._spill_bytes == 256
    assert allocator._reload_bytes == 256
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cost_aware_allocator.py -v`
Expected: FAIL with missing allocator implementation

- [ ] **Step 3: Write minimal allocator scaffolding**

Create `src/nnc_py/passes/strategies/cost_aware_allocator.py` with:

```python
class CostAwareAllocator(MemoryAllocationStrategy):
    DEFAULT_ALIGNMENT = 16

    @property
    def name(self) -> str:
        return "cost_aware"

    @property
    def strategy_type(self) -> AllocationStrategy:
        return AllocationStrategy.COST_AWARE

    def _evict_score(self, *, next_use_distance: int | None, size: int) -> float:
        if next_use_distance is None:
            return float("inf")
        return next_use_distance / max(size, 1)
```

Also add simple free-list helpers and byte counters.

- [ ] **Step 4: Expand the allocator to produce a real unified plan**

Implement an event-driven scan that:

- reloads required inputs before node execution
- frees tensors whose last use has completed
- allocates outputs into reusable fast regions
- evicts currently resident tensors by highest eviction score when needed
- emits `spill_points`, `reload_points`, `spill_bytes`, `reload_bytes`, `total_transfer_bytes`

Minimal eviction candidate payload:

```python
candidate = {
    "tensor_name": tensor_name,
    "size": alloc.size,
    "next_use_distance": next_use_distance,
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_cost_aware_allocator.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_cost_aware_allocator.py src/nnc_py/passes/strategies/cost_aware_allocator.py src/nnc_py/passes/memory_strategy.py
git commit -m "feat(memory): add cost-aware allocator"
```

## Task 4: Integrate Unified Spill/Reload Planning

**Files:**
- Modify: `src/nnc_py/passes/spill.py`
- Modify: `src/nnc_py/codegen/x86_backend.py`
- Modify: `tests/test_memory_limit_enforcement.py`
- Modify: `tests/test_spill_verification.py`
- Modify: `tests/test_spill_correctness.py`

- [ ] **Step 1: Write the failing integration tests for unified spill handling**

Add assertions like:

```python
def test_cost_aware_plan_emits_transfer_metrics_under_memory_limit(...):
    compiler = Compiler(target="x86", opt_level=2)
    compiler.compile(onnx_path, output_dir, max_memory="2KB")
    # inspect generated artifacts or plan-facing metadata helper
    assert transfer_bytes > 0


def test_cost_aware_spill_path_keeps_runtime_output_correct(...):
    # compile constrained model with opt_level=2
    # build and execute
    # compare output to numpy or ORT reference
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_limit_enforcement.py tests/test_spill_verification.py tests/test_spill_correctness.py -v`
Expected: FAIL because the legacy spill flow still owns constrained-memory behavior

- [ ] **Step 3: Make legacy spill analysis coexist with unified plans**

In `src/nnc_py/passes/spill.py`, add an early return path:

```python
alloc_plan = ctx.metadata.get("memory_allocation_plan")
if alloc_plan is not None and alloc_plan.has_spill:
    ctx.metadata["spill_plan"] = None
    return
```

This prevents `SpillAnalysisPass` from re-planning spill/reload on top of allocator-produced spill decisions.

- [ ] **Step 4: Verify the backend uses unified spill/reload codegen**

Keep `src/nnc_py/codegen/x86_backend.py` on the unified path:

```python
if alloc_plan is not None and alloc_plan.has_spill:
    source = self._generate_source_with_unified_spill(ctx, alloc_plan)
```

Add or update tests to ensure constrained-memory builds from `cost_aware` generate fast/slow pool accesses correctly.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_memory_limit_enforcement.py tests/test_spill_verification.py tests/test_spill_correctness.py -v`
Expected: PASS or environment-appropriate skips for optional runtime checks

- [ ] **Step 6: Commit**

```bash
git add src/nnc_py/passes/spill.py src/nnc_py/codegen/x86_backend.py tests/test_memory_limit_enforcement.py tests/test_spill_verification.py tests/test_spill_correctness.py
git commit -m "feat(memory): unify cost-aware spill planning"
```

## Task 5: End-To-End Verification And Benchmark Comparison

**Files:**
- Modify: `tests/test_memory_limit_enforcement.py`
- Modify: `tests/test_spill_verification.py`
- Modify: `tests/test_spill_correctness.py`
- Reference: `benchmarks/harness.py`

- [ ] **Step 1: Add a regression test that compares transfer cost against the basic strategy**

```python
def test_cost_aware_transfer_bytes_do_not_exceed_basic(tmp_path):
    # compile the same constrained model twice:
    # 1. memory_strategy="basic"
    # 2. default O2 cost-aware path
    # assert cost-aware total_transfer_bytes <= basic total_transfer_bytes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_limit_enforcement.py::test_cost_aware_transfer_bytes_do_not_exceed_basic -v`
Expected: FAIL until plan metrics are surfaced and comparable

- [ ] **Step 3: Surface transfer metrics for test/benchmark inspection**

If needed, add lightweight helpers in the planner layer so tests can inspect:

```python
plan.spill_bytes
plan.reload_bytes
plan.total_transfer_bytes
```

Avoid adding benchmark-only logic to the compiler.

- [ ] **Step 4: Run focused verification suites**

Run: `pytest tests/test_liveness_analysis.py tests/test_memory_strategy_selection.py tests/test_cost_aware_allocator.py tests/test_memory_limit_enforcement.py -q`
Expected: PASS

Run: `pytest tests/test_spill_verification.py tests/test_spill_correctness.py -q`
Expected: PASS or environment-appropriate skips

- [ ] **Step 5: Run benchmark comparison**

Run:

```bash
python -m benchmarks.harness --model resnet18 --batch-sizes 1 --output benchmarks/results/resnet18-cost-aware.json
```

Run a constrained-memory comparison using a fixed `max_memory`-driven workflow if benchmark harness support is added during implementation; otherwise capture the planner metrics from tests and document benchmark follow-up separately.

Expected:

- constrained-memory scenarios show reduced or equal `total_transfer_bytes`
- end-to-end latency does not regress in the target scenario

- [ ] **Step 6: Commit**

```bash
git add tests/test_liveness_analysis.py tests/test_memory_strategy_selection.py tests/test_cost_aware_allocator.py tests/test_memory_limit_enforcement.py tests/test_spill_verification.py tests/test_spill_correctness.py
git commit -m "test(memory): verify cost-aware planner behavior"
```

## Task 6: Documentation And Developer Guidance

**Files:**
- Modify: `README.md`
- Modify: `docs/OPTIMIZATION_PASSES.md`

- [ ] **Step 1: Write the failing doc assertions mentally and list what must be true**

Required doc updates:

- `O0` uses conservative planning
- `O1/O2/O3` default to `cost_aware`
- `fast memory` is a constraint, not a minimization target
- planner optimizes `total_transfer_bytes`

- [ ] **Step 2: Update documentation**

Add concise wording like:

```markdown
- `O0`: conservative memory planning with the legacy `basic` allocator
- `O1/O2/O3`: transfer-aware memory planning with the `cost_aware` allocator
```

and explain that spill/reload reduction is the target under constrained memory.

- [ ] **Step 3: Run the relevant test/documentation verification**

Run: `pytest tests/test_project_metadata.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add README.md docs/OPTIMIZATION_PASSES.md
git commit -m "docs(memory): describe cost-aware planner defaults"
```
