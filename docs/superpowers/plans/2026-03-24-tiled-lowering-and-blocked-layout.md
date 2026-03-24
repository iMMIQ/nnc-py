# Tiled Lowering And Blocked Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a schedule-aware tiled lowering pipeline to `nnc-py` so large operators can compile under about `1 MB` fast memory using generic blocked layouts and late target-physical layout mapping.

**Architecture:** Keep the semantic graph IR intact, add structured execution-plan objects plus `ScheduleAnalysisPass`, `TiledLoweringPass`, `LayoutPlanningPass`, and `MemoryPlanningPassV3`, then teach the backend to emit tile-oriented fast-memory regions and late layout mapping hooks. Phase 1 targets `Conv` and pooling first, then extends tile-compatible elementwise handoff and backend code generation.

**Tech Stack:** Python 3, pytest, existing `nnc-py` IR/passes/codegen pipeline, ONNX Runtime numerical comparison tests, generated C backend artifacts.

---

## File Structure

### New files

- `src/nnc_py/ir/execution_plan.py`
  - Execution-plan dataclasses for layout class, tile region, tensor access, node execution plan, memory region kinds.
- `src/nnc_py/passes/schedule_analysis.py`
  - Decides which nodes require tiled execution and attaches schedule candidates.
- `src/nnc_py/passes/layout_planning.py`
  - Chooses generic blocked layouts and block factors without emitting target-physical layout.
- `src/nnc_py/passes/tiled_lowering.py`
  - Builds concrete tiled execution plans for phase-1 operators.
- `tests/test_execution_plan_ir.py`
  - Dataclass and serialization/metadata tests for execution-plan objects.
- `tests/test_schedule_analysis_pass.py`
  - Unit tests for node eligibility and schedule-candidate generation.
- `tests/test_layout_planning_pass.py`
  - Unit tests for blocked-layout planning.
- `tests/test_tiled_lowering_pass.py`
  - Unit tests for tiled conv / maxpool lowering metadata and legality.
- `tests/test_memory_planning_v3.py`
  - Unit tests for tile-aware fast-memory accounting and region reuse.
- `tests/test_codegen_tiled_layout.py`
  - Snapshot-style or string-based tests for emitted tile pools / comments / symbols.

### Modified files

- `src/nnc_py/ir/types.py`
  - Add generic blocked-layout enums or bridge types if needed.
- `src/nnc_py/ir/context.py`
  - Store typed execution-plan metadata cleanly.
- `src/nnc_py/passes/base.py`
  - Update `O3` pass ordering to include schedule/layout/tiled passes and V3 memory planning.
- `src/nnc_py/passes/__init__.py`
  - Export new passes and plan accessors.
- `src/nnc_py/passes/prepack_lowering.py`
  - Narrow role to static lowering hints and weight-prepack facts consumed by tiled lowering.
- `src/nnc_py/passes/memory_planning.py`
  - Introduce `MemoryPlanningPassV3` and shared accessors.
- `src/nnc_py/passes/memory_strategy.py`
  - Extend plan types with logical regions suitable for tile/scratch/stage accounting.
- `src/nnc_py/codegen/x86_backend.py`
  - Emit logical fast-memory regions and tile-aware declarations/comments/codegen hooks.
- `src/nnc_py/codegen/c_emitter.py`
  - Respect execution-plan entrypoints or tile-loop wrappers where phase 1 requires it.
- `benchmarks/metrics.py`
  - Read phase-1 tiled fast-memory declarations if new macros are added.
- `tests/test_pass_manager.py`
  - Assert new `O3` pass ordering.
- `tests/test_memory_strategy_selection.py`
  - Assert new V3 planning path defaults.
- `tests/test_benchmark_metrics.py`
  - Assert metrics parser supports tiled region macro layout.
- `tests/test_prepack_lowering_pass.py`
  - Lock narrowed lowering metadata contract if behavior changes.
- `tests/test_specialized_codegen.py`
  - Adjust only if specialized entrypoint naming changes under tiled codegen hooks.

## Task 1: Add Execution-Plan IR Primitives

**Files:**
- Create: `src/nnc_py/ir/execution_plan.py`
- Modify: `src/nnc_py/ir/types.py`
- Modify: `src/nnc_py/ir/context.py`
- Modify: `src/nnc_py/passes/__init__.py`
- Test: `tests/test_execution_plan_ir.py`

- [ ] **Step 1: Write the failing tests**

```python
from nnc_py.ir.execution_plan import LayoutClass, MemoryRegionKind, NodeExecutionPlan


def test_execution_plan_records_tile_and_layout_metadata():
    plan = NodeExecutionPlan(
        node_name="conv0",
        op_family="conv2d",
        tile_axes=("h", "w"),
        layout_class=LayoutClass.BLOCKED_ACTIVATION,
        memory_regions=(MemoryRegionKind.TILE, MemoryRegionKind.SCRATCH),
    )

    assert plan.node_name == "conv0"
    assert plan.layout_class is LayoutClass.BLOCKED_ACTIVATION
    assert MemoryRegionKind.SCRATCH in plan.memory_regions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_execution_plan_ir.py -q`
Expected: FAIL with import errors because `execution_plan.py` and the new enums do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
class LayoutClass(Enum):
    PLAIN = "plain"
    BLOCKED_ACTIVATION = "blocked_activation"
    BLOCKED_WEIGHT = "blocked_weight"
    TARGET_PHYSICAL = "target_physical"
```

Add focused dataclasses in `src/nnc_py/ir/execution_plan.py` and keep them immutable where practical.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_execution_plan_ir.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nnc_py/ir/execution_plan.py src/nnc_py/ir/types.py src/nnc_py/ir/context.py src/nnc_py/passes/__init__.py tests/test_execution_plan_ir.py
git commit -m "feat: add tiled execution plan ir"
```

## Task 2: Add `ScheduleAnalysisPass`

**Files:**
- Create: `src/nnc_py/passes/schedule_analysis.py`
- Modify: `src/nnc_py/passes/__init__.py`
- Test: `tests/test_schedule_analysis_pass.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_schedule_analysis_marks_large_conv_as_tiled_candidate():
    ctx = make_resnet_like_conv_context()

    ScheduleAnalysisPass().run(ctx)

    schedule = ctx.metadata["schedule_candidates"]
    assert schedule["conv0"].must_tile is True
    assert schedule["conv0"].reason == "peak_working_set"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schedule_analysis_pass.py -q`
Expected: FAIL because `ScheduleAnalysisPass` and `schedule_candidates` do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
class ScheduleAnalysisPass(PassBase):
    def _execute(self, ctx):
        ctx.metadata["schedule_candidates"] = analyze_nodes(ctx.graph)
```

Use current tensor sizes and operator families to mark obvious phase-1 tiling candidates:
- `Conv`
- `MaxPool`
- `AveragePool`
- `GlobalAveragePool`
- `Gemm` / `MatMul`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schedule_analysis_pass.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nnc_py/passes/schedule_analysis.py src/nnc_py/passes/__init__.py tests/test_schedule_analysis_pass.py
git commit -m "feat: add schedule analysis pass"
```

## Task 3: Add `LayoutPlanningPass`

**Files:**
- Create: `src/nnc_py/passes/layout_planning.py`
- Modify: `src/nnc_py/passes/__init__.py`
- Modify: `src/nnc_py/ir/types.py`
- Test: `tests/test_layout_planning_pass.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_layout_planning_assigns_generic_blocked_layout_to_conv_activation_and_weight():
    ctx = make_conv_context()
    ctx.metadata["schedule_candidates"] = {"conv0": candidate_for_conv("conv0")}

    LayoutPlanningPass().run(ctx)

    plan = ctx.metadata["layout_plans"]["conv0"]
    assert plan.input_layout.name == "blocked_activation"
    assert plan.weight_layout.name == "blocked_weight"
    assert plan.target_physical_layout is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layout_planning_pass.py -q`
Expected: FAIL because layout planning is not implemented.

- [ ] **Step 3: Write the minimal implementation**

```python
class LayoutPlanningPass(PassBase):
    def _execute(self, ctx):
        ctx.metadata["layout_plans"] = build_generic_blocked_layouts(ctx)
```

Pick conservative phase-1 defaults:
- activation blocked on `C`
- weights blocked on `K` and `C`
- no target-physical mapping at this stage

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_layout_planning_pass.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nnc_py/passes/layout_planning.py src/nnc_py/passes/__init__.py src/nnc_py/ir/types.py tests/test_layout_planning_pass.py
git commit -m "feat: add generic blocked layout planning"
```

## Task 4: Add `TiledLoweringPass` For `Conv` And Pooling

**Files:**
- Create: `src/nnc_py/passes/tiled_lowering.py`
- Modify: `src/nnc_py/passes/prepack_lowering.py`
- Modify: `src/nnc_py/passes/__init__.py`
- Test: `tests/test_tiled_lowering_pass.py`
- Test: `tests/test_prepack_lowering_pass.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_tiled_lowering_emits_conv_plan_with_halo_and_scratch():
    ctx = make_conv_context()
    seed_schedule_and_layout(ctx, node_name="conv0")

    TiledLoweringPass().run(ctx)

    plan = ctx.metadata["node_execution_plans"]["conv0"]
    assert plan.tile_axes == ("h", "w")
    assert plan.input_halo == (1, 1)
    assert plan.scratch_bytes >= 0
```

```python
def test_tiled_lowering_emits_pool_plan_without_weight_layout():
    ctx = make_maxpool_context()
    seed_schedule_and_layout(ctx, node_name="pool0")

    TiledLoweringPass().run(ctx)

    plan = ctx.metadata["node_execution_plans"]["pool0"]
    assert plan.op_family == "maxpool2d"
    assert plan.weight_access is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tiled_lowering_pass.py tests/test_prepack_lowering_pass.py -q`
Expected: FAIL because no execution plans are produced yet.

- [ ] **Step 3: Write the minimal implementation**

```python
class TiledLoweringPass(PassBase):
    def _execute(self, ctx):
        ctx.metadata["node_execution_plans"] = lower_phase1_nodes(ctx)
```

Adjust `PrepackLoweringPass` only so it keeps static facts needed by tiled lowering and does not grow into the execution scheduler.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tiled_lowering_pass.py tests/test_prepack_lowering_pass.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nnc_py/passes/tiled_lowering.py src/nnc_py/passes/prepack_lowering.py src/nnc_py/passes/__init__.py tests/test_tiled_lowering_pass.py tests/test_prepack_lowering_pass.py
git commit -m "feat: add phase1 tiled lowering pass"
```

## Task 5: Introduce `MemoryPlanningPassV3`

**Files:**
- Modify: `src/nnc_py/passes/memory_planning.py`
- Modify: `src/nnc_py/passes/memory_strategy.py`
- Modify: `src/nnc_py/passes/base.py`
- Modify: `src/nnc_py/passes/__init__.py`
- Test: `tests/test_memory_planning_v3.py`
- Test: `tests/test_pass_manager.py`
- Test: `tests/test_memory_strategy_selection.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_memory_planning_v3_uses_tile_regions_not_full_tensor_peak():
    ctx = make_tiled_conv_context()
    attach_phase1_execution_plan(ctx, input_tile_bytes=262144, output_tile_bytes=131072)

    MemoryPlanningPassV3().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.total_fast_memory <= 1024 * 1024
    assert plan.logical_regions["tile"].size_bytes >= 262144
```

```python
def test_o3_pass_order_includes_schedule_layout_and_tiled_lowering_before_v3():
    names = [p.__class__.__name__ for p in PassManager.get_default_passes(3)]
    assert names.index("ScheduleAnalysisPass") < names.index("MemoryPlanningPassV3")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_planning_v3.py tests/test_pass_manager.py tests/test_memory_strategy_selection.py -q`
Expected: FAIL because V3 planning and new pass ordering are not present.

- [ ] **Step 3: Write the minimal implementation**

```python
class MemoryPlanningPassV3(PassBase):
    def _execute(self, ctx):
        ctx.metadata["memory_allocation_plan"] = allocate_tile_regions(ctx)
```

Keep `MemoryPlanningPassV2` intact for compatibility, but make `O3` switch to V3 only when tiled planning metadata exists or when the advanced pipeline is enabled.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_planning_v3.py tests/test_pass_manager.py tests/test_memory_strategy_selection.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nnc_py/passes/memory_planning.py src/nnc_py/passes/memory_strategy.py src/nnc_py/passes/base.py src/nnc_py/passes/__init__.py tests/test_memory_planning_v3.py tests/test_pass_manager.py tests/test_memory_strategy_selection.py
git commit -m "feat: add tile-aware memory planning v3"
```

## Task 6: Emit Tile-Aware Memory Regions In Codegen

**Files:**
- Modify: `src/nnc_py/codegen/x86_backend.py`
- Modify: `src/nnc_py/codegen/c_emitter.py`
- Modify: `benchmarks/metrics.py`
- Test: `tests/test_codegen_tiled_layout.py`
- Test: `tests/test_benchmark_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_x86_backend_emits_tiled_fast_memory_regions():
    code = compile_model_with_tiled_plan()
    assert "NNC_TILE_MEMORY_SIZE" in code["tensors.c"]
    assert "NNC_SCRATCH_MEMORY_SIZE" in code["tensors.c"]
```

```python
def test_benchmark_metrics_reads_tile_and_scratch_pool_sizes(tmp_path):
    tensors_c = tmp_path / "tensors.c"
    tensors_c.write_text(
        "#define NNC_FAST_MEMORY_SIZE 524288\n"
        "#define NNC_TILE_MEMORY_SIZE 262144\n"
        "#define NNC_SCRATCH_MEMORY_SIZE 131072\n"
    )
    metrics = extract_memory_pool_sizes(tensors_c)
    assert metrics["fast_memory_bytes"] == 524288
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_codegen_tiled_layout.py tests/test_benchmark_metrics.py -q`
Expected: FAIL because tiled region emission is not implemented.

- [ ] **Step 3: Write the minimal implementation**

```python
lines.extend([
    "#define NNC_TILE_MEMORY_SIZE ...",
    "#define NNC_SCRATCH_MEMORY_SIZE ...",
])
```

Emit logical region declarations and keep code generation conservative in phase 1:
- region symbols
- tile-aware comments
- wrapper hooks for future tile-loop code

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codegen_tiled_layout.py tests/test_benchmark_metrics.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nnc_py/codegen/x86_backend.py src/nnc_py/codegen/c_emitter.py benchmarks/metrics.py tests/test_codegen_tiled_layout.py tests/test_benchmark_metrics.py
git commit -m "feat: emit tiled memory regions in x86 backend"
```

## Task 7: Add End-To-End Tiled Correctness Tests Against ONNX Runtime

**Files:**
- Modify: `tests/test_x86_runtime.py`
- Create: `tests/test_tiled_runtime_correctness.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_tiled_conv_matches_onnxruntime_reference():
    result = run_tiled_model_against_reference("conv")
    assert result.max_abs_diff == 0.0
```

```python
def test_tiled_maxpool_matches_onnxruntime_reference():
    result = run_tiled_model_against_reference("maxpool")
    assert result.max_abs_diff == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tiled_runtime_correctness.py -q`
Expected: FAIL because phase-1 tiled execution is not active yet.

- [ ] **Step 3: Write the minimal implementation**

Use the existing ONNX Runtime comparison harnesses and add focused helpers for:
- tiled conv
- tiled maxpool
- tile-compatible add/relu handoff

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tiled_runtime_correctness.py tests/test_x86_runtime.py -q -k "tiled or conv or maxpool"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_tiled_runtime_correctness.py tests/test_x86_runtime.py
git commit -m "test: verify tiled lowering against onnxruntime"
```

## Task 8: Prove `resnet18` Fast-Memory Reduction Path

**Files:**
- Modify: `tests/test_benchmark_metrics.py`
- Modify: `tests/test_snapshots_resnet18.py`
- Create: `tests/test_resnet18_tiled_memory_budget.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_resnet18_o3_tiled_plan_reports_fast_memory_within_1mb():
    metrics = compile_and_measure_resnet18(max_memory="1M", opt_level=3)
    assert metrics["fast_memory_bytes"] <= 1024 * 1024
```

```python
def test_resnet18_o3_tiled_codegen_builds_under_memory_budget():
    report = compile_resnet18_with_tiled_pipeline(max_memory="1M")
    assert report["compiled"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resnet18_tiled_memory_budget.py -q`
Expected: FAIL because the current whole-tensor path bottoms out around 4 MB.

- [ ] **Step 3: Write the minimal implementation**

Thread the tiled pipeline through the real `resnet18` compile path and expose measurable artifact metrics.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resnet18_tiled_memory_budget.py tests/test_snapshots_resnet18.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_resnet18_tiled_memory_budget.py tests/test_snapshots_resnet18.py tests/test_benchmark_metrics.py
git commit -m "test: lock resnet18 tiled fast-memory budget"
```

## Task 9: Add Late Target-Physical Layout Mapping Hooks

**Files:**
- Modify: `src/nnc_py/codegen/x86_backend.py`
- Modify: `src/nnc_py/ir/execution_plan.py`
- Modify: `tests/test_codegen_tiled_layout.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_execution_plan_keeps_generic_blocked_layout_before_physical_mapping():
    plan = compile_to_execution_plan_only()
    assert plan.layout_class.value == "blocked_activation"
    assert plan.target_physical_layout is None
```

```python
def test_backend_records_target_physical_layout_mapping_comment():
    code = compile_model_with_tiled_plan()
    assert "target_physical_layout" in code["model.c"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_codegen_tiled_layout.py -q -k "physical_layout or blocked_layout"`
Expected: FAIL because backend mapping hooks are absent.

- [ ] **Step 3: Write the minimal implementation**

Add a late mapping hook that records physical-layout intent without overcommitting to a full device backend in phase 1.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codegen_tiled_layout.py -q -k "physical_layout or blocked_layout"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nnc_py/codegen/x86_backend.py src/nnc_py/ir/execution_plan.py tests/test_codegen_tiled_layout.py
git commit -m "feat: add late target physical layout mapping hooks"
```

## Task 10: Final Verification And Documentation Sync

**Files:**
- Modify: `docs/superpowers/specs/2026-03-23-tiled-lowering-and-blocked-layout-design.md`
- Modify: `docs/superpowers/plans/2026-03-24-tiled-lowering-and-blocked-layout.md`

- [ ] **Step 1: Run focused unit and integration verification**

Run:

```bash
pytest tests/test_execution_plan_ir.py tests/test_schedule_analysis_pass.py tests/test_layout_planning_pass.py tests/test_tiled_lowering_pass.py tests/test_memory_planning_v3.py tests/test_codegen_tiled_layout.py tests/test_tiled_runtime_correctness.py tests/test_resnet18_tiled_memory_budget.py -q
```

Expected: PASS

- [ ] **Step 2: Run existing regression suites touched by the pipeline**

Run:

```bash
pytest tests/test_prepack_lowering_pass.py tests/test_pass_manager.py tests/test_memory_strategy_selection.py tests/test_x86_runtime.py tests/test_benchmark_metrics.py tests/test_snapshots_resnet18.py -q
```

Expected: PASS

- [ ] **Step 3: Update spec/plan notes if the implementation deviated in a meaningful way**

Record only real architectural deltas, not implementation trivia.

- [ ] **Step 4: Confirm clean worktree**

Run: `git status --short --branch`
Expected: no uncommitted changes

- [ ] **Step 5: Commit final docs sync if needed**

```bash
git add docs/superpowers/specs/2026-03-23-tiled-lowering-and-blocked-layout-design.md docs/superpowers/plans/2026-03-24-tiled-lowering-and-blocked-layout.md
git commit -m "docs: sync tiled lowering implementation notes"
```

## Notes For Execution

- Keep scope tight to phase 1. Do not attempt full `NZ/ZZ` physical layout materialization before generic blocked layout and tile-aware memory planning are correct.
- Prefer new focused test files rather than dumping all new assertions into existing large test modules.
- Keep `MemoryPlanningPassV2` available until V3 is stable; phase it out only after tiled planning is proven on real models.
- Do not bury the new pipeline inside `node.metadata["lowering"]` blobs. Use typed plan objects and explicit metadata slots on `CompileContext`.
- Numerical verification against ONNX Runtime is mandatory before claiming tiled lowering correctness.
- Whole-model `resnet18` verification is the acceptance gate for the `1 MB` fast-memory objective.
