# Joint SRAM Offset Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing joint tiling/schedule contract so the external solver owns fast-memory fragmentation and offset assignment end-to-end, and the compiler imports and validates solver-authored placement instead of reallocating SRAM internally.

**Architecture:** Extend the current external joint IR in place with first-class SRAM items and allocations, plus residency identities. Build compiler-known fixed SRAM items in the problem, require the solver to return dynamic residency items and offsets, validate the combined time/space placement, then materialize the imported placement into the internal schedule IR and a joint-path memory-import compatibility pass. Replace the joint path’s current dependency on `ScheduledMemoryPlanningPass` allocation logic with a pass that only validates/imports solver-authored placement.

**Tech Stack:** Python 3.11+, dataclasses, existing `nnc_py` pass pipeline, pytest, current joint schedule builders/validators/materializer, x86 scheduled codegen path.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/nnc_py/ir/joint_tiling_schedule.py` | Modify | Extend the external problem/solution IR with SRAM items, allocations, and residency identity |
| `src/nnc_py/ir/pipeline_schedule.py` | Modify | Add an internal carrier for imported solver-authored SRAM offsets and stable item identity |
| `src/nnc_py/joint_schedule/recipes.py` | Modify | Emit compiler-known fixed SRAM items in the joint problem |
| `src/nnc_py/joint_schedule/solver.py` | Modify | Upgrade the baseline solver to emit generated residency items and offsets |
| `src/nnc_py/joint_schedule/validation.py` | Modify | Validate item ownership, lifetime semantics, overlap, alignment, and offset legality |
| `src/nnc_py/joint_schedule/materialize.py` | Modify | Import solver-authored SRAM placement into internal schedule/result objects and compatibility metadata |
| `src/nnc_py/passes/joint_tiling_schedule.py` | Modify | Keep problem/solve/materialize passes aligned with the upgraded contract |
| `src/nnc_py/passes/joint_schedule_memory_import.py` | Create | Joint-path memory import/validation pass that replaces internal SRAM reallocation |
| `src/nnc_py/passes/base.py` | Modify | Swap the joint O3 path from `ScheduledMemoryPlanningPass` to the new import pass |
| `src/nnc_py/passes/__init__.py` | Modify | Export the new joint-path memory import pass cleanly |
| `src/nnc_py/compiler.py` | Modify | Keep compiler entrypoints and joint pipeline assembly aligned with the import-only memory path |
| `tests/test_joint_tiling_schedule_ir.py` | Modify | Cover new problem/solution IR fields and round-trips |
| `tests/test_pipeline_schedule_ir.py` | Modify | Cover the internal SRAM interval carrier for imported offsets |
| `tests/test_joint_tiling_schedule_problem_builder.py` | Modify | Verify problem builder emits fixed SRAM items with correct ownership |
| `tests/test_joint_tiling_schedule_solver.py` | Modify | Verify baseline and CLI solutions carry the upgraded placement fields |
| `tests/test_joint_tiling_schedule_validation.py` | Modify | Verify new placement validation failures and success cases |
| `tests/test_joint_tiling_schedule_materialize.py` | Modify | Verify materialization preserves imported offsets and item identity |
| `tests/test_joint_schedule_memory_import.py` | Create | Focused tests for the new import/validation pass |
| `tests/test_pipeline_pass_integration.py` | Modify | Confirm the joint path no longer reallocates SRAM internally |
| `tests/fake_joint_solver.py` | Modify | Keep the external test solver aligned with the required upgraded contract |
| `tests/test_pipeline_scheduler_e2e.py` | Modify | End-to-end compile/build coverage for solver-authored offsets |
| `tests/test_codegen_pipeline_schedule.py` | Modify | Verify downstream codegen compatibility with imported solver-authored SRAM placement |

---

### Task 1: Extend The External And Internal IR Carriers

**Files:**
- Modify: `src/nnc_py/ir/joint_tiling_schedule.py`
- Modify: `src/nnc_py/ir/pipeline_schedule.py`
- Modify: `tests/test_joint_tiling_schedule_ir.py`
- Modify: `tests/test_pipeline_schedule_ir.py`

- [ ] **Step 1: Write the failing joint IR tests**

```python
from nnc_py.ir.joint_tiling_schedule import (
    JointProblem,
    JointResidencyWindow,
    JointSramAllocation,
    JointSramItem,
    JointSramItemKind,
    JointSolution,
)


def test_joint_problem_round_trips_sram_items_and_alignment():
    problem = JointProblem(
        ...,
        sram_items=(
            JointSramItem(
                item_id="matmul0.temp",
                kind=JointSramItemKind.TEMP_INTERVAL,
                size_bytes=128,
                alignment_bytes=16,
                is_optional=False,
                owner_action_id="matmul0.compute",
                owner_value_id=None,
                owner_residency_id=None,
            ),
        ),
        default_alignment_bytes=16,
    )
    restored = JointProblem.from_json(problem.to_json())
    assert restored.sram_items[0].item_id == "matmul0.temp"
    assert restored.default_alignment_bytes == 16


def test_joint_solution_requires_residency_identity_generated_items_and_allocations():
    solution = JointSolution(
        ...,
        residency_windows=(
            JointResidencyWindow(
                residency_id="mid@0",
                value_id="mid",
                start_time=10,
                end_time=24,
            ),
        ),
        generated_sram_items=(
            JointSramItem(
                item_id="mid@0.item",
                kind=JointSramItemKind.RESIDENT_WINDOW,
                size_bytes=96,
                alignment_bytes=16,
                is_optional=False,
                owner_action_id=None,
                owner_value_id="mid",
                owner_residency_id="mid@0",
            ),
        ),
        sram_allocations=(
            JointSramAllocation(item_id="mid@0.item", offset=64),
        ),
    )
    assert solution.generated_sram_items[0].owner_residency_id == "mid@0"
```

- [ ] **Step 2: Write the failing internal carrier test**

```python
from nnc_py.ir.pipeline_schedule import SramAllocationInterval


def test_sram_allocation_interval_tracks_imported_offset_and_item_identity():
    interval = SramAllocationInterval(
        value_name="mid.resident@10",
        item_id="mid@0.item",
        item_kind="resident_window",
        buffer_id="joint_buf_0",
        offset=64,
        start_time=10,
        end_time=24,
        size_bytes=96,
    )
    assert interval.offset == 64
    assert interval.item_id == "mid@0.item"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_joint_tiling_schedule_ir.py tests/test_pipeline_schedule_ir.py -v
```

Expected: FAIL because the new IR fields and internal carrier fields do not exist yet.

- [ ] **Step 4: Implement the upgraded IR**

In `src/nnc_py/ir/joint_tiling_schedule.py`, add:

- `JointSramItemKind`
- `JointSramItem`
- `JointSramAllocation`

Extend:

- `JointProblem` with `sram_items` and `default_alignment_bytes`
- `JointResidencyWindow` with `residency_id`
- `JointSolution` with `generated_sram_items` and `sram_allocations`

Required semantics:

- `alignment_bytes` is required on every SRAM item
- `generated_sram_items` are solution-only dynamic items
- `residency_id` is required on every residency window
- `sram_allocations` are required solver-authored item placements

In `src/nnc_py/ir/pipeline_schedule.py`, extend `SramAllocationInterval` so it can carry:

- `item_id`
- `item_kind`
- `offset`

Keep the existing internal names and current schema names; do not add `V2` types or schema IDs.

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_joint_tiling_schedule_ir.py tests/test_pipeline_schedule_ir.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_joint_tiling_schedule_ir.py tests/test_pipeline_schedule_ir.py src/nnc_py/ir/joint_tiling_schedule.py src/nnc_py/ir/pipeline_schedule.py
git commit -m "feat: extend joint ir with sram items and offsets"
```

### Task 2: Emit Fixed SRAM Items And Upgrade The Baseline Solver

**Files:**
- Modify: `src/nnc_py/joint_schedule/recipes.py`
- Modify: `src/nnc_py/joint_schedule/solver.py`
- Modify: `tests/test_joint_tiling_schedule_problem_builder.py`
- Modify: `tests/test_joint_tiling_schedule_solver.py`

- [ ] **Step 1: Write the failing builder and solver tests**

```python
def test_problem_builder_emits_compute_temp_sram_items(ctx_with_tiled_group):
    problem = build_joint_problem(ctx_with_tiled_group)
    temp_items = [item for item in problem.sram_items if item.kind.value == "temp_interval"]
    assert temp_items
    assert temp_items[0].owner_action_id


def test_problem_builder_emits_transfer_buffer_items_when_lowered_semantics_require_them(
    ctx_with_transfer_buffer_requirement,
):
    problem = build_joint_problem(ctx_with_transfer_buffer_requirement)
    transfer_items = [item for item in problem.sram_items if item.kind.value == "transfer_buffer"]
    assert transfer_items
    assert all(item.owner_action_id for item in transfer_items)


def test_baseline_solver_emits_residency_items_and_allocations(valid_problem):
    result = BaselineJointScheduleSolver().solve(valid_problem)
    assert result.generated_sram_items
    assert result.sram_allocations
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_joint_tiling_schedule_problem_builder.py tests/test_joint_tiling_schedule_solver.py -v
```

Expected: FAIL because the problem builder does not emit SRAM items and the baseline solver does not return allocations.

- [ ] **Step 3: Implement fixed item construction in the problem builder**

Update `src/nnc_py/joint_schedule/recipes.py` so `build_joint_problem()` emits:

- one fixed `temp_interval` SRAM item per compute action with `temp_bytes > 0`
- any deterministic transfer-buffer items already implied by lowered semantics
- problem-wide `default_alignment_bytes`

Use stable IDs like:

```python
item_id=f"{action_id}.temp"
```

Do not pre-enumerate residency items in the problem.

- [ ] **Step 4: Implement baseline solver support for dynamic items and offsets**

Update `src/nnc_py/joint_schedule/solver.py` so the baseline solver:

- generates one `resident_window` item per emitted residency window
- emits `sram_allocations` for both:
  - problem-declared fixed items
  - solution-generated residency items
- carries forward any problem-declared `transfer_buffer` items into the same allocation pass
- uses a simple deterministic packing strategy good enough for regression tests

Minimal acceptable policy:

```python
offset = 0
for active_item in sorted_items:
    allocations.append(JointSramAllocation(item_id=active_item.item_id, offset=offset))
    offset += _align(active_item.size_bytes, active_item.alignment_bytes)
```

This is not meant to be optimal; it only needs to produce valid baseline contract payloads.

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_joint_tiling_schedule_problem_builder.py tests/test_joint_tiling_schedule_solver.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_joint_tiling_schedule_problem_builder.py tests/test_joint_tiling_schedule_solver.py src/nnc_py/joint_schedule/recipes.py src/nnc_py/joint_schedule/solver.py
git commit -m "feat: build fixed sram items and solver allocations"
```

### Task 3: Validate Time-Space Placement And Ownership Rules

**Files:**
- Modify: `src/nnc_py/joint_schedule/validation.py`
- Modify: `tests/test_joint_tiling_schedule_validation.py`

- [ ] **Step 1: Write the failing placement validation tests**

```python
def test_validator_rejects_missing_allocation_for_active_temp_item(valid_problem, valid_solution):
    broken = replace(valid_solution, sram_allocations=())
    failure = validate_joint_solution(valid_problem, broken)
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION


def test_validator_rejects_capacity_overflow(valid_problem, valid_solution):
    broken = replace(
        valid_solution,
        sram_allocations=(
            JointSramAllocation(item_id="mid@0.item", offset=valid_problem.sram_capacity_bytes),
        ),
    )
    failure = validate_joint_solution(valid_problem, broken)
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION


def test_validator_rejects_residency_cardinality_mismatch(valid_problem, valid_solution):
    broken = replace(valid_solution, generated_sram_items=())
    failure = validate_joint_solution(valid_problem, broken)
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION


def test_validator_rejects_invalid_item_ownership(valid_problem, valid_solution):
    broken_item = replace(
        valid_solution.generated_sram_items[0],
        owner_value_id=None,
        owner_residency_id=None,
    )
    broken = replace(valid_solution, generated_sram_items=(broken_item,))
    failure = validate_joint_solution(valid_problem, broken)
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION


def test_validator_rejects_overlapping_offsets_for_time_overlapping_items(valid_problem, valid_solution):
    broken = replace(
        valid_solution,
        sram_allocations=(
            JointSramAllocation(item_id="matmul0.temp", offset=0),
            JointSramAllocation(item_id="mid@0.item", offset=0),
        ),
    )
    failure = validate_joint_solution(valid_problem, broken)
    assert failure.error_category is JointFailureCategory.RESOURCE_OVERLAP


def test_validator_rejects_misaligned_offset(valid_problem, valid_solution):
    broken = replace(
        valid_solution,
        sram_allocations=(JointSramAllocation(item_id="mid@0.item", offset=3),),
    )
    failure = validate_joint_solution(valid_problem, broken)
    assert failure.error_category is JointFailureCategory.INVALID_SOLUTION
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_joint_tiling_schedule_validation.py -v
```

Expected: FAIL because placement validation does not yet check allocations, alignment, or overlap.

- [ ] **Step 3: Implement placement validation**

Update `src/nnc_py/joint_schedule/validation.py` to add:

- active item discovery for:
  - problem-declared fixed items
  - solver-generated residency items
- exact one-to-one cardinality between `residency_windows` and `resident_window` items
- exact one allocation per active item
- half-open interval semantics `[start_time, end_time)`
- alignment checks
- negative-offset rejection
- `offset + size <= sram_capacity_bytes`
- address overlap rejection for time-overlapping items
- ownership checks for:
  - `temp_interval`
  - `transfer_buffer`
  - `resident_window`

Use the same lifetime rules everywhere:

```python
def _intervals_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_joint_tiling_schedule_validation.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_joint_tiling_schedule_validation.py src/nnc_py/joint_schedule/validation.py
git commit -m "feat: validate joint time-space placement"
```

### Task 4: Import Solver Placement Into Internal Schedule Results

**Files:**
- Modify: `src/nnc_py/joint_schedule/materialize.py`
- Modify: `tests/test_joint_tiling_schedule_materialize.py`

- [ ] **Step 1: Write the failing materialization tests**

```python
def test_materialize_joint_solution_preserves_imported_offsets(valid_problem, valid_solution):
    problem, result = materialize_joint_solution(valid_problem, valid_solution)
    assert result.sram_intervals
    assert result.sram_intervals[0].offset >= 0


def test_materialize_joint_solution_keeps_temp_and_resident_item_identity(valid_problem, valid_solution):
    _, result = materialize_joint_solution(valid_problem, valid_solution)
    assert {interval.item_kind for interval in result.sram_intervals} >= {
        "temp_interval",
        "resident_window",
    }


def test_materialize_joint_solution_preserves_transfer_buffer_identity(
    problem_with_transfer_buffer,
    solution_with_transfer_buffer,
):
    _, result = materialize_joint_solution(problem_with_transfer_buffer, solution_with_transfer_buffer)
    assert "transfer_buffer" in {interval.item_kind for interval in result.sram_intervals}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_joint_tiling_schedule_materialize.py -v
```

Expected: FAIL because materialization currently invents placeholder buffers and drops solver offsets.

- [ ] **Step 3: Implement offset-preserving materialization**

Update `src/nnc_py/joint_schedule/materialize.py` so it:

- imports `solution.generated_sram_items`
- resolves all active items and their allocations
- emits `PipelineScheduleResult.sram_intervals` with:
  - stable `item_id`
  - `item_kind`
  - imported `offset`
- preserves imported identities for `transfer_buffer` items in the same carrier
- stops inventing placeholder `joint_buf_<index>` allocations as the authoritative placement

Keep `scheduled_steps`, `scheduled_values`, and `residency_windows` compatible with the current downstream path.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_joint_tiling_schedule_materialize.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_joint_tiling_schedule_materialize.py src/nnc_py/joint_schedule/materialize.py
git commit -m "feat: materialize solver-authored sram offsets"
```

### Task 5: Replace Joint-Path Internal Allocation With Import Validation

**Files:**
- Create: `src/nnc_py/passes/joint_schedule_memory_import.py`
- Modify: `src/nnc_py/passes/joint_tiling_schedule.py`
- Modify: `src/nnc_py/passes/base.py`
- Modify: `src/nnc_py/passes/__init__.py`
- Modify: `src/nnc_py/compiler.py`
- Create: `tests/test_joint_schedule_memory_import.py`
- Modify: `tests/test_pipeline_pass_integration.py`

- [ ] **Step 1: Write the failing pass tests**

```python
def test_joint_o3_path_uses_memory_import_pass_not_scheduled_memory_planning():
    names = [p.__class__.__name__ for p in PassManager.get_joint_tiling_schedule_o3_passes()]
    assert "JointScheduleMemoryImportPass" in names
    assert "ScheduledMemoryPlanningPass" not in names


def test_joint_memory_import_pass_populates_compat_metadata(ctx_with_joint_result):
    JointScheduleMemoryImportPass().run(ctx_with_joint_result)
    assert "scheduled_memory_plan" in ctx_with_joint_result.metadata
    assert ctx_with_joint_result.metadata["memory_allocation_plan"].strategy_name == "joint_solver_import"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_joint_schedule_memory_import.py tests/test_pipeline_pass_integration.py tests/test_codegen_pipeline_schedule.py -v
```

Expected: FAIL because the new pass does not exist and the joint O3 path still uses `ScheduledMemoryPlanningPass`.

- [ ] **Step 3: Implement the joint-path import/validation pass**

Create `src/nnc_py/passes/joint_schedule_memory_import.py` with a pass that:

- reads `ctx.pipeline_schedule_problem` and `ctx.pipeline_schedule_result`
- validates imported `sram_intervals` are complete and offset-bearing
- builds compatibility metadata required by downstream codegen:
  - `scheduled_memory_plan`
  - `memory_allocation_plan`
- never reallocates or rewrites offsets

Use a distinct strategy name such as:

```python
strategy_name = "joint_solver_import"
```

- [ ] **Step 4: Rewire the joint O3 path**

Update:

- `src/nnc_py/passes/base.py`
- `src/nnc_py/passes/__init__.py`
- `src/nnc_py/compiler.py`

so the joint O3 path becomes:

1. existing O3 passes through `TiledLoweringPass`
2. `JointTilingScheduleProblemPass`
3. `JointTilingScheduleSolvePass`
4. `JointTilingScheduleMaterializationPass`
5. `LivenessAnalysisPass`
6. `JointScheduleMemoryImportPass`

Do not run `ScheduledMemoryPlanningPass` on the joint path anymore.

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_joint_schedule_memory_import.py tests/test_pipeline_pass_integration.py tests/test_codegen_pipeline_schedule.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_joint_schedule_memory_import.py tests/test_pipeline_pass_integration.py src/nnc_py/passes/joint_schedule_memory_import.py src/nnc_py/passes/joint_tiling_schedule.py src/nnc_py/passes/base.py src/nnc_py/passes/__init__.py src/nnc_py/compiler.py
git commit -m "feat: import joint solver memory placement"
```

### Task 6: Restore End-To-End Coverage For External Offset Ownership

**Files:**
- Modify: `tests/fake_joint_solver.py`
- Modify: `tests/test_pipeline_scheduler_e2e.py`

- [ ] **Step 1: Write the failing end-to-end assertions**

```python
def test_joint_contract_external_solver_solution_preserves_imported_offsets(tmp_path):
    ctx, output_dir = _compile_model(
        tmp_path,
        enable_pipeline_scheduler=None,
        metadata={
            "enable_joint_tiling_schedule_contract": True,
            "joint_tiling_schedule_solver_command": _joint_solver_command("solution"),
        },
    )
    assert ctx.pipeline_schedule_result.sram_intervals
    assert all(interval.offset >= 0 for interval in ctx.pipeline_schedule_result.sram_intervals)
    _build_generated_x86_source(output_dir)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_pipeline_scheduler_e2e.py -v --run-integration
```

Expected: FAIL because the fake solver and/or joint path do not yet preserve imported offsets end-to-end.

- [ ] **Step 3: Upgrade the fake external solver**

Update `tests/fake_joint_solver.py` so `"solution"` mode returns a fully valid upgraded solution payload, including:

- `residency_windows` with `residency_id`
- `generated_sram_items`
- `sram_allocations`

Use the baseline solver as the primary helper and upgrade all in-repo fixtures/tests that still rely on the old minimal contract. Do not preserve a fallback path for underspecified legacy payloads; the upgraded `v1` contract is required and should fail fast when required fields are absent.

- [ ] **Step 4: Strengthen end-to-end coverage**

Update `tests/test_pipeline_scheduler_e2e.py` to verify:

- joint baseline path imports offsets
- external solver success path imports offsets
- external solver failure path still surfaces standardized categories
- generated x86 code still builds under `make`

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_joint_tiling_schedule_solver.py tests/test_pipeline_scheduler_e2e.py -v --run-integration
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/fake_joint_solver.py tests/test_pipeline_scheduler_e2e.py
git commit -m "test: cover joint external sram offset ownership"
```

### Task 7: Final Focused Regression

**Files:**
- Test only

- [ ] **Step 1: Run the focused regression suite**

Run:

```bash
pytest \
  tests/test_joint_tiling_schedule_ir.py \
  tests/test_pipeline_schedule_ir.py \
  tests/test_joint_tiling_schedule_problem_builder.py \
  tests/test_joint_tiling_schedule_solver.py \
  tests/test_joint_tiling_schedule_validation.py \
  tests/test_joint_tiling_schedule_materialize.py \
  tests/test_joint_schedule_memory_import.py \
  tests/test_pipeline_pass_integration.py \
  tests/test_codegen_pipeline_schedule.py \
  tests/test_pipeline_scheduler_e2e.py \
  tests/test_scheduled_memory_planning.py \
  tests/test_pipeline_scheduler.py \
  -v --run-integration
```

Expected: PASS

- [ ] **Step 2: Record any regressions that require follow-up**

If this suite exposes non-joint-path regressions, stop and split them into a separate fix before merging.

- [ ] **Step 3: Commit any last targeted fixes if needed**

```bash
git add <files>
git commit -m "fix: resolve joint offset regression"
```
