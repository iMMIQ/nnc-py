# Scheduled O3 Native Spill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `-O3` a single scheduled path that natively handles `--max-memory` by modeling spill/reload as real DMA work and generating matching parallel x86 C code.

**Architecture:** Extend the existing scheduled-O3 IR so staged values, residency windows, and transfer steps are first-class objects. Insert a dedicated scheduled-memory expansion pass before scheduling, upgrade scheduling to reason about DMA spill/reload and SRAM capacity, then replace legacy V4 compatibility exports with a scheduled-native memory planner consumed directly by x86 codegen.

**Tech Stack:** Python 3.11+, existing `nnc_py` pass pipeline and scheduling IR, pytest, current x86 backend/runtime, ONNX CLI compile entrypoint.

---

### Task 1: Extend Scheduled O3 IR For Native Spill Semantics

**Files:**
- Modify: `src/nnc_py/ir/pipeline_schedule.py`
- Modify: `src/nnc_py/ir/context.py`
- Test: `tests/test_pipeline_schedule_ir.py`

- [ ] **Step 1: Write the failing IR tests**

```python
from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    PipelineScheduleProblem,
    ResidencyWindow,
    ScheduleStep,
    ScheduleStepKind,
    ScheduledValue,
    ScheduledValueHomeTier,
    TransferStep,
    TransferStepKind,
)


def test_scheduled_value_tracks_home_tier_and_graph_tensor_name():
    value = ScheduledValue(
        name="sram|node|4:add0|tensor|1:y",
        graph_tensor_name="y",
        size_bytes=64,
        home_tier=ScheduledValueHomeTier.SLOW,
    )

    assert value.graph_tensor_name == "y"
    assert value.home_tier is ScheduledValueHomeTier.SLOW


def test_transfer_step_uses_dma_resource_and_names_moved_value():
    step = TransferStep(
        id="y.reload0",
        node_name="reload:y",
        transfer_kind=TransferStepKind.RELOAD_DMA,
        moved_value_name="y",
        bytes=64,
    )

    assert step.resource_kind is PipelineResourceKind.DMA
    assert step.step_kind is ScheduleStepKind.RELOAD_DMA
```

- [ ] **Step 2: Run the IR tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_schedule_ir.py -q`
Expected: FAIL because the new spill-native IR symbols do not exist yet.

- [ ] **Step 3: Extend the scheduling IR**

Update `src/nnc_py/ir/pipeline_schedule.py` to add:

```python
class ScheduleStepKind(str, Enum):
    DMA_IN = "dma_in"
    SHAPE_PREP = "shape_prep"
    COMPUTE = "compute"
    SPILL_DMA = "spill_dma"
    RELOAD_DMA = "reload_dma"
    DMA_OUT = "dma_out"


class ScheduledValueHomeTier(str, Enum):
    INPUT = "input"
    CONST = "const"
    SLOW = "slow"
    SRAM = "sram"


@dataclass(frozen=True)
class ScheduledValue:
    name: str
    graph_tensor_name: str | None
    size_bytes: int
    producer_step_id: str | None = None
    consumer_step_ids: tuple[str, ...] = ()
    must_reside_in_sram: bool = False
    can_alias: bool = False
    home_tier: ScheduledValueHomeTier = ScheduledValueHomeTier.SRAM


@dataclass(frozen=True)
class ResidencyWindow:
    value_name: str
    residency_id: str
    opened_by_step_id: str
    closed_by_step_id: str | None = None


@dataclass(frozen=True)
class TransferStep(ScheduleStep):
    transfer_kind: TransferStepKind = TransferStepKind.DMA_IN
    moved_value_name: str = ""
    src_tier: ScheduledValueHomeTier = ScheduledValueHomeTier.SLOW
    dst_tier: ScheduledValueHomeTier = ScheduledValueHomeTier.SRAM
    bytes: int = 0
```

Also extend `PipelineScheduleProblem` and `PipelineScheduleResult` to carry:

- `scheduled_values`
- `residency_windows`
- transfer-specific diagnostics needed by later passes

- [ ] **Step 4: Add context helpers**

Update `src/nnc_py/ir/context.py` with accessors for the new metadata fields so later passes do not poke raw dictionaries.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pipeline_schedule_ir.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/nnc_py/ir/pipeline_schedule.py src/nnc_py/ir/context.py tests/test_pipeline_schedule_ir.py
git commit -m "feat: extend scheduled O3 IR for native spill"
```

### Task 2: Emit Scheduled Values Without Legacy Spill Assumptions

**Files:**
- Modify: `src/nnc_py/passes/pipeline_step_lowering.py`
- Test: `tests/test_pipeline_step_lowering.py`
- Test: `tests/test_codegen_pipeline_schedule.py`

- [ ] **Step 1: Write the failing lowering tests**

```python
def test_pipeline_step_lowering_marks_external_values_with_home_tier():
    ctx = _make_conv_relu_context()
    _run_pipeline_step_lowering(ctx)

    problem = ctx.pipeline_schedule_problem
    input_value = next(value for value in problem.scheduled_values if value.name == "input")
    assert input_value.home_tier.value == "input"


def test_pipeline_step_lowering_keeps_staged_outputs_as_sram_values():
    ctx = _make_conv_relu_context()
    _run_pipeline_step_lowering(ctx)

    problem = ctx.pipeline_schedule_problem
    staged = next(value for value in problem.scheduled_values if value.name.startswith("sram|node|"))
    assert staged.home_tier.value == "sram"
```

- [ ] **Step 2: Run the lowering tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_step_lowering.py -q`
Expected: FAIL because scheduled values still use the old `SramValue`-only model.

- [ ] **Step 3: Refactor lowering to produce canonical scheduled values**

Update `src/nnc_py/passes/pipeline_step_lowering.py` so:

- external inputs/constants are emitted as `ScheduledValue(home_tier=INPUT/CONST/SLOW)`
- staged outputs keep the encoded staged name but also carry `graph_tensor_name`
- raw `SramValue` creation is replaced by the new scheduled value object
- base DMA steps remain explicit and continue to use the DMA resource

Use a helper like:

```python
def _scheduled_value_for_access(..., home_tier: ScheduledValueHomeTier) -> ScheduledValue:
    return ScheduledValue(
        name=staged_name,
        graph_tensor_name=access.tensor_name,
        size_bytes=max(size_bytes, 1),
        producer_step_id=producer_step_id,
        consumer_step_ids=consumer_ids,
        can_alias=True,
        home_tier=home_tier,
    )
```

- [ ] **Step 4: Update codegen metadata tests**

Adjust `tests/test_codegen_pipeline_schedule.py` to use `scheduled_values` instead of the legacy `sram_values` field wherever applicable.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pipeline_step_lowering.py -q tests/test_codegen_pipeline_schedule.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/nnc_py/passes/pipeline_step_lowering.py tests/test_pipeline_step_lowering.py tests/test_codegen_pipeline_schedule.py
git commit -m "refactor: lower scheduled values for native spill"
```

### Task 3: Add Scheduled Memory Expansion Pass With Real DMA Spill/Reload Steps

**Files:**
- Create: `src/nnc_py/passes/scheduled_memory_expansion.py`
- Modify: `src/nnc_py/passes/__init__.py`
- Modify: `src/nnc_py/passes/base.py`
- Test: `tests/test_scheduled_memory_expansion.py`
- Test: `tests/test_pipeline_pass_integration.py`

- [ ] **Step 1: Write the failing expansion-pass tests**

```python
from nnc_py.passes.scheduled_memory_expansion import ScheduledMemoryExpansionPass


def test_expansion_adds_spill_and_reload_dma_steps_when_budget_is_tight():
    ctx = make_scheduled_spill_context(max_memory=64)

    ScheduledMemoryExpansionPass().run(ctx)

    problem = ctx.pipeline_schedule_problem
    step_ids = [step.id for step in problem.steps]
    assert "value0.spill0" in step_ids
    assert "value0.reload0" in step_ids


def test_expansion_rejects_must_reside_value_that_cannot_fit():
    ctx = make_must_reside_context(max_memory=32)

    with pytest.raises(RuntimeError, match="must reside in SRAM"):
        ScheduledMemoryExpansionPass().run(ctx)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_scheduled_memory_expansion.py -q`
Expected: FAIL because the pass does not exist yet.

- [ ] **Step 3: Implement scheduled memory expansion**

Create `src/nnc_py/passes/scheduled_memory_expansion.py` with a pass that:

- reads `ctx.pipeline_schedule_problem`
- reads `ctx.metadata["max_memory"]` when present
- identifies values that cannot remain resident across all consumers
- creates `TransferStep(..., transfer_kind=SPILL_DMA)` and `TransferStep(..., transfer_kind=RELOAD_DMA)` nodes
- adds dependency edges so each reload precedes the consumer it serves
- records `ResidencyWindow` metadata for the resulting problem

The first implementation can be greedy, but it must be legal:

```python
for value in spill_candidates:
    spill_step = TransferStep(
        id=f"{value.name}.spill0",
        node_name=f"spill:{value.name}",
        transfer_kind=TransferStepKind.SPILL_DMA,
        moved_value_name=value.name,
        src_tier=ScheduledValueHomeTier.SRAM,
        dst_tier=ScheduledValueHomeTier.SLOW,
        bytes=value.size_bytes,
    )
```

- [ ] **Step 4: Insert the new pass into O3**

Update `src/nnc_py/passes/base.py` so scheduled O3 includes `ScheduledMemoryExpansionPass` before `PipelineSchedulingPass`.

Also update `tests/test_pipeline_pass_integration.py` to assert:

- `ScheduledMemoryExpansionPass` exists in the O3 chain
- `SpillAnalysisPass` does not exist in the O3 chain

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_scheduled_memory_expansion.py -q tests/test_pipeline_pass_integration.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/nnc_py/passes/scheduled_memory_expansion.py src/nnc_py/passes/__init__.py src/nnc_py/passes/base.py tests/test_scheduled_memory_expansion.py tests/test_pipeline_pass_integration.py
git commit -m "feat: expand scheduled O3 with dma spill steps"
```

### Task 4: Upgrade Scheduling To Enforce SRAM Capacity And DMA Spill Legality

**Files:**
- Modify: `src/nnc_py/passes/pipeline_scheduling.py`
- Modify: `src/nnc_py/scheduler/list_scheduler.py`
- Test: `tests/test_pipeline_scheduler.py`
- Test: `tests/test_pipeline_scheduler_e2e.py`

- [ ] **Step 1: Write the failing scheduler tests**

```python
def test_list_scheduler_places_reload_before_its_consumer():
    problem = make_problem_with_reload_step()
    result = ListPipelineScheduler().solve(problem)

    assert result.feasible is True
    by_id = {step.step_id: step for step in result.scheduled_steps}
    assert by_id["value0.reload0"].end_time <= by_id["consumer0.compute"].start_time


def test_list_scheduler_reports_budget_failure_without_keyerror():
    problem = make_impossible_spill_problem()
    result = ListPipelineScheduler().solve(problem)

    assert result.feasible is False
    assert result.diagnostics["reason"] == "no_feasible_schedule_under_budget"
```

- [ ] **Step 2: Run the scheduler tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_scheduler.py -q tests/test_pipeline_scheduler_e2e.py -q`
Expected: FAIL because the scheduler does not yet account for spill/reload residency windows and new diagnostics.

- [ ] **Step 3: Upgrade the scheduler**

Update `src/nnc_py/scheduler/list_scheduler.py` to:

- validate transfer steps against `scheduled_values`
- track residency windows rather than only monolithic `sram_values`
- count SRAM occupancy over time including reload-created windows
- keep all DMA transfer kinds on the DMA resource
- emit explicit diagnostics instead of falling through to internal exceptions

The key invariants are:

```python
if step.step_kind is ScheduleStepKind.RELOAD_DMA:
    # consumer must not start until reload completes
    ...
if active_sram_bytes > problem.sram_capacity_bytes:
    return PipelineScheduleResult(
        feasible=False,
        solver_name="list",
        diagnostics={"reason": "no_feasible_schedule_under_budget"},
    )
```

- [ ] **Step 4: Wire the upgraded result through the scheduling pass**

Update `src/nnc_py/passes/pipeline_scheduling.py` so O3 errors preserve the new diagnostics and never attempt legacy fallback semantics.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pipeline_scheduler.py -q tests/test_pipeline_scheduler_e2e.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/nnc_py/passes/pipeline_scheduling.py src/nnc_py/scheduler/list_scheduler.py tests/test_pipeline_scheduler.py tests/test_pipeline_scheduler_e2e.py
git commit -m "feat: schedule dma spill and reload under memory budgets"
```

### Task 5: Replace Legacy V4 Compatibility With Scheduled-Native Memory Planning

**Files:**
- Create: `src/nnc_py/passes/scheduled_memory_planning.py`
- Modify: `src/nnc_py/passes/__init__.py`
- Modify: `src/nnc_py/passes/base.py`
- Modify: `src/nnc_py/compiler.py`
- Test: `tests/test_scheduled_memory_planning.py`
- Modify: `tests/test_memory_planning_v4.py`
- Modify: `tests/test_pipeline_pass_integration.py`

- [ ] **Step 1: Write the failing planning tests**

```python
from nnc_py.passes.scheduled_memory_planning import ScheduledMemoryPlanningPass


def test_scheduled_memory_planning_assigns_slow_offsets_to_spilled_values():
    ctx = make_spilled_schedule_context()

    ScheduledMemoryPlanningPass().run(ctx)

    plan = ctx.metadata["scheduled_memory_plan"]
    assert plan.slow_allocations["value0"].offset >= 0
    assert plan.transfer_points


def test_scheduled_memory_planning_does_not_write_legacy_memory_plan():
    ctx = make_spilled_schedule_context()

    ScheduledMemoryPlanningPass().run(ctx)

    assert "memory_plan" not in ctx.metadata
    assert "spill_plan" not in ctx.metadata
```

- [ ] **Step 2: Run the planning tests to verify they fail**

Run: `uv run pytest tests/test_scheduled_memory_planning.py -q`
Expected: FAIL because the scheduled-native planner does not exist yet.

- [ ] **Step 3: Implement the new planner**

Create `src/nnc_py/passes/scheduled_memory_planning.py` with:

- a `ScheduledMemoryPlanningPass`
- dataclasses for final SRAM allocations, slow allocations, and transfer points
- no legacy compatibility export

The planner should consume the scheduled result plus residency windows and produce one new metadata object, for example:

```python
ctx.metadata["scheduled_memory_plan"] = ScheduledMemoryPlan(
    total_fast_memory=...,
    total_slow_memory=...,
    value_allocations={...},
    slow_allocations={...},
    transfer_points=(...),
)
```

- [ ] **Step 4: Switch O3 to the new planner and remove legacy semantics**

Update:

- `src/nnc_py/passes/base.py` to use `ScheduledMemoryPlanningPass` in O3
- `src/nnc_py/compiler.py` to remove O3 fallback/legacy compatibility assumptions
- `tests/test_pipeline_pass_integration.py` to assert the new pass name

Trim or replace `tests/test_memory_planning_v4.py` so it no longer encodes legacy spill compatibility as a requirement.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_scheduled_memory_planning.py -q tests/test_memory_planning_v4.py -q tests/test_pipeline_pass_integration.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/nnc_py/passes/scheduled_memory_planning.py src/nnc_py/passes/__init__.py src/nnc_py/passes/base.py src/nnc_py/compiler.py tests/test_scheduled_memory_planning.py tests/test_memory_planning_v4.py tests/test_pipeline_pass_integration.py
git commit -m "refactor: make O3 memory planning scheduled-native"
```

### Task 6: Make x86 Codegen And CLI Consume Native Spill DMA Plans

**Files:**
- Modify: `src/nnc_py/codegen/x86_backend.py`
- Modify: `src/nnc_py/cli.py`
- Test: `tests/test_codegen_pipeline_schedule.py`
- Test: `tests/test_pipeline_scheduler_e2e.py`
- Test: `tests/test_resnet18_tiled_memory_budget.py`
- Test: `tests/test_snapshots_operator_coverage_large.py`

- [ ] **Step 1: Write the failing codegen and CLI tests**

```python
def test_codegen_emits_real_dma_spill_and_reload_worker_steps():
    ctx = make_codegen_context_with_native_spill()
    artifacts = X86Backend().generate(ctx)
    model_c = _artifact_text(artifacts, "model.c")

    assert "spill_dma" in model_c
    assert "reload_dma" in model_c
    assert "nnc_pipeline_run_parallel" in model_c


def test_cli_o3_budget_failure_surfaces_scheduled_error_not_keyerror(tmp_path):
    result = subprocess.run(
        [
            "uv", "run", "python", "-m", "nnc_py.cli", "compile",
            "models/operator_coverage_large.onnx",
            "-o", str(tmp_path / "out"),
            "-t", "x86",
            "-O", "3",
            "--max-memory", "1M",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert "KeyError" not in result.stderr + result.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_codegen_pipeline_schedule.py -q tests/test_pipeline_scheduler_e2e.py -q tests/test_snapshots_operator_coverage_large.py -q`
Expected: FAIL because the backend still consumes legacy plan fields and does not render scheduled-native spill DMA steps.

- [ ] **Step 3: Update x86 backend to consume `scheduled_memory_plan`**

Refactor `src/nnc_py/codegen/x86_backend.py` so it:

- reads only scheduled-native memory metadata for O3
- emits worker functions for `spill_dma` and `reload_dma`
- places these on the DMA worker path
- stops using legacy `memory_plan` / `spill_plan` for scheduled O3

Key shape:

```python
if step_kind == "spill_dma":
    dma_lines.append(
        f"memcpy(_nnc_slow_pool + {slow_offset}, _nnc_fast_pool + {fast_offset}, {size});"
    )
```

- [ ] **Step 4: Improve CLI diagnostics**

Update `src/nnc_py/cli.py` or the compiler-layer exception path so infeasible scheduled O3 memory cases print explicit diagnostics and never expose raw staged SRAM key strings.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_codegen_pipeline_schedule.py -q tests/test_pipeline_scheduler_e2e.py -q tests/test_resnet18_tiled_memory_budget.py -q tests/test_snapshots_operator_coverage_large.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/nnc_py/codegen/x86_backend.py src/nnc_py/cli.py tests/test_codegen_pipeline_schedule.py tests/test_pipeline_scheduler_e2e.py tests/test_resnet18_tiled_memory_budget.py tests/test_snapshots_operator_coverage_large.py
git commit -m "feat: emit native spill dma code for scheduled O3"
```

### Task 7: Delete O3 Legacy Spill Dependencies And Run End-To-End Verification

**Files:**
- Modify: `src/nnc_py/passes/base.py`
- Modify: `src/nnc_py/passes/spill.py`
- Modify: `src/nnc_py/passes/memory_planning_v4.py`
- Modify: `src/nnc_py/compiler.py`
- Test: `tests/test_pipeline_pass_integration.py`
- Test: `tests/test_memory_limit_enforcement.py`

- [ ] **Step 1: Write the failing regression tests**

```python
def test_scheduled_o3_pass_chain_contains_no_legacy_spill_pass():
    names = [pass_obj.__class__.__name__ for pass_obj in PassManager.get_scheduled_o3_passes()]
    assert "SpillAnalysisPass" not in names


def test_o3_no_longer_exports_legacy_memory_plan_for_scheduled_path(monkeypatch, tmp_path):
    ctx = _compile_graph(monkeypatch, tmp_path, max_memory="64")
    assert "memory_plan" not in ctx.metadata
```

- [ ] **Step 2: Run the regression tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_pass_integration.py -q tests/test_memory_limit_enforcement.py -q`
Expected: FAIL because O3 still carries legacy spill compatibility code.

- [ ] **Step 3: Remove the remaining O3 legacy hooks**

Update:

- `src/nnc_py/passes/base.py` to keep `SpillAnalysisPass` out of all O3 paths
- `src/nnc_py/passes/spill.py` to document it as non-O3 legacy-only or delete the now-unused scheduled compatibility assumptions
- `src/nnc_py/passes/memory_planning_v4.py` to stop exporting legacy plan formats for O3
- `src/nnc_py/compiler.py` to treat O3 failures as scheduled-O3 failures only

- [ ] **Step 4: Run full verification**

Run:

```bash
uv run pytest \
  tests/test_pipeline_schedule_ir.py \
  tests/test_pipeline_step_lowering.py \
  tests/test_scheduled_memory_expansion.py \
  tests/test_pipeline_scheduler.py \
  tests/test_pipeline_scheduler_e2e.py \
  tests/test_scheduled_memory_planning.py \
  tests/test_codegen_pipeline_schedule.py \
  tests/test_pipeline_pass_integration.py \
  tests/test_memory_limit_enforcement.py \
  tests/test_resnet18_tiled_memory_budget.py \
  tests/test_snapshots_operator_coverage_large.py -q
```

Expected: PASS

Then run real CLI coverage:

```bash
uv run python -m nnc_py.cli compile models/operator_coverage_large.onnx -o /tmp/nnc-o3-native-spill-large -t x86 -O 3 --max-memory 1M
uv run python -m nnc_py.cli compile models/resnet18.onnx -o /tmp/nnc-o3-native-spill-resnet18 -t x86 -O 3 --max-memory 1M
make -C /tmp/nnc-o3-native-spill-large
make -C /tmp/nnc-o3-native-spill-resnet18
```

Expected:

- no staged SRAM `KeyError`
- successful compile or explicit scheduled-O3 feasibility error
- generated outputs build when compilation succeeds

- [ ] **Step 5: Commit**

```bash
git add src/nnc_py/passes/base.py src/nnc_py/passes/spill.py src/nnc_py/passes/memory_planning_v4.py src/nnc_py/compiler.py tests/test_pipeline_pass_integration.py tests/test_memory_limit_enforcement.py
git commit -m "refactor: remove legacy spill semantics from O3"
```
