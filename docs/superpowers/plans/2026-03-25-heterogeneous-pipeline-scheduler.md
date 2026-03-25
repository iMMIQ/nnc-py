# Heterogeneous Pipeline Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a baseline heterogeneous pipeline scheduler that models `MATMUL` / `SHAPE` / `DMA` / `OTHER`, keeps SRAM as a hard constraint, surfaces the schedule in generated `x86` simulation code, and cleanly decouples problem construction from cost modeling and solving.

**Architecture:** Extend the existing O3 tiled path with a standalone schedule-problem IR, pluggable cost-model providers, a baseline heuristic scheduler, and a time-aware memory planner. Keep the problem builder, solver, and backend consumers separate so future solver upgrades can reuse the same compiler-side interfaces.

**Tech Stack:** Python 3.11+, dataclasses, existing `nnc_py` pass pipeline, pytest, current x86 C code emitter/backend.

---

### Task 1: Add Pipeline Scheduling IR And Context Accessors

**Files:**
- Create: `src/nnc_py/ir/pipeline_schedule.py`
- Modify: `src/nnc_py/ir/context.py`
- Test: `tests/test_pipeline_schedule_ir.py`

- [ ] **Step 1: Write the failing IR tests**

```python
from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    PipelineScheduleProblem,
    PipelineScheduleResult,
    ScheduleDependencyKind,
    ScheduleEdge,
    ScheduleStep,
    ScheduleStepKind,
    SramValue,
)


def test_pipeline_schedule_problem_keeps_steps_edges_and_budget():
    step = ScheduleStep(
        id="conv0.compute",
        node_name="conv0",
        tile_id="tile0",
        step_kind=ScheduleStepKind.COMPUTE,
        resource_kind=PipelineResourceKind.MATMUL,
        duration=17,
        launch_overhead=3,
    )
    value = SramValue(name="conv0.tile0.out", size_bytes=4096)
    problem = PipelineScheduleProblem(
        steps=(step,),
        edges=(ScheduleEdge("a", "b", ScheduleDependencyKind.DATA),),
        sram_values=(value,),
        sram_capacity_bytes=64 * 1024,
    )

    assert problem.sram_capacity_bytes == 64 * 1024
    assert problem.steps[0].resource_kind is PipelineResourceKind.MATMUL


def test_compile_context_exposes_schedule_problem_and_result_without_mutation():
    ...
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_pipeline_schedule_ir.py -v`
Expected: FAIL with import or attribute errors because the new IR module and context accessors do not exist yet.

- [ ] **Step 3: Implement the standalone scheduling IR**

Create [pipeline_schedule.py](/home/ayd/code/nnc-py/src/nnc_py/ir/pipeline_schedule.py) with:

```python
from dataclasses import dataclass, field
from enum import Enum


class PipelineResourceKind(Enum):
    MATMUL = "matmul"
    SHAPE = "shape"
    DMA = "dma"
    OTHER = "other"


class ScheduleStepKind(Enum):
    DMA_IN = "dma_in"
    SHAPE_PREP = "shape_prep"
    COMPUTE = "compute"
    DMA_OUT = "dma_out"


class ScheduleDependencyKind(Enum):
    DATA = "data"
    ORDER = "order"
    SAME_NODE_SEQUENCE = "same_node_sequence"


@dataclass(frozen=True)
class ScheduleStep:
    id: str
    node_name: str
    tile_id: str | None = None
    step_kind: ScheduleStepKind = ScheduleStepKind.COMPUTE
    resource_kind: PipelineResourceKind = PipelineResourceKind.OTHER
    duration: int = 0
    launch_overhead: int = 0
    sram_input_names: tuple[str, ...] = ()
    sram_output_names: tuple[str, ...] = ()
    sram_temp_bytes: int = 0
    attrs: dict[str, object] = field(default_factory=dict)
```

Also add immutable dataclasses for:

- `SramValue`
- `ScheduleEdge`
- `ScheduledStep`
- `SramAllocationInterval`
- `PipelineScheduleProblem`
- `PipelineScheduleResult`

Add metadata helper functions, matching the existing execution-plan style:

- `get_pipeline_schedule_problem(ctx)`
- `set_pipeline_schedule_problem(ctx, problem)`
- `get_pipeline_schedule_result(ctx)`
- `set_pipeline_schedule_result(ctx, result)`

- [ ] **Step 4: Add typed context accessors**

Update [context.py](/home/ayd/code/nnc-py/src/nnc_py/ir/context.py) to expose read-only properties and lookup helpers for:

- `pipeline_schedule_problem`
- `pipeline_schedule_result`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_pipeline_schedule_ir.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_pipeline_schedule_ir.py src/nnc_py/ir/pipeline_schedule.py src/nnc_py/ir/context.py
git commit -m "feat: add pipeline scheduling IR"
```

### Task 2: Add Cost Model Provider Interfaces And Fallback Model

**Files:**
- Create: `src/nnc_py/cost_model/__init__.py`
- Create: `src/nnc_py/cost_model/base.py`
- Create: `src/nnc_py/cost_model/simple.py`
- Create: `src/nnc_py/cost_model/cli.py`
- Test: `tests/test_cost_model.py`

- [ ] **Step 1: Write the failing cost-model tests**

```python
from nnc_py.cost_model.simple import SimpleCostModelProvider
from nnc_py.ir.pipeline_schedule import PipelineResourceKind, ScheduleStepKind


def test_simple_cost_model_applies_non_zero_launch_overhead():
    provider = SimpleCostModelProvider()
    estimate = provider.estimate_step(
        op_type="Reshape",
        step_kind=ScheduleStepKind.SHAPE_PREP,
        resource_kind=PipelineResourceKind.SHAPE,
        input_shapes=((1, 64, 1, 1),),
        output_shapes=((1, 64),),
        dtypes=("float32",),
        tensor_bytes=256,
    )
    assert estimate.latency > 0
    assert estimate.launch_overhead > 0


def test_cli_cost_model_falls_back_when_command_is_missing():
    ...
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_cost_model.py -v`
Expected: FAIL because the cost-model package does not exist yet.

- [ ] **Step 3: Define the provider contract**

Create [base.py](/home/ayd/code/nnc-py/src/nnc_py/cost_model/base.py) with:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CostEstimate:
    latency: int
    launch_overhead: int
    source: str
    breakdown: dict[str, int] = field(default_factory=dict)


class CostModelProvider(ABC):
    @abstractmethod
    def estimate_step(
        self,
        *,
        op_type: str,
        step_kind,
        resource_kind,
        input_shapes,
        output_shapes,
        dtypes,
        tensor_bytes: int,
        attrs: dict[str, object] | None = None,
    ) -> CostEstimate:
        raise NotImplementedError
```

- [ ] **Step 4: Implement the fallback simple model**

Create [simple.py](/home/ayd/code/nnc-py/src/nnc_py/cost_model/simple.py) with an intentionally simple model:

```python
class SimpleCostModelProvider(CostModelProvider):
    DMA_LAUNCH = 12
    SHAPE_LAUNCH = 8
    MATMUL_LAUNCH = 16
    OTHER_LAUNCH = 10
    DMA_BW = 32
    SHAPE_TPUT = 64
    MATMUL_TPUT = 128
    OTHER_TPUT = 64
```

Use:

- `latency = launch + ceil(bytes / DMA_BW)` for `DMA`
- `latency = launch + ceil(elements / SHAPE_TPUT)` for `SHAPE`
- `latency = launch + ceil(macs / MATMUL_TPUT)` for `MATMUL`
- `latency = launch + ceil(work / OTHER_TPUT)` for `OTHER`

Keep the math simple and deterministic. The important property is non-zero launch overhead.

- [ ] **Step 5: Implement the CLI adapter with fallback**

Create [cli.py](/home/ayd/code/nnc-py/src/nnc_py/cost_model/cli.py) with:

- a provider that shells out to an external command
- a cache keyed by op/shape/dtype/step/resource
- graceful fallback to `SimpleCostModelProvider`
- bounded timeout

Expose a simple constructor shape:

```python
CliCostModelProvider(command: list[str] | None = None, fallback: CostModelProvider | None = None)
```

- [ ] **Step 6: Export the package API**

Create [__init__.py](/home/ayd/code/nnc-py/src/nnc_py/cost_model/__init__.py) exporting:

- `CostEstimate`
- `CostModelProvider`
- `CliCostModelProvider`
- `SimpleCostModelProvider`

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest tests/test_cost_model.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add tests/test_cost_model.py src/nnc_py/cost_model/__init__.py src/nnc_py/cost_model/base.py src/nnc_py/cost_model/simple.py src/nnc_py/cost_model/cli.py
git commit -m "feat: add pipeline cost model providers"
```

### Task 3: Lower Node Execution Plans Into Schedule Problems

**Files:**
- Create: `src/nnc_py/passes/pipeline_step_lowering.py`
- Modify: `src/nnc_py/passes/base.py`
- Modify: `src/nnc_py/passes/__init__.py`
- Test: `tests/test_pipeline_step_lowering.py`

- [ ] **Step 1: Write the failing pass tests**

```python
def test_pipeline_step_lowering_emits_mixed_granularity_steps_for_tiled_conv():
    ctx = make_tiled_conv_context()
    PipelineStepLoweringPass().run(ctx)

    problem = ctx.pipeline_schedule_problem
    assert problem is not None
    assert [step.step_kind.value for step in problem.steps] == [
        "dma_in",
        "compute",
        "dma_out",
    ]


def test_pipeline_step_lowering_keeps_small_relu_as_single_other_compute_step():
    ...
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pipeline_step_lowering.py -v`
Expected: FAIL because `PipelineStepLoweringPass` does not exist yet.

- [ ] **Step 3: Implement the pass**

Create [pipeline_step_lowering.py](/home/ayd/code/nnc-py/src/nnc_py/passes/pipeline_step_lowering.py) with a pass that:

- reads `ctx.node_execution_plans`
- classifies each node as:
  - large tiled op -> small multi-step sequence
  - ordinary op -> one `COMPUTE` step
- uses the cost-model provider to fill:
  - `duration`
  - `launch_overhead`
- emits:
  - `PipelineScheduleProblem`
  - step dependency edges
  - SRAM value objects for produced/consumed staged data

For the baseline:

- tiled `conv2d`, `gemm`, `matmul`:
  - `DMA_IN`
  - optional `SHAPE_PREP`
  - `COMPUTE`
  - `DMA_OUT`
- non-tiled shape ops:
  - one `SHAPE_PREP` or `COMPUTE` on `SHAPE`
- elementwise/general ops:
  - one `COMPUTE` on `OTHER`

- [ ] **Step 4: Register the pass without inserting it into the default O3 chain yet**

Update [__init__.py](/home/ayd/code/nnc-py/src/nnc_py/passes/__init__.py) to export `PipelineStepLoweringPass`.

Do not change the default pass order yet. Keep integration isolated until the scheduler and `V4` planner exist.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_pipeline_step_lowering.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_pipeline_step_lowering.py src/nnc_py/passes/pipeline_step_lowering.py src/nnc_py/passes/__init__.py
git commit -m "feat: lower tiled plans into pipeline schedule problems"
```

### Task 4: Implement The Baseline Heuristic Scheduler

**Files:**
- Create: `src/nnc_py/scheduler/__init__.py`
- Create: `src/nnc_py/scheduler/base.py`
- Create: `src/nnc_py/scheduler/list_scheduler.py`
- Create: `src/nnc_py/passes/pipeline_scheduling.py`
- Modify: `src/nnc_py/passes/__init__.py`
- Test: `tests/test_pipeline_scheduler.py`
- Test: `tests/test_pipeline_scheduling_pass.py`

- [ ] **Step 1: Write the failing scheduler tests**

```python
def test_list_scheduler_allows_dma_to_overlap_with_matmul():
    problem = make_dma_and_matmul_problem()
    result = ListPipelineScheduler().solve(problem)

    assert result.feasible is True
    dma = step_by_id(result, "dma_in")
    mm = step_by_id(result, "matmul")
    assert dma.start_time <= mm.start_time < dma.end_time


def test_list_scheduler_serializes_two_dma_steps():
    ...


def test_list_scheduler_delays_ready_step_when_sram_budget_would_overflow():
    ...
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pipeline_scheduler.py tests/test_pipeline_scheduling_pass.py -v`
Expected: FAIL because the scheduler package and scheduling pass do not exist.

- [ ] **Step 3: Implement the scheduler interface**

Create [base.py](/home/ayd/code/nnc-py/src/nnc_py/scheduler/base.py) with:

```python
class PipelineScheduler(ABC):
    @abstractmethod
    def solve(self, problem: PipelineScheduleProblem) -> PipelineScheduleResult:
        raise NotImplementedError
```

- [ ] **Step 4: Implement the list scheduler**

Create [list_scheduler.py](/home/ayd/code/nnc-py/src/nnc_py/scheduler/list_scheduler.py) with:

- topological readiness tracking
- earliest-feasible-start computation
- one slot per resource kind
- SRAM capacity feasibility checks
- priority based on:
  - critical-path length
  - resource preference (`MATMUL` first)
  - stable topological tie-break

Return a result with:

- `scheduled_steps`
- `makespan`
- `feasible`
- diagnostics for infeasible cases

- [ ] **Step 5: Implement the pass wrapper**

Create [pipeline_scheduling.py](/home/ayd/code/nnc-py/src/nnc_py/passes/pipeline_scheduling.py) with a pass that:

- reads `ctx.pipeline_schedule_problem`
- invokes a scheduler instance
- stores `ctx.pipeline_schedule_result`
- records whether it used the heuristic path or a serial fallback

- [ ] **Step 6: Export the new pass and scheduler**

Update:

- [__init__.py](/home/ayd/code/nnc-py/src/nnc_py/passes/__init__.py)
- [__init__.py](/home/ayd/code/nnc-py/src/nnc_py/scheduler/__init__.py)

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest tests/test_pipeline_scheduler.py tests/test_pipeline_scheduling_pass.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add tests/test_pipeline_scheduler.py tests/test_pipeline_scheduling_pass.py src/nnc_py/scheduler/__init__.py src/nnc_py/scheduler/base.py src/nnc_py/scheduler/list_scheduler.py src/nnc_py/passes/pipeline_scheduling.py src/nnc_py/passes/__init__.py
git commit -m "feat: add baseline heterogeneous pipeline scheduler"
```

### Task 5: Add Time-Aware SRAM Memory Planning V4

**Files:**
- Create: `src/nnc_py/passes/memory_planning_v4.py`
- Modify: `src/nnc_py/passes/base.py`
- Modify: `src/nnc_py/passes/__init__.py`
- Test: `tests/test_memory_planning_v4.py`

- [ ] **Step 1: Write the failing V4 tests**

```python
def test_memory_planning_v4_reuses_buffer_for_non_overlapping_scheduled_values():
    ctx = make_scheduled_two_stage_context()
    MemoryPlanningPassV4().run(ctx)

    plan = ctx.metadata["memory_allocation_plan"]
    assert plan.strategy_name == "schedule_time_v4"
    assert plan.total_fast_memory < sum_sizes_without_reuse(ctx)


def test_memory_planning_v4_falls_back_when_no_schedule_result_exists():
    ...
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_memory_planning_v4.py -v`
Expected: FAIL because `MemoryPlanningPassV4` does not exist yet.

- [ ] **Step 3: Implement the time-aware planner**

Create [memory_planning_v4.py](/home/ayd/code/nnc-py/src/nnc_py/passes/memory_planning_v4.py) with a pass that:

- consumes `ctx.pipeline_schedule_result`
- derives live intervals from scheduled producer/consumer windows
- assigns fast-memory buffers by interval non-overlap
- emits a `MemoryAllocationPlan`
- sets a distinct `strategy_name`, for example `schedule_time_v4`

Keep the first version conservative:

- only plan fast SRAM
- do not absorb all spill optimization into `V4`
- if required metadata is missing, return a clear fallback signal

- [ ] **Step 4: Add a compatibility wrapper for fallback**

Update [__init__.py](/home/ayd/code/nnc-py/src/nnc_py/passes/__init__.py) to export `MemoryPlanningPassV4`.

Do not remove `V2` or `V3`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_memory_planning_v4.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_memory_planning_v4.py src/nnc_py/passes/memory_planning_v4.py src/nnc_py/passes/__init__.py
git commit -m "feat: add time-aware memory planning v4"
```

### Task 6: Integrate The New O3 Path And Serial Fallbacks

**Files:**
- Modify: `src/nnc_py/passes/base.py`
- Modify: `src/nnc_py/compiler.py`
- Modify: `src/nnc_py/passes/__init__.py`
- Test: `tests/test_pipeline_pass_integration.py`

- [ ] **Step 1: Write the failing integration tests**

```python
def test_o3_pipeline_registers_pipeline_scheduler_after_tiled_lowering():
    passes = PassManager.get_default_passes(3)
    names = [p.name for p in passes]
    assert "PipelineStepLowering" in names
    assert "PipelineScheduling" in names
    assert "MemoryPlanningV4" in names


def test_pipeline_scheduler_metadata_records_serial_fallback_when_disabled():
    ...
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pipeline_pass_integration.py -v`
Expected: FAIL because the new passes are not part of the O3 default chain yet.

- [ ] **Step 3: Insert the new passes into the O3 path**

Update [base.py](/home/ayd/code/nnc-py/src/nnc_py/passes/base.py) so O3 becomes:

- `ScheduleAnalysisPass`
- `LayoutPlanningPass`
- `TiledLoweringPass`
- `PipelineStepLoweringPass`
- `PipelineSchedulingPass`
- `LivenessAnalysisPass`
- `MemoryPlanningPassV4`
- `SpillAnalysisPass`

Inside `MemoryPlanningPassV4`, keep explicit fallback to the current `V3` behavior when the new schedule path is unavailable or infeasible.

- [ ] **Step 4: Add compile-time metadata switches**

Update [compiler.py](/home/ayd/code/nnc-py/src/nnc_py/compiler.py) to allow:

- setting an external cost-model command in `ctx.metadata`
- optionally disabling the scheduler for debugging

Keep the defaults conservative:

- scheduler enabled for O3
- fallback provider always available

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_pipeline_pass_integration.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_pipeline_pass_integration.py src/nnc_py/passes/base.py src/nnc_py/compiler.py src/nnc_py/passes/__init__.py
git commit -m "feat: wire heterogeneous scheduler into o3 pipeline"
```

### Task 7: Surface The Schedule In x86 Simulation Codegen

**Files:**
- Modify: `src/nnc_py/codegen/x86_backend.py`
- Modify: `src/nnc_py/codegen/c_emitter.py`
- Test: `tests/test_codegen_pipeline_schedule.py`

- [ ] **Step 1: Write the failing backend tests**

```python
def test_x86_backend_emits_schedule_trace_for_pipeline_steps():
    ctx = compile_ctx_with_pipeline_schedule()
    artifacts = X86Backend().generate(ctx)
    model_c = next(f.content for f in artifacts.files if f.filename == "model.c")

    assert "pipeline schedule:" in model_c
    assert "resource=matmul" in model_c
    assert "resource=dma" in model_c


def test_x86_backend_keeps_compiling_when_pipeline_schedule_falls_back_to_serial():
    ...
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_codegen_pipeline_schedule.py -v`
Expected: FAIL because backend codegen does not emit schedule-aware traces yet.

- [ ] **Step 3: Extend backend metadata consumption**

Update [x86_backend.py](/home/ayd/code/nnc-py/src/nnc_py/codegen/x86_backend.py) so it:

- reads `ctx.pipeline_schedule_result`
- reads the `V4` memory plan when available
- prepares backend metadata for:
  - step trace comments
  - wrapper grouping
  - fallback-mode diagnostics

The backend must stay simulation-oriented. It does not need real host parallelism.

- [ ] **Step 4: Emit schedule-aware wrapper comments or trace blocks**

Update [c_emitter.py](/home/ayd/code/nnc-py/src/nnc_py/codegen/c_emitter.py) so generated code contains:

- a schedule summary comment block
- per-step annotations with:
  - `step_id`
  - `resource`
  - `start`
  - `end`
  - `duration`
  - `cost_source`
  - `sram binding`

Keep execution single-threaded. The point is visibility and deterministic simulation.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_codegen_pipeline_schedule.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_codegen_pipeline_schedule.py src/nnc_py/codegen/x86_backend.py src/nnc_py/codegen/c_emitter.py
git commit -m "feat: emit pipeline schedule traces in x86 backend"
```

### Task 8: Add End-To-End Coverage For Cost Model, Scheduling, And Fallback Paths

**Files:**
- Modify: `tests/test_complex_e2e.py`
- Create: `tests/test_pipeline_scheduler_e2e.py`

- [ ] **Step 1: Write the failing end-to-end tests**

```python
def test_o3_end_to_end_produces_pipeline_schedule_metadata_for_tiled_graph():
    ctx = compile_pipeline_ready_model()
    assert ctx.pipeline_schedule_problem is not None
    assert ctx.pipeline_schedule_result is not None
    assert ctx.metadata["memory_allocation_plan"].strategy_name in {
        "schedule_time_v4",
        "tile_regions_v3",
    }


def test_missing_cli_cost_model_uses_simple_fallback_without_failing_compile():
    ...
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pipeline_scheduler_e2e.py -v`
Expected: FAIL until all prior tasks are integrated.

- [ ] **Step 3: Add the end-to-end assertions**

Focus on:

- tiled conv or gemm graph that exercises the new O3 path
- fallback when the CLI cost model is unavailable
- serial fallback path when the schedule is unsupported
- generated source still building under the existing x86 simulation assumptions

- [ ] **Step 4: Run the focused tests**

Run: `pytest tests/test_pipeline_scheduler_e2e.py tests/test_codegen_pipeline_schedule.py -v`
Expected: PASS

- [ ] **Step 5: Run the broader regression subset**

Run: `pytest tests/test_execution_plan_ir.py tests/test_memory_planning_v3.py tests/test_tiled_execution_groups.py tests/test_codegen_tiled_layout.py tests/test_pipeline_schedule_ir.py tests/test_cost_model.py tests/test_pipeline_step_lowering.py tests/test_pipeline_scheduler.py tests/test_pipeline_scheduling_pass.py tests/test_memory_planning_v4.py tests/test_pipeline_pass_integration.py tests/test_pipeline_scheduler_e2e.py tests/test_codegen_pipeline_schedule.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_complex_e2e.py tests/test_pipeline_scheduler_e2e.py
git commit -m "test: add end-to-end coverage for pipeline scheduling"
```

### Task 9: Final Verification And Cleanup

**Files:**
- Modify: only files touched by previous tasks, if needed

- [ ] **Step 1: Run the full targeted verification set**

Run:

```bash
pytest tests/test_execution_plan_ir.py \
  tests/test_memory_planning_v3.py \
  tests/test_tiled_execution_groups.py \
  tests/test_codegen_tiled_layout.py \
  tests/test_pipeline_schedule_ir.py \
  tests/test_cost_model.py \
  tests/test_pipeline_step_lowering.py \
  tests/test_pipeline_scheduler.py \
  tests/test_pipeline_scheduling_pass.py \
  tests/test_memory_planning_v4.py \
  tests/test_pipeline_pass_integration.py \
  tests/test_pipeline_scheduler_e2e.py \
  tests/test_codegen_pipeline_schedule.py -v
```

Expected: PASS

- [ ] **Step 2: Inspect the generated pass order and backend trace output manually**

Run:

```bash
pytest tests/test_pipeline_pass_integration.py::test_o3_pipeline_registers_pipeline_scheduler_after_tiled_lowering -v -s
pytest tests/test_codegen_pipeline_schedule.py::test_x86_backend_emits_schedule_trace_for_pipeline_steps -v -s
```

Expected:

- O3 includes the new passes in the intended order
- generated code visibly contains pipeline schedule annotations

- [ ] **Step 3: Commit any final follow-up fixes**

```bash
git add src/nnc_py tests
git commit -m "fix: polish heterogeneous pipeline scheduler integration"
```
