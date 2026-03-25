# Heterogeneous Pipeline Scheduler Design

## Status

Proposed and user-approved at the design level on 2026-03-25.

This design targets a small-SRAM heterogeneous embedded device with multiple
independent execution pipelines:

- `MATMUL`
- `SHAPE`
- `DMA`
- `OTHER`

The pipelines may execute in parallel, but they share the same SRAM budget.

## Context

`nnc-py` already contains the compiler pieces needed to support a more realistic
embedded execution model:

- graph-level semantic IR
- schedule-candidate analysis
- generic blocked layout planning
- tiled lowering
- tile-aware fast-memory planning

The current pipeline, however, still assumes an essentially serial node
execution order. Existing execution-plan metadata describes how a node may be
tiled and which logical memory regions it needs, but it does not describe:

- execution time
- resource occupancy
- overlap between different compute pipelines
- shared-SRAM contention in the time dimension

For the target device, these omissions are decisive. A realistic compiler path
must model:

- resource-constrained non-preemptive scheduling
- shared SRAM as a hard capacity constraint
- single-channel DMA that cannot overlap with other DMA, but may overlap with
  compute
- launch overhead for each issued step, so the compiler does not over-fragment
  execution into tiny low-value micro-steps

This scheduling problem is fundamentally hard and will require long-term
algorithmic optimization. The implementation therefore must separate:

- schedule-problem construction
- cost modeling
- solver strategy
- schedule consumption by memory planning and backend simulation

That separation allows the same schedule problem to be solved later by a more
specialized algorithm, an external solver, or another LLM focused on
optimization quality.

## Goals

- Add a compile-time heterogeneous pipeline scheduler for the small-SRAM target.
- Model `MATMUL`, `SHAPE`, `DMA`, and `OTHER` as separate execution resources.
- Treat SRAM capacity as a hard scheduling constraint.
- Minimize makespan first.
- Support mixed scheduling granularity:
  - large tiled operators use a small number of pipeline steps
  - ordinary operators stay at node granularity
- Preserve a hard interface boundary so the scheduling problem can be handed to
  a stronger solver later without rewriting the compiler pipeline.
- Surface the schedule in generated `x86` code for simulation and inspection.

## Non-Goals

- Do not solve the global optimum scheduling problem in the first version.
- Do not model bank conflicts, multi-port SRAM arbitration, or sub-cycle hazards.
- Do not make steps preemptive.
- Do not support multiple DMA channels in the base version.
- Do not require full ONNX operator coverage in the initial implementation.
- Do not force all nodes into tile/step-level scheduling.

## Fixed Execution Assumptions

The following constraints were explicitly confirmed for the first version:

- objective: minimize total latency
- scheduling is offline and compile-time
- all scheduled tasks are non-preemptive
- DMA is a dedicated single resource
- DMA may overlap with compute resources
- multiple DMA steps may not overlap with one another
- all execution resources share one SRAM capacity budget
- the compiler must account for fixed launch overhead
- large operators use step-level scheduling, while ordinary operators remain
  node-level

## Alternative Approaches Considered

### Approach A: Node-Level Resource Scheduling Only

Treat each node as one task, assign it to one pipeline, and run a
resource-constrained list scheduler over node dependencies.

Pros:

- minimal compiler changes
- easy to integrate with the current pass pipeline

Cons:

- cannot express `DMA -> shape -> compute -> DMA` overlap
- cannot model SRAM occupancy with enough fidelity
- too coarse for tiled operators

Rejected.

### Approach B: Full Micro-Op Scheduling

Split every operator into many very small actions and schedule the complete
low-level DAG.

Pros:

- highest theoretical fidelity

Cons:

- launch overhead dominates many tiny tasks
- problem size explodes quickly
- too fragile for a first integrated compiler version

Rejected for the base version.

### Approach C: Layered Mixed-Granularity Scheduling

Keep semantic node plans, lower only selected nodes into a small number of
pipeline steps, then solve a resource-constrained scheduling problem over those
steps.

Pros:

- matches the target hardware model well enough
- preserves existing semantic and tiled-lowering structure
- avoids over-fragmentation
- leaves a clean solver boundary

Chosen.

## High-Level Architecture

The O3 pipeline is extended from:

`ScheduleAnalysis -> LayoutPlanning -> TiledLowering -> Liveness -> MemoryPlanningV3`

to:

`ScheduleAnalysis -> LayoutPlanning -> TiledLowering -> PipelineStepLowering -> PipelineScheduling -> MemoryPlanningV4 -> SpillAnalysis -> x86 simulation codegen`

The new layers are:

- `PipelineStepLoweringPass`
  - converts semantic/tiled node execution plans into scheduling steps
  - assigns a cost estimate to each step through a cost-model provider
  - emits a standalone scheduling problem IR

- `PipelineSchedulingPass`
  - invokes a scheduler interface on the emitted problem
  - stores the chosen schedule as metadata

- `MemoryPlanningPassV4`
  - consumes schedule time intervals instead of pure topological node indices
  - allocates SRAM buffers using scheduled live intervals

- `x86` schedule-aware simulation codegen
  - emits wrappers and traces that make the chosen schedule visible
  - remains a simulation target, not a real asynchronous hardware backend

## Decoupling Boundary

The implementation must hard-separate four concerns.

### 1. Problem IR

A stable, serializable description of the scheduling instance.

### 2. Cost Model

A replaceable provider used only to estimate step duration.

### 3. Solver

A replaceable component that takes a problem and returns a schedule result.

### 4. Consumers

Memory planning and backend codegen consume the schedule result but do not solve
the schedule problem themselves.

This boundary is the central design requirement for future optimization work.
The scheduling problem must be exportable as a standalone object that another
solver can own.

## Scheduling Problem IR

Add a new module, for example:

- `src/nnc_py/ir/pipeline_schedule.py`

Core types:

### `PipelineResourceKind`

- `MATMUL`
- `SHAPE`
- `DMA`
- `OTHER`

### `ScheduleStepKind`

- `DMA_IN`
- `SHAPE_PREP`
- `COMPUTE`
- `DMA_OUT`

### `ScheduleStep`

Fields:

- `id`
- `node_name`
- `tile_id | None`
- `step_kind`
- `resource_kind`
- `duration`
- `launch_overhead`
- `sram_input_names`
- `sram_output_names`
- `sram_temp_bytes`
- `attrs`

A step is the smallest non-preemptive unit seen by the scheduler.

### `SramValue`

Fields:

- `name`
- `size_bytes`
- `producer_step_id | None`
- `consumer_step_ids`
- `must_reside_in_sram`
- `can_alias`

This models values that remain live across step boundaries, including tile
buffers and staged intermediate outputs.

### `ScheduleEdge`

Fields:

- `src_step_id`
- `dst_step_id`
- `kind`

Supported edge kinds:

- `DATA`
- `ORDER`
- `SAME_NODE_SEQUENCE`

### `PipelineScheduleProblem`

Fields:

- `steps`
- `edges`
- `resources`
- `sram_capacity_bytes`
- `objective = MIN_MAKESPAN`
- `metadata`

This object is the formal solver input and should be exportable to JSON without
backend-specific details.

## Schedule Result IR

The solver returns a separate result object, also serializable.

### `ScheduledStep`

Fields:

- `step_id`
- `resource_kind`
- `resource_slot`
- `start_time`
- `end_time`

### `SramAllocationInterval`

Fields:

- `value_name`
- `buffer_id`
- `start_time`
- `end_time`
- `size_bytes`

### `PipelineScheduleResult`

Fields:

- `scheduled_steps`
- `sram_intervals`
- `makespan`
- `feasible`
- `solver_name`
- `diagnostics`

This object is the standard output contract for any scheduling algorithm.

## Cost Model Integration

The project already has, or will have, an external CLI cost-model program that
accepts:

- operator type
- tensor shapes
- tensor dtypes

and returns estimated latency.

The compiler must integrate that program through a provider interface rather
than calling it directly from passes.

Recommended modules:

- `src/nnc_py/cost_model/base.py`
- `src/nnc_py/cost_model/cli.py`
- `src/nnc_py/cost_model/simple.py`

### `CostModelProvider`

The provider interface should accept:

- `op_type`
- step kind
- input tensor shapes
- output tensor shapes
- dtypes
- resource kind
- optional target/layout metadata

and return:

- `estimated_latency`
- `source`
- optional breakdown/debug metadata

### External CLI Provider

`CliCostModelProvider` should:

- invoke the external CLI
- cache results by op/shape/dtype/signature
- sanitize invalid outputs
- fall back cleanly if the CLI is unavailable or fails

### Required Fallback Provider

The compiler must ship a simple internal fallback provider so the pipeline
remains runnable without the external CLI.

Base formulas for the fallback model:

- `DMA`: `launch_overhead + bytes / bandwidth`
- `MATMUL`: `launch_overhead + macs / throughput`
- `SHAPE`: `launch_overhead + elements / throughput`
- `OTHER`: `launch_overhead + work / throughput`

The critical rule is that launch overhead must be explicit and non-zero. The
fallback model must not allow tiny steps to appear effectively free, because
that would bias the scheduler toward pathological over-fragmentation.

## Mixed-Granularity Step Lowering

`PipelineStepLoweringPass` lowers graph/node execution plans into scheduling
steps.

Recommended modules:

- `src/nnc_py/passes/pipeline_step_lowering.py`

Rules for the first version:

- tiled large operators may lower to a small sequence of:
  - `DMA_IN`
  - `SHAPE_PREP`
  - `COMPUTE`
  - `DMA_OUT`
- ordinary operators stay as a single `COMPUTE` step
- only introduce step-level detail when it maps to a real resource pipeline or
  a real SRAM residency change

This pass is responsible for problem construction, not optimization.

It should:

- inspect `NodeExecutionPlan`
- decide which nodes are step-lowered
- derive SRAM value objects
- query the cost model for each step duration
- emit `PipelineScheduleProblem`

## Base Scheduling Algorithm

The first implementation should use a replaceable scheduler interface with a
conservative heuristic solver.

Recommended modules:

- `src/nnc_py/scheduler/base.py`
- `src/nnc_py/scheduler/list_scheduler.py`

### Interface

`PipelineScheduler.solve(problem) -> PipelineScheduleResult`

### Base Solver

Use resource-constrained list scheduling over the step DAG:

- maintain a ready set of dependency-satisfied steps
- compute each step's earliest feasible start time
- enforce:
  - predecessor completion
  - resource availability
  - SRAM hard-cap feasibility
- select one ready step at a time using a heuristic priority

Recommended priority order:

1. larger critical-path length first
2. `MATMUL` over `SHAPE`/`OTHER`
3. steps that unblock future `MATMUL`
4. steps that reduce future SRAM pressure
5. stable topological tie-break

This solver is not expected to be globally optimal. Its role is to establish:

- a correct schedule contract
- a feasible baseline
- a stable input/output format for later solver upgrades

## SRAM Constraint Handling

The first version should keep scheduling-time SRAM reasoning simpler than full
joint time-and-offset optimization.

### During Scheduling

Treat SRAM as a hard capacity check:

A step is feasible only if, at its candidate issue time:

- currently live SRAM values
- plus the step's temporary SRAM requirement
- plus newly produced SRAM outputs

fit within the SRAM budget.

### After Scheduling

Run a dedicated SRAM buffer assignment pass using scheduled time intervals.

This keeps the base scheduler focused on temporal feasibility while allowing
offset/buffer reuse to remain a separate concern.

## Time-Aware Memory Planning

Current memory planning logic is primarily topological. The new pipeline needs a
time-aware planner.

Recommended implementation:

- add `MemoryPlanningPassV4`
- keep `V2` and `V3` for existing paths

`MemoryPlanningPassV4` should:

- consume `PipelineScheduleResult`
- derive live intervals in schedule time
- assign fast-memory buffers to non-overlapping intervals
- compute peak SRAM usage over time
- emit a memory plan that backend codegen can consume

The base version should focus on SRAM fast memory. Existing spill logic may
remain as a separate conservative fallback instead of being fully absorbed into
the new scheduler.

## Pass Pipeline Integration

Recommended O3 order:

- `IdentityEliminationPass`
- `DeadCodeEliminationPass`
- `PatternFusionPass`
- `PrepackLoweringPass`
- `DominatorFusionPass`
- `ScheduleAnalysisPass`
- `LayoutPlanningPass`
- `TiledLoweringPass`
- `PipelineStepLoweringPass`
- `PipelineSchedulingPass`
- `LivenessAnalysisPass`
- `MemoryPlanningPassV4`
- `SpillAnalysisPass`

Fallback behavior:

- if no pipeline schedule problem is emitted, use current planning flow
- if scheduling fails, mark the failure and fall back to conservative serial
  execution order
- if `V4` cannot use the schedule result, fall back to `V3`

This preserves the current compiler behavior while allowing opt-in advancement.

## x86 Simulation Backend

The `x86` backend remains a simulation backend. It does not need to implement
true asynchronous execution. It does need to make the schedule visible and
testable.

Recommended backend behavior:

- consume the schedule result and time-aware memory plan when available
- emit schedule-oriented wrapper functions for supported step patterns
- emit schedule traces or comments containing:
  - step id
  - node/tile origin
  - resource kind
  - start/end time
  - duration
  - cost-model source
  - SRAM binding

The first version may execute scheduled wrappers in timestamp order inside a
single-threaded simulator. The purpose is traceability and behavioral modeling,
not actual host-side concurrency.

## Base Version Scope

The first version should explicitly limit itself to:

- tiled `Conv`
- tiled `Gemm/MatMul`
- `Reshape`/`Transpose`/`Flatten` on the `SHAPE` pipeline
- selected elementwise operators on the `OTHER` pipeline
- mixed granularity with a small fixed step vocabulary

Do not expand operator support until the IR, scheduler interface, time-aware
memory planning, and backend trace path are stable.

## Failure and Fallback Strategy

The new pipeline must degrade safely.

- external cost model unavailable
  - fall back to `SimpleCostModelProvider`
- schedule infeasible or unsupported
  - fall back to conservative serial scheduling metadata
- time-aware memory planning unsupported for a graph
  - fall back to existing `MemoryPlanningPassV3`
- backend step pattern unsupported
  - emit ordinary node wrappers with schedule diagnostics/comments

Each fallback must be explicit in metadata so tests can assert which path was
used.

## Testing Strategy

Testing should be split by layer.

### IR / Problem Construction

- problem objects contain expected steps, edges, resources, and SRAM values
- mixed-granularity lowering only expands intended nodes

### Cost Model

- CLI provider caches and sanitizes results
- fallback provider enforces non-zero launch overhead
- tiny problems still produce meaningful positive duration

### Scheduler

- ready-set scheduling respects dependencies
- DMA steps do not overlap with other DMA
- DMA may overlap with compute
- SRAM hard-cap blocks otherwise-ready steps
- heuristics produce stable deterministic schedules

### Memory Planning

- scheduled intervals reuse buffers when non-overlapping
- peak SRAM is computed from schedule time, not node order
- changing a schedule changes the memory layout when expected

### Backend Simulation

- generated code contains schedule-visible wrappers or trace comments
- timing/resource annotations match the schedule result
- fallback paths remain compilable

## Recommended Implementation Modules

- `src/nnc_py/ir/pipeline_schedule.py`
- `src/nnc_py/cost_model/base.py`
- `src/nnc_py/cost_model/simple.py`
- `src/nnc_py/cost_model/cli.py`
- `src/nnc_py/scheduler/base.py`
- `src/nnc_py/scheduler/list_scheduler.py`
- `src/nnc_py/passes/pipeline_step_lowering.py`
- `src/nnc_py/passes/pipeline_scheduling.py`
- `src/nnc_py/passes/memory_planning_v4.py`

Existing files that will need coordinated integration:

- `src/nnc_py/passes/base.py`
- `src/nnc_py/passes/memory_planning.py`
- `src/nnc_py/ir/execution_plan.py`
- `src/nnc_py/codegen/x86_backend.py`
- `src/nnc_py/codegen/c_emitter.py`

## Decision Summary

The project should implement a layered mixed-granularity heterogeneous pipeline
scheduler with:

- a standalone schedule-problem IR
- a pluggable cost-model provider, including external CLI integration
- a replaceable scheduler interface
- a baseline heuristic list scheduler
- time-aware SRAM planning
- schedule-visible `x86` simulation code generation

This gives the project a practical first version while preserving the ability to
hand the scheduling problem to a stronger algorithmic solver later.
