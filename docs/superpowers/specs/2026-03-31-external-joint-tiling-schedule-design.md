# External Joint Tiling Schedule Design

## Status

Proposed and approved at the design level on 2026-03-31.

## Goal

Refactor the current scheduled `-O3` path so an external optimization team can
solve a reusable, abstract problem that jointly chooses:

- tiling strategy
- global schedule
- SRAM residency
- spill / reload actions

The compiler must retain ownership of legality, parameter estimation,
validation, materialization, and code generation.

## Problem

`nnc-py` already has a useful internal schedule IR:

- `PipelineScheduleProblem`
- `PipelineScheduleResult`
- schedule-aware memory planning

This is a strong foundation, but it is not yet the right collaboration boundary
for an external optimization team.

Today, the compiler still bakes in too many decisions before solving:

- tiling is largely fixed before scheduling
- spill / reload insertion is decided inside the compiler
- the problem shape is tied to current pass structure

That boundary is too low-level and too coupled to compiler engineering.

The collaboration target is different:

- the optimization team should work on a clean mathematical problem
- the compiler team should own all engineering-heavy legality and lowering work
- the interface should remain reusable even if internal passes evolve

## Design Principles

- Expose optimization decisions, not compiler mechanics
- Keep the external contract versioned and JSON-serializable
- Allow both single-op and fused-group tiling decisions
- Preserve a strict compiler-side legality boundary
- Keep final byte-offset allocation out of the external solver
- Require compiler-side validation of every external solution

## Recommended Architecture

Introduce a new external-facing optimization layer above the existing internal
pipeline schedule IR.

The system is split into four responsibilities:

1. `RegionBuilder`
   - compiler-owned
   - partitions the graph into `tileable region` units
   - a region may be a single op or a fused group

2. `RecipeEnumerator`
   - compiler-owned
   - enumerates a small set of legal tiling/layout execution candidates for
     each region
   - each candidate is a `recipe`

3. `External Joint Optimization Problem`
   - shared contract between compiler and optimization team
   - purely declarative and versioned
   - contains regions, recipes, values, actions, resources, and budgets

4. `SolutionValidator` and `SolutionMaterializer`
   - compiler-owned
   - validates the external solution
   - converts it into existing internal schedule and memory-plan artifacts

## Decision Unit

The optimization decision unit is `tileable region`.

A `tileable region`:

- may be a single operator
- may be a pre-defined fused group such as `conv + add + relu`
- is created by compiler logic before the external problem is built

The external solver does not decide which nodes are legal to group. It only
chooses among legal recipes for already-defined regions.

## Why Regions Instead of Raw Nodes

At the tiling stage, fused groups and single ops are materially different:

- fused groups have cross-op legality constraints
- fused groups can avoid materializing internal intermediates
- fused groups change SRAM pressure and DMA behavior more than single-op tiling
- fused-group cost is not equal to the sum of per-op standalone cost

Because of that, the optimization model must support both single-op and
fused-group units while treating them as the same abstract object type:
`tileable region`.

## External Problem Schema

Define a new versioned schema, for example:

- `joint_tiling_schedule_problem_v1`

This schema is not a direct export of `PipelineScheduleProblem`. It is a more
stable, higher-level collaboration boundary.

### Core Objects

#### `Region`

Represents one optimization decision unit.

Suggested fields:

- `region_id`
- `kind`
- `member_nodes`
- `input_value_ids`
- `output_value_ids`
- `predecessor_region_ids`
- `successor_region_ids`

#### `Recipe`

Represents one legal execution strategy for a region.

Suggested fields:

- `recipe_id`
- `region_id`
- `tile_spec`
- `layout_spec`
- `action_template`
- `value_footprint`
- `cost_parameters`
- `compatibility_tags`

Each recipe must already be legal. The external solver selects recipes; it does
not prove tile legality.

#### `Value`

Represents a global memory object in the optimization problem.

Suggested fields:

- `value_id`
- `size_bytes`
- `home_tier`
- `producer`
- `consumers`
- `must_keep`
- `spillable`

`Value` is not restricted to graph tensors. It may describe region outputs,
staged values, or other scheduled memory objects that matter to the joint
optimization problem.

#### `Action`

Represents a schedulable atomic activity.

Suggested kinds:

- `compute`
- `dma_in`
- `dma_out`
- `spill`
- `reload`

Suggested fields:

- `action_id`
- `region_id`
- `recipe_id`
- `resource_kind`
- `duration`
- `launch_overhead`
- `predecessor_action_ids`
- `reads`
- `writes`
- `temp_bytes`

#### `ResourceModel`

Represents globally shared execution resources.

Initial design assumption:

- one `DMA` channel
- one `MATMUL` slot
- one `SHAPE` slot
- one `OTHER` slot

This matches current project assumptions and keeps the first formulation
aligned with existing infrastructure.

#### `Budget` and `Objective`

Suggested fields:

- `sram_capacity_bytes`
- `objective = min_makespan`

Optional secondary objectives may exist, such as fewer spills or reloads, but
must only serve as tie-breakers behind makespan.

## Recipe Design

`Recipe` is the key abstraction. It must preserve the effect of tiling without
exposing compiler internals.

Each recipe should capture:

- tile shape and tile-domain tags
- relevant layout tags
- the local action pattern the region would execute
- intermediate and scratch memory footprint
- cost parameters for the local action sequence
- only the minimal compatibility information needed to compose with neighbors

To keep the external problem size reasonable, the compiler should emit a small
Pareto-pruned candidate set per region rather than a large raw search space.

Recommended first target:

- roughly 4 to 12 recipes per region

## Solver Responsibilities

The external solver owns only optimization decisions.

It is responsible for:

- selecting one recipe per region
- globally scheduling actions
- choosing SRAM residency windows
- choosing legal spill / reload behavior for spillable values

It is not responsible for:

- graph partitioning into regions
- recipe legality generation
- duration or size estimation
- final byte-offset allocation
- code generation
- pass orchestration

## Compiler Responsibilities

The compiler retains responsibility for:

- region construction
- recipe enumeration
- legality filtering
- cost and memory parameter estimation
- schema serialization
- external solver invocation
- solution validation
- materialization into existing internal artifacts
- final physical memory offset allocation
- backend code generation

## Solver Output Contract

Define a separate versioned schema, for example:

- `joint_tiling_schedule_solution_v1`

The output must be declarative, not procedural.

Suggested fields:

- `selected_recipes`
- `scheduled_actions`
- `enabled_transfers`
- `residency_windows`
- `objective_value`
- `diagnostics`

### `selected_recipes`

Maps each `region_id` to exactly one `recipe_id`.

### `scheduled_actions`

Provides:

- `action_id`
- `start_time`
- `end_time`
- `resource_slot`

### `enabled_transfers`

Identifies which spill and reload transfers are selected.

### `residency_windows`

Describes when each value resides in SRAM.

### `diagnostics`

May include:

- solver status
- optimality gap
- lower bound
- timeout information

The solver output must not include final SRAM or slow-memory byte offsets.
Those remain compiler-owned materialization details.

## Validation Requirements

Every external solution must be checked by a compiler-owned validator before it
can influence lowering or code generation.

The validator must at least enforce:

- every region selects exactly one recipe
- every scheduled action belongs to a selected recipe or a legal transfer
- dependency constraints are satisfied
- resource-slot overlap is legal
- SRAM capacity is never exceeded
- residency windows are consistent with action reads and writes
- spill and reload actions are only used on allowed values
- final required outputs are produced and reachable

## Standard Failure Semantics

Validation and solver integration must use stable failure categories rather than
project-internal exception strings.

Recommended categories:

- `invalid_solution`
- `incomplete_solution`
- `dependency_violation`
- `resource_overlap`
- `sram_capacity_exceeded`
- `illegal_transfer`
- `incompatible_recipe_boundary`
- `solver_reported_infeasible`

## Integration with Current Project

The current internal IR remains useful as a lower layer.

Recommended direction:

- keep `PipelineScheduleProblem` and `PipelineScheduleResult` as internal
  schedule/materialization structures
- add the new external joint optimization schema above them
- materialize the external solution back into the current scheduled pipeline
  path

This means the existing scheduled-O3 pipeline can evolve from:

- fixed tiling
- compiler-chosen spill insertion
- direct schedule-problem construction

toward:

- compiler-built regions
- compiler-built legal recipes
- external joint recipe/schedule/residency optimization
- compiler validation and materialization

## Migration Plan

### Phase 1: External Schema and Adapter

- introduce versioned problem and solution schemas
- add JSON serialization and deserialization
- add an external solver adapter, likely using stdin/stdout JSON exchange

### Phase 2: Region and Recipe Layer

- add `RegionBuilder`
- add `RecipeEnumerator`
- emit regions and Pareto-pruned recipe candidates

### Phase 3: Validation and Materialization

- add `SolutionValidator`
- add `SolutionMaterializer`
- lower validated external solutions into current internal schedule artifacts

### Phase 4: Baseline Internal Solver Compatibility

- provide an internal baseline heuristic that consumes the new schema
- keep regression tests independent from any external solver dependency

### Phase 5: Replace Direct Fixed-Problem Construction

- progressively retire direct fixed schedule-problem construction for the
  scheduled O3 path
- derive internal schedule artifacts from selected recipes instead

## Testing Strategy

Testing should be structured around boundaries, not implementation details.

Required coverage:

- schema round-trip tests
- region and recipe enumeration contract tests
- validator rejection tests for each standard failure category
- materialization tests that confirm compatibility with existing scheduled
  memory planning and code generation
- baseline internal-solver tests on the new schema
- end-to-end scheduled O3 tests using the adapter path

## Non-Goals

- exposing compiler passes or metadata structures to the external team
- making the external solver responsible for legality proofs
- requiring the external solver to produce final physical offsets
- supporting arbitrary dynamic re-grouping of graph regions in the first version
- solving continuous tiling search directly in the first version

## Recommended Outcome

The recommended collaboration boundary is:

- external team solves a versioned `region + recipe + action + value + budget`
  optimization problem
- compiler team owns legality, estimation, validation, materialization, and
  codegen

This preserves the importance of tiling in the objective while preventing the
external team from spending time on compiler engineering details.
