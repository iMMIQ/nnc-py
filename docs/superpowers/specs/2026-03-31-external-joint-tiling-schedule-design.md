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

## Normative v1 Contract

`joint_tiling_schedule_problem_v1` must be a versioned JSON object with these
required top-level fields:

- `schema_version`
- `regions`
- `recipes`
- `values`
- `actions`
- `boundary_constraints`
- `dependency_edges`
- `resources`
- `sram_capacity_bytes`
- `objective`

V1 contract rules:

- `regions`, `recipes`, `values`, `actions`, `boundary_constraints`,
  `dependency_edges`, and `resources` are JSON arrays of objects
- all IDs are strings and unique within their object class
- all integer quantities are non-negative unless explicitly documented
- all times are integer-valued
- required fields must always be present, even when their array value is empty
- nullability must be explicit and stable
- any additional fields must be documented as forward-compatible extensions
- consumers must ignore unknown fields at any object level when
  `schema_version` is a recognized v1 literal

`joint_tiling_schedule_solution_v1` must be a versioned JSON object with these
required top-level fields:

- `schema_version`
- `selected_recipes`
- `scheduled_actions`
- `residency_windows`
- `objective_value`
- `diagnostics`

V1 solution rules:

- `selected_recipes`, `scheduled_actions`, and `residency_windows` are JSON
  arrays of objects
- `selected_recipes` contains exactly one recipe per region
- `scheduled_actions` contains every mandatory active action exactly once
- `scheduled_actions` may additionally contain optional actions chosen by the
  solver, each at most once
- any action omitted from `scheduled_actions` is interpreted as inactive
- every `residency_window` refers to a declared `value_id`

`joint_tiling_schedule_failure_v1` must be a versioned JSON object with these
required top-level fields:

- `schema_version`
- `status`
- `error_category`
- `diagnostics`

Minimum required object fields in v1:

- `Region`
  - required: `region_id`, `kind`, `input_value_ids`, `output_value_ids`
  - optional: `member_nodes`, `predecessor_region_ids`, `successor_region_ids`
- `Recipe`
  - required: `recipe_id`, `region_id`, `tile_spec`, `layout_spec`,
    `activates_action_ids`, `value_footprint`, `cost_parameters`
- `Value`
  - required: `value_id`, `size_bytes`, `initial_tier`, `required_final_tier`,
    `must_keep`, `spillable`, `allows_multiple_sram_windows`, `producer`,
    `consumers`
- `Action`
  - required: `action_id`, `kind`, `resource_kind`, `duration`,
    `launch_overhead`, `reads`, `writes`, `temp_bytes`, `is_optional`,
    `region_id`, `recipe_id`, `optional_value_id`
- `BoundaryConstraint`
  - required: `boundary_id`, `src_region_id`, `dst_region_id`,
    `compatible_recipe_pairs`
  - optional: `required_layout_relations`, `required_tile_domain_relations`
- `DependencyEdge`
  - required: `src_action_id`, `dst_action_id`, `kind`
  - optional: none in v1
- `ResourceModel`
  - required: `resource_kind`, `slot_count`
  - fixed in v1: `slot_count = 1`
- `SelectedRecipe`
  - required: `region_id`, `recipe_id`
- `ScheduledAction`
  - required: `action_id`, `start_time`
- `ResidencyWindow`
  - required: `value_id`, `start_time`, `end_time`

Presence and nullability rules in v1:

- required fields must always be present, even when their value is `null` or `[]`
- `Value.producer` is required and may be `null`
- `Value.consumers` is required and may be `[]`
- `Action.region_id`, `Action.recipe_id`, and `Action.optional_value_id` are
  required and may be `null`

Normative field shapes in v1:

- `schema_version`
  - problem literal: `"joint_tiling_schedule_problem_v1"`
  - solution literal: `"joint_tiling_schedule_solution_v1"`
- `Region.kind`
  - enum: `"single_op"` | `"fused_group"`
- `DependencyEdge.kind`
  - enum: `"data"` | `"order"`
- `tile_spec`
  - object with required fields:
    - `axes`: array of strings
    - `shape`: array of non-negative integers
- `layout_spec`
  - object with required field:
    - `layout_tags`: array of strings
- `value_footprint`
  - object with required fields:
    - `resident_bytes`: non-negative integer
    - `scratch_bytes`: non-negative integer
    - `transfer_bytes`: non-negative integer
- `cost_parameters`
  - object with required fields:
    - `latency`: non-negative integer
    - `launch_overhead`: non-negative integer
- `reads`
  - array of logical `value_id` strings
- `writes`
  - array of logical `value_id` strings
- `producer`
  - either `null` or object `{ "action_id": string }`
- `consumers`
  - array of objects `{ "action_id": string }`
- `compatible_recipe_pairs`
  - array of objects
    `{ "src_recipe_id": string, "dst_recipe_id": string }`
- `required_layout_relations`
  - array of strings
- `required_tile_domain_relations`
  - array of strings

Dataflow consistency invariants in v1:

- `Region.input_value_ids` and `Region.output_value_ids` define the region-level
  interface only
- `Value.producer` and `Value.consumers` define the authoritative action-level
  producer and consumer set for each value
- `Action.reads` and `Action.writes` define the authoritative per-action value
  access set used by legality, liveness, and residency checks
- every `value_id` in `Region.input_value_ids` or `Region.output_value_ids` must
  also appear in the global `values` array
- if a region lists a `value_id` in `output_value_ids`, then some action in that
  region must write that value or the region interface is invalid
- if a region lists a `value_id` in `input_value_ids`, then some action in that
  region must read that value or the region interface is invalid
- `Value.producer.action_id`, when non-null, must refer to exactly one action
  whose `writes` includes that `value_id`
- every `Value.consumers[*].action_id` must refer to an action whose `reads`
  includes that `value_id`
- if these representations disagree, `Action.reads`/`writes` and
  `Value.producer`/`consumers` take precedence over region interface metadata,
  and the problem instance is invalid

### Core Objects

#### `Region`

Represents one optimization decision unit.

V1 fields:

- `region_id`
- `kind`
- `member_nodes`
- `input_value_ids`
- `output_value_ids`
- `predecessor_region_ids`
- `successor_region_ids`

Region edges provide the coarse producer-consumer structure. They are not the
final authoritative schedule graph by themselves.

In v1:

- `predecessor_region_ids` and `successor_region_ids` are informational only
- authoritative region adjacency is defined by `BoundaryConstraint` objects

#### `Recipe`

Represents one legal execution strategy for a region.

V1 fields:

- `recipe_id`
- `region_id`
- `tile_spec`
- `layout_spec`
- `activates_action_ids`
- `value_footprint`
- `cost_parameters`

Each recipe must already be legal. The external solver selects recipes; it does
not prove tile legality.

The problem must use one canonical action model:

- the compiler emits explicit candidate actions and transfer actions in the
  problem object
- each recipe declares which action IDs become active when that recipe is
  selected
- the mandatory active action set is exactly the union of
  `activates_action_ids` for the selected recipes
- the external solver is not allowed to synthesize new actions

Activation invariants in v1:

- every non-optional action must have non-null `recipe_id`
- every non-optional action ID must appear in exactly one recipe's
  `activates_action_ids`
- that unique recipe ID must equal the action's `recipe_id`
- if these invariants are violated, the problem instance is invalid

This keeps scheduling and transfer decisions declarative and validator-friendly.

#### `Value`

Represents a global memory object in the optimization problem.

V1 fields:

- `value_id`
- `size_bytes`
- `initial_tier`
- `required_final_tier`
- `producer`
- `consumers`
- `must_keep`
- `spillable`
- `allows_multiple_sram_windows`

`Value` is not restricted to graph tensors. It may describe region outputs,
staged values, or other scheduled memory objects that matter to the joint
optimization problem.

Value semantics must be explicit:

- `initial_tier` defines where the value exists before any scheduled action
- `required_final_tier` defines where the value must exist after execution
- `must_keep` means the value must remain available without spill across its
  required live interval
- `spillable` means compiler legality permits spill and reload actions
- `must_keep=true` and `spillable=true` is illegal
- `allows_multiple_sram_windows` determines whether the value may leave and
  later re-enter SRAM

V1 tier enum is fixed to:

- `unmaterialized`
- `input`
- `const`
- `slow`
- `sram`

Tier rules:

- `initial_tier` and `required_final_tier` must use the v1 tier enum
- `initial_tier = unmaterialized` means the value does not exist in any tier
  before its declared producer action completes
- `reads` and `writes` on actions always refer to logical `value_id`s, not
  tier-specific copies
- transfer actions carry tier movement through their `kind`
- `dma_in` moves a value from `input` or `const` into `sram`
- `spill` moves a spillable value from `sram` to `slow`
- `reload` moves a spillable value from `slow` to `sram`
- `dma_out` makes a produced value available in its declared final non-SRAM
  backing tier

Producer rules:

- if `producer = null`, the value must start in its declared `initial_tier`
- if `producer != null`, the value must use `initial_tier = unmaterialized`
- a produced value first becomes available only at the `end_time` of its
  producer action or of a later transfer action that materializes it in another
  tier

Final-tier legality rules:

- `required_final_tier = unmaterialized` is illegal in v1
- if `producer != null`, `required_final_tier` must be `slow` or `sram`
- if `producer = null` and `initial_tier = input`, then `required_final_tier`
  must be `input` or `sram`
- if `producer = null` and `initial_tier = const`, then `required_final_tier`
  must be `const` or `sram`
- if `producer = null` and `initial_tier = slow`, then `required_final_tier`
  must be `slow` or `sram`
- if `producer = null` and `initial_tier = sram`, then `required_final_tier`
  must be `slow` or `sram`

Transfer legality rules:

- `dma_in`
  - source value is available in its declared `input` or `const` backing tier
    for the full action interval
  - destination SRAM availability begins at the action `end_time`
  - source backing remains available after transfer
- `spill`
  - source value must be resident in SRAM for the full action interval
  - slow-tier availability begins at the action `end_time`
  - the associated SRAM residency window may end at the same `end_time` or
    later if another rule requires continued residence
- `reload`
  - source value must already be available in slow tier before action start
  - destination SRAM availability begins at the action `end_time`
  - slow-tier backing remains available after reload
- `dma_out`
  - source value must be resident in SRAM for the full action interval
  - required final slow-tier backing availability begins at the action
    `end_time`

Any solution that violates these transfer rules is `illegal_transfer`.

Per-kind `reads` and `writes` rules:

- `compute`
  - `reads` lists logical values that must be resident in SRAM for the full
    action interval
  - `writes` lists logical values produced by the compute action
  - if a produced value is intended to become SRAM-resident, its first
    residency window may start at the compute action `end_time`
- `dma_in`
  - `reads` contains exactly one logical value whose current backing is
    `input` or `const`
  - `writes` contains that same logical value and opens an SRAM residency
    window at action `end_time`
- `spill`
  - `reads` contains exactly one logical value currently resident in SRAM
  - `writes` contains that same logical value and makes it available in `slow`
    at action `end_time`
- `reload`
  - `reads` contains exactly one logical value currently available in `slow`
  - `writes` contains that same logical value and opens an SRAM residency
    window at action `end_time`
- `dma_out`
  - `reads` contains exactly one logical value currently resident in SRAM
  - `writes` contains that same logical value and makes it available in `slow`
    at action `end_time`

Residency and liveness rules:

- there is no implicit SRAM residence in v1
- if a value is available in SRAM at time `0`, the solution must contain a
  residency window with `start_time = 0`
- if `required_final_tier = sram`, the solution must contain a final residency
  window whose `end_time = objective_value`
- `must_keep=true` means once the value becomes resident in SRAM, it must remain
  resident until the end of its last active consumer action
- the last active consumer is computed from the active action set selected by
  recipes plus any scheduled optional actions
- every non-initial SRAM window after the first must begin at the `end_time` of
  an active `compute`, `dma_in`, or `reload` action that writes the value
- every gap between two SRAM windows for a spillable value must be justified by
  an active `spill` ending the earlier window and, if the value returns to
  SRAM, an active `reload` opening the next window

Optional transfer multiplicity rules:

- the compiler must predeclare every optional spill and reload action instance
  that may be used in v1
- each optional transfer action instance may be scheduled at most once
- the solver may not infer additional spill or reload instances from residency
  windows alone

The first version should explicitly classify inputs, constants, staged
intermediates, and required outputs using these fields.

#### `Action`

Represents a schedulable candidate atomic activity.

V1 `kind` enum:

- `compute`
- `dma_in`
- `dma_out`
- `spill`
- `reload`

V1 fields:

- `action_id`
- `region_id`
- `recipe_id`
- `kind`
- `resource_kind`
- `duration`
- `launch_overhead`
- `reads`
- `writes`
- `temp_bytes`
- `is_optional`
- `optional_value_id`

Schedule semantics are fixed:

- time is discrete integer time
- all intervals use half-open semantics `[start_time, end_time)`
- all actions are non-preemptive
- each action occupies exactly one resource kind for its full active duration
- `occupied_duration = duration + launch_overhead`
- `end_time = start_time + occupied_duration`
- the solution may report `end_time`, but the validator must recompute it from
  `start_time` and action parameters
- actions may overlap only if they use different resource kinds and all
  dependency and memory constraints remain satisfied

In the first version, multi-resource actions are out of scope.

Action activation rules are fixed:

- `is_optional=false` means the action is mandatory if and only if its owning
  recipe is selected
- `is_optional=true` means the action is an explicit optional candidate that the
  solver may schedule only if its referenced values and dependencies are legal
- spill and reload actions are the only optional actions in v1
- mandatory recipe-owned actions must have non-null `recipe_id`
- optional transfer actions may use `recipe_id = null` and must use
  non-null `optional_value_id`

The active action set in v1 is exactly:

- all non-optional actions whose `recipe_id` belongs to the selected recipe set
- all optional actions that appear in `scheduled_actions`

#### `BoundaryConstraint`

Represents compatibility rules across a region edge.

V1 fields:

- `boundary_id`
- `src_region_id`
- `dst_region_id`
- `compatible_recipe_pairs`
- `required_layout_relations`
- `required_tile_domain_relations`

The authoritative compatibility rule is:

- a region pair `(src_region_id, dst_region_id)` is adjacent in v1 if the two
  regions share a declared interface value, meaning some `value_id` appears in
  both `src.output_value_ids` and `dst.input_value_ids`
- adjacent regions may only select recipe pairs explicitly allowed by the
  boundary constraint set
- every adjacent ordered region pair must have exactly one `BoundaryConstraint`
  object in v1
- omission of a boundary constraint is illegal, not unconstrained

This makes `incompatible_recipe_boundary` a precise validator outcome rather
than an implied interpretation of free-form tags.

In v1:

- `compatible_recipe_pairs` is the only normative compatibility field used by
  validation
- `required_layout_relations` and `required_tile_domain_relations` are
  descriptive metadata for debugging, reporting, and future schema revisions
- validators may ignore those relation arrays when deciding legality

#### `DependencyEdge`

Represents an authoritative precedence relation between candidate actions.

V1 fields:

- `src_action_id`
- `dst_action_id`
- `kind`

The precedence model must have one single source of truth:

- region edges define coarse graph structure
- recipes activate explicit action IDs
- boundary constraints define which recipe pairs may coexist
- dependency edges define the exact action-level precedence graph checked by the
  solver and validator

The compiler is responsible for generating all action-level dependency edges
before exporting the external problem.

`DependencyEdge` is the only authoritative precedence representation in v1.
Action objects must not carry an independent predecessor list.

Edge activation rules are fixed:

- a dependency edge is active if and only if both endpoint actions are active
- edges incident to inactive optional actions are ignored by the validator
- a mandatory active action that requires an inactive optional predecessor is
  still invalid unless its reads, writes, and residency requirements are
  satisfied through other active actions and legal value state
- an active dependency edge is satisfied if and only if
  `end_time(src_action_id) <= start_time(dst_action_id)`
- equality is legal; strict separation is not required

#### `ResourceModel`

Represents globally shared execution resources.

V1 required resources:

- one `DMA` channel
- one `MATMUL` slot
- one `SHAPE` slot
- one `OTHER` slot

This matches current project assumptions and keeps the first formulation
aligned with existing infrastructure.

The first version assumes:

- one slot per listed resource kind
- no action may migrate between resource kinds
- no action may occupy more than one resource kind simultaneously

Because v1 has one slot per resource kind, `resource_slot` does not appear in
the normative v1 solution schema.

`resources` is an array of `ResourceModel` objects, each with:

- `resource_kind`
- `slot_count`

V1 requires exactly these resource entries:

- `DMA` with `slot_count = 1`
- `MATMUL` with `slot_count = 1`
- `SHAPE` with `slot_count = 1`
- `OTHER` with `slot_count = 1`

#### `Budget` and `Objective`

V1 fields:

- `sram_capacity_bytes`
- `objective`

Optional secondary objectives may exist, such as fewer spills or reloads, but
must only serve as tie-breakers behind makespan.

In v1:

- `objective` is the string literal `"min_makespan"`
- `objective_value` in the solution is the makespan
- `objective_value = max(end_time)` over all scheduled actions
- if no actions are scheduled, `objective_value = 0`
- residency windows may end at `objective_value` but do not extend the makespan
- any secondary tie-break information belongs in `diagnostics`, not in
  `objective_value`

## Recipe Design

`Recipe` is the key abstraction. It must preserve the effect of tiling without
exposing compiler internals.

Each recipe should capture:

- tile shape and tile-domain tags
- relevant layout tags
- the explicit action IDs the recipe activates
- intermediate and scratch memory footprint
- cost parameters for the local action sequence
- explicit compatibility information needed to compose with neighbors

Recipe-local action ordering must not remain implicit in prose. It must be
exported through the problem's explicit `DependencyEdge` set.

In v1, `value_footprint` is descriptive metadata for recipe selection and
cross-checking only. It does not add a separate SRAM legality term beyond:

- resident `Value.size_bytes`
- active `Action.temp_bytes`

In v1, `cost_parameters` is also descriptive metadata:

- schedule legality and makespan are computed only from active action timing
- validators do not recompute or enforce recipe-level `latency` or
  `launch_overhead`
- compiler and solver may use recipe-level cost metadata for ranking, warm
  starts, heuristics, or diagnostics

To keep the external problem size reasonable, the compiler should emit a small
Pareto-pruned candidate set per region rather than a large raw search space.

Recommended first target:

- roughly 4 to 12 recipes per region

## Solver Responsibilities

The external solver owns only optimization decisions.

It is responsible for:

- selecting one recipe per region
- globally scheduling active actions
- choosing SRAM residency windows for values that may reside in fast memory
- choosing among legal optional spill / reload actions for spillable values

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

V1 fields:

- `selected_recipes`
- `scheduled_actions`
- `residency_windows`
- `objective_value`
- `diagnostics`

### `selected_recipes`

Maps each `region_id` to exactly one `recipe_id`.

`selected_recipes` is an array of `SelectedRecipe` objects, each with:

- `region_id`
- `recipe_id`

### `scheduled_actions`

Provides:

- `action_id`
- `start_time`

`end_time` is derived from the problem definition and does not need to be a
free solver decision. The solution may include it for convenience, but the
validator must recompute and verify it.

The v1 contract requires:

- every mandatory active action appears exactly once
- every optional scheduled action appears exactly once
- omitted optional actions are treated as inactive
- actions whose owner recipe is not selected must not appear

`scheduled_actions` is an array of `ScheduledAction` objects, each with:

- `action_id`
- `start_time`

### `residency_windows`

Describes when each value resides in SRAM.

Residency semantics must allow multiple disjoint windows only when the
corresponding value declares `allows_multiple_sram_windows=true`.

Each residency window uses half-open integer semantics:

- `window = [start_time, end_time)`
- `start_time < end_time`

`residency_windows` is an array of `ResidencyWindow` objects, each with:

- `value_id`
- `start_time`
- `end_time`

Normalization rules:

- windows for the same `value_id` must be strictly non-overlapping
- windows for the same `value_id` must be non-adjacent; adjacent windows must
  be merged before validation
- because overlapping windows for one value are illegal, capacity counting sees
  at most one active window per value at any time

The validator must use these rules:

- for `compute` actions, every read value `v` must be resident in SRAM for the
  full interval of the action
- transfer actions use the per-kind source and destination availability rules
  defined above rather than the generic compute-read rule
- a producing action makes its written values available at the end of the
  action interval
- `temp_bytes` counts against SRAM capacity for the full interval of the action
- for every integer time `t`, total resident value bytes covering `t` plus
  total `temp_bytes` of active actions covering `t` must not exceed
  `sram_capacity_bytes`

### `diagnostics`

`diagnostics` is a JSON object. Recommended fields:

- solver status
- optimality gap
- lower bound
- timeout information

The solver output must not include final SRAM or slow-memory byte offsets.
Those remain compiler-owned materialization details.

## Failure Response Contract

For v1 over a JSON stdin/stdout adapter:

- a successful solve returns `joint_tiling_schedule_solution_v1` on stdout and
  process exit code `0`
- an expected non-solution outcome returns
  `joint_tiling_schedule_failure_v1` on stdout and process exit code `0`
- malformed input, transport failure, or internal solver crash returns a
  non-zero process exit code and may emit human-readable stderr

`joint_tiling_schedule_failure_v1` fields:

- `schema_version = "joint_tiling_schedule_failure_v1"`
- `status`
  - enum: `"infeasible"` | `"timeout"` | `"invalid_problem"` | `"error"`
- `error_category`
  - one of the normative v1 failure categories listed below when applicable
- `diagnostics`
  - JSON object with solver-specific details

## Validation Requirements

Every external solution must be checked by a compiler-owned validator before it
can influence lowering or code generation.

The validator must at least enforce:

- every region selects exactly one recipe
- every scheduled action is either:
  - a mandatory action activated by the selected recipe set, or
  - a legal optional action explicitly chosen by the solution
- every optional spill or reload action used by the solution is legal for the
  corresponding value
- dependency constraints are satisfied
- same-resource action overlap is illegal
- SRAM capacity is never exceeded
- residency windows are consistent with action reads and writes
- selected adjacent recipes satisfy all boundary constraints
- final required outputs are produced and reachable

## Standard Failure Semantics

Validation and solver integration must use stable failure categories rather than
project-internal exception strings.

Normative `error_category` enum in v1:

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

The intended layering is:

- external schema carries regions, recipes, candidate actions, dependency edges,
  values, boundary constraints, resources, and budgets
- internal schedule IR remains the lowered representation consumed by current
  schedule-aware memory planning and code generation

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
- emit regions, Pareto-pruned recipe candidates, explicit candidate actions,
  dependency edges, and boundary constraints

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
