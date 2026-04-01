## Summary

Upgrade the existing external joint tiling/schedule contract so the external
solver is responsible for the complete fast-memory placement problem, including
SRAM fragmentation and offset assignment. Keep the current high-level abstraction
around regions, recipes, values, and actions, but extend the contract so the
solver now returns both dynamic residency objects and offsets for all active
fast-memory items. Do not introduce a parallel `v2` naming scheme; evolve the
existing contract in place.

## Goals

- Keep the collaboration boundary abstract and solver-friendly.
- Move fast-memory offset assignment and fragmentation handling out of the
  compiler allocator and into the external solver.
- Preserve the current compiler-owned responsibilities for problem building,
  validation, materialization, and codegen compatibility.
- Avoid exposing internal `ScheduledMemoryPlanningPass` implementation details
  directly to the external team.

## Non-Goals

- Do not redesign region formation or recipe enumeration.
- Do not add a second public schema family such as `JointProblemV2`.
- Do not keep the current joint path behavior where internal memory planning
  recomputes SRAM offsets after the solver returns.

## Architecture

The existing joint contract remains the only external boundary:

- compiler builds `JointProblem`
- external solver returns `JointSolution` or `JointFailure`
- compiler validates the solution
- compiler materializes the solution back into internal scheduled IR

The key change is that the external solver now owns the complete fast-memory
placement decision:

- recipe selection
- action timing
- dynamic residency generation
- spill/reload selection
- SRAM item offset assignment

The compiler no longer performs fast-memory offset allocation on the joint path.
The current scheduled memory planning stage must be replaced or reduced to
validation/compatibility-only behavior for this path.

## Ownership Boundary

### Compiler-Owned

- region discovery
- recipe construction
- fixed action semantics
- fixed value semantics
- fixed resource constraints
- fixed SRAM item construction for deterministic fast-memory users
- problem validation
- solution validation
- solution materialization
- downstream codegen compatibility

### External Solver-Owned

- selected recipe per region
- action schedule
- generated residency windows
- generated dynamic SRAM items associated with residency
- offset assignment for all active SRAM items
- fragmentation avoidance

## Contract Changes

The existing schema names remain:

- `joint_tiling_schedule_problem_v1`
- `joint_tiling_schedule_solution_v1`

Their contents are extended in place.

### Compatibility Rule

This is an atomic rollout, not a mixed-version protocol.

- a compiler that emits the upgraded `problem_v1` must only talk to a solver that
  understands the upgraded required fields
- a solver that emits the upgraded `solution_v1` must only talk to a compiler
  that validates and materializes the upgraded required fields
- old solvers that merely ignore unknown `problem_v1` fields are not valid peers
  for the upgraded compiler
- old compilers that only understand the pre-upgrade `solution_v1` are not valid
  peers for the upgraded solver

The implementation should therefore treat the new fields as required for the
upgraded v1 contract and fail fast on absence rather than relying on unknown-
field tolerance.

### New Problem Fields

Add to `JointProblem`:

- `sram_items`
- `default_alignment_bytes`

These represent compiler-known fixed fast-memory users and alignment policy.

### New Solution Fields

Add to `JointSolution`:

- `generated_sram_items`
- `sram_allocations`

These let the solver create dynamic SRAM objects, especially residency-backed
objects, and assign offsets to all active fast-memory items.

### Residency Identity

Extend `JointResidencyWindow` to include:

- `residency_id`

The residency identifier is required so generated SRAM items and allocations can
refer to a stable residency object rather than relying on tuple position or
reconstructing identity from `(value_id, start_time, end_time)`.

## SRAM Item Model

Introduce a first-class SRAM allocation object in the external IR.

### `JointSramItem`

Required fields:

- `item_id`
- `kind`
- `size_bytes`
- `alignment_bytes`
- `is_optional`
- `owner_action_id`
- `owner_value_id`
- `owner_residency_id`

Not every owner field is populated for every kind, but each item must have
 exactly one clear semantic owner.

### `JointSramItemKind`

Initial kinds:

- `temp_interval`
- `resident_window`
- `transfer_buffer`

### `JointSramAllocation`

Required fields:

- `item_id`
- `offset`

Offsets are solver outputs and become the authoritative fast-memory layout for
the joint path.

### Residency Cardinality

The contract is one-to-one:

- each `resident_window` SRAM item must reference exactly one `residency_id`
- each `residency_window` must have exactly one corresponding `resident_window`
  SRAM item

Zero-item and multi-item residency windows are not allowed in the upgraded
contract.

## Item Construction Policy

### Compiler-Declared Fixed Items

The compiler constructs `JointProblem.sram_items` only for deterministic
fast-memory users:

- compute temp intervals
- explicit transfer staging buffers when required by the lowered semantics
- optional-action-owned fixed items whose identity is known before solve time
  even if activation is solver-controlled

For compute temp intervals, there is no solver freedom around existence or time
range. The solver only chooses the offset.

### Solver-Generated Dynamic Items

The solver generates only residency-backed items in
`JointSolution.generated_sram_items`.

The solver also generates the corresponding `residency_windows`. The compiler
does not pre-enumerate candidate residency item slots.

This keeps the contract flexible for:

- zero windows
- one window
- multiple windows per value
- spill/reload-driven re-entry into SRAM

### Exhaustive Ownership Rule

Ownership of SRAM items is exhaustive and disjoint:

- `problem.sram_items`
  contains every non-residency item whose semantic identity is already known at
  problem-build time
- `solution.generated_sram_items`
  contains only `resident_window` items created by the solver

This means:

- compute temp items are always problem-declared
- transfer buffer items, including those tied to optional spill/reload actions,
  are still problem-declared if their semantic identity is known before solving
- the solver never invents new temp or transfer item kinds
- the solver only invents residency-backed items

## Validation Rules

Extend joint solution validation with fast-memory placement checks.

### Structural Checks

- every active fixed SRAM item has exactly one allocation
- every generated SRAM item has exactly one allocation
- allocations may only reference known problem items or generated solution items
- generated residency-backed items must reference a valid `residency_id`

### Alignment Checks

- every allocated offset satisfies the item alignment
- `alignment_bytes` is required on every SRAM item
- `default_alignment_bytes` exists to constrain compiler-built defaults and for
  validation sanity checks, not as a runtime omission fallback

### Capacity Checks

- for every allocated item: `offset >= 0`
- for every allocated item: `offset + size_bytes <= sram_capacity_bytes`

### Overlap Checks

For any two active items whose time intervals overlap:

- their address intervals must not overlap

This is the rule that upgrades the external contract from pure capacity
accounting to true fragmentation-aware placement.

### Lifetime Semantics

All item lifetimes are half-open intervals `[start_time, end_time)`.

- action-owned item lifetimes include the action launch overhead and therefore
  span the full scheduled action interval
- `temp_interval` lifetime is exactly the owning compute action interval
- `transfer_buffer` lifetime is exactly the owning transfer action interval
- `resident_window` lifetime is exactly the owning residency window interval

Solver, validator, and materializer must all use the same half-open interval
rule when evaluating overlap.

### Semantic Checks

- each `temp_interval` item must match its owning action execution interval
- each `resident_window` item must match its owning residency window interval
- each `transfer_buffer` item must match its owning transfer action semantics
- items for optional actions may only exist when those actions are active

## Materialization Changes

The joint materialization stage must stop inventing placeholder SRAM intervals
such as `joint_buf_<index>` without offsets.

Instead it must:

- import the solver-provided SRAM items and offsets
- extend the internal carrier so imported offsets are preserved explicitly
- construct internal `SramAllocationInterval` objects with stable identity and
  imported offset information
- emit compatibility metadata needed by downstream runtime/codegen readers
- preserve the solver’s offsets without recomputation

Materialization remains compiler-owned, but it becomes an import/conversion step
instead of a pre-allocation step.

Concretely, the implementation must extend the current internal schedule IR
carrier before this is implementable:

- `SramAllocationInterval` must gain an explicit `offset` field and stable item
  identity fields sufficient to distinguish residency, temp, and transfer items
- the imported joint-path memory result must carry enough information to build
  downstream compatibility metadata without running a fresh allocator

## Pipeline Changes

The current joint O3 path ends with:

- `JointTilingScheduleMaterializationPass`
- `LivenessAnalysisPass`
- `ScheduledMemoryPlanningPass`

This is no longer correct once the solver owns offsets.

The joint path must be changed so that the current scheduled memory planning pass
does not perform allocation on this path.

Two acceptable implementation directions:

1. Replace it with a dedicated joint-memory import/validation pass.
2. Refactor the scheduled memory planning pass so the joint path only uses its
   compatibility/export logic and never its allocator.

The preferred direction is a dedicated joint-path import/validation pass because
it keeps the ownership boundary explicit and avoids accidental offset rewrites.

## Internal Compatibility Target

After materialization/import, the joint path still needs to feed the existing
downstream codegen path. The imported result must therefore be sufficient to
produce:

- internal schedule steps
- scheduled values
- residency windows
- solver-authored SRAM interval placement
- any compatibility memory metadata required by generated C/X86 output

The downstream codegen should remain unaware of whether offsets came from the
internal allocator or the external solver.

The implementation must name one authoritative internal representation for this
imported placement. The recommended choice is:

- extend `PipelineScheduleResult.sram_intervals` so it can carry solver-authored
  offsets and item identity directly
- build joint-path compatibility metadata from that imported placement instead
  of from `ScheduledMemoryPlanningPass` allocator output

## Testing

Add or extend tests to cover:

- IR round-trip for new SRAM item and allocation fields
- problem builder output for fixed SRAM items
- solution validation for:
  - missing allocations
  - misalignment
  - capacity overflow
  - overlapping time-overlapping items
  - residency item ownership mismatches
- materialization preserving solver offsets
- joint-path pass integration confirming no internal fast-memory reallocation
- end-to-end compile/build coverage with external solver-provided offsets

## Migration Constraint

Do not introduce `V2` type names or schema identifiers. The existing joint
contract is upgraded directly. All code paths, tests, and documentation should
use the current names.
