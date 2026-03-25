# Scheduled O3 Native Spill Design

## Goal

Make `-O3` a single scheduled compilation path that natively supports `--max-memory` by modeling spill and reload as real DMA work. Remove all legacy spill and fallback semantics from O3 so the generated result stays legal, predictable, and aligned with the actual parallel C runtime.

## Problem Statement

The current O3 path mixes two incompatible worlds:

- scheduled O3 produces staged SRAM values such as `sram|node|...|tensor|...`
- legacy spill analysis expects graph-tensor and graph-node semantics

This creates illegal internal states under `-O3 --max-memory`. The most visible symptom is a `KeyError` on staged SRAM value names, but the deeper issue is architectural: scheduled memory semantics and legacy spill semantics are not compatible.

O3 must stop mixing these models. The compiler should have exactly one O3 meaning:

- schedule-aware lowering
- schedule-aware memory planning
- schedule-native spill/reload
- true parallel C code generation

## Non-Goals

- Preserve any legacy O3 spill path
- Auto-fallback from scheduled O3 to conservative O3
- Keep exposing legacy `memory_plan` or `spill_plan` as O3 truth
- Simulate DMA with comments or metadata only

## User-Facing Contract

After this change:

- `-O3` always means scheduled O3
- `--max-memory` always uses schedule-native semantics on O3
- spill and reload are emitted as real DMA steps
- generated x86 simulation code continues to run on the real 4-worker parallel runtime
- failures are reported as explicit scheduled-O3 feasibility or implementation errors, never as internal Python lookup failures

## Recommended Architecture

Use a single scheduled O3 pipeline with four distinct responsibilities:

1. `PipelineStepLoweringPass`
2. `ScheduledMemoryExpansionPass`
3. `PipelineSchedulingPass`
4. `ScheduledMemoryPlanningPass`

The design intentionally keeps lowering, memory-action construction, scheduling, and final storage planning separate. This avoids turning `MemoryPlanningPassV4` into an unbounded catch-all and prevents a second source of truth from reappearing later.

## Pass Pipeline

O3 should converge to this pass chain:

- `IdentityEliminationPass`
- `DeadCodeEliminationPass`
- `PatternFusionPass`
- `PrepackLoweringPass`
- `DominatorFusionPass`
- `ScheduleAnalysisPass`
- `LayoutPlanningPass`
- `TiledLoweringPass`
- `PipelineStepLoweringPass`
- `ScheduledMemoryExpansionPass`
- `PipelineSchedulingPass`
- `ScheduledMemoryPlanningPass`

The following O3 behaviors must be removed:

- `SpillAnalysisPass` participation in O3
- legacy spill compatibility logic inside O3 planning
- O3 fallback-to-legacy scheduling or memory semantics

## Core Data Model

### ScheduledValue

Represents the canonical memory object in scheduled O3. This replaces graph-tensor semantics as the unit consumed by scheduling, memory planning, and DMA modeling.

Suggested fields:

- `name`
- `graph_tensor_name`
- `producer_step_id`
- `consumer_step_ids`
- `size_bytes`
- `must_reside_in_sram`
- `can_alias`
- `home_tier` in `{input, const, slow, sram}`

### TransferStep

Represents a real DMA operation and participates in scheduling exactly like compute work.

Suggested kinds:

- `dma_in`
- `dma_out`
- `spill_dma`
- `reload_dma`

Suggested fields:

- `moved_value_name`
- `src_tier`
- `dst_tier`
- `bytes`

### ResidencyWindow

Represents one contiguous period during which a value is resident in SRAM. A value may have multiple residency windows across spill/reload cycles.

This object is necessary because one value can:

- be produced into SRAM
- consumed by several steps
- be spilled out
- later be reloaded
- be consumed again

The compiler must therefore model multiple legal SRAM lifetimes for one logical value.

## Dependency Rules

The dependency graph must enforce the following:

### Production

A value becomes resident in SRAM only after one of:

- its producer compute step completes
- a `reload_dma` for that value completes

### Consumption

Any compute step that reads a value must depend on the most recent event that made that value resident in SRAM.

### Spill

A `spill_dma(value)` must:

- occur after the producer or reload that created the current residency window
- occur after the final SRAM consumer covered by that residency window

This is not equivalent to the legacy rule "spill after last use" because one logical value may have several residency windows.

### Reload

A `reload_dma(value)` must:

- depend on a valid backing source for the value
- complete before the consumer compute step it serves

### DMA Resource Competition

All of the following compete for the same DMA resource pool:

- `dma_in`
- `dma_out`
- `spill_dma`
- `reload_dma`

This is required so that cost modeling, schedule legality, and generated runtime behavior stay aligned.

### SRAM Capacity

At any time point, SRAM usage is:

- sum of all active value residency windows
- plus step-local SRAM temp usage

A schedule is legal only if this total never exceeds `max_memory`.

## Component Responsibilities

### 1. `PipelineStepLoweringPass`

Responsibilities:

- lower node execution plans into base scheduled steps
- emit initial compute and explicit non-spill DMA steps
- emit initial `ScheduledValue` records

Non-responsibilities:

- do not decide spill points
- do not choose final SRAM or slow-memory offsets

### 2. `ScheduledMemoryExpansionPass`

Responsibilities:

- analyze `ScheduledValue` lifetimes against `max_memory`
- decide which values are spill-capable and which are not
- create candidate `spill_dma` and `reload_dma` steps
- create dependency edges required for legal residency transitions

This pass turns memory pressure into explicit schedule entities rather than leaving it as an implicit post-pass side effect.

### 3. `PipelineSchedulingPass`

Responsibilities:

- schedule the complete step graph including spill/reload DMA steps
- enforce dependency legality
- enforce DMA resource competition
- enforce SRAM capacity feasibility

Failure modes should be explicit, such as:

- `sram_capacity_exceeded`
- `reload_without_backing`
- `value_must_reside_conflict`
- `no_feasible_schedule_under_budget`

### 4. `ScheduledMemoryPlanningPass`

Responsibilities:

- assign final SRAM offsets to staged values and residency windows
- assign slow-memory backing offsets for spilled values
- emit one codegen-ready memory/transfer plan for scheduled O3

Non-responsibilities:

- do not perform legacy compatibility export
- do not invoke or depend on legacy `MemoryPlan`
- do not re-run a separate spill decision phase

## Code Generation Requirements

The x86 backend must consume the unified scheduled plan directly.

It must emit worker-visible real steps for:

- `dma_in`
- `spill_dma`
- `reload_dma`
- `dma_out`
- compute

Constraints:

- the existing 4-worker parallel runtime remains the execution target
- DMA workers perform actual copies or moves
- compute workers do not hide spill/reload behavior internally
- schedule comments remain descriptive only; they must not carry semantics that the runtime does not execute

## Error Handling

CLI-visible failures must become explicit scheduled-O3 diagnostics.

Good examples:

- `scheduled O3 failed: no feasible schedule under max-memory`
- `scheduled O3 failed: value <name> must reside in SRAM but exceeds capacity`
- `scheduled O3 failed: spill/reload expansion unsupported for op family <family>`

Bad examples:

- raw `KeyError`
- legacy pass failure names
- hidden fallback to another O3 meaning

## Testing Strategy

### IR and Pass Tests

Add unit tests for:

- `PipelineStepLoweringPass` output shape for scheduled values
- `ScheduledMemoryExpansionPass` creation of `spill_dma` and `reload_dma`
- legality of dependency edges around residency transitions

### Scheduler Tests

Add tests that verify:

- spill and reload DMA steps enter the actual schedule
- SRAM capacity is enforced over time
- DMA resource contention affects feasibility and order
- infeasible budgets return explicit scheduled diagnostics

### Codegen Tests

Add tests that verify:

- generated `model.c` includes real DMA spill/reload step helpers
- generated runtime metadata reflects spill/reload steps
- generated code no longer depends on legacy `memory_plan` or `spill_plan`
- generated output still compiles

### CLI End-to-End Tests

Required coverage:

- `cli -O3 --max-memory` on `models/operator_coverage_large.onnx`
- `cli -O3 --max-memory` on `models/resnet18.onnx`

Assertions:

- successful cases produce buildable output
- failure cases produce explicit scheduled-O3 errors
- no staged SRAM `KeyError` is exposed

## Migration Plan

1. Remove `SpillAnalysisPass` from O3 pass composition.
2. Introduce `ScheduledMemoryExpansionPass` and new scheduled memory IR records.
3. Upgrade `PipelineSchedulingPass` to schedule spill/reload DMA work.
4. Replace `MemoryPlanningPassV4` compatibility semantics with scheduled-native planning only.
5. Update x86 codegen to consume the unified scheduled memory/transfer plan.
6. Delete remaining O3 dependencies on legacy memory-plan compatibility paths.

## Acceptance Criteria

The change is complete only when all of the following are true:

- O3 has exactly one scheduled meaning
- O3 no longer uses legacy spill or fallback semantics
- `--max-memory` on O3 affects actual DMA-aware scheduling
- spill/reload appear as real DMA work in generated C
- runtime behavior matches scheduled semantics
- failures are explicit scheduled-O3 diagnostics, not internal exceptions
