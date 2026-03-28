# X86 Codegen IR Refactor Design

**Date:** 2026-03-27

**Status:** Approved in chat, written for implementation planning

## Goal

Restructure x86 code generation so that `X86Backend` no longer mixes orchestration, scheduled-O3 lowering, serial lowering, and C text emission in one module. Introduce a codegen-focused lowered IR and move emitters to narrower modules while preserving current x86 behavior.

## Context

The current x86 backend has grown into the dominant complexity center of the repository. It handles:

- backend orchestration
- compile-path branching for serial vs scheduled O3
- schedule-aware metadata shaping
- runtime helper generation
- per-file C emission
- output artifact assembly

This creates three problems:

1. scheduled O3 concerns leak across the backend instead of staying localized
2. ordinary x86 emission is harder to reason about because it shares helpers with schedule-specific behavior
3. future work on performance, diagnostics, and backend abstractions is blocked by file size and responsibility overlap

## Non-Goals

This refactor will not:

- implement or redesign `npu`
- add new optimization passes
- redesign the benchmark system
- introduce new compiler features unrelated to x86 codegen boundaries
- attempt a perfect long-term backend abstraction for all targets

## Constraints

- Preserve current x86 behavior as the main success criterion.
- Opportunistic cleanup is allowed where it simplifies interfaces, metadata flow, or error presentation.
- Non-semantic generated-code layout changes are acceptable.
- NPU remains a placeholder and is excluded from the new abstraction work.

## Recommended Approach

Use a codegen-focused lowered IR as the seam between compile-path decisions and C emission.

This is the chosen approach because it removes the largest current source of complexity without requiring a full compiler redesign. The backend should decide what needs to be generated, produce a structured representation of that output, and hand it to narrow emitters that only format artifacts.

## Architecture

### 1. Backend Orchestration Layer

`X86Backend.generate()` becomes a thin coordinator that:

- validates the context it needs
- assigns symbols and prepares shared state
- selects the serial or scheduled lowering path
- builds one lowered codegen package
- invokes emitters for output files
- returns the generated artifact bundle

It should not directly contain large C string construction logic.

### 2. Lowered Codegen IR Layer

Introduce a new internal representation for x86 code generation. This is not a general compiler IR. It is a codegen contract that describes what the emitters need to output.

Expected categories in this IR:

- public entry point and generated helper functions
- tensor and constant declarations
- memory-pool definitions and allocation metadata
- serial execution steps
- scheduled execution steps
- schedule summary and per-node annotations
- parallel runtime metadata for scheduled execution
- debug/runtime support requirements
- artifact-level metadata needed by emitters

This layer absorbs scheduled-O3 special cases so they stop leaking into generic emission helpers.

### 3. Emitter Layer

Split file emission into focused modules, for example:

- header emitter
- model source emitter
- tensor emitter
- constants loader emitter
- build artifact emitter
- test runner emitter

Emitters accept lowered codegen IR and produce text or binary outputs. They should not re-derive schedule decisions from `CompileContext`.

## Compile-Path Boundaries

Two compile paths remain for x86:

- serial x86 codegen
- scheduled O3 x86 codegen

The important change is where they diverge and converge:

- they diverge in lowering
- they converge at the lowered codegen IR boundary
- they share emitters after lowering

This keeps schedule-specific behavior explicit without duplicating artifact writers.

## Scope of This Refactor

### Included

- create the lowered x86 codegen IR
- move scheduled O3 shaping logic into lowering
- move serial shaping logic into lowering
- extract emitters/helpers out of `x86_backend.py`
- simplify backend orchestration responsibilities
- clean up metadata and error-flow edges where they become clearer during the split

### Excluded

- NPU backend work
- benchmark redesign
- pass-pipeline redesign beyond what is needed to support the new backend seam
- broad IR rewrites outside x86 code generation

## Testing Strategy

Follow characterization-first refactoring.

### Before structural changes

Add or strengthen tests that pin:

- basic x86 compile artifact generation
- scheduled O3 artifact features
- schedule-summary and parallel-runtime related output markers
- sanitized compile-error behavior that must remain user-facing

These tests should focus on stable behavioral signals, not brittle full-file snapshots unless a snapshot already provides useful coverage.

### During refactor

Run narrow test slices after each boundary move, then full test suite at the end.

Priority test areas:

- CLI compile behavior
- backend/codegen tests
- scheduled pipeline tests
- snapshot tests touching generated artifacts

## Implementation Order

1. Add characterization coverage for current x86 outputs and scheduled-O3 signals.
2. Define minimal lowered x86 codegen IR types.
3. Move scheduled O3 lowering into dedicated lowering code.
4. Move serial lowering into dedicated lowering code.
5. Extract emitters for generated artifacts.
6. Reduce `X86Backend.generate()` to orchestration over lowering + emitters.
7. Run full verification and clean up temporary compatibility shims.

## Risks

### Hidden coupling in scheduled codegen

Scheduled O3 behavior may depend on metadata and helper interactions that are not obvious until moved.

Mitigation:

- migrate scheduled lowering first
- keep intermediate adapters temporarily
- verify with focused scheduled tests after each extraction

### Overly brittle output tests

Exact generated-text assertions can make a behavior-preserving refactor unnecessarily expensive.

Mitigation:

- prefer key-fragment assertions for new characterization tests
- keep existing snapshots where they already provide value

### Mixed decision and formatting logic

Some helpers likely combine semantic decisions with final text formatting.

Mitigation:

- split in two phases when needed: first isolate decision data, then move formatting

## Success Criteria

- `x86_backend.py` is substantially smaller and no longer owns orchestration, lowering, and emission at once
- serial and scheduled x86 paths have explicit lowering entry points
- emitters consume lowered codegen IR rather than raw `CompileContext`
- existing x86 behavior remains compatible
- full test suite passes after refactor

## Follow-On Work

Once this seam exists, the next engineering gains become cheaper:

- benchmark expansion across more models and memory modes
- cleaner schedule diagnostics
- backend capability boundaries that do not depend on one giant x86 file
- future target abstraction work if `npu` becomes real later
