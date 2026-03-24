# Tiled Lowering And Blocked Layout Design

## Status

Proposed and user-approved at the design level on 2026-03-23.

Partially implemented and revalidated during execution on 2026-03-24.

Current implementation status in `nnc-py`:

- structured execution-plan IR is in place
- `ScheduleAnalysisPass`, `LayoutPlanningPass`, `TiledLoweringPass`, and `MemoryPlanningPassV3` are implemented
- phase-1 tile-aware execution/storage support exists for a conservative supported subset, including:
  - `Conv`
  - `MaxPool`
  - tile-compatible `Add` / `Relu` group handoff
  - a minimal tail `Gemm` coverage path for `resnet18`
- logical tile/scratch region metadata is emitted into code generation
- late target-physical layout intent is recorded as deferred metadata/comments rather than materialized
- `resnet18` now compiles successfully at `O3` with `max_memory='1M'`
  - current emitted fast-memory size: `1,039,616` bytes

Still intentionally deferred:

- full device-specific physical layout materialization
- broad graph-wide tile executor beyond the currently supported safe subsets
- full operator coverage outside the documented phase-1 path

## Context

`nnc-py` currently has the right high-level compiler shape for deployment on a heterogeneous embedded device:

- semantic graph IR
- optimization passes
- liveness analysis
- memory planning
- backend code generation

But its execution model is still mostly whole-tensor oriented. The current `cost_aware` allocator materially reduces fast-memory usage by reusing buffers across nodes, but it does not reduce the per-node working-set floor for large operators such as `Conv` and `MaxPool`.

Measured on `resnet18` in the current tree:

- `O0` fast memory: about `70.3 MB`
- `O3 + basic allocator`: about `61.1 MB`
- `O3 + cost_aware allocator`: about `4.6 MB`
- with a tighter budget, the graph can compact to about `4.0 MB`
- below that, compilation fails because the peak single-node demand is still about `4.0 MB`

For the target device, fast memory is about `1 MB`. This makes whole-tensor execution fundamentally insufficient. The compiler must lower large operators into tiled execution plans and must model blocked layouts before final target-specific layout mapping.

The target physical memory layout is similar to Ascend-style `NZ/ZZ`, not NVIDIA-style layouts. However, the compiler should not expose `NZ/ZZ` as an early IR concept. The design should use a generic blocked/tiled layout abstraction and map to the final physical layout at the backend boundary.

## Goals

- Make `1 MB` fast-memory compilation a primary compiler objective.
- Introduce a general tiled execution framework rather than model-specific hacks.
- Keep semantic graph IR independent from target physical layout.
- Use generic blocked layout internally and map to `NZ/ZZ`-like physical layout late.
- Let memory planning reason about tile buffers, scratch buffers, pack buffers, and staged movement.
- Preserve correctness through ONNX Runtime equivalence checks for compiler-driven transformations.

## Non-Goals

- Do not optimize around x86 runtime kernels as the main strategy.
- Do not encode target physical layout directly into frontend IR.
- Do not use graph-level `Split`/`Concat` cloning as the primary execution strategy.
- Do not attempt full coverage for every ONNX operator in the first phase.
- Do not require the whole graph to be tiled in phase 1.

## Design Decision

The main design direction remains inside `nnc-py`, not `onnxsplit`.

`onnxsplit` is useful as a reference for:

- plan-first transformation workflow
- split legality analysis
- memory-constrained adjustment flow
- ONNX Runtime equivalence validation

`onnxsplit` is not the right primary architecture here because it is a graph-splitting tool, not a tiled compiler for a small SRAM target. Its core assumptions are too weak for this project:

- it models memory as roughly `total_memory / parts`
- it primarily reasons at graph level rather than schedule level
- it uses `Split`/`Slice`/`Concat` graph rewriting as the execution mechanism
- its axis rules are oriented toward generic ONNX execution, especially batch splitting
- it does not model halo, scratch, blocked layout, or target physical layout mapping

The chosen design is a schedule-aware tiled lowering architecture inside `nnc-py`.

## Architecture Overview

The compiler is split into four conceptual layers.

### 1. Semantic IR Layer

This remains the current graph-level operator view:

- `Conv`
- `Add`
- `Pool`
- `Relu`
- `Gemm`
- other semantic operators

This layer is used by frontend loading, graph cleanup, fusion, legality checks, and most high-level reasoning.

### 2. Schedule / Lowering Layer

This layer introduces structured execution plans for operators without replacing the semantic nodes themselves.

Each tiled-capable node gets a lowering plan that describes:

- tile axes and tile sizes
- block shapes
- halo requirements
- scratch requirements
- pack / unpack requirements
- stage buffer requirements
- whether the node can participate in a fused tile group
- whether double-buffering is supported

This is the core new capability.

### 3. Memory Planning Layer

Memory planning no longer reasons only in terms of full tensor lifetimes. It must reason about:

- persistent tensors
- input/output tile buffers
- scratch buffers
- layout pack buffers
- stage buffers between memory levels

Peak fast-memory demand becomes a function of the execution plan, not just the tensor graph.

### 4. Backend Mapping Layer

The backend maps generic blocked layouts into target physical layouts similar to `NZ/ZZ`.

This mapping happens late so that earlier passes remain target-agnostic enough to be reusable.

## Alternative Approaches Considered

### Approach A: Keep Extending `node.metadata["lowering"]`

Pros:

- minimal short-term change
- easy to continue from current prepack-lowering work

Cons:

- tile loops, halo, pack buffers, and layout mapping rapidly become unstructured metadata
- pass boundaries become unclear
- memory planning remains fragile

Rejected as the long-term architecture.

### Approach B: Use Graph-Level Split/Concat Transformation Like `onnxsplit`

Pros:

- conceptually simple
- leverages ONNX graph manipulation patterns
- easy to verify with existing runtimes

Cons:

- wrong abstraction level for a compiler targeting explicit memory hierarchies
- poor fit for halo, scratch, and blocked-layout execution
- leads to graph bloat and backend complexity

Rejected as the primary strategy. It may remain a fallback mechanism for some unsupported operators in the future.

### Approach C: Add A Dedicated Schedule / Layout Lowering Layer

Pros:

- aligns with the real bottleneck: per-node working-set size
- allows explicit tile, scratch, and layout reasoning
- keeps semantic IR and target physical layout decoupled

Cons:

- requires new plan structures and a new memory-planning model
- larger upfront compiler work

Chosen.

## New Core Concepts

### Layout Layers

The design uses three layout levels.

#### Semantic Layout

Examples:

- `NCHW`
- `OIHW`

Used for operator meaning and frontend compatibility.

#### Generic Blocked Layout

Examples:

- activation blocked on `C`
- weights blocked on `K` and `C`

This is the main layout representation used by lowering and memory planning.

#### Target Physical Layout

Examples:

- `NZ`-like
- `ZZ`-like

Used only at backend mapping and code generation time.

### Execution Planning Objects

The implementation should introduce structured objects equivalent to the following concepts:

- `LayoutClass`
  - `plain`
  - `blocked_activation`
  - `blocked_weight`
  - `target_physical`
- `TileRegion`
  - logical extents
  - halo extents
  - block alignment requirements
- `TensorAccessPlan`
  - how each input or output is materialized for one execution step
- `NodeExecutionPlan`
  - the full tiled execution contract for one node
- `MemoryRegionKind`
  - `persistent`
  - `tile`
  - `scratch`
  - `pack`
  - `stage`

The exact Python types can evolve, but the separation of concerns should stay intact.

## Pass Pipeline

The advanced pipeline should evolve toward:

1. `IdentityEliminationPass`
2. `DeadCodeEliminationPass`
3. basic pattern fusion
4. `PrepackLoweringPass`
5. `ScheduleAnalysisPass`
6. `TiledLoweringPass`
7. `LayoutPlanningPass`
8. `LivenessAnalysisPass`
9. `MemoryPlanningPassV3`
10. `SpillOrStagingPass`
11. backend physical-layout mapping
12. code generation

### Pass Responsibilities

#### `PrepackLoweringPass`

Keep it, but narrow its role:

- weight prepacking
- kernel family hints
- static attributes useful to later tiled lowering

It should not become the place where the full execution strategy is decided.

#### `ScheduleAnalysisPass`

Analyzes which nodes:

- must be tiled
- may remain whole-tensor
- should share tile boundaries with neighbors
- can participate in a small fused tile group

It emits candidate scheduling information only.

#### `TiledLoweringPass`

Builds concrete execution plans for selected operators and schedule groups.

This is where the compiler decides:

- tile shapes
- halo
- scratch
- reusable tile boundaries
- tile-friendly fused groups

#### `LayoutPlanningPass`

Chooses generic blocked layouts and block factors. It does not emit final target physical layout yet.

#### `MemoryPlanningPassV3`

Plans fast memory from execution plans, not from whole-tensor operator boundaries.

#### `SpillOrStagingPass`

Handles staged movement between memory levels in a tile-aware way. This should replace the current whole-tensor-overflow mindset as the primary advanced path.

## Operator Scope

### Phase 1 Required Coverage

- `Conv`
- `MaxPool`
- `AveragePool`
- `GlobalAveragePool`
- `Gemm`
- `MatMul`
- `Add`
- `Relu`

### Phase 1 Restricted Coverage

- `Concat`
  - only tile-compatible cases
- `Flatten`
- `Reshape`
- `Transpose`
  - only when they do not destroy tile flow or can be represented as cheap metadata reshaping

### Later Phases

- `BatchNormalization`
- `LayerNormalization`
- `ReduceMean`
- `ReduceSum`
- `Softmax`
- `Mul`
- `Sub`
- `Div`
- `Sigmoid`
- `Tanh`

### Explicitly Deferred

- `LSTM`
- `Gather`
- heavily dynamic-shape operators
- fully general transpose/layout transforms

## Memory Model

`MemoryPlanningPassV3` must model five memory object classes:

- `persistent`
- `tile_io`
- `scratch`
- `pack`
- `stage`

The per-plan fast-memory demand is:

`peak_fast(plan) = live_persistent + input_tiles + output_tiles + scratch + pack + stage`

This replaces whole-tensor node demand as the primary optimization quantity.

### Region Strategy

Memory planning should proceed in two stages.

#### Stage 1: Region Sizing

Estimate upper bounds for:

- persistent pool
- tile pool
- scratch pool
- pack pool
- stage pool

#### Stage 2: Region Scheduling

Exploit reuse across execution order:

- reuse tile buffers inside a schedule group
- allow direct handoff from producer output tile to consumer input tile when legal
- aggressively reuse scratch
- isolate pack and stage buffers to avoid polluting persistent allocation

The final generated code may still place these logical regions inside one physical fast pool, but the planner must reason about them separately.

## Why This Reaches 1 MB

The current compiler hits a hard floor because large operators still require whole input and whole output to coexist.

For example, a large `MaxPool` currently behaves like:

- full input activation resident
- full output activation resident

Under tiled lowering it instead behaves like:

- one input tile with halo
- one output tile
- minimal scratch

This is the difference between a multi-megabyte per-node floor and a design that can plausibly fit within `1 MB`.

## Backend Layout Mapping

The backend should map generic blocked layout into target physical layout late.

Phase 1 should support only a limited set of generic blocked layouts:

- activation blocked on `C`
- weight blocked on `K` and `C`

This keeps the first implementation tractable while leaving room for richer fractal-style layouts later.

The backend mapping stage is responsible for:

- final index order
- inner/outer dimension folding
- physical alignment constraints
- final pack strategy
- any target-specific layout materialization

## Correctness And Validation

Validation must happen at four levels.

### 1. Lowering Validation

Check that every generated execution plan is internally valid:

- tile shapes legal
- halo legal
- block alignment legal
- scratch sizes deterministic and non-negative

### 2. Memory-Model Validation

Check that the planned fast-memory usage stays within the requested budget, especially around `1 MB`.

This is more important than microbenchmarks for the first phase.

### 3. Numerical Validation

Use the current ONNX Runtime comparison flow to verify that compiler-lowered tiled execution matches the reference implementation.

Tests should cover:

- tiled `Conv`
- tiled `MaxPool`
- `Add + Relu`
- residual paths
- pack / unpack correctness around blocked layout

### 4. Whole-Model Viability

Use real models such as `resnet18` to validate:

- the compiler can produce code under a `1 MB` fast-memory budget
- generated fast / tile / scratch pools match expectations
- the solution is not dependent on runtime kernel specialization

## Error Handling

When the compiler cannot produce a legal tiled plan, it must fail for the right reason.

Expected error classes include:

- no legal tile shape under current memory budget
- unsupported operator for tiled lowering
- incompatible layout handoff between adjacent plans
- impossible halo or pack requirements
- physical-layout mapping not implemented for a chosen generic blocked layout

These failures should be explicit and actionable. Silent fallback to whole-tensor execution under a `1 MB` budget should be avoided for operators that would obviously exceed the budget.

## Implementation Strategy

Implementation should proceed incrementally.

### Milestone 1

- add plan data structures
- add `ScheduleAnalysisPass`
- add `TiledLoweringPass` for `Conv` and `MaxPool`
- add generic blocked activation / weight layouts
- add `MemoryPlanningPassV3` with logical region support

### Milestone 2

- extend tiled lowering to `Add`, `Relu`, `AveragePool`, `GlobalAveragePool`, `Gemm`, `MatMul`
- support small tile-friendly schedule groups
- add pack / stage planning

### Milestone 3

- backend mapping from generic blocked layout to target physical layout
- phase-1 whole-model validation under `1 MB`

## Open Questions

- exact default block sizes for activations and weights
- whether phase 1 should expose double-buffering structure immediately or reserve the API only
- which restricted `Transpose` cases should be treated as metadata-only layout reinterpretation

These questions affect implementation details, not the main architecture.

## Final Decision

Build the next stage of `nnc-py` around schedule-aware tiled lowering, generic blocked layouts, and tile-aware memory planning. Use `onnxsplit` only as a reference for planning discipline and verification style, not as the main structural model.
