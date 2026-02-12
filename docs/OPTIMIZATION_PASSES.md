# Optimization Passes

This document describes the optimization passes available in nnc-py.

## Overview

The compiler applies optimization passes based on the optimization level (`opt_level`).
Each pass transforms the computation graph to improve efficiency or reduce code size.

## Pass Categories

### Essential Passes (O0)

These passes are required for correct code generation:

- **LivenessAnalysisPass**: Analyzes tensor lifetimes for memory planning
- **MemoryPlanningPassV2**: Plans memory allocation for tensors

### Basic Optimizations (O1)

Applied at `opt_level=1` and above:

- **IdentityEliminationPass**: Removes Identity operations from the graph
  - Replaces all references to an Identity node's output with its input
  - Handles chains of Identity nodes correctly
  - Reduces unnecessary operations

- **DeadCodeEliminationPass**: Removes unused nodes from the graph
  - Identifies nodes whose outputs are not used
  - Preserves nodes that produce graph outputs
  - Preserves nodes that produce inputs to live nodes

### Intermediate Optimizations (O2)

Applied at `opt_level=2` and above:

All O1 passes plus:

- **SpillAnalysisPass**: Handles memory overflow when `max_memory` is set

### Advanced Optimizations (O3)

Applied at `opt_level=3` and above:

All O2 passes plus:

- **PatternFusionPass**: Declarative pattern-based operator fusion
  - Uses TVM-style Dataflow Pattern Language (DFPL)
  - Built-in fusion patterns:
    - Conv + ReLU → FusedConvRelu (reduces memory traffic)
    - Add + ReLU → FusedAddRelu (faster activation)
    - Conv + Sigmoid → FusedConvSigmoid
    - Add + Sigmoid → FusedAddSigmoid
    - MatMul + ReLU → FusedMatMulRelu
  - Only fuses when producer output has single consumer
  - Preserves graph semantics exactly
  - Supports custom pattern registration via `register_pattern()`

See `docs/pattern_matching_guide.md` for details on defining custom fusion patterns.

## Implementation Details

### Adding a New Pass

1. Create a new file in `src/nnc_py/passes/`
2. Inherit from `PassBase`
3. Implement the `_execute` method
4. Import and register in `PassManager.get_default_passes()`

Example:

```python
from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassBase

class MyOptimizationPass(PassBase):
    @property
    def name(self) -> str:
        return "MyOptimization"

    def _execute(self, ctx: CompileContext) -> None:
        # Transform ctx.graph here
        pass
```

### Pass Execution Order

Passes are executed in the order they are registered. For O3:

1. IdentityEliminationPass - removes no-op operations
2. DeadCodeEliminationPass - removes now-unused nodes
3. OperatorFusionPass - fuses compatible operator patterns
4. LivenessAnalysisPass - analyzes lifetimes on optimized graph
5. MemoryPlanningPassV2 - plans memory based on lifetimes
6. SpillAnalysisPass - handles overflow if needed
