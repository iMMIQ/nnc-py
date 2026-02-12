# Pattern Matching Guide

This guide explains how to use nnc-py's pattern matching system for operator fusion.

## Overview

nnc-py uses a TVM-style Dataflow Pattern Language (DFPL) for declarative operator fusion. This allows you to define fusion patterns declaratively using composition operators, rather than writing imperative pattern matching code.

## Pattern Building

### Basic Patterns

```python
from nnc_py.pattern import OpPattern, WildcardPattern

# Match any node
wildcard = WildcardPattern("any")

# Match specific operator
conv = OpPattern(OpType.CONV2D, "conv")
relu = OpPattern(OpType.RELU, "relu")
```

### Pattern Composition

The DFPL provides several operators for combining patterns:

```python
from nnc_py.pattern import OpPattern

# Or (|): Matches either pattern
activation = OpPattern(OpType.RELU) | OpPattern(OpType.SIGMOID)

# And (&): Matches both patterns
pattern = conv & relu

# Use constraint: Output consumed by another pattern
conv_then_relu = conv.used_by(relu)

# Exclusive use: Output ONLY consumed by one pattern
conv_exclusive_relu = conv.only_used_by(relu)
```

### Attribute Patterns

Match nodes with specific attribute values:

```python
from nnc_py.pattern import OpPattern

# Match conv with specific kernel size
conv_3x3 = OpPattern(OpType.CONV2D).has_attr(kernel_shape=[3, 3])
```

## Registering Fusion Patterns

To create a custom fusion pattern:

```python
from nnc_py.pattern.registry import register_pattern
from nnc_py.pattern import OpPattern
from nnc_py.ir.node import OpType, Node
from nnc_py.ir.graph import Graph
from nnc_py.pattern.base import PatternMatch

def create_my_fused_node(graph: Graph, match: PatternMatch, name: str) -> Node:
    """Custom fusion logic for creating the fused node."""
    # Extract matched nodes from bindings
    conv_node = match.bindings["conv"]
    relu_node = match.bindings["relu"]

    # Create fused node with combined attributes
    return Node(
        op_type=OpType.MY_FUSED_OP,
        name=name,
        inputs=list(conv_node.inputs),
        outputs=list(relu_node.outputs),
        attrs=conv_node.attrs.copy(),
        metadata={"fused_from": [conv_node.name, relu_node.name]}
    )

# Register the pattern
register_pattern(
    name="my_custom_conv_relu",
    pattern=conv().only_used_by(relu()),
    priority=150,  # Lower priority than built-in patterns
    description="My custom Conv+ReLU fusion",
    replace_func=create_my_fused_node,
    fused_op_type=OpType.MY_FUSED_OP,
)
```

## Pattern Priority

Patterns are matched in priority order (higher values first). Built-in patterns use:
- Priority 200: Common patterns (Conv+ReLU, Add+ReLU, etc.)
- Priority 190-100: Less common patterns

Custom patterns should use priority values based on their expected benefit.

## Built-in Patterns

The following patterns are registered by default:

| Pattern Name | Description | Priority |
|--------------|-------------|----------|
| conv_relu | Conv2D + ReLU | 200 |
| conv_sigmoid | Conv2D + Sigmoid | 200 |
| add_relu | Add + ReLU | 200 |
| add_sigmoid | Add + Sigmoid | 200 |
| matmul_relu | MatMul + ReLU | 190 |

## Example: End-to-End Usage

```python
from nnc_py import compile_model
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph

# Compile with pattern-based fusion (opt-level 3+)
ctx = CompileContext(
    graph=graph,
    target="x86",
    optimization_level=3  # Enables pattern fusion
)

result = compile_model(ctx)
# Fused operators will be applied automatically
```
