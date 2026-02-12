# Pattern-Based Fusion System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将TVM的Dataflow Pattern Language (DFPL)完整移植到nnc-py，替代现有的硬编码算子融合策略

**Architecture:** 基于DFPattern的声明式模式匹配系统，通过组合运算符(|, &, >>, ^)构建复杂模式，注册表管理融合模式，支持优先级和自定义约束

**Tech Stack:** Python 3.10+, dataclasses, abc, typing, NetworkX (已有), pytest

---

## Task 1: 创建pattern模块目录结构

**Files:**
- Create: `src/nnc_py/pattern/__init__.py`
- Create: `src/nnc_py/pattern/base.py`
- Create: `src/nnc_py/pattern/patterns.py`
- Create: `src/nnc_py/pattern/matcher.py`
- Create: `src/nnc_py/pattern/registry.py`
- Create: `src/nnc_py/pattern/fusion_patterns.py`

**Step 1: 创建__init__.py**

```python
"""Dataflow Pattern Language for nnc-py.

This module provides TVM-style declarative pattern matching for operator fusion.
"""

from nnc_py.pattern.base import (
    DFPattern,
    PatternMatch,
    MatchContext,
)

from nnc_py.pattern.patterns import (
    WildcardPattern,
    OpPattern,
    OrPattern,
    AndPattern,
    AttrPattern,
    UsePattern,
    ExclusiveUsePattern,
)

from nnc_py.pattern.matcher import PatternMatcher
from nnc_py.pattern.registry import PatternRegistry, register_pattern, FusionPattern

__all__ = [
    "DFPattern",
    "PatternMatch",
    "MatchContext",
    "WildcardPattern",
    "OpPattern",
    "OrPattern",
    "AndPattern",
    "AttrPattern",
    "UsePattern",
    "ExclusiveUsePattern",
    "PatternMatcher",
    "PatternRegistry",
    "register_pattern",
    "FusionPattern",
]
```

**Step 2: 运行验证模块导入**

Run: `python -c "from nnc_py.pattern import DFPattern; print('Import OK')"`
Expected: "Import OK"

**Step 3: Commit**

```bash
git add src/nnc_py/pattern/__init__.py
git commit -m "feat(pattern): add pattern module structure"
```

---

## Task 2: 实现base.py核心类

**Files:**
- Create: `src/nnc_py/pattern/base.py`
- Test: `tests/test_pattern_base.py`

**Step 1: 编写基础类**

```python
"""Core pattern classes for dataflow pattern matching."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set


@dataclass
class PatternMatch:
    """Represents a successful pattern match with bindings.

    Attributes:
        bindings: Maps pattern variable names to matched nodes/tensors
        anchor: The root node of the matched pattern
        nodes: All nodes in the matched subgraph
    """
    bindings: Dict[str, Any]
    anchor: 'Node'
    nodes: Set['Node']


@dataclass
class MatchContext:
    """Context for pattern matching with memoization.

    Attributes:
        bindings: Current variable bindings
        memo: Cache for (node_id, pattern_id) -> match result
    """
    bindings: Dict[str, Any] = field(default_factory=dict)
    memo: Dict[tuple, Optional['PatternMatch']] = field(default_factory=dict)

    def with_bindings(self, new_bindings: Dict[str, Any]) -> 'MatchContext':
        """Create a new context with additional bindings."""
        merged = self.bindings.copy()
        merged.update(new_bindings)
        return MatchContext(bindings=merged, memo=self.memo)

    def get_cached(self, node: 'Node', pattern: 'DFPattern') -> Optional['PatternMatch']:
        """Get cached match result."""
        return self.memo.get((id(node), id(pattern)))

    def set_cached(self, node: 'Node', pattern: 'DFPattern', result: Optional['PatternMatch']) -> None:
        """Cache match result."""
        self.memo[(id(node), id(pattern))] = result


class DFPattern(ABC):
    """Base class for all dataflow patterns.

    Provides composition operators similar to TVM's DFPL:
    - | (or): Matches either pattern
    - & (and): Matches both patterns
    - used_by: Output is consumed by another pattern
    - only_used_by: Output is ONLY consumed by another pattern
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or f"pat_{id(self)}"

    def __or__(self, other: 'DFPattern') -> 'OrPattern':
        """Create an OrPattern - matches either pattern."""
        # Import here to avoid circular dependency
        from nnc_py.pattern.patterns import OrPattern
        return OrPattern(self, other)

    def __and__(self, other: 'DFPattern') -> 'AndPattern':
        """Create an AndPattern - matches both patterns."""
        from nnc_py.pattern.patterns import AndPattern
        return AndPattern(self, other)

    def has_attr(self, **kwargs) -> 'AttrPattern':
        """Match nodes with specific attribute values."""
        from nnc_py.pattern.patterns import AttrPattern
        return AttrPattern(self, **kwargs)

    def used_by(self, pattern: 'DFPattern') -> 'UsePattern':
        """Match when this pattern's output is used by another pattern."""
        from nnc_py.pattern.patterns import UsePattern
        return UsePattern(self, pattern)

    def only_used_by(self, pattern: 'DFPattern') -> 'ExclusiveUsePattern':
        """Match when this pattern's output is ONLY used by another pattern."""
        from nnc_py.pattern.patterns import ExclusiveUsePattern
        return ExclusiveUsePattern(self, pattern)

    @abstractmethod
    def match(self, node: 'Node', graph: 'Graph', context: MatchContext) -> Optional[PatternMatch]:
        """Try to match this pattern against a node.

        Args:
            node: The node to match against
            graph: The computation graph
            context: Current matching context (bindings, memoization)

        Returns:
            PatternMatch if successful, None otherwise
        """
        pass

    def extract(self, match: PatternMatch) -> Any:
        """Extract captured values from a match."""
        return match.bindings.get(self.name)
```

**Step 2: 编写测试**

```python
# tests/test_pattern_base.py
import pytest
from nnc_py.pattern.base import DFPattern, PatternMatch, MatchContext

def test_pattern_match_dataclass():
    """Test PatternMatch dataclass structure."""
    match = PatternMatch(
        bindings={"x": "node1"},
        anchor="node1",
        nodes={"node1"}
    )
    assert match.bindings["x"] == "node1"
    assert match.anchor == "node1"
    assert "node1" in match.nodes

def test_match_context_with_bindings():
    """Test MatchContext binding extension."""
    ctx = MatchContext(bindings={"a": 1})
    new_ctx = ctx.with_bindings({"b": 2})
    assert new_ctx.bindings == {"a": 1, "b": 2}
    assert ctx.bindings == {"a": 1}  # Original unchanged

def test_match_context_memoization():
    """Test MatchContext memo cache."""
    from unittest.mock import Mock
    ctx = MatchContext()

    pattern = Mock()
    node = Mock()
    match = Mock()

    ctx.set_cached(node, pattern, match)
    assert ctx.get_cached(node, pattern) == match
```

**Step 3: 运行测试**

Run: `pytest tests/test_pattern_base.py -v`
Expected: 全部PASS

**Step 4: Commit**

```bash
git add src/nnc_py/pattern/base.py tests/test_pattern_base.py
git commit -m "feat(pattern): implement DFPattern base class and MatchContext"
```

---

## Task 3: 实现具体模式类

**Files:**
- Create: `src/nnc_py/pattern/patterns.py`
- Test: `tests/test_pattern_concrete.py`

**Step 1: 实现WildcardPattern和OpPattern**

```python
"""Concrete pattern implementations."""

from typing import Optional
from nnc_py.pattern.base import DFPattern, PatternMatch, MatchContext
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.graph import Graph


class WildcardPattern(DFPattern):
    """Matches any node (like . in regex)."""

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # Wildcard matches anything
        return PatternMatch(
            bindings={self.name: node},
            anchor=node,
            nodes={node}
        )


class OpPattern(DFPattern):
    """Matches a specific operator type."""

    def __init__(self, op_type: OpType, name: Optional[str] = None):
        super().__init__(name)
        self.op_type = op_type

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # Check memoization
        cached = context.get_cached(node, self)
        if cached is not None:
            return cached

        # Match if operator types match
        if node.op_type != self.op_type:
            result = None
        else:
            result = PatternMatch(
                bindings={self.name: node},
                anchor=node,
                nodes={node}
            )

        context.set_cached(node, self, result)
        return result
```

**Step 2: 实现OrPattern和AndPattern**

```python
# 添加到patterns.py

class OrPattern(DFPattern):
    """Matches if either left OR right pattern matches."""

    def __init__(self, left: DFPattern, right: DFPattern):
        super().__init__()
        self.left = left
        self.right = right

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # Try left first, then right
        result = self.left.match(node, graph, context)
        if result is None:
            result = self.right.match(node, graph, context)
        return result


class AndPattern(DFPattern):
    """Matches if both patterns match (structural composition).

    Used for combining patterns that match different parts of the graph.
    """

    def __init__(self, left: DFPattern, right: DFPattern):
        super().__init__()
        self.left = left
        self.right = right

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # Try to match left pattern
        left_match = self.left.match(node, graph, context)
        if left_match is None:
            return None

        # Try to match right pattern with updated bindings
        right_match = self.right.match(node, graph, context.with_bindings(left_match.bindings))
        if right_match is None:
            return None

        # Merge bindings and nodes
        merged = left_match.bindings.copy()
        merged.update(right_match.bindings)
        merged_nodes = left_match.nodes | right_match.nodes

        # Anchor is the last matched node
        return PatternMatch(
            bindings=merged,
            anchor=right_match.anchor or left_match.anchor,
            nodes=merged_nodes
        )
```

**Step 3: 实现UsePattern和ExclusiveUsePattern**

```python
# 添加到patterns.py

class UsePattern(DFPattern):
    """Matches when producer's output is used by consumer pattern."""

    def __init__(self, producer: DFPattern, consumer: DFPattern):
        super().__init__()
        self.producer = producer
        self.consumer = consumer

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # Try to match producer at this node
        prod_match = self.producer.match(node, graph, context)
        if prod_match is None:
            return None

        # Check if any consumer matches
        anchor = prod_match.anchor
        if not anchor.outputs:
            return None

        for out_tensor in anchor.outputs:
            for consumer_node in graph.get_consumers(out_tensor):
                cons_match = self.consumer.match(
                    consumer_node,
                    graph,
                    context.with_bindings(prod_match.bindings)
                )
                if cons_match:
                    # Merge bindings
                    merged = prod_match.bindings.copy()
                    merged.update(cons_match.bindings)
                    merged_nodes = prod_match.nodes | cons_match.nodes
                    return PatternMatch(
                        bindings=merged,
                        anchor=cons_match.anchor,
                        nodes=merged_nodes
                    )
        return None


class ExclusiveUsePattern(DFPattern):
    """Matches when producer's output is ONLY used by consumer pattern.

    This is the key pattern for safe fusion - ensures the producer's
    output has exactly one consumer.
    """

    def __init__(self, producer: DFPattern, consumer: DFPattern):
        super().__init__()
        self.producer = producer
        self.consumer = consumer

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # Try to match producer at this node
        prod_match = self.producer.match(node, graph, context)
        if prod_match is None:
            return None

        # Check exclusive use condition
        anchor = prod_match.anchor
        if not anchor.outputs:
            return None

        for out_tensor in anchor.outputs:
            consumers = graph.get_consumers(out_tensor)
            if len(consumers) != 1:
                return None  # Not exclusively used

            cons_match = self.consumer.match(
                consumers[0],
                graph,
                context.with_bindings(prod_match.bindings)
            )
            if cons_match:
                merged = prod_match.bindings.copy()
                merged.update(cons_match.bindings)
                merged_nodes = prod_match.nodes | cons_match.nodes
                return PatternMatch(
                    bindings=merged,
                    anchor=cons_match.anchor,
                    nodes=merged_nodes
                )
        return None


class AttrPattern(DFPattern):
    """Matches nodes with specific attribute values."""

    def __init__(self, pattern: DFPattern, **attrs):
        super().__init__()
        self.pattern = pattern
        self.attrs = attrs

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # First match inner pattern
        inner_match = self.pattern.match(node, graph, context)
        if inner_match is None:
            return None

        # Check attributes
        anchor = inner_match.anchor
        for key, expected_value in self.attrs.items():
            actual_value = anchor.attrs.get(key)
            if actual_value != expected_value:
                return None

        return inner_match
```

**Step 4: 编写测试**

```python
# tests/test_pattern_concrete.py
import pytest
from nnc_py.pattern.patterns import WildcardPattern, OpPattern, OrPattern, AndPattern
from nnc_py.pattern.patterns import UsePattern, ExclusiveUsePattern
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.graph import Graph
from nnc_py.ir.tensor import TensorType


def test_wildcard_matches_anything():
    """Test that WildcardPattern matches any node."""
    pattern = WildcardPattern("wc")

    conv_node = Node(op_type=OpType.CONV2D, name="conv1", inputs=[], outputs=["out1"])
    relu_node = Node(op_type=OpType.RELU, name="relu1", inputs=["out1"], outputs=[])

    graph = Graph()
    graph.add_node(conv_node)
    graph.add_node(relu_node)

    from nnc_py.pattern.base import MatchContext
    ctx = MatchContext()

    result1 = pattern.match(conv_node, graph, ctx)
    result2 = pattern.match(relu_node, graph, ctx)

    assert result1 is not None
    assert result2 is not None
    assert result1.bindings["wc"] == conv_node
    assert result2.bindings["wc"] == relu_node


def test_op_pattern_matches_specific_type():
    """Test OpPattern matches only its operator type."""
    conv_pattern = OpPattern(OpType.CONV2D, "conv")
    relu_pattern = OpPattern(OpType.RELU, "relu")

    conv_node = Node(op_type=OpType.CONV2D, name="conv1", inputs=[], outputs=["out1"])

    graph = Graph()
    graph.add_node(conv_node)

    from nnc_py.pattern.base import MatchContext
    ctx = MatchContext()

    assert conv_pattern.match(conv_node, graph, ctx) is not None
    assert relu_pattern.match(conv_node, graph, ctx) is None


def test_or_pattern():
    """Test OrPattern matches either pattern."""
    pattern = OpPattern(OpType.RELU, "act") | OpPattern(OpType.SIGMOID, "act")

    relu_node = Node(op_type=OpType.RELU, name="relu1", inputs=[], outputs=[])
    sigmoid_node = Node(op_type=OpType.SIGMOID, name="sig1", inputs=[], outputs=[])
    tanh_node = Node(op_type=OpType.TANH, name="tanh1", inputs=[], outputs=[])

    graph = Graph()
    for n in [relu_node, sigmoid_node, tanh_node]:
        graph.add_node(n)

    from nnc_py.pattern.base import MatchContext
    ctx = MatchContext()

    assert pattern.match(relu_node, graph, ctx) is not None
    assert pattern.match(sigmoid_node, graph, ctx) is not None
    assert pattern.match(tanh_node, graph, ctx) is None


def test_exclusive_use_pattern():
    """Test that only_used_by requires single consumer."""
    from nnc_py.pattern.base import MatchContext

    conv = OpPattern(OpType.CONV2D, "conv")
    relu = OpPattern(OpType.RELU, "relu")
    pattern = conv.only_used_by(relu)

    # Case 1: Single consumer - should match
    conv_node = Node(op_type=OpType.CONV2D, name="conv1", inputs=[], outputs=["conv_out"])
    relu_node = Node(op_type=OpType.RELU, name="relu1", inputs=["conv_out"], outputs=["relu_out"])

    graph1 = Graph()
    graph1.add_node(conv_node)
    graph1.add_node(relu_node)

    ctx1 = MatchContext()
    result1 = pattern.match(conv_node, graph1, ctx1)
    assert result1 is not None
    assert "conv" in result1.bindings
    assert "relu" in result1.bindings

    # Case 2: Multiple consumers - should NOT match
    conv_node2 = Node(op_type=OpType.CONV2D, name="conv2", inputs=[], outputs=["conv_out2"])
    relu_node2 = Node(op_type=OpType.RELU, name="relu2", inputs=["conv_out2"], outputs=["relu_out2"])
    other_node = Node(op_type=OpType.SIGMOID, name="other", inputs=["conv_out2"], outputs=[])

    graph2 = Graph()
    graph2.add_node(conv_node2)
    graph2.add_node(relu_node2)
    graph2.add_node(other_node)

    ctx2 = MatchContext()
    result2 = pattern.match(conv_node2, graph2, ctx2)
    assert result2 is None  # Should fail - multiple consumers
```

**Step 5: 运行测试**

Run: `pytest tests/test_pattern_concrete.py -v`
Expected: 全部PASS

**Step 6: Commit**

```bash
git add src/nnc_py/pattern/patterns.py tests/test_pattern_concrete.py
git commit -m "feat(pattern): implement concrete pattern classes (Wildcard, Op, Or, And, Use)"
```

---

## Task 4: 实现PatternMatcher

**Files:**
- Create: `src/nnc_py/pattern/matcher.py`
- Test: `tests/test_pattern_matcher.py`

**Step 1: 实现PatternMatcher类**

```python
"""Pattern matching engine for finding pattern matches in graphs."""

from typing import List, Set, Tuple
from nnc_py.pattern.base import DFPattern, PatternMatch, MatchContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node


class PatternMatcher:
    """Efficient pattern matching engine for nnc-py graphs.

    Uses top-down DFS with memoization for efficient matching.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def match_pattern(self, pattern: DFPattern) -> List[PatternMatch]:
        """Find all matches of a pattern in the graph.

        Args:
            pattern: The pattern to match

        Returns:
            List of all successful matches (non-overlapping preferred)
        """
        matches = []
        context = MatchContext()

        # Try matching from each node (topological order preferred)
        for node in self.graph.topological_sort():
            match = pattern.match(node, self.graph, context)
            if match:
                matches.append(match)

        # Filter for non-overlapping matches (greedy by topological order)
        return self._filter_non_overlapping(matches)

    def _filter_non_overlapping(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Filter matches to return non-overlapping ones.

        Uses greedy selection: prefer matches that appear earlier
        in topological order (upstream nodes first).
        """
        if not matches:
            return []

        # Sort by anchor position in topological order
        topo_order = {n: i for i, n in enumerate(self.graph.topological_sort())}
        matches.sort(key=lambda m: topo_order.get(m.anchor, float('inf')))

        selected: List[PatternMatch] = []
        used_nodes: Set[Node] = set()

        for match in matches:
            # Check if any node in this match is already used
            if match.nodes.isdisjoint(used_nodes):
                selected.append(match)
                used_nodes.update(match.nodes)

        return selected

    def match_all_patterns(
        self,
        patterns: List[Tuple[DFPattern, 'FusionPattern']]
    ) -> List[Tuple[PatternMatch, 'FusionPattern']]:
        """Match multiple patterns, returning (match, fusion_pattern) pairs.

        Args:
            patterns: List of (pattern, fusion_pattern) tuples ordered by priority

        Returns:
            List of matches with their associated fusion patterns
        """
        results = []

        for pattern, fusion_pattern in patterns:
            matches = self.match_pattern(pattern)
            for match in matches:
                results.append((match, fusion_pattern))

        # Filter by priority and overlap
        return self._select_by_priority(results)

    def _select_by_priority(
        self,
        results: List[Tuple[PatternMatch, 'FusionPattern']]
    ) -> List[Tuple[PatternMatch, 'FusionPattern']]:
        """Select matches based on pattern priority.

        Higher priority patterns are preferred. When conflicts occur,
        higher priority pattern wins.
        """
        # Sort by priority (higher first)
        results.sort(key=lambda x: x[1].priority, reverse=True)

        selected: List[Tuple[PatternMatch, 'FusionPattern']] = []
        used_nodes: Set[Node] = set()

        for match, fusion_pattern in results:
            if match.nodes.isdisjoint(used_nodes):
                selected.append((match, fusion_pattern))
                used_nodes.update(match.nodes)

        return selected
```

**Step 2: 编写测试**

```python
# tests/test_pattern_matcher.py
import pytest
from nnc_py.pattern.matcher import PatternMatcher
from nnc_py.pattern.patterns import OpPattern, WildcardPattern
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.graph import Graph


def test_matcher_finds_all_matches():
    """Test that PatternMatcher finds all pattern matches."""
    graph = Graph()

    # Create a simple chain: conv -> relu
    conv = Node(op_type=OpType.CONV2D, name="conv1", inputs=[], outputs=["c_out"])
    relu = Node(op_type=OpType.RELU, name="relu1", inputs=["c_out"], outputs=["r_out"])
    relu2 = Node(op_type=OpType.RELU, name="relu2", inputs=[], outputs=[])

    graph.add_node(conv)
    graph.add_node(relu)
    graph.add_node(relu2)

    pattern = OpPattern(OpType.RELU, "r")
    matcher = PatternMatcher(graph)

    matches = matcher.match_pattern(pattern)

    # Should find both relu nodes
    assert len(matches) == 2


def test_matcher_filters_overlapping():
    """Test that matcher returns non-overlapping matches."""
    graph = Graph()

    # Create overlapping potential matches
    conv1 = Node(op_type=OpType.CONV2D, name="conv1", inputs=[], outputs=["c1_out"])
    conv2 = Node(op_type=OpType.CONV2D, name="conv2", inputs=[], outputs=["c2_out"])

    # Both convs are valid matches - should get both since they don't overlap
    graph.add_node(conv1)
    graph.add_node(conv2)

    pattern = OpPattern(OpType.CONV2D, "c")
    matcher = PatternMatcher(graph)

    matches = matcher.match_pattern(pattern)

    assert len(matches) == 2
```

**Step 3: 运行测试**

Run: `pytest tests/test_pattern_matcher.py -v`
Expected: 全部PASS

**Step 4: Commit**

```bash
git add src/nnc_py/pattern/matcher.py tests/test_pattern_matcher.py
git commit -m "feat(pattern): implement PatternMatcher with non-overlapping selection"
```

---

## Task 5: 实现PatternRegistry

**Files:**
- Create: `src/nnc_py/pattern/registry.py`
- Test: `tests/test_pattern_registry.py`

**Step 1: 实现注册表**

```python
"""Pattern registry for fusion patterns."""

from typing import Callable, Dict, List, Optional
from dataclasses import dataclass
from nnc_py.pattern.base import DFPattern, PatternMatch
from nnc_py.ir.node import OpType
from nnc_py.ir.graph import Graph
from nnc_py.ir.context import CompileContext


@dataclass
class FusionPattern:
    """Defines a fusion pattern with metadata and handlers.

    Attributes:
        name: Unique name for this pattern
        pattern: The pattern to match
        priority: Higher values are matched first (default: 100)
        description: Human-readable description
        check_func: Optional validation function
        replace_func: Optional function to create fused node
        fused_op_type: Target OpType for fused operation
    """
    name: str
    pattern: DFPattern
    priority: int = 100
    description: str = ""
    check_func: Optional[Callable[[Graph, PatternMatch], bool]] = None
    replace_func: Optional[Callable[[Graph, PatternMatch, str], 'Node']] = None
    fused_op_type: Optional[OpType] = None


class PatternRegistry:
    """Global registry for fusion patterns.

    Singleton pattern for centralized pattern management.
    """
    _instance: Optional['PatternRegistry'] = None
    _patterns: Dict[str, FusionPattern] = {}

    def __new__(cls) -> 'PatternRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, pattern: FusionPattern) -> None:
        """Register a fusion pattern."""
        if pattern.name in cls._patterns:
            raise ValueError(f"Pattern '{pattern.name}' already registered")
        cls._patterns[pattern.name] = pattern

    @classmethod
    def get(cls, name: str) -> Optional[FusionPattern]:
        """Get a registered pattern by name."""
        return cls._patterns.get(name)

    @classmethod
    def get_all(cls) -> List[FusionPattern]:
        """Get all registered patterns, sorted by priority."""
        return sorted(cls._patterns.values(), key=lambda p: -p.priority)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered patterns (for testing)."""
        cls._patterns.clear()


def register_pattern(
    name: str,
    pattern: DFPattern,
    priority: int = 100,
    description: str = "",
    check_func: Optional[Callable[[Graph, PatternMatch], bool]] = None,
    replace_func: Optional[Callable[[Graph, PatternMatch, str], 'Node']] = None,
    fused_op_type: Optional[OpType] = None,
) -> FusionPattern:
    """Register a fusion pattern.

    Example:
        conv = OpPattern(OpType.CONV2D, "conv")
        relu = OpPattern(OpType.RELU, "relu")

        register_pattern(
            name="conv_relu",
            pattern=conv.only_used_by(relu),
            priority=200,
            description="Conv + ReLU fusion",
            fused_op_type=OpType.FUSED_CONV_RELU
        )

    Args:
        name: Unique pattern name
        pattern: DFPattern to match
        priority: Matching priority (higher first)
        description: Human-readable description
        check_func: Optional validation callback
        replace_func: Optional custom node creation callback
        fused_op_type: Target OpType for fused node

    Returns:
        The registered FusionPattern
    """
    fusion_pattern = FusionPattern(
        name=name,
        pattern=pattern,
        priority=priority,
        description=description,
        check_func=check_func,
        replace_func=replace_func,
        fused_op_type=fused_op_type,
    )
    PatternRegistry.register(fusion_pattern)
    return fusion_pattern
```

**Step 2: 编写测试**

```python
# tests/test_pattern_registry.py
import pytest
from nnc_py.pattern.registry import PatternRegistry, register_pattern, FusionPattern
from nnc_py.pattern.patterns import OpPattern, WildcardPattern


def test_register_pattern():
    """Test registering a new pattern."""
    PatternRegistry.clear()  # Start fresh

    pattern = OpPattern(OpType.CONV2D, "conv")
    fp = register_pattern(
        name="test_conv",
        pattern=pattern,
        priority=100,
        description="Test pattern"
    )

    assert fp.name == "test_conv"
    assert PatternRegistry.get("test_conv") == fp


def test_duplicate_registration_raises():
    """Test that duplicate registration raises error."""
    PatternRegistry.clear()

    pattern = WildcardPattern("wc")
    register_pattern(name="dup", pattern=pattern)

    with pytest.raises(ValueError, match="already registered"):
        register_pattern(name="dup", pattern=pattern)


def test_priority_ordering():
    """Test that patterns are returned in priority order."""
    PatternRegistry.clear()

    register_pattern(name="low", pattern=WildcardPattern("a"), priority=50)
    register_pattern(name="high", pattern=WildcardPattern("b"), priority=200)
    register_pattern(name="mid", pattern=WildcardPattern("c"), priority=100)

    patterns = PatternRegistry.get_all()

    assert patterns[0].name == "high"   # 200
    assert patterns[1].name == "mid"    # 100
    assert patterns[2].name == "low"    # 50


def test_get_pattern_by_name():
    """Test retrieving pattern by name."""
    PatternRegistry.clear()

    pattern = OpPattern(OpType.RELU, "relu")
    fp = register_pattern(name="my_relu", pattern=pattern, priority=150)

    retrieved = PatternRegistry.get("my_relu")
    assert retrieved is fp
    assert retrieved.name == "my_relu"
    assert retrieved.priority == 150
```

**Step 3: 运行测试**

Run: `pytest tests/test_pattern_registry.py -v`
Expected: 全部PASS

**Step 4: Commit**

```bash
git add src/nnc_py/pattern/registry.py tests/test_pattern_registry.py
git commit -m "feat(pattern): implement PatternRegistry and register_pattern function"
```

---

## Task 6: 实现PatternFusionPass

**Files:**
- Create: `src/nnc_py/passes/pattern_fusion.py`
- Modify: `src/nnc_py/passes/__init__.py`

**Step 1: 实现PatternFusionPass**

```python
"""Pattern-based operator fusion pass."""

from typing import Set
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node
from nnc_py.passes.base import PassBase
from nnc_py.pattern.matcher import PatternMatcher
from nnc_py.pattern.registry import PatternRegistry, FusionPattern
from nnc_py.pattern.base import PatternMatch


class PatternFusionPass(PassBase):
    """Fuse operators using declarative pattern matching.

    This pass uses the PatternRegistry to find and apply fusion patterns.
    It replaces the hardcoded pattern matching in OperatorFusionPass.
    """

    @property
    def name(self) -> str:
        return "PatternFusion"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute pattern-based fusion."""
        graph = ctx.graph

        # Get all registered patterns
        fusion_patterns = PatternRegistry.get_all()

        if not fusion_patterns:
            return  # No patterns registered

        # Build (pattern, fusion_pattern) pairs
        pattern_pairs = [(fp.pattern, fp) for fp in fusion_patterns]

        # Create matcher and find all matches
        matcher = PatternMatcher(graph)
        matches_with_patterns = matcher.match_all_patterns(pattern_pairs)

        # Apply fusions
        fused_nodes: Set[str] = set()
        patterns_found = {}

        for match, fusion_pattern in matches_with_patterns:
            # Skip if any nodes already fused
            if any(n.name in fused_nodes for n in match.nodes):
                continue

            # Run check function if provided
            if fusion_pattern.check_func:
                if not fusion_pattern.check_func(graph, match):
                    continue

            # Apply the fusion
            self._apply_fusion(graph, match, fusion_pattern, fused_nodes)
            patterns_found[fusion_pattern.name] = patterns_found.get(fusion_pattern.name, 0) + 1

        # Log results if debug mode
        if ctx.debug:
            self._log_summary(patterns_found, len(graph.nodes))

    def _apply_fusion(
        self,
        graph: Graph,
        match: PatternMatch,
        fusion_pattern: FusionPattern,
        fused_nodes: Set[str],
    ) -> None:
        """Apply a single fusion transformation.

        Args:
            graph: The computation graph
            match: The pattern match
            fusion_pattern: The fusion pattern definition
            fused_nodes: Set to track fused node names
        """
        # Generate unique name for fused node
        fused_name = f"fused_{fusion_pattern.name}_{len(fused_nodes) + 1}"

        # Create fused node
        if fusion_pattern.replace_func:
            fused_node = fusion_pattern.replace_func(graph, match, fused_name)
        elif fusion_pattern.fused_op_type:
            fused_node = self._default_fusion(graph, match, fusion_pattern, fused_name)
        else:
            raise ValueError(f"Fusion pattern {fusion_pattern.name} has no replacement")

        # Add fused node to graph
        graph.add_node(fused_node)

        # Update graph outputs if needed
        self._update_graph_outputs(graph, match, fused_node)

        # Remove original nodes
        for node in match.nodes:
            del graph.nodes[node.name]
            fused_nodes.add(node.name)

    def _default_fusion(
        self,
        graph: Graph,
        match: PatternMatch,
        fusion_pattern: FusionPattern,
        fused_name: str,
    ) -> Node:
        """Default fusion behavior when no replace_func provided.

        Assumes a simple chain pattern and takes inputs from the first node
        and outputs from the last node.
        """
        # Get nodes in topological order
        topo_nodes = self.graph.topological_sort()
        sorted_nodes = [n for n in match.nodes if n in topo_nodes]

        if not sorted_nodes:
            sorted_nodes = list(match.nodes)

        first_node = sorted_nodes[0]
        last_node = sorted_nodes[-1]

        return Node(
            op_type=fusion_pattern.fused_op_type,
            name=fused_name,
            inputs=list(first_node.inputs),
            outputs=list(last_node.outputs),
            attrs=first_node.attrs.copy(),
            metadata={"fused_from": [n.name for n in sorted_nodes]}
        )

    def _update_graph_outputs(
        self,
        graph: Graph,
        match: PatternMatch,
        fused_node: Node,
    ) -> None:
        """Update graph outputs after fusion."""
        # Find all outputs from matched nodes
        for node in match.nodes:
            for out_tensor in node.outputs:
                if out_tensor in graph.outputs:
                    # Replace with fused node's outputs
                    for i, old_out in enumerate(graph.outputs):
                        if old_out == out_tensor:
                            graph.outputs[i] = fused_node.outputs[0]

    def _log_summary(self, patterns_found: dict, node_count: int) -> None:
        """Log fusion summary."""
        print(f"\n{'='*60}")
        print(f"Pattern Fusion Summary")
        print(f"{'='*60}")
        print(f"Total fusions: {sum(patterns_found.values())}")
        print(f"Patterns found:")
        for pattern, count in sorted(patterns_found.items()):
            print(f"  - {pattern}: {count}")
        print(f"Nodes after fusion: {node_count}")
        print(f"{'='*60}")
```

**Step 2: 更新__init__.py导出**

```python
# src/nnc_py/passes/__init__.py 添加
from nnc_py.passes.pattern_fusion import PatternFusionPass

__all__ = [..., "PatternFusionPass"]
```

**Step 3: 编写测试**

```python
# tests/test_pattern_fusion_pass.py
import pytest
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.passes.pattern_fusion import PatternFusionPass
from nnc_py.pattern.registry import register_pattern
from nnc_py.pattern.patterns import OpPattern


def test_pattern_fusion_pass():
    """Test PatternFusionPass with a simple pattern."""
    # Register a test pattern
    register_pattern(
        name="test_conv_relu",
        pattern=OpPattern(OpType.CONV2D, "conv").only_used_by(
            OpPattern(OpType.RELU, "relu")
        ),
        priority=100,
        fused_op_type=OpType.FUSED_CONV_RELU,
    )

    # Create test graph: conv -> relu
    graph = Graph("test")
    conv = Node(
        op_type=OpType.CONV2D,
        name="conv1",
        inputs=["input"],
        outputs=["conv_out"],
        attrs={"kernel_shape": [3, 3, 1, 1]}
    )
    relu = Node(
        op_type=OpType.RELU,
        name="relu1",
        inputs=["conv_out"],
        outputs=["output"]
    )
    graph.add_node(conv)
    graph.add_node(relu)
    graph.outputs = ["output"]

    # Run pass
    ctx = CompileContext()
    ctx.graph = graph
    ctx.debug = False

    pass_obj = PatternFusionPass()
    pass_obj.run(ctx)

    # Verify fusion occurred
    assert "fused_test_conv_relu_1" in graph.nodes
    fused_node = graph.nodes["fused_test_conv_relu_1"]
    assert fused_node.op_type == OpType.FUSED_CONV_RELU
    assert fused_node.inputs == ["input"]
    assert fused_node.outputs == ["output"]

    # Verify original nodes removed
    assert "conv1" not in graph.nodes
    assert "relu1" not in graph.nodes
```

**Step 4: 运行测试**

Run: `pytest tests/test_pattern_fusion_pass.py -v`
Expected: 全部PASS

**Step 5: Commit**

```bash
git add src/nnc_py/passes/pattern_fusion.py src/nnc_py/passes/__init__.py tests/test_pattern_fusion_pass.py
git commit -m "feat(passes): implement PatternFusionPass"
```

---

## Task 7: 实现内置融合模式

**Files:**
- Create: `src/nnc_py/pattern/fusion_patterns.py`
- Modify: `src/nnc_py/pattern/__init__.py`
- Modify: `src/nnc_py/passes/base.py`

**Step 1: 实现融合模式定义**

```python
"""Built-in fusion pattern definitions."""

from nnc_py.pattern.base import PatternMatch
from nnc_py.pattern.registry import register_pattern
from nnc_py.pattern.patterns import OpPattern, WildcardPattern
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.graph import Graph


# Pattern building helpers
def conv(name: str = "conv") -> OpPattern:
    """Create a Conv2D pattern."""
    return OpPattern(OpType.CONV2D, name)

def matmul(name: str = "matmul") -> OpPattern:
    """Create a MatMul pattern."""
    return OpPattern(OpType.MATMUL, name)

def add(name: str = "add") -> OpPattern:
    """Create an Add pattern."""
    return OpPattern(OpType.ADD, name)

def relu(name: str = "relu") -> OpPattern:
    """Create a ReLU pattern."""
    return OpPattern(OpType.RELU, name)

def sigmoid(name: str = "sigmoid") -> OpPattern:
    """Create a Sigmoid pattern."""
    return OpPattern(OpType.SIGMOID, name)

def tanh(name: str = "tanh") -> OpPattern:
    """Create a Tanh pattern."""
    return OpPattern(OpType.TANH, name)

def wildcard(name: str = "wildcard") -> WildcardPattern:
    """Create a wildcard pattern."""
    return WildcardPattern(name)


# Fusion helpers
def _create_fused_conv_relu(graph: Graph, match: PatternMatch, name: str) -> Node:
    """Create fused Conv+ReLU node."""
    conv_node = match.bindings["conv"]
    relu_node = match.bindings["relu"]
    return Node(
        op_type=OpType.FUSED_CONV_RELU,
        name=name,
        inputs=list(conv_node.inputs),
        outputs=list(relu_node.outputs),
        attrs=conv_node.attrs.copy(),
        metadata={"fused_from": [conv_node.name, relu_node.name]}
    )


def _create_fused_conv_sigmoid(graph: Graph, match: PatternMatch, name: str) -> Node:
    """Create fused Conv+Sigmoid node."""
    conv_node = match.bindings["conv"]
    sigmoid_node = match.bindings["sigmoid"]
    return Node(
        op_type=OpType.FUSED_CONV_SIGMOID,
        name=name,
        inputs=list(conv_node.inputs),
        outputs=list(sigmoid_node.outputs),
        attrs=conv_node.attrs.copy(),
        metadata={"fused_from": [conv_node.name, sigmoid_node.name]}
    )


def _create_fused_add_relu(graph: Graph, match: PatternMatch, name: str) -> Node:
    """Create fused Add+ReLU node."""
    add_node = match.bindings["add"]
    relu_node = match.bindings["relu"]
    return Node(
        op_type=OpType.FUSED_ADD_RELU,
        name=name,
        inputs=list(add_node.inputs),
        outputs=list(relu_node.outputs),
        attrs=add_node.attrs.copy(),
        metadata={"fused_from": [add_node.name, relu_node.name]}
    )


def _create_fused_add_sigmoid(graph: Graph, match: PatternMatch, name: str) -> Node:
    """Create fused Add+Sigmoid node."""
    add_node = match.bindings["add"]
    sigmoid_node = match.bindings["sigmoid"]
    return Node(
        op_type=OpType.FUSED_ADD_SIGMOID,
        name=name,
        inputs=list(add_node.inputs),
        outputs=list(sigmoid_node.outputs),
        attrs=add_node.attrs.copy(),
        metadata={"fused_from": [add_node.name, sigmoid_node.name]}
    )


def _create_fused_matmul_relu(graph: Graph, match: PatternMatch, name: str) -> Node:
    """Create fused MatMul+ReLU node."""
    matmul_node = match.bindings["matmul"]
    relu_node = match.bindings["relu"]
    return Node(
        op_type=OpType.FUSED_MATMUL_RELU,
        name=name,
        inputs=list(matmul_node.inputs),
        outputs=list(relu_node.outputs),
        attrs=matmul_node.attrs.copy(),
        metadata={"fused_from": [matmul_node.name, relu_node.name]}
    )


# Register built-in patterns

# Conv + ReLU (high priority - common pattern)
register_pattern(
    name="conv_relu",
    pattern=conv().only_used_by(relu()),
    priority=200,
    description="Conv + ReLU fusion",
    fused_op_type=OpType.FUSED_CONV_RELU,
    replace_func=_create_fused_conv_relu,
)

# Conv + Sigmoid
register_pattern(
    name="conv_sigmoid",
    pattern=conv().only_used_by(sigmoid()),
    priority=200,
    description="Conv + Sigmoid fusion",
    fused_op_type=OpType.FUSED_CONV_SIGMOID,
    replace_func=_create_fused_conv_sigmoid,
)

# Add + ReLU
register_pattern(
    name="add_relu",
    pattern=add().only_used_by(relu()),
    priority=200,
    description="Add + ReLU fusion",
    fused_op_type=OpType.FUSED_ADD_RELU,
    replace_func=_create_fused_add_relu,
)

# Add + Sigmoid
register_pattern(
    name="add_sigmoid",
    pattern=add().only_used_by(sigmoid()),
    priority=200,
    description="Add + Sigmoid fusion",
    fused_op_type=OpType.FUSED_ADD_SIGMOID,
    replace_func=_create_fused_add_sigmoid,
)

# MatMul + ReLU
register_pattern(
    name="matmul_relu",
    pattern=matmul().only_used_by(relu()),
    priority=190,
    description="MatMul + ReLU fusion",
    fused_op_type=OpType.FUSED_MATMUL_RELU,
    replace_func=_create_fused_matmul_relu,
)
```

**Step 2: 更新pattern/__init__.py导出**

```python
# src/nnc_py/pattern/__init__.py 添加
from nnc_py.pattern import fusion_patterns

# Re-export pattern helpers
__all__.extend([
    "conv", "matmul", "add", "relu", "sigmoid", "tanh", "wildcard",
])
```

**Step 3: 添加新的OpType到node.py**

```python
# src/nnc_py/ir/node.py 添加到OpType枚举
class OpType(Enum):
    # ... 现有类型 ...

    # Existing fused operators
    FUSED_CONV_RELU = "FusedConvRelu"
    FUSED_CONV_BIAS_RELU = "FusedConvBiasRelu"
    FUSED_CONV_SIGMOID = "FusedConvSigmoid"
    FUSED_ADD_RELU = "FusedAddRelu"
    FUSED_ADD_SIGMOID = "FusedAddSigmoid"

    # NEW: Extended fused operators
    FUSED_MATMUL_RELU = "FusedMatMulRelu"
```

**Step 4: 更新PassManager使用PatternFusionPass**

```python
# src/nnc_py/passes/base.py 修改get_default_passes方法
# 在导入部分添加:
from nnc_py.passes.pattern_fusion import PatternFusionPass

# 在opt_level >= 3的返回列表中，替换:
if opt_level >= 3:
    return [
        IdentityEliminationPass(),
        DeadCodeEliminationPass(),
        PatternFusionPass(),      # NEW: Pattern-based fusion
        LivenessAnalysisPass(),
        MemoryPlanningPassV2(),
        SpillAnalysisPass(),
    ]
```

**Step 5: 编写测试**

```python
# tests/test_builtin_fusion_patterns.py
import pytest
from nnc_py.pattern import fusion_patterns
from nnc_py.pattern.registry import PatternRegistry


def test_builtin_patterns_registered():
    """Test that all built-in patterns are registered."""
    patterns = PatternRegistry.get_all()

    pattern_names = {p.name for p in patterns}
    assert "conv_relu" in pattern_names
    assert "conv_sigmoid" in pattern_names
    assert "add_relu" in pattern_names
    assert "add_sigmoid" in pattern_names
    assert "matmul_relu" in pattern_names


def test_conv_relu_pattern_definition():
    """Test Conv+ReLU pattern structure."""
    patterns = PatternRegistry.get_all()
    conv_relu = next(p for p in patterns if p.name == "conv_relu")

    assert conv_relu is not None
    assert conv_relu.priority == 200
    assert "Conv + ReLU" in conv_relu.description
```

**Step 6: 运行测试**

Run: `pytest tests/test_builtin_fusion_patterns.py -v`
Expected: 全部PASS

**Step 7: Commit**

```bash
git add src/nnc_py/pattern/fusion_patterns.py src/nnc_py/pattern/__init__.py src/nnc_py/ir/node.py src/nnc_py/passes/base.py tests/test_builtin_fusion_patterns.py
git commit -m "feat(pattern): add built-in fusion patterns (Conv+ReLU, MatMul+ReLU, etc.)"
```

---

## Task 8: 更新C代码生成器支持新融合类型

**Files:**
- Modify: `src/nnc_py/codegen/c_emitter.py`
- Test: `tests/test_c_emitter_coverage.py`

**Step 1: 添加FUSED_MATMUL_RELU处理**

```python
# src/nnc_py/codegen/c_emitter.py 添加到_emit_operator_call方法

    elif node.op_type == OpType.FUSED_MATMUL_RELU:
        self._emit_fused_matmul_relu_call(ctx, node)

# 添加新方法
def _emit_fused_matmul_relu_call(self, ctx: CompileContext, node: Node) -> None:
    """Emit MatMul+ReLU fused operation."""
    # Similar to _emit_fused_conv_relu_call
    input_name = node.inputs[0]
    weight_name = node.inputs[1] if len(node.inputs) > 1 else "NULL"
    output_name = node.outputs[0]

    self._emit_comment(f"Fused MatMul+ReLU: {node.name}")
    self.write_line(f"nnc_matmul_relu({input_name}, {weight_name}, {output_name});")
```

**Step 2: 更新覆盖率测试**

```python
# tests/test_c_emitter_coverage.py 添加
def test_fused_matmul_relu_emission():
    """Test FUSED_MATMUL_RELU code generation."""
    # Create a fused matmul relu node and verify C code generation
    ...
```

**Step 3: 运行测试**

Run: `pytest tests/test_c_emitter_coverage.py::test_fused_matmul_relu_emission -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/nnc_py/codegen/c_emitter.py tests/test_c_emitter_coverage.py
git commit -m "feat(codegen): add FUSED_MATMUL_RELU code generation"
```

---

## Task 9: 运行所有现有测试确保兼容性

**Files:**
- Test: 全部现有测试

**Step 1: 运行所有单元测试**

Run: `pytest tests/ -v -k "not slow"`
Expected: 大部分PASS，某些测试可能需要更新

**Step 2: 运行E2E测试**

Run: `pytest tests/test_operator_fusion_e2e.py -v`
Expected: 如果pattern系统正确实现，应该PASS

**Step 3: 如果有失败，修复问题**

Run: 根据失败信息修复
Expected: 最终全部PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "fix: ensure backward compatibility with existing tests"
```

---

## Task 10: 添加文档

**Files:**
- Create: `docs/pattern_matching_guide.md`

**Step 1: 编写模式匹配指南**

```markdown
# Pattern Matching Guide

This guide explains how to use nnc-py's pattern matching system for operator fusion.

## Overview

nnc-py uses a TVM-style Dataflow Pattern Language (DFPL) for declarative operator fusion.

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

```python
# Or: matches either pattern
pattern = OpPattern(OpType.RELU) | OpPattern(OpType.SIGMOID)

# Use constraint: output consumed by another pattern
pattern = conv.only_used_by(relu)

# And: both patterns match
pattern = conv & relu
```

## Registering Fusion Patterns

```python
from nnc_py.pattern.registry import register_pattern

def create_my_fused_node(graph, match, name):
    # Custom fusion logic
    return Node(...)

register_pattern(
    name="my_pattern",
    pattern=conv().only_used_by(relu()),
    priority=150,
    description="My custom fusion",
    replace_func=create_my_fused_node,
    fused_op_type=OpType.MY_FUSED_OP,
)
```
```

**Step 2: 更新OPTIMIZATION_PASSES.md**

```markdown
# Optimization Passes

## Pattern Fusion Pass

The PatternFusionPass uses a declarative pattern matching system for operator fusion.

### Built-in Patterns

- `conv_relu`: Conv2D + ReLU
- `conv_sigmoid`: Conv2D + Sigmoid
- `add_relu`: Add + ReLU
- `add_sigmoid`: Add + Sigmoid
- `matmul_relu`: MatMul + ReLU

### Adding Custom Patterns

See `docs/pattern_matching_guide.md` for details.
```

**Step 3: Commit**

```bash
git add docs/pattern_matching_guide.md docs/OPTIMIZATION_PASSES.md
git commit -m "docs: add pattern matching guide and update optimization docs"
```

---

## 验证计划

1. **运行所有单元测试**: `pytest tests/ -v`
2. **运行E2E测试**: `pytest tests/test_operator_fusion_e2e.py -v`
3. **编译一个ONNX模型并验证融合**: `python -m nnc_py compile model.onnx --opt-level 3`
4. **检查生成的C代码**: 验证融合算子的代码生成正确

---

**计划完成并保存到 `docs/plans/2025-02-12-pattern-fusion.md`**

**执行选项:**

**1. Subagent-Driven (当前会话)** - 我在此会话中分派子代理执行每个任务，任务间审查，快速迭代

**2. Parallel Session (独立会话)** - 在新会话中使用executing-plans，批量执行并设置检查点

**选择哪种方式？**
