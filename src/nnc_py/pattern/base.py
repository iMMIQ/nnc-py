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
