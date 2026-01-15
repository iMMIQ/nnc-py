"""Operator splitting rules and registry.

This module defines the data structures and registry for managing
how operators can be split to reduce memory usage.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

from nnc_py.ir.node import Node, OpType


class SplitAxisBehavior(Enum):
    """How an axis behaves under splitting."""

    FULLY_SPLITTABLE = "full"  # Can split at any point (batch, channels)
    REDUCTION_FORBIDDEN = "reduction"  # Cannot split (reduction axis)
    SHAPE_CHANGE_FORBIDDEN = "shape"  # Cannot split (reshape/transpose)
    REQUIRES_BROADCAST = "broadcast"  # Split requires broadcast handling


@dataclass
class SplitAxisRule:
    """Rule for splitting a specific axis."""

    axis_index: int
    behavior: SplitAxisBehavior
    min_chunk_size: int = 1
    alignment: int = 1


@dataclass
class OperatorSplitRule:
    """Split rule for an operator type."""

    op_type: OpType
    # Which input axes can be split (per input)
    input_split_rules: List[List[SplitAxisRule]]
    # How the split affects outputs (per output)
    output_split_behavior: List[SplitAxisBehavior]
    # Which inputs are reused across all splits (e.g., weights, bias)
    reused_inputs: Set[int] = field(default_factory=set)
    # Function to determine how split propagates to dependent ops
    propagate_split: Optional[Callable[[int], Optional[int]]] = None


@dataclass
class SplitInfo:
    """Information about how to split a single operator."""

    original_node: Node
    split_axis: int  # Which axis to split on
    num_splits: int  # How many pieces to split into
    chunk_sizes: List[int] = field(default_factory=list)  # Size of each chunk
    split_nodes: List[Node] = field(default_factory=list)  # Generated split nodes


@dataclass
class SplitPlan:
    """Plan for splitting operators in a graph."""

    splits: List[SplitInfo] = field(default_factory=list)
    cascades: List["CascadeInfo"] = field(default_factory=list)  # For dependent ops


@dataclass
class CascadeInfo:
    """Information about cascading splits to dependent operators."""

    source_node: Node
    target_node: Node
    source_axis: int
    target_axis: int
    required_splits: int  # Must match source num_splits


class SplitRegistry:
    """Registry of operator split rules.

    This class maintains a mapping from OpType to OperatorSplitRule,
    allowing the system to query how an operator can be split.
    """

    _rules: Dict[OpType, OperatorSplitRule] = {}

    @classmethod
    def register(cls, rule: OperatorSplitRule) -> None:
        """Register a split rule for an operator type.

        Args:
            rule: The split rule to register. If a rule already exists
                  for this op_type, it will be overwritten.
        """
        cls._rules[rule.op_type] = rule

    @classmethod
    def get_rule(cls, op_type: OpType) -> Optional[OperatorSplitRule]:
        """Get the split rule for an operator type.

        Args:
            op_type: The operator type to query.

        Returns:
            The split rule if registered, None otherwise.
        """
        return cls._rules.get(op_type)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered rules (useful for testing)."""
        cls._rules.clear()
