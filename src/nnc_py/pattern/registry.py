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
