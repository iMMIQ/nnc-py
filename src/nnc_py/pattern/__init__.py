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

# Import fusion_patterns to trigger pattern registration
from nnc_py.pattern import fusion_patterns  # noqa: F401

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
