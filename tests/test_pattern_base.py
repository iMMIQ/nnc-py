# tests/test_pattern_base.py
import pytest
from nnc_py.pattern.base import DFPattern, PatternMatch, MatchContext


def test_pattern_match_dataclass():
    """Test PatternMatch dataclass structure."""
    # Mock node-like objects
    class MockNode:
        def __init__(self, name):
            self.name = name
        def __hash__(self):
            return hash(self.name)
        def __eq__(self, other):
            return isinstance(other, MockNode) and self.name == other.name

    node1 = MockNode("node1")
    match = PatternMatch(
        bindings={"x": node1},
        anchor=node1,
        node_names={node1.name}
    )
    assert match.bindings["x"] == node1
    assert match.anchor == node1
    assert node1.name in match.node_names


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
