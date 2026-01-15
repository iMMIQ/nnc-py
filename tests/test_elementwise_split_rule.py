"""Tests for elementwise operator split rules (TDD Cycle 14)."""

import pytest

from nnc_py.ir.node import OpType
from nnc_py.ir.split_rules import (
    SplitAxisBehavior,
    SplitRegistry,
)
from nnc_py.passes.operators.elementwise_rules import (
    register_add_split_rule,
    register_mul_split_rule,
    register_sub_split_rule,
    register_div_split_rule,
)


@pytest.fixture(autouse=True)
def register_rules():
    """Register elementwise rules for tests."""
    SplitRegistry.clear()
    register_add_split_rule()
    register_mul_split_rule()
    register_sub_split_rule()
    register_div_split_rule()
    yield
    SplitRegistry.clear()


class TestAddSplitRule:
    """Test Add split rules."""

    def test_add_has_split_rule(self):
        """Test that Add has a registered split rule."""
        rule = SplitRegistry.get_rule(OpType.ADD)
        assert rule is not None
        assert rule.op_type == OpType.ADD

    def test_add_all_axes_splittable(self):
        """Test that all axes are splittable for Add."""
        rule = SplitRegistry.get_rule(OpType.ADD)

        # Should have rules for at least first 4 axes
        assert len(rule.input_split_rules[0]) >= 4

        # All should be fully splittable
        for axis_rule in rule.input_split_rules[0]:
            assert axis_rule.behavior == SplitAxisBehavior.FULLY_SPLITTABLE

    def test_add_second_input_splittable(self):
        """Test that second input is also splittable for Add."""
        rule = SplitRegistry.get_rule(OpType.ADD)

        # Second input should have split rules (may require broadcast handling)
        assert len(rule.input_split_rules) >= 2

    def test_add_output_splittable(self):
        """Test that Add output is splittable."""
        rule = SplitRegistry.get_rule(OpType.ADD)

        assert rule.output_split_behavior[0] == SplitAxisBehavior.FULLY_SPLITTABLE


class TestMulSplitRule:
    """Test Mul split rules."""

    def test_mul_has_split_rule(self):
        """Test that Mul has a registered split rule."""
        rule = SplitRegistry.get_rule(OpType.MUL)
        assert rule is not None
        assert rule.op_type == OpType.MUL

    def test_mul_all_axes_splittable(self):
        """Test that all axes are splittable for Mul."""
        rule = SplitRegistry.get_rule(OpType.MUL)

        assert len(rule.input_split_rules[0]) >= 4
        for axis_rule in rule.input_split_rules[0]:
            assert axis_rule.behavior == SplitAxisBehavior.FULLY_SPLITTABLE


class TestSubSplitRule:
    """Test Sub split rules."""

    def test_sub_has_split_rule(self):
        """Test that Sub has a registered split rule."""
        rule = SplitRegistry.get_rule(OpType.SUB)
        assert rule is not None
        assert rule.op_type == OpType.SUB


class TestDivSplitRule:
    """Test Div split rules."""

    def test_div_has_split_rule(self):
        """Test that Div has a registered split rule."""
        rule = SplitRegistry.get_rule(OpType.DIV)
        assert rule is not None
        assert rule.op_type == OpType.DIV


class TestElementwiseBroadcastHandling:
    """Test broadcast handling for elementwise ops."""

    def test_broadcast_input_marked(self):
        """Test that broadcast inputs are properly marked."""
        rule = SplitRegistry.get_rule(OpType.ADD)

        # Second input may be broadcast - check for special handling
        # For now, we just verify the rule exists and has proper structure
        assert len(rule.input_split_rules) >= 2

    def test_elementwise_propagate_split(self):
        """Test that elementwise ops propagate splits correctly."""
        rule = SplitRegistry.get_rule(OpType.ADD)

        # Elementwise ops should propagate split axis
        assert rule.propagate_split is not None
        assert rule.propagate_split(0) == 0
