"""Tests for reduction operator split rules (TDD Cycle 15)."""

import pytest

from nnc_py.ir.node import OpType
from nnc_py.ir.split_rules import (
    SplitAxisBehavior,
    SplitRegistry,
)
from nnc_py.passes.operators.reduction_rules import (
    register_reduce_mean_split_rule,
    register_reduce_sum_split_rule,
)


@pytest.fixture(autouse=True)
def register_rules():
    """Register reduction rules for tests."""
    SplitRegistry.clear()
    register_reduce_mean_split_rule()
    register_reduce_sum_split_rule()
    yield
    SplitRegistry.clear()


class TestReduceMeanSplitRule:
    """Test ReduceMean split rules."""

    def test_reduce_mean_has_split_rule(self):
        """Test that ReduceMean has a registered split rule."""
        rule = SplitRegistry.get_rule(OpType.REDUCE_MEAN)
        assert rule is not None
        assert rule.op_type == OpType.REDUCE_MEAN

    def test_reduce_mean_non_reduction_axes_splittable(self):
        """Test that non-reduction axes are splittable."""
        rule = SplitRegistry.get_rule(OpType.REDUCE_MEAN)
        assert rule is not None

        # Should have rules for non-reduction axes
        assert len(rule.input_split_rules[0]) > 0

        # Check that we have at least one fully splittable axis
        splittable = [r for r in rule.input_split_rules[0]
                      if r.behavior == SplitAxisBehavior.FULLY_SPLITTABLE]
        assert len(splittable) > 0

    def test_reduce_mean_has_propagate_split(self):
        """Test that ReduceMean has a propagate_split function."""
        rule = SplitRegistry.get_rule(OpType.REDUCE_MEAN)

        # Reduction ops don't propagate splits in the same way
        # because the output shape is different
        assert rule.propagate_split is not None


class TestReduceSumSplitRule:
    """Test ReduceSum split rules."""

    def test_reduce_sum_has_split_rule(self):
        """Test that ReduceSum has a registered split rule."""
        rule = SplitRegistry.get_rule(OpType.REDUCE_SUM)
        assert rule is not None
        assert rule.op_type == OpType.REDUCE_SUM

    def test_reduce_sum_non_reduction_axes_splittable(self):
        """Test that non-reduction axes are splittable for ReduceSum."""
        rule = SplitRegistry.get_rule(OpType.REDUCE_SUM)

        splittable = [r for r in rule.input_split_rules[0]
                      if r.behavior == SplitAxisBehavior.FULLY_SPLITTABLE]
        assert len(splittable) > 0


class TestReductionOutputShape:
    """Test that reduction ops handle output shape correctly."""

    def test_reduce_mean_output_shape_change(self):
        """Test that ReduceMean output has shape change behavior."""
        rule = SplitRegistry.get_rule(OpType.REDUCE_MEAN)

        # Reduction ops change the output shape
        # This is indicated by the output split behavior
        assert len(rule.output_split_behavior) > 0

    def test_reduce_sum_output_shape_change(self):
        """Test that ReduceSum output has shape change behavior."""
        rule = SplitRegistry.get_rule(OpType.REDUCE_SUM)

        assert len(rule.output_split_behavior) > 0
