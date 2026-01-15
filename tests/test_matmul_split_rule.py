"""Tests for MatMul split rules (TDD Cycle 13)."""

import pytest

from nnc_py.ir.node import OpType
from nnc_py.ir.split_rules import (
    SplitAxisBehavior,
    SplitAxisRule,
    SplitRegistry,
)
from nnc_py.passes.operators.matmul_rules import register_matmul_split_rule


@pytest.fixture(autouse=True)
def register_rule():
    """Register MatMul rule for tests."""
    SplitRegistry.clear()
    register_matmul_split_rule()
    yield
    SplitRegistry.clear()


class TestMatMulSplitRule:
    """Test MatMul split rules."""

    def test_matmul_has_split_rule(self):
        """Test that MatMul has a registered split rule."""
        rule = SplitRegistry.get_rule(OpType.MATMUL)
        assert rule is not None
        assert rule.op_type == OpType.MATMUL

    def test_matmul_batch_axis_is_splittable(self):
        """Test that batch axis (axis 0) is fully splittable."""
        rule = SplitRegistry.get_rule(OpType.MATMUL)
        assert rule is not None

        # First input should have batch axis splittable
        batch_rules = [r for r in rule.input_split_rules[0] if r.axis_index == 0]
        assert len(batch_rules) == 1
        assert batch_rules[0].behavior == SplitAxisBehavior.FULLY_SPLITTABLE

    def test_matmul_m_axis_is_splittable(self):
        """Test that M axis (axis 1) is fully splittable."""
        rule = SplitRegistry.get_rule(OpType.MATMUL)

        m_rules = [r for r in rule.input_split_rules[0] if r.axis_index == 1]
        assert len(m_rules) == 1
        assert m_rules[0].behavior == SplitAxisBehavior.FULLY_SPLITTABLE

    def test_matmul_second_input_not_splittable(self):
        """Test that second input (weights) is not splittable (reused)."""
        rule = SplitRegistry.get_rule(OpType.MATMUL)

        # Second input should be in reused_inputs
        assert 1 in rule.reused_inputs

    def test_matmul_output_propagates_split(self):
        """Test that MatMul output split propagates from input."""
        rule = SplitRegistry.get_rule(OpType.MATMUL)

        # Output should have split behavior
        assert len(rule.output_split_behavior) > 0
        assert rule.output_split_behavior[0] == SplitAxisBehavior.FULLY_SPLITTABLE

    def test_matmul_has_propagate_split_function(self):
        """Test that MatMul has a propagate_split function."""
        rule = SplitRegistry.get_rule(OpType.MATMUL)

        assert rule.propagate_split is not None
        # Split on axis 0 should propagate to axis 0
        assert rule.propagate_split(0) == 0
        # Split on axis 1 should propagate to axis 1
        assert rule.propagate_split(1) == 1
