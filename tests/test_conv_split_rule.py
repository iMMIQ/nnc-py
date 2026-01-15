"""Tests for Conv2D operator split rule (TDD Cycle 2)."""

from nnc_py.ir.node import OpType
from nnc_py.ir.split_rules import (
    SplitRegistry,
    SplitAxisBehavior,
)
from nnc_py.passes.operators.conv_rules import register_conv2d_split_rule


class TestConv2DSplitRule:
    """Test Conv2D split rule registration and properties."""

    def test_conv2d_has_split_rule(self):
        """Test that Conv2D has a registered split rule."""
        register_conv2d_split_rule()
        rule = SplitRegistry.get_rule(OpType.CONV2D)
        assert rule is not None

    def test_conv2d_batch_axis_is_splittable(self):
        """Test that batch axis (0) is splittable for Conv2D."""
        register_conv2d_split_rule()
        rule = SplitRegistry.get_rule(OpType.CONV2D)

        # Get splittable axes for first input
        splittable_axes = [
            r.axis_index for r in rule.input_split_rules[0]
            if r.behavior == SplitAxisBehavior.FULLY_SPLITTABLE
        ]
        assert 0 in splittable_axes  # Batch axis

    def test_conv2d_channel_axis_is_splittable(self):
        """Test that channel axis (1) is splittable for Conv2D."""
        register_conv2d_split_rule()
        rule = SplitRegistry.get_rule(OpType.CONV2D)

        splittable_axes = [
            r.axis_index for r in rule.input_split_rules[0]
            if r.behavior == SplitAxisBehavior.FULLY_SPLITTABLE
        ]
        assert 1 in splittable_axes  # Channel axis (NCHW layout)

    def test_conv2d_spatial_axes_are_splittable(self):
        """Test that spatial axes (2, 3) are splittable for Conv2D."""
        register_conv2d_split_rule()
        rule = SplitRegistry.get_rule(OpType.CONV2D)

        splittable_axes = [
            r.axis_index for r in rule.input_split_rules[0]
            if r.behavior == SplitAxisBehavior.FULLY_SPLITTABLE
        ]
        assert 2 in splittable_axes  # Height
        assert 3 in splittable_axes  # Width

    def test_conv2d_reused_inputs(self):
        """Test that Conv2D weights and bias are marked as reused inputs."""
        register_conv2d_split_rule()
        rule = SplitRegistry.get_rule(OpType.CONV2D)

        # Input 1 (weights) should be reused
        assert 1 in rule.reused_inputs
        # Input 2 (bias) should be reused if present
        assert 2 in rule.reused_inputs

    def test_conv2d_output_split_behavior(self):
        """Test that Conv2D output can be split along corresponding axes."""
        register_conv2d_split_rule()
        rule = SplitRegistry.get_rule(OpType.CONV2D)

        # Output should support splitting
        assert len(rule.output_split_behavior) >= 1
        assert rule.output_split_behavior[0] == SplitAxisBehavior.FULLY_SPLITTABLE

    def test_conv2d_weights_require_broadcast(self):
        """Test that Conv2D weights are marked as requiring broadcast handling."""
        register_conv2d_split_rule()
        rule = SplitRegistry.get_rule(OpType.CONV2D)

        # Weights (input 1) should have broadcast behavior
        weight_rules = rule.input_split_rules[1]
        assert len(weight_rules) > 0
        assert weight_rules[0].behavior == SplitAxisBehavior.REQUIRES_BROADCAST
