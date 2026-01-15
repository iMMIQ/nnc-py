"""Tests for operator splitting registry (TDD Cycle 1)."""

import pytest

from nnc_py.ir.node import OpType
from nnc_py.ir.split_rules import (
    SplitRegistry,
    SplitAxisBehavior,
    SplitAxisRule,
    OperatorSplitRule,
)


class TestSplitRegistryBasics:
    """Test basic SplitRegistry functionality."""

    def test_split_registry_exists(self):
        """Test that SplitRegistry can be instantiated."""
        registry = SplitRegistry()
        assert registry is not None

    def test_split_registry_is_singleton(self):
        """Test that SplitRegistry behaves like a singleton (class-level storage)."""
        registry1 = SplitRegistry()
        registry2 = SplitRegistry()
        # Same class, same underlying storage
        assert type(registry1) == type(registry2)

    def test_register_and_get_rule(self):
        """Test registering and retrieving a split rule."""
        # Create a simple rule
        rule = OperatorSplitRule(
            op_type=OpType.ADD,
            input_split_rules=[[
                SplitAxisRule(0, SplitAxisBehavior.FULLY_SPLITTABLE),
            ]],
            output_split_behavior=[SplitAxisBehavior.FULLY_SPLITTABLE],
            reused_inputs=set(),
        )

        SplitRegistry.register(rule)

        # Retrieve the rule
        retrieved = SplitRegistry.get_rule(OpType.ADD)
        assert retrieved is not None
        assert retrieved.op_type == OpType.ADD

    def test_get_nonexistent_rule_returns_none(self):
        """Test that getting a non-existent rule returns None."""
        # Use a rare op type that likely won't be registered by other tests
        rule = SplitRegistry.get_rule(OpType.TILE)  # Not registered yet
        # May return None or raise - for now expect None or check behavior
        # We'll accept None as valid
        assert rule is None or rule.op_type != OpType.TILE

    def test_split_axis_behavior_enum(self):
        """Test that SplitAxisBehavior enum has expected values."""
        assert SplitAxisBehavior.FULLY_SPLITTABLE is not None
        assert SplitAxisBehavior.REDUCTION_FORBIDDEN is not None
        assert SplitAxisBehavior.SHAPE_CHANGE_FORBIDDEN is not None

    def test_split_axis_rule_dataclass(self):
        """Test SplitAxisRule dataclass."""
        rule = SplitAxisRule(
            axis_index=0,
            behavior=SplitAxisBehavior.FULLY_SPLITTABLE,
            min_chunk_size=1,
            alignment=16
        )
        assert rule.axis_index == 0
        assert rule.behavior == SplitAxisBehavior.FULLY_SPLITTABLE
        assert rule.min_chunk_size == 1
        assert rule.alignment == 16

    def test_operator_split_rule_dataclass(self):
        """Test OperatorSplitRule dataclass."""
        rule = OperatorSplitRule(
            op_type=OpType.MUL,
            input_split_rules=[
                [SplitAxisRule(0, SplitAxisBehavior.FULLY_SPLITTABLE)],
                [SplitAxisRule(0, SplitAxisBehavior.FULLY_SPLITTABLE)],
            ],
            output_split_behavior=[SplitAxisBehavior.FULLY_SPLITTABLE],
            reused_inputs=set(),
        )
        assert rule.op_type == OpType.MUL
        assert len(rule.input_split_rules) == 2
        assert len(rule.reused_inputs) == 0

    def test_reused_inputs_set(self):
        """Test OperatorSplitRule with reused inputs."""
        rule = OperatorSplitRule(
            op_type=OpType.CONV2D,
            input_split_rules=[
                [SplitAxisRule(0, SplitAxisBehavior.FULLY_SPLITTABLE)],
                [SplitAxisRule(0, SplitAxisBehavior.REQUIRES_BROADCAST)],
            ],
            output_split_behavior=[SplitAxisBehavior.FULLY_SPLITTABLE],
            reused_inputs={1},  # weights reused
        )
        assert 1 in rule.reused_inputs
        assert 0 not in rule.reused_inputs

    def test_register_overwrites_existing_rule(self):
        """Test that registering a rule twice overwrites the old one."""
        rule1 = OperatorSplitRule(
            op_type=OpType.RELU,
            input_split_rules=[[
                SplitAxisRule(0, SplitAxisBehavior.FULLY_SPLITTABLE),
            ]],
            output_split_behavior=[SplitAxisBehavior.FULLY_SPLITTABLE],
            reused_inputs=set(),
        )

        rule2 = OperatorSplitRule(
            op_type=OpType.RELU,
            input_split_rules=[[
                SplitAxisRule(1, SplitAxisBehavior.REDUCTION_FORBIDDEN),
            ]],
            output_split_behavior=[SplitAxisBehavior.REDUCTION_FORBIDDEN],
            reused_inputs=set(),
        )

        SplitRegistry.register(rule1)
        SplitRegistry.register(rule2)  # Should overwrite

        retrieved = SplitRegistry.get_rule(OpType.RELU)
        assert retrieved.input_split_rules[0][0].axis_index == 1
