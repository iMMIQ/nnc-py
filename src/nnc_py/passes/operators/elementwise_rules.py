"""Elementwise operator split rules (Add, Mul, Sub, Div)."""

from nnc_py.ir.node import OpType
from nnc_py.ir.split_rules import (
    SplitAxisBehavior,
    SplitAxisRule,
    SplitRegistry,
    OperatorSplitRule,
)


def _create_elementwise_rule(op_type: OpType) -> OperatorSplitRule:
    """Create a split rule for elementwise operators.

    Elementwise operators work on each element independently:
    - All axes are fully splittable
    - Output shape matches input shape
    - Broadcast inputs need special handling (marked as REQUIRES_BROADCAST)
    - Split axis propagates directly to output

    Args:
        op_type: The operator type (Add, Mul, Sub, Div)

    Returns:
        An OperatorSplitRule for the elementwise operator.
    """
    # Support splitting on first 4 axes (can be extended)
    # First input: all axes fully splittable
    first_input_rules = [
        SplitAxisRule(i, SplitAxisBehavior.FULLY_SPLITTABLE)
        for i in range(4)
    ]

    # Second input: first axis may require broadcast handling
    second_input_rules = [
        SplitAxisRule(0, SplitAxisBehavior.REQUIRES_BROADCAST),
    ] + [
        SplitAxisRule(i, SplitAxisBehavior.FULLY_SPLITTABLE)
        for i in range(1, 4)
    ]

    return OperatorSplitRule(
        op_type=op_type,
        input_split_rules=[first_input_rules, second_input_rules],
        output_split_behavior=[SplitAxisBehavior.FULLY_SPLITTABLE],
        reused_inputs=set(),  # No inputs are reused
        propagate_split=lambda axis: axis,  # Direct propagation
    )


def register_add_split_rule() -> None:
    """Register the Add split rule."""
    rule = _create_elementwise_rule(OpType.ADD)
    SplitRegistry.register(rule)


def register_mul_split_rule() -> None:
    """Register the Mul split rule."""
    rule = _create_elementwise_rule(OpType.MUL)
    SplitRegistry.register(rule)


def register_sub_split_rule() -> None:
    """Register the Sub split rule."""
    rule = _create_elementwise_rule(OpType.SUB)
    SplitRegistry.register(rule)


def register_div_split_rule() -> None:
    """Register the Div split rule."""
    rule = _create_elementwise_rule(OpType.DIV)
    SplitRegistry.register(rule)


def register_all_elementwise_rules() -> None:
    """Register all elementwise operator split rules."""
    register_add_split_rule()
    register_mul_split_rule()
    register_sub_split_rule()
    register_div_split_rule()
