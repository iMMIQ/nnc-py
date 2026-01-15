"""Reduction operator split rules (ReduceMean, ReduceSum)."""

from typing import Optional

from nnc_py.ir.node import OpType
from nnc_py.ir.split_rules import (
    SplitAxisBehavior,
    SplitAxisRule,
    SplitRegistry,
    OperatorSplitRule,
)


def _create_reduction_rule(
    op_type: OpType,
    default_reduce_axis: int = 1
) -> OperatorSplitRule:
    """Create a split rule for reduction operators.

    Reduction operators reduce along a specific axis:
    - The reduction axis cannot be split (REDUCTION_FORBIDDEN)
    - Other axes are fully splittable
    - Output shape changes (one less dimension)

    Args:
        op_type: The operator type (ReduceMean, ReduceSum)
        default_reduce_axis: Default axis to reduce (can be overridden by attrs)

    Returns:
        An OperatorSplitRule for the reduction operator.
    """
    # For a typical 4D tensor [Batch, C, H, W] reducing on axis 1 (C):
    # - Axis 0 (Batch): FULLY_SPLITTABLE
    # - Axis 1 (Channels): REDUCTION_FORBIDDEN (this is the reduction axis)
    # - Axis 2 (Height): FULLY_SPLITTABLE
    # - Axis 3 (Width): FULLY_SPLITTABLE

    input_split_rules = [[
        SplitAxisRule(0, SplitAxisBehavior.FULLY_SPLITTABLE),      # Batch
        SplitAxisRule(default_reduce_axis, SplitAxisBehavior.REDUCTION_FORBIDDEN),  # Reduction axis
        SplitAxisRule(2, SplitAxisBehavior.FULLY_SPLITTABLE),      # Other axes
        SplitAxisRule(3, SplitAxisBehavior.FULLY_SPLITTABLE),
    ]]

    # Reduction changes output shape - the reduction axis is removed
    # Splitting on a non-reduction axis (e.g., batch) means:
    # - Each split produces a smaller output along that axis
    # - But the reduction axis is still removed
    def propagate_reduction_split(axis: int) -> Optional[int]:
        """Propagate split for reduction ops.

        If we split on axis i < reduction_axis: output split on axis i
        If we split on axis i > reduction_axis: output split on axis i-1
        (because the reduction axis is removed)
        """
        if axis == default_reduce_axis:
            return None  # Cannot split on reduction axis
        if axis < default_reduce_axis:
            return axis
        # axis > reduction_axis: shift down by 1
        return max(0, axis - 1)

    return OperatorSplitRule(
        op_type=op_type,
        input_split_rules=input_split_rules,
        output_split_behavior=[SplitAxisBehavior.SHAPE_CHANGE_FORBIDDEN],
        reused_inputs=set(),
        propagate_split=propagate_reduction_split,
    )


def register_reduce_mean_split_rule() -> None:
    """Register the ReduceMean split rule."""
    rule = _create_reduction_rule(OpType.REDUCE_MEAN, default_reduce_axis=1)
    SplitRegistry.register(rule)


def register_reduce_sum_split_rule() -> None:
    """Register the ReduceSum split rule."""
    rule = _create_reduction_rule(OpType.REDUCE_SUM, default_reduce_axis=1)
    SplitRegistry.register(rule)


def register_all_reduction_rules() -> None:
    """Register all reduction operator split rules."""
    register_reduce_mean_split_rule()
    register_reduce_sum_split_rule()
