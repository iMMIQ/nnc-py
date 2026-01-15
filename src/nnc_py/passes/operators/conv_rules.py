"""Conv2D operator split rules."""

from nnc_py.ir.node import OpType
from nnc_py.ir.split_rules import (
    SplitRegistry,
    SplitAxisBehavior,
    SplitAxisRule,
    OperatorSplitRule,
)


def register_conv2d_split_rule() -> None:
    """Register the split rule for Conv2D operator.

    Conv2D can be split along:
    - Batch dimension (axis 0)
    - Output channels dimension (axis 1 for NCHW)
    - Spatial dimensions (axes 2, 3 for height, width)

    Weights and bias are reused across all splits.
    """
    rule = OperatorSplitRule(
        op_type=OpType.CONV2D,
        # Input 0: activation tensor [N, C, H, W]
        input_split_rules=[
            [
                SplitAxisRule(0, SplitAxisBehavior.FULLY_SPLITTABLE),  # Batch
                SplitAxisRule(1, SplitAxisBehavior.FULLY_SPLITTABLE),  # Channels
                SplitAxisRule(2, SplitAxisBehavior.FULLY_SPLITTABLE),  # Height
                SplitAxisRule(3, SplitAxisBehavior.FULLY_SPLITTABLE),  # Width
            ],
            # Input 1: weights [O, I, kH, kW] - reused across splits
            [
                SplitAxisRule(0, SplitAxisBehavior.REQUIRES_BROADCAST),
            ],
            # Input 2: bias [O] - reused across splits
            [
                SplitAxisRule(0, SplitAxisBehavior.REQUIRES_BROADCAST),
            ],
        ],
        output_split_behavior=[SplitAxisBehavior.FULLY_SPLITTABLE],
        reused_inputs={1, 2},  # Weights and bias are reused
    )

    SplitRegistry.register(rule)
