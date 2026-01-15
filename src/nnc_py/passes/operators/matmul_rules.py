"""MatMul operator split rules."""

from nnc_py.ir.node import OpType
from nnc_py.ir.split_rules import (
    SplitAxisBehavior,
    SplitAxisRule,
    SplitRegistry,
    OperatorSplitRule,
)


def register_matmul_split_rule() -> None:
    """Register the MatMul split rule.

    MatMul: [Batch, M, K] @ [K, N] = [Batch, M, N]

    Splittable axes:
    - Batch (axis 0): Fully splittable
    - M (axis 1): Fully splittable
    - K (axis 2): Not splittable (requires both inputs to be split together)
    - N (axis 3): Not splittable for now (second input)

    The second input (weights/bias) is reused across all splits.
    """
    rule = OperatorSplitRule(
        op_type=OpType.MATMUL,
        # First input: [Batch, M, K]
        input_split_rules=[
            [
                SplitAxisRule(0, SplitAxisBehavior.FULLY_SPLITTABLE),  # Batch
                SplitAxisRule(1, SplitAxisBehavior.FULLY_SPLITTABLE),  # M
            ],
        ],
        # Output: [Batch, M, N] - same split behavior as input
        output_split_behavior=[SplitAxisBehavior.FULLY_SPLITTABLE],
        # Second input is reused (weights)
        reused_inputs={1},
        # Propagate split: axis 0->0, axis 1->1
        propagate_split=lambda axis: axis if axis in (0, 1) else None,
    )
    SplitRegistry.register(rule)
