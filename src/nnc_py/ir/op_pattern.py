"""Operator pattern classification for fusion.

This module defines the pattern kinds for operators based on TVM's fusion strategy.
The pattern kind determines how an operator can be fused with others in the graph.
"""

from enum import IntEnum
from typing import Dict

from nnc_py.ir.node import OpType


class OpPatternKind(IntEnum):
    """Operator pattern kind for fusion classification.

    These values follow TVM's pattern classification strategy:
    - kOpaque (0): Cannot be fused (e.g., pooling, reduction)
    - kElemWise (1): Element-wise operation
    - kBroadcast (2): Broadcasting operation (e.g., add with different shapes)
    - kInjective (3): Injective operation (reshape, transpose)
    - kOutEWiseFusable (4): Output element-wise fusable (e.g., conv2d, matmul)
    """

    kOpaque = 0
    kElemWise = 1
    kBroadcast = 2
    kInjective = 3
    kOutEWiseFusable = 4


# Pattern classification lookup table
# Based on TVM's strategy for dominator-based fusion
_OP_PATTERN_MAP: Dict[OpType, OpPatternKind] = {
    # kOutEWiseFusable: Reduction-like ops that can fuse with element-wise
    OpType.CONV2D: OpPatternKind.kOutEWiseFusable,
    OpType.MATMUL: OpPatternKind.kOutEWiseFusable,
    OpType.GEMM: OpPatternKind.kOutEWiseFusable,

    # kElemWise: Element-wise operations
    OpType.RELU: OpPatternKind.kElemWise,
    OpType.SIGMOID: OpPatternKind.kElemWise,
    OpType.TANH: OpPatternKind.kElemWise,
    OpType.ADD: OpPatternKind.kElemWise,
    OpType.MUL: OpPatternKind.kElemWise,
    OpType.SUB: OpPatternKind.kElemWise,
    OpType.DIV: OpPatternKind.kElemWise,
    OpType.POW: OpPatternKind.kElemWise,
    OpType.EQUAL: OpPatternKind.kElemWise,
    OpType.LESS: OpPatternKind.kElemWise,
    OpType.GREATER: OpPatternKind.kElemWise,
    OpType.AND: OpPatternKind.kElemWise,
    OpType.OR: OpPatternKind.kElemWise,
    OpType.XOR: OpPatternKind.kElemWise,
    OpType.NOT: OpPatternKind.kElemWise,
    OpType.SQRT: OpPatternKind.kElemWise,
    OpType.EXP: OpPatternKind.kElemWise,
    OpType.LOG: OpPatternKind.kElemWise,
    OpType.ABS: OpPatternKind.kElemWise,
    OpType.NEG: OpPatternKind.kElemWise,
    OpType.CLIP: OpPatternKind.kElemWise,
    OpType.IDENTITY: OpPatternKind.kElemWise,

    # kBroadcast: Broadcasting operations
    OpType.BATCH_NORM: OpPatternKind.kBroadcast,
    OpType.LAYER_NORM: OpPatternKind.kBroadcast,
    OpType.SOFTMAX: OpPatternKind.kBroadcast,
    OpType.CONCAT: OpPatternKind.kBroadcast,
    OpType.EXPAND: OpPatternKind.kBroadcast,

    # kInjective: Shape manipulation operations
    OpType.RESHAPE: OpPatternKind.kInjective,
    OpType.FLATTEN: OpPatternKind.kInjective,
    OpType.TRANSPOSE: OpPatternKind.kInjective,
    OpType.SQUEEZE: OpPatternKind.kInjective,
    OpType.UNSQUEEZE: OpPatternKind.kInjective,
    OpType.TILE: OpPatternKind.kInjective,
    OpType.SPLIT: OpPatternKind.kInjective,
    OpType.CAST: OpPatternKind.kInjective,

    # kOpaque: Operations that cannot be fused
    OpType.MAXPOOL: OpPatternKind.kOpaque,
    OpType.AVGPOOL: OpPatternKind.kOpaque,
    OpType.GLOBAL_MAXPOOL: OpPatternKind.kOpaque,
    OpType.GLOBAL_AVGPOOL: OpPatternKind.kOpaque,
    OpType.REDUCE_MEAN: OpPatternKind.kOpaque,
    OpType.REDUCE_SUM: OpPatternKind.kOpaque,
    OpType.SHAPE: OpPatternKind.kOpaque,
    OpType.CONSTANT: OpPatternKind.kOpaque,
    OpType.CONSTANT_OF_SHAPE: OpPatternKind.kOpaque,
    OpType.LSTM: OpPatternKind.kOpaque,
    OpType.GATHER: OpPatternKind.kOpaque,

    # Fused operators are treated as opaque since they're already fused
    OpType.FUSED_CONV_RELU: OpPatternKind.kOpaque,
    OpType.FUSED_CONV_BIAS_RELU: OpPatternKind.kOpaque,
    OpType.FUSED_CONV_SIGMOID: OpPatternKind.kOpaque,
    OpType.FUSED_ADD_RELU: OpPatternKind.kOpaque,
    OpType.FUSED_ADD_SIGMOID: OpPatternKind.kOpaque,
    OpType.FUSED_MATMUL_RELU: OpPatternKind.kOpaque,
}


def get_op_pattern_kind(op_type: OpType) -> OpPatternKind:
    """Get the pattern kind for a given operator type.

    Args:
        op_type: The operator type to classify.

    Returns:
        The OpPatternKind classification for the operator.

    Raises:
        ValueError: If the operator type is not recognized.
    """
    if op_type not in _OP_PATTERN_MAP:
        raise ValueError(f"Unknown operator type: {op_type}")
    return _OP_PATTERN_MAP[op_type]


def combine_pattern_kind(p1: OpPatternKind, p2: OpPatternKind) -> OpPatternKind:
    """Combine two pattern kinds to determine the resulting fusion pattern.

    The combination follows TVM's strategy:
    - If either is kOpaque, result is kOpaque (cannot fuse)
    - If either is kOutEWiseFusable, result is kOutEWiseFusable
    - Otherwise, take the maximum (more restrictive pattern)

    Args:
        p1: First pattern kind.
        p2: Second pattern kind.

    Returns:
        The combined pattern kind.
    """
    # Opaque blocks everything
    if p1 == OpPatternKind.kOpaque or p2 == OpPatternKind.kOpaque:
        return OpPatternKind.kOpaque

    # kOutEWiseFusable propagates
    if p1 == OpPatternKind.kOutEWiseFusable or p2 == OpPatternKind.kOutEWiseFusable:
        return OpPatternKind.kOutEWiseFusable

    # For kElemWise, kBroadcast, kInjective: take the maximum
    # (kBroadcast > kElemWise, kInjective > kBroadcast)
    return OpPatternKind(max(p1, p2))
