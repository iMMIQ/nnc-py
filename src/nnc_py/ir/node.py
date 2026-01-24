"""Computation graph node definition."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class OpType(Enum):
    """Operator type enumeration."""

    # Neural network ops
    CONV2D = "Conv"
    RELU = "Relu"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"
    SOFTMAX = "Softmax"

    # Pooling ops
    MAXPOOL = "MaxPool"
    AVGPOOL = "AveragePool"
    GLOBAL_MAXPOOL = "GlobalMaxPool"
    GLOBAL_AVGPOOL = "GlobalAveragePool"

    # Arithmetic ops
    ADD = "Add"
    MUL = "Mul"
    SUB = "Sub"
    DIV = "Div"
    POW = "Pow"

    # Comparison ops
    EQUAL = "Equal"
    LESS = "Less"
    GREATER = "Greater"

    # Logical ops
    AND = "And"
    OR = "Or"
    XOR = "Xor"
    NOT = "Not"

    # Math ops (unary element-wise)
    SQRT = "Sqrt"
    EXP = "Exp"
    LOG = "Log"
    ABS = "Abs"
    NEG = "Neg"

    # Matrix ops
    MATMUL = "MatMul"
    GEMM = "Gemm"

    # Shape ops
    RESHAPE = "Reshape"
    FLATTEN = "Flatten"
    TRANSPOSE = "Transpose"
    SQUEEZE = "Squeeze"
    UNSQUEEZE = "Unsqueeze"
    SPLIT = "Split"
    TILE = "Tile"
    SHAPE = "Shape"

    # Reduction ops
    REDUCE_MEAN = "ReduceMean"
    REDUCE_SUM = "ReduceSum"

    # Other ops
    CONCAT = "Concat"
    BATCH_NORM = "BatchNormalization"
    LAYER_NORM = "LayerNormalization"
    IDENTITY = "Identity"
    CONSTANT = "Constant"
    CONSTANT_OF_SHAPE = "ConstantOfShape"
    EXPAND = "Expand"
    CLIP = "Clip"
    CAST = "Cast"


@dataclass
class Node:
    """Computation graph node."""

    op_type: OpType
    name: str
    inputs: List[str]  # Input tensor names
    outputs: List[str]  # Output tensor names
    attrs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_computational(self) -> bool:
        """Check if this is a computational node (vs. shape manipulation)."""
        non_computational = {
            OpType.RESHAPE,
            OpType.FLATTEN,
            OpType.TRANSPOSE,
            OpType.SQUEEZE,
            OpType.UNSQUEEZE,
            OpType.TILE,
            OpType.SHAPE,
        }
        return self.op_type not in non_computational

    def get_attr(self, key: str, default: Any = None) -> Any:
        """Get an attribute value with a default."""
        return self.attrs.get(key, default)

    def __repr__(self) -> str:
        return f"Node({self.name}: {self.op_type.value})"
