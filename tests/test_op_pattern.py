import pytest
from nnc_py.ir.node import OpType
from nnc_py.ir.op_pattern import get_op_pattern_kind, OpPatternKind


def test_conv_pattern_kind():
    """Conv2D should be kOutEWiseFusable."""
    assert get_op_pattern_kind(OpType.CONV2D) == OpPatternKind.kOutEWiseFusable


def test_elemwise_pattern_kind():
    """Element-wise ops should be kElemWise."""
    assert get_op_pattern_kind(OpType.RELU) == OpPatternKind.kElemWise
    assert get_op_pattern_kind(OpType.ADD) == OpPatternKind.kElemWise


def test_injective_pattern_kind():
    """Injective ops like reshape should be kInjective."""
    assert get_op_pattern_kind(OpType.RESHAPE) == OpPatternKind.kInjective


def test_opaque_pattern_kind():
    """Opaque ops like pooling should be kOpaque."""
    assert get_op_pattern_kind(OpType.MAXPOOL) == OpPatternKind.kOpaque
