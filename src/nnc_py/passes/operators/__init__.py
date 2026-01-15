"""Operator-specific split rules.

This package contains split rule definitions for individual operator types.
"""

from nnc_py.passes.operators.conv_rules import register_conv2d_split_rule


__all__ = ["register_conv2d_split_rule"]
