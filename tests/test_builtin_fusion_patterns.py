# tests/test_builtin_fusion_patterns.py
import pytest
from nnc_py.pattern.registry import PatternRegistry


def test_builtin_patterns_registered():
    """Test that all built-in patterns are registered."""
    # Patterns are registered when fusion_patterns module is imported
    # which happens when importing nnc_py.pattern

    patterns = PatternRegistry.get_all()

    pattern_names = {p.name for p in patterns}
    assert "conv_relu" in pattern_names
    assert "conv_sigmoid" in pattern_names
    assert "add_relu" in pattern_names
    assert "add_sigmoid" in pattern_names
    assert "matmul_relu" in pattern_names


def test_conv_relu_pattern_definition():
    """Test Conv+ReLU pattern structure."""
    patterns = PatternRegistry.get_all()
    conv_relu = next((p for p in patterns if p.name == "conv_relu"), None)

    assert conv_relu is not None
    assert conv_relu.priority == 200
    assert "Conv + ReLU" in conv_relu.description
