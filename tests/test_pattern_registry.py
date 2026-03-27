# tests/test_pattern_registry.py
import pytest
from nnc_py.pattern.registry import PatternRegistry, register_pattern, FusionPattern
from nnc_py.pattern.patterns import OpPattern, WildcardPattern
from nnc_py.ir.node import OpType


def test_register_pattern():
    """Test registering a new pattern."""
    PatternRegistry.clear()  # Start fresh

    pattern = OpPattern(OpType.CONV2D, "conv")
    fp = register_pattern(
        name="test_conv",
        pattern=pattern,
        priority=100,
        description="Test pattern"
    )

    assert fp.name == "test_conv"
    assert PatternRegistry.get("test_conv") == fp


def test_duplicate_registration_raises():
    """Test that duplicate registration raises error."""
    PatternRegistry.clear()

    pattern = WildcardPattern("wc")
    register_pattern(name="dup", pattern=pattern)

    with pytest.raises(ValueError, match="already registered"):
        register_pattern(name="dup", pattern=pattern)


def test_priority_ordering():
    """Test that patterns are returned in priority order."""
    PatternRegistry.clear()

    register_pattern(name="low", pattern=WildcardPattern("a"), priority=50)
    register_pattern(name="high", pattern=WildcardPattern("b"), priority=200)
    register_pattern(name="mid", pattern=WildcardPattern("c"), priority=100)

    patterns = PatternRegistry.get_all()

    assert patterns[0].name == "high"   # 200
    assert patterns[1].name == "mid"    # 100
    assert patterns[2].name == "low"    # 50


def test_get_pattern_by_name():
    """Test retrieving pattern by name."""
    PatternRegistry.clear()

    pattern = OpPattern(OpType.RELU, "relu")
    fp = register_pattern(name="my_relu", pattern=pattern, priority=150)

    retrieved = PatternRegistry.get("my_relu")
    assert retrieved is fp
    assert retrieved.name == "my_relu"
    assert retrieved.priority == 150


def test_registry_snapshot_restore_round_trips_builtin_patterns():
    """Snapshot/restore should preserve builtins and discard test-only patterns."""
    baseline_names = {pattern.name for pattern in PatternRegistry.get_all()}
    assert "conv_relu" in baseline_names

    snapshot = PatternRegistry.snapshot()

    PatternRegistry.clear()
    register_pattern(
        name="my_relu",
        pattern=OpPattern(OpType.RELU, "relu"),
        priority=150,
    )

    polluted_names = {pattern.name for pattern in PatternRegistry.get_all()}
    assert polluted_names == {"my_relu"}

    PatternRegistry.restore(snapshot)

    restored_names = {pattern.name for pattern in PatternRegistry.get_all()}
    assert "conv_relu" in restored_names
    assert "my_relu" not in restored_names
