"""Tests for NameManager and C identifier sanitization."""

import pytest

from nnc_py.utils.name_manager import NameManager


class TestNameManager:
    """Test name sanitization and symbol mapping."""

    def test_simple_name(self):
        """Test simple valid names pass through."""
        nm = NameManager()
        assert nm.get_symbol("input") == "input"
        assert nm.get_symbol("output") == "output"
        assert nm.get_symbol("my_tensor") == "my_tensor"

    def test_double_colon(self):
        """Test ONNX scope operator (::) is replaced."""
        nm = NameManager()
        # onnx::Reshape_0 -> onnx__Reshape_0
        result = nm.get_symbol("onnx::Reshape_0")
        assert "::" not in result
        assert "onnx_Reshape_0" in result or "onnx__Reshape_0" in result

    def test_dot_replacement(self):
        """Test dots are replaced with underscores."""
        nm = NameManager()
        result = nm.get_symbol("weight.data")
        assert "." not in result
        assert "weight_data" in result

    def test_leading_digit(self):
        """Test names starting with digits get prefixed."""
        nm = NameManager()
        result = nm.get_symbol("68_shape")
        assert not result[0].isdigit()
        assert result in ["n68_shape", "_68_shape"]

    def test_slash_replacement(self):
        """Test forward slashes are replaced."""
        nm = NameManager()
        result = nm.get_symbol("model/layer/weight")
        assert "/" not in result
        assert "model_layer_weight" in result

    def test_complex_onnx_name(self):
        """Test complex ONNX names with multiple invalid chars."""
        nm = NameManager()
        # Example: onnx::Conv_234.weight
        result = nm.get_symbol("onnx::Conv_234.weight")
        # Should not contain invalid chars
        assert "::" not in result
        assert "." not in result
        # Should be valid C identifier
        assert result.replace("_", "").isalnum() or result.isidentifier()

    def test_empty_name(self):
        """Test empty names become 'unnamed'."""
        nm = NameManager()
        result = nm.get_symbol("")
        assert result == "unnamed"

    def test_special_chars_only(self):
        """Test names with only special characters."""
        nm = NameManager()
        result = nm.get_symbol("...")
        assert result == "unnamed"

    def test_uniqueness(self):
        """Test that colliding sanitized names get unique suffixes."""
        nm = NameManager()
        # Two different ONNX names that sanitize to the same C symbol
        name1 = nm.get_symbol("tensor-1")
        name2 = nm.get_symbol("tensor.1")
        # Both should sanitize to "tensor_1", so second gets suffix
        assert "tensor_1" in name1
        assert name1 != name2

    def test_c_keywords(self):
        """Test that C keywords get suffix."""
        nm = NameManager()
        for keyword in ["if", "while", "for", "return", "int"]:
            result = nm.get_symbol(keyword)
            assert result == f"{keyword}_"

    def test_with_prefix(self):
        """Test names with prefix."""
        nm = NameManager()
        result = nm.get_symbol("input", prefix="var_")
        assert result.startswith("var_")

    def test_multiple_invalid_chars_collapse(self):
        """Test that multiple consecutive invalid chars collapse to single underscore."""
        nm = NameManager()
        result = nm.get_symbol("tensor...name")
        # Should not have multiple underscores in a row
        assert "__" not in result

    def test_hyphen_replacement(self):
        """Test hyphens are replaced."""
        nm = NameManager()
        result = nm.get_symbol("layer-1-output")
        assert "-" not in result
        assert "layer_1_output" in result

    def test_at_sign_replacement(self):
        """Test @ signs are replaced."""
        nm = NameManager()
        result = nm.get_symbol("attention@mask")
        assert "@" not in result

    def test_space_replacement(self):
        """Test spaces are replaced."""
        nm = NameManager()
        result = nm.get_symbol("my tensor")
        assert " " not in result
        assert "my_tensor" in result


class TestX86BackendSymbolAssignment:
    """Test X86Backend symbol assignment integration."""

    def test_assign_symbols_with_invalid_names(self):
        """Test that backend properly assigns C symbols for invalid names."""
        from nnc_py.ir.context import CompileContext
        from nnc_py.ir.graph import Graph
        from nnc_py.ir.tensor import TensorType, TensorShape
        from nnc_py.ir.types import DataType
        from nnc_py.codegen.x86_backend import X86Backend

        backend = X86Backend()
        graph = Graph(name="test")

        # Add tensors with problematic names
        problematic_names = [
            "onnx::Reshape_0.data",
            "68_shape",
            "model/layer/weight",
            "output::0",
        ]

        for name in problematic_names:
            graph.add_tensor(TensorType(
                dtype=DataType.FLOAT32,
                shape=TensorShape(dims=[1, 2, 3]),
                name=name
            ))

        ctx = CompileContext(graph, "x86", 0)
        backend._assign_symbols(ctx)

        # Verify all assigned symbols are valid C identifiers
        for orig_name in problematic_names:
            symbol = ctx.tensor_symbols.get(orig_name)
            assert symbol is not None
            # Check it doesn't start with digit
            assert not symbol[0].isdigit()
            # Check no invalid chars
            assert all(c.isalnum() or c == "_" for c in symbol)
            # Check not a C keyword
            assert symbol not in ["if", "while", "for", "return", "int"]
