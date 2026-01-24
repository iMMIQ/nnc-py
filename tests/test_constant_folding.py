"""Tests for onnxsim integration in ONNX frontend."""

import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

import pytest

from nnc_py.frontend.onnx_loader import ONNXFrontend, HAS_ONNXSIM


# Skip tests that require onnxsim if it's not available
pytestmark = pytest.mark.skipif(
    not HAS_ONNXSIM,
    reason="onnxsim is not installed"
)


def _create_simple_add_model() -> onnx.ModelProto:
    """Create a simple ONNX model with constant Add operation."""
    # Create constants
    const_a = helper.make_tensor('const_a', onnx.TensorProto.FLOAT, [3], [1.0, 2.0, 3.0])
    const_b = helper.make_tensor('const_b', onnx.TensorProto.FLOAT, [3], [4.0, 5.0, 6.0])

    # Create Add node
    add_node = helper.make_node('Add', inputs=['const_a', 'const_b'], outputs=['output'])

    # Create graph
    graph = helper.make_graph(
        [add_node],
        'simple_add',
        [],  # no inputs
        [helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [3])],
        [const_a, const_b]  # initializers
    )

    # Create model
    model = helper.make_model(graph)
    return model


def _create_relu_chain_model() -> onnx.ModelProto:
    """Create an ONNX model with a chain of operations on constants."""
    # Create input constant
    const_input = helper.make_tensor('input', onnx.TensorProto.FLOAT, [3], [1.0, 2.0, 3.0])

    # Create nodes: input -> Add (input + input) -> ReLU -> output
    add_node = helper.make_node('Add', inputs=['input', 'input'], outputs=['mid1'])
    relu_node = helper.make_node('Relu', inputs=['mid1'], outputs=['output'])

    # Create graph
    graph = helper.make_graph(
        [add_node, relu_node],
        'relu_chain',
        [],
        [helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [3])],
        [const_input]
    )

    model = helper.make_model(graph)
    return model


def _create_mixed_model() -> onnx.ModelProto:
    """Create a model with both constant and variable inputs."""
    # Only one constant
    const_a = helper.make_tensor('const_a', onnx.TensorProto.FLOAT, [3], [1.0, 2.0, 3.0])

    # Add node with one constant and one variable input
    add_node = helper.make_node('Add', inputs=['const_a', 'input_var'], outputs=['output'])

    # Create graph
    graph = helper.make_graph(
        [add_node],
        'mixed_add',
        [helper.make_tensor_value_info('input_var', onnx.TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [3])],
        [const_a]
    )

    model = helper.make_model(graph)
    return model


class TestOnnxsimIntegration:
    """Test suite for onnxsim integration."""

    def test_onnxsim_simplifies_constant_add(self):
        """Test that onnxsim folds constant Add operations."""
        # Create model with constant operations
        model = _create_simple_add_model()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            temp_path = f.name
            try:
                onnx.save(model, temp_path)

                # Load with simplification enabled
                frontend = ONNXFrontend(enable_simplify=True)
                graph = frontend.load(temp_path)

                # With onnxsim, the constant Add should be folded
                # The output should be in constants
                assert 'output' in graph.constants
                result = graph.constants['output']
                expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
                np.testing.assert_array_equal(result, expected)
            finally:
                Path(temp_path).unlink(missing_ok=True)

    def test_onnxsim_simplifies_relu_chain(self):
        """Test that onnxsim folds a chain of constant operations."""
        model = _create_relu_chain_model()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            temp_path = f.name
            try:
                onnx.save(model, temp_path)

                frontend = ONNXFrontend(enable_simplify=True)
                graph = frontend.load(temp_path)

                # The entire chain should be folded
                assert 'output' in graph.constants
                result = graph.constants['output']
                # relu(input + input) = relu([2, 4, 6]) = [2, 4, 6]
                expected = np.array([2.0, 4.0, 6.0], dtype=np.float32)
                np.testing.assert_array_equal(result, expected)
            finally:
                Path(temp_path).unlink(missing_ok=True)

    def test_onnxsim_disabled_preserves_graph(self):
        """Test that disabling onnxsim preserves the original graph structure."""
        model = _create_simple_add_model()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            temp_path = f.name
            try:
                onnx.save(model, temp_path)

                # Load with simplification disabled
                frontend = ONNXFrontend(enable_simplify=False)
                graph = frontend.load(temp_path)

                # The Add node should still be present
                assert len(graph.nodes) > 0
                # Output should not be in constants (not folded)
                assert 'output' not in graph.constants
            finally:
                Path(temp_path).unlink(missing_ok=True)

    def test_mixed_constant_variable_inputs(self):
        """Test that operations with mixed inputs are handled correctly."""
        model = _create_mixed_model()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            temp_path = f.name
            try:
                onnx.save(model, temp_path)

                frontend = ONNXFrontend(enable_simplify=True)
                graph = frontend.load(temp_path)

                # The Add node should still exist (one input is variable)
                assert len(graph.nodes) > 0
                # Constant should be preserved
                assert 'const_a' in graph.constants
                # Output should not be a constant (has variable input)
                assert 'output' not in graph.constants
            finally:
                Path(temp_path).unlink(missing_ok=True)

    def test_frontend_init_options(self):
        """Test ONNXFrontend initialization options."""
        # With simplification enabled (default)
        frontend_enabled = ONNXFrontend(enable_simplify=True)
        # Should be True since onnxsim is available (test is skipped otherwise)
        assert frontend_enabled.enable_simplify is True

        # With simplification disabled
        frontend_disabled = ONNXFrontend(enable_simplify=False)
        assert frontend_disabled.enable_simplify is False

        # Default behavior - simplification enabled by default
        frontend_default = ONNXFrontend()
        assert frontend_default.enable_simplify is True
