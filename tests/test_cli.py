"""Tests for CLI functionality."""

from pathlib import Path

import onnx
import pytest
from click.testing import CliRunner
from onnx import helper

from nnc_py.cli import main


@pytest.fixture
def runner():
    """Create a Click CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def simple_onnx_model(tmp_path):
    """Create a simple ONNX model for testing."""
    # Create a simple linear model: y = x + 1
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["X", "const"], "Y"),
        ],
        "simple_model",
        [
            helper.make_tensor_value_info("X", helper.TensorProto.FLOAT, [2, 2]),
        ],
        [
            helper.make_tensor_value_info("Y", helper.TensorProto.FLOAT, [2, 2]),
        ],
        [
            helper.make_tensor("const", helper.TensorProto.FLOAT, [2, 2], [1.0, 1.0, 1.0, 1.0]),
        ],
    )
    model = helper.make_model(graph)
    model_path = tmp_path / "simple_model.onnx"
    onnx.save(model, model_path)
    return model_path


def test_cli_main_command_exists(runner):
    """Test that the main CLI group is accessible."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "NNC - Neural Network Compiler" in result.output


def test_targets_command(runner):
    """Test 'nnc targets' command lists available targets."""
    result = runner.invoke(main, ["targets"])
    assert result.exit_code == 0
    assert "Available Targets" in result.output
    assert "x86" in result.output
    assert "npu" in result.output
    assert "Generate code for x86 simulation" in result.output
