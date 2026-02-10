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


def test_info_command(runner, simple_onnx_model):
    """Test 'nnc info' command displays model information."""
    result = runner.invoke(main, ["info", str(simple_onnx_model)])
    assert result.exit_code == 0
    assert "Model Information" in result.output
    assert "simple_model.onnx" in result.output
    assert "Nodes:" in result.output
    assert "Inputs:" in result.output
    assert "Outputs:" in result.output
    assert "Operators:" in result.output
    assert "Add: 1" in result.output


def test_info_command_invalid_file(runner):
    """Test 'nnc info' with non-existent file."""
    result = runner.invoke(main, ["info", "nonexistent.onnx"])
    assert result.exit_code != 0
    assert "Error loading model" in result.output


def test_compile_command_basic(runner, simple_onnx_model, tmp_path):
    """Test 'nnc compile' command with basic options."""
    output_dir = tmp_path / "output"
    result = runner.invoke(main, [
        "compile",
        str(simple_onnx_model),
        "-o", str(output_dir),
        "-t", "x86",
    ])
    assert result.exit_code == 0
    assert "Compiling" in result.output
    assert "Target: x86" in result.output
    # Verify output directory was created
    assert output_dir.exists()
    # Check for generated C files
    c_files = list(output_dir.glob("*.c"))
    assert len(c_files) > 0, "Expected at least one .c file to be generated"


def test_compile_command_with_opt_level(runner, simple_onnx_model, tmp_path):
    """Test 'nnc compile' with optimization level."""
    output_dir = tmp_path / "output_opt"
    result = runner.invoke(main, [
        "compile",
        str(simple_onnx_model),
        "-o", str(output_dir),
        "-O", "2",
    ])
    assert result.exit_code == 0
    assert "Optimization: O2" in result.output


def test_compile_command_npu_target(runner, simple_onnx_model, tmp_path):
    """Test 'nnc compile' with NPU target."""
    output_dir = tmp_path / "output_npu"
    result = runner.invoke(main, [
        "compile",
        str(simple_onnx_model),
        "-o", str(output_dir),
        "-t", "npu",
    ])
    assert result.exit_code == 0
    assert "Target: npu" in result.output


def test_compile_command_invalid_target(runner, simple_onnx_model, tmp_path):
    """Test 'nnc compile' with invalid target (should be caught by Click)."""
    output_dir = tmp_path / "output_invalid"
    result = runner.invoke(main, [
        "compile",
        str(simple_onnx_model),
        "-o", str(output_dir),
        "-t", "invalid",
    ])
    assert result.exit_code != 0
    assert "Invalid value for '-t'" in result.output
