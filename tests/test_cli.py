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
    assert "not yet implemented" in result.output.lower()


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
    assert result.exit_code != 0
    assert "Target: npu" in result.output
    assert "not implemented" in result.output.lower()


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


def test_compile_command_with_max_memory(runner, simple_onnx_model, tmp_path):
    """Test 'nnc compile' with max memory limit."""
    output_dir = tmp_path / "output_memory"
    result = runner.invoke(main, [
        "compile",
        str(simple_onnx_model),
        "-o", str(output_dir),
        "--max-memory", "256K",
    ])
    assert result.exit_code == 0
    assert "Max Memory: 256K" in result.output


def test_compile_command_with_memory_strategy(runner, simple_onnx_model, tmp_path):
    """Test 'nnc compile' with memory strategy."""
    output_dir = tmp_path / "output_strategy"
    result = runner.invoke(main, [
        "compile",
        str(simple_onnx_model),
        "-o", str(output_dir),
        "--memory-strategy", "basic",
    ])
    assert result.exit_code == 0
    assert "Memory Strategy: basic" in result.output


def test_compile_command_with_entry_name(runner, simple_onnx_model, tmp_path):
    """Test 'nnc compile' with custom entry name."""
    output_dir = tmp_path / "output_entry"
    result = runner.invoke(main, [
        "compile",
        str(simple_onnx_model),
        "-o", str(output_dir),
        "--entry-name", "my_infer",
    ])
    assert result.exit_code == 0
    model_header = (output_dir / "model.h").read_text()
    model_source = (output_dir / "model.c").read_text()
    assert "void my_infer(void);" in model_header
    assert "void my_infer(void) {" in model_source


def test_compile_command_constant_folding_enabled(runner, simple_onnx_model, tmp_path):
    """Test 'nnc compile' with constant folding enabled (default)."""
    output_dir = tmp_path / "output_cf_on"
    result = runner.invoke(main, [
        "compile",
        str(simple_onnx_model),
        "-o", str(output_dir),
        "--enable-constant-folding",
    ])
    assert result.exit_code == 0
    assert "Constant Folding: enabled" in result.output


def test_compile_command_constant_folding_disabled(runner, simple_onnx_model, tmp_path):
    """Test 'nnc compile' with constant folding disabled."""
    output_dir = tmp_path / "output_cf_off"
    result = runner.invoke(main, [
        "compile",
        str(simple_onnx_model),
        "-o", str(output_dir),
        "--disable-constant-folding",
    ])
    assert result.exit_code == 0
    assert "Constant Folding: disabled" in result.output


def test_compile_command_debug_mode(runner, simple_onnx_model, tmp_path):
    """Test 'nnc compile' with debug mode enabled."""
    output_dir = tmp_path / "output_debug"
    result = runner.invoke(main, [
        "compile",
        str(simple_onnx_model),
        "-o", str(output_dir),
        "--debug",
    ])
    assert result.exit_code == 0
    assert "Debug Mode: enabled" in result.output


def test_compile_command_invalid_model(runner, tmp_path):
    """Test 'nnc compile' with non-existent model file."""
    output_dir = tmp_path / "output"
    result = runner.invoke(main, [
        "compile",
        "nonexistent.onnx",
        "-o", str(output_dir),
    ])
    assert result.exit_code != 0


def test_compile_verbose_mode(runner, simple_onnx_model, tmp_path):
    """Test 'nnc compile' with verbose mode (on error)."""
    # Create a model that will fail compilation
    # (empty graph without proper inputs/outputs)
    graph = helper.make_graph([], "empty", [], [])
    model = helper.make_model(graph)
    bad_model_path = tmp_path / "bad_model.onnx"
    onnx.save(model, bad_model_path)

    output_dir = tmp_path / "output_bad"
    result = runner.invoke(main, [
        "compile",
        str(bad_model_path),
        "-o", str(output_dir),
        "-v",
    ])
    assert result.exit_code != 0


@pytest.fixture
def debug_output_file(tmp_path):
    """Create a mock debug output file for testing compare command."""
    debug_file = tmp_path / "debug_output.txt"
    debug_file.write_text("""# NNC Debug Output
Tensor: input
Shape: [2, 2]
Data: [[1.0, 2.0], [3.0, 4.0]]

Tensor: output
Shape: [2, 2]
Data: [[2.0, 3.0], [4.0, 5.0]]
""")
    return debug_file


def test_debug_compare_command(runner, debug_output_file, simple_onnx_model):
    """Test 'nnc debug compare' command."""
    result = runner.invoke(main, [
        "debug", "compare",
        str(debug_output_file),
        str(simple_onnx_model),
    ])
    # The command will fail due to mismatched data, but CLI should handle it
    assert "Comparing:" in result.output
    assert "Tolerances:" in result.output


def test_debug_compare_with_custom_tolerances(runner, debug_output_file, simple_onnx_model):
    """Test 'nnc debug compare' with custom tolerances."""
    result = runner.invoke(main, [
        "debug", "compare",
        str(debug_output_file),
        str(simple_onnx_model),
        "--rtol", "1e-3",
        "--atol", "1e-4",
    ])
    assert "rtol=0.001" in result.output or "rtol=1e-3" in result.output
    assert "atol=0.0001" in result.output or "atol=1e-4" in result.output


def test_debug_compare_with_json_output(runner, debug_output_file, simple_onnx_model, tmp_path):
    """Test 'nnc debug compare' with JSON output."""
    json_output = tmp_path / "results.json"
    result = runner.invoke(main, [
        "debug", "compare",
        str(debug_output_file),
        str(simple_onnx_model),
        "--output", str(json_output),
    ])
    # Command may fail comparison but JSON should be written
    # (depending on implementation)
    assert result.exit_code in [0, 1]  # May pass or fail depending on data
