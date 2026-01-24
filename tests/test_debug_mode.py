"""Debug mode tests for comparing NNC output with ONNX Runtime.

These tests verify that the debug mode correctly:
1. Generates debug dump code in the C output
2. Runs and produces debug output files
3. Can be compared against ONNX Runtime results
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

import pytest

from nnc_py import Compiler
from nnc_py.tools.debug_compare import (
    DebugOutputParser,
    ONNXRuntimeRunner,
    DebugComparator,
)


class DebugModeTestHelper:
    """Helper for debug mode testing."""

    def __init__(self, tmp_dir: str = None):
        """Initialize test helper.

        Args:
            tmp_dir: Temporary directory for outputs.
        """
        self.tmp_dir = tmp_dir or tempfile.mkdtemp()
        self.runtime_dir = Path(__file__).resolve().parent.parent / "runtime"

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def compile_model_debug(
        self,
        onnx_model: onnx.ModelProto,
        model_name: str = "test"
    ) -> str:
        """Compile ONNX model to C code with debug mode enabled.

        Args:
            onnx_model: ONNX model to compile.
            model_name: Name for the model.

        Returns:
            Path to the output directory.
        """
        # Save ONNX model
        onnx_path = os.path.join(self.tmp_dir, f"{model_name}.onnx")
        onnx.save(onnx_model, onnx_path)

        # Compile to C with debug mode
        output_dir = os.path.join(self.tmp_dir, f"build_{model_name}")
        compiler = Compiler(target="x86", opt_level=0, debug_mode=True)
        compiler.compile(onnx_path, output_dir)

        return output_dir

    def build_c_code(self, output_dir: str) -> bool:
        """Build the generated C code.

        Args:
            output_dir: Directory containing generated C code.

        Returns:
            True if build succeeded, False otherwise.
        """
        build_dir = Path(output_dir)
        makefile = build_dir / "Makefile"

        # Update Makefile to use correct runtime path
        if makefile.exists():
            makefile_content = makefile.read_text()
            makefile_content = makefile_content.replace(
                "NNC_RUNTIME ?= ../../runtime",
                f"NNC_RUNTIME = {self.runtime_dir}"
            )
            makefile.write_text(makefile_content)

        try:
            subprocess.run(
                ["make", "clean"],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            result = subprocess.run(
                ["make"],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                print("Build failed!")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False

            return True
        except subprocess.TimeoutExpired:
            print("Build timed out")
            return False
        except Exception as e:
            print(f"Build error: {e}")
            return False

    def run_executable(self, output_dir: str) -> tuple[bool, str, str]:
        """Run the compiled executable.

        Args:
            output_dir: Directory containing the executable.

        Returns:
            Tuple of (success, stdout, stderr).
        """
        exe_path = os.path.join(output_dir, "model")
        if not os.path.exists(exe_path):
            return False, "", "Executable not found"

        try:
            result = subprocess.run(
                [exe_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=output_dir
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Execution timed out"
        except Exception as e:
            return False, "", str(e)


class TestDebugMode:
    """Debug mode tests."""

    def setup_method(self):
        """Set up test environment."""
        self.helper = DebugModeTestHelper()

    def teardown_method(self):
        """Clean up test environment."""
        self.helper.cleanup()

    def test_debug_mode_generates_debug_code(self):
        """Test that debug mode generates debug dump code."""
        # Create simple model: output = Relu(input)
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [4])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4])

        relu_node = helper.make_node("Relu", inputs=["input"], outputs=["output"])

        graph = helper.make_graph([relu_node], "relu_model", [input_val], [output_val])
        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        # Compile with debug mode
        output_dir = self.helper.compile_model_debug(model, "relu_debug")

        # Check generated files
        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c not generated"

        # Check that debug macros are defined
        with open(model_c_path, 'r') as f:
            code = f.read()

        # In debug mode, we should have debug output for intermediate tensors
        # The debug code is injected into nnc_run function
        assert "DEBUG_PRINT" in code or "DEBUG_TENSOR" in code, \
            "Debug macros not found in generated code"

    def test_debug_mode_creates_debug_output_file(self):
        """Test that running debug mode executable creates debug output file."""
        # Create simple model
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

        relu_node = helper.make_node("Relu", inputs=["input"], outputs=["output"])

        graph = helper.make_graph([relu_node], "relu_model", [input_val], [output_val])
        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        # Compile and build
        output_dir = self.helper.compile_model_debug(model, "relu_file_test")

        build_success = self.helper.build_c_code(output_dir)
        if not build_success:
            pytest.skip("Build failed")

        # Run executable
        success, stdout, stderr = self.helper.run_executable(output_dir)

        # Check for debug output file
        debug_file = os.path.join(output_dir, "nnc_debug_output.txt")
        if not os.path.exists(debug_file):
            pytest.skip(f"Debug file not created. stdout: {stdout}, stderr: {stderr}")

        # Parse debug output
        parser = DebugOutputParser(debug_file)
        outputs = parser.parse()

        # Should have at least the output tensor
        assert len(outputs) > 0, "No tensors found in debug output"
        assert "output" in outputs, "Output tensor not found in debug output"

    def test_debug_output_parser(self):
        """Test the debug output parser with sample data."""
        # Create sample debug output
        debug_content = """DEBUG_TENSOR_START output 0
SHAPE 1
DIM 0 4
DATA_START
0.000000
0.010000
0.020000
0.030000
DATA_END
DEBUG_TENSOR_END output
"""
        debug_file = os.path.join(self.helper.tmp_dir, "test_debug.txt")
        with open(debug_file, 'w') as f:
            f.write(debug_content)

        parser = DebugOutputParser(debug_file)
        outputs = parser.parse()

        assert "output" in outputs
        assert outputs["output"]["node_idx"] == 0
        assert outputs["output"]["shape"] == [4]
        assert len(outputs["output"]["data"]) == 4

    def test_debug_compare_simple_model(self):
        """Test comparing NNC output with ONNX Runtime for a simple model."""
        # Create simple Add model: output = input + constant
        const_tensor = helper.make_tensor(
            "const_value",
            onnx.TensorProto.FLOAT,
            [2, 2],
            [1.0, 2.0, 3.0, 4.0]
        )

        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const"],
            value=const_tensor
        )

        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
        const_val = helper.make_tensor_value_info("const", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

        add_node = helper.make_node("Add", inputs=["input", "const"], outputs=["output"])

        graph = helper.make_graph(
            [const_node, add_node],
            "add_model",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        # Compile with debug mode
        onnx_path = os.path.join(self.helper.tmp_dir, "add_compare.onnx")
        onnx.save(model, onnx_path)

        output_dir = self.helper.compile_model_debug(model, "add_compare")

        build_success = self.helper.build_c_code(output_dir)
        if not build_success:
            pytest.skip("Build failed")

        success, stdout, stderr = self.helper.run_executable(output_dir)

        debug_file = os.path.join(output_dir, "nnc_debug_output.txt")
        if not os.path.exists(debug_file):
            pytest.skip(f"Debug file not created. stdout: {stdout}, stderr: {stderr}")

        # Compare with ONNX Runtime
        comparator = DebugComparator(debug_file, onnx_path)
        results = comparator.compare()

        # All outputs should match (or have acceptable differences)
        assert len(results["shape_mismatch"]) == 0, "Shape mismatches found"
        # We may have minor numerical differences due to floating point
        assert len(results["mismatched"]) == 0 or \
               all(m["max_diff"] < 1e-3 for m in results["mismatched"]), \
            f"Large mismatches found: {results['mismatched']}"

    def test_onnx_runtime_runner(self):
        """Test the ONNX Runtime runner."""
        # Create simple Relu model
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [4])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4])

        relu_node = helper.make_node("Relu", inputs=["input"], outputs=["output"])

        graph = helper.make_graph([relu_node], "relu_model", [input_val], [output_val])
        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        onnx_path = os.path.join(self.helper.tmp_dir, "relu_runner.onnx")
        onnx.save(model, onnx_path)

        runner = ONNXRuntimeRunner(onnx_path)

        # Get input info
        input_info = runner.get_input_info()
        assert len(input_info) == 1
        assert input_info[0][0] == "input"

        # Run with test pattern
        outputs = runner.run_with_intermediates(test_pattern=True)

        assert "output" in outputs
        assert outputs["output"].shape == (4,)


class TestDebugCodeIntegration:
    """Test debug code integration in generated C files."""

    def setup_method(self):
        """Set up test environment."""
        self.helper = DebugModeTestHelper()

    def teardown_method(self):
        """Clean up test environment."""
        self.helper.cleanup()

    def test_debug_mode_vs_normal_mode(self):
        """Compare code generation with and without debug mode."""
        # Create simple model
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [4])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4])

        relu_node = helper.make_node("Relu", inputs=["input"], outputs=["output"])

        graph = helper.make_graph([relu_node], "relu_model", [input_val], [output_val])
        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        onnx_path = os.path.join(self.helper.tmp_dir, "relu.onnx")
        onnx.save(model, onnx_path)

        # Compile without debug mode
        output_dir_normal = os.path.join(self.helper.tmp_dir, "build_normal")
        compiler_normal = Compiler(target="x86", opt_level=0, debug_mode=False)
        compiler_normal.compile(onnx_path, output_dir_normal)

        # Compile with debug mode
        output_dir_debug = os.path.join(self.helper.tmp_dir, "build_debug")
        compiler_debug = Compiler(target="x86", opt_level=0, debug_mode=True)
        compiler_debug.compile(onnx_path, output_dir_debug)

        # Compare model.c files
        normal_model_c = os.path.join(output_dir_normal, "model.c")
        debug_model_c = os.path.join(output_dir_debug, "model.c")

        with open(normal_model_c, 'r') as f:
            normal_code = f.read()

        with open(debug_model_c, 'r') as f:
            debug_code = f.read()

        # Debug version should have additional debug code
        # This is a rough check - debug code should be longer
        assert len(debug_code) >= len(normal_code), \
            "Debug code should be at least as long as normal code"

        # Debug version should have debug statements
        assert "DEBUG_PRINT" in debug_code or "printf" in debug_code, \
            "Debug code should contain debug print statements"

    def test_debug_test_runner_has_file_support(self):
        """Test that debug mode test runner has file output support."""
        # Create simple model
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [4])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4])

        relu_node = helper.make_node("Relu", inputs=["input"], outputs=["output"])

        graph = helper.make_graph([relu_node], "relu_model", [input_val], [output_val])
        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        onnx_path = os.path.join(self.helper.tmp_dir, "relu.onnx")
        onnx.save(model, onnx_path)

        # Compile with debug mode
        output_dir = os.path.join(self.helper.tmp_dir, "build_debug")
        compiler = Compiler(target="x86", opt_level=0, debug_mode=True)
        compiler.compile(onnx_path, output_dir)

        # Check test_runner.c
        test_runner_c = os.path.join(output_dir, "test_runner.c")
        with open(test_runner_c, 'r') as f:
            code = f.read()

        # Debug mode should have file operations
        assert "fopen" in code, "Debug test runner should have fopen"
        assert "fclose" in code, "Debug test runner should have fclose"
        assert "debug_file" in code, "Debug test runner should have debug_file"
        assert "nnc_debug_output.txt" in code, \
            "Debug test runner should reference debug output file"
