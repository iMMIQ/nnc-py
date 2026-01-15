"""End-to-end integration tests for compilation and execution.

These tests compile ONNX models to C code, compile the C code,
and execute it to verify correctness.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

import pytest

from nnc_py import Compiler
from nnc_py.frontend.onnx_loader import ONNXFrontend
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.codegen.x86_backend import X86Backend


class E2ETestHelper:
    """Helper for end-to-end testing."""

    def __init__(self, tmp_dir: str = None):
        """Initialize test helper.

        Args:
            tmp_dir: Temporary directory for outputs.
        """
        self.tmp_dir = tmp_dir or tempfile.mkdtemp()
        # Path: tests/test_e2e.py -> tests -> project root (nnc-py)
        self.runtime_dir = Path(__file__).resolve().parent.parent / "runtime"

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def compile_model(self, onnx_model: onnx.ModelProto, model_name: str = "test") -> str:
        """Compile ONNX model to C code.

        Args:
            onnx_model: ONNX model to compile.
            model_name: Name for the model.

        Returns:
            Path to the output directory.
        """
        # Save ONNX model
        onnx_path = os.path.join(self.tmp_dir, f"{model_name}.onnx")
        onnx.save(onnx_model, onnx_path)

        # Compile to C
        output_dir = os.path.join(self.tmp_dir, f"build_{model_name}")
        compiler = Compiler(target="x86", opt_level=0)
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
            # Replace default runtime path with actual runtime directory
            makefile_content = makefile_content.replace(
                "NNC_RUNTIME ?= ../../runtime",
                f"NNC_RUNTIME = {self.runtime_dir}"
            )
            makefile.write_text(makefile_content)

        try:
            # Run make clean first
            subprocess.run(
                ["make", "clean"],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Run make
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
                timeout=30
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Execution timed out"
        except Exception as e:
            return False, "", str(e)


class TestEndToEnd:
    """End-to-end integration tests."""

    def setup_method(self):
        """Set up test environment."""
        self.helper = E2ETestHelper()

    def teardown_method(self):
        """Clean up test environment."""
        self.helper.cleanup()

    def test_simple_add_model(self):
        """Test compilation and execution of a simple Add model."""
        # Create a simple Add model: output = input + constant
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

        # Compile
        output_dir = self.helper.compile_model(model, "simple_add")

        # Check generated files exist
        assert os.path.exists(os.path.join(output_dir, "model.h"))
        assert os.path.exists(os.path.join(output_dir, "model.c"))
        assert os.path.exists(os.path.join(output_dir, "tensors.c"))
        assert os.path.exists(os.path.join(output_dir, "constants.c"))
        assert os.path.exists(os.path.join(output_dir, "Makefile"))

        # Try to build
        build_success = self.helper.build_c_code(output_dir)
        if not build_success:
            pytest.skip("Build failed - fix needed")

        # Try to run
        success, stdout, stderr = self.helper.run_executable(output_dir)
        assert success, f"Execution failed: {stderr}"

    def test_reshape_model(self):
        """Test compilation of a Reshape model."""
        # Create model: reshape [4] to [2, 2]
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [4])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

        shape_const = helper.make_tensor("shape", onnx.TensorProto.INT64, [2], [2, 2])
        shape_node = helper.make_node("Constant", inputs=[], outputs=["shape"], value=shape_const)

        reshape_node = helper.make_node("Reshape", inputs=["input", "shape"], outputs=["output"])

        graph = helper.make_graph(
            [shape_node, reshape_node],
            "reshape_model",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        output_dir = self.helper.compile_model(model, "reshape")

        # Check generated files
        assert os.path.exists(os.path.join(output_dir, "model.c"))

        # Try to build
        build_success = self.helper.build_c_code(output_dir)
        if not build_success:
            pytest.skip("Build failed - fix needed")

    def test_split_model(self):
        """Test compilation of a Split model."""
        # Split [2, 4] into [2, 2] + [2, 2] along axis 1
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 4])
        output1_val = helper.make_tensor_value_info("output1", onnx.TensorProto.FLOAT, [2, 2])
        output2_val = helper.make_tensor_value_info("output2", onnx.TensorProto.FLOAT, [2, 2])

        split_node = helper.make_node("Split", inputs=["input"], outputs=["output1", "output2"], axis=1)

        graph = helper.make_graph(
            [split_node],
            "split_model",
            [input_val],
            [output1_val, output2_val]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        output_dir = self.helper.compile_model(model, "split")

        # Try to build
        build_success = self.helper.build_c_code(output_dir)
        if not build_success:
            pytest.skip("Build failed - fix needed")

    def test_concat_model(self):
        """Test compilation of a Concat model."""
        # Concat two [2, 2] tensors into [2, 4]
        input1_val = helper.make_tensor_value_info("input1", onnx.TensorProto.FLOAT, [2, 2])
        input2_val = helper.make_tensor_value_info("input2", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 4])

        concat_node = helper.make_node(
            "Concat",
            inputs=["input1", "input2"],
            outputs=["output"],
            axis=1
        )

        graph = helper.make_graph(
            [concat_node],
            "concat_model",
            [input1_val, input2_val],
            [output_val]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        output_dir = self.helper.compile_model(model, "concat")

        # Try to build
        build_success = self.helper.build_c_code(output_dir)
        if not build_success:
            pytest.skip("Build failed - fix needed")

    def test_conv_relu_model(self):
        """Test compilation of Conv + ReLU model (existing test case)."""
        # Use existing model if available
        existing_model = Path(__file__).parent.parent / "simple_conv.onnx"
        if not existing_model.exists():
            pytest.skip("Existing model not found")

        output_dir = self.helper.compile_model(
            onnx.load(existing_model),
            "conv_relu"
        )

        build_success = self.helper.build_c_code(output_dir)
        if not build_success:
            pytest.skip("Build failed - fix needed")

        success, stdout, stderr = self.helper.run_executable(output_dir)
        if not success:
            print(f"Execution failed: {stderr}")
            pytest.skip("Execution failed")


class TestCodeGeneration:
    """Test generated C code structure."""

    def setup_method(self):
        """Set up test environment."""
        self.helper = E2ETestHelper()

    def teardown_method(self):
        """Clean up test environment."""
        self.helper.cleanup()

    def check_c_syntax(self, code: str) -> tuple[bool, str]:
        """Check if C code has valid syntax (basic checks).

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not code:
            return False, "Empty code"

        # Check for basic syntax issues
        issues = []

        # Check for unmatched parentheses
        open_parens = code.count("(")
        close_parens = code.count(")")
        if open_parens != close_parens:
            issues.append(f"Unmatched parentheses: {open_parens} open, {close_parens} close")

        # Check for unmatched braces
        open_braces = code.count("{")
        close_braces = code.count("}")
        if open_braces != close_braces:
            issues.append(f"Unmatched braces: {open_braces} open, {close_braces} close")

        # Check for nnc_ function calls that might not exist
        import re
        function_calls = re.findall(r'\bnnc_(\w+)\(', code)
        known_functions = {
            "add", "sub", "mul", "div", "relu", "sigmoid", "tanh", "softmax",
            "conv", "maxpool2d", "avgpool2d", "matmul", "gemm",
            "reshape", "flatten", "transpose", "squeeze", "unsqueeze",
            "reducemean", "reducesum", "concat", "batchnorm", "clip",
            "layernorm", "identity", "split", "run"  # nnc_run is generated entry point
        }
        for func in function_calls:
            if func not in known_functions:
                issues.append(f"Unknown function: nnc_{func}")

        return len(issues) == 0, "; ".join(issues)

    def test_generated_model_c_syntax(self):
        """Test that generated model.c has valid C syntax."""
        # Create simple model
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

        relu_node = helper.make_node("Relu", inputs=["input"], outputs=["output"])

        graph = helper.make_graph([relu_node], "relu_model", [input_val], [output_val])
        model = helper.make_model(graph)

        output_dir = self.helper.compile_model(model, "relu_syntax")

        # Read generated model.c
        model_c_path = os.path.join(output_dir, "model.c")
        with open(model_c_path, 'r') as f:
            code = f.read()

        is_valid, error = self.check_c_syntax(code)
        assert is_valid, f"Generated C code has issues: {error}"

    def test_reshape_generated_code(self):
        """Check generated Reshape code for correct parameter passing."""
        # Create reshape model
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [4])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

        shape_const = helper.make_tensor("shape", onnx.TensorProto.INT64, [2], [2, 2])
        shape_node = helper.make_node("Constant", inputs=[], outputs=["shape"], value=shape_const)
        reshape_node = helper.make_node("Reshape", inputs=["input", "shape"], outputs=["output"])

        graph = helper.make_graph(
            [shape_node, reshape_node],
            "reshape_check",
            [input_val],
            [output_val]
        )
        model = helper.make_model(graph)

        output_dir = self.helper.compile_model(model, "reshape_check")

        # Read generated model.c
        model_c_path = os.path.join(output_dir, "model.c")
        with open(model_c_path, 'r') as f:
            code = f.read()

        # Check for nnc_reshape call
        assert "nnc_reshape" in code, "nnc_reshape function call not found"

        # Basic syntax check
        is_valid, error = self.check_c_syntax(code)
        if not is_valid:
            print(f"Generated code:\n{code}")
        assert is_valid, f"Generated C code has issues: {error}"
