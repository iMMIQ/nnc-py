"""Snapshot tests for Operator Coverage model with large parameters and memory constraints.

This test file extends the operator coverage model with larger tensor dimensions
to test memory-constrained scenarios. It verifies that:

1. The model compiles correctly with memory limits (spill/reload code generation)
2. Results match reference implementations even when memory is constrained
3. All operators still work correctly under memory pressure

Run with: pytest tests/test_snapshots_operator_coverage_large.py
Update snapshots: pytest tests/test_snapshots_operator_coverage_large.py --snapshot-update

Generate the model with: python tools/dump_classic_models.py
"""

import os
import re
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest

from nnc_py import Compiler
from nnc_py.frontend.onnx_loader import ONNXFrontend
from test_common import BaseSnapshotTest, GraphSnapshotWrapper, is_lsan_ptrace_error


class TestIRSnapshots(BaseSnapshotTest):
    """IR snapshot tests for Large Operator Coverage model."""

    def test_large_operator_coverage_ir_snapshot(self, snapshot):
        """Test Large Operator Coverage IR structure snapshot."""
        model_path = self.models_dir / "operator_coverage_large.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}. Generate with tools/dump_classic_models.py")

        graph = self.frontend.load(str(model_path))
        wrapper = GraphSnapshotWrapper(graph)

        assert wrapper == snapshot


class TestCodegenSnapshots(BaseSnapshotTest):
    """Codegen snapshot tests for Large Operator Coverage model."""

    def test_large_operator_coverage_codegen_snapshot(self, snapshot):
        """Test Large Operator Coverage generated C code snapshot."""
        model_path = self.models_dir / "operator_coverage_large.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}. Generate with tools/dump_classic_models.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = Compiler(target="x86", opt_level=0)
            compiler.compile(str(model_path), tmpdir)

            normalized_code = self._get_normalized_code(tmpdir)
            assert normalized_code == snapshot

    def test_large_operator_coverage_codegen_with_memory_constraint(self, snapshot):
        """Test Large Operator Coverage with memory constraint generates spill code."""
        model_path = self.models_dir / "operator_coverage_large.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}. Generate with tools/dump_classic_models.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set memory limit to force spill (32KB should be too small for this model)
            compiler = Compiler(target="x86", opt_level=0)
            compiler.compile(str(model_path), tmpdir, max_memory="32KB", memory_strategy="basic")

            # Check that spill code was generated
            tensors_c = Path(tmpdir) / "tensors.c"
            content = tensors_c.read_text()

            has_slow_pool = "_nnc_slow_pool" in content
            has_fast_pool = "_nnc_fast_pool" in content or "_nnc_memory_pool" in content

            assert has_slow_pool, "Expected slow pool to be generated for memory-constrained compilation"
            assert has_fast_pool, "Expected fast pool to be generated"

            normalized_code = self._get_normalized_code(tmpdir)
            assert normalized_code == snapshot

    def test_large_operator_coverage_runtime_with_memory_constraint(self):
        """Test Large Operator Coverage with memory constraint produces correct results.

        This is a comprehensive test that verifies:
        1. Model compiles with memory constraints
        2. Generated code runs without errors
        3. Output matches reference (PyTorch/ONNX Runtime)
        """
        model_path = self.models_dir / "operator_coverage_large.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}. Generate with tools/dump_classic_models.py")

        runtime_dir = Path(__file__).parent.parent / "runtime"

        # Load model to get input shape
        onnx_model = onnx.load(str(model_path))
        input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        input_size = np.prod(input_shape)

        # Generate test input
        np.random.seed(42)
        input_data = np.random.randn(*input_shape).astype(np.float32)

        print(f"\n[Large Operator Coverage] Input shape: {input_shape}")
        print(f"  Memory constraint: 32KB (forces spill)")

        # Get reference output
        ref_outputs = self._get_reference_output(model_path, input_data)
        ref_output = list(ref_outputs.values())[0]
        print(f"  Reference output shape: {ref_output.shape}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Compile with memory constraint
            compiler = Compiler(target="x86", opt_level=0)
            compiler.compile(str(model_path), tmpdir, max_memory="32KB", memory_strategy="basic")

            # Check for spill
            tensors_c = Path(tmpdir) / "tensors.c"
            content = tensors_c.read_text()
            has_slow_pool = "_nnc_slow_pool" in content
            print(f"  Slow pool generated: {has_slow_pool}")

            # Create test runner
            test_runner_c = Path(tmpdir) / "test_runner.c"
            input_flat_size = int(np.prod(input_shape))
            output_flat_size = int(np.prod(ref_output.shape))

            test_runner_code = f"""
#include <stdio.h>
#include <math.h>
#include "model.h"
#include "nnc_ops.h"

extern Tensor tensor_input;
extern Tensor tensor_output;

int main(void) {{
    // Initialize input with test data
    float *input_data = (float *)tensor_input.data;
    int input_size = {input_flat_size};

    // Simple pattern for initialization
    for (int i = 0; i < input_size; i++) {{
        input_data[i] = (float)i * 0.01f - 10.0f;
    }}

    // Run inference
    nnc_run();

    // Print output
    float *output_data = (float *)tensor_output.data;
    int output_size = {output_flat_size};

    printf("NNC Large Model Test\\n");
    printf("Output size: %d\\n", output_size);
    printf("Values:");
    for (int i = 0; i < output_size; i++) {{
        printf(" %.6f", output_data[i]);
    }}
    printf("\\n");

    return 0;
}}
"""
            test_runner_c.write_text(test_runner_code)

            # Compile with test runner
            exe_path = self._compile_with_sanitizer(tmpdir, runtime_dir)

            # Run executable
            stdout, stderr, returncode = self._run_executable(exe_path)

            if is_lsan_ptrace_error(stderr):
                pytest.skip(
                    "LeakSanitizer is unavailable in this execution environment "
                    "(ptrace restriction)."
                )

            assert returncode == 0, f"Program failed with return code {returncode}\\nstdout: {stdout}\\nstderr: {stderr}"
            assert "ERROR: AddressSanitizer" not in stderr, f"AddressSanitizer detected errors:\\n{stderr}"

            # Compute expected output with same initialization pattern
            test_input = np.arange(input_flat_size, dtype=np.float32) * 0.01 - 10.0
            test_input = test_input.reshape(input_shape)
            expected_outputs = self._get_reference_output(model_path, test_input)
            expected_output = list(expected_outputs.values())[0]

            # Parse C output
            match = re.search(r'Values:(.+)', stdout)
            if match:
                actual_output = np.array([float(x) for x in match.group(1).strip().split()])

                # Compare
                match, msg = self._compare_outputs(actual_output, expected_output.flatten())
                assert match, f"Output mismatch: {msg}"
                print(f"  Output matches reference! {msg}")
            else:
                # Alternative parsing
                c_output_flat = self._parse_c_output(stdout)
                match, msg = self._compare_outputs(c_output_flat, expected_output.flatten())
                assert match, f"Output mismatch: {msg}"
                print(f"  Output matches reference! {msg}")

    def test_large_operator_coverage_codegen_with_runtime(self):
        """Test Large Operator Coverage code compiles, runs with sanitizers, and output matches reference.

        This test verifies that the generated C code produces the same output as ONNX Runtime
        without memory constraints (baseline test).
        """
        model_path = self.models_dir / "operator_coverage_large.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}. Generate with tools/dump_classic_models.py")

        onnx_model = onnx.load(str(model_path))
        input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        input_size = np.prod(input_shape)
        input_data = np.arange(input_size, dtype=np.float32) * 0.01
        input_data = input_data.reshape(input_shape)

        print(f"\\n[Large Operator Coverage] Computing reference output...")
        ref_outputs = self._get_reference_output(model_path, input_data)
        ref_output = list(ref_outputs.values())[0]
        print(f"  Reference output shape: {ref_output.shape}")

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = Compiler(target="x86", opt_level=0)
            compiler.compile(str(model_path), tmpdir)

            runtime_dir = Path(__file__).parent.parent / "runtime"
            exe_path = self._compile_with_sanitizer(tmpdir, runtime_dir)

            stdout, stderr, returncode = self._run_executable(exe_path)

            if is_lsan_ptrace_error(stderr):
                pytest.skip(
                    "LeakSanitizer is unavailable in this execution environment "
                    "(ptrace restriction)."
                )

            assert returncode == 0, f"Program failed with return code {returncode}\\nstdout: {stdout}\\nstderr: {stderr}"
            assert "ERROR: AddressSanitizer" not in stderr, f"AddressSanitizer detected errors:\\n{stderr}"
            assert "NNC Model Runner" in stdout or "Inference complete" in stdout

            print(f"[Large Operator Coverage] Comparing C output with reference...")
            c_output_flat = self._parse_c_output(stdout)
            print(f"  C output size: {len(c_output_flat)} values")

            ref_output_flat = ref_output.flatten()[:len(c_output_flat)]
            match, msg = self._compare_outputs(c_output_flat, ref_output_flat)
            assert match, f"Output mismatch: {msg}"
            print(f"  C output matches reference! {msg}")


if __name__ == "__main__":
    # Generate the model when run directly
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
    from dump_classic_models import export_large_operator_coverage_model

    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating large operator coverage model...")
    export_large_operator_coverage_model(models_dir)
    print(f"Model saved to: {models_dir / 'operator_coverage_large.onnx'}")
