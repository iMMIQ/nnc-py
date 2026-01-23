"""Snapshot tests for nnc-py compiler.

These tests verify that:
1. ONNX models are parsed correctly into IR graphs
2. Generated C code remains consistent across runs
3. Compiler behavior is deterministic
4. Generated code compiles and runs correctly with sanitizers
5. Runtime results match reference implementation

Run with: pytest tests/test_snapshots.py
Update snapshots: pytest tests/test_snapshots.py --snapshot-update
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from syrupy.extensions.amber import AmberSnapshotExtension

from nnc_py import Compiler
from nnc_py.frontend.onnx_loader import ONNXFrontend


class RuntimeErrorWrapper:
    """Wrapper to capture runtime execution results for snapshot testing."""

    def __init__(self, stdout: str, stderr: str, returncode: int):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    def __repr__(self) -> str:
        """Return representation for snapshot comparison."""
        lines = [
            "# Runtime Execution Result",
            f"# Return code: {self.returncode}",
            "",
            "## STDOUT",
            "---------",
        ]
        if self.stdout:
            lines.append(self.stdout)
        else:
            lines.append("(empty)")

        lines.append("")
        lines.append("## STDERR")
        lines.append("---------")
        if self.stderr:
            # Filter out ASan output that may vary between runs
            filtered_stderr = self._filter_asan_output(self.stderr)
            lines.append(filtered_stderr)
        else:
            lines.append("(empty)")

        return "\n".join(lines)

    def _filter_asan_output(self, stderr: str) -> str:
        """Filter out non-deterministic parts of ASan output."""
        lines = []
        for line in stderr.split("\n"):
            # Filter out memory addresses and pointer values
            line = re.sub(r'0x[0-9a-fA-F]+', '<PTR>', line)
            # Filter out thread IDs
            line = re.sub(r'T\d+', '<TID>', line)
            # Filter out process IDs
            line = re.sub(r'pid \d+', 'pid <PID>', line)
            lines.append(line)
        return "\n".join(lines)


class GraphSnapshotWrapper:
    """Wrapper to serialize Graph for snapshot testing."""

    def __init__(self, graph):
        self.graph = graph

    def _repr_pretty_(self):
        """Return a string representation for snapshot comparison."""
        lines = [
            f"# Graph: {self.graph.name}",
            f"# Inputs: {self.graph.inputs}",
            f"# Outputs: {self.graph.outputs}",
            "",
            "## Nodes",
            "",
        ]

        # Sort nodes by name for deterministic output
        for name in sorted(self.graph.nodes.keys()):
            node = self.graph.nodes[name]
            attrs_str = ", ".join(f"{k}={v}" for k, v in sorted(node.attrs.items()))
            if attrs_str:
                attrs_str = " [" + attrs_str + "]"
            lines.append(f"{name}: {node.op_type.value}{attrs_str}")
            lines.append(f"  inputs: {node.inputs}")
            lines.append(f"  outputs: {node.outputs}")
            lines.append("")

        lines.extend([
            "## Tensors",
            "",
        ])

        for name in sorted(self.graph.tensors.keys()):
            tensor = self.graph.tensors[name]
            lines.append(f"{name}: {tensor.dtype} {tensor.shape}")

        return "\n".join(lines)


class CodeSnapshotWrapper:
    """Wrapper to serialize generated C code for snapshot testing.

    Note: Memory allocation in the compiler has non-deterministic behavior
    due to unordered set usage. This wrapper normalizes memory offsets and
    other unstable values for stable snapshot comparison.
    """

    # Patterns to normalize non-deterministic values in generated code
    NORMALIZATION_PATTERNS = [
        # Memory pool offsets: _nnc_memory_pool + <number> -> _nnc_memory_pool + <OFFSET>
        # Note: handle variable spacing
        (r'_nnc_memory_pool\s*\+\s*\d+', '_nnc_memory_pool + <OFFSET>'),
        # Pointer addresses: 0x<hex> -> <PTR>
        (r'0x[0-9a-fA-F]+', '<PTR>'),
        # Array data values (can vary due to initialization) - for constants we keep them
        # but we normalize any embedded addresses
    ]

    def __init__(self, source_files: dict, exclude_files: list = None):
        """
        Args:
            source_files: Dict mapping filename to content
            exclude_files: List of filenames to exclude from snapshot (e.g., constants.c with large weights)
        """
        # Store normalized content directly for snapshot comparison
        self._normalized_content = self._normalize_source_files(source_files, exclude_files)

    def _normalize_code(self, content: str) -> str:
        """Normalize non-deterministic values in generated code."""
        import re

        normalized = content
        for pattern, replacement in self.NORMALIZATION_PATTERNS:
            normalized = re.sub(pattern, replacement, normalized)
        return normalized

    def _extract_structure(self, content: str, filename: str) -> str:
        """Extract structural parts of the code, excluding large data sections.

        This makes snapshots more readable and focused on structure.
        """
        lines = content.split('\n')
        result = []
        in_large_data = False
        data_line_count = 0
        max_data_lines_to_show = 5

        for line in lines:
            # Skip large constant arrays (like weights)
            if 'static const float tensor_' in line and '_data[' in line:
                in_large_data = True
                data_line_count = 0
                result.append(line + '  // ... <data array> ...')
                continue

            if in_large_data:
                data_line_count += 1
                if '};' in line:
                    in_large_data = False
                    if data_line_count > max_data_lines_to_show:
                        result.append('    // ... ' + str(data_line_count - max_data_lines_to_show) + ' more lines ...')
                        result.append(line)
                    else:
                        result.append(line)
                elif data_line_count <= max_data_lines_to_show:
                    result.append(line)
                continue

            result.append(line)

        return '\n'.join(result)

    def _normalize_source_files(self, source_files: dict, exclude_files: list = None) -> str:
        """Convert source files to normalized string representation."""
        exclude = set(exclude_files or [])
        lines = []

        for filename in sorted(source_files.keys()):
            if filename in exclude:
                lines.append(f"// {filename}")
                lines.append(f"// <excluded from snapshot - {source_files[filename].count(chr(10))} lines>")
                lines.append("")
                continue

            content = self._normalize_code(source_files[filename])
            if filename == "constants.c":
                # For constants, just show structure summary
                line_count = len(content.split('\n'))
                array_count = content.count('static const float tensor_')
                lines.append(f"// {filename}")
                lines.append(f"// Summary: {array_count} constant arrays, {line_count} total lines")
                lines.append("// <full content excluded - use --snapshot-details to include>")
            else:
                content = self._extract_structure(content, filename)
                lines.append(f"// {filename}")
                lines.append("// " + "=" * 60)
                lines.append(content)
            lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return representation for snapshot comparison."""
        return self._normalized_content

    def __eq__(self, other: object) -> bool:
        """Compare with another CodeSnapshotWrapper or string."""
        if isinstance(other, CodeSnapshotWrapper):
            return self._normalized_content == other._normalized_content
        if isinstance(other, str):
            return self._normalized_content == other
        return False


class TestIRSnapshots:
    """Snapshot tests for IR graph structure."""

    def setup_method(self):
        """Set up test fixtures."""
        self.frontend = ONNXFrontend()

    def test_lenet5_ir_snapshot(self, snapshot, models_dir):
        """Test LeNet-5 IR structure snapshot."""
        model_path = models_dir / "lenet5.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        graph = self.frontend.load(str(model_path))
        wrapper = GraphSnapshotWrapper(graph)

        assert wrapper == snapshot

    def test_resnet18_ir_snapshot(self, snapshot, models_dir):
        """Test ResNet-18 IR structure snapshot."""
        model_path = models_dir / "resnet18.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        graph = self.frontend.load(str(model_path))
        wrapper = GraphSnapshotWrapper(graph)

        assert wrapper == snapshot

    def test_simple_cnn_ir_snapshot(self, snapshot, models_dir):
        """Test Simple CNN IR structure snapshot."""
        model_path = models_dir / "simple_cnn.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        graph = self.frontend.load(str(model_path))
        wrapper = GraphSnapshotWrapper(graph)

        assert wrapper == snapshot

    def test_simple_mlp_ir_snapshot(self, snapshot, models_dir):
        """Test Simple MLP IR structure snapshot."""
        model_path = models_dir / "simple_mlp.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        graph = self.frontend.load(str(model_path))
        wrapper = GraphSnapshotWrapper(graph)

        assert wrapper == snapshot


class TestCodegenSnapshots:
    """Snapshot tests for generated C code."""

    def _compile_with_sanitizer(self, tmpdir: str, runtime_dir: Path, opt_level: str = "O0") -> str:
        """Compile the generated code with -g -fsanitize=address.

        Args:
            tmpdir: Temporary directory path
            runtime_dir: Runtime directory path
            opt_level: Optimization level (e.g., "O0", "O2", "O3")

        Returns:
            Path to the compiled executable.
        """
        source_files = ["model.c", "tensors.c", "test_runner.c", "constants.c"]
        obj_files = []

        # Get runtime path
        runtime_include = runtime_dir / "include"
        runtime_ops = runtime_dir / "x86" / "ops.c"

        # Compile flags with sanitizers
        cflags = ["-g", f"-{opt_level}", "-Wall", "-Wextra", "-fsanitize=address",
                  "-fno-common", "-std=c11", "-fPIC"]

        # Compile each source file
        for filename in source_files:
            filepath = Path(tmpdir) / filename
            if not filepath.exists():
                continue

            obj_file = filepath.with_suffix(".o")
            obj_files.append(obj_file)

            cmd = [
                "gcc",
                *cflags,
                f"-I{runtime_include}",
                "-c",
                str(filepath),
                "-o",
                str(obj_file),
            ]

            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to compile {filename}:\n{result.stderr}\n{result.stdout}"
                )

        # Compile runtime ops
        ops_obj = Path(tmpdir) / "ops.o"
        obj_files.append(ops_obj)

        cmd = [
            "gcc",
            *cflags,
            f"-I{runtime_include}",
            "-c",
            str(runtime_ops),
            "-o",
            str(ops_obj),
        ]

        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to compile ops.c:\n{result.stderr}\n{result.stdout}"
            )

        # Link
        exe_path = Path(tmpdir) / "model"
        cmd = [
            "gcc",
            "-fsanitize=address",
            "-o",
            str(exe_path),
        ] + [str(f) for f in obj_files] + ["-lm"]

        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to link:\n{result.stderr}\n{result.stdout}"
            )

        return str(exe_path)

    def _run_executable(self, exe_path: str, timeout: int = 30) -> tuple[str, str, int]:
        """Run the compiled executable and capture output.

        Returns:
            Tuple of (stdout, stderr, returncode).
        """
        result = subprocess.run(
            [exe_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"ASAN_OPTIONS": "detect_leaks=1"},
        )

        return result.stdout, result.stderr, result.returncode

    def _get_reference_output(self, model_path: Path, input_data: np.ndarray) -> dict:
        """Get reference output using PyTorch by loading ONNX model.

        Args:
            model_path: Path to ONNX model file
            input_data: Input tensor as numpy array

        Returns:
            Dict with output tensor names and values.
        """
        # Load ONNX model
        onnx_model = onnx.load(str(model_path))

        # Create a PyTorch model equivalent to the ONNX model
        # For now, we'll use a simpler approach with ONNX Runtime if available
        # or manual computation with numpy/torch

        # Get output names
        output_names = [out.name for out in onnx_model.graph.output]

        # Try using torch for inference by converting ONNX to torch
        # This is a simplified approach - for complex models, use onnxruntime
        import subprocess
        import sys

        # Try to import onnxruntime
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(model_path))
            inputs = {session.get_inputs()[0].name: input_data}
            outputs = session.run(None, inputs)
            return {name: val for name, val in zip(output_names, outputs)}
        except ImportError:
            # Fallback: use torch to manually compute
            return self._compute_with_torch(onnx_model, input_data)

    def _compute_with_torch(self, onnx_model: onnx.ModelProto, input_data: np.ndarray) -> dict:
        """Compute reference output using PyTorch based on ONNX graph.

        This handles simple models (Conv, Relu, etc.) by reconstructing
        the operations in PyTorch.
        """
        import torch

        input_tensor = torch.from_numpy(input_data)

        # Extract weights and create PyTorch modules
        weights = {}
        for init in onnx_model.graph.initializer:
            weights[init.name] = torch.from_numpy(onnx.numpy_helper.to_array(init))

        # Track tensors by name for complex models
        tensors = {}
        tensors[onnx_model.graph.input[0].name] = input_tensor

        output_names = [out.name for out in onnx_model.graph.output]
        outputs = {}

        for node in onnx_model.graph.node:
            op_type = node.op_type

            # Get input tensor(s)
            if len(node.input) == 0:
                continue

            # Get the first input (most ops have single input)
            input_name = node.input[0]
            if input_name in weights:
                # Input is a constant weight
                current_input = weights[input_name]
            elif input_name in tensors:
                current_input = tensors[input_name]
            else:
                # Skip if input not available yet
                continue

            if op_type == "Conv":
                # Get weight
                weight = weights[node.input[1]]
                bias = weights.get(node.input[2]) if len(node.input) > 2 else None

                # Get attributes
                kernel_shape = None
                pads = None
                strides = None
                for attr in node.attribute:
                    if attr.name == "kernel_shape":
                        kernel_shape = attr.ints
                    elif attr.name == "pads":
                        pads = attr.ints
                    elif attr.name == "strides":
                        strides = attr.ints

                # Convert to list
                kernel_shape = list(kernel_shape) if kernel_shape else [1, 1]
                strides = list(strides) if strides else [1, 1]
                pads = list(pads) if pads else [0, 0, 0, 0]

                # PyTorch uses [out_channels, in_channels, H, W]
                # Need to permute if needed
                weight = weight.contiguous()

                # Apply convolution
                result = F.conv2d(
                    current_input,
                    weight,
                    bias=bias,
                    stride=strides[0] if strides else 1,
                    padding=[pads[0], pads[2]] if pads else 0,
                )

                # Store output
                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "Relu":
                result = torch.relu(current_input)

                # Store output
                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "MaxPool":
                kernel_shape = None
                pads = None
                strides = None
                for attr in node.attribute:
                    if attr.name == "kernel_shape":
                        kernel_shape = attr.ints
                    elif attr.name == "pads":
                        pads = attr.ints
                    elif attr.name == "strides":
                        strides = attr.ints

                kernel_shape = list(kernel_shape) if kernel_shape else [1, 1]
                strides = list(strides) if strides else [1, 1]
                pads = list(pads) if pads else [0, 0, 0, 0]

                result = F.max_pool2d(
                    current_input,
                    kernel_size=kernel_shape,
                    stride=strides,
                    padding=[pads[0], pads[2]] if pads else 0,
                )

                # Store output
                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "Flatten":
                axis = 1  # Default
                for attr in node.attribute:
                    if attr.name == "axis":
                        axis = attr.i
                result = current_input.flatten(axis)

                # Store output
                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "Gemm":
                # Linear layer
                weight = weights[node.input[1]]
                bias = weights.get(node.input[2]) if len(node.input) > 2 else None

                # Transpose weight if needed (ONNX uses [K, N], PyTorch uses [N, K])
                # Actually ONNX Gemm: Y = alpha * A^T * B + beta * C
                # But we need to check the transpose attributes
                trans_a = 0
                trans_b = 0
                for attr in node.attribute:
                    if attr.name == "transA":
                        trans_a = attr.i
                    elif attr.name == "transB":
                        trans_b = attr.i

                # Handle transposition for PyTorch F.linear
                # F.linear computes: output = input * weight^T + bias
                # ONNX Gemm with transB=1 computes: output = input * weight^T (same as PyTorch)
                # ONNX Gemm with transB=0 computes: output = input * weight (need to transpose for PyTorch)
                # So: when transB=0, we transpose; when transB=1, we don't
                if trans_b == 0:
                    weight = weight.t()

                # For 2D input (batch, features)
                if len(current_input.shape) == 2:
                    result = torch.nn.functional.linear(current_input, weight, bias)
                else:
                    # Flatten first
                    result = current_input.flatten(1)
                    result = torch.nn.functional.linear(result, weight, bias)

                # Store output
                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "Add":
                # Get second input
                if len(node.input) < 2:
                    continue
                input2_name = node.input[1]
                if input2_name in weights:
                    input2 = weights[input2_name]
                elif input2_name in tensors:
                    input2 = tensors[input2_name]
                else:
                    continue

                result = current_input + input2

                # Store output
                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "GlobalAveragePool":
                # Global average pooling: average over all spatial dimensions
                # For [N, C, H, W], output is [N, C, 1, 1]
                if len(current_input.shape) == 4:
                    result = F.adaptive_avg_pool2d(current_input, (1, 1))
                else:
                    # For other shapes, just average over all dims except batch
                    result = current_input.mean(dim=list(range(1, len(current_input.shape))), keepdim=True)

                # Store output
                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "Identity":
                # Identity just passes through
                result = current_input

                # Store output
                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

        # Return all outputs
        return outputs

    def _parse_c_output(self, stdout: str) -> np.ndarray:
        """Parse output values from C program stdout.

        The test_runner prints output values like:
          output[0] = 0.123456
          output[1] = 0.234567
        """
        import re

        values = []
        for line in stdout.split("\n"):
            match = re.search(r"output\[(\d+)\]\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)", line)
            if match:
                values.append(float(match.group(2)))

        return np.array(values, dtype=np.float32)

    def _compare_outputs(self, c_output: np.ndarray, ref_output: np.ndarray,
                        rtol: float = 1e-3, atol: float = 1e-5) -> tuple[bool, str]:
        """Compare C output with reference output.

        Returns:
            Tuple of (match, message).
        """
        if c_output.shape != ref_output.shape:
            c_flat = c_output.flatten()
            ref_flat = ref_output.flatten()
        else:
            c_flat = c_output
            ref_flat = ref_output

        if len(c_flat) != len(ref_flat):
            return False, f"Size mismatch: C output has {len(c_flat)} elements, reference has {len(ref_flat)}"

        # Compare element-wise
        max_diff = 0.0
        max_rel_diff = 0.0
        mismatch_count = 0

        for i, (c_val, ref_val) in enumerate(zip(c_flat, ref_flat)):
            diff = abs(c_val - ref_val)
            max_diff = max(max_diff, diff)

            if abs(ref_val) > atol:
                rel_diff = diff / abs(ref_val)
                max_rel_diff = max(max_rel_diff, rel_diff)

            if diff > atol and diff > rtol * abs(ref_val):
                mismatch_count += 1
                if mismatch_count <= 5:  # Only show first 5 mismatches
                    print(f"  Mismatch at index {i}: C={c_val:.6f}, Ref={ref_val:.6f}, diff={diff:.6e}")

        match = mismatch_count == 0
        msg = f"max_diff={max_diff:.6e}, max_rel_diff={max_rel_diff:.6e}"
        if mismatch_count:
            msg += f", {mismatch_count}/{len(c_flat)} mismatches"

        return match, msg

    def _get_normalized_code(self, tmpdir: str) -> str:
        """Get normalized generated code for snapshot comparison."""
        source_files = {}
        for filename in ["model.h", "model.c", "tensors.c"]:
            filepath = Path(tmpdir) / filename
            if filepath.exists():
                source_files[filename] = filepath.read_text()

        # Normalize memory offsets and other non-deterministic values
        import re
        normalized_files = {}
        for filename, content in source_files.items():
            # Replace memory pool offsets
            content = re.sub(r'_nnc_memory_pool\s*\+\s*\d+', '_nnc_memory_pool + <OFFSET>', content)
            # Remove pointer addresses
            content = re.sub(r'0x[0-9a-fA-F]+', '<PTR>', content)
            # Normalize memory size comments and definitions
            content = re.sub(r'\/\* Total size: \d+ bytes \([0-9.]+ [KMGT]?B\) \*\/', '/* Total size: <SIZE> bytes (<SIZE_FORMAT>) */', content)
            content = re.sub(r'\/\* Buffers: \d+, Tensors: \d+ \*\/', '/* Buffers: <COUNT>, Tensors: <COUNT> */', content)
            content = re.sub(r'#define NNC_MEMORY_SIZE \d+', '#define NNC_MEMORY_SIZE <SIZE>', content)
            content = re.sub(r'nbytes = \d+,', 'nbytes = <SIZE>,', content)
            normalized_files[filename] = content

        # Build output
        lines = []
        for filename in sorted(normalized_files.keys()):
            content = normalized_files[filename]
            lines.append(f"// {filename}")
            lines.append("// " + "=" * 60)
            lines.append(content)
            lines.append("")
        return "\n".join(lines)

    def test_lenet5_codegen_snapshot(self, snapshot, models_dir):
        """Test LeNet-5 generated C code snapshot."""
        model_path = models_dir / "lenet5.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = Compiler(target="x86", opt_level=0)
            compiler.compile(str(model_path), tmpdir)

            normalized_code = self._get_normalized_code(tmpdir)
            assert normalized_code == snapshot

    # Note: test_lenet5_codegen_with_runtime is skipped due to Gemm operator
    # code generation issue - the generated nnc_gemm call has incorrect arguments

    def test_simple_mlp_codegen_snapshot(self, snapshot, models_dir):
        """Test Simple MLP generated C code snapshot."""
        model_path = models_dir / "simple_mlp.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = Compiler(target="x86", opt_level=0)
            compiler.compile(str(model_path), tmpdir)

            normalized_code = self._get_normalized_code(tmpdir)
            assert normalized_code == snapshot

    def test_simple_conv_codegen_with_runtime(self):
        """Test simple_conv code compiles, runs with sanitizers, and output matches PyTorch."""
        model_path = Path(__file__).parent / "simple_conv.onnx"
        self._run_runtime_test(model_path, "simple_conv")

    def test_simple_mlp_codegen_with_runtime(self, models_dir):
        """Test simple_mlp code compiles, runs with sanitizers, and output matches PyTorch."""
        model_path = models_dir / "simple_mlp.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "simple_mlp")

    def test_lenet5_codegen_with_runtime(self, models_dir):
        """Test lenet5 code compiles, runs with sanitizers, and output matches PyTorch."""
        model_path = models_dir / "lenet5.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "lenet5")

    def test_simple_cnn_codegen_with_runtime(self, models_dir):
        """Test simple_cnn code compiles, runs with sanitizers, and output matches PyTorch."""
        model_path = models_dir / "simple_cnn.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "simple_cnn")

    def test_resnet18_codegen_with_runtime(self, models_dir):
        """Test resnet18 code compiles, runs with sanitizers, and output matches PyTorch."""
        model_path = models_dir / "resnet18.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "resnet18")

    def _run_runtime_test(self, model_path: Path, model_name: str, opt_level: str = "O0") -> None:
        """Run runtime test with sanitizers and compare output with PyTorch reference.

        Args:
            model_path: Path to ONNX model file
            model_name: Name of the model for logging
            opt_level: Optimization level (e.g., "O0", "O2", "O3")
        """
        # Get runtime directory
        runtime_dir = Path(__file__).parent.parent / "runtime"

        # Create test input data (same pattern as used in C test runner)
        # The C test runner initializes with: ((float)i * 0.01f)
        onnx_model = onnx.load(str(model_path))
        input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        input_size = np.prod(input_shape)
        input_data = np.arange(input_size, dtype=np.float32) * 0.01
        input_data = input_data.reshape(input_shape)

        # Get reference output using PyTorch
        print(f"\\n[{model_name}] Computing reference output with PyTorch...")
        ref_outputs = self._get_reference_output(model_path, input_data)
        ref_output = list(ref_outputs.values())[0]
        print(f"  Reference output shape: {ref_output.shape}")

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = Compiler(target="x86", opt_level=0)
            compiler.compile(str(model_path), tmpdir)

            # Compile with sanitizers
            exe_path = self._compile_with_sanitizer(tmpdir, runtime_dir, opt_level)

            # Run the executable
            stdout, stderr, returncode = self._run_executable(exe_path)

            # Check that the program ran successfully
            assert returncode == 0, f"Program failed with return code {returncode}\\nstdout: {stdout}\\nstderr: {stderr}"

            # Check for ASan errors in stderr
            assert "ERROR: AddressSanitizer" not in stderr, f"AddressSanitizer detected errors:\\n{stderr}"

            # Verify basic output patterns
            assert "NNC Model Runner" in stdout
            assert "Inference complete" in stdout

            # Parse C output and compare with reference
            # Note: The C test_runner only outputs first 10 values
            print(f"[{model_name}] Comparing C output with PyTorch reference...")
            c_output_flat = self._parse_c_output(stdout)
            print(f"  C output size: {len(c_output_flat)} values")

            # Compare only the values that C outputs (first 10)
            ref_output_flat = ref_output.flatten()[:len(c_output_flat)]
            match, msg = self._compare_outputs(c_output_flat, ref_output_flat)
            assert match, f"Output mismatch: {msg}\\nC output:\\n{c_output_flat}\\nReference:\\n{ref_output_flat}"
            print(f"  [{model_name}] {msg}")
            print(f"  [{model_name}] C output matches PyTorch reference!")

    # Tests with -O3 optimization level

    def test_simple_conv_codegen_with_runtime_o3(self):
        """Test simple_conv code compiles and runs with -O3 optimization and sanitizers."""
        model_path = Path(__file__).parent / "simple_conv.onnx"
        self._run_runtime_test(model_path, "simple_conv", opt_level="O3")

    def test_simple_mlp_codegen_with_runtime_o3(self, models_dir):
        """Test simple_mlp code compiles and runs with -O3 optimization and sanitizers."""
        model_path = models_dir / "simple_mlp.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "simple_mlp", opt_level="O3")

    def test_lenet5_codegen_with_runtime_o3(self, models_dir):
        """Test lenet5 code compiles and runs with -O3 optimization and sanitizers."""
        model_path = models_dir / "lenet5.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "lenet5", opt_level="O3")

    def test_simple_cnn_codegen_with_runtime_o3(self, models_dir):
        """Test simple_cnn code compiles and runs with -O3 optimization and sanitizers."""
        model_path = models_dir / "simple_cnn.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "simple_cnn", opt_level="O3")

    def test_resnet18_codegen_with_runtime_o3(self, models_dir):
        """Test resnet18 code compiles and runs with -O3 optimization and sanitizers."""
        model_path = models_dir / "resnet18.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "resnet18", opt_level="O3")
