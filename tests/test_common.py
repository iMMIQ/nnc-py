"""Shared utilities for snapshot tests."""

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
import torch.nn.functional as F
from syrupy.extensions.amber import AmberSnapshotExtension

from nnc_py import Compiler
from nnc_py.frontend.onnx_loader import ONNXFrontend


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
        r'_nnc_memory_pool\s*\+\s*\d+', '_nnc_memory_pool + <OFFSET>',
        # Pointer addresses: 0x<hex> -> <PTR>
        r'0x[0-9a-fA-F]+', '<PTR>',
    ]

    def __init__(self, source_files: dict, exclude_files: list = None):
        self._normalized_content = self._normalize_source_files(source_files, exclude_files)

    def _normalize_code(self, content: str) -> str:
        """Normalize non-deterministic values in generated code."""
        import re

        normalized = content
        for pattern, replacement in self.NORMALIZATION_PATTERNS:
            normalized = re.sub(pattern, replacement, normalized)
        return normalized

    def _extract_structure(self, content: str, filename: str) -> str:
        """Extract structural parts of the code, excluding large data sections."""
        lines = content.split('\n')
        result = []
        in_large_data = False
        data_line_count = 0
        max_data_lines_to_show = 5

        for line in lines:
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
                    elif data_line_count <= max_data_lines_to_show:
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
                line_count = len(content.split('\n'))
                array_count = content.count('static const float tensor_')
                lines.append(f"// {filename}")
                lines.append(f"// Summary: {array_count} constant arrays, {line_count} total lines")
                lines.append("// <full content excluded>")
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


class BaseSnapshotTest:
    """Base class for snapshot tests with common utilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.frontend = ONNXFrontend()
        self.models_dir = Path(__file__).parent.parent / "models"

    def _compile_with_sanitizer(self, tmpdir: str, runtime_dir: Path, opt_level: str = "O0") -> str:
        """Compile the generated code with -g -fsanitize=address."""
        # Try constants_loader.c first, fall back to constants.c for backward compatibility
        constants_file = "constants_loader.c"
        if not (Path(tmpdir) / constants_file).exists():
            constants_file = "constants.c"
        source_files = ["model.c", "tensors.c", "test_runner.c", constants_file]
        obj_files = []

        runtime_include = runtime_dir / "include"
        runtime_ops = runtime_dir / "x86" / "ops.c"

        cflags = ["-g", f"-{opt_level}", "-Wall", "-Wextra", "-fsanitize=address",
                  "-fno-common", "-std=c11", "-fPIC"]

        for filename in source_files:
            filepath = Path(tmpdir) / filename
            if not filepath.exists():
                continue

            obj_file = filepath.with_suffix(".o")
            obj_files.append(obj_file)

            cmd = [
                "gcc", *cflags, f"-I{runtime_include}", "-c",
                str(filepath), "-o", str(obj_file),
            ]

            result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to compile {filename}:\n{result.stderr}\n{result.stdout}")

        # Compile runtime ops
        ops_obj = Path(tmpdir) / "ops.o"
        obj_files.append(ops_obj)

        cmd = ["gcc", *cflags, f"-I{runtime_include}", "-c", str(runtime_ops), "-o", str(ops_obj)]
        result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile ops.c:\n{result.stderr}\n{result.stdout}")

        # Link
        exe_path = Path(tmpdir) / "model"
        cmd = ["gcc", "-fsanitize=address", "-o", str(exe_path)] + [str(f) for f in obj_files] + ["-lm"]
        result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to link:\n{result.stderr}\n{result.stdout}")

        return str(exe_path)

    def _run_executable(self, exe_path: str, timeout: int = 30) -> tuple[str, str, int]:
        """Run the compiled executable and capture output."""
        exe_dir = Path(exe_path).parent
        result = subprocess.run(
            [exe_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(exe_dir),  # Set working directory to exe location for constants.bin
            env={"ASAN_OPTIONS": "detect_leaks=1"},
        )
        return result.stdout, result.stderr, result.returncode

    def _get_reference_output(self, model_path: Path, input_data: np.ndarray) -> dict:
        """Get reference output using ONNX Runtime fallback or PyTorch."""
        onnx_model = onnx.load(str(model_path))
        output_names = [out.name for out in onnx_model.graph.output]

        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(model_path))
            inputs = {session.get_inputs()[0].name: input_data}
            outputs = session.run(None, inputs)
            return {name: val for name, val in zip(output_names, outputs)}
        except ImportError:
            return self._compute_with_torch(onnx_model, input_data)

    def _compute_with_torch(self, onnx_model: onnx.ModelProto, input_data: np.ndarray) -> dict:
        """Compute reference output using PyTorch based on ONNX graph."""
        import torch

        input_tensor = torch.from_numpy(input_data)

        weights = {}
        for init in onnx_model.graph.initializer:
            weights[init.name] = torch.from_numpy(onnx.numpy_helper.to_array(init))

        tensors = {}
        tensors[onnx_model.graph.input[0].name] = input_tensor

        output_names = [out.name for out in onnx_model.graph.output]
        outputs = {}

        # Helper to get attribute value from ONNX node
        def get_attr(node, attr_name, default=None):
            for attr in node.attribute:
                if attr.name == attr_name:
                    if attr.HasField('i'):
                        return attr.i
                    elif attr.HasField('f'):
                        return attr.f
                    elif attr.HasField('s'):
                        return attr.s
                    elif attr.ints:
                        return list(attr.ints)
                    elif attr.floats:
                        return list(attr.floats)
                    elif attr.strings:
                        return list(attr.strings)
            return default

        for node in onnx_model.graph.node:
            op_type = node.op_type

            if len(node.input) == 0:
                continue

            input_name = node.input[0]
            if input_name in weights:
                current_input = weights[input_name]
            elif input_name in tensors:
                current_input = tensors[input_name]
            else:
                continue

            if op_type == "Conv":
                weight = weights[node.input[1]]
                bias = weights.get(node.input[2]) if len(node.input) > 2 else None

                kernel_shape = list(get_attr(node, "kernel_shape", [1, 1]))
                strides = list(get_attr(node, "strides", [1, 1]))
                pads = list(get_attr(node, "pads", [0, 0, 0, 0]))

                weight = weight.contiguous()
                result = F.conv2d(current_input, weight, bias=bias,
                                  stride=strides[0] if strides else 1,
                                  padding=[pads[0], pads[2]] if pads else 0)

                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "Relu":
                result = torch.relu(current_input)
                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "MaxPool":
                kernel_shape = list(get_attr(node, "kernel_shape", [1, 1]))
                strides = list(get_attr(node, "strides", [1, 1]))
                pads = list(get_attr(node, "pads", [0, 0, 0, 0]))

                result = F.max_pool2d(current_input, kernel_size=kernel_shape,
                                      stride=strides, padding=[pads[0], pads[2]] if pads else 0)

                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "Flatten":
                axis = get_attr(node, "axis", 1)
                result = current_input.flatten(axis)
                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "Gemm":
                weight = weights[node.input[1]]
                bias = weights.get(node.input[2]) if len(node.input) > 2 else None

                trans_a = get_attr(node, "transA", 0)
                trans_b = get_attr(node, "transB", 0)

                if trans_b == 0:
                    weight = weight.t()

                if len(current_input.shape) == 2:
                    result = torch.nn.functional.linear(current_input, weight, bias)
                else:
                    result = current_input.flatten(1)
                    result = torch.nn.functional.linear(result, weight, bias)

                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "Add":
                input2_name = node.input[1]
                if input2_name in weights:
                    input2 = weights[input2_name]
                elif input2_name in tensors:
                    input2 = tensors[input2_name]
                else:
                    continue

                result = current_input + input2
                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "GlobalAveragePool":
                if len(current_input.shape) == 4:
                    result = F.adaptive_avg_pool2d(current_input, (1, 1))
                else:
                    result = current_input.mean(dim=list(range(1, len(current_input.shape))), keepdim=True)

                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

            elif op_type == "Identity":
                result = current_input
                for out_name in node.output:
                    tensors[out_name] = result
                    if out_name in output_names:
                        outputs[out_name] = result.numpy()

        return outputs

    def _parse_c_output(self, stdout: str) -> np.ndarray:
        """Parse output values from C program stdout."""
        values = []
        for line in stdout.split("\n"):
            match = re.search(r"output\[(\d+)\]\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)", line)
            if match:
                values.append(float(match.group(2)))
        return np.array(values, dtype=np.float32)

    def _compare_outputs(self, c_output: np.ndarray, ref_output: np.ndarray,
                        rtol: float = 1e-3, atol: float = 1e-5) -> tuple[bool, str]:
        """Compare C output with reference output."""
        if c_output.shape != ref_output.shape:
            c_flat = c_output.flatten()
            ref_flat = ref_output.flatten()
        else:
            c_flat = c_output
            ref_flat = ref_output

        if len(c_flat) != len(ref_flat):
            return False, f"Size mismatch: C output has {len(c_flat)} elements, reference has {len(ref_flat)}"

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
                if mismatch_count <= 5:
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

        import re
        normalized_files = {}
        for filename, content in source_files.items():
            content = re.sub(r'_nnc_memory_pool\s*\+\s*\d+', '_nnc_memory_pool + <OFFSET>', content)
            content = re.sub(r'0x[0-9a-fA-F]+', '<PTR>', content)
            content = re.sub(r'\/\* Total size: \d+ bytes \([0-9.]+ [KMGT]?B\) \*\/', '/* Total size: <SIZE> bytes (<SIZE_FORMAT>) */', content)
            content = re.sub(r'\/\* Buffers: \d+, Tensors: \d+ \*\/', '/* Buffers: <COUNT>, Tensors: <COUNT> */', content)
            content = re.sub(r'#define NNC_MEMORY_SIZE \d+', '#define NNC_MEMORY_SIZE <SIZE>', content)
            content = re.sub(r'nbytes = \d+,', 'nbytes = <SIZE>,', content)
            # Normalize node names that contain memory addresses (e.g., MatMul_140157833117536_MatMul)
            # This handles both code and comments
            # Match pattern: OP_<ID>_OP where <ID> is a number
            ops_pattern = r'(MatMul|Add|Mul|Div|Softmax|Relu|Transpose|Flatten|Gemm|Conv|Constant|Split|Reshape|Concat)'
            content = re.sub(fr'_{ops_pattern}_\d+_\1', r'_\1_<ID>_\1', content)
            content = re.sub(fr'node_{ops_pattern}_\d+_\1', r'node_\1_<ID>_\1', content)
            content = re.sub(fr'tensor_{ops_pattern}_\d+_\1', r'tensor_\1_<ID>_\1', content)
            content = re.sub(fr'/\* {ops_pattern}: \1_\d+_\1 \*/', r'/* \1: \1_<ID>_\1 */', content)
            normalized_files[filename] = content

        lines = []
        for filename in sorted(normalized_files.keys()):
            content = normalized_files[filename]
            lines.append(f"// {filename}")
            lines.append("// " + "=" * 60)
            lines.append(content)
            lines.append("")
        return "\n".join(lines)

    def _run_runtime_test(self, model_path: Path, model_name: str, opt_level: str = "O0") -> None:
        """Run runtime test with sanitizers and compare intermediate outputs with ONNX Runtime.

        Uses debug mode to capture intermediate tensor outputs and compare each layer's
        output with ONNX Runtime reference implementation.
        """
        from nnc_py.tools.debug_compare import DebugComparator

        runtime_dir = Path(__file__).parent.parent / "runtime"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Compile with debug mode enabled
            print(f"\\n[{model_name}] Compiling with debug mode...")
            compiler = Compiler(target="x86", opt_level=0, debug_mode=True)
            compiler.compile(str(model_path), tmpdir)

            exe_path = self._compile_with_sanitizer(tmpdir, runtime_dir, opt_level)

            stdout, stderr, returncode = self._run_executable(exe_path)

            assert returncode == 0, f"Program failed with return code {returncode}\\nstdout: {stdout}\\nstderr: {stderr}"
            assert "ERROR: AddressSanitizer" not in stderr, f"AddressSanitizer detected errors:\\n{stderr}"
            assert "NNC Model Runner" in stdout
            assert "Inference complete" in stdout

            # Check that debug output file was created
            debug_file = Path(tmpdir) / "nnc_debug_output.txt"
            assert debug_file.exists(), f"Debug output file not created. stdout: {stdout}"

            print(f"[{model_name}] Comparing NNC outputs with ONNX Runtime...")

            # Compare NNC debug output with ONNX Runtime
            # Use relaxed tolerances for floating point differences across implementations
            # ResNet and deeper models may have accumulated floating point differences
            comparator = DebugComparator(str(debug_file), str(model_path), rtol=1e-1, atol=1e-3)
            results = comparator.compare()

            # Report results
            total = len(results["matched"]) + len(results["mismatched"]) + len(results["shape_mismatch"])
            print(f"  Total tensors compared: {total}")
            print(f"  ✓ Matched: {len(results['matched'])}")
            print(f"  ✗ Mismatched: {len(results['mismatched'])}")
            print(f"  ⚠ Shape mismatch: {len(results['shape_mismatch'])}")

            # Report mismatches if any
            if results["mismatched"]:
                print(f"  [{model_name}] Mismatched tensors:")
                for item in results["mismatched"][:5]:  # Show first 5
                    if "max_diff" in item:
                        print(f"    - {item['tensor']}: max_diff={item['max_diff']:.6e}")
                if len(results["mismatched"]) > 5:
                    print(f"    ... and {len(results['mismatched']) - 5} more")

            if results["shape_mismatch"]:
                print(f"  [{model_name}] Shape mismatches:")
                for item in results["shape_mismatch"][:5]:
                    print(f"    - {item['tensor']}: NNC {item['nnc_shape']} vs ONNX {item['onnx_shape']}")

            # Assert that all outputs match
            assert len(results["shape_mismatch"]) == 0, \
                f"Shape mismatches found: {len(results['shape_mismatch'])}"
            assert len(results["mismatched"]) == 0, \
                f"Output mismatches found: {len(results['mismatched'])}"

            print(f"  [{model_name}] All {total} tensor outputs match ONNX Runtime!")

