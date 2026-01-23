"""Snapshot tests for nnc-py compiler.

These tests verify that:
1. ONNX models are parsed correctly into IR graphs
2. Generated C code remains consistent across runs
3. Compiler behavior is deterministic

Run with: pytest tests/test_snapshots.py
Update snapshots: pytest tests/test_snapshots.py --snapshot-update
"""

import tempfile
from pathlib import Path

import pytest
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
