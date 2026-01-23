"""Snapshot tests for Simple Conv model.

Run with: pytest tests/test_snapshots_simple_conv.py
Update snapshots: pytest tests/test_snapshots_simple_conv.py --snapshot-update
"""

import tempfile
from pathlib import Path

import pytest
from syrupy.extensions.amber import AmberSnapshotExtension

from nnc_py import Compiler
from nnc_py.frontend.onnx_loader import ONNXFrontend
from test_common import BaseSnapshotTest, GraphSnapshotWrapper, CodeSnapshotWrapper


class TestIRSnapshots(BaseSnapshotTest):
    """IR snapshot tests for Simple Conv."""

    def test_simple_conv_ir_snapshot(self, snapshot):
        """Test Simple Conv IR structure snapshot."""
        model_path = Path(__file__).parent / "simple_conv.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        graph = self.frontend.load(str(model_path))
        wrapper = GraphSnapshotWrapper(graph)

        assert wrapper == snapshot


class TestCodegenSnapshots(BaseSnapshotTest):
    """Codegen snapshot tests for Simple Conv."""

    def test_simple_conv_codegen_snapshot(self, snapshot):
        """Test Simple Conv generated C code snapshot."""
        model_path = Path(__file__).parent / "simple_conv.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = Compiler(target="x86", opt_level=0)
            compiler.compile(str(model_path), tmpdir)

            normalized_code = self._get_normalized_code(tmpdir)
            assert normalized_code == snapshot

    def test_simple_conv_codegen_with_runtime(self):
        """Test Simple Conv code compiles, runs with sanitizers, and output matches PyTorch."""
        model_path = Path(__file__).parent / "simple_conv.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "simple_conv")

    def test_simple_conv_codegen_with_runtime_o3(self):
        """Test Simple Conv code compiles and runs with -O3 optimization and sanitizers."""
        model_path = Path(__file__).parent / "simple_conv.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "simple_conv", opt_level="O3")
