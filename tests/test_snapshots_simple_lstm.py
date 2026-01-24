"""Snapshot tests for Simple LSTM model.

Run with: pytest tests/test_snapshots_simple_lstm.py
Update snapshots: pytest tests/test_snapshots_simple_lstm.py --snapshot-update

Note: LSTM operator is not yet supported by nnc-py. This test file serves as
a reference for when LSTM support is added. The tests will currently fail
with "unsupported operator" errors.
"""

import tempfile

import pytest
from syrupy.extensions.amber import AmberSnapshotExtension

from nnc_py import Compiler
from nnc_py.frontend.onnx_loader import ONNXFrontend
from test_common import BaseSnapshotTest, GraphSnapshotWrapper, CodeSnapshotWrapper


class TestIRSnapshots(BaseSnapshotTest):
    """IR snapshot tests for Simple LSTM."""

    def test_simple_lstm_ir_snapshot(self, snapshot):
        """Test Simple LSTM IR structure snapshot."""
        model_path = self.models_dir / "simple_lstm.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        graph = self.frontend.load(str(model_path))
        wrapper = GraphSnapshotWrapper(graph)

        assert wrapper == snapshot


class TestCodegenSnapshots(BaseSnapshotTest):
    """Codegen snapshot tests for Simple LSTM."""

    def test_simple_lstm_codegen_snapshot(self, snapshot):
        """Test Simple LSTM generated C code snapshot."""
        model_path = self.models_dir / "simple_lstm.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = Compiler(target="x86", opt_level=0)
            compiler.compile(str(model_path), tmpdir)

            normalized_code = self._get_normalized_code(tmpdir)
            assert normalized_code == snapshot

    def test_simple_lstm_codegen_with_runtime(self):
        """Test Simple LSTM code compiles, runs with sanitizers, and output matches PyTorch."""
        model_path = self.models_dir / "simple_lstm.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "simple_lstm")

    def test_simple_lstm_codegen_with_runtime_o3(self):
        """Test Simple LSTM code compiles and runs with -O3 optimization and sanitizers."""
        model_path = self.models_dir / "simple_lstm.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "simple_lstm", opt_level="O3")
