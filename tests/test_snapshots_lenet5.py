"""Snapshot tests for LeNet-5 model.

Run with: pytest tests/test_snapshots_lenet5.py
Update snapshots: pytest tests/test_snapshots_lenet5.py --snapshot-update
"""

import tempfile

import pytest
from syrupy.extensions.amber import AmberSnapshotExtension

from nnc_py import Compiler
from nnc_py.frontend.onnx_loader import ONNXFrontend
from test_common import BaseSnapshotTest, GraphSnapshotWrapper, CodeSnapshotWrapper


class TestIRSnapshots(BaseSnapshotTest):
    """IR snapshot tests for LeNet-5."""

    def test_lenet5_ir_snapshot(self, snapshot):
        """Test LeNet-5 IR structure snapshot."""
        model_path = self.models_dir / "lenet5.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        graph = self.frontend.load(str(model_path))
        wrapper = GraphSnapshotWrapper(graph)

        assert wrapper == snapshot


class TestCodegenSnapshots(BaseSnapshotTest):
    """Codegen snapshot tests for LeNet-5."""

    def test_lenet5_codegen_snapshot(self, snapshot):
        """Test LeNet-5 generated C code snapshot."""
        model_path = self.models_dir / "lenet5.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = Compiler(target="x86", opt_level=0)
            compiler.compile(str(model_path), tmpdir)

            normalized_code = self._get_normalized_code(tmpdir)
            assert normalized_code == snapshot

    def test_lenet5_codegen_with_runtime(self):
        """Test LeNet-5 code compiles, runs with sanitizers, and output matches PyTorch."""
        model_path = self.models_dir / "lenet5.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "lenet5")

    def test_lenet5_codegen_with_runtime_o3(self):
        """Test LeNet-5 code compiles and runs with -O3 optimization and sanitizers."""
        model_path = self.models_dir / "lenet5.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "lenet5", opt_level="O3")
