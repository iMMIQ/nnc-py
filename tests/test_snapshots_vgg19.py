"""Snapshot tests for VGG-19 model.

Run with: pytest tests/test_snapshots_vgg19.py
Update snapshots: pytest tests/test_snapshots_vgg19.py --snapshot-update
"""

import tempfile

import pytest
from syrupy.extensions.amber import AmberSnapshotExtension

from nnc_py import Compiler
from nnc_py.frontend.onnx_loader import ONNXFrontend
from test_common import BaseSnapshotTest, GraphSnapshotWrapper, CodeSnapshotWrapper


class TestIRSnapshots(BaseSnapshotTest):
    """IR snapshot tests for VGG-19."""

    def test_vgg19_ir_snapshot(self, snapshot):
        """Test VGG-19 IR structure snapshot."""
        model_path = self.models_dir / "vgg19.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        graph = self.frontend.load(str(model_path))
        wrapper = GraphSnapshotWrapper(graph)

        assert wrapper == snapshot


class TestCodegenSnapshots(BaseSnapshotTest):
    """Codegen snapshot tests for VGG-19."""

    def test_vgg19_codegen_snapshot(self, snapshot):
        """Test VGG-19 generated C code snapshot."""
        model_path = self.models_dir / "vgg19.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = Compiler(target="x86", opt_level=0)
            compiler.compile(str(model_path), tmpdir)

            normalized_code = self._get_normalized_code(tmpdir)
            assert normalized_code == snapshot

    def test_vgg19_codegen_with_runtime_o3(self):
        """Test VGG-19 code compiles and runs with -O3 optimization and sanitizers.

        Note: VGG-19 is a large model (~143M parameters, ~550MB weights).
        Only O3 mode is tested to reduce test time.
        """
        model_path = self.models_dir / "vgg19.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        # VGG-19 is large, needs extended timeout even with O3
        self._run_runtime_test(model_path, "vgg19", opt_level="O3", timeout=300)
