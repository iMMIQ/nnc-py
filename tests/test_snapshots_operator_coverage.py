"""Snapshot tests for Operator Coverage model.

This model is designed to test all core ONNX operators supported by nnc-py:
- Arithmetic: Add, Sub, Mul, Div, Pow, Sqrt
- Logical: And, Or, Not, Equal
- Shape: Reshape, Transpose, Unsqueeze, Split, Concat, Tile
- Reduction: ReduceMean, ReduceSum
- Activation: Relu, Clip
- Other: Constant, MatMul, Cast

Run with: pytest tests/test_snapshots_operator_coverage.py
Update snapshots: pytest tests/test_snapshots_operator_coverage.py --snapshot-update
"""

import tempfile

import pytest

from nnc_py import Compiler
from nnc_py.frontend.onnx_loader import ONNXFrontend
from test_common import BaseSnapshotTest, GraphSnapshotWrapper


class TestIRSnapshots(BaseSnapshotTest):
    """IR snapshot tests for Operator Coverage model."""

    def test_operator_coverage_ir_snapshot(self, snapshot):
        """Test Operator Coverage IR structure snapshot."""
        model_path = self.models_dir / "operator_coverage.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        graph = self.frontend.load(str(model_path))
        wrapper = GraphSnapshotWrapper(graph)

        assert wrapper == snapshot


class TestCodegenSnapshots(BaseSnapshotTest):
    """Codegen snapshot tests for Operator Coverage model."""

    def test_operator_coverage_codegen_snapshot(self, snapshot):
        """Test Operator Coverage generated C code snapshot."""
        model_path = self.models_dir / "operator_coverage.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = Compiler(target="x86", opt_level=0)
            compiler.compile(str(model_path), tmpdir)

            normalized_code = self._get_normalized_code(tmpdir)
            assert normalized_code == snapshot

    def test_operator_coverage_codegen_with_runtime(self):
        """Test Operator Coverage code compiles, runs with sanitizers, and output matches reference.

        This test verifies that the generated C code produces the same output as ONNX Runtime.
        Specifically tests:
        - Reshape with computed constant shape (Concat of Constants)
        - Greater operation with broadcasting (tensor > scalar)
        - Boolean operations (And, Or, Not) and their cast to float
        - Concatenation of boolean results and mean computation
        """
        model_path = self.models_dir / "operator_coverage.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "operator_coverage")

    def test_operator_coverage_codegen_with_runtime_o3(self):
        """Test Operator Coverage code compiles and runs with -O3 optimization and sanitizers."""
        model_path = self.models_dir / "operator_coverage.onnx"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        self._run_runtime_test(model_path, "operator_coverage", opt_level="O3")
