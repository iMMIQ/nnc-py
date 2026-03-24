"""End-to-end tiled-planning correctness tests against ONNX Runtime."""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from nnc_py import Compiler
from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassManager
from tests.test_common import BaseSnapshotTest


@dataclass(frozen=True)
class TiledRuntimeCaseResult:
    model_c: str
    tensors_c: str
    comparison_results: dict
    max_abs_diff: float


def _make_tiled_conv_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 96, 96])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 96, 96])

    conv_weight = helper.make_tensor(
        "conv_weight",
        TensorProto.FLOAT,
        [16, 16, 3, 3],
        (np.arange(16 * 16 * 3 * 3, dtype=np.float32) / 2048.0).reshape(-1).tolist(),
    )

    conv = helper.make_node(
        "Conv",
        inputs=["input", "conv_weight"],
        outputs=["output"],
        name="conv0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )

    graph = helper.make_graph(
        [conv],
        "tiled_conv_correctness",
        [input_info],
        [output_info],
        [conv_weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_tiled_maxpool_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 192, 192])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 96, 96])

    maxpool = helper.make_node(
        "MaxPool",
        inputs=["input"],
        outputs=["output"],
        name="pool0",
        kernel_shape=[2, 2],
        strides=[2, 2],
    )

    graph = helper.make_graph(
        [maxpool],
        "tiled_maxpool_correctness",
        [input_info],
        [output_info],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_tiled_conv_add_relu_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 96, 96])
    residual_info = helper.make_tensor_value_info("residual", TensorProto.FLOAT, [1, 16, 96, 96])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 96, 96])

    conv_weight = helper.make_tensor(
        "conv_weight",
        TensorProto.FLOAT,
        [16, 16, 3, 3],
        (np.arange(16 * 16 * 3 * 3, dtype=np.float32) / 2048.0).reshape(-1).tolist(),
    )

    conv = helper.make_node(
        "Conv",
        inputs=["input", "conv_weight"],
        outputs=["conv_out"],
        name="conv0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    add = helper.make_node(
        "Add",
        inputs=["conv_out", "residual"],
        outputs=["add_out"],
        name="add0",
    )
    relu = helper.make_node(
        "Relu",
        inputs=["add_out"],
        outputs=["output"],
        name="relu0",
    )

    graph = helper.make_graph(
        [conv, add, relu],
        "tiled_conv_add_relu_correctness",
        [input_info, residual_info],
        [output_info],
        [conv_weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _assert_comparison_results_clean(
    results: dict,
    *,
    allowed_missing_in_nnc: set[str] | None = None,
) -> None:
    allowed_missing_in_nnc = allowed_missing_in_nnc or set()
    assert len(results["shape_mismatch"]) == 0, results["shape_mismatch"]
    assert len(results["mismatched"]) == 0, results["mismatched"]
    unexpected_missing = [
        tensor_name
        for tensor_name in results["missing_in_nnc"]
        if tensor_name not in allowed_missing_in_nnc
    ]
    assert len(unexpected_missing) == 0, unexpected_missing
    assert len(results["missing_in_onnx"]) == 0, results["missing_in_onnx"]


def test_debug_comparator_is_imported_lazily():
    assert "DebugComparator" not in globals()


def test_assert_comparison_results_rejects_missing_tensors():
    with pytest.raises(AssertionError):
        _assert_comparison_results_clean(
            {
                "matched": ["output"],
                "mismatched": [],
                "missing_in_nnc": ["conv_out"],
                "missing_in_onnx": [],
                "shape_mismatch": [],
            }
        )


class TestTiledRuntimeCorrectness(BaseSnapshotTest):
    def setup_method(self):
        super().setup_method()
        pytest.importorskip("onnxruntime")

    def _assert_phase1_tiled_metadata(
        self,
        model_path: Path,
        *,
        tiled_node_name: str,
        expected_op_family: str,
    ) -> None:
        graph = self.frontend.load(str(model_path))
        ctx = CompileContext(graph=graph, target="x86", optimization_level=3)

        pass_manager = PassManager()
        for pass_obj in PassManager.get_default_passes(3):
            pass_manager.register(pass_obj)
        pass_manager.run(ctx)

        schedule_candidates = ctx.metadata.get("schedule_candidates", {})
        assert tiled_node_name in schedule_candidates, f"Missing schedule candidate for {tiled_node_name}"
        candidate = schedule_candidates[tiled_node_name]
        assert candidate.op_family == expected_op_family
        assert candidate.must_tile is True

        execution_plans = ctx.metadata.get("node_execution_plans", {})
        assert tiled_node_name in execution_plans, f"Missing execution plan for {tiled_node_name}"
        plan = execution_plans[tiled_node_name]
        assert plan.op_family == expected_op_family
        assert plan.tile_axes

    def _run_tiled_runtime_case(
        self,
        model: onnx.ModelProto,
        *,
        model_name: str,
        tiled_node_name: str,
        expected_op_family: str,
        allowed_missing_in_nnc: set[str] | None = None,
        timeout: int = 90,
    ) -> TiledRuntimeCaseResult:
        runtime_dir = Path(__file__).resolve().parent.parent / "runtime"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            model_path = tmpdir_path / f"{model_name}.onnx"
            onnx.save(model, model_path)

            self._assert_phase1_tiled_metadata(
                model_path,
                tiled_node_name=tiled_node_name,
                expected_op_family=expected_op_family,
            )

            compiler = Compiler(target="x86", opt_level=3, debug_mode=True)
            compiler.compile(str(model_path), tmpdir)
            model_c = (tmpdir_path / "model.c").read_text()
            tensors_c = (tmpdir_path / "tensors.c").read_text()

            exe_path = self._compile_with_sanitizer(tmpdir, runtime_dir, opt_level="O3")
            result = subprocess.run(
                [exe_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(tmpdir_path),
                env={"ASAN_OPTIONS": "detect_leaks=0"},
            )
            stdout, stderr, returncode = result.stdout, result.stderr, result.returncode

            assert returncode == 0, (
                f"Program failed with return code {returncode}\nstdout: {stdout}\nstderr: {stderr}"
            )
            assert "ERROR: AddressSanitizer" not in stderr, stderr
            assert "NNC Model Runner" in stdout
            assert "Inference complete" in stdout

            debug_file = tmpdir_path / "nnc_debug_output.txt"
            assert debug_file.exists(), f"Debug output file not created. stdout: {stdout}"

            from nnc_py.tools.debug_compare import DebugComparator

            comparator = DebugComparator(str(debug_file), str(model_path), rtol=1e-3, atol=1e-5)
            results = comparator.compare()

            _assert_comparison_results_clean(
                results,
                allowed_missing_in_nnc=allowed_missing_in_nnc,
            )

            max_abs_diff = max(
                (
                    float(item["max_diff"])
                    for item in results["mismatched"]
                    if "max_diff" in item
                ),
                default=0.0,
            )
            return TiledRuntimeCaseResult(
                model_c=model_c,
                tensors_c=tensors_c,
                comparison_results=results,
                max_abs_diff=max_abs_diff,
            )

    def test_tiled_conv_runtime_path_matches_onnxruntime_reference(self):
        result = self._run_tiled_runtime_case(
            _make_tiled_conv_model(),
            model_name="tiled_conv",
            tiled_node_name="conv0",
            expected_op_family="conv2d",
        )
        assert "tile-aware wrapper" in result.model_c
        assert ".data = NULL" not in result.tensors_c
        assert "_nnc_input_buffer_tensor_input" in result.tensors_c
        assert result.max_abs_diff == 0.0

    def test_tiled_maxpool_runtime_path_matches_onnxruntime_reference(self):
        result = self._run_tiled_runtime_case(
            _make_tiled_maxpool_model(),
            model_name="tiled_maxpool",
            tiled_node_name="pool0",
            expected_op_family="maxpool",
        )
        assert "tile-aware wrapper" in result.model_c
        assert ".data = NULL" not in result.tensors_c
        assert "_nnc_input_buffer_tensor_input" in result.tensors_c
        assert result.max_abs_diff == 0.0

    def test_tiled_conv_add_relu_runtime_path_matches_onnxruntime_reference(self):
        result = self._run_tiled_runtime_case(
            _make_tiled_conv_add_relu_model(),
            model_name="tiled_conv_add_relu",
            tiled_node_name="conv0",
            expected_op_family="conv2d",
            allowed_missing_in_nnc={"add_out"},
        )
        assert result.model_c.count("tile-aware wrapper") >= 2
        assert "node_conv0_body();" in result.model_c
        assert "node_fused_add_relu_1_body();" in result.model_c
        assert ".data = NULL" not in result.tensors_c
        assert "_nnc_input_buffer_tensor_input" in result.tensors_c
        assert "_nnc_input_buffer_tensor_residual" in result.tensors_c
        assert result.max_abs_diff == 0.0
