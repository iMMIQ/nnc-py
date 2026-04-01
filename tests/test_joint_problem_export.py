from __future__ import annotations

import json

import onnx
from onnx import TensorProto, helper

from nnc_py.tools.joint_problem_export import (
    build_joint_problem_from_onnx,
    export_joint_problem_to_path,
)


def _write_gemm_model(path) -> None:
    input_value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_value = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
    weight_init = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [4, 3],
        [1.0] * 12,
    )
    node = helper.make_node("Gemm", ["input", "weight"], ["output"], name="gemm0")
    graph = helper.make_graph(
        [node],
        "gemm_graph",
        [input_value],
        [output_value],
        [weight_init],
    )
    model = helper.make_model(graph, producer_name="nnc-py-test")
    onnx.save(model, path)


def test_build_joint_problem_from_onnx_exports_problem(tmp_path):
    model_path = tmp_path / "gemm.onnx"
    _write_gemm_model(model_path)

    problem = build_joint_problem_from_onnx(model_path, max_memory="1M")

    assert problem.schema_version == "joint_tiling_schedule_problem_v1"
    assert problem.objective == "min_makespan"
    assert problem.sram_capacity_bytes == 1024 * 1024
    assert len(problem.regions) == 1
    assert len(problem.recipes) == 1


def test_export_joint_problem_to_path_writes_json_file(tmp_path):
    model_path = tmp_path / "gemm.onnx"
    output_path = tmp_path / "problem.json"
    _write_gemm_model(model_path)

    export_joint_problem_to_path(model_path, output_path, max_memory="1M")

    payload = json.loads(output_path.read_text())
    assert payload["schema_version"] == "joint_tiling_schedule_problem_v1"
    assert payload["objective"] == "min_makespan"
    assert payload["sram_capacity_bytes"] == 1024 * 1024
