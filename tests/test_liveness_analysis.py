import onnx
from onnx import TensorProto, helper

from nnc_py.frontend.onnx_loader import ONNXFrontend
from nnc_py.ir.context import CompileContext
from nnc_py.passes.liveness import LivenessAnalysisPass, TensorLiveness


def _run_liveness(graph):
    ctx = CompileContext(graph=graph, target="x86", optimization_level=0)
    LivenessAnalysisPass().run(ctx)
    return ctx.metadata["tensor_liveness"]


def test_tensor_liveness_tracks_use_positions_for_branching_graph(tmp_path):
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["input"], ["left"]),
            helper.make_node("Sigmoid", ["input"], ["right"]),
            helper.make_node("Add", ["left", "right"], ["output"]),
        ],
        "branch",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    path = tmp_path / "branch.onnx"
    onnx.save(model, path)

    frontend = ONNXFrontend()
    graph = frontend.load(str(path))
    liveness = _run_liveness(graph)

    assert liveness["left"].use_positions == [2]
    assert liveness["right"].use_positions == [2]
    assert liveness["input"].use_positions == [0, 1]


def test_tensor_liveness_reports_next_use_and_remaining_counts():
    info = TensorLiveness(
        tensor_name="x",
        live_start=0,
        live_end=5,
        use_positions=[1, 3, 5],
    )

    assert info.next_use_after(0) == 1
    assert info.next_use_after(1) == 3
    assert info.next_use_after(4) == 5
    assert info.next_use_after(5) is None

    assert info.remaining_uses_after(0) == 3
    assert info.remaining_uses_after(1) == 2
    assert info.remaining_uses_after(3) == 1
    assert info.remaining_uses_after(5) == 0
