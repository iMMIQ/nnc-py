from pathlib import Path

import pytest

from benchmarks.cases import BenchmarkCase, get_benchmark_case, list_benchmark_cases


def test_resnet18_case_defaults():
    case = get_benchmark_case("resnet18")
    assert isinstance(case, BenchmarkCase)
    assert case.name == "resnet18"
    assert case.model_path == Path("models/resnet18.onnx")
    assert case.workload_batch_sizes == [1, 8, 16, 32]
    assert case.warmup_iterations == 5
    assert case.measure_iterations == 20


def test_unknown_case_raises_clear_error():
    with pytest.raises(KeyError) as exc:
        get_benchmark_case("missing")

    assert "missing" in str(exc.value)


def test_list_cases_includes_resnet18():
    assert "resnet18" in list_benchmark_cases()
