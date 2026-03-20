from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    model_path: Path
    workload_batch_sizes: list[int]
    warmup_iterations: int
    measure_iterations: int


_CASES: dict[str, BenchmarkCase] = {}


def _register_case(case: BenchmarkCase) -> None:
    if case.name in _CASES:
        raise ValueError(f"benchmark case already registered: {case.name}")

    _CASES[case.name] = case


_register_case(
    BenchmarkCase(
        name="resnet18",
        model_path=Path("models/resnet18.onnx"),
        workload_batch_sizes=[1, 8, 16, 32],
        warmup_iterations=5,
        measure_iterations=20,
    )
)


def get_benchmark_case(name: str) -> BenchmarkCase:
    try:
        return _CASES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark case: {name}") from exc


def list_benchmark_cases() -> list[str]:
    return sorted(_CASES)
