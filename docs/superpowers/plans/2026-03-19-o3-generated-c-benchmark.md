# O3 Generated C Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a host-side benchmark framework that measures the performance and static memory footprint of O3-generated C code and compares results across algorithm versions, starting with `ResNet18`.

**Architecture:** Add an independent `benchmarks/` package at the repo root that orchestrates compile/build/run/report without modifying the existing `nnc` CLI. The harness will compile `models/resnet18.onnx` with `Compiler(target="x86", opt_level=3)`, generate a dedicated benchmark runner, build the generated C code with `gcc`, collect timing and static memory metrics, then write structured JSON and optional baseline comparisons.

**Tech Stack:** Python 3.10+, `dataclasses`, `json`, `subprocess`, `pathlib`, existing `nnc_py.Compiler`, generated C runner using `clock_gettime`, `pytest`

---

### Task 1: Benchmark Case Registry

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/cases.py`
- Test: `tests/test_benchmark_cases.py`

- [ ] **Step 1: Write the failing case-registry tests**

```python
from pathlib import Path

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
    try:
        get_benchmark_case("missing")
    except KeyError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected missing case to raise")


def test_list_cases_includes_resnet18():
    assert "resnet18" in list_benchmark_cases()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_cases.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors for `benchmarks.cases`

- [ ] **Step 3: Write minimal benchmark case implementation**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    model_path: Path
    workload_batch_sizes: list[int]
    warmup_iterations: int
    measure_iterations: int


_CASES = {
    "resnet18": BenchmarkCase(
        name="resnet18",
        model_path=Path("models/resnet18.onnx"),
        workload_batch_sizes=[1, 8, 16, 32],
        warmup_iterations=5,
        measure_iterations=20,
    )
}


def get_benchmark_case(name: str) -> BenchmarkCase:
    try:
        return _CASES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark case: {name}") from exc


def list_benchmark_cases() -> list[str]:
    return sorted(_CASES)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_cases.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/__init__.py benchmarks/cases.py tests/test_benchmark_cases.py
git commit -m "feat: add benchmark case registry"
```

### Task 2: Static Memory and Artifact Metrics

**Files:**
- Create: `benchmarks/metrics.py`
- Test: `tests/test_benchmark_metrics.py`

- [ ] **Step 1: Write the failing metrics tests**

```python
import json

from benchmarks.metrics import collect_artifact_metrics, extract_memory_pool_sizes


def test_extract_memory_pool_sizes(tmp_path):
    tensors_c = tmp_path / "tensors.c"
    tensors_c.write_text(
        "#define NNC_FAST_MEMORY_SIZE 4096\n"
        "#define NNC_SLOW_MEMORY_SIZE 1024\n"
    )

    sizes = extract_memory_pool_sizes(tensors_c)
    assert sizes["fast_memory_bytes"] == 4096
    assert sizes["slow_memory_bytes"] == 1024


def test_collect_artifact_metrics_sums_static_sizes(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "tensors.c").write_text("#define NNC_FAST_MEMORY_SIZE 256\n")
    (build_dir / "constants.bin").write_bytes(b"12345678")
    exe = build_dir / "resnet18_bench"
    exe.write_bytes(b"binary")

    metrics = collect_artifact_metrics(build_dir, exe)
    assert metrics["fast_memory_bytes"] == 256
    assert metrics["slow_memory_bytes"] == 0
    assert metrics["constants_bytes"] == 8
    assert metrics["binary_size_bytes"] == 6
    assert metrics["total_static_bytes"] == 264
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing function errors

- [ ] **Step 3: Write minimal metrics extraction implementation**

```python
import re
from pathlib import Path


def extract_memory_pool_sizes(tensors_c_path: Path) -> dict[str, int]:
    content = tensors_c_path.read_text()

    def _extract(name: str) -> int:
        match = re.search(rf"#define\\s+{name}\\s+(\\d+)", content)
        return int(match.group(1)) if match else 0

    return {
        "fast_memory_bytes": _extract("NNC_FAST_MEMORY_SIZE"),
        "slow_memory_bytes": _extract("NNC_SLOW_MEMORY_SIZE"),
    }


def collect_artifact_metrics(build_dir: Path, executable_path: Path) -> dict[str, int]:
    pool_sizes = extract_memory_pool_sizes(build_dir / "tensors.c")
    constants_path = build_dir / "constants.bin"
    constants_bytes = constants_path.stat().st_size if constants_path.exists() else 0
    binary_size_bytes = executable_path.stat().st_size

    return {
        **pool_sizes,
        "constants_bytes": constants_bytes,
        "binary_size_bytes": binary_size_bytes,
        "total_static_bytes": (
            pool_sizes["fast_memory_bytes"]
            + pool_sizes["slow_memory_bytes"]
            + constants_bytes
        ),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/metrics.py tests/test_benchmark_metrics.py
git commit -m "feat: add benchmark artifact metrics"
```

### Task 3: Baseline Comparison Logic

**Files:**
- Create: `benchmarks/compare.py`
- Test: `tests/test_benchmark_compare.py`

- [ ] **Step 1: Write the failing comparison tests**

```python
from benchmarks.compare import compare_results


def test_compare_results_reports_latency_and_memory_deltas():
    baseline = {
        "commit": "base123",
        "runs": [{"batch_size": 1, "latency_ms_p50": 10.0, "throughput_samples_per_sec": 100.0}],
        "memory": {"total_static_bytes": 1000},
    }
    candidate = {
        "commit": "cand456",
        "runs": [{"batch_size": 1, "latency_ms_p50": 8.0, "throughput_samples_per_sec": 120.0}],
        "memory": {"total_static_bytes": 900},
    }

    diff = compare_results(baseline, candidate)
    assert diff["baseline_commit"] == "base123"
    assert diff["candidate_commit"] == "cand456"
    assert diff["runs"][0]["batch_size"] == 1
    assert diff["runs"][0]["latency_delta_pct"] == -20.0
    assert diff["runs"][0]["throughput_delta_pct"] == 20.0
    assert diff["memory"]["total_static_bytes_delta"] == -100


def test_compare_results_rejects_mismatched_batch_sets():
    baseline = {"commit": "base", "runs": [{"batch_size": 1}], "memory": {"total_static_bytes": 1}}
    candidate = {"commit": "cand", "runs": [{"batch_size": 8}], "memory": {"total_static_bytes": 1}}

    try:
        compare_results(baseline, candidate)
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("expected mismatched batch sizes to raise")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_compare.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing function errors

- [ ] **Step 3: Write minimal comparison implementation**

```python
def _pct_delta(baseline_value: float, candidate_value: float) -> float:
    return round(((candidate_value - baseline_value) / baseline_value) * 100.0, 4)


def compare_results(baseline: dict, candidate: dict) -> dict:
    baseline_runs = {run["batch_size"]: run for run in baseline["runs"]}
    candidate_runs = {run["batch_size"]: run for run in candidate["runs"]}

    if baseline_runs.keys() != candidate_runs.keys():
        raise ValueError("baseline and candidate batch_size sets must match")

    run_diffs = []
    for batch_size in sorted(baseline_runs):
        base_run = baseline_runs[batch_size]
        cand_run = candidate_runs[batch_size]
        run_diffs.append(
            {
                "batch_size": batch_size,
                "latency_delta_pct": _pct_delta(base_run["latency_ms_p50"], cand_run["latency_ms_p50"]),
                "throughput_delta_pct": _pct_delta(
                    base_run["throughput_samples_per_sec"],
                    cand_run["throughput_samples_per_sec"],
                ),
            }
        )

    return {
        "baseline_commit": baseline["commit"],
        "candidate_commit": candidate["commit"],
        "runs": run_diffs,
        "memory": {
            "total_static_bytes_delta": (
                candidate["memory"]["total_static_bytes"] - baseline["memory"]["total_static_bytes"]
            )
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_compare.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/compare.py tests/test_benchmark_compare.py
git commit -m "feat: add benchmark result comparison"
```

### Task 4: Dedicated Benchmark Runner Generator

**Files:**
- Create: `benchmarks/runner_gen.py`
- Test: `tests/test_benchmark_runner_gen.py`

- [ ] **Step 1: Write the failing runner-generator tests**

```python
from benchmarks.runner_gen import generate_benchmark_runner


def test_runner_uses_monotonic_clock_and_json_output():
    source = generate_benchmark_runner(
        model_name="resnet18",
        workload_batch_sizes=[1, 8],
        warmup_iterations=5,
        measure_iterations=20,
        entry_point="nnc_run",
    )

    assert "clock_gettime(CLOCK_MONOTONIC" in source
    assert '"runs": [' in source
    assert "for (int batch_iter = 0; batch_iter < workload_batch; batch_iter++)" in source
    assert "throughput_samples_per_sec" in source


def test_runner_loads_constants_when_requested():
    source = generate_benchmark_runner(
        model_name="resnet18",
        workload_batch_sizes=[1],
        warmup_iterations=1,
        measure_iterations=2,
        entry_point="nnc_run",
        has_constants=True,
    )

    assert "nnc_load_constants(\"constants.bin\")" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_runner_gen.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing function errors

- [ ] **Step 3: Write minimal runner generation implementation**

```python
def generate_benchmark_runner(...):
    return f"""#include <time.h>
#include <stdio.h>
#include "model.h"

static double now_ms(void) {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}}

int main(void) {{
    int workloads[] = {{1, 8}};
    printf("{{\\"model\\": \\"resnet18\\", \\"runs\\": [");
    for (int workload_idx = 0; workload_idx < 2; workload_idx++) {{
        int workload_batch = workloads[workload_idx];
        for (int warmup = 0; warmup < 5; warmup++) {{
            for (int batch_iter = 0; batch_iter < workload_batch; batch_iter++) {{
                nnc_run();
            }}
        }}
        double start_ms = now_ms();
        for (int iter = 0; iter < 20; iter++) {{
            for (int batch_iter = 0; batch_iter < workload_batch; batch_iter++) {{
                nnc_run();
            }}
        }}
        double elapsed_ms = now_ms() - start_ms;
        double throughput_samples_per_sec = (20.0 * workload_batch) / (elapsed_ms / 1000.0);
        printf("{{\\"batch_size\\": %d, \\"throughput_samples_per_sec\\": %.6f}}", workload_batch, throughput_samples_per_sec);
    }}
    printf("]}}");
    return 0;
}}"""
```

- [ ] **Step 4: Flesh out the implementation to match the spec before calling the task done**

Implementation requirements:
- Initialize graph inputs once with deterministic data
- Load `constants.bin` when present
- Emit per-run latency samples so Python can compute mean/p50/p95
- Use workload batch semantics: one measured iteration runs `nnc_run()` N times
- Avoid trailing-comma JSON bugs

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_benchmark_runner_gen.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add benchmarks/runner_gen.py tests/test_benchmark_runner_gen.py
git commit -m "feat: add benchmark runner generator"
```

### Task 5: Harness Orchestration and Result Persistence

**Files:**
- Create: `benchmarks/harness.py`
- Modify: `benchmarks/cases.py`
- Modify: `benchmarks/runner_gen.py`
- Modify: `benchmarks/metrics.py`
- Modify: `benchmarks/compare.py`
- Test: `tests/test_benchmark_harness.py`

- [ ] **Step 1: Write the failing harness tests**

```python
import json
from pathlib import Path

from benchmarks.harness import build_result_payload, parse_runner_output


def test_parse_runner_output_returns_json_payload():
    payload = parse_runner_output('{"model":"resnet18","runs":[{"batch_size":1}]}')
    assert payload["model"] == "resnet18"
    assert payload["runs"][0]["batch_size"] == 1


def test_build_result_payload_includes_commit_memory_and_runs(tmp_path):
    runner_payload = {
        "model": "resnet18",
        "runs": [{"batch_size": 1, "latency_ms_mean": 1.0, "latency_ms_p50": 1.0, "latency_ms_p95": 1.0, "throughput_samples_per_sec": 100.0}],
    }
    artifact_metrics = {
        "fast_memory_bytes": 256,
        "slow_memory_bytes": 0,
        "constants_bytes": 8,
        "binary_size_bytes": 16,
        "total_static_bytes": 264,
    }

    result = build_result_payload(
        model_name="resnet18",
        commit="abc1234",
        runner_payload=runner_payload,
        artifact_metrics=artifact_metrics,
        output_dir=tmp_path / "build",
        executable_path=tmp_path / "build" / "resnet18_bench",
        cflags=["-O3", "-std=c11"],
    )

    assert result["model"] == "resnet18"
    assert result["commit"] == "abc1234"
    assert result["compiler_config"]["opt_level"] == 3
    assert result["memory"]["total_static_bytes"] == 264
    assert result["runs"][0]["batch_size"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_harness.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing function errors

- [ ] **Step 3: Implement harness helpers first**

```python
def parse_runner_output(stdout: str) -> dict:
    return json.loads(stdout)


def build_result_payload(...):
    return {
        "model": model_name,
        "commit": commit,
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "compiler_config": {"target": "x86", "opt_level": 3},
        "build_config": {"cc": "gcc", "cflags": cflags},
        "runs": runner_payload["runs"],
        "memory": artifact_metrics,
        "artifacts": {
            "output_dir": str(output_dir),
            "executable_path": str(executable_path),
        },
    }
```

- [ ] **Step 4: Expand `harness.py` to orchestrate the full flow**

Implementation requirements:
- Parse CLI args for `--model`, `--batch-sizes`, `--baseline-result`, `--output`
- Load the selected case from `benchmarks.cases`
- Compile the ONNX model with `Compiler(target="x86", opt_level=3)`
- Write the generated benchmark runner into the compiler output directory
- Compile generated sources plus `runtime/x86/ops.c` into a benchmark executable
- Execute the benchmark process and parse its JSON stdout
- Collect artifact metrics and persist final result JSON under `benchmarks/results/`
- When `--baseline-result` is provided, call `compare_results()` and write a sibling diff JSON

- [ ] **Step 5: Add a harness integration test with monkeypatched subprocess**

```python
def test_harness_writes_result_file(tmp_path, monkeypatch):
    ...
    # Monkeypatch Compiler.compile, subprocess.run, and git commit lookup
    # Assert the result JSON is written with the expected fields
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_benchmark_harness.py -v`
Expected: PASS

- [ ] **Step 7: Run the focused benchmark test suite**

Run: `pytest tests/test_benchmark_cases.py tests/test_benchmark_metrics.py tests/test_benchmark_compare.py tests/test_benchmark_runner_gen.py tests/test_benchmark_harness.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add benchmarks/cases.py benchmarks/runner_gen.py benchmarks/metrics.py benchmarks/compare.py benchmarks/harness.py tests/test_benchmark_cases.py tests/test_benchmark_metrics.py tests/test_benchmark_compare.py tests/test_benchmark_runner_gen.py tests/test_benchmark_harness.py
git commit -m "feat: add O3 generated C benchmark harness"
```

### Task 6: Documentation and End-to-End Manual Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-03-19-o3-generated-c-benchmark-design.md`
- Test: manual benchmark run

- [ ] **Step 1: Add benchmark usage documentation**

Document:
- benchmark goal and scope
- workload batch semantics
- example commands
- output result file locations
- baseline comparison workflow

- [ ] **Step 2: Run a real benchmark smoke test on `resnet18`**

Run: `python -m benchmarks.harness --model resnet18 --batch-sizes 1`
Expected:
- generated benchmark executable builds successfully
- runner prints valid JSON
- result file appears under `benchmarks/results/`

- [ ] **Step 3: Run a baseline comparison smoke test**

Run:
```bash
python -m benchmarks.harness --model resnet18 --batch-sizes 1 --output benchmarks/results/resnet18-current.json
python -m benchmarks.compare benchmarks/results/resnet18-current.json benchmarks/results/resnet18-current.json
```

Expected:
- compare command completes successfully
- diff output reports zero deltas

- [ ] **Step 4: Re-run the focused automated tests after docs and smoke validation**

Run: `pytest tests/test_benchmark_cases.py tests/test_benchmark_metrics.py tests/test_benchmark_compare.py tests/test_benchmark_runner_gen.py tests/test_benchmark_harness.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md docs/superpowers/specs/2026-03-19-o3-generated-c-benchmark-design.md benchmarks tests
git commit -m "docs: add benchmark framework usage"
```
