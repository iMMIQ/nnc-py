from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from nnc_py import Compiler

from benchmarks.cases import get_benchmark_case
from benchmarks.compare import compare_results
from benchmarks.metrics import collect_artifact_metrics, has_memory_layout_defines
from benchmarks.runner_gen import generate_benchmark_runner


def parse_runner_output(stdout: str) -> dict:
    if not isinstance(stdout, str):
        raise ValueError("runner stdout must be a string")

    raw = stdout.strip()
    if not raw:
        raise ValueError("runner stdout is empty; expected JSON payload on stdout")

    def _try_parse(candidate: str) -> dict | None:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            raise ValueError("runner stdout JSON must be an object")
        return payload

    parsed = _try_parse(raw)
    if parsed is not None:
        return parsed

    # If stdout includes logs, try the last non-empty line as the payload.
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        parsed_last = _try_parse(last)
        if parsed_last is not None:
            return parsed_last

    snippet = raw
    if len(snippet) > 200:
        snippet = snippet[:200] + "...(truncated)"
    last_line = lines[-1] if lines else ""
    if len(last_line) > 200:
        last_line = last_line[:200] + "...(truncated)"
    raise ValueError(
        "runner did not emit a clean JSON payload on stdout. "
        f"stdout_snippet={snippet!r} last_non_empty_line={last_line!r}"
    )


def build_result_payload(
    *,
    model_name: str,
    commit: str,
    runner_payload: dict,
    artifact_metrics: dict[str, int],
    output_dir: Path,
    executable_path: Path,
    cflags: list[str],
) -> dict:
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


def get_repo_root() -> Path:
    # benchmarks/ is a top-level package at repo root.
    return Path(__file__).resolve().parents[1]


def get_git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Failed to determine git commit: {stderr}")
    return (result.stdout or "").strip()


def _parse_batch_sizes(tokens: list[str] | None) -> list[int] | None:
    if tokens is None:
        return None
    combined = ",".join(tokens)
    parts = [p for p in combined.replace(" ", ",").split(",") if p]
    sizes: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid batch size: {part!r}") from exc
        if value <= 0:
            raise ValueError(f"Batch size must be > 0, got: {value}")
        sizes.append(value)
    if not sizes:
        return None
    return sizes


def _nearest_rank_percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if pct <= 0.0:
        return sorted_values[0]
    if pct >= 1.0:
        return sorted_values[-1]
    n = len(sorted_values)
    # Nearest-rank: k = ceil(p * n), 1-indexed.
    k = int(math.ceil(pct * n))
    return sorted_values[max(0, min(n - 1, k - 1))]


def _summarize_runs(runner_payload: dict) -> dict:
    runs_in = runner_payload.get("runs", [])
    summarized: list[dict] = []
    for run in runs_in:
        samples = run.get("latency_ms_samples", [])
        if not isinstance(samples, list):
            raise ValueError("runner payload latency_ms_samples must be a list")
        samples_f = [float(x) for x in samples]
        samples_sorted = sorted(samples_f)
        mean = (sum(samples_f) / len(samples_f)) if samples_f else None
        p50 = _nearest_rank_percentile(samples_sorted, 0.50)
        p95 = _nearest_rank_percentile(samples_sorted, 0.95)
        summarized.append(
            {
                "batch_size": int(run["batch_size"]),
                "latency_ms_mean": mean,
                "latency_ms_p50": p50,
                "latency_ms_p95": p95,
                "throughput_samples_per_sec": float(run["throughput_samples_per_sec"]),
                "latency_ms_samples": samples_f,
            }
        )
    return {"model": runner_payload.get("model"), "runs": summarized}


def _default_output_path(*, repo_root: Path, model_name: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = repo_root / "benchmarks" / "results"
    return results_dir / f"{model_name}-{ts}.json"


def _default_build_dir(*, repo_root: Path, model_name: str, output: Path | None) -> Path:
    if output is not None:
        return output.parent / f"{output.stem}_build"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "benchmarks" / "build" / f"{model_name}-{ts}"


def _diff_path_for_output(output_path: Path) -> Path:
    if output_path.suffix == ".json":
        return output_path.with_name(f"{output_path.stem}.diff.json")
    return output_path.with_name(f"{output_path.name}.diff.json")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _benchmark_build_sources(build_dir: Path) -> list[str]:
    sources: list[str] = []
    for path in sorted(build_dir.glob("*.c")):
        if path.name == "test_runner.c":
            continue
        sources.append(str(path))
    return sources


def run_benchmark(
    *,
    model_name: str,
    batch_sizes: list[int] | None = None,
    baseline_result: Path | None = None,
    output: Path | None = None,
) -> tuple[Path, Path | None]:
    repo_root = get_repo_root()
    case = get_benchmark_case(model_name)

    selected_batch_sizes = batch_sizes or list(case.workload_batch_sizes)
    output_path = (output or _default_output_path(repo_root=repo_root, model_name=model_name)).resolve()
    build_dir = _default_build_dir(repo_root=repo_root, model_name=model_name, output=output_path).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    compiler = Compiler(target="x86", opt_level=3)
    model_path = repo_root / case.model_path
    compiler.compile(str(model_path), str(build_dir), entry_point="nnc_run")

    tensors_c_path = build_dir / "tensors.c"
    if not tensors_c_path.exists():
        raise FileNotFoundError(
            f"Required build artifact missing: {tensors_c_path} (cannot compute memory metrics)"
        )
    if not has_memory_layout_defines(tensors_c_path):
        raise ValueError(
            f"Malformed tensors.c: {tensors_c_path} is missing required memory pool defines "
            "(expected NNC_MEMORY_SIZE or NNC_FAST_MEMORY_SIZE and/or NNC_SLOW_MEMORY_SIZE)"
        )

    has_constants = (build_dir / "constants.bin").exists()
    input_tensor_symbols = getattr(case, "input_tensor_symbols", None)

    runner_source = generate_benchmark_runner(
        model_name=model_name,
        workload_batch_sizes=selected_batch_sizes,
        warmup_iterations=case.warmup_iterations,
        measure_iterations=case.measure_iterations,
        entry_point="nnc_run",
        has_constants=has_constants,
        input_tensor_symbols=input_tensor_symbols,
    )
    runner_path = build_dir / "benchmark_runner.c"
    runner_path.write_text(runner_source)

    runtime_include = repo_root / "runtime" / "include"
    runtime_ops = repo_root / "runtime" / "x86" / "ops.c"

    cflags = ["-O3", "-std=c11"]
    source_files = _benchmark_build_sources(build_dir)
    exe_path = build_dir / f"{model_name}_bench"

    build_cmd = [
        "gcc",
        *cflags,
        f"-I{runtime_include}",
        f"-I{build_dir}",
        *source_files,
        str(runtime_ops),
        "-lm",
        "-o",
        str(exe_path),
    ]
    build_result = subprocess.run(
        build_cmd, capture_output=True, text=True, cwd=str(build_dir), timeout=300
    )
    if build_result.returncode != 0:
        raise RuntimeError(
            "Failed to build benchmark executable:\n"
            f"{build_result.stderr}\n{build_result.stdout}"
        )

    run_result = subprocess.run(
        [str(exe_path)],
        capture_output=True,
        text=True,
        cwd=str(build_dir),  # required for constants.bin relative load
        timeout=300,
    )
    if run_result.returncode != 0:
        raise RuntimeError(
            "Benchmark executable failed:\n"
            f"{run_result.stderr}\n{run_result.stdout}"
        )

    runner_raw = parse_runner_output(run_result.stdout)
    runner_summary = _summarize_runs(runner_raw)

    commit = get_git_commit(repo_root)
    artifact_metrics = collect_artifact_metrics(build_dir, exe_path)
    result_payload = build_result_payload(
        model_name=model_name,
        commit=commit,
        runner_payload=runner_summary,
        artifact_metrics=artifact_metrics,
        output_dir=build_dir,
        executable_path=exe_path,
        cflags=cflags,
    )

    _write_json(output_path, result_payload)

    diff_path: Path | None = None
    if baseline_result is not None:
        baseline_payload = json.loads(Path(baseline_result).read_text())
        diff_payload = compare_results(baseline_payload, result_payload)
        diff_path = _diff_path_for_output(output_path)
        _write_json(diff_path, diff_payload)

    return output_path, diff_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NNC generated-C benchmark harness")
    parser.add_argument("--model", default="resnet18", help="Benchmark case name")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        default=None,
        help="Override batch sizes (e.g. --batch-sizes 1 8 16 or --batch-sizes 1,8,16)",
    )
    parser.add_argument(
        "--baseline-result",
        default=None,
        help="Path to a baseline result JSON to compare against",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write result JSON (default: benchmarks/results/...)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    baseline_result = Path(args.baseline_result) if args.baseline_result else None
    output = Path(args.output) if args.output else None

    run_benchmark(
        model_name=args.model,
        batch_sizes=batch_sizes,
        baseline_result=baseline_result,
        output=output,
    )


if __name__ == "__main__":
    main()
