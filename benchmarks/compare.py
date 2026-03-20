from __future__ import annotations


def _pct_delta(baseline_value: float | int | None, candidate_value: float | int | None) -> float | None:
    """
    Compute percentage delta for candidate vs baseline.

    Robustness requirements:
    - If baseline is 0, delta is undefined; return None instead of raising.
    - If either value is None, return None.
    """
    if baseline_value is None or candidate_value is None:
        return None
    try:
        baseline_f = float(baseline_value)
        candidate_f = float(candidate_value)
    except (TypeError, ValueError):
        return None
    if baseline_f == 0.0:
        return None
    return round(((candidate_f - baseline_f) / baseline_f) * 100.0, 4)


def _validate_payload(payload: dict, label: str) -> tuple[str, list[dict], dict]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} payload must be a dictionary")
    if "commit" not in payload:
        raise ValueError(f"{label} payload is missing 'commit'")
    if "runs" not in payload:
        raise ValueError(f"{label} payload is missing 'runs'")
    if "memory" not in payload:
        raise ValueError(f"{label} payload is missing 'memory'")

    runs = payload["runs"]
    if not isinstance(runs, list):
        raise ValueError(f"{label} runs entry must be a list")

    memory = payload["memory"]
    if not isinstance(memory, dict):
        raise ValueError(f"{label} memory entry must be a dictionary")
    if "total_static_bytes" not in memory:
        raise ValueError(f"{label} memory is missing 'total_static_bytes'")

    return payload["commit"], runs, memory


def _build_run_map(runs: list[dict], label: str) -> dict[int, dict]:
    run_map: dict[int, dict] = {}
    seen: set[int] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"{label} run at index {index} must be a dictionary")
        if "batch_size" not in run:
            raise ValueError(f"{label} run at index {index} is missing 'batch_size'")
        if "latency_ms_p50" not in run:
            raise ValueError(f"{label} run at index {index} is missing 'latency_ms_p50'")
        if "throughput_samples_per_sec" not in run:
            raise ValueError(
                f"{label} run at index {index} is missing 'throughput_samples_per_sec'"
            )

        batch_size = run["batch_size"]
        if not isinstance(batch_size, int):
            raise ValueError(f"{label} batch_size must be an integer at index {index}")
        if batch_size in seen:
            raise ValueError(f"duplicate batch_size {batch_size} in {label} runs")
        seen.add(batch_size)

        run_map[batch_size] = run

    return run_map


def compare_results(baseline: dict, candidate: dict) -> dict:
    baseline_commit, baseline_runs_raw, baseline_memory = _validate_payload(baseline, "baseline")
    candidate_commit, candidate_runs_raw, candidate_memory = _validate_payload(
        candidate, "candidate"
    )

    baseline_runs = _build_run_map(baseline_runs_raw, "baseline")
    candidate_runs = _build_run_map(candidate_runs_raw, "candidate")

    if baseline_runs.keys() != candidate_runs.keys():
        raise ValueError(
            "baseline and candidate batch_size sets must match: "
            f"{sorted(baseline_runs.keys())} vs {sorted(candidate_runs.keys())}"
        )

    run_diffs: list[dict] = []
    for batch_size in sorted(baseline_runs.keys()):
        base_run = baseline_runs[batch_size]
        cand_run = candidate_runs[batch_size]
        run_diffs.append(
            {
                "batch_size": batch_size,
                "latency_delta_pct": _pct_delta(
                    base_run["latency_ms_p50"], cand_run["latency_ms_p50"]
                ),
                "throughput_delta_pct": _pct_delta(
                    base_run["throughput_samples_per_sec"],
                    cand_run["throughput_samples_per_sec"],
                ),
            }
        )

    base_mem = baseline_memory.get("total_static_bytes")
    cand_mem = candidate_memory.get("total_static_bytes")
    mem_delta: int | None
    if base_mem is None or cand_mem is None:
        mem_delta = None
    else:
        try:
            mem_delta = int(cand_mem) - int(base_mem)
        except (TypeError, ValueError):
            mem_delta = None

    return {
        "baseline_commit": baseline_commit,
        "candidate_commit": candidate_commit,
        "runs": run_diffs,
        "memory": {
            "total_static_bytes_delta": mem_delta
        },
    }
