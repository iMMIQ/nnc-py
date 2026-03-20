import pytest

from benchmarks.compare import compare_results


def test_compare_results_reports_latency_and_memory_deltas():
    baseline = {
        "commit": "base123",
        "runs": [
            {
                "batch_size": 1,
                "latency_ms_p50": 10.0,
                "throughput_samples_per_sec": 100.0,
            }
        ],
        "memory": {"total_static_bytes": 1000},
    }
    candidate = {
        "commit": "cand456",
        "runs": [
            {
                "batch_size": 1,
                "latency_ms_p50": 8.0,
                "throughput_samples_per_sec": 120.0,
            }
        ],
        "memory": {"total_static_bytes": 900},
    }

    diff = compare_results(baseline, candidate)
    assert diff["baseline_commit"] == "base123"
    assert diff["candidate_commit"] == "cand456"
    assert diff["runs"][0]["batch_size"] == 1
    assert diff["runs"][0]["latency_delta_pct"] == -20.0
    assert diff["runs"][0]["throughput_delta_pct"] == 20.0
    assert diff["memory"]["total_static_bytes_delta"] == -100


def test_compare_results_orders_runs_by_batch_size():
    baseline = {
        "commit": "base123",
        "runs": [
            {
                "batch_size": 16,
                "latency_ms_p50": 40.0,
                "throughput_samples_per_sec": 250.0,
            },
            {
                "batch_size": 8,
                "latency_ms_p50": 30.0,
                "throughput_samples_per_sec": 300.0,
            },
        ],
        "memory": {"total_static_bytes": 800},
    }
    candidate = {
        "commit": "cand456",
        "runs": [
            {
                "batch_size": 8,
                "latency_ms_p50": 20.0,
                "throughput_samples_per_sec": 400.0,
            },
            {
                "batch_size": 16,
                "latency_ms_p50": 32.0,
                "throughput_samples_per_sec": 310.0,
            },
        ],
        "memory": {"total_static_bytes": 700},
    }

    diff = compare_results(baseline, candidate)
    assert [run["batch_size"] for run in diff["runs"]] == [8, 16]


def test_compare_results_rejects_mismatched_batch_sets():
    baseline = {
        "commit": "base",
        "runs": [
            {
                "batch_size": 1,
                "latency_ms_p50": 10.0,
                "throughput_samples_per_sec": 10.0,
            }
        ],
        "memory": {"total_static_bytes": 1},
    }
    candidate = {
        "commit": "cand",
        "runs": [
            {
                "batch_size": 8,
                "latency_ms_p50": 10.0,
                "throughput_samples_per_sec": 10.0,
            }
        ],
        "memory": {"total_static_bytes": 1},
    }

    with pytest.raises(ValueError, match="batch_size"):
        compare_results(baseline, candidate)


def test_compare_results_baseline_zero_latency_error():
    baseline = {
        "commit": "base",
        "runs": [
            {
                "batch_size": 1,
                "latency_ms_p50": 0.0,
                "throughput_samples_per_sec": 1.0,
            }
        ],
        "memory": {"total_static_bytes": 1},
    }
    candidate = {
        "commit": "cand",
        "runs": [
            {
                "batch_size": 1,
                "latency_ms_p50": 1.0,
                "throughput_samples_per_sec": 2.0,
            }
        ],
        "memory": {"total_static_bytes": 1},
    }

    diff = compare_results(baseline, candidate)
    assert diff["runs"][0]["latency_delta_pct"] is None
    assert diff["runs"][0]["throughput_delta_pct"] == 100.0


@pytest.mark.parametrize("label", ["baseline", "candidate"])
def test_compare_results_detects_duplicate_batch_sizes(label):
    runs = [
        {
            "batch_size": 1,
            "latency_ms_p50": 10.0,
            "throughput_samples_per_sec": 10.0,
        },
        {
            "batch_size": 1,
            "latency_ms_p50": 11.0,
            "throughput_samples_per_sec": 11.0,
        },
    ]

    baseline = {
        "commit": "base",
        "runs": runs if label == "baseline" else [{"batch_size": 1, "latency_ms_p50": 10.0, "throughput_samples_per_sec": 10.0}],
        "memory": {"total_static_bytes": 1},
    }
    candidate = {
        "commit": "cand",
        "runs": runs if label == "candidate" else [{"batch_size": 1, "latency_ms_p50": 10.0, "throughput_samples_per_sec": 10.0}],
        "memory": {"total_static_bytes": 1},
    }

    with pytest.raises(ValueError, match="duplicate batch_size"):
        compare_results(baseline, candidate)


def test_compare_results_returns_none_delta_when_baseline_percentile_is_none():
    baseline = {
        "commit": "base",
        "runs": [{"batch_size": 1, "latency_ms_p50": None, "throughput_samples_per_sec": 10.0}],
        "memory": {"total_static_bytes": 10},
    }
    candidate = {
        "commit": "cand",
        "runs": [{"batch_size": 1, "latency_ms_p50": 1.0, "throughput_samples_per_sec": 12.0}],
        "memory": {"total_static_bytes": 8},
    }

    diff = compare_results(baseline, candidate)
    assert diff["runs"][0]["latency_delta_pct"] is None
    assert diff["runs"][0]["throughput_delta_pct"] == 20.0


def test_compare_results_allows_empty_runs_and_returns_stable_shape():
    baseline = {"commit": "base", "runs": [], "memory": {"total_static_bytes": 100}}
    candidate = {"commit": "cand", "runs": [], "memory": {"total_static_bytes": 80}}

    diff = compare_results(baseline, candidate)
    assert diff["baseline_commit"] == "base"
    assert diff["candidate_commit"] == "cand"
    assert diff["runs"] == []
    assert diff["memory"]["total_static_bytes_delta"] == -20
