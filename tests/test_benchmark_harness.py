import json
from pathlib import Path

import subprocess
import pytest

from benchmarks.harness import build_result_payload, parse_runner_output, run_benchmark


def test_parse_runner_output_returns_json_payload():
    payload = parse_runner_output('{"model":"resnet18","runs":[{"batch_size":1}]}')
    assert payload["model"] == "resnet18"
    assert payload["runs"][0]["batch_size"] == 1


def test_parse_runner_output_extracts_last_non_empty_line_when_stdout_is_noisy():
    stdout = "some log line\n\n" + '{"model":"resnet18","runs":[{"batch_size":1}]}' + "\n"
    payload = parse_runner_output(stdout)
    assert payload["model"] == "resnet18"
    assert payload["runs"][0]["batch_size"] == 1


def test_parse_runner_output_raises_value_error_on_invalid_stdout():
    with pytest.raises(ValueError) as excinfo:
        parse_runner_output("not json at all\n")

    msg = str(excinfo.value).lower()
    assert "json" in msg
    assert "stdout" in msg


def test_build_result_payload_includes_commit_memory_and_runs(tmp_path):
    runner_payload = {
        "model": "resnet18",
        "runs": [
            {
                "batch_size": 1,
                "latency_ms_mean": 1.0,
                "latency_ms_p50": 1.0,
                "latency_ms_p95": 1.0,
                "throughput_samples_per_sec": 100.0,
            }
        ],
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


def test_harness_writes_result_and_diff_files(tmp_path, monkeypatch):
    # Arrange: baseline result file.
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
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
        )
    )

    # Arrange: monkeypatch Compiler.compile to drop minimal artifacts into output_dir.
    def fake_compile(self, onnx_path: str, output_dir: str, entry_point: str = "nnc_run", **kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # These files are discovered/used by the harness (metrics + build source enumeration).
        (out / "model.c").write_text("int dummy_model_c(void){return 0;}\n")
        (out / "tensors.c").write_text("#define NNC_FAST_MEMORY_SIZE 256\n")
        (out / "model.h").write_text("void nnc_run(void);\n")
        (out / "constants.bin").write_bytes(b"12345678")

    import benchmarks.harness as harness

    monkeypatch.setattr(harness.Compiler, "compile", fake_compile, raising=True)
    monkeypatch.setattr(harness, "get_git_commit", lambda repo_root: "abc1234", raising=True)

    # Arrange: monkeypatch subprocess.run for gcc + benchmark execution.
    runner_stdout = json.dumps(
        {
            "model": "resnet18",
            "runs": [
                {
                    "batch_size": 1,
                    "latency_ms_samples": [1.0, 1.0, 1.0],
                    "throughput_samples_per_sec": 100.0,
                }
            ],
        }
    )

    def fake_run(cmd, **kwargs):
        # gcc build step: succeed and create the output executable the harness expects.
        if isinstance(cmd, list) and cmd and cmd[0] == "gcc":
            if "-o" in cmd:
                exe_path = Path(cmd[cmd.index("-o") + 1])
                exe_path.write_bytes(b"binary")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        # benchmark execution: return JSON on stdout
        if isinstance(cmd, list) and cmd and str(cmd[0]).endswith("_bench"):
            return subprocess.CompletedProcess(cmd, 0, stdout=runner_stdout, stderr="")

        raise AssertionError(f"unexpected subprocess.run call: {cmd!r}")

    monkeypatch.setattr(harness.subprocess, "run", fake_run, raising=True)

    # Act
    output_path = tmp_path / "result.json"
    result_path, diff_path = run_benchmark(
        model_name="resnet18",
        batch_sizes=[1],
        baseline_result=baseline_path,
        output=output_path,
    )

    # Assert
    assert result_path == output_path
    assert result_path.exists()
    payload = json.loads(result_path.read_text())
    assert payload["model"] == "resnet18"
    assert payload["commit"] == "abc1234"
    assert payload["compiler_config"]["opt_level"] == 3
    assert payload["runs"][0]["batch_size"] == 1
    assert "memory" in payload and payload["memory"]["total_static_bytes"] > 0

    assert diff_path is not None
    assert diff_path.exists()
    diff_payload = json.loads(diff_path.read_text())
    assert diff_payload["baseline_commit"] == "base123"
    assert diff_payload["candidate_commit"] == "abc1234"


def test_harness_accepts_single_pool_tensors_layout(tmp_path, monkeypatch):
    import benchmarks.harness as harness

    def fake_compile_single_pool(self, onnx_path: str, output_dir: str, entry_point: str = "nnc_run", **kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.c").write_text("int dummy_model_c(void){return 0;}\n")
        (out / "model.h").write_text("void nnc_run(void);\n")
        (out / "tensors.c").write_text("#define NNC_MEMORY_SIZE 512\n")
        (out / "constants.bin").write_bytes(b"1234")

    monkeypatch.setattr(harness.Compiler, "compile", fake_compile_single_pool, raising=True)
    monkeypatch.setattr(harness, "get_git_commit", lambda repo_root: "abc1234", raising=True)

    runner_stdout = json.dumps(
        {
            "model": "resnet18",
            "runs": [
                {
                    "batch_size": 1,
                    "latency_ms_samples": [1.0, 2.0],
                    "throughput_samples_per_sec": 100.0,
                }
            ],
        }
    )

    def fake_run(cmd, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == "gcc":
            if "-o" in cmd:
                exe_path = Path(cmd[cmd.index("-o") + 1])
                exe_path.write_bytes(b"binary")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if isinstance(cmd, list) and cmd and str(cmd[0]).endswith("_bench"):
            return subprocess.CompletedProcess(cmd, 0, stdout=runner_stdout, stderr="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd!r}")

    monkeypatch.setattr(harness.subprocess, "run", fake_run, raising=True)

    result_path, diff_path = run_benchmark(
        model_name="resnet18",
        batch_sizes=[1],
        output=tmp_path / "single-pool.json",
    )

    assert diff_path is None
    payload = json.loads(result_path.read_text())
    assert payload["memory"]["fast_memory_bytes"] == 512
    assert payload["memory"]["slow_memory_bytes"] == 0
    assert payload["memory"]["total_static_bytes"] == 516


def test_harness_excludes_compiler_test_runner_from_build(tmp_path, monkeypatch):
    import benchmarks.harness as harness

    def fake_compile_with_test_runner(
        self, onnx_path: str, output_dir: str, entry_point: str = "nnc_run", **kwargs
    ):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.c").write_text("int dummy_model_c(void){return 0;}\n")
        (out / "model.h").write_text("void nnc_run(void);\n")
        (out / "tensors.c").write_text("#define NNC_MEMORY_SIZE 512\n")
        (out / "test_runner.c").write_text("int main(void){return 0;}\n")

    monkeypatch.setattr(
        harness.Compiler, "compile", fake_compile_with_test_runner, raising=True
    )
    monkeypatch.setattr(harness, "get_git_commit", lambda repo_root: "abc1234", raising=True)

    runner_stdout = json.dumps(
        {
            "model": "resnet18",
            "runs": [
                {
                    "batch_size": 1,
                    "latency_ms_samples": [1.0],
                    "throughput_samples_per_sec": 100.0,
                }
            ],
        }
    )
    observed_build_cmd: list[str] = []

    def fake_run(cmd, **kwargs):
        nonlocal observed_build_cmd
        if isinstance(cmd, list) and cmd and cmd[0] == "gcc":
            observed_build_cmd = cmd
            assert any(str(arg).endswith("benchmark_runner.c") for arg in cmd)
            assert not any(str(arg).endswith("test_runner.c") for arg in cmd)
            if "-o" in cmd:
                exe_path = Path(cmd[cmd.index("-o") + 1])
                exe_path.write_bytes(b"binary")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if isinstance(cmd, list) and cmd and str(cmd[0]).endswith("_bench"):
            return subprocess.CompletedProcess(cmd, 0, stdout=runner_stdout, stderr="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd!r}")

    monkeypatch.setattr(harness.subprocess, "run", fake_run, raising=True)

    run_benchmark(
        model_name="resnet18",
        batch_sizes=[1],
        output=tmp_path / "exclude-test-runner.json",
    )

    assert observed_build_cmd


def test_harness_fails_clearly_when_tensors_c_is_missing(tmp_path, monkeypatch):
    import benchmarks.harness as harness

    def fake_compile_missing_tensors(self, onnx_path: str, output_dir: str, entry_point: str = "nnc_run", **kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.c").write_text("int dummy_model_c(void){return 0;}\n")
        (out / "model.h").write_text("void nnc_run(void);\n")

    monkeypatch.setattr(harness.Compiler, "compile", fake_compile_missing_tensors, raising=True)
    monkeypatch.setattr(harness, "get_git_commit", lambda repo_root: "abc1234", raising=True)

    def fake_run(cmd, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == "gcc":
            if "-o" in cmd:
                exe_path = Path(cmd[cmd.index("-o") + 1])
                exe_path.write_bytes(b"binary")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd!r}")

    monkeypatch.setattr(harness.subprocess, "run", fake_run, raising=True)

    with pytest.raises(FileNotFoundError) as excinfo:
        run_benchmark(model_name="resnet18", batch_sizes=[1], output=tmp_path / "out.json")

    assert "tensors.c" in str(excinfo.value)


def test_harness_fails_clearly_when_tensors_c_is_malformed(tmp_path, monkeypatch):
    import benchmarks.harness as harness

    def fake_compile_malformed_tensors(self, onnx_path: str, output_dir: str, entry_point: str = "nnc_run", **kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.c").write_text("int dummy_model_c(void){return 0;}\n")
        (out / "model.h").write_text("void nnc_run(void);\n")
        (out / "tensors.c").write_text("/* missing pool macros */\n")

    monkeypatch.setattr(harness.Compiler, "compile", fake_compile_malformed_tensors, raising=True)
    monkeypatch.setattr(harness, "get_git_commit", lambda repo_root: "abc1234", raising=True)

    def fake_run(cmd, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == "gcc":
            if "-o" in cmd:
                exe_path = Path(cmd[cmd.index("-o") + 1])
                exe_path.write_bytes(b"binary")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd!r}")

    monkeypatch.setattr(harness.subprocess, "run", fake_run, raising=True)

    with pytest.raises(ValueError) as excinfo:
        run_benchmark(model_name="resnet18", batch_sizes=[1], output=tmp_path / "out.json")

    assert "tensors.c" in str(excinfo.value).lower()
