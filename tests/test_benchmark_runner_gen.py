import re

import pytest

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
    assert (
        "for (int batch_iter = 0; batch_iter < workload_batch; batch_iter++)"
        in source
    )
    assert "throughput_samples_per_sec" in source


def test_runner_enables_posix_clock_gettime_before_time_header():
    source = generate_benchmark_runner(
        model_name="resnet18",
        workload_batch_sizes=[1],
        warmup_iterations=1,
        measure_iterations=1,
        entry_point="nnc_run",
    )

    feature_macro = "#define _POSIX_C_SOURCE 200809L"
    time_include = "#include <time.h>"
    assert feature_macro in source
    assert time_include in source
    assert source.index(feature_macro) < source.index(time_include)


def test_runner_loads_constants_when_requested():
    source = generate_benchmark_runner(
        model_name="resnet18",
        workload_batch_sizes=[1],
        warmup_iterations=1,
        measure_iterations=2,
        entry_point="nnc_run",
        has_constants=True,
    )

    assert 'nnc_load_constants("constants.bin")' in source


def test_invalid_entry_point_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        generate_benchmark_runner(
            model_name="resnet18",
            workload_batch_sizes=[1],
            warmup_iterations=1,
            measure_iterations=1,
            entry_point="nnc_run(); /* injected */",
        )

    msg = str(excinfo.value)
    assert "entry_point" in msg
    assert "identifier" in msg.lower()


def test_invalid_args_are_rejected():
    with pytest.raises(ValueError):
        generate_benchmark_runner(
            model_name="resnet18",
            workload_batch_sizes=[],
            warmup_iterations=0,
            measure_iterations=0,
            entry_point="nnc_run",
        )

    with pytest.raises(ValueError):
        generate_benchmark_runner(
            model_name="resnet18",
            workload_batch_sizes=[1],
            warmup_iterations=-1,
            measure_iterations=0,
            entry_point="nnc_run",
        )

    with pytest.raises(ValueError):
        generate_benchmark_runner(
            model_name="resnet18",
            workload_batch_sizes=[0],
            warmup_iterations=0,
            measure_iterations=0,
            entry_point="nnc_run",
        )


def test_runner_escapes_unsafe_model_name_safely_in_generated_source():
    model_name = 'resnet18"\n\\\\weird'
    source = generate_benchmark_runner(
        model_name=model_name,
        workload_batch_sizes=[1],
        warmup_iterations=1,
        measure_iterations=1,
        entry_point="nnc_run",
    )

    # Raw model_name must not be spliced into the C string literal (would break compilation).
    assert model_name not in source

    # We generate a JSON-string literal at build time and print it with %s.
    assert "static const char* model_json" in source
    assert "printf" in source and "model_json" in source

    # Should include escapes consistent with JSON + C embedding.
    # Expect escaped quote and escaped newline in the generated C string literal content.
    assert re.search(r'model_json\s*=\s*".*\\\\\\".*\\\\n', source)


def test_configurable_input_tensor_symbols_reflected_in_generated_code():
    source = generate_benchmark_runner(
        model_name="resnet18",
        workload_batch_sizes=[1],
        warmup_iterations=1,
        measure_iterations=1,
        entry_point="nnc_run",
        input_tensor_symbols=["my_input0", "my_input1"],
    )

    assert "&my_input0" in source
    assert "&my_input1" in source


def test_invalid_input_tensor_symbol_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        generate_benchmark_runner(
            model_name="resnet18",
            workload_batch_sizes=[1],
            warmup_iterations=1,
            measure_iterations=1,
            entry_point="nnc_run",
            input_tensor_symbols=["bad-name"],
        )
    assert "input_tensor_symbols" in str(excinfo.value)


def test_runner_avoids_trailing_comma_in_runs_json():
    source = generate_benchmark_runner(
        model_name="resnet18",
        workload_batch_sizes=[1, 8, 16],
        warmup_iterations=1,
        measure_iterations=2,
        entry_point="nnc_run",
    )

    # We can’t execute C here, but we can assert the generator includes
    # a comma guard so the emitted JSON is valid.
    assert "if (workload_idx > 0)" in source


def test_runner_emits_latency_samples_array():
    source = generate_benchmark_runner(
        model_name="resnet18",
        workload_batch_sizes=[1],
        warmup_iterations=1,
        measure_iterations=3,
        entry_point="nnc_run",
    )

    assert "latency_ms_samples" in source


def test_runner_includes_float32_dtype_guard_for_deterministic_init():
    source = generate_benchmark_runner(
        model_name="resnet18",
        workload_batch_sizes=[1],
        warmup_iterations=1,
        measure_iterations=1,
        entry_point="nnc_run",
    )

    # Deterministic init should only write for float32 tensors.
    assert "NNC_DTYPE_FLOAT32" in source
    assert "t->dtype" in source


def test_runner_emits_final_json_only_after_measurements_and_no_error_json():
    source = generate_benchmark_runner(
        model_name="resnet18",
        workload_batch_sizes=[1, 8],
        warmup_iterations=1,
        measure_iterations=2,
        entry_point="nnc_run",
    )

    # No stdout JSON should be printed as part of timing failure handling.
    assert "clock_gettime_failed" not in source

    # The generator should collect measurements first, then emit JSON.
    collect_idx = source.index("/* Collect measurements */")
    emit_idx = source.index("/* Emit JSON */")
    assert emit_idx > collect_idx


def test_runner_handles_measure_iterations_zero_without_malloc0_failure_check():
    source = generate_benchmark_runner(
        model_name="resnet18",
        workload_batch_sizes=[1],
        warmup_iterations=1,
        measure_iterations=0,
        entry_point="nnc_run",
    )

    # The generated C should not treat malloc(0) returning NULL as an allocation failure.
    assert "lat_samples_count" in source
    assert "lat_samples_count > 0 && !lat_samples" in source

    # Still emits the expected JSON shape (empty samples array for each run).
    assert "latency_ms_samples" in source
