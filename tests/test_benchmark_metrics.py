import pytest

from benchmarks.metrics import (
    collect_artifact_metrics,
    extract_memory_pool_sizes,
    has_memory_layout_defines,
)


def test_extract_memory_pool_sizes(tmp_path):
    tensors_c = tmp_path / "tensors.c"
    tensors_c.write_text(
        "#define NNC_FAST_MEMORY_SIZE 4096\n"
        "#define NNC_SLOW_MEMORY_SIZE 1024\n"
    )

    sizes = extract_memory_pool_sizes(tensors_c)
    assert sizes["fast_memory_bytes"] == 4096
    assert sizes["slow_memory_bytes"] == 1024


def test_extract_memory_pool_sizes_supports_single_pool_layout(tmp_path):
    tensors_c = tmp_path / "tensors.c"
    tensors_c.write_text("#define NNC_MEMORY_SIZE 61081920\n")

    sizes = extract_memory_pool_sizes(tensors_c)
    assert sizes["fast_memory_bytes"] == 61081920
    assert sizes["slow_memory_bytes"] == 0


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


def test_collect_artifact_metrics_supports_single_pool_layout(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "tensors.c").write_text("#define NNC_MEMORY_SIZE 512\n")
    (build_dir / "constants.bin").write_bytes(b"1234")
    exe = build_dir / "resnet18_bench"
    exe.write_bytes(b"binary")

    metrics = collect_artifact_metrics(build_dir, exe)
    assert metrics["fast_memory_bytes"] == 512
    assert metrics["slow_memory_bytes"] == 0
    assert metrics["constants_bytes"] == 4
    assert metrics["binary_size_bytes"] == 6
    assert metrics["total_static_bytes"] == 516


def test_benchmark_metrics_reads_tile_and_scratch_pool_sizes(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "tensors.c").write_text(
        "#define NNC_FAST_MEMORY_SIZE 524288\n"
        "#define NNC_TILE_MEMORY_SIZE 262144\n"
        "#define NNC_SCRATCH_MEMORY_SIZE 131072\n"
    )
    exe = build_dir / "resnet18_bench"
    exe.write_bytes(b"binary")

    metrics = collect_artifact_metrics(build_dir, exe)
    assert metrics["fast_memory_bytes"] == 524288
    assert metrics["tile_memory_bytes"] == 262144
    assert metrics["scratch_memory_bytes"] == 131072
    assert metrics["total_static_bytes"] == 524288


def test_benchmark_metrics_collects_generic_logical_region_sizes(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "tensors.c").write_text(
        "#define NNC_FAST_MEMORY_SIZE 524288\n"
        "#define NNC_TILE_MEMORY_SIZE 262144\n"
        "#define NNC_SCRATCH_MEMORY_SIZE 131072\n"
        "#define NNC_PERSISTENT_MEMORY_SIZE 65536\n"
    )
    exe = build_dir / "resnet18_bench"
    exe.write_bytes(b"binary")

    metrics = collect_artifact_metrics(build_dir, exe)
    assert metrics["tile_memory_bytes"] == 262144
    assert metrics["scratch_memory_bytes"] == 131072
    assert metrics["logical_region_bytes"] == {
        "tile": 262144,
        "scratch": 131072,
        "persistent": 65536,
    }


def test_has_memory_layout_defines_accepts_tiled_region_metadata(tmp_path):
    tensors_c = tmp_path / "tensors.c"
    tensors_c.write_text(
        "#define NNC_FAST_MEMORY_SIZE 524288\n"
        "#define NNC_TILE_MEMORY_SIZE 262144\n"
    )

    assert has_memory_layout_defines(tensors_c) is True


def test_extract_memory_pool_sizes_missing_tensors(tmp_path):
    sizes = extract_memory_pool_sizes(tmp_path / "missing_tensors.c")
    assert sizes["fast_memory_bytes"] == 0
    assert sizes["slow_memory_bytes"] == 0


def test_collect_artifact_metrics_missing_executable(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "tensors.c").write_text("#define NNC_FAST_MEMORY_SIZE 128\n")
    exe = build_dir / "missing_bin"

    with pytest.raises(FileNotFoundError) as exc:
        collect_artifact_metrics(build_dir, exe)

    assert "executable path" in str(exc.value).lower()
