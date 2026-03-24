from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.metrics import extract_memory_pool_sizes, has_memory_layout_defines
from nnc_py import Compiler
from test_snapshots_resnet18 import require_resnet18_model_path


def compile_resnet18_with_tiled_pipeline(
    *,
    max_memory: str,
    opt_level: int = 3,
) -> dict[str, object]:
    model_path = require_resnet18_model_path()

    with TemporaryDirectory(prefix="resnet18-budget-") as tmpdir:
        build_dir = Path(tmpdir)
        compiler = Compiler(target="x86", opt_level=opt_level)

        try:
            compiler.compile(str(model_path), str(build_dir), max_memory=max_memory)
        except Exception as exc:
            return {
                "compiled": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "exception": exc,
                "artifact_files": sorted(path.name for path in build_dir.iterdir()),
            }

        tensors_c = build_dir / "tensors.c"
        if not has_memory_layout_defines(tensors_c):
            raise AssertionError(
                f"Compiled resnet18 artifacts are missing memory layout defines: {tensors_c}"
            )

        return {
            "compiled": True,
            "metrics": extract_memory_pool_sizes(tensors_c),
            "artifact_files": sorted(path.name for path in build_dir.iterdir()),
        }


def compile_and_measure_resnet18(*, max_memory: str, opt_level: int = 3) -> dict[str, int]:
    report = compile_resnet18_with_tiled_pipeline(
        max_memory=max_memory,
        opt_level=opt_level,
    )
    if not report["compiled"]:
        raise report["exception"]
    return report["metrics"]


def test_resnet18_o3_tiled_plan_reports_fast_memory_within_1mb():
    metrics = compile_and_measure_resnet18(max_memory="1M", opt_level=3)
    assert metrics["fast_memory_bytes"] <= 1024 * 1024


def test_resnet18_o3_tiled_codegen_builds_under_memory_budget():
    report = compile_resnet18_with_tiled_pipeline(max_memory="1M", opt_level=3)
    assert report["compiled"] is True
