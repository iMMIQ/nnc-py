"""Pytest configuration and fixtures for nnc-py tests."""

from pathlib import Path
import sys

import pytest
from nnc_py.pattern.registry import PatternRegistry

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SNAPSHOT_PREFIX = "test_snapshots_"
_INTEGRATION_FILENAMES = {
    "test_debug_mode.py",
    "test_memory_safety.py",
    "test_numerical_accuracy.py",
    "test_reload_code_generation.py",
    "test_reproduce_issue.py",
    "test_spill_correctness.py",
    "test_tiled_runtime_correctness.py",
}


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "snapshot: mark test as a snapshot test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-snapshot",
        action="store_true",
        default=False,
        help="Run snapshot tests (they are skipped by default in CI unless explicitly enabled)",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that compile or execute generated artifacts",
    )


def _is_snapshot_test(path: Path) -> bool:
    return path.name.startswith(_SNAPSHOT_PREFIX)


def _is_integration_test(path: Path, test_name: str) -> bool:
    if "integration" in path.parts or "e2e" in path.stem:
        return True
    if path.name in _INTEGRATION_FILENAMES:
        return True
    return _is_snapshot_test(path) and "runtime" in test_name


def pytest_collection_modifyitems(config, items):
    """Mark and gate expensive test groups behind explicit flags."""
    run_snapshot = config.getoption("--run-snapshot")
    run_integration = config.getoption("--run-integration")
    skip_snapshot = pytest.mark.skip(
        reason="snapshot tests are skipped by default; use --run-snapshot"
    )
    skip_integration = pytest.mark.skip(
        reason="integration tests are skipped by default; use --run-integration"
    )

    for item in items:
        path = Path(str(item.fspath)).resolve().relative_to(ROOT)
        is_snapshot = _is_snapshot_test(path)
        is_integration = _is_integration_test(path, item.name)

        if is_snapshot:
            item.add_marker("snapshot")
            if not run_snapshot:
                item.add_marker(skip_snapshot)
                continue

        if is_integration:
            item.add_marker("integration")
            if not run_integration:
                item.add_marker(skip_integration)


@pytest.fixture
def models_dir():
    """Path to the directory containing classic ONNX models."""
    return Path(__file__).parent.parent / "models"


@pytest.fixture
def snapshot_dir():
    """Path to the snapshot directory."""
    return Path(__file__).parent / "__snapshots__"


@pytest.fixture(autouse=True)
def restore_pattern_registry():
    """Keep global fusion-pattern registry state isolated per test."""
    snapshot = PatternRegistry.snapshot()
    try:
        yield
    finally:
        PatternRegistry.restore(snapshot)
