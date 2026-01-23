"""Pytest configuration and fixtures for nnc-py tests."""

from pathlib import Path

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "snapshot: mark test as a snapshot test")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-snapshot",
        action="store_true",
        default=False,
        help="Run snapshot tests (they are skipped by default in CI unless explicitly enabled)",
    )


@pytest.fixture
def models_dir():
    """Path to the directory containing classic ONNX models."""
    return Path(__file__).parent.parent / "models"


@pytest.fixture
def snapshot_dir():
    """Path to the snapshot directory."""
    return Path(__file__).parent / "snapshots"
