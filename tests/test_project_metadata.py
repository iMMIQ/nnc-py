"""Project metadata and test configuration consistency checks."""

from pathlib import Path
import tomllib


def test_pyproject_uses_syrupy_tool_section():
    """Snapshot configuration should use the syrupy tool section name."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text())

    assert "tool" in data
    assert "syrupy" in data["tool"]


def test_snapshot_directory_matches_pyproject_config():
    """Snapshot configuration should point at the actual snapshot directory."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text())

    config = data["tool"]["syrupy"]
    snapshots_path = Path(config["snapshots_path"])

    assert snapshots_path == Path("tests/__snapshots__")
    assert (Path(__file__).parent.parent / snapshots_path).exists()
