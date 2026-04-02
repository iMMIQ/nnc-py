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


def test_joint_schedule_modules_are_importable():
    import nnc_py.joint_schedule as joint_schedule
    from nnc_py.passes import JointTilingScheduleProblemPass

    assert callable(joint_schedule.build_joint_problem)
    assert callable(joint_schedule.build_joint_regions)
    assert JointTilingScheduleProblemPass.__name__ == "JointTilingScheduleProblemPass"


def test_pyproject_metadata_has_no_placeholder_author_or_urls():
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text())

    author = data["project"]["authors"][0]
    urls = data["project"]["urls"]

    assert author["name"] != "Your Name"
    assert author["email"] != "your.email@example.com"
    assert urls["Homepage"] == "https://github.com/iMMIQ/nnc-py"
    assert urls["Repository"] == "https://github.com/iMMIQ/nnc-py"
    assert urls["Issues"] == "https://github.com/iMMIQ/nnc-py/issues"


def test_readme_has_no_deleted_config_entry_or_clone_placeholder():
    readme = (Path(__file__).parent.parent / "README.md").read_text()

    assert "<repository-url>" not in readme
    assert "config.py           # Configuration management" not in readme
