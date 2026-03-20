import re
from pathlib import Path


_MEMORY_DEFINE_RE = r"#define\s+{name}\s+\(?([0-9][0-9_]*)\)?[uUlL]*"


def _extract_defined_size(content: str, name: str) -> int:
    pattern = re.compile(_MEMORY_DEFINE_RE.format(name=re.escape(name)))
    match = pattern.search(content)
    if not match:
        return 0
    numeric = match.group(1).replace("_", "")
    return int(numeric) if numeric else 0


def has_memory_layout_defines(tensors_c_path: Path) -> bool:
    if not tensors_c_path.exists():
        return False
    content = tensors_c_path.read_text()
    return any(
        define in content
        for define in (
            "NNC_MEMORY_SIZE",
            "NNC_FAST_MEMORY_SIZE",
            "NNC_SLOW_MEMORY_SIZE",
        )
    )


def extract_memory_pool_sizes(tensors_c_path: Path) -> dict[str, int]:
    content = tensors_c_path.read_text() if tensors_c_path.exists() else ""
    single_pool_bytes = _extract_defined_size(content, "NNC_MEMORY_SIZE")
    fast_memory_bytes = _extract_defined_size(content, "NNC_FAST_MEMORY_SIZE")
    slow_memory_bytes = _extract_defined_size(content, "NNC_SLOW_MEMORY_SIZE")

    if single_pool_bytes and not fast_memory_bytes and not slow_memory_bytes:
        fast_memory_bytes = single_pool_bytes

    return {
        "fast_memory_bytes": fast_memory_bytes,
        "slow_memory_bytes": slow_memory_bytes,
    }


def collect_artifact_metrics(build_dir: Path, executable_path: Path) -> dict[str, int]:
    if not executable_path.exists():
        raise FileNotFoundError(
            f"Executable path {executable_path} does not exist; cannot measure binary size."
        )

    tensors_c_path = build_dir / "tensors.c"
    pool_sizes = extract_memory_pool_sizes(tensors_c_path)
    constants_path = build_dir / "constants.bin"
    constants_bytes = constants_path.stat().st_size if constants_path.exists() else 0
    binary_size_bytes = executable_path.stat().st_size
    total_static_bytes = (
        pool_sizes["fast_memory_bytes"]
        + pool_sizes["slow_memory_bytes"]
        + constants_bytes
    )

    return {
        **pool_sizes,
        "constants_bytes": constants_bytes,
        "binary_size_bytes": binary_size_bytes,
        "total_static_bytes": total_static_bytes,
    }
