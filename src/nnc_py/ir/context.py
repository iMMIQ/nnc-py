"""Compilation context for passing information through the pipeline."""

from dataclasses import dataclass, field
from typing import Any

from nnc_py.ir.graph import Graph


@dataclass
class CompileContext:
    """Compilation context - passes information through the compilation pipeline."""

    graph: Graph
    target: str  # "x86" or "npu"
    optimization_level: int = 0
    debug: bool = False

    # Symbol tables
    tensor_symbols: dict[str, str] = field(default_factory=dict)  # ONNX -> C
    node_symbols: dict[str, str] = field(default_factory=dict)  # ONNX -> C

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if self.tensor_symbols is None:
            self.tensor_symbols = {}
        if self.node_symbols is None:
            self.node_symbols = {}
        if self.metadata is None:
            self.metadata = {}
