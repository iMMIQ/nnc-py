"""Base classes for code generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CodeArtifact:
    """A generated code artifact."""

    filename: str
    content: str | bytes
    file_type: str  # "source" or "header" or "build" or "binary"


@dataclass
class CodeGenResult:
    """Result of code generation."""

    files: list[CodeArtifact] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_file(self, filename: str, content: str | bytes, file_type: str = "source"):
        """Add a generated file."""
        self.files.append(CodeArtifact(filename, content, file_type))


class BackendBase(ABC):
    """Base class for code generation backends."""

    @abstractmethod
    def generate(self, ctx) -> CodeGenResult:
        """Generate code for the given compilation context.

        Args:
            ctx: Compilation context with the IR graph.

        Returns:
            CodeGenResult containing generated files.
        """
        pass
