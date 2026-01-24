"""Configuration management for the compiler."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CompilerConfig:
    """Compiler configuration."""

    target: str = "x86"
    opt_level: int = 0
    debug: bool = False

    # Frontend options
    enable_constant_folding: bool = True

    # Pass configuration
    enable_fusion: bool = False
    enable_layout_opt: bool = True

    # Code generation options
    emit_comments: bool = True
    emit_symbols: bool = True

    # Target-specific options
    target_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompilerConfig":
        """Create config from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "target": self.target,
            "opt_level": self.opt_level,
            "debug": self.debug,
            "enable_constant_folding": self.enable_constant_folding,
            "enable_fusion": self.enable_fusion,
            "enable_layout_opt": self.enable_layout_opt,
            "emit_comments": self.emit_comments,
            "emit_symbols": self.emit_symbols,
            "target_config": self.target_config,
        }
