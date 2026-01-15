"""Name management utilities for code generation."""

import re
from typing import Dict


class NameManager:
    """Manages C symbol names generated from ONNX names."""

    def __init__(self):
        self._symbol_map: Dict[str, str] = {}
        self._used_names: Dict[str, int] = {}

    def get_symbol(self, onnx_name: str, prefix: str = "") -> str:
        """Get or create a C symbol name for an ONNX name.

        Args:
            onnx_name: Original ONNX name.
            prefix: Optional prefix for the symbol.

        Returns:
            Valid C identifier.
        """
        if onnx_name in self._symbol_map:
            return self._symbol_map[onnx_name]

        # Sanitize the name
        sanitized = self._sanitize_name(onnx_name)
        symbol = f"{prefix}{sanitized}" if prefix else sanitized

        # Ensure uniqueness
        if symbol in self._used_names:
            counter = self._used_names[symbol]
            unique_symbol = f"{symbol}_{counter}"
            while unique_symbol in self._used_names:
                counter += 1
                unique_symbol = f"{symbol}_{counter}"
            symbol = unique_symbol
            self._used_names[f"{prefix}{sanitized}"] = counter + 1

        self._symbol_map[onnx_name] = symbol
        self._used_names[symbol] = 1
        return symbol

    def _sanitize_name(self, name: str) -> str:
        """Sanitize an ONNX name to be a valid C identifier.

        Handles:
        - Invalid C characters (., ::, /, -, etc.) -> underscore
        - Names starting with digits -> prefix with underscore
        - C keywords -> suffix with underscore
        - Empty names -> 'unnamed'
        """
        if not name:
            return "unnamed"

        # Replace invalid C characters with underscores
        # Valid C identifiers: letters, digits, underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)

        # Collapse multiple underscores into one
        sanitized = re.sub(r"_+", "_", sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")

        # Ensure it doesn't start with a digit
        if sanitized and sanitized[0].isdigit():
            sanitized = f"n{sanitized}"

        # Handle empty names (resulted from all invalid chars)
        if not sanitized:
            sanitized = "unnamed"

        # Avoid C keywords
        c_keywords = {
            "auto", "break", "case", "char", "const", "continue", "default",
            "do", "double", "else", "enum", "extern", "float", "for", "goto",
            "if", "int", "long", "register", "return", "short", "signed",
            "sizeof", "static", "struct", "switch", "typedef", "union",
            "unsigned", "void", "volatile", "while", "inline", "restrict",
            "bool", "true", "false", "NULL", "null"
        }
        if sanitized in c_keywords:
            sanitized = f"{sanitized}_"

        return sanitized

    def reset(self):
        """Reset all mappings."""
        self._symbol_map.clear()
        self._used_names.clear()
