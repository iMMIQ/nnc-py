"""NPU backend for actual chip deployment."""

from typing import TYPE_CHECKING

from nnc_py.codegen.base import BackendBase, CodeGenResult

if TYPE_CHECKING:
    from nnc_py.ir.context import CompileContext


class NPUBackend(BackendBase):
    """NPU target backend - generates code linking to actual chip binary."""

    def __init__(self, debug_mode: bool = False):
        """Initialize the NPU backend.

        Args:
            debug_mode: Whether to enable debug mode with intermediate tensor dumps.
        """
        self.debug_mode = debug_mode

    def generate(self, ctx: "CompileContext") -> CodeGenResult:
        """Generate NPU C code.

        This backend generates code that links to the actual NPU chip binary.
        It marks which operators can be accelerated by the NPU and generates
        appropriate memory layout descriptions.
        """
        result = CodeGenResult()

        # TODO: Implement NPU-specific code generation
        # 1. Annotate nodes for NPU acceleration
        # 2. Generate NPU-specific API calls
        # 3. Generate memory layout descriptions
        # 4. Handle CPU fallback for unsupported ops

        # For now, return a placeholder
        result.add_file(
            "model_npu.c",
            "/* NPU backend - TODO */\n",
            "source"
        )

        return result
