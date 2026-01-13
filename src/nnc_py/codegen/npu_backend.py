"""NPU backend for actual chip deployment."""

from nnc_py.codegen.base import BackendBase, CodeGenResult


class NPUBackend(BackendBase):
    """NPU target backend - generates code linking to actual chip binary."""

    def generate(self, ctx) -> CodeGenResult:
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
