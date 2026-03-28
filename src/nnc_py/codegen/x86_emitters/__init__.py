"""Emitters for x86 code generation artifacts."""

from nnc_py.codegen.x86_emitters.build_files import emit_makefile, emit_test_runner
from nnc_py.codegen.x86_emitters.constants_loader import emit_constants_loader
from nnc_py.codegen.x86_emitters.header import emit_header
from nnc_py.codegen.x86_emitters.model_source import emit_model_source
from nnc_py.codegen.x86_emitters.tensors import emit_tensors

__all__ = [
    "emit_header",
    "emit_model_source",
    "emit_tensors",
    "emit_constants_loader",
    "emit_makefile",
    "emit_test_runner",
]
