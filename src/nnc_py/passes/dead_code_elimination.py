"""Dead code elimination pass.

This pass removes nodes from the computation graph whose outputs are not used
by any other node and are not graph outputs.
"""

from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassBase


class DeadCodeEliminationPass(PassBase):
    """Remove unused nodes from the computation graph."""

    @property
    def name(self) -> str:
        return "DeadCodeElimination"

    def _execute(self, ctx: CompileContext) -> None:
        # TODO: implement
        pass
