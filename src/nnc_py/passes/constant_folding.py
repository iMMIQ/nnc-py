"""Constant folding pass - evaluate constant expressions at compile time."""

import numpy as np
from typing import Optional

from nnc_py.ir.context import CompileContext
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.graph import Graph
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.passes.base import PassBase


class ConstantFoldingPass(PassBase):
    """Fold constant operations by evaluating them at compile time.

    This pass identifies nodes where all inputs are constants, evaluates
    the operation, and replaces the output with a constant tensor.
    """

    @property
    def name(self) -> str:
        return "constant_folding"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute constant folding on the graph.

        Args:
            ctx: Compilation context to transform.
        """
        graph = ctx.graph
        folded_count = 0

        # Get nodes in topological order
        nodes = graph.topological_sort()

        for node in nodes:
            if self._try_fold_node(graph, node):
                folded_count += 1

        if ctx.debug and folded_count > 0:
            print(f"[ConstantFolding] Folded {folded_count} nodes")

    def _try_fold_node(self, graph: Graph, node: Node) -> bool:
        """Try to fold a single node.

        Args:
            graph: The computation graph.
            node: The node to potentially fold.

        Returns:
            True if the node was folded, False otherwise.
        """
        # Check if all inputs are constants
        input_constants = []
        for input_name in node.inputs:
            if input_name not in graph.constants:
                return False
            input_constants.append(graph.constants[input_name])

        # All inputs are constants - evaluate the operation
        try:
            result = self._evaluate_op(node, input_constants)
            if result is None:
                return False

            # Store the result as a constant
            output_name = node.outputs[0]
            graph.constants[output_name] = result

            # Update tensor type if it exists
            if output_name in graph.tensors:
                tensor = graph.tensors[output_name]
                # Update shape if it was static
                if tensor.shape.is_static():
                    new_shape = TensorShape(dims=list(result.shape), layout=tensor.shape.layout)
                    tensor.shape = new_shape

            # Mark node as folded (metadata for debugging)
            node.metadata["folded"] = True
            return True

        except Exception as e:
            # Folding failed - leave node as-is
            node.metadata["fold_error"] = str(e)
            return False

    def _evaluate_op(self, node: Node, inputs: list[np.ndarray]) -> Optional[np.ndarray]:
        """Evaluate an operation with constant inputs.

        Args:
            node: The node to evaluate.
            inputs: List of constant input arrays.

        Returns:
            Result array, or None if operation cannot be evaluated.
        """
        op_type = node.op_type

        # Unary operations
        if len(inputs) == 1:
            x = inputs[0]
            if op_type == OpType.RELU:
                return np.maximum(x, 0).astype(x.dtype)
            elif op_type == OpType.SIGMOID:
                return (1 / (1 + np.exp(-x))).astype(x.dtype)
            elif op_type == OpType.TANH:
                return np.tanh(x).astype(x.dtype)
            elif op_type == OpType.SOFTMAX:
                axis = node.attrs.get("axis", -1)
                e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
                return (e_x / e_x.sum(axis=axis, keepdims=True)).astype(x.dtype)
            elif op_type == OpType.TRANSPOSE:
                perms = node.attrs.get("perm", None)
                if perms:
                    return np.transpose(x, perms)
                return np.transpose(x)
            elif op_type == OpType.RESHAPE:
                shape = node.attrs.get("shape", None)
                if shape:
                    return x.reshape(shape)
                return x
            elif op_type == OpType.FLATTEN:
                return x.flatten()
            elif op_type == OpType.CLIP:
                min_val = node.attrs.get("min", -np.inf)
                max_val = node.attrs.get("max", np.inf)
                return np.clip(x, min_val, max_val).astype(x.dtype)

        # Binary operations
        elif len(inputs) == 2:
            a, b = inputs

            if op_type == OpType.ADD:
                return (a + b).astype(a.dtype)
            elif op_type == OpType.SUB:
                return (a - b).astype(a.dtype)
            elif op_type == OpType.MUL:
                return (a * b).astype(a.dtype)
            elif op_type == OpType.DIV:
                return (a / b).astype(a.dtype)
            elif op_type == OpType.POW:
                return np.power(a, b).astype(a.dtype)
            elif op_type == OpType.CONCAT:
                axis = node.attrs.get("axis", 0)
                return np.concatenate([a, b], axis=axis).astype(a.dtype)

        # Ternary operations
        elif len(inputs) == 3:
            if op_type == OpType.CONCAT:
                axis = node.attrs.get("axis", 0)
                return np.concatenate(inputs, axis=axis).astype(inputs[0].dtype)

        # N-ary operations
        if op_type == OpType.CONCAT:
            axis = node.attrs.get("axis", 0)
            return np.concatenate(inputs, axis=axis).astype(inputs[0].dtype)

        # Operations not supported for constant folding
        return None
