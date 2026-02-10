"""Identity elimination pass.

This pass removes Identity operations from the computation graph by replacing
all references to an Identity node's output with its input, then removing the node.
"""

from typing import Dict, List, Set

from nnc_py.ir.context import CompileContext
from nnc_py.ir.node import OpType
from nnc_py.passes.base import PassBase


class IdentityEliminationPass(PassBase):
    """Remove Identity operations from the computation graph.

    This pass:
    1. Finds all Identity nodes in the graph
    2. Builds a mapping from Identity output tensor to Identity input tensor
    3. Updates all consumer nodes to use the input tensor directly
    4. Updates graph outputs if needed
    5. Removes the Identity nodes

    The pass handles chains of Identity nodes correctly.
    """

    @property
    def name(self) -> str:
        return "IdentityElimination"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute Identity elimination."""
        graph = ctx.graph

        # Step 1: Find all Identity nodes and build replacement mapping
        # We need to handle chains, so we iteratively resolve the mapping
        replacement_map: Dict[str, str] = {}
        identity_nodes = [
            node for node in graph.nodes.values()
            if node.op_type == OpType.IDENTITY
        ]

        for identity_node in identity_nodes:
            if len(identity_node.inputs) == 1 and len(identity_node.outputs) == 1:
                output_tensor = identity_node.outputs[0]
                input_tensor = identity_node.inputs[0]
                replacement_map[output_tensor] = input_tensor

        # Step 2: Resolve chains (if A->B->C are identities, map C to A)
        replacement_map = self._resolve_chains(replacement_map)

        # Step 3: Update consumer nodes
        for node in graph.nodes.values():
            # Skip identity nodes (they will be removed)
            if node.op_type == OpType.IDENTITY:
                continue

            # Update inputs
            updated_inputs = []
            for input_tensor in node.inputs:
                updated_inputs.append(replacement_map.get(input_tensor, input_tensor))
            node.inputs = updated_inputs

        # Step 4: Update graph outputs
        updated_outputs = []
        for output_tensor in graph.outputs:
            updated_outputs.append(replacement_map.get(output_tensor, output_tensor))
        graph.outputs = updated_outputs

        # Step 5: Update graph inputs (in case an identity was marked as input)
        # This is less common but possible
        updated_inputs = []
        for input_tensor in graph.inputs:
            updated_inputs.append(replacement_map.get(input_tensor, input_tensor))
        graph.inputs = updated_inputs

        # Step 6: Remove Identity nodes
        nodes_to_remove = [
            node.name for node in identity_nodes
            if node.name in graph.nodes
        ]
        for node_name in nodes_to_remove:
            del graph.nodes[node_name]

        # Log summary if debug mode is on
        if ctx.debug:
            print(f"\n{'='*60}")
            print(f"Identity Elimination Summary")
            print(f"{'='*60}")
            print(f"Identity nodes removed: {len(nodes_to_remove)}")
            if replacement_map:
                print(f"Replacements: {len(replacement_map)} tensors remapped")
            print(f"{'='*60}")

    def _resolve_chains(self, replacement_map: Dict[str, str]) -> Dict[str, str]:
        """Resolve chains of Identity nodes.

        If we have mappings: B->A, C->B, we want to map C->A directly.

        Args:
            replacement_map: Initial mapping from output tensor to input tensor

        Returns:
            Resolved mapping where all chains are flattened
        """
        resolved = {}
        for output_tensor, input_tensor in replacement_map.items():
            # Follow the chain until we hit a tensor not in the map
            current = input_tensor
            visited: Set[str] = set()
            while current in replacement_map and current not in visited:
                visited.add(current)
                current = replacement_map[current]
            resolved[output_tensor] = current
        return resolved
