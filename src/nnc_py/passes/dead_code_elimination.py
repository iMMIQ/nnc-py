"""Dead code elimination pass.

This pass removes nodes from the computation graph whose outputs are not used
by any other node and are not graph outputs.
"""

from collections import deque

from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassBase


class DeadCodeEliminationPass(PassBase):
    """Remove unused nodes from the computation graph."""

    @property
    def name(self) -> str:
        return "DeadCodeElimination"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute dead code elimination.

        Algorithm:
        1. Mark all nodes as "dead" initially
        2. Mark nodes that produce graph outputs as "live"
        3. Backward propagate: mark nodes that produce inputs to live nodes as "live"
        4. Remove all dead nodes
        """
        graph = ctx.graph

        # Collect all live tensors (graph outputs and inputs)
        live_tensors = set(graph.outputs)
        live_tensors.update(graph.inputs)  # Keep input tensors marked

        # Work backwards: find nodes that produce live tensors
        live_nodes = set()

        # Initialize queue with output tensors
        queue = deque(graph.outputs)

        while queue:
            tensor_name = queue.popleft()

            # Find nodes that produce this tensor
            producers = graph.get_producers(tensor_name)

            for producer in producers:
                if producer.name not in live_nodes:
                    live_nodes.add(producer.name)
                    # Add this node's inputs to the queue
                    for input_tensor in producer.inputs:
                        if input_tensor not in live_tensors:
                            live_tensors.add(input_tensor)
                            queue.append(input_tensor)

        # Remove dead nodes
        nodes_to_remove = [
            node_name for node_name in graph.nodes
            if node_name not in live_nodes
        ]

        for node_name in nodes_to_remove:
            del graph.nodes[node_name]

        # Log summary if debug mode is on
        if ctx.debug:
            print(f"\n{'='*60}")
            print(f"Dead Code Elimination Summary")
            print(f"{'='*60}")
            print(f"Nodes before: {len(graph.nodes) + len(nodes_to_remove)}")
            print(f"Nodes removed: {len(nodes_to_remove)}")
            print(f"Nodes after: {len(graph.nodes)}")
            if nodes_to_remove:
                print(f"Removed nodes: {', '.join(nodes_to_remove)}")
            print(f"{'='*60}")
