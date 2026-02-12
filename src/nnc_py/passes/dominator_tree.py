"""Post-dominator tree for dominator-based operator fusion.

This module provides a post-dominator tree implementation using NetworkX.
Post-dominance is used in fusion analysis to determine which nodes must
execute together on all paths to the exit.

Key concepts:
- A node y post-dominates node x if all paths from x to exit go through y
- Post-dominance is computed as dominance on the reversed graph
- A virtual exit node is added to connect all graph outputs
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import networkx as nx

from nnc_py.passes.indexed_forward_graph import IndexedForwardGraph, NodeEntry

if TYPE_CHECKING:
    from nnc_py.ir.graph import Graph


VIRTUAL_EXIT = "__EXIT__"


class DominatorTree:
    """Post-dominator tree for fusion analysis.

    This class computes the post-dominator relationship between nodes
    in a computation graph. Post-dominance is essential for identifying
    safe fusion opportunities in operator fusion passes.

    Example:
        >>> graph = Graph("test")
        >>> # ... build graph ...
        >>> ifg = IndexedForwardGraph(graph)
        >>> dom_tree = DominatorTree(ifg)
        >>> idom = dom_tree.get_immediate_dominator("node_name")
    """

    def __init__(self, indexed_graph: IndexedForwardGraph):
        """Build the post-dominator tree from an indexed forward graph.

        Args:
            indexed_graph: The indexed forward graph to analyze.
        """
        self.indexed_graph: IndexedForwardGraph = indexed_graph
        self.graph: Graph = indexed_graph.graph
        self._immediate_dominators: Dict[str, Optional[str]] = {}
        self._dominance_frontier: Dict[str, List[str]] = {}

        self._build()

    def _build(self) -> None:
        """Build the post-dominator tree.

        Algorithm:
        1. Build a NetworkX directed graph from the computation graph
        2. Add a virtual exit node connected to all graph outputs
        3. Reverse the graph (post-dominance = dominance in reversed graph)
        4. Compute immediate dominators using NetworkX
        5. Filter out the virtual exit from results
        """
        # Step 1: Build NetworkX graph from computation graph
        nx_graph = nx.DiGraph()

        # Add all nodes
        for node_name in self.graph.nodes:
            nx_graph.add_node(node_name)

        # Add edges based on data flow
        for node_name, node in self.graph.nodes.items():
            for input_tensor in node.inputs:
                # Find producers of this input tensor
                for producer in self.graph.get_producers(input_tensor):
                    if producer.name in self.graph.nodes:
                        nx_graph.add_edge(producer.name, node_name)

        # Step 2: Add virtual exit node
        # Connect all graph outputs to the virtual exit
        exit_node = VIRTUAL_EXIT
        nx_graph.add_node(exit_node)

        for output_name in self.graph.outputs:
            # Find producers of this output
            for producer in self.graph.get_producers(output_name):
                if producer.name in self.graph.nodes:
                    nx_graph.add_edge(producer.name, exit_node)

        # Also handle nodes marked as extern_ref in indexed_graph
        for node_name, entry in self.indexed_graph.node_map.items():
            if entry.extern_ref:
                nx_graph.add_edge(node_name, exit_node)

        # Step 3: Reverse the graph for post-dominance computation
        nx_graph_reversed = nx_graph.reverse()

        # Step 4: Compute immediate dominators using the virtual exit as start
        # In the reversed graph, we start from the exit node
        if exit_node in nx_graph_reversed:
            try:
                idom = nx.immediate_dominators(nx_graph_reversed, exit_node)

                # Step 5: Extract results, filtering out virtual exit
                for node_name in self.graph.nodes:
                    dominator = idom.get(node_name)
                    if dominator == VIRTUAL_EXIT:
                        # Nodes directly dominated by exit have no post-dominator
                        self._immediate_dominators[node_name] = None
                    elif dominator is None or dominator == node_name:
                        # Exit node or self-loop
                        self._immediate_dominators[node_name] = None
                    else:
                        self._immediate_dominators[node_name] = dominator
            except nx.NetworkXError:
                # Graph might not be connected properly, set all to None
                for node_name in self.graph.nodes:
                    self._immediate_dominators[node_name] = None
        else:
            # No exit node found, graph might be empty
            for node_name in self.graph.nodes:
                self._immediate_dominators[node_name] = None

    def get_immediate_dominator(self, node_name: str) -> Optional[str]:
        """Get the immediate post-dominator of a node.

        The immediate post-dominator is the closest node that post-dominates
        the given node on all paths to the exit.

        Args:
            node_name: Name of the node to query.

        Returns:
            Name of the immediate post-dominator, or None if the node
            is not post-dominated (e.g., output nodes).
        """
        return self._immediate_dominators.get(node_name)

    def get_post_dominator_chain(self, node_name: str) -> List[str]:
        """Get the chain of post-dominators from a node to the exit.

        Returns a list starting with the node's immediate post-dominator,
        followed by its post-dominator, and so on, until reaching a node
        with no post-dominator.

        Args:
            node_name: Name of the starting node.

        Returns:
            List of post-dominator names in order from closest to farthest.
        """
        chain: List[str] = []
        current = self.get_immediate_dominator(node_name)

        while current is not None:
            chain.append(current)
            current = self.get_immediate_dominator(current)

        return chain

    def find_common_post_dominator(self, node_names: List[str]) -> Optional[str]:
        """Find the nearest common post-dominator of multiple nodes.

        This is useful for determining fusion boundaries - nodes can be
        safely fused if they share a common post-dominator and meet
        other safety criteria.

        Args:
            node_names: List of node names to find common post-dominator for.

        Returns:
            Name of the nearest common post-dominator, or None if none exists.
        """
        if not node_names:
            return None

        # Get post-dominator chains for all nodes
        chains = []
        for name in node_names:
            chain = self.get_post_dominator_chain(name)
            chains.append(chain)

        # Find intersection of all chains
        if not chains:
            return None

        # Start with the first chain and find common elements
        common = set(chains[0])

        for chain in chains[1:]:
            common &= set(chain)
            if not common:
                return None

        # Return the nearest (first in chain) common post-dominator
        for node in chains[0]:
            if node in common:
                return node

        return None

    def does_post_dominate(self, node: str, potential_dominator: str) -> bool:
        """Check if potential_dominator post-dominates node.

        Args:
            node: The node to check.
            potential_dominator: The potential post-dominator.

        Returns:
            True if potential_dominator post-dominates node.
        """
        current = self.get_immediate_dominator(node)
        while current is not None:
            if current == potential_dominator:
                return True
            current = self.get_immediate_dominator(current)
        return False


__all__ = [
    "DominatorTree",
    "VIRTUAL_EXIT",
]
