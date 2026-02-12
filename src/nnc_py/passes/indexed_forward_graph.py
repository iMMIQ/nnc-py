"""Indexed forward graph for dominator-based operator fusion.

This module provides a data structure that indexes a computation graph
for efficient fusion analysis. It follows TVM's approach by building
a post-DFS order and maintaining output edge information.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from nnc_py.ir.graph import Graph
from nnc_py.ir.op_pattern import OpPatternKind, get_op_pattern_kind

if TYPE_CHECKING:
    from nnc_py.ir.node import Node


@dataclass
class Edge:
    """Edge in the indexed forward graph.

    Attributes:
        node: The destination NodeEntry this edge points to.
        pattern: The pattern kind of the destination node.
    """

    node: "NodeEntry"
    pattern: OpPatternKind


@dataclass
class NodeEntry:
    """Entry for a node in the indexed forward graph.

    Attributes:
        node: The original IR node.
        index: Index in the post-DFS order.
        pattern: The pattern kind of this node.
        outputs: List of output edges from this node.
        extern_ref: Whether this node is referenced externally (output node).
    """

    node: "Node"
    index: int
    pattern: OpPatternKind
    outputs: List[Edge] = field(default_factory=list)
    extern_ref: bool = False


class IndexedForwardGraph:
    """Indexed forward graph for fusion analysis.

    This data structure provides:
    - Post-DFS ordering of nodes (reverse topological sort)
    - Fast lookup of node entries by name
    - Output edge information for each node
    - Pattern kind classification for all nodes

    The post-DFS order ensures that descendants appear before ancestors,
    which is essential for dominator-based fusion algorithms.
    """

    def __init__(self, graph: Graph):
        """Build the indexed forward graph from a computation graph.

        Args:
            graph: The input computation graph to index.
        """
        self.graph: Graph = graph
        self.node_map: Dict[str, NodeEntry] = {}
        self.post_dfs_order: List[NodeEntry] = []

        self._build()

    def _build(self) -> None:
        """Build the indexed forward graph structure.

        This method:
        1. Creates NodeEntry for all nodes with pattern kinds
        2. Builds output edges between nodes
        3. Computes post-DFS order (reverse topological sort)
        4. Marks nodes that are graph outputs as extern_ref
        """
        # Step 1: Create entries for all nodes
        for node in self.graph.nodes.values():
            pattern = get_op_pattern_kind(node.op_type)
            entry = NodeEntry(
                node=node,
                index=-1,  # Will be set during ordering
                pattern=pattern,
                outputs=[],
                extern_ref=False,
            )
            self.node_map[node.name] = entry

        # Step 2: Build output edges
        for node in self.graph.nodes.values():
            entry = self.node_map[node.name]
            for output_tensor in node.outputs:
                # Find all consumers of this tensor
                consumers = self.graph.get_consumers(output_tensor)
                for consumer in consumers:
                    consumer_entry = self.node_map[consumer.name]
                    edge = Edge(
                        node=consumer_entry,
                        pattern=consumer_entry.pattern,
                    )
                    entry.outputs.append(edge)

        # Step 3: Build post-DFS order using reverse topological sort
        topo_order = self.graph.topological_sort()
        # Reverse to get post-DFS order (leaves first)
        for i, node in enumerate(reversed(topo_order)):
            entry = self.node_map[node.name]
            entry.index = i
            self.post_dfs_order.append(entry)

        # Step 4: Mark extern_ref for output nodes
        for output_name in self.graph.outputs:
            # Find the node that produces this output
            producers = self.graph.get_producers(output_name)
            for producer in producers:
                if producer.name in self.node_map:
                    self.node_map[producer.name].extern_ref = True

    def get_node_entry(self, node_name: str) -> Optional[NodeEntry]:
        """Get the NodeEntry for a node by name.

        Args:
            node_name: Name of the node to look up.

        Returns:
            The NodeEntry if found, None otherwise.
        """
        return self.node_map.get(node_name)

    def get_output_entries(self, node_name: str) -> List[NodeEntry]:
        """Get the NodeEntries for all outputs of a node.

        Args:
            node_name: Name of the source node.

        Returns:
            List of NodeEntries for nodes that consume this node's outputs.
        """
        entry = self.node_map.get(node_name)
        if entry is None:
            return []
        return [edge.node for edge in entry.outputs]


__all__ = [
    "Edge",
    "NodeEntry",
    "IndexedForwardGraph",
]
