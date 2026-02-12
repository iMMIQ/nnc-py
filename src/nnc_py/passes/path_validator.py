"""Path validation for safe operator fusion.

This module provides path validation to ensure safe fusion operations
by checking paths between nodes satisfy pattern constraints.
"""

from typing import Callable, Optional, Set, List

from nnc_py.passes.indexed_forward_graph import IndexedForwardGraph, NodeEntry, Edge
from nnc_py.ir.op_pattern import OpPatternKind


class PathValidator:
    """Validates paths between nodes for safe operator fusion."""

    def __init__(self, graph: IndexedForwardGraph):
        """Initialize path validator with indexed forward graph.

        Args:
            graph: The indexed forward graph to validate paths in.
        """
        self.graph = graph

    def check_path(self, src: NodeEntry, dst: NodeEntry, max_kind: OpPatternKind) -> bool:
        """Validate all paths from src to dst satisfy pattern constraint.

        Args:
            src: Source node entry
            dst: Destination node entry
            max_kind: Maximum pattern kind allowed on the path

        Returns:
            True if all paths from src to dst have pattern kinds <= max_kind,
            False otherwise.
        """
        # Get all paths from src to dst using DFS
        paths = self._find_all_paths(src, dst)

        # If no path exists, we consider it valid
        if not paths:
            return True

        # Check if any path contains a node with pattern kind > max_kind
        for path in paths:
            for node_entry in path:
                if node_entry.pattern.value > max_kind.value:
                    return False

        return True

    def _find_all_paths(self, src: NodeEntry, dst: NodeEntry) -> list[list[NodeEntry]]:
        """Find all simple paths from src to dst."""
        paths = []
        current_path = []

        def dfs(current: NodeEntry):
            current_path.append(current)

            if current == dst:
                paths.append(list(current_path))
            else:
                for edge in current.outputs:
                    # Check if we've already visited this node in the current path
                    if edge.node not in current_path:
                        dfs(edge.node)

            current_path.pop()

        dfs(src)
        return paths

    def check_path_with_condition(self, src: NodeEntry, dst: NodeEntry, condition: Callable[[NodeEntry], bool]) -> bool:
        """Validate path from src to dst satisfies custom condition.

        Args:
            src: Source node entry
            dst: Destination node entry
            condition: Function that returns True if node is allowed on path

        Returns:
            True if all nodes on all paths from src to dst satisfy the condition,
            False otherwise.
        """
        paths = self._find_all_paths(src, dst)

        # Check if any path contains a node that doesn't satisfy the condition
        for path in paths:
            for node_entry in path:
                if not condition(node_entry):
                    return False

        return True

    def _check_path_condition_recursive(self,
                                       current: NodeEntry,
                                       dst: NodeEntry,
                                       condition: Callable[[NodeEntry], bool],
                                       current_path: list[NodeEntry]) -> bool:
        """Recursive helper for condition checking."""
        current_path.append(current)

        if current == dst:
            result = True
        else:
            result = False
            for edge in current.outputs:
                if condition(edge.node) and edge.node not in current_path:
                    if self._check_path_condition_recursive(edge.node, dst, condition, current_path):
                        result = True
                        break

        current_path.pop()
        return result

    def count_nodes_on_path(self, src: NodeEntry, dst: NodeEntry) -> int:
        """Count number of nodes on shortest path from src to dst.

        Args:
            src: Source node entry
            dst: Destination node entry

        Returns:
            Number of nodes on the shortest path from src to dst (including src and dst),
            or 0 if no path exists.
        """
        if src == dst:
            return 1

        queue = [(src, 1)]  # (node, count)
        visited = set()

        while queue:
            current, count = queue.pop(0)

            # Use node name for hashing
            node_key = (current.node.name, current.index)
            if node_key in visited:
                continue

            visited.add(node_key)

            if current == dst:
                return count

            for edge in current.outputs:
                edge_key = (edge.node.node.name, edge.node.index)
                if edge_key not in visited:
                    queue.append((edge.node, count + 1))

        return 0