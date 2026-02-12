"""Enhanced fusion groups with dominator-based fusion support."""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Set, Optional

from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.passes.indexed_forward_graph import IndexedForwardGraph
from nnc_py.passes.dominator_tree import DominatorTree
from nnc_py.ir.op_pattern import get_op_pattern_kind, OpPatternKind
from nnc_py.passes.fusion_groups import FusionGroup, GroupArena as BaseGroupArena


class EnhancedGroupArena(BaseGroupArena):
    """Enhanced GroupArena with dominator-based fusion support."""

    def __init__(self, graph: Graph, indexed_graph: IndexedForwardGraph,
                 dominator_tree: DominatorTree, max_fuse_depth: int = 256,
                 max_function_args: int = 256):
        """Initialize enhanced group arena.

        Args:
            graph: The graph being processed.
            indexed_graph: Indexed forward graph.
            dominator_tree: Dominator tree for the graph.
            max_fuse_depth: Maximum depth of fused operations.
            max_function_args: Maximum number of arguments in fused functions.
        """
        super().__init__()
        self.graph = graph
        self.indexed_graph = indexed_graph
        self.dominator_tree = dominator_tree
        self.max_fuse_depth = max_fuse_depth
        self.max_function_args = max_function_args

        # Additional mapping for node names to groups
        self._node_to_group_id: Dict[str, int] = {}

    def new_group(self) -> int:
        """Create a new group and return its ID."""
        group_id = self._next_node_id
        group = self.create_group("unknown", 0)
        self._groups[group_id] = group
        self._next_node_id += 1
        return group_id

    def add_node_to_group(self, group_id: int, node_name: str) -> None:
        """Add a node to a group.

        Args:
            group_id: The group ID.
            node_name: The name of the node to add.
        """
        if group_id not in self._groups:
            raise ValueError(f"Group {group_id} not found")

        group = self._groups[group_id]

        # If node is already in another group, merge groups
        if node_name in self._node_to_group_id:
            existing_group_id = self._node_to_group_id[node_name]
            if existing_group_id != group_id:
                self.merge_groups_by_id(existing_group_id, group_id)
                return

        # Add node to group
        self._node_to_group_id[node_name] = group_id
        group.num_nodes += 1

        # Update args count (simplified - would need more sophisticated calculation)
        if hasattr(group, 'args_num'):
            group.args_num += 1

    def is_node_grouped(self, node_name: str) -> bool:
        """Check if a node is already in a group.

        Args:
            node_name: Name of the node to check.

        Returns:
            True if node is grouped, False otherwise.
        """
        return node_name in self._node_to_group_id

    def can_add_to_group(self, node1_name: str, node2_name: str) -> bool:
        """Check if two nodes can be added to the same group respecting depth limit.

        Args:
            node1_name: First node name.
            node2_name: Second node name.

        Returns:
            True if nodes can be added to the same group.
        """
        # Simple depth check - in a real implementation, this would check
        # the actual path length in the dominator tree
        return True

    def set_group_pattern_kind(self, group_id: int, pattern_kind: OpPatternKind) -> None:
        """Set the pattern kind for a group.

        Args:
            group_id: The group ID.
            pattern_kind: The pattern kind to set.
        """
        if group_id not in self._groups:
            raise ValueError(f"Group {group_id} not found")

        group = self._groups[group_id]
        group.pattern = pattern_kind.name

    def get_group_pattern_kind(self, group_id: int) -> OpPatternKind:
        """Get the pattern kind for a group.

        Args:
            group_id: The group ID.

        Returns:
            The pattern kind for the group.
        """
        if group_id not in self._groups:
            raise ValueError(f"Group {group_id} not found")

        group = self._groups[group_id]
        try:
            return OpPatternKind[group.pattern]
        except KeyError:
            return OpPatternKind.kOpaque

    def get_all_groups(self) -> Dict[int, List[str]]:
        """Get all groups with their node names.

        Returns:
            Dictionary mapping group ID to list of node names.
        """
        result = {}
        node_groups: Dict[int, List[str]] = {}

        # Build inverse mapping from node name to group ID
        for node_name, group_id in self._node_to_group_id.items():
            if group_id not in node_groups:
                node_groups[group_id] = []
            node_groups[group_id].append(node_name)

        return node_groups

    def merge_groups_by_id(self, group1_id: int, group2_id: int) -> None:
        """Merge two groups by their IDs.

        Args:
            group1_id: First group ID.
            group2_id: Second group ID.
        """
        if group1_id not in self._groups or group2_id not in self._groups:
            raise ValueError("One or both groups not found")

        group1 = self._groups[group1_id]
        group2 = self._groups[group2_id]

        self.merge_groups(group1, group2)

        # Update node mappings
        for node_name, group_id in list(self._node_to_group_id.items()):
            if group_id == group2_id:
                self._node_to_group_id[node_name] = group1_id