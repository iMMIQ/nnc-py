from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class FusionGroup:
    """Represents a fusion group with Union-Find capabilities."""
    parent: 'FusionGroup'  # Parent pointer for Union-Find
    pattern: str
    root_node: int  # ID of the root node
    num_nodes: int  # Number of nodes in this group
    args_num: int
    attrs: Dict[str, Any]
    node_id: int  # Unique identifier for this node

    def __eq__(self, other):
        if isinstance(other, FusionGroup):
            return self.node_id == other.node_id
        return False

    def __hash__(self):
        return hash(self.node_id)

    def __init__(self, pattern: str, args_num: int, attrs: Dict[str, Any], node_id: int):
        self.parent = self  # Initially points to itself
        self.pattern = pattern
        self.root_node = node_id
        self.num_nodes = 1
        self.args_num = args_num
        self.attrs = attrs.copy()
        self.node_id = node_id

    def find_root(self) -> 'FusionGroup':
        """Find the root of this group with path compression."""
        if self.parent != self:
            # Path compression: point directly to root
            self.parent = self.parent.find_root()
        return self.parent

    def merge_from(self, other: 'FusionGroup') -> None:
        """Merge another group into this one."""
        other_root = other.find_root()
        if self.find_root() != other_root:
            # Attach other's root to this one
            other_root.parent = self.find_root()
            # Update node count
            self.find_root().num_nodes += other_root.num_nodes


class GroupArena:
    """Manages a collection of fusion groups using Union-Find."""

    def __init__(self):
        self._groups: Dict[int, FusionGroup] = {}  # Maps node_id to group data
        self._next_node_id = 1

    def _get_group_by_node(self, node_id: int) -> FusionGroup:
        """Get group by node_id."""
        return self._groups.get(node_id)

    def create_group(self, pattern_kind: str, args_num: int, attrs: Dict[str, Any] = None) -> FusionGroup:
        """Create a new fusion group."""
        if attrs is None:
            attrs = {}

        group = FusionGroup(
            pattern=pattern_kind,
            args_num=args_num,
            attrs=attrs,
            node_id=self._next_node_id
        )

        self._groups[self._next_node_id] = group
        self._next_node_id += 1
        return group

    def get_root(self, group: FusionGroup) -> FusionGroup:
        """Get the root group for the given group."""
        return group.find_root()

    def merge_groups(self, group1: FusionGroup, group2: FusionGroup) -> None:
        """Merge two groups together."""
        root1 = self.get_root(group1)
        root2 = self.get_root(group2)

        if root1 != root2:
            # Merge root2 into root1
            root1.merge_from(root2)

    def get_all_roots(self) -> List[FusionGroup]:
        """Get all root groups in the arena."""
        seen_roots = []
        for group in self._groups.values():
            root = group.find_root()
            if root not in seen_roots:
                seen_roots.append(root)
        return seen_roots