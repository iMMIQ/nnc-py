import pytest
from nnc_py.passes.fusion_groups import FusionGroup, GroupArena


def test_group_creation():
    """Test creating groups with different pattern_kind values."""
    arena = GroupArena()

    # Create groups with different pattern kinds
    group1 = arena.create_group(pattern_kind="matmul", args_num=2, attrs={"dtype": "float32"})
    group2 = arena.create_group(pattern_kind="relu", args_num=1, attrs={"alpha": 0.1})

    # Verify group properties
    assert group1.pattern == "matmul"
    assert group1.args_num == 2
    assert group1.attrs == {"dtype": "float32"}
    assert group1.num_nodes == 1
    assert group1.root_node == 1
    assert group1.parent == group1

    assert group2.pattern == "relu"
    assert group2.args_num == 1
    assert group2.attrs == {"alpha": 0.1}
    assert group2.num_nodes == 1
    assert group2.root_node == 2
    assert group2.parent == group2


def test_group_merge():
    """Test merging groups works correctly."""
    arena = GroupArena()

    # Create two groups
    group1 = arena.create_group(pattern_kind="matmul", args_num=2)
    group2 = arena.create_group(pattern_kind="relu", args_num=1)

    # Verify initial state
    assert arena.get_root(group1) == group1
    assert arena.get_root(group2) == group2
    assert arena.get_all_roots() == [group1, group2]

    # Merge group2 into group1
    arena.merge_groups(group1, group2)

    # Verify merge result
    root1 = arena.get_root(group1)
    root2 = arena.get_root(group2)

    assert root1 == root2  # Both should have the same root
    assert root1.num_nodes == 2  # Combined node count
    assert root1.pattern == "matmul"  # Keep the root's pattern

    # Verify only one root exists
    assert len(arena.get_all_roots()) == 1


def test_path_compression():
    """Test path compression in find_root() method."""
    arena = GroupArena()

    # Create groups: 1 -> 2 -> 3 (chain)
    group1 = arena.create_group(pattern_kind="A", args_num=1)
    group2 = arena.create_group(pattern_kind="B", args_num=1)
    group3 = arena.create_group(pattern_kind="C", args_num=1)

    # Create a chain: group1.parent = group3, group2.parent = group3
    arena._get_group_by_node(1).parent = group3
    arena._get_group_by_node(2).parent = group3
    group3.parent = group3  # Self parent for root

    # Merge groups to create longer chains
    arena.merge_groups(group1, group2)
    arena.merge_groups(group2, group3)

    # Before path compression, accessing root should compress paths
    root = arena.get_root(group1)
    assert root == arena.get_root(group2) == arena.get_root(group3)

    # Verify path compression worked by checking parent pointers
    # After path compression, all nodes should point directly to root
    root_node = root
    assert root_node.parent == root_node  # Root should point to itself

    # Child nodes should point directly to root after path compression
    assert arena._get_group_by_node(1).parent == root_node
    assert arena._get_group_by_node(2).parent == root_node


def test_node_counting():
    """Test num_nodes updates correctly on merge."""
    arena = GroupArena()

    # Create three groups
    group1 = arena.create_group(pattern_kind="A", args_num=1)
    group2 = arena.create_group(pattern_kind="B", args_num=1)
    group3 = arena.create_group(pattern_kind="C", args_num=1)

    # Check initial counts
    assert arena._get_group_by_node(1).num_nodes == 1
    assert arena._get_group_by_node(2).num_nodes == 1
    assert arena._get_group_by_node(3).num_nodes == 1

    # Merge group2 into group1
    arena.merge_groups(group1, group2)

    # Check updated counts
    root1 = arena.get_root(group1)
    assert root1.num_nodes == 2
    assert arena._get_group_by_node(3).num_nodes == 1  # Unchanged

    # Merge group3 into the merged group
    arena.merge_groups(root1, group3)

    # Check final count
    final_root = arena.get_root(root1)
    assert final_root.num_nodes == 3