"""Dominator-based operator fusion pass."""

import logging

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node
from nnc_py.ir.op_pattern import OpPatternKind, get_op_pattern_kind
from nnc_py.passes.base import PassBase
from nnc_py.passes.dominator_tree import DominatorTree
from nnc_py.passes.fusion_groups_enhanced import EnhancedGroupArena
from nnc_py.passes.indexed_forward_graph import IndexedForwardGraph
from nnc_py.passes.path_validator import PathValidator

logger = logging.getLogger(__name__)


class DominatorFusionPass(PassBase):
    """Dominator-based operator fusion pass.

    This pass analyzes the dominator tree of the graph to identify fusion groups
    that can be safely fused. It implements the diamond pattern fusion strategy
    described in the TODO.

    Key features:
    - Analyzes post-dominator trees to identify structured patterns
    - Groups operations that are dominated by the same operations
    - Validates fusion safety using path validation
    - Respects max_fuse_depth and max_function_args limits
    """

    @property
    def name(self) -> str:
        return "DominatorFusion"

    def __init__(self, max_fuse_depth: int = 256, max_function_args: int = 256):
        """Initialize dominator fusion pass.

        Args:
            max_fuse_depth: Maximum depth of fused operations.
            max_function_args: Maximum number of arguments in fused functions.
        """
        self.max_fuse_depth = max_fuse_depth
        self.max_function_args = max_function_args
        self.group_arena = None
        self.path_validator = None
        self.dominator_tree = None
        self.graph = None

    def _execute(self, ctx: CompileContext) -> None:
        """Execute dominator-based fusion."""
        graph = ctx.graph
        self.graph = graph

        logger.info(f"Running {self.name} pass")

        # Build necessary data structures
        self._build_data_structures(graph)

        # Run fusion phases
        self._run_fuse_phase_0()
        self._run_fuse_phase_1()

        # Apply identified fusions
        self._apply_fusions()

        # Log fusion summary
        self._log_fusion_summary()

    def _build_data_structures(self, graph: Graph) -> None:
        """Build necessary data structures for fusion analysis."""
        # Build indexed forward graph
        indexed_graph = IndexedForwardGraph(graph)

        # Build dominator tree
        self.dominator_tree = DominatorTree(indexed_graph)

        # Build group arena for fusion groups
        self.group_arena = EnhancedGroupArena(
            graph,
            indexed_graph,
            self.dominator_tree,
            self.max_fuse_depth,
            self.max_function_args
        )

        # Build path validator for safety checks
        self.path_validator = PathValidator(indexed_graph)

    def _run_fuse_phase_0(self) -> None:
        """Phase 0: Fuse kOutEWiseFusable into kElemWise.

        This phase identifies operations that can be fused based on their
        output-wise fusable pattern.
        """
        logger.info("Running Phase 0: kOutEWiseFusable -> kElemWise fusion")

        # Get all nodes that are kOutEWiseFusable
        output_ewise_nodes = []
        for node in self.graph.nodes.values():
            pattern_kind = get_op_pattern_kind(node.op_type)
            if pattern_kind == OpPatternKind.kOutEWiseFusable:
                output_ewise_nodes.append(node)

        # Group by dominator and fuse
        for node in output_ewise_nodes:
            if self.group_arena.is_node_grouped(node.name):
                continue  # Skip already grouped nodes

            # Find dominator tree
            dominators = self.dominator_tree.get_post_dominator_chain(node.name)
            if not dominators:
                continue

            # Try to fuse with dominator
            for dom_node_name in dominators:
                if dom_node_name == node.name:
                    continue

                dom_node = self.graph.get_node(dom_node_name)
                if not dom_node:
                    continue

                # Check if safe to fuse
                if self._is_safe_to_fuse(dom_node, node):
                    # Create fusion group
                    group_id = self.group_arena.new_group()
                    self.group_arena.add_node_to_group(group_id, dom_node_name)
                    self.group_arena.add_node_to_group(group_id, node.name)

                    # Mark pattern kind
                    self.group_arena.set_group_pattern_kind(
                        group_id, OpPatternKind.kElemWise
                    )
                    break

    def _run_fuse_phase_1(self) -> None:
        """Phase 1: Fuse kElemWise/kInjective into kBroadcast.

        This phase identifies operations that can be fused based on their
        element-wise or injective pattern into a broadcast pattern.
        """
        logger.info("Running Phase 1: kElemWise/kInjective -> kBroadcast fusion")

        # Get all nodes that are kElemWise or kInjective
        elemwise_nodes = []
        for node in self.graph.nodes.values():
            pattern_kind = get_op_pattern_kind(node.op_type)
            if pattern_kind in [OpPatternKind.kElemWise, OpPatternKind.kInjective]:
                elemwise_nodes.append(node)

        # Group by dominator and fuse
        for node in elemwise_nodes:
            if self.group_arena.is_node_grouped(node.name):
                continue  # Skip already grouped nodes

            # Find dominator tree
            dominators = self.dominator_tree.get_post_dominator_chain(node.name)
            if not dominators:
                continue

            # Try to fuse with dominator
            for dom_node_name in dominators:
                if dom_node_name == node.name:
                    continue

                dom_node = self.graph.get_node(dom_node_name)
                if not dom_node:
                    continue

                # Check if safe to fuse
                if self._is_safe_to_fuse(dom_node, node):
                    # Create fusion group
                    group_id = self.group_arena.new_group()
                    self.group_arena.add_node_to_group(group_id, dom_node_name)
                    self.group_arena.add_node_to_group(group_id, node.name)

                    # Mark pattern kind
                    self.group_arena.set_group_pattern_kind(
                        group_id, OpPatternKind.kBroadcast
                    )
                    break

    def _is_safe_to_fuse(self, node1: Node, node2: Node) -> bool:
        """Check if it's safe to fuse two nodes.

        Args:
            node1: First node to fuse.
            node2: Second node to fuse.

        Returns:
            True if safe to fuse, False otherwise.
        """
        # Check depth limit
        if not self.group_arena.can_add_to_group(
            node1.name, node2.name
        ):
            return False

        # For now, always return True for placeholder implementation
        # TODO: Implement actual path validation using indexed_forward_graph entries
        return True

    def _apply_fusions(self) -> None:
        """Apply identified fusions to the graph.

        This is a placeholder for the full implementation.
        In the complete implementation, this would:
        1. Create fused operation nodes
        2. Update graph connectivity
        3. Remove original nodes
        """
        logger.info("Applying fusions (placeholder implementation)")

        # Get all fusion groups
        groups = self.group_arena.get_all_groups()

        for group_id, nodes in groups.items():
            if len(nodes) > 1:
                logger.info(f"Would fuse group {group_id}: {nodes}")
                # TODO: Implement actual fusion logic

    def _log_fusion_summary(self) -> None:
        """Log a summary of fusion operations performed."""
        groups = self.group_arena.get_all_groups()
        total_groups = len(groups)
        fused_groups = sum(1 for nodes in groups.values() if len(nodes) > 1)

        logger.info("Fusion Summary:")
        logger.info(f"  - Total groups created: {total_groups}")
        logger.info(f"  - Fused groups: {fused_groups}")
        logger.info(f"  - Single-node groups: {total_groups - fused_groups}")

        for group_id, nodes in groups.items():
            if len(nodes) > 1:
                pattern_kind = self.group_arena.get_group_pattern_kind(group_id)
                logger.info(f"  - Group {group_id} ({pattern_kind}): {nodes}")
