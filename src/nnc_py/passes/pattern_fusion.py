"""Pattern-based operator fusion pass."""

from typing import Set
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node
from nnc_py.passes.base import PassBase
from nnc_py.pattern.matcher import PatternMatcher
from nnc_py.pattern.registry import PatternRegistry, FusionPattern
from nnc_py.pattern.base import PatternMatch


class PatternFusionPass(PassBase):
    """Fuse operators using declarative pattern matching.

    This pass uses the PatternRegistry to find and apply fusion patterns.
    It replaces the hardcoded pattern matching in OperatorFusionPass.
    """

    @property
    def name(self) -> str:
        return "PatternFusion"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute pattern-based fusion."""
        graph = ctx.graph

        # Get all registered patterns
        fusion_patterns = PatternRegistry.get_all()

        if not fusion_patterns:
            return  # No patterns registered

        # Build (pattern, fusion_pattern) pairs
        pattern_pairs = [(fp.pattern, fp) for fp in fusion_patterns]

        # Create matcher and find all matches
        matcher = PatternMatcher(graph)
        matches_with_patterns = matcher.match_all_patterns(pattern_pairs)

        # Apply fusions
        fused_nodes: Set[str] = set()
        patterns_found = {}

        for match, fusion_pattern in matches_with_patterns:
            # Skip if any nodes already fused
            if any(n in fused_nodes for n in match.node_names):
                continue

            # Run check function if provided
            if fusion_pattern.check_func:
                if not fusion_pattern.check_func(graph, match):
                    continue

            # Apply the fusion
            self._apply_fusion(graph, match, fusion_pattern, fused_nodes)
            patterns_found[fusion_pattern.name] = patterns_found.get(fusion_pattern.name, 0) + 1

        # Log results if debug mode
        if ctx.debug:
            self._log_summary(patterns_found, len(graph.nodes))

    def _apply_fusion(
        self,
        graph: Graph,
        match: PatternMatch,
        fusion_pattern: FusionPattern,
        fused_nodes: Set[str],
    ) -> None:
        """Apply a single fusion transformation.

        Args:
            graph: The computation graph
            match: The pattern match
            fusion_pattern: The fusion pattern definition
            fused_nodes: Set to track fused node names
        """
        # Generate unique name for fused node
        fused_name = f"fused_{fusion_pattern.name}_{len(fused_nodes) + 1}"

        # Create fused node
        if fusion_pattern.replace_func:
            fused_node = fusion_pattern.replace_func(graph, match, fused_name)
        elif fusion_pattern.fused_op_type:
            fused_node = self._default_fusion(graph, match, fusion_pattern, fused_name)
        else:
            raise ValueError(f"Fusion pattern {fusion_pattern.name} has no replacement")

        # Add fused node to graph
        graph.add_node(fused_node)

        # Update graph outputs if needed
        self._update_graph_outputs(graph, match, fused_node)

        # Remove original nodes
        for node_name in match.node_names:
            if node_name in graph.nodes:
                del graph.nodes[node_name]
                fused_nodes.add(node_name)

    def _default_fusion(
        self,
        graph: Graph,
        match: PatternMatch,
        fusion_pattern: FusionPattern,
        fused_name: str,
    ) -> Node:
        """Default fusion behavior when no replace_func provided.

        Assumes a simple chain pattern and takes inputs from the first node
        and outputs from the last node.
        """
        # Get nodes in topological order
        topo_nodes = graph.topological_sort()
        matched_nodes = [n for n in topo_nodes if n.name in match.node_names]

        if not matched_nodes:
            # Fallback: get nodes from bindings
            matched_nodes = list(match.bindings.values())

        if not matched_nodes:
            raise ValueError(f"Cannot determine nodes for fusion {fusion_pattern.name}")

        first_node = matched_nodes[0]
        last_node = matched_nodes[-1]

        return Node(
            op_type=fusion_pattern.fused_op_type,
            name=fused_name,
            inputs=list(first_node.inputs),
            outputs=list(last_node.outputs),
            attrs=first_node.attrs.copy(),
            metadata={"fused_from": [n.name for n in matched_nodes]}
        )

    def _update_graph_outputs(
        self,
        graph: Graph,
        match: PatternMatch,
        fused_node: Node,
    ) -> None:
        """Update graph outputs after fusion."""
        # Find all outputs from matched nodes
        for node_name in match.node_names:
            if node_name not in graph.nodes:
                continue
            node = graph.nodes[node_name]
            for out_tensor in node.outputs:
                if out_tensor in graph.outputs:
                    # Replace with fused node's outputs
                    for i, old_out in enumerate(graph.outputs):
                        if old_out == out_tensor:
                            graph.outputs[i] = fused_node.outputs[0]

    def _log_summary(self, patterns_found: dict, node_count: int) -> None:
        """Log fusion summary."""
        print(f"\n{'='*60}")
        print(f"Pattern Fusion Summary")
        print(f"{'='*60}")
        print(f"Total fusions: {sum(patterns_found.values())}")
        print(f"Patterns found:")
        for pattern, count in sorted(patterns_found.items()):
            print(f"  - {pattern}: {count}")
        print(f"Nodes after fusion: {node_count}")
        print(f"{'='*60}")
