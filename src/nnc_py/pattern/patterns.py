"""Concrete pattern implementations."""

from typing import Optional
from nnc_py.pattern.base import DFPattern, PatternMatch, MatchContext
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.graph import Graph


class WildcardPattern(DFPattern):
    """Matches any node (like . in regex)."""

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # Wildcard matches anything
        return PatternMatch(
            bindings={self.name: node},
            anchor=node,
            nodes={node}
        )


class OpPattern(DFPattern):
    """Matches a specific operator type."""

    def __init__(self, op_type: OpType, name: Optional[str] = None):
        super().__init__(name)
        self.op_type = op_type

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # Check memoization
        cached = context.get_cached(node, self)
        if cached is not None:
            return cached

        # Match if operator types match
        if node.op_type != self.op_type:
            result = None
        else:
            result = PatternMatch(
                bindings={self.name: node},
                anchor=node,
                nodes={node}
            )

        context.set_cached(node, self, result)
        return result


class OrPattern(DFPattern):
    """Matches if either left OR right pattern matches."""

    def __init__(self, left: DFPattern, right: DFPattern):
        super().__init__()
        self.left = left
        self.right = right

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # Try left first, then right
        result = self.left.match(node, graph, context)
        if result is None:
            result = self.right.match(node, graph, context)
        return result


class AndPattern(DFPattern):
    """Matches if both patterns match (structural composition).

    Used for combining patterns that match different parts of the graph.
    """

    def __init__(self, left: DFPattern, right: DFPattern):
        super().__init__()
        self.left = left
        self.right = right

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # Try to match left pattern
        left_match = self.left.match(node, graph, context)
        if left_match is None:
            return None

        # Try to match right pattern with updated bindings
        right_match = self.right.match(node, graph, context.with_bindings(left_match.bindings))
        if right_match is None:
            return None

        # Merge bindings and nodes
        merged = left_match.bindings.copy()
        merged.update(right_match.bindings)
        merged_nodes = left_match.nodes | right_match.nodes

        # Anchor is the last matched node
        return PatternMatch(
            bindings=merged,
            anchor=right_match.anchor or left_match.anchor,
            nodes=merged_nodes
        )


class UsePattern(DFPattern):
    """Matches when producer's output is used by consumer pattern."""

    def __init__(self, producer: DFPattern, consumer: DFPattern):
        super().__init__()
        self.producer = producer
        self.consumer = consumer

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # Try to match producer at this node
        prod_match = self.producer.match(node, graph, context)
        if prod_match is None:
            return None

        # Check if any consumer matches
        anchor = prod_match.anchor
        if not anchor.outputs:
            return None

        for out_tensor in anchor.outputs:
            for consumer_node in graph.get_consumers(out_tensor):
                cons_match = self.consumer.match(
                    consumer_node,
                    graph,
                    context.with_bindings(prod_match.bindings)
                )
                if cons_match:
                    # Merge bindings
                    merged = prod_match.bindings.copy()
                    merged.update(cons_match.bindings)
                    merged_nodes = prod_match.nodes | cons_match.nodes
                    return PatternMatch(
                        bindings=merged,
                        anchor=cons_match.anchor,
                        nodes=merged_nodes
                    )
        return None


class ExclusiveUsePattern(DFPattern):
    """Matches when producer's output is ONLY used by consumer pattern.

    This is the key pattern for safe fusion - ensures the producer's
    output has exactly one consumer.
    """

    def __init__(self, producer: DFPattern, consumer: DFPattern):
        super().__init__()
        self.producer = producer
        self.consumer = consumer

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # Try to match producer at this node
        prod_match = self.producer.match(node, graph, context)
        if prod_match is None:
            return None

        # Check exclusive use condition
        anchor = prod_match.anchor
        if not anchor.outputs:
            return None

        for out_tensor in anchor.outputs:
            consumers = graph.get_consumers(out_tensor)
            if len(consumers) != 1:
                return None  # Not exclusively used

            cons_match = self.consumer.match(
                consumers[0],
                graph,
                context.with_bindings(prod_match.bindings)
            )
            if cons_match:
                merged = prod_match.bindings.copy()
                merged.update(cons_match.bindings)
                merged_nodes = prod_match.nodes | cons_match.nodes
                return PatternMatch(
                    bindings=merged,
                    anchor=cons_match.anchor,
                    nodes=merged_nodes
                )
        return None


class AttrPattern(DFPattern):
    """Matches nodes with specific attribute values."""

    def __init__(self, pattern: DFPattern, **attrs):
        super().__init__()
        self.pattern = pattern
        self.attrs = attrs

    def match(self, node: Node, graph: Graph, context: MatchContext) -> Optional[PatternMatch]:
        # First match inner pattern
        inner_match = self.pattern.match(node, graph, context)
        if inner_match is None:
            return None

        # Check attributes
        anchor = inner_match.anchor
        for key, expected_value in self.attrs.items():
            actual_value = anchor.attrs.get(key)
            if actual_value != expected_value:
                return None

        return inner_match
