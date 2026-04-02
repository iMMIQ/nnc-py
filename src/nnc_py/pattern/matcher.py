"""Pattern matching engine for finding pattern matches in graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nnc_py.ir.graph import Graph
from nnc_py.pattern.base import DFPattern, MatchContext, PatternMatch

if TYPE_CHECKING:
    from nnc_py.pattern.registry import FusionPattern


class PatternMatcher:
    """Efficient pattern matching engine for nnc-py graphs.

    Uses top-down DFS with memoization for efficient matching.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def match_pattern(self, pattern: DFPattern) -> list[PatternMatch]:
        """Find all matches of a pattern in the graph.

        Args:
            pattern: The pattern to match

        Returns:
            List of all successful matches (non-overlapping preferred)
        """
        matches = []
        context = MatchContext()

        # Try matching from each node (topological order preferred)
        for node in self.graph.topological_sort():
            match = pattern.match(node, self.graph, context)
            if match:
                matches.append(match)

        # Filter for non-overlapping matches (greedy by topological order)
        return self._filter_non_overlapping(matches)

    def _filter_non_overlapping(self, matches: list[PatternMatch]) -> list[PatternMatch]:
        """Filter matches to return non-overlapping ones.

        Uses greedy selection: prefer matches that appear earlier
        in topological order (upstream nodes first).
        """
        if not matches:
            return []

        # Sort by anchor position in topological order
        topo_nodes = self.graph.topological_sort()
        topo_order = {n.name: i for i, n in enumerate(topo_nodes)}
        matches.sort(key=lambda m: topo_order.get(m.anchor.name, float('inf')))

        selected: list[PatternMatch] = []
        used_names: set[str] = set()

        for match in matches:
            # Check if any node in this match is already used
            if match.node_names.isdisjoint(used_names):
                selected.append(match)
                used_names.update(match.node_names)

        return selected

    def match_all_patterns(
        self,
        patterns: list[tuple[DFPattern, FusionPattern]]
    ) -> list[tuple[PatternMatch, FusionPattern]]:
        """Match multiple patterns, returning (match, fusion_pattern) pairs.

        Args:
            patterns: List of (pattern, fusion_pattern) tuples ordered by priority

        Returns:
            List of matches with their associated fusion patterns
        """
        results = []

        for pattern, fusion_pattern in patterns:
            matches = self.match_pattern(pattern)
            for match in matches:
                results.append((match, fusion_pattern))

        # Filter by priority and overlap
        return self._select_by_priority(results)

    def _select_by_priority(
        self,
        results: list[tuple[PatternMatch, FusionPattern]]
    ) -> list[tuple[PatternMatch, FusionPattern]]:
        """Select matches based on pattern priority.

        Higher priority patterns are preferred. When conflicts occur,
        higher priority pattern wins.
        """
        # Sort by priority (higher first)
        results.sort(key=lambda x: x[1].priority, reverse=True)

        selected: list[tuple[PatternMatch, FusionPattern]] = []
        used_names: set[str] = set()

        for match, fusion_pattern in results:
            if match.node_names.isdisjoint(used_names):
                selected.append((match, fusion_pattern))
                used_names.update(match.node_names)

        return selected
