"""Computation graph structure."""

from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from nnc_py.ir.node import Node
from nnc_py.ir.tensor import TensorType


class Graph:
    """Computation graph - core IR structure."""

    def __init__(self, name: str = "main"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.tensors: Dict[str, TensorType] = {}
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.constants: Dict[str, np.ndarray] = {}

        # NetworkX graph for analysis
        self._nx_graph: Optional[nx.DiGraph] = None

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.name] = node
        self._nx_graph = None  # Invalidate cached graph

    def add_tensor(self, tensor: TensorType) -> None:
        """Add a tensor definition to the graph."""
        self.tensors[tensor.name] = tensor

    def get_node(self, name: str) -> Node:
        """Get a node by name."""
        return self.nodes[name]

    def get_tensor(self, name: str) -> TensorType:
        """Get a tensor by name."""
        return self.tensors[name]

    def get_producers(self, tensor_name: str) -> List[Node]:
        """Get all nodes that produce the given tensor."""
        return [
            node for node in self.nodes.values() if tensor_name in node.outputs
        ]

    def get_consumers(self, tensor_name: str) -> List[Node]:
        """Get all nodes that consume the given tensor."""
        return [
            node for node in self.nodes.values() if tensor_name in node.inputs
        ]

    def topological_sort(self) -> List[Node]:
        """Return nodes in topological order."""
        self._ensure_nx_graph()
        sorted_names = list(nx.topological_sort(self._nx_graph))
        return [self.nodes[name] for name in sorted_names]

    def _ensure_nx_graph(self):
        """Ensure the NetworkX graph is built."""
        if self._nx_graph is None:
            self._build_graph()

    def _build_graph(self) -> None:
        """Build the NetworkX graph structure."""
        self._nx_graph = nx.DiGraph()

        # Add all nodes
        for node_name, node in self.nodes.items():
            self._nx_graph.add_node(node_name, node=node)

        # Add edges based on tensor dependencies
        for node_name, node in self.nodes.items():
            for input_tensor in node.inputs:
                # Find the producer of this tensor
                for producer in self.get_producers(input_tensor):
                    self._nx_graph.add_edge(producer.name, node_name)

    def __repr__(self) -> str:
        return (
            f"Graph(name={self.name}, "
            f"nodes={len(self.nodes)}, "
            f"inputs={len(self.inputs)}, "
            f"outputs={len(self.outputs)})"
        )
