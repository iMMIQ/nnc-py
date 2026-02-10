"""Operator fusion pass.

This pass fuses compatible operator patterns (e.g., Conv+ReLU, Add+Activation)
into single fused operations for improved performance.
"""

from typing import Dict, Optional, Set

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.passes.base import PassBase


class OperatorFusionPass(PassBase):
    """Fuse compatible operator patterns into single fused operations.

    This pass identifies and fuses common operator patterns:
    - Conv + ReLU -> FUSED_CONV_RELU
    - Add + ReLU -> FUSED_ADD_RELU
    - Conv + Sigmoid -> FUSED_CONV_SIGMOID
    - Add + Sigmoid -> FUSED_ADD_SIGMOID

    Fusion is only performed when:
    1. The producer's output has exactly one consumer
    2. The pattern is recognized and safe to fuse
    3. The fusion would preserve graph semantics
    """

    @property
    def name(self) -> str:
        return "OperatorFusion"

    def _execute(self, ctx: CompileContext) -> None:
        """Execute operator fusion."""
        graph = ctx.graph

        # Keep track of fused nodes to avoid double-processing
        fused_nodes: Set[str] = set()

        # Get nodes in topological order for deterministic processing
        nodes = graph.topological_sort()

        # Track fusion statistics
        fusion_count = 0
        patterns_found: Dict[str, int] = {}

        for node in nodes:
            # Skip already fused nodes
            if node.name in fused_nodes:
                continue

            # Try to fuse this node with its producer
            fusion_result = self._try_fusion_with_producer(graph, node, fused_nodes)

            if fusion_result:
                fusion_count += 1
                pattern_name = fusion_result
                patterns_found[pattern_name] = patterns_found.get(pattern_name, 0) + 1

        # Log summary if debug mode is on
        if ctx.debug:
            self._log_summary(fusion_count, patterns_found, len(graph.nodes))

    def _try_fusion_with_producer(
        self,
        graph: Graph,
        consumer: Node,
        fused_nodes: Set[str],
    ) -> Optional[str]:
        """Try to fuse the consumer node with its producer.

        Args:
            graph: The computation graph
            consumer: The consumer node
            fused_nodes: Set of already fused node names

        Returns:
            Pattern name if fusion occurred, None otherwise
        """
        # Only fuse single-input consumers (element-wise ops)
        if len(consumer.inputs) != 1:
            return None

        input_tensor = consumer.inputs[0]

        # Find producers of this tensor
        producers = graph.get_producers(input_tensor)

        # Need exactly one producer for fusion
        if len(producers) != 1:
            return None

        producer = producers[0]

        # Don't fuse if producer is already fused
        if producer.name in fused_nodes:
            return None

        # Check that producer's output has only one consumer
        consumers = graph.get_consumers(input_tensor)
        if len(consumers) != 1:
            return None

        # Try specific fusion patterns
        if producer.op_type == OpType.CONV2D and consumer.op_type == OpType.RELU:
            return self._fuse_conv_relu(graph, producer, consumer, fused_nodes)
        elif producer.op_type == OpType.CONV2D and consumer.op_type == OpType.SIGMOID:
            return self._fuse_conv_sigmoid(graph, producer, consumer, fused_nodes)
        elif producer.op_type == OpType.ADD and consumer.op_type == OpType.RELU:
            return self._fuse_add_relu(graph, producer, consumer, fused_nodes)
        elif producer.op_type == OpType.ADD and consumer.op_type == OpType.SIGMOID:
            return self._fuse_add_sigmoid(graph, producer, consumer, fused_nodes)

        return None

    def _fuse_conv_relu(
        self,
        graph: Graph,
        conv: Node,
        relu: Node,
        fused_nodes: Set[str],
    ) -> str:
        """Fuse Conv + ReLU into FUSED_CONV_RELU."""
        # Create fused node
        fused_node = Node(
            op_type=OpType.FUSED_CONV_RELU,
            name=f"fused_conv_relu_{len(fused_nodes) + 1}",
            inputs=conv.inputs,  # Take conv inputs
            outputs=relu.outputs,  # Output relu's output
            attrs=conv.attrs.copy(),  # Copy conv attributes
            metadata={"fused_from": [conv.name, relu.name]},
        )
        graph.add_node(fused_node)

        # Update graph outputs if needed
        self._update_graph_outputs(graph, conv.outputs[0], relu.outputs[0])

        # Remove original nodes
        del graph.nodes[conv.name]
        del graph.nodes[relu.name]

        # Mark as fused
        fused_nodes.add(conv.name)
        fused_nodes.add(relu.name)

        return "Conv+ReLU"

    def _fuse_conv_sigmoid(
        self,
        graph: Graph,
        conv: Node,
        sigmoid: Node,
        fused_nodes: Set[str],
    ) -> str:
        """Fuse Conv + Sigmoid into FUSED_CONV_SIGMOID."""
        fused_node = Node(
            op_type=OpType.FUSED_CONV_SIGMOID,
            name=f"fused_conv_sigmoid_{len(fused_nodes) + 1}",
            inputs=conv.inputs,
            outputs=sigmoid.outputs,
            attrs=conv.attrs.copy(),
            metadata={"fused_from": [conv.name, sigmoid.name]},
        )
        graph.add_node(fused_node)

        self._update_graph_outputs(graph, conv.outputs[0], sigmoid.outputs[0])

        del graph.nodes[conv.name]
        del graph.nodes[sigmoid.name]

        fused_nodes.add(conv.name)
        fused_nodes.add(sigmoid.name)

        return "Conv+Sigmoid"

    def _fuse_add_relu(
        self,
        graph: Graph,
        add: Node,
        relu: Node,
        fused_nodes: Set[str],
    ) -> str:
        """Fuse Add + ReLU into FUSED_ADD_RELU."""
        fused_node = Node(
            op_type=OpType.FUSED_ADD_RELU,
            name=f"fused_add_relu_{len(fused_nodes) + 1}",
            inputs=add.inputs,
            outputs=relu.outputs,
            attrs=add.attrs.copy(),
            metadata={"fused_from": [add.name, relu.name]},
        )
        graph.add_node(fused_node)

        self._update_graph_outputs(graph, add.outputs[0], relu.outputs[0])

        del graph.nodes[add.name]
        del graph.nodes[relu.name]

        fused_nodes.add(add.name)
        fused_nodes.add(relu.name)

        return "Add+ReLU"

    def _fuse_add_sigmoid(
        self,
        graph: Graph,
        add: Node,
        sigmoid: Node,
        fused_nodes: Set[str],
    ) -> str:
        """Fuse Add + Sigmoid into FUSED_ADD_SIGMOID."""
        fused_node = Node(
            op_type=OpType.FUSED_ADD_SIGMOID,
            name=f"fused_add_sigmoid_{len(fused_nodes) + 1}",
            inputs=add.inputs,
            outputs=sigmoid.outputs,
            attrs=add.attrs.copy(),
            metadata={"fused_from": [add.name, sigmoid.name]},
        )
        graph.add_node(fused_node)

        self._update_graph_outputs(graph, add.outputs[0], sigmoid.outputs[0])

        del graph.nodes[add.name]
        del graph.nodes[sigmoid.name]

        fused_nodes.add(add.name)
        fused_nodes.add(sigmoid.name)

        return "Add+Sigmoid"

    def _update_graph_outputs(self, graph: Graph, old_tensor: str, new_tensor: str) -> None:
        """Update graph outputs if old_tensor was an output."""
        if old_tensor in graph.outputs:
            # Replace old tensor with new tensor
            new_outputs = [new_tensor if t == old_tensor else t for t in graph.outputs]
            graph.outputs = new_outputs

    def _log_summary(self, fusion_count: int, patterns_found: Dict[str, int], node_count: int) -> None:
        """Log a summary of fusion results."""
        print(f"\n{'='*60}")
        print(f"Operator Fusion Summary")
        print(f"{'='*60}")
        print(f"Total fusions: {fusion_count}")
        print(f"Patterns found:")
        for pattern, count in sorted(patterns_found.items()):
            print(f"  - {pattern}: {count}")
        print(f"Nodes after fusion: {node_count}")
        print(f"{'='*60}")
