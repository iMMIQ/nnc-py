"""Built-in fusion pattern definitions."""

from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.pattern.base import PatternMatch
from nnc_py.pattern.patterns import OpPattern
from nnc_py.pattern.registry import register_pattern


# Pattern building helpers
def conv(name: str = "conv") -> OpPattern:
    """Create a Conv2D pattern."""
    return OpPattern(OpType.CONV2D, name)

def matmul(name: str = "matmul") -> OpPattern:
    """Create a MatMul pattern."""
    return OpPattern(OpType.MATMUL, name)

def add(name: str = "add") -> OpPattern:
    """Create an Add pattern."""
    return OpPattern(OpType.ADD, name)

def relu(name: str = "relu") -> OpPattern:
    """Create a ReLU pattern."""
    return OpPattern(OpType.RELU, name)

def sigmoid(name: str = "sigmoid") -> OpPattern:
    """Create a Sigmoid pattern."""
    return OpPattern(OpType.SIGMOID, name)

# Fusion helpers
def _create_fused_conv_relu(graph: Graph, match: PatternMatch, name: str) -> Node:
    """Create fused Conv+ReLU node."""
    conv_node = match.bindings["conv"]
    relu_node = match.bindings["relu"]
    return Node(
        op_type=OpType.FUSED_CONV_RELU,
        name=name,
        inputs=list(conv_node.inputs),
        outputs=list(relu_node.outputs),
        attrs=conv_node.attrs.copy(),
        metadata={"fused_from": [conv_node.name, relu_node.name]}
    )


def _create_fused_conv_sigmoid(graph: Graph, match: PatternMatch, name: str) -> Node:
    """Create fused Conv+Sigmoid node."""
    conv_node = match.bindings["conv"]
    sigmoid_node = match.bindings["sigmoid"]
    return Node(
        op_type=OpType.FUSED_CONV_SIGMOID,
        name=name,
        inputs=list(conv_node.inputs),
        outputs=list(sigmoid_node.outputs),
        attrs=conv_node.attrs.copy(),
        metadata={"fused_from": [conv_node.name, sigmoid_node.name]}
    )


def _create_fused_add_relu(graph: Graph, match: PatternMatch, name: str) -> Node:
    """Create fused Add+ReLU node."""
    add_node = match.bindings["add"]
    relu_node = match.bindings["relu"]
    return Node(
        op_type=OpType.FUSED_ADD_RELU,
        name=name,
        inputs=list(add_node.inputs),
        outputs=list(relu_node.outputs),
        attrs=add_node.attrs.copy(),
        metadata={"fused_from": [add_node.name, relu_node.name]}
    )


def _create_fused_add_sigmoid(graph: Graph, match: PatternMatch, name: str) -> Node:
    """Create fused Add+Sigmoid node."""
    add_node = match.bindings["add"]
    sigmoid_node = match.bindings["sigmoid"]
    return Node(
        op_type=OpType.FUSED_ADD_SIGMOID,
        name=name,
        inputs=list(add_node.inputs),
        outputs=list(sigmoid_node.outputs),
        attrs=add_node.attrs.copy(),
        metadata={"fused_from": [add_node.name, sigmoid_node.name]}
    )


def _create_fused_matmul_relu(graph: Graph, match: PatternMatch, name: str) -> Node:
    """Create fused MatMul+ReLU node."""
    matmul_node = match.bindings["matmul"]
    relu_node = match.bindings["relu"]
    return Node(
        op_type=OpType.FUSED_MATMUL_RELU,
        name=name,
        inputs=list(matmul_node.inputs),
        outputs=list(relu_node.outputs),
        attrs=matmul_node.attrs.copy(),
        metadata={"fused_from": [matmul_node.name, relu_node.name]}
    )


# Register built-in patterns

# Conv + ReLU (high priority - common pattern)
register_pattern(
    name="conv_relu",
    pattern=conv().only_used_by(relu()),
    priority=200,
    description="Conv + ReLU fusion",
    fused_op_type=OpType.FUSED_CONV_RELU,
    replace_func=_create_fused_conv_relu,
)

# Conv + Sigmoid
register_pattern(
    name="conv_sigmoid",
    pattern=conv().only_used_by(sigmoid()),
    priority=200,
    description="Conv + Sigmoid fusion",
    fused_op_type=OpType.FUSED_CONV_SIGMOID,
    replace_func=_create_fused_conv_sigmoid,
)

# Add + ReLU
register_pattern(
    name="add_relu",
    pattern=add().only_used_by(relu()),
    priority=200,
    description="Add + ReLU fusion",
    fused_op_type=OpType.FUSED_ADD_RELU,
    replace_func=_create_fused_add_relu,
)

# Add + Sigmoid
register_pattern(
    name="add_sigmoid",
    pattern=add().only_used_by(sigmoid()),
    priority=200,
    description="Add + Sigmoid fusion",
    fused_op_type=OpType.FUSED_ADD_SIGMOID,
    replace_func=_create_fused_add_sigmoid,
)

# MatMul + ReLU
register_pattern(
    name="matmul_relu",
    pattern=matmul().only_used_by(relu()),
    priority=190,
    description="MatMul + ReLU fusion",
    fused_op_type=OpType.FUSED_MATMUL_RELU,
    replace_func=_create_fused_matmul_relu,
)
