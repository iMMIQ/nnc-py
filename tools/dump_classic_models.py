#!/usr/bin/env python3
"""Dump classic ML models to ONNX format for snapshot testing.

Usage:
    python tools/dump_classic_models.py

This script exports classic neural network models to ONNX format
which can be used for snapshot testing of the nnc-py compiler.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models


class LeNet5(nn.Module):
    """LeNet-5: Classic CNN for MNIST handwritten digit recognition.

    Original architecture by Yann LeCun et al., 1998.
    Structure: Input -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> ReLU -> FC -> ReLU -> FC
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: tuple,
    model_name: str,
    opset_version: int = 13,
) -> None:
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export.
        output_path: Path to save the ONNX model.
        input_shape: Input tensor shape (batch, channels, height, width).
        model_name: Name of the model for logging.
        opset_version: ONNX opset version to use.
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    print(f"Exporting {model_name} to {output_path}...")

    # Use legacy exporter for better opset 13 compatibility
    # This avoids Shape operators that aren't supported by nnc-py
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # Use legacy exporter to avoid Shape op in newer torch versions
        dynamo=False,
    )

    # Verify the model
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Print model info
    print(f"  Input shape: {input_shape}")
    print(f"  ONNX opset: {opset_version}")
    print(f"  Model size: {output_path.stat().st_size / 1024:.1f} KB")

    # Count ops
    ops = {}
    for node in onnx_model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"  Operators: {dict(ops)}")


def export_lenet5(output_dir: Path) -> Path:
    """Export LeNet-5 model to ONNX.

    Args:
        output_dir: Directory to save the model.

    Returns:
        Path to the exported model.
    """
    model = LeNet5(num_classes=10)
    output_path = output_dir / "lenet5.onnx"
    export_to_onnx(model, output_path, (1, 1, 28, 28), "LeNet-5")
    return output_path


def export_resnet18(output_dir: Path) -> Path:
    """Export ResNet-18 model to ONNX.

    Args:
        output_dir: Directory to save the model.

    Returns:
        Path to the exported model.
    """
    # Load pretrained ResNet-18
    model = models.resnet18(weights=None)
    model.eval()

    output_path = output_dir / "resnet18.onnx"
    export_to_onnx(model, output_path, (1, 3, 224, 224), "ResNet-18")
    return output_path


def export_simple_cnn(output_dir: Path) -> Path:
    """Export a simple CNN model to ONNX.

    This is a minimal CNN useful for quick testing.

    Args:
        output_dir: Directory to save the model.

    Returns:
        Path to the exported model.
    """
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 8 * 8, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = SimpleCNN()
    output_path = output_dir / "simple_cnn.onnx"
    export_to_onnx(model, output_path, (1, 3, 32, 32), "Simple CNN")
    return output_path


def export_simple_mlp(output_dir: Path) -> Path:
    """Export a simple MLP model to ONNX.

    Args:
        output_dir: Directory to save the model.

    Returns:
        Path to the exported model.
    """
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleMLP()
    output_path = output_dir / "simple_mlp.onnx"
    export_to_onnx(model, output_path, (1, 1, 28, 28), "Simple MLP")
    return output_path


class SelfAttention(nn.Module):
    """Single self-attention layer without multi-head complexity.

    This is the core building block of Transformer architectures.
    Uses scaled dot-product attention: softmax(QK^T / sqrt(d_k))V

    Note: Uses separate Q, K, V projections to avoid torch.chunk which
    generates ONNX Slice/Gather/Shape operators.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        # Separate Query, Key, Value projections
        # Using separate layers avoids chunk() which creates Slice/Gather ops
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        """Input shape: (batch, seq_len, embed_dim)."""
        # Generate Q, K, V using separate projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Scaled dot-product attention
        # (batch, seq_len, embed_dim) @ (batch, embed_dim, seq_len)
        # -> (batch, seq_len, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply attention to values
        # (batch, seq_len, seq_len) @ (batch, seq_len, embed_dim)
        # -> (batch, seq_len, embed_dim)
        attn_output = attn_weights @ v

        # Output projection
        return self.out_proj(attn_output)


class SimpleTransformer(nn.Module):
    """A minimal Transformer model for testing.

    Architecture:
    1. Token embedding (Linear projection)
    2. 2 Self-Attention layers with ReLU activation
    3. Layer normalization
    4. Classification head

    This is NOT a full BERT/GPT-style transformer, but focuses on
    the core attention mechanism that defines the architecture.
    """

    def __init__(self, seq_len: int = 16, input_dim: int = 32,
                 embed_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Input embedding: projects (seq_len, input_dim) -> (seq_len, embed_dim)
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Two self-attention layers
        self.attn1 = SelfAttention(embed_dim)
        self.attn2 = SelfAttention(embed_dim)

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head: global average pool -> linear
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        """Input shape: (batch, seq_len, input_dim)."""
        # Embedding
        x = self.embedding(x)
        x = torch.relu(x)

        # Self-attention layers
        x = self.attn1(x)
        x = torch.relu(x)
        x = self.attn2(x)
        x = torch.relu(x)

        # Layer norm
        x = self.norm(x)

        # Classification
        x = self.classifier(x)
        return x


def export_simple_transformer(output_dir: Path) -> Path:
    """Export a simple Transformer model to ONNX.

    This model demonstrates self-attention mechanism, the core of
    modern transformer architectures used in BERT, GPT, etc.

    Args:
        output_dir: Directory to save the model.

    Returns:
        Path to the exported model.
    """
    model = SimpleTransformer(
        seq_len=16,      # Short sequence for quick testing
        input_dim=32,     # Input feature dimension per token
        embed_dim=64,     # Attention embedding dimension
        num_classes=10,
    )
    output_path = output_dir / "simple_transformer.onnx"
    export_to_onnx(model, output_path, (1, 16, 32), "Simple Transformer")
    return output_path


class OperatorCoverageModel(nn.Module):
    """A model designed to cover all core ONNX operators supported by nnc-py.

    This model includes the following operators:
    - Add, And, Cast, Clip, Concat, Constant, Div, Equal
    - MatMul, Mul, Not, Or, Pow, ReduceMean, ReduceSum
    - Relu, Reshape, Split, Sqrt, Sub, Tile, Transpose, Unsqueeze

    The model performs a series of transformations on input data:
    1. Feature processing with arithmetic ops (Add, Sub, Mul, Div, Pow, Sqrt)
    2. Logical operations (And, Or, Not, Equal)
    3. Shape manipulation (Reshape, Transpose, Unsqueeze, Split, Concat, Tile)
    4. Reduction operations (ReduceMean, ReduceSum)
    5. Activation (Relu, Clip)
    """

    def __init__(self):
        super().__init__()
        # Learnable parameters for various operations
        self.weight = nn.Parameter(torch.randn(4, 8))
        self.bias = nn.Parameter(torch.randn(8))
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        """Input shape: (batch, 4, 8)"""

        # Constant (implicit in ONNX through learnable parameters)
        constant_val = torch.tensor(1.0, dtype=x.dtype)

        # Add: x + constant
        x = x + constant_val

        # Mul: element-wise multiplication
        x = x * self.scale

        # Sub: subtraction
        x = x - 0.5

        # Div: division
        x = x / 2.0

        # Pow: power operation
        x = torch.pow(x, 2.0)

        # Sqrt: square root (use Abs to avoid negative values, but that's another op)
        # Using max(0, x) to avoid negative instead of abs
        x_sqrt_input = torch.clamp(x, min=0.0) + 1e-6
        x = torch.sqrt(x_sqrt_input)

        # Clip: clamp values
        x = torch.clamp(x, min=-1.0, max=1.0)

        # Relu: activation
        x = torch.relu(x)

        # MatMul: matrix multiplication
        # x shape: (batch, 4, 8) @ (8, 4) -> (batch, 4, 4)
        x = torch.matmul(x, self.weight.t())

        # Reshape: flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # (batch, 16)

        # ReduceMean: mean along dimension
        mean_val = torch.mean(x, dim=1, keepdim=True)  # (batch, 1)

        # ReduceSum: sum along dimension
        sum_val = torch.sum(x, dim=1, keepdim=True)  # (batch, 1)

        # Concat: concatenate tensors
        x = torch.cat([mean_val, sum_val], dim=1)  # (batch, 2)

        # Unsqueeze: add dimension
        x = x.unsqueeze(-1)  # (batch, 2, 1)

        # Tile: repeat along dimensions
        x = x.repeat(1, 1, 4)  # (batch, 2, 4)

        # Transpose: swap dimensions
        x = x.transpose(1, 2)  # (batch, 4, 2)

        # Split: split tensor into parts
        x1, x2 = torch.split(x, 1, dim=2)  # each: (batch, 4, 1)

        # Equal: comparison (returns bool) - squeeze to make shapes match
        # Use a slice of x1 to create equal comparison
        x1_sliced = x1[:, :, 0:1]  # (batch, 4, 1)
        x2_sliced = x1[:, :, 0:1]  # (batch, 4, 1) - same tensor, so Equal is True
        eq_result = torch.eq(x1_sliced, x2_sliced)  # (batch, 4, 1) - all True

        # Cast: bool to float
        eq_result = eq_result.to(torch.float32)

        # And: logical AND (use Greater comparisons)
        x1_gt_zero = x1 > 0  # bool: (batch, 4, 1)
        x1_gt_one = x1 > 1.0  # bool: (batch, 4, 1)
        and_result = x1_gt_zero & x1_gt_one  # both must be True

        # Or: logical OR
        or_result = x1_gt_zero | x1_gt_one  # at least one True

        # Not: logical NOT
        not_result = torch.logical_not(x1_gt_zero)

        # Convert boolean results to float and combine
        and_result = and_result.to(torch.float32)
        or_result = or_result.to(torch.float32)
        not_result = not_result.to(torch.float32)

        # Combine all results
        x = torch.cat([and_result, or_result, not_result], dim=1)  # (batch, 4, 3)

        # Final aggregation
        x = x.mean(dim=1)  # (batch, 3)

        return x


class SimpleLSTM(nn.Module):
    """A simple LSTM model for sequence processing.

    This model demonstrates LSTM (Long Short-Term Memory) functionality,
    which is the core of many sequence-to-sequence models used in NLP,
    time series forecasting, and more.

    Architecture:
    1. Input embedding (Linear projection)
    2. LSTM layer
    3. Output head (Linear)

    The LSTM processes sequences and maintains hidden state across time steps.
    """

    def __init__(self, input_size: int = 32, hidden_size: int = 64,
                 num_layers: int = 1, num_classes: int = 10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input embedding: projects (input_dim) -> (hidden_size)
        self.embedding = nn.Linear(input_size, hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # Classification head: use final hidden state
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        """Input shape: (batch, seq_len, input_dim)."""
        # Embedding
        x = self.embedding(x)
        x = torch.relu(x)

        # LSTM: output shape (batch, seq_len, hidden_size)
        # hidden shape (num_layers, batch, hidden_size)
        lstm_out, (hidden, _) = self.lstm(x)

        # Use the final hidden state from the last layer
        final_hidden = hidden[-1]  # (batch, hidden_size)

        # Classification
        x = self.classifier(final_hidden)
        return x


def export_simple_lstm(output_dir: Path) -> Path:
    """Export a simple LSTM model to ONNX.

    This model demonstrates LSTM functionality, the core of recurrent
    neural networks used for sequence processing tasks.

    Args:
        output_dir: Directory to save the model.

    Returns:
        Path to the exported model.
    """
    model = SimpleLSTM(
        input_size=32,     # Input feature dimension per time step
        hidden_size=64,    # LSTM hidden state size
        num_layers=1,      # Single LSTM layer
        num_classes=10,
    )
    output_path = output_dir / "simple_lstm.onnx"
    export_to_onnx(model, output_path, (1, 16, 32), "Simple LSTM")
    return output_path


class LargeOperatorCoverageModel(nn.Module):
    """Operator coverage model with large parameters for memory constraint testing.

    This model uses the same operator set as OperatorCoverageModel but with
    larger tensor dimensions to force memory spill under constraints:
    - Input shape: (4, 64, 128) instead of (1, 4, 8)
    - Weight matrices: 128x64 instead of 8x4
    - Multiple intermediate tensors that must spill to slow memory

    Operators covered (same as small model):
    - Arithmetic: Add, Sub, Mul, Div, Pow, Sqrt
    - Logical: And, Or, Not, Equal
    - Shape: Reshape, Transpose, Unsqueeze, Split, Concat, Tile
    - Reduction: ReduceMean, ReduceSum
    - Activation: Relu, Clip
    - Other: Constant, MatMul, Cast
    """

    def __init__(self):
        super().__init__()
        # Large learnable parameters
        self.weight = nn.Parameter(torch.randn(128, 64))
        self.bias = nn.Parameter(torch.randn(128))
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        """Input shape: (batch, 64, 128)"""
        # Constant operations
        constant_val = torch.tensor(1.0, dtype=x.dtype)

        # Arithmetic chain
        x = x + constant_val          # Add
        x = x * self.scale            # Mul
        x = x - 0.5                   # Sub
        x = x / 2.0                   # Div
        x = torch.pow(x, 2.0)         # Pow

        # Sqrt with non-negative input
        x_sqrt_input = torch.clamp(x, min=0.0) + 1e-6
        x = torch.sqrt(x_sqrt_input)  # Sqrt

        # Clip and Relu
        x = torch.clamp(x, min=-1.0, max=1.0)  # Clip
        x = torch.relu(x)                       # Relu

        # MatMul: x is (batch, 64, 128), weight is (128, 64)
        # We want (batch, 64, 128) @ (128, 64) -> (batch, 64, 64)
        # So use weight directly (not transposed)
        x = torch.matmul(x, self.weight)       # (batch, 64, 64)

        # Reshape
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # (batch, 4096)

        # Reductions
        mean_val = torch.mean(x, dim=1, keepdim=True)  # (batch, 1)
        sum_val = torch.sum(x, dim=1, keepdim=True)    # (batch, 1)

        # Concat
        x = torch.cat([mean_val, sum_val], dim=1)  # (batch, 2)

        # Unsqueeze
        x = x.unsqueeze(-1)  # (batch, 2, 1)

        # Tile (repeat)
        x = x.repeat(1, 1, 64)  # (batch, 2, 64)

        # Transpose
        x = x.transpose(1, 2)  # (batch, 64, 2)

        # Split
        x1, x2 = torch.split(x, 1, dim=2)  # each: (batch, 64, 1)

        # Logical operations with comparisons
        x1_gt_zero = x1 > 0
        x1_gt_one = x1 > 1.0
        x2_gt_zero = x2 > 0

        # And, Or, Not
        and_result = x1_gt_zero & x1_gt_one
        or_result = x1_gt_zero | x2_gt_zero
        not_result = torch.logical_not(x1_gt_zero)

        # Cast to float
        and_result = and_result.to(torch.float32)
        or_result = or_result.to(torch.float32)
        not_result = not_result.to(torch.float32)

        # Combine results
        x = torch.cat([and_result, or_result, not_result], dim=1)  # (batch, 64, 3)

        # Final aggregation
        x = x.mean(dim=1)  # (batch, 3)

        return x


def export_operator_coverage_model(output_dir: Path) -> Path:
    """Export the operator coverage model to ONNX.

    This model is designed to test all core ONNX operators supported by nnc-py.

    Args:
        output_dir: Directory to save the model.

    Returns:
        Path to the exported model.
    """
    import onnx
    from onnx import helper

    model = OperatorCoverageModel()
    output_path = output_dir / "operator_coverage.onnx"

    # First export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 4, 8)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )

    # PyTorch's ONNX export sometimes optimizes away the Equal operator.
    # Manually add it to ensure it's present for coverage testing.
    onnx_model = onnx.load(output_path)
    ops = {node.op_type for node in onnx_model.graph.node}

    if "Equal" not in ops:
        # Find a Split output to connect Equal to
        split_output = None
        for node in onnx_model.graph.node:
            if node.op_type == "Split":
                split_output = node.output[0]
                break

        if split_output:
            # Create an Equal node: split_output == 0
            equal_node = helper.make_node(
                "Equal",
                inputs=[split_output, "const_zero"],
                outputs=["equal_output"],
                name="Equal_manual",
            )

            # Add constant zero initializer
            const_zero = helper.make_tensor(
                "const_zero", onnx.TensorProto.FLOAT, [1], [0.0]
            )
            onnx_model.graph.initializer.append(const_zero)

            # Insert Equal node after Split
            split_idx = next(
                i for i, n in enumerate(onnx_model.graph.node) if n.op_type == "Split"
            )
            onnx_model.graph.node.insert(split_idx + 1, equal_node)

            # Save updated model
            onnx.save(onnx_model, output_path)

    # Verify the model
    onnx.checker.check_model(onnx_model)

    # Print model info
    print(f"  Input shape: (1, 4, 8)")
    print(f"  ONNX opset: 13")
    print(f"  Model size: {output_path.stat().st_size / 1024:.1f} KB")

    # Count ops
    ops = {}
    for node in onnx_model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"  Operators: {dict(ops)}")

    return output_path


def export_large_operator_coverage_model(output_dir: Path) -> Path:
    """Export the large operator coverage model to ONNX for memory constraint testing.

    This model uses the same operators as operator_coverage.onnx but with larger
    tensor dimensions to force memory spill under constraints.

    Args:
        output_dir: Directory to save the model.

    Returns:
        Path to the exported model.
    """
    import onnx
    from onnx import helper

    model = LargeOperatorCoverageModel()
    output_path = output_dir / "operator_coverage_large.onnx"

    # Export to ONNX with large input shape
    model.eval()
    dummy_input = torch.randn(4, 64, 128)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )

    # Manually add Equal operator if not present
    onnx_model = onnx.load(output_path)
    ops = {node.op_type for node in onnx_model.graph.node}

    if "Equal" not in ops:
        split_output = None
        for node in onnx_model.graph.node:
            if node.op_type == "Split":
                split_output = node.output[0]
                break

        if split_output:
            equal_node = helper.make_node(
                "Equal",
                inputs=[split_output, "const_zero"],
                outputs=["equal_output"],
                name="Equal_manual",
            )
            const_zero = helper.make_tensor(
                "const_zero", onnx.TensorProto.FLOAT, [1], [0.0]
            )
            onnx_model.graph.initializer.append(const_zero)

            split_idx = next(
                i for i, n in enumerate(onnx_model.graph.node) if n.op_type == "Split"
            )
            onnx_model.graph.node.insert(split_idx + 1, equal_node)
            onnx.save(onnx_model, output_path)
            onnx_model = onnx.load(output_path)

    onnx.checker.check_model(onnx_model)

    # Print model info
    print(f"  Input shape: (4, 64, 128)")
    print(f"  Output shape: (4, 3)")
    print(f"  ONNX opset: 13")
    print(f"  Model size: {output_path.stat().st_size / 1024:.1f} KB")

    # Count ops
    ops = {}
    for node in onnx_model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"  Operators: {dict(ops)}")

    return output_path


def main() -> int:
    """Main entry point."""
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    models_dir = project_root / "models"

    # Create models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting classic models to: {models_dir}")
    print("-" * 60)

    # Check for torch and torchvision
    try:
        import torch
        import torchvision
        print(f"PyTorch version: {torch.__version__}")
        print(f"TorchVision version: {torchvision.__version__}")
    except ImportError as e:
        print(f"Error: Required package not found: {e}")
        print("\nInstall with: pip install torch torchvision onnx")
        return 1

    # Export models
    models_to_export = [
        export_lenet5,
        export_resnet18,
        export_simple_cnn,
        export_simple_mlp,
        export_simple_lstm,
        export_simple_transformer,
        export_operator_coverage_model,
        export_large_operator_coverage_model,
    ]

    exported = []
    for export_fn in models_to_export:
        try:
            path = export_fn(models_dir)
            exported.append(path)
            print()
        except Exception as e:
            print(f"Error exporting {export_fn.__name__}: {e}")
            print()
            continue

    # Summary
    print("-" * 60)
    print(f"Successfully exported {len(exported)} model(s):")
    for path in exported:
        print(f"  - {path.name}")

    # Create README for models directory
    readme_path = models_dir / "README.md"
    readme_content = """# Classic ML Models for Snapshot Testing

This directory contains ONNX exports of classic neural network models
used for snapshot testing the nnc-py compiler.

## Models

| Model | Input Shape | Use Case |
|-------|-------------|----------|
| `lenet5.onnx` | (1, 1, 28, 28) | Classic CNN, MNIST classification |
| `resnet18.onnx` | (1, 3, 224, 224) | Deep residual network, complex testing |
| `simple_cnn.onnx` | (1, 3, 32, 32) | Minimal CNN for quick tests |
| `simple_mlp.onnx` | (1, 1, 28, 28) | Simple feedforward network |
| `simple_lstm.onnx` | (1, 16, 32) | LSTM for sequence processing |
| `simple_transformer.onnx` | (1, 16, 32) | Self-attention mechanism, Transformer core |
| `operator_coverage.onnx` | (1, 4, 8) | Comprehensive operator coverage test |
| `operator_coverage_large.onnx` | (4, 64, 128) | Large model for memory constraint testing |

### Operator Coverage Model

The `operator_coverage.onnx` model includes the following operators:
- **Arithmetic**: Add, Sub, Mul, Div, Pow, Sqrt
- **Logical**: And, Or, Not, Equal
- **Shape**: Reshape, Transpose, Unsqueeze, Split, Concat, Tile
- **Reduction**: ReduceMean, ReduceSum
- **Activation**: Relu, Clip
- **Other**: Constant, MatMul, Cast

### Large Operator Coverage Model

The `operator_coverage_large.onnx` model is a memory-constrained variant with the same
operator coverage but larger tensor dimensions:
- Input: (4, 64, 128) vs (1, 4, 8) for the small model
- Forces memory spill when compiled with constraints (e.g., 32KB limit)
- Tests correctness of spill/reload code generation

## Usage

```python
from nnc_py import Compiler

compiler = Compiler(target="x86", opt_level=2)
compiler.compile("models/lenet5.onnx", "output/lenet5")
```

## Regenerating Models

Run the export script:
```bash
python tools/dump_classic_models.py
```

Requires: `pip install torch torchvision onnx`
"""
    readme_path.write_text(readme_content)
    print(f"\nCreated {readme_path.relative_to(project_root)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
