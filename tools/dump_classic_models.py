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
    3. Classification head

    This is NOT a full BERT/GPT-style transformer, but focuses on
    the core attention mechanism that defines the architecture.

    Note: LayerNorm is omitted to avoid ONNX operators (Slice, Gather, Shape)
    that may not be supported by the compiler.
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
        export_simple_transformer,
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
| `simple_transformer.onnx` | (1, 16, 32) | Self-attention mechanism, Transformer core |

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
