# Classic ML Models for Snapshot Testing

This directory contains ONNX exports of classic neural network models
used for snapshot testing the nnc-py compiler.

## Models

| Model | Input Shape | Use Case |
|-------|-------------|----------|
| `lenet5.onnx` | (1, 1, 28, 28) | Classic CNN, MNIST classification |
| `resnet18.onnx` | (1, 3, 224, 224) | Deep residual network, complex testing |
| `simple_cnn.onnx` | (1, 3, 32, 32) | Minimal CNN for quick tests |
| `simple_mlp.onnx` | (1, 1, 28, 28) | Simple feedforward network |

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
