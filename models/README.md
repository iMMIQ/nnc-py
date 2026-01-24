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
