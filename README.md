# NNC-Py

> Neural Network Compiler - ONNX to C compiler for edge inference

NNC-Py is a compiler that converts ONNX neural network models to C code for embedded and edge devices. It targets x86 and NPU architectures with support for operator fusion, memory optimization, and pseudo-instruction acceleration.

## Features

- **ONNX Frontend**: Load and parse ONNX models with automatic shape inference
- **Multi-Target Backend**: Generate code for x86 simulation or NPU deployment
- **Optimization Passes**: Constant folding, dead code elimination, and more (O0-O3)
- **Runtime Library**: Optimized C runtime for common neural network operators
- **CLI Interface**: Simple command-line interface for compiling models

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/nnc-py.git
cd nnc-py
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Compile an ONNX model
nnc compile model.onnx -o ./build -t x86 -O1

# Compile with O3 optimization
nnc compile model.onnx -o ./build -O3
```

This generates:
- `model.h` - Header file with tensor declarations
- `model.c` - Main inference code
- `tensors.c` - Tensor definitions
- `constants.c` - Constant data arrays
- `Makefile` - Build configuration
- `test_runner.c` - Test executable

Build and run:
```bash
cd build
make
./model
```

## Usage

### Python API

```python
from nnc_py import Compiler

# Compile with default settings (O0, x86)
compiler = Compiler(target="x86", opt_level=0)
compiler.compile("model.onnx", "./build")

# Compile with optimization
compiler = Compiler(target="x86", opt_level=3)
artifacts = compiler.compile("model.onnx", "./build")
```

### Command Line

```bash
nnc compile MODEL.onnx -o OUTPUT_DIR [OPTIONS]

Options:
  -o, --output PATH          Output directory
  -t, --target [x86|npu]     Target architecture (default: x86)
  -O, --opt-level INT        Optimization level 0-3 (default: 0)
  --entry-name TEXT          Entry point function name
  -v, --verbose              Verbose output
```

## Supported Operators

| Operator | Status | Notes |
|----------|--------|-------|
| Conv2D | ✅ | 2D convolution |
| MaxPool2D | ✅ | 2D max pooling |
| AvgPool2D | ✅ | 2D average pooling |
| MatMul/Gemm | ✅ | Matrix multiplication (2D, batched, vector) |
| Add | ✅ | Element-wise addition |
| Sub | ✅ | Element-wise subtraction |
| Mul | ✅ | Element-wise multiplication |
| Div | ✅ | Element-wise division |
| Relu | ✅ | ReLU activation |
| Sigmoid | ✅ | Sigmoid activation |
| Tanh | ✅ | Tanh activation |
| Softmax | ✅ | Softmax along axis |
| Transpose | ✅ | Tensor transpose with permutation |
| Reshape | ✅ | Reshape to specified shape |
| Flatten | ✅ | Flatten to 1D |
| Concat | ✅ | Concatenate along axis |
| Split | ✅ | Split along axis |
| LayerNormalization | ✅ | Layer normalization |
| Identity | ✅ | Identity operation |
| Constant | ✅ | Constant tensor |
| Clip | ✅ | Clip values to range |
| ReduceMean | ✅ | Mean reduction |
| ReduceSum | ✅ | Sum reduction |

## Optimization Levels

- **O0**: No optimization - direct translation
- **O1**: Basic optimizations - constant folding, dead code elimination
- **O2**: Structure optimization - reshape/transpose elimination, layout canonicalization
- **O3**: Advanced optimizations - operator fusion, memory planning

## Project Structure

```
nnc-py/
├── src/nnc_py/
│   ├── cli.py              # Command-line interface
│   ├── compiler.py         # Main compiler class
│   ├── config.py           # Configuration management
│   ├── codegen/            # Code generation
│   │   ├── base.py         # Backend base class
│   │   ├── c_emitter.py    # C code emitter
│   │   └── x86_backend.py  # x86 backend
│   ├── frontend/           # Frontend loaders
│   │   └── onnx_loader.py  # ONNX model loader
│   ├── ir/                 # Intermediate representation
│   │   ├── graph.py        # Computation graph
│   │   ├── node.py         # Operation nodes
│   │   ├── tensor.py       # Tensor types
│   │   └── types.py        # Data types
│   └── passes/             # Optimization passes
│       ├── base.py         # Pass manager
│       └── constant_folding.py
├── runtime/                # C runtime library
│   ├── include/            # Runtime headers
│   └── x86/                # x86 implementation
└── tests/                  # Test suite
```

## License

MIT License - see LICENSE file for details.
