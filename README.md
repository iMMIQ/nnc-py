# NNC-Py

> Neural Network Compiler - ONNX to C compiler for edge inference

NNC-Py is a compiler that converts ONNX neural network models to C code for embedded and edge devices. The x86 backend is implemented today; the NPU target is reserved but not implemented yet. The compiler includes operator fusion, memory optimization, and a C runtime for generated models.

## Features

- **ONNX Frontend**: Load and parse ONNX models with automatic shape inference
- **Backend**: Generate and run x86 C code today, with an NPU target planned
- **Optimization Passes**: Identity elimination, dead code elimination, fusion, memory planning, and spill analysis (O0-O3)
- **Runtime Library**: Optimized C runtime for common neural network operators
- **CLI Interface**: Simple command-line interface for compiling models

## Installation

```bash
# Install from source
git clone <repository-url>
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
- `constants.bin` - Serialized constant data
- `constants_loader.c` - Constant loader implementation
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

# Compile with optimization and a public entry-point alias
compiler = Compiler(target="x86", opt_level=3)
compiler.compile("model.onnx", "./build", entry_point="my_infer")
```

Note: `entry_point=` in the Python API corresponds to `--entry-name` in the `nnc compile` CLI.

### Command Line

```bash
nnc compile MODEL.onnx -o OUTPUT_DIR [OPTIONS]

Options:
  -o, --output PATH          Output directory
  -t, --target [x86|npu]     Target architecture (default: x86)
  -O, --opt-level INT        Optimization level 0-3 (default: 0)
  --entry-name TEXT          Public inference entry point (default: nnc_run)
  -v, --verbose              Verbose output
```

Notes:
- `x86` is the only implemented backend at the moment.
- `npu` is accepted as a reserved target name, but compilation fails with a clear "not implemented yet" error.
- `--entry-name my_infer` adds a public wrapper like `void my_infer(void)` that calls `nnc_run()`.

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

- **O0**: Required analysis only - liveness + memory planning
- **O1**: Basic graph cleanup - identity elimination + dead code elimination
- **O2**: O1 + spill analysis for memory-constrained builds
- **O3**: O2 + pattern fusion + dominator fusion

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

## Snapshot Testing

The project includes comprehensive snapshot testing that verifies the entire compilation pipeline:

1. **IR Graph Snapshots** - Verifies ONNX models are parsed correctly into IR graphs
2. **Code Generation Snapshots** - Ensures generated C code remains consistent across runs
3. **Runtime Execution Testing** - Compiles and runs generated code with sanitizers when the environment supports it
4. **Result Verification** - Compares C program output with PyTorch reference

### Running Snapshot Tests

```bash
# Run one snapshot suite
pytest tests/test_snapshots_simple_conv.py -v

# Update snapshots
pytest tests/test_snapshots_simple_conv.py --snapshot-update

# Run runtime tests with output comparison
pytest tests/test_snapshots_simple_conv.py::TestCodegenSnapshots::test_simple_conv_codegen_with_runtime -v -s
```

### Snapshot Test Workflow

For each ONNX model, the snapshot tests:

1. **Generate C Code** - Compiles ONNX model to C using the Compiler
2. **Compile with Sanitizers** - Uses `-g -fsanitize=address` to catch memory issues
3. **Execute Program** - Runs the compiled executable and captures output
4. **Compare with Reference** - Computes expected output using PyTorch and compares

The runtime layer is environment-sensitive because LeakSanitizer can fail under restricted or ptrace-controlled environments. Treat those failures separately from compiler regressions.

## Benchmarks (O3 Generated C)

This repo includes a host-side benchmark harness under `benchmarks/` intended to measure:
- End-to-end performance of generated **x86 C** for a fixed ONNX model at `opt_level=3` (O3)
- Static memory footprint and artifact sizes from the generated build outputs

Out of scope: compile time, embedded/cross targets, and system-level profiling (perf/valgrind/RSS/cache-miss).

### Workload Batch Semantics

For models with fixed input shapes (like `models/resnet18.onnx`), `--batch-sizes` are **workload batches**, not a runtime change to the model input tensor shape.

One measured iteration for `batch_size=N` runs `nnc_run()` **N times sequentially** in-process, and the harness reports:
- `batch_size=1`: single-inference latency distribution (mean/p50/p95) and throughput
- `batch_size>1`: throughput and per-iteration latency when executing `N` sequential inferences per measured iteration

### Running A Benchmark

Run the default `resnet18` case (defaults to the case’s configured batch sizes):
```bash
python -m benchmarks.harness --model resnet18
```

Override workload batches:
```bash
python -m benchmarks.harness --model resnet18 --batch-sizes 1 8 16 32
# or:
python -m benchmarks.harness --model resnet18 --batch-sizes 1,8,16,32
```

Write the result JSON to a specific location:
```bash
python -m benchmarks.harness --model resnet18 --batch-sizes 1 --output benchmarks/results/resnet18-current.json
```

### Result Locations

By default, the harness writes a timestamped result file under:
- `benchmarks/results/<model>-<UTC timestamp>.json`

Build artifacts (generated C, runner, executable, constants) are placed under:
- `benchmarks/build/<model>-<UTC timestamp>/`

If `--output PATH.json` is provided, the build directory is created next to it as:
- `<PATH.stem>_build/` (for example `benchmarks/results/resnet18-current_build/`)

The result JSON includes the build directory and executable path under the `artifacts` field.

### Baseline Comparison Workflow

To compare a new run against a saved baseline, pass `--baseline-result` and the harness will write a sibling diff JSON next to the result:
```bash
python -m benchmarks.harness \
  --model resnet18 \
  --batch-sizes 1 8 16 32 \
  --baseline-result benchmarks/results/resnet18-baseline.json \
  --output benchmarks/results/resnet18-candidate.json
```

This writes:
- `benchmarks/results/resnet18-candidate.json`
- `benchmarks/results/resnet18-candidate.diff.json` (batch-level throughput/latency deltas and memory delta)

## License

MIT License - see LICENSE file for details.
