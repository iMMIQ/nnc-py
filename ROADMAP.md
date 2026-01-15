# NNC-Py Development Roadmap

> Neural Network Compiler for Python - ONNX to C with x86 and NPU support

---

## Phase 1: Foundation ✅
- [x] Project initialization
- [x] Dependency setup (ONNX, NumPy, NetworkX, Click, Rich, Pydantic)
- [x] IR data structures (Graph, Node, Tensor, Types)
- [x] Basic CLI framework
- [x] Configuration management
- [x] README documentation

**Goal**: ✅ Working project skeleton that can parse and validate basic structure

---

## Phase 2: ONNX Frontend ✅
- [x] ONNX model loader
- [x] Graph to IR conversion
- [x] Type inference
- [x] Shape inference (using ONNX shape_inference)
- [x] Constant loading
- [x] Attribute parsing

**Goal**: ✅ Load and convert ONNX models to internal IR

---

## Phase 3: Optimization Passes ⚠️

### Framework Status
- [x] Pass manager framework
- [x] Optimization level system (O0-O3)

### Pass Implementation Plan

#### Batch 1: Basic Passes (O1)
| # | Pass | File | Status | Description |
|---|------|------|--------|-------------|
| 1 | ConstantFoldingPass | `passes/constant_folding.py` | ✅ | Fold constant expressions (e.g., `Add(const1, const2)` → const) |
| 2 | DeadCodeEliminationPass | `passes/dce.py` | 🔜 | Remove unused nodes and tensors |
| 3 | IdentityEliminationPass | `passes/identity_elim.py` | 🔜 | Remove identity operations (`Identity(x)` → `x`) |

#### Batch 2: Structure Optimization (O2)
| # | Pass | File | Status | Description |
|---|------|------|--------|-------------|
| 4 | ReshapeEliminationPass | `passes/reshape_elim.py` | 🔜 | Eliminate redundant reshapes (`Reshape(Reshape(x))`) |
| 5 | TransposeEliminationPass | `passes/transpose_elim.py` | 🔜 | Merge/eliminate adjacent transposes |
| 6 | LayoutCanonicalizationPass | `passes/layout_canon.py` | 🔜 | Unify data layout (NCHW ↔ NHWC) |

#### Batch 3: Operator Fusion (O2/O3)
| # | Pass | File | Status | Description |
|---|------|------|--------|-------------|
| 7 | ConvBNFusionPass | `passes/conv_bn_fusion.py` | 🔜 | Fuse Conv + BatchNorm → Conv |
| 8 | ConvReluFusionPass | `passes/conv_relu_fusion.py` | 🔜 | Fuse Conv + ReLU → Conv(with activation) |
| 9 | ConvBnReluFusionPass | `passes/conv_bn_relu_fusion.py` | 🔜 | Fuse Conv + BN + ReLU → Conv |
| 10 | AffineFusionPass | `passes/affine_fusion.py` | 🔜 | Fuse `Mul(x, c) + Add(x, c)` → affine transform |

#### Batch 4: Memory Optimization (O2)
| # | Pass | File | Status | Description |
|---|------|------|--------|-------------|
| 11 | LivenessAnalysisPass | `passes/liveness.py` | ✅ | Analyze tensor lifetimes for memory reuse |
| 12 | MemoryPlanningPass | `passes/memory_plan.py` | ✅ | Static memory allocation with buffer sharing |
| 13 | InplaceOpPass | `passes/inplace.py` | 🔜 | Identify in-place operations (ReLU, etc.) |

### Optimization Level Configuration
```
O0: [] (no optimization)
O1: [ConstantFolding]
O2: [O1 + LivenessAnalysis, MemoryPlanning]
O3: [O2 + InplaceOp, (more passes planned)]
```

**Goal**: ⚠️ Optimize IR for better performance (ConstantFolding, LivenessAnalysis, MemoryPlanning implemented)

---

## Phase 4: Memory Planning ✅
- [x] Liveness analysis
- [x] Memory size calculation
- [x] Static buffer layout
- [x] Memory reuse analysis
- [ ] User-specified memory overrides

**Goal**: ✅ Efficient static memory allocation (implemented for O2+)

---

## Phase 5: Pseudo-Instruction Layer 🔜
- [x] Pseudo-instruction ISA definition (conceptual)
- [ ] IR lowering to pseudo-instructions
- [ ] Pattern matching for common ops
- [ ] Instruction scheduling

**Goal**: Abstract hardware acceleration layer

---

## Phase 6: Code Generation ✅
- [x] C code emitter
- [x] Runtime API definition (nnc_runtime.h, nnc_ops.h)
- [x] Data serialization (binary format)
- [x] Header file generation
- [x] x86 backend implementation
- [ ] NPU backend implementation (TODO)
- [x] Test runner generation

**Goal**: ✅ Generate compilable C code (x86 complete, NPU pending)

---

## Phase 7: Testing & Validation ✅
- [x] Basic unit tests
- [x] Integration tests with real models
- [x] Runtime operator tests (vs numpy)
- [ ] Reference implementation comparison (vs ONNX Runtime)
- [ ] Performance benchmarks
- [ ] Correctness validation suite
- [ ] CI/CD pipeline

**Goal**: ✅ Verified and tested codebase (unit tests, e2e tests, runtime tests)

---

## Phase 8: Advanced Features 🔜
- [ ] Quantization support (INT8/INT16)
- [ ] Dynamic shape support
- [ ] More operators (see list below)
- [ ] Multi-platform runtime stubs
- [ ] Debug visualization tools
- [ ] SIMD optimizations (AVX/SSE)

**Goal**: Production-ready compiler

---

## Supported Operators

### High Priority ✅ Implemented
- [x] Conv2D
- [x] MatMul / Gemm
- [x] MaxPool2D
- [x] AvgPool2D
- [x] Relu
- [x] Sigmoid
- [x] Tanh
- [x] Softmax
- [x] Add / Sub / Mul / Div
- [x] Transpose
- [x] Reshape
- [x] Flatten
- [x] Concat
- [x] Split
- [x] ReduceMean
- [x] ReduceSum
- [x] LayerNormalization
- [x] Identity
- [x] Constant
- [x] Clip

### Medium Priority 🔜 To Implement
- [ ] LeakyReLU
- [ ] BatchNormalization
- [ ] Gather
- [ ] Pad
- [ ] Slice
- [ ] Tile
- [ ] Squeeze / Unsqueeze

### Lower Priority 📋 Future
- [ ] Cast
- [ ] Pow
- [ ] Expand
- [ ] Where
- [ ] TopK
- [ ] Attention (MHA/MQGA)

---

## Milestones

### v0.1.0 - MVP ✅
- [x] Load simple ONNX models
- [x] Generate C code with basic operators
- [x] Static memory allocation
- [x] CLI with compile command
- [x] x86 backend
- [x] End-to-end testing

### v0.2.0 - Optimization ✅
- [x] Graph optimization framework
- [x] Constant folding pass
- [ ] Operator fusion
- [x] Memory planning (liveness + buffer sharing)
- [ ] Transpose elimination

### v0.3.0 - Production Ready 📋
- [ ] Full operator set (~30 ops)
- [ ] Comprehensive testing
- [x] Documentation
- [ ] Examples and tutorials

### v0.4.0 - NPU Support 📋
- [ ] NPU backend implementation
- [ ] Pseudo-instruction layer
- [ ] Hardware-specific optimizations

### v0.5.0 - Advanced 📋
- [ ] Quantization (PTQ/QAT)
- [ ] Dynamic shapes
- [ ] Performance profiling
- [ ] Visualization tools

---

## Quick Reference

| Component | File | Status |
|-----------|------|--------|
| CLI | `cli.py` | ✅ |
| Compiler | `compiler.py` | ✅ |
| ONNX Loader | `frontend/onnx_loader.py` | ✅ |
| IR | `ir/*.py` | ✅ |
| x86 Backend | `codegen/x86_backend.py` | ✅ |
| NPU Backend | `codegen/npu_backend.py` | 🔜 TODO |
| C Emitter | `codegen/c_emitter.py` | ✅ |
| Pass Framework | `passes/base.py` | ✅ |
| ConstantFolding | `passes/constant_folding.py` | ✅ |
| LivenessAnalysis | `passes/liveness.py` | ✅ |
| MemoryPlanning | `passes/memory_plan.py` | ✅ |
| Runtime | `runtime/` | ✅ |
| Tests | `tests/` | ✅ |
| Documentation | `README.md`, `ROADMAP.md`, `docs/MEMORY_ALLOCATION.md` | ✅ |
