# Scheduled Residual Tile Streaming Implementation Plan

**Goal:** Extend scheduled-native O3 tile streaming to residual `conv + add + relu` groups under a 1MB `fast_mem` budget.

## Task 1: Add failing scheduled tile-streaming tests

- [ ] Add codegen tests for scheduled `conv + add + relu` and `conv + fused_add_relu`
- [ ] Add a runtime correctness test that compiles generated C and compares outputs with ONNX Runtime
- [ ] Add a residual/downsample regression test if support is enabled in this round

## Task 2: Implement scheduled-native residual group streaming

- [ ] Extend scheduled group-plan acceptance beyond conv-only groups
- [ ] Add safe multi-buffer helper generation for residual/add groups
- [ ] Replace grouped compute steps with a single helper invocation
- [ ] Keep detached home storage only for tensors not fully covered by streaming

## Task 3: Re-verify 1MB resnet18 behavior

- [ ] Run targeted pytest coverage for scheduled tile codegen/runtime
- [ ] Compile `resnet18` with `-O3 --max-memory 1M`
- [ ] Compare runtime output against ONNX Runtime
- [ ] Inspect generated C to confirm which steps are streaming vs home execution
