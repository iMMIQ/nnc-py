# Fix C Code Output Discrepancy Debug Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Debug and fix why the generated C code outputs 0.5 instead of 0.333 (matching ONNX runtime)

**Architecture:** The ONNX model contains boolean operations (Greater, And, Or, Not) that produce a final mean of 0.333 in ONNX runtime but 0.5 in C code. Boolean operations have been fixed, but there's still a discrepancy.

**Tech Stack:** ONNX, onnxruntime, C (generated), numpy, Python

---

## Current State

- ONNX runtime output: **0.333**
- C code output: **0.5**
- Fixed so far: `nnc_and` BOOL handling, type inference for Greater/Or/Not/Cast

## Task 1: Verify Model Consistency

**Files:**
- Check: `tests/models/operator_coverage.onnx`
- Check: `/tmp/op_test_debug/constants.bin`

**Step 1: Verify ONNX model weight**

Run: `python -c "import onnx; from onnx import numpy_helper; m = onnx.load('tests/models/operator_coverage.onnx'); print([i.name for i in m.graph.initializer])"`
Expected: Output showing `weight`, `scale`, `const_zero`, and Constant nodes

**Step 2: Verify constants.bin has same weight**

Run: `python -c "import struct; data = open('/tmp/op_test_debug/constants.bin', 'rb').read(); print('weight found:', 'tensor_weight' in data.decode('utf-8', errors='ignore'))"`
Expected: `weight found: True`

**Step 3: Run ONNX runtime to get baseline**

Run: `python -c "import onnxruntime as ort; import numpy as np; s = ort.InferenceSession('tests/models/operator_coverage.onnx'); x = np.arange(32, dtype=np.float32).reshape(1, 4, 8) / 100; print(s.run(None, {'input': x})[0][0, 0])"`
Expected: `0.3333333432674408`

**Step 4: Run C code to compare**

Run: `cd /tmp/op_test_debug && ./model`
Expected: Should output 0.333 (currently outputs 0.5 - this is the bug)

---

## Task 2: Add Debug Output to C Code

**Files:**
- Create: `/tmp/op_test_debug/debug_main.c`

**Step 1: Create debug main**

```c
#include <stdio.h>
#include "model.h"

// Declare external tensors we want to inspect
extern Tensor tensor_Tile_output_0;
extern Tensor tensor_Transpose_1_output_0;
extern Tensor tensor_Split_output_0;
extern Tensor tensor_Greater_output_0;
extern Tensor tensor_Greater_1_output_0;
extern Tensor tensor_And_output_0;
extern Tensor tensor_Or_output_0;
extern Tensor tensor_Not_output_0;
extern Tensor tensor_Concat_2_output_0;

int main() {
    nnc_load_constants("constants.bin");

    // Set input
    for (int i = 0; i < 32; i++) {
        ((float*)tensor_input.data)[i] = (float)i / 100.0f;
    }

    // Run inference
    nnc_run();

    // Print intermediate values
    printf("=== Debug Output ===\n");

    // Tile output
    float* tile_data = (float*)tensor_Tile_output_0.data;
    printf("Tile_output_0 (first 4): %.4f, %.4f, %.4f, %.4f\n",
           tile_data[0], tile_data[1], tile_data[2], tile_data[3]);

    // Transpose output
    float* trans_data = (float*)tensor_Transpose_1_output_0.data;
    printf("Transpose_1_output_0 (first 4): %.4f, %.4f, %.4f, %.4f\n",
           trans_data[0], trans_data[1], trans_data[2], trans_data[3]);

    // Split output
    float* split_data = (float*)tensor_Split_output_0.data;
    printf("Split_output_0 (first 4): %.4f, %.4f, %.4f, %.4f\n",
           split_data[0], split_data[1], split_data[2], split_data[3]);

    // Boolean ops
    uint8_t* and_data = (uint8_t*)tensor_And_output_0.data;
    uint8_t* or_data = (uint8_t*)tensor_Or_output_0.data;
    uint8_t* not_data = (uint8_t*)tensor_Not_output_0.data;

    printf("And (first 4): %d, %d, %d, %d\n", and_data[0], and_data[1], and_data[2], and_data[3]);
    printf("Or (first 4): %d, %d, %d, %d\n", or_data[0], or_data[1], or_data[2], or_data[3]);
    printf("Not (first 4): %d, %d, %d, %d\n", not_data[0], not_data[1], not_data[2], not_data[3]);

    // Concat_2
    float* concat_data = (float*)tensor_Concat_2_output_0.data;
    printf("Concat_2_output_0 (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.1f ", concat_data[i]);
    printf("\n");

    // Final output
    printf("Final output: %.4f\n", ((float*)tensor_output.data)[0]);

    return 0;
}
```

**Step 2: Compile and run debug main**

Run:
```bash
cd /tmp/op_test_debug
gcc -std=c11 -O2 -I/home/ayd/code/nnc-py/runtime/include debug_main.c model.o tensors.o constants_loader.o /home/ayd/code/nnc-py/runtime/x86/ops.c -lm -o debug_main
./debug_main
```

Expected: Output showing all intermediate values

---

## Task 3: Get Corresponding ONNX Intermediate Values

**Files:**
- Create: `/tmp/compare_onnx.py`

**Step 1: Create Python script to extract ONNX intermediate values**

```python
import onnx
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('tests/models/operator_coverage.onnx')
test_input = np.arange(32, dtype=np.float32).reshape(1, 4, 8) / 100.0

# Get intermediate outputs by adding them to graph
model = onnx.load('tests/models/operator_coverage.onnx')

# Find key nodes and their output shapes
print("=== ONNX Graph Analysis ===")
for node in model.graph.node:
    if node.op_type in ['Tile', 'Transpose', 'Split', 'Greater', 'And', 'Or', 'Not']:
        print(f"{node.op_type}: {node.input} -> {node.output}")

# Use Python to trace through equivalent computation
print("\n=== Python Trace ===")
import torch
import sys
sys.path.insert(0, 'tools')
from dump_classic_models import OperatorCoverageModel
from onnx import numpy_helper

# Get weight from ONNX
for init in model.graph.initializer:
    if 'weight' in init.name:
        weight = numpy_helper.to_array(init)
        break

pytorch_model = OperatorCoverageModel()
pytorch_model.eval()
with torch.no_grad():
    pytorch_model.weight[:] = torch.from_numpy(weight)
    result = pytorch_model(torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) / 100)
    print(f"PyTorch output: {result.item()}")
```

**Step 2: Run comparison script**

Run: `python /tmp/compare_onnx.py`

Expected: PyTorch output showing 0.333 and graph structure

---

## Task 4: Compare Boolean Operation Inputs

**Files:**
- Modify: `runtime/x86/ops.c` (if bug found)

**Step 1: Check Greater operation in C code**

The C code debug output should show:
- Greater_output_0: all 0 (x > 0.0, all x values are negative)
- Greater_1_output_0: all 0 (x > 1.0, all x values are negative)

If Greater shows non-zero values, the input to Greater is wrong.

**Step 2: Trace back to find where values diverge**

If Split values don't match between C and ONNX, work backwards:
1. Check Transpose_1 output
2. Check Tile output
3. Check Expand output
4. Check Unsqueeze_2 output
5. Check Concat_1 output (ReduceMean + ReduceSum)

**Step 3: Verify each operation**

For each operation, compare:
- Input shape (C vs ONNX)
- Output shape (C vs ONNX)
- Sample values (first 4 elements)

---

## Task 5: Identify the Divergent Operation

**Files:**
- Check: `src/nnc_py/codegen/x86_backend.py`
- Check: `runtime/x86/ops.c`

**Step 1: Compare shapes through the pipeline**

Key shapes after each operation:
- Reshape_output_0: (1, 4, 4)
- ReduceMean_output_0: (1, 1) with keepdims=1 → (1, 1, 4)
- ReduceSum_output_0: (1, 1) with keepdims=1, axis=-1 → (1, 1, 4)
- Concat_1_output_0: (1, 2, 4)
- Unsqueeze_2_output_0: (1, 2, 1, 4) or (1, 2, 4, 1)
- Expand_output_0: broadcasts to...
- Tile_output_0: repeats along...
- Transpose_1_output_0: (1, 4, 2) or (1, 2, 4)
- Split_output_0: (1, 4, 1)

**Step 2: Check axis parameters**

Common bugs:
- Wrong axis in ReduceMean/ReduceSum (should be 1, keepdims=1)
- Wrong axis in Concat (should be 1)
- Wrong axis in Split (should be 2)
- Wrong perm in Transpose_1 (should be [0, 2, 1])

**Step 3: Verify axis values in generated code**

Run: `grep -n "nnc_reducemean\|nnc_split\|nnc_concat\|nnc_transpose" /tmp/op_test_debug/model.c | head -20`

Expected output:
```c
nnc_reducemean(&tensor_Reshape_output_0, &tensor_ReduceMean_output_0, 1, 1);  // axis=1, keepdims=1
nnc_reducesum(&tensor_Reshape_output_0, &tensor_ReduceSum_output_0, -1, 1);  // axis=-1, keepdims=1
nnc_concat(..., 2, 1);  // num_inputs=2, axis=1
nnc_transpose(..., (int64_t*)node_Transpose_1_perm, 3);  // perm=[0,2,1], ndim=3
nnc_split(..., 2, 2);  // axis=2
```

---

## Task 6: Fix the Identified Bug

**Files:**
- Modify: `runtime/x86/ops.c` or `src/nnc_py/codegen/x86_backend.py` or `src/nnc_py/codegen/c_emitter.py`

**Step 1: Implement fix based on findings**

Possible fixes:
- If axis is wrong: fix the axis parameter passing
- If shape is wrong: fix the shape inference
- If operation implementation is wrong: fix the C function

**Step 2: Add test case**

Create test in `tests/test_runtime_ops.py`:

```python
def test_boolean_concat_mean():
    """Test that boolean ops cast to float and concatenate correctly."""
    # This tests the specific pattern: Greater -> And/Or/Not -> Cast -> Concat -> Mean
    pass
```

**Step 3: Verify fix**

Run:
```bash
cd /tmp/op_test_debug
make clean
make
./model
```

Expected: Output shows `0.500000` changed to `0.333333`

---

## Task 7: Run Full Test Suite

**Files:**
- Test: `tests/test_snapshots_operator_coverage.py`

**Step 1: Run snapshot test**

Run: `python -m pytest tests/test_snapshots_operator_coverage.py -xvs`

Expected: All tests pass

**Step 2: Regenerate snapshot if needed**

If test fails due to output change:
Run: `python -m pytest tests/test_snapshots_operator_coverage.py --snapshot-update -xvs`

**Step 3: Commit**

```bash
git add -A
git commit -m "fix: correct [operation] axis/shape handling"
```
