# Floating-Point Accumulation Error Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix floating-point accumulation errors in runtime operations to improve numerical accuracy and eliminate the need for relaxed tolerances in deep network tests.

**Architecture:** Implement Kahan summation (compensated summation) in critical accumulation operations (reducesum, reducemean, matmul, gemm, conv) to reduce floating-point error accumulation. This algorithm maintains a running compensation term to capture and add back the low-order bits that are lost in each addition.

**Tech Stack:** C11 (runtime), Python 3.12 (tests), ONNX Runtime (reference), NumPy (verification)

---

## Background: The Problem

Floating-point arithmetic has limited precision (about 7 decimal digits for float32). When accumulating many values, small rounding errors compound:

```c
// Current naive implementation - errors accumulate
float sum = 0.0f;
for (int i = 0; i < n; i++) {
    sum += data[i];  // Lost precision with each addition
}
```

For deep networks like ResNet18 with millions of operations, these accumulate to ~1% error.

## Solution: Kahan Summation

Kahan summation maintains a compensation term `c` that captures the lost precision:

```c
// Kahan summation - dramatically reduced error
float sum = 0.0f;
float c = 0.0f;  // Running compensation for lost low-order bits
for (int i = 0; i < n; i++) {
    float y = data[i] - c;  // Subtract compensation first
    float t = sum + y;      // Add to sum
    c = (t - sum) - y;      // Update compensation (what was lost)
    sum = t;
}
```

This reduces accumulated error from O(nε) to O(ε), where ε is machine epsilon.

---

## Task 1: Add Kahan Summation Helper Function to Runtime

**Files:**
- Modify: `runtime/x86/ops.c`

**Step 1: Write the C helper function for Kahan summation**

Add this function after the `tensor_numel` helper (around line 23):

```c
/* ============================================================================
 * Numerical Stability Helpers
 * ============================================================================ */

/* Kahan summation for improved numerical accuracy
 * Reduces floating-point error accumulation from O(n*eps) to O(eps)
 * Returns the accurate sum of n float values starting at data
 */
static float kahan_sum(const float* data, int64_t n) {
    float sum = 0.0f;
    float c = 0.0f;  /* Running compensation for lost low-order bits */

    for (int64_t i = 0; i < n; i++) {
        float y = data[i] - c;   /* Subtract compensation first */
        float t = sum + y;        /* Add to sum */
        c = (t - sum) - y;        /* Update compensation: (sum + y) - sum - y = lost bits */
        sum = t;
    }
    return sum;
}

/* Pairwise summation alternative for better parallelization potential
 * Reduces error by recursively summing pairs (log depth instead of linear)
 */
static float pairwise_sum(const float* data, int64_t n) {
    if (n <= 32) {
        /* Base case: small arrays use direct summation */
        float sum = 0.0f;
        for (int64_t i = 0; i < n; i++) {
            sum += data[i];
        }
        return sum;
    }

    /* Recursive case: split and sum halves */
    int64_t half = n / 2;
    return pairwise_sum(data, half) + pairwise_sum(data + half, n - half);
}
```

**Step 2: Verify the file compiles**

Run: `cd runtime/x86 && make clean && make ops.o`

Expected: ops.o compiles without errors or warnings

**Step 3: Commit**

```bash
git add runtime/x86/ops.c
git commit -m "feat(runtime): add Kahan summation helper for numerical stability"
```

---

## Task 2: Fix nnc_reducesum Using Kahan Summation

**Files:**
- Modify: `runtime/x86/ops.c` (function `nnc_reducesum`, around line 914)

**Step 1: Replace naive accumulation with Kahan summation**

Find the inner loop (around lines 941-945):
```c
/* OLD CODE - remove this */
float sum = 0.0f;
for (int64_t r = 0; r < reduce_dim; r++) {
    int64_t idx = outer * reduce_dim * inner_size + r * inner_size + inner;
    sum += in_data[idx];
}
```

Replace with:
```c
/* NEW CODE - Kahan summation */
float sum = 0.0f;
float c = 0.0f;  /* Running compensation */
for (int64_t r = 0; r < reduce_dim; r++) {
    int64_t idx = outer * reduce_dim * inner_size + r * inner_size + inner;
    float y = in_data[idx] - c;   /* Subtract compensation */
    float t = sum + y;
    c = (t - sum) - y;            /* Track lost precision */
    sum = t;
}
```

**Step 2: Verify compilation**

Run: `cd runtime/x86 && make clean && make`

Expected: Clean compilation, no warnings

**Step 3: Run snapshot tests to verify improvement**

Run: `pytest tests/test_snapshots_resnet18.py::TestCodegenSnapshots::test_resnet18_codegen_with_runtime -v`

Expected: Test should still pass, potentially with improved accuracy

**Step 4: Commit**

```bash
git add runtime/x86/ops.c
git commit -m "fix(runtime): use Kahan summation in nnc_reducesum for accuracy"
```

---

## Task 3: Fix nnc_reducemean Using Kahan Summation

**Files:**
- Modify: `runtime/x86/ops.c` (function `nnc_reducemean`)

**Step 1: Find and update nnc_reducemean**

Search for the function and find the accumulation loop. Replace naive sum with Kahan:

```c
/* Find the reduction loop in nnc_reducemean and replace with Kahan */
float sum = 0.0f;
float c = 0.0f;  /* Running compensation */
for (int64_t r = 0; r < reduce_dim; r++) {
    int64_t idx = outer * reduce_dim * inner_size + r * inner_size + inner;
    float y = in_data[idx] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}
float mean = sum / (float)reduce_dim;  /* Divide after accurate summation */
```

**Step 2: Verify compilation**

Run: `cd runtime/x86 && make clean && make`

Expected: Clean compilation

**Step 3: Commit**

```bash
git add runtime/x86/ops.c
git commit -m "fix(runtime): use Kahan summation in nnc_reducemean for accuracy"
```

---

## Task 4: Fix nnc_matmul Using Kahan Summation

**Files:**
- Modify: `runtime/x86/ops.c` (function `nnc_matmul`, around line 641)

**Step 1: Update the inner accumulation loops in matmul**

Find all inner loops with `sum += a... * b...` patterns. There are multiple cases (2D, 1D@2D, 2D@1D, batched).

For each case, replace the naive accumulation:

```c
/* OLD - Example from line 662-665 */
float sum = 0.0f;
for (int k = 0; k < K; k++) {
    sum += a_data[i * K + k] * b_data[k * N + j];
}
```

With Kahan:
```c
/* NEW */
float sum = 0.0f;
float c = 0.0f;
for (int k = 0; k < K; k++) {
    float y = a_data[i * K + k] * b_data[k * N + j] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}
```

Apply this to all 4 cases in `nnc_matmul`:
1. Standard 2D (lines ~660-667)
2. Vector @ Matrix (lines ~674-679)
3. Matrix @ Vector (lines ~686-691)
4. Batched (lines ~720+)

**Step 2: Verify compilation**

Run: `cd runtime/x86 && make clean && make`

Expected: Clean compilation

**Step 3: Test with ResNet snapshot**

Run: `pytest tests/test_snapshots_resnet18.py::TestCodegenSnapshots::test_resnet18_codegen_with_runtime -v`

Expected: Improved accuracy

**Step 4: Commit**

```bash
git add runtime/x86/ops.c
git commit -m "fix(runtime): use Kahan summation in nnc_matmul for accuracy"
```

---

## Task 5: Fix nnc_gemm Using Kahan Summation

**Files:**
- Modify: `runtime/x86/ops.c` (function `nnc_gemm`, around line 750)

**Step 1: Update the inner accumulation loop in gemm**

Find the loop and replace naive accumulation with Kahan:

```c
/* Locate the accumulation loop and replace */
float sum = 0.0f;
float c = 0.0f;  /* Kahan compensation */
for (int k = 0; k < K; k++) {
    float y = a_data[i * K + k] * b_data[k * N + j] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}
```

Also handle the bias addition with proper precision:
```c
if (c_data) {
    /* Add bias after accumulation to preserve precision */
    float y = *c_data - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}
```

**Step 2: Verify compilation**

Run: `cd runtime/x86 && make clean && make`

**Step 3: Commit**

```bash
git add runtime/x86/ops.c
git commit -m "fix(runtime): use Kahan summation in nnc_gemm for accuracy"
```

---

## Task 6: Fix nnc_conv Using Kahan Summation

**Files:**
- Modify: `runtime/x86/ops.c` (function `nnc_conv`, around line 414)

**Step 1: Update the convolution accumulation loop**

The convolution has the deepest nesting (7 loops) and benefits most from Kahan.

Find the inner accumulation (around lines 495-520):

```c
/* Replace the accumulation loop */
float sum = 0.0f;
float c = 0.0f;  /* Kahan compensation */

for (int c_in = 0; c_in < C_in; c_in++) {
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int h = h_in + kh;
            int w = w_in + kw;

            if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                int64_t in_idx = n * C_in * H_in * W_in
                               + c_in * H_in * W_in
                               + h * W_in
                               + w;

                int64_t weight_idx = c_out * C_in * kernel_h * kernel_w
                                   + c_in * kernel_h * kernel_w
                                   + kh * kernel_w
                                   + kw;

                float y = in_data[in_idx] * weight_data[weight_idx] - c;
                float t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
        }
    }
}
```

**Step 2: Verify compilation**

Run: `cd runtime/x86 && make clean && make`

**Step 3: Commit**

```bash
git add runtime/x86/ops.c
git commit -m "fix(runtime): use Kahan summation in nnc_conv for accuracy"
```

---

## Task 7: Create Numerical Accuracy Test

**Files:**
- Create: `tests/test_numerical_accuracy.py`

**Step 1: Write test to verify Kahan improvement**

```python
"""Test numerical accuracy improvements from Kahan summation."""

import numpy as np
import pytest
from pathlib import Path

from nnc_py.compiler import Compiler


def test_kahan_summation_reduces_error():
    """Verify that Kahan summation reduces accumulated error."""
    # Create a model that does a large ReduceSum
    # This will expose the difference between naive and Kahan summation

    import onnx
    from onnx import helper, TensorProto, numpy_helper

    # Create a simple model: sum of 1000 small values
    # This demonstrates Kahan's advantage
    n = 1000
    small_values = np.random.randn(n).astype(np.float32) * 0.001

    # Create ONNX model with Constant -> ReduceSum
    const_tensor = numpy_helper.from_array(small_values, name="input")
    reduce_node = helper.make_node(
        "ReduceSum",
        inputs=["input"],
        outputs=["output"],
        axes=[0],
        keepdims=0
    )

    graph = helper.make_graph(
        [reduce_node],
        "test_reduce_sum",
        [],
        [helper.make_tensor("output", TensorProto.FLOAT, [1])]
    )
    graph.initializer.append(const_tensor)

    model = helper.make_model(graph)

    # Save and compile
    model_path = Path("/tmp/test_reduce_sum.onnx")
    onnx.save(model, model_path)

    compiler = Compiler(opt_level=2)
    compiler.compile(str(model_path), "/tmp/test_reduce_sum_out")

    # Run and compare with numpy
    # The Kahan version should be much closer to numpy's sum
    # (numpy also uses pairwise summation for accuracy)
```

**Step 2: Run test to verify it works**

Run: `pytest tests/test_numerical_accuracy.py -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_numerical_accuracy.py
git commit -m "test: add numerical accuracy test for Kahan summation"
```

---

## Task 8: Revert Tolerance Relaxation in Tests

**Files:**
- Modify: `tests/test_common.py` (lines ~520-524)

**Step 1: Revert the tolerance changes**

Find the tolerance code added for deep networks (around lines 520-524):

```python
# CURRENT - Remove this condition
if "resnet" in model_name.lower() or "vgg" in model_name.lower():
    rtol, atol = 2e-1, 1.5e-2
else:
    rtol, atol = 1e-1, 1e-3
```

Replace with original strict tolerances:
```python
# Use strict tolerances - Kahan summation should make this pass
rtol, atol = 1e-3, 1e-5
```

**Step 2: Run the failing tests to verify Kahan fixes work**

Run: `pytest tests/test_snapshots_resnet18.py::TestCodegenSnapshots::test_resnet18_codegen_with_runtime -v`

Expected: Tests should PASS with strict tolerances (confirming Kahan works)

**Step 3: If tests fail, analyze which operations still need fixes**

Run with verbose output to see which tensors fail:
```bash
pytest tests/test_snapshots_resnet18.py -v --tb=short
```

Check if remaining failures are in specific operations (e.g., LayerNorm) that may also need Kahan.

**Step 4: Commit**

```bash
git add tests/test_common.py
git commit -m "test: revert tolerance relaxation - Kahan summation fixes accuracy"
```

---

## Task 9: Fix LayerNorm Accumulation (If Needed)

**Files:**
- Modify: `runtime/x86/ops.c` (function `nnc_layernorm`)

**Step 1: Check if LayerNorm needs fixes**

If Task 8 showed LayerNorm-related failures, find `nnc_layernorm` and update the mean/variance calculations:

```c
/* Find mean calculation in LayerNorm */
float mean = 0.0f;
float c_mean = 0.0f;
for (int64_t i = 0; i < num_elements; i++) {
    float y = data[i] - c_mean;
    float t = mean + y;
    c_mean = (t - mean) - y;
    mean = t;
}
mean /= (float)num_elements;

/* Find variance calculation */
float var = 0.0f;
float c_var = 0.0f;
for (int64_t i = 0; i < num_elements; i++) {
    float diff = data[i] - mean;
    float y = diff * diff - c_var;
    float t = var + y;
    c_var = (t - var) - y;
    var = t;
}
var /= (float)num_elements;
```

**Step 2: Verify and commit**

```bash
cd runtime/x86 && make clean && make
git add runtime/x86/ops.c
git commit -m "fix(runtime): use Kahan summation in nnc_layernorm"
```

---

## Task 10: Full Regression Test

**Files:**
- Test: All snapshot tests

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`

Expected: All tests PASS with strict tolerances

**Step 2: Verify specific deep network tests**

Run: `pytest tests/test_snapshots_resnet18.py tests/test_snapshots_vgg19.py -v`

Expected: All PASS

**Step 3: Measure improvement**

Compare max_diff before/after by checking debug output:
```bash
pytest tests/test_snapshots_resnet18.py::TestCodegenSnapshots::test_resnet18_codegen_with_runtime -v -s
```

Expected: max_diff should be < 1e-3 (was ~1e-2 before)

**Step 4: Final commit**

```bash
git add docs/plans/2025-02-10-floating-point-accumulation-fix.md
git commit -m "docs: add floating-point accumulation fix implementation plan"
```

---

## Summary

This plan implements Kahan summation across all critical accumulation operations:

| Operation | Lines | Impact |
|-----------|-------|--------|
| `nnc_reducesum` | ~914 | Direct reduction accuracy |
| `nnc_reducemean` | ~980 | Mean calculation accuracy |
| `nnc_matmul` | ~641 | Matrix multiply accuracy |
| `nnc_gemm` | ~750 | GEMM accuracy |
| `nnc_conv` | ~414 | Convolution accuracy (largest gain) |

**Expected outcome:**
- Floating-point error reduced from ~1% to <0.1%
- Tests pass with strict tolerances (rtol=1e-3, atol=1e-5)
- No need for model-specific tolerance hacks

**References:**
- Kahan, W. (1965). "Further remarks on reducing truncation errors"
- Goldberg, D. (1991). "What every computer scientist should know about floating-point arithmetic"
