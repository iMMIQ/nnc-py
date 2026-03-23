"""Test x86 runtime operators against numpy reference implementation.

This test compiles and runs C code for each operator and compares
the results with numpy's implementation.
"""

import ctypes
import ctypes.util
import numpy as np
import pytest
import subprocess
import tempfile
from pathlib import Path


def _build_conv_logging_probe(tmp_path: Path) -> Path:
    runtime_dir = Path(__file__).parent.parent / "runtime"
    source_path = tmp_path / "conv_logging_probe.c"
    source_path.write_text(
        """
#include "nnc_ops.h"

int main(void) {
    float input_data[1] = {1.0f};
    float weight_data[1] = {2.0f};
    float output_data[1] = {0.0f};

    int64_t input_shape[4] = {1, 1, 1, 1};
    int64_t weight_shape[4] = {1, 1, 1, 1};
    int64_t output_shape[4] = {1, 1, 1, 1};

    Tensor input = {
        .data = input_data,
        .dtype = NNC_DTYPE_FLOAT32,
        .shape = input_shape,
        .ndim = 4,
        .nbytes = sizeof(input_data),
    };
    Tensor weight = {
        .data = weight_data,
        .dtype = NNC_DTYPE_FLOAT32,
        .shape = weight_shape,
        .ndim = 4,
        .nbytes = sizeof(weight_data),
    };
    Tensor output = {
        .data = output_data,
        .dtype = NNC_DTYPE_FLOAT32,
        .shape = output_shape,
        .ndim = 4,
        .nbytes = sizeof(output_data),
    };

    nnc_conv(&input, &weight, NULL, &output, 1, 1, 1, 1, 0, 0);
    return output_data[0] == 2.0f ? 0 : 1;
}
"""
    )

    exe_path = tmp_path / "conv_logging_probe"
    cmd = [
        "gcc",
        "-std=c11",
        "-O2",
        f"-I{runtime_dir / 'include'}",
    ]
    cmd.extend(
        [
            str(runtime_dir / "x86" / "ops.c"),
            str(source_path),
            "-lm",
            "-o",
            str(exe_path),
        ]
    )
    build = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert build.returncode == 0, build.stderr or build.stdout
    return exe_path


def _reference_conv_nchw(
    input_data: np.ndarray,
    weight_data: np.ndarray,
    bias_data: np.ndarray | None,
    *,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
) -> np.ndarray:
    n, c_in, h_in, w_in = input_data.shape
    c_out, _, kernel_h, kernel_w = weight_data.shape
    h_out = ((h_in + 2 * pad_h - kernel_h) // stride_h) + 1
    w_out = ((w_in + 2 * pad_w - kernel_w) // stride_w) + 1
    output = np.zeros((n, c_out, h_out, w_out), dtype=np.float32)

    for batch in range(n):
        for out_ch in range(c_out):
            for h_out_idx in range(h_out):
                h_base = h_out_idx * stride_h - pad_h
                for w_out_idx in range(w_out):
                    w_base = w_out_idx * stride_w - pad_w
                    acc = 0.0
                    for in_ch in range(c_in):
                        for kh in range(kernel_h):
                            h = h_base + kh
                            if h < 0 or h >= h_in:
                                continue
                            for kw in range(kernel_w):
                                w = w_base + kw
                                if w < 0 or w >= w_in:
                                    continue
                                acc += (
                                    float(input_data[batch, in_ch, h, w])
                                    * float(weight_data[out_ch, in_ch, kh, kw])
                                )
                    if bias_data is not None:
                        acc += float(bias_data[out_ch])
                    output[batch, out_ch, h_out_idx, w_out_idx] = acc

    return output


def _reference_gemm(
    a_data: np.ndarray,
    b_data: np.ndarray,
    c_data: np.ndarray | None,
    *,
    alpha: float,
    beta: float,
    trans_a: int,
    trans_b: int,
) -> np.ndarray:
    a_mat = a_data.T if trans_a else a_data
    b_mat = b_data.T if trans_b else b_data
    result = alpha * np.matmul(a_mat, b_mat)
    if c_data is not None:
        result = result + (beta * c_data)
    return result.astype(np.float32, copy=False)


def test_conv_debug_logging_is_disabled_by_default(tmp_path):
    exe_path = _build_conv_logging_probe(tmp_path)

    run = subprocess.run([str(exe_path)], capture_output=True, text=True, timeout=60)

    assert run.returncode == 0, run.stderr or run.stdout
    assert run.stderr == ""


class TestX86Runtime:
    """Test x86 runtime operator implementations."""

    @pytest.fixture(autouse=True)
    def setup_runtime(self):
        """Compile the runtime library for testing."""
        self.runtime_dir = Path(__file__).parent.parent / "runtime"
        self.build_dir = Path(tempfile.mkdtemp())

        # Keep references to prevent GC
        self._tensor_refs = []  # Store (shape_arr, data_arr) tuples

        # Compile runtime ops
        self._compile_runtime()

        yield

        # Cleanup
        import shutil
        try:
            shutil.rmtree(self.build_dir)
        except:
            pass

    def _compile_runtime(self):
        """Compile the runtime library."""
        # Create a simple test program
        makefile = self.build_dir / "Makefile"

        # Generate Makefile
        # Note: ASan disabled for shared library tests - causes hangs with ctypes
        makefile_content = f"""CC = gcc
CFLAGS = -std=c11 -O2 -Wall -fPIC
LDFLAGS = -lm

NNC_RUNTIME = {self.runtime_dir}
CFLAGS += -I$(NNC_RUNTIME)/include

.PHONY: all clean

all: libruntime.so

libruntime.so: ops.o
	$(CC) -shared -o $@ $^ $(LDFLAGS)

ops.o:
	$(CC) $(CFLAGS) -c $(NNC_RUNTIME)/x86/ops.c -o $@

clean:
	rm -f *.o *.so
"""
        makefile.write_text(makefile_content)

        # Compile
        result = subprocess.run(
            ["make"],
            cwd=self.build_dir,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile runtime: {result.stderr}")

        # Load the shared library
        lib_path = self.build_dir / "libruntime.so"
        self.lib = ctypes.CDLL(str(lib_path))

        # Set up function signatures
        self._setup_function_signatures()

    def _setup_function_signatures(self):
        """Set up ctypes function signatures."""
        # Tensor structure
        class Tensor(ctypes.Structure):
            _fields_ = [
                ("data", ctypes.c_void_p),
                ("dtype", ctypes.c_int),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("ndim", ctypes.c_int),
                ("nbytes", ctypes.c_int64),
            ]

        self.Tensor = Tensor

        # nnc_add(Tensor* a, Tensor* b, Tensor* out)
        self.lib.nnc_add.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_add.restype = None

        # nnc_mul(Tensor* a, Tensor* b, Tensor* out)
        self.lib.nnc_mul.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_mul.restype = None

        # nnc_sub(Tensor* a, Tensor* b, Tensor* out)
        self.lib.nnc_sub.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_sub.restype = None

        # nnc_div(Tensor* a, Tensor* b, Tensor* out)
        self.lib.nnc_div.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_div.restype = None

        # nnc_relu(Tensor* input, Tensor* output)
        self.lib.nnc_relu.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_relu.restype = None

        # nnc_sigmoid(Tensor* input, Tensor* output)
        self.lib.nnc_sigmoid.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_sigmoid.restype = None

        # nnc_tanh(Tensor* input, Tensor* output)
        self.lib.nnc_tanh.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_tanh.restype = None

        # nnc_conv(Tensor* input, Tensor* weight, Tensor* bias, Tensor* output, ...)
        self.lib.nnc_conv.argtypes = [
            ctypes.POINTER(Tensor),
            ctypes.POINTER(Tensor),
            ctypes.POINTER(Tensor),
            ctypes.POINTER(Tensor),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.nnc_conv.restype = None

        # nnc_conv_relu(Tensor* input, Tensor* weight, Tensor* bias, Tensor* output, ...)
        self.lib.nnc_conv_relu.argtypes = list(self.lib.nnc_conv.argtypes)
        self.lib.nnc_conv_relu.restype = None
        self.lib.nnc_conv1x1.argtypes = [
            ctypes.POINTER(Tensor),
            ctypes.POINTER(Tensor),
            ctypes.POINTER(Tensor),
            ctypes.POINTER(Tensor),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.nnc_conv1x1.restype = None
        self.lib.nnc_conv3x3_s1.argtypes = [
            ctypes.POINTER(Tensor),
            ctypes.POINTER(Tensor),
            ctypes.POINTER(Tensor),
            ctypes.POINTER(Tensor),
        ]
        self.lib.nnc_conv3x3_s1.restype = None
        self.lib.nnc_conv7x7_s2.argtypes = list(self.lib.nnc_conv3x3_s1.argtypes)
        self.lib.nnc_conv7x7_s2.restype = None
        self.lib.nnc_conv_relu1x1.argtypes = list(self.lib.nnc_conv1x1.argtypes)
        self.lib.nnc_conv_relu1x1.restype = None
        self.lib.nnc_conv_relu3x3_s1.argtypes = list(self.lib.nnc_conv3x3_s1.argtypes)
        self.lib.nnc_conv_relu3x3_s1.restype = None
        self.lib.nnc_conv_relu7x7_s2.argtypes = list(self.lib.nnc_conv3x3_s1.argtypes)
        self.lib.nnc_conv_relu7x7_s2.restype = None

        # nnc_softmax(Tensor* input, Tensor* output, int axis)
        self.lib.nnc_softmax.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_int]
        self.lib.nnc_softmax.restype = None

        # nnc_matmul(Tensor* a, Tensor* b, Tensor* output)
        self.lib.nnc_matmul.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_matmul.restype = None
        self.lib.nnc_gemm_nt.argtypes = [
            ctypes.POINTER(Tensor),
            ctypes.POINTER(Tensor),
            ctypes.POINTER(Tensor),
            ctypes.POINTER(Tensor),
            ctypes.c_float,
            ctypes.c_float,
        ]
        self.lib.nnc_gemm_nt.restype = None

        # nnc_transpose(Tensor* input, Tensor* output, int64_t* perm, int ndim)
        self.lib.nnc_transpose.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
        self.lib.nnc_transpose.restype = None

        # nnc_reshape(Tensor* input, Tensor* output, int64_t* shape, int ndim)
        self.lib.nnc_reshape.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
        self.lib.nnc_reshape.restype = None

        # nnc_concat(Tensor** inputs, Tensor* output, int num_inputs, int axis)
        self.lib.nnc_concat.argtypes = [ctypes.POINTER(ctypes.POINTER(Tensor)), ctypes.POINTER(Tensor), ctypes.c_int, ctypes.c_int]
        self.lib.nnc_concat.restype = None

        # nnc_split(Tensor* input, Tensor** outputs, int num_outputs, int axis)
        self.lib.nnc_split.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(ctypes.POINTER(Tensor)), ctypes.c_int, ctypes.c_int]
        self.lib.nnc_split.restype = None

        # nnc_reducemean(Tensor* input, Tensor* output, int axis, int keepdims)
        self.lib.nnc_reducemean.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_int, ctypes.c_int]
        self.lib.nnc_reducemean.restype = None

        # nnc_reducesum(Tensor* input, Tensor* output, int axis, int keepdims)
        self.lib.nnc_reducesum.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_int, ctypes.c_int]
        self.lib.nnc_reducesum.restype = None

        # nnc_tile(Tensor* input, Tensor* output, int64_t* repeats, int ndim)
        self.lib.nnc_tile.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
        self.lib.nnc_tile.restype = None

        # nnc_equal(Tensor* a, Tensor* b, Tensor* out)
        self.lib.nnc_equal.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_equal.restype = None

        # nnc_and(Tensor* a, Tensor* b, Tensor* out)
        self.lib.nnc_and.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_and.restype = None

        # Math operations (unary)
        self.lib.nnc_sqrt.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_sqrt.restype = None
        self.lib.nnc_exp.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_exp.restype = None
        self.lib.nnc_log.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_log.restype = None
        self.lib.nnc_abs.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_abs.restype = None
        self.lib.nnc_neg.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_neg.restype = None

    def _make_tensor(self, np_array):
        """Create a Tensor from numpy array."""
        # Ensure C-contiguous and correct dtype
        arr = np.ascontiguousarray(np_array.astype(np.float32))
        data_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        shape_arr = np.array(np_array.shape, dtype=np.int64)
        shape_ptr = shape_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

        tensor = self.Tensor(
            data=ctypes.cast(data_ptr, ctypes.c_void_p),
            dtype=0,  # NNC_DTYPE_FLOAT32
            shape=shape_ptr,
            ndim=len(np_array.shape),
            nbytes=arr.nbytes
        )

        # Keep references to prevent GC
        self._tensor_refs.append((shape_arr, arr))

        return tensor, arr

    def _compare_results(self, c_result, np_result, tol=1e-5):
        """Compare C result with numpy result."""
        c_flat = c_result.flatten()
        np_flat = np_result.flatten()

        assert len(c_flat) == len(np_flat), f"Size mismatch: {len(c_flat)} vs {len(np_flat)}"

        max_diff = 0
        for i, (c_val, np_val) in enumerate(zip(c_flat, np_flat)):
            diff = abs(c_val - np_val)
            max_diff = max(max_diff, diff)
            if diff > tol:
                # For very small values, use relative tolerance
                if abs(np_val) > 1e-10:
                    rel_diff = diff / abs(np_val)
                    assert rel_diff < 1e-3, f"Mismatch at index {i}: C={c_val}, numpy={np_val}, diff={diff}"

        return max_diff

    def test_add(self):
        """Test nnc_add."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        expected = a + b

        tensor_a, data_a = self._make_tensor(a)
        tensor_b, data_b = self._make_tensor(b)

        output = np.zeros_like(a)
        tensor_out, data_out = self._make_tensor(output)

        self.lib.nnc_add(ctypes.byref(tensor_a), ctypes.byref(tensor_b), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  add max_diff: {max_diff}")

    def test_mul(self):
        """Test nnc_mul."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        expected = a * b

        tensor_a, data_a = self._make_tensor(a)
        tensor_b, data_b = self._make_tensor(b)

        out = np.zeros_like(a)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_mul(ctypes.byref(tensor_a), ctypes.byref(tensor_b), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  mul max_diff: {max_diff}")

    def test_sub(self):
        """Test nnc_sub."""
        a = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected = a - b

        tensor_a, data_a = self._make_tensor(a)
        tensor_b, data_b = self._make_tensor(b)

        out = np.zeros_like(a)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_sub(ctypes.byref(tensor_a), ctypes.byref(tensor_b), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  sub max_diff: {max_diff}")

    def test_div(self):
        """Test nnc_div."""
        a = np.array([[10.0, 12.0], [14.0, 16.0]], dtype=np.float32)
        b = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        expected = a / b

        tensor_a, data_a = self._make_tensor(a)
        tensor_b, data_b = self._make_tensor(b)

        out = np.zeros_like(a)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_div(ctypes.byref(tensor_a), ctypes.byref(tensor_b), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  div max_diff: {max_diff}")

    def test_relu(self):
        """Test nnc_relu."""
        a = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
        expected = np.maximum(0, a)

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(a)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_relu(ctypes.byref(tensor_a), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  relu max_diff: {max_diff}")

    def test_sigmoid(self):
        """Test nnc_sigmoid."""
        a = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
        expected = 1 / (1 + np.exp(-a))

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(a)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_sigmoid(ctypes.byref(tensor_a), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  sigmoid max_diff: {max_diff}")

    def test_tanh(self):
        """Test nnc_tanh."""
        a = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
        expected = np.tanh(a)

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(a)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_tanh(ctypes.byref(tensor_a), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  tanh max_diff: {max_diff}")

    def test_softmax(self):
        """Test nnc_softmax along last axis."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

        # Compute softmax along axis 1
        exp_a = np.exp(a - np.max(a, axis=1, keepdims=True))
        expected = exp_a / np.sum(exp_a, axis=1, keepdims=True)

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(a)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_softmax(ctypes.byref(tensor_a), ctypes.byref(tensor_out), 1)

        max_diff = self._compare_results(data_out, expected, tol=1e-4)
        print(f"  softmax max_diff: {max_diff}")

    def test_conv_no_padding_matches_reference(self):
        """Test nnc_conv fast path for no-padding 3x3 convolution."""
        input_data = np.arange(1, 26, dtype=np.float32).reshape(1, 1, 5, 5)
        weight_data = np.array(
            [[[[1.0, 0.0, -1.0], [0.5, 0.25, -0.5], [1.5, -0.25, 0.75]]]],
            dtype=np.float32,
        )
        bias_data = np.array([0.75], dtype=np.float32)
        expected = _reference_conv_nchw(
            input_data,
            weight_data,
            bias_data,
            stride_h=1,
            stride_w=1,
            pad_h=0,
            pad_w=0,
        )

        tensor_input, _ = self._make_tensor(input_data)
        tensor_weight, _ = self._make_tensor(weight_data)
        tensor_bias, _ = self._make_tensor(bias_data)
        tensor_output, data_output = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_conv(
            ctypes.byref(tensor_input),
            ctypes.byref(tensor_weight),
            ctypes.byref(tensor_bias),
            ctypes.byref(tensor_output),
            3,
            3,
            1,
            1,
            0,
            0,
        )

        max_diff = self._compare_results(data_output, expected)
        print(f"  conv_no_padding max_diff: {max_diff}")

    def test_conv_same_padding_matches_reference(self):
        """Test nnc_conv for the common 3x3 stride-1 same-padding case."""
        input_data = np.array(
            [
                [
                    [[1.0, -2.0, 3.0, 0.5, -1.5], [2.5, 0.0, -1.0, 4.0, 1.5], [-3.0, 2.0, 1.0, -0.5, 2.5], [1.5, -1.5, 2.5, 3.0, -2.0], [0.25, 1.25, -2.5, 0.75, 1.0]],
                    [[-1.0, 0.5, 2.0, -2.5, 1.0], [3.0, -1.5, 0.25, 1.5, -0.75], [1.25, 2.5, -3.0, 0.5, 2.0], [-2.0, 1.0, 1.5, -1.0, 0.25], [0.5, -0.5, 2.25, 1.75, -1.25]],
                ]
            ],
            dtype=np.float32,
        )
        weight_data = np.array(
            [
                [
                    [[0.5, -1.0, 0.25], [1.5, 0.75, -0.5], [-0.25, 0.5, 1.0]],
                    [[-0.75, 0.5, 1.25], [0.0, -1.5, 0.5], [1.0, -0.25, 0.75]],
                ],
                [
                    [[1.0, 0.25, -0.5], [-1.25, 0.5, 0.75], [0.5, -0.75, 1.5]],
                    [[0.25, -1.0, 0.5], [1.5, 0.25, -0.25], [-0.5, 1.0, 0.75]],
                ],
            ],
            dtype=np.float32,
        )
        bias_data = np.array([0.5, -0.25], dtype=np.float32)
        expected = _reference_conv_nchw(
            input_data,
            weight_data,
            bias_data,
            stride_h=1,
            stride_w=1,
            pad_h=1,
            pad_w=1,
        )

        tensor_input, _ = self._make_tensor(input_data)
        tensor_weight, _ = self._make_tensor(weight_data)
        tensor_bias, _ = self._make_tensor(bias_data)
        tensor_output, data_output = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_conv(
            ctypes.byref(tensor_input),
            ctypes.byref(tensor_weight),
            ctypes.byref(tensor_bias),
            ctypes.byref(tensor_output),
            3,
            3,
            1,
            1,
            1,
            1,
        )

        max_diff = self._compare_results(data_output, expected)
        print(f"  conv_same_padding max_diff: {max_diff}")

    def test_conv_same_padding_wide_matches_reference(self):
        """Test nnc_conv for a wider 3x3 stride-1 same-padding output row."""
        input_data = np.arange(1, 1 + (1 * 2 * 6 * 10), dtype=np.float32).reshape(1, 2, 6, 10)
        weight_data = np.array(
            [
                [
                    [[0.5, -1.0, 0.25], [1.5, 0.75, -0.5], [-0.25, 0.5, 1.0]],
                    [[-0.75, 0.5, 1.25], [0.0, -1.5, 0.5], [1.0, -0.25, 0.75]],
                ],
                [
                    [[1.0, 0.25, -0.5], [-1.25, 0.5, 0.75], [0.5, -0.75, 1.5]],
                    [[0.25, -1.0, 0.5], [1.5, 0.25, -0.25], [-0.5, 1.0, 0.75]],
                ],
            ],
            dtype=np.float32,
        )
        bias_data = np.array([0.5, -0.25], dtype=np.float32)
        expected = _reference_conv_nchw(
            input_data,
            weight_data,
            bias_data,
            stride_h=1,
            stride_w=1,
            pad_h=1,
            pad_w=1,
        )

        tensor_input, _ = self._make_tensor(input_data)
        tensor_weight, _ = self._make_tensor(weight_data)
        tensor_bias, _ = self._make_tensor(bias_data)
        tensor_output, data_output = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_conv(
            ctypes.byref(tensor_input),
            ctypes.byref(tensor_weight),
            ctypes.byref(tensor_bias),
            ctypes.byref(tensor_output),
            3,
            3,
            1,
            1,
            1,
            1,
        )

        max_diff = self._compare_results(data_output, expected)
        print(f"  conv_same_padding_wide max_diff: {max_diff}")

    def test_conv_same_padding_stride2_matches_reference(self):
        """Test nnc_conv for the common 3x3 stride-2 same-padding case."""
        input_data = np.array(
            [
                [
                    [[1.0, -2.0, 3.0, 0.5, -1.5, 2.0], [2.5, 0.0, -1.0, 4.0, 1.5, -0.25], [-3.0, 2.0, 1.0, -0.5, 2.5, 1.25], [1.5, -1.5, 2.5, 3.0, -2.0, 0.75], [0.25, 1.25, -2.5, 0.75, 1.0, -1.0], [2.0, -0.75, 1.5, -2.0, 0.5, 3.0]],
                    [[-1.0, 0.5, 2.0, -2.5, 1.0, 0.25], [3.0, -1.5, 0.25, 1.5, -0.75, 2.25], [1.25, 2.5, -3.0, 0.5, 2.0, -1.5], [-2.0, 1.0, 1.5, -1.0, 0.25, 1.75], [0.5, -0.5, 2.25, 1.75, -1.25, 0.0], [1.5, -2.25, 0.75, 2.5, -0.5, 1.0]],
                ]
            ],
            dtype=np.float32,
        )
        weight_data = np.array(
            [
                [
                    [[0.5, -1.0, 0.25], [1.5, 0.75, -0.5], [-0.25, 0.5, 1.0]],
                    [[-0.75, 0.5, 1.25], [0.0, -1.5, 0.5], [1.0, -0.25, 0.75]],
                ],
                [
                    [[1.0, 0.25, -0.5], [-1.25, 0.5, 0.75], [0.5, -0.75, 1.5]],
                    [[0.25, -1.0, 0.5], [1.5, 0.25, -0.25], [-0.5, 1.0, 0.75]],
                ],
            ],
            dtype=np.float32,
        )
        bias_data = np.array([0.5, -0.25], dtype=np.float32)
        expected = _reference_conv_nchw(
            input_data,
            weight_data,
            bias_data,
            stride_h=2,
            stride_w=2,
            pad_h=1,
            pad_w=1,
        )

        tensor_input, _ = self._make_tensor(input_data)
        tensor_weight, _ = self._make_tensor(weight_data)
        tensor_bias, _ = self._make_tensor(bias_data)
        tensor_output, data_output = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_conv(
            ctypes.byref(tensor_input),
            ctypes.byref(tensor_weight),
            ctypes.byref(tensor_bias),
            ctypes.byref(tensor_output),
            3,
            3,
            2,
            2,
            1,
            1,
        )

        max_diff = self._compare_results(data_output, expected)
        print(f"  conv_same_padding_stride2 max_diff: {max_diff}")

    def test_conv_same_padding_stride2_wide_matches_reference(self):
        """Test nnc_conv for a wider 3x3 stride-2 same-padding output row."""
        input_data = np.arange(1, 1 + (1 * 2 * 8 * 12), dtype=np.float32).reshape(1, 2, 8, 12)
        weight_data = np.array(
            [
                [
                    [[0.5, -1.0, 0.25], [1.5, 0.75, -0.5], [-0.25, 0.5, 1.0]],
                    [[-0.75, 0.5, 1.25], [0.0, -1.5, 0.5], [1.0, -0.25, 0.75]],
                ],
                [
                    [[1.0, 0.25, -0.5], [-1.25, 0.5, 0.75], [0.5, -0.75, 1.5]],
                    [[0.25, -1.0, 0.5], [1.5, 0.25, -0.25], [-0.5, 1.0, 0.75]],
                ],
            ],
            dtype=np.float32,
        )
        bias_data = np.array([0.5, -0.25], dtype=np.float32)
        expected = _reference_conv_nchw(
            input_data,
            weight_data,
            bias_data,
            stride_h=2,
            stride_w=2,
            pad_h=1,
            pad_w=1,
        )

        tensor_input, _ = self._make_tensor(input_data)
        tensor_weight, _ = self._make_tensor(weight_data)
        tensor_bias, _ = self._make_tensor(bias_data)
        tensor_output, data_output = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_conv(
            ctypes.byref(tensor_input),
            ctypes.byref(tensor_weight),
            ctypes.byref(tensor_bias),
            ctypes.byref(tensor_output),
            3,
            3,
            2,
            2,
            1,
            1,
        )

        max_diff = self._compare_results(data_output, expected)
        print(f"  conv_same_padding_stride2_wide max_diff: {max_diff}")

    def test_conv_1x1_matches_reference(self):
        """Test nnc_conv fast path for 1x1 convolution."""
        input_data = np.array(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[-1.0, -2.0], [-3.0, -4.0]],
                ]
            ],
            dtype=np.float32,
        )
        weight_data = np.array(
            [
                [[[2.0]], [[-1.0]]],
                [[[0.5]], [[3.0]]],
            ],
            dtype=np.float32,
        )
        bias_data = np.array([0.25, -0.5], dtype=np.float32)
        expected = _reference_conv_nchw(
            input_data,
            weight_data,
            bias_data,
            stride_h=1,
            stride_w=1,
            pad_h=0,
            pad_w=0,
        )

        tensor_input, _ = self._make_tensor(input_data)
        tensor_weight, _ = self._make_tensor(weight_data)
        tensor_bias, _ = self._make_tensor(bias_data)
        tensor_output, data_output = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_conv(
            ctypes.byref(tensor_input),
            ctypes.byref(tensor_weight),
            ctypes.byref(tensor_bias),
            ctypes.byref(tensor_output),
            1,
            1,
            1,
            1,
            0,
            0,
        )

        max_diff = self._compare_results(data_output, expected)
        print(f"  conv_1x1 max_diff: {max_diff}")

    def test_conv1x1_matches_reference(self):
        """Test specialized nnc_conv1x1 kernel."""
        input_data = np.array(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[-1.0, -2.0], [-3.0, -4.0]],
                ]
            ],
            dtype=np.float32,
        )
        weight_data = np.array(
            [
                [[[2.0]], [[-1.0]]],
                [[[0.5]], [[3.0]]],
            ],
            dtype=np.float32,
        )
        bias_data = np.array([0.25, -0.5], dtype=np.float32)
        expected = _reference_conv_nchw(
            input_data,
            weight_data,
            bias_data,
            stride_h=1,
            stride_w=1,
            pad_h=0,
            pad_w=0,
        )

        tensor_input, _ = self._make_tensor(input_data)
        tensor_weight, _ = self._make_tensor(weight_data)
        tensor_bias, _ = self._make_tensor(bias_data)
        tensor_output, data_output = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_conv1x1(
            ctypes.byref(tensor_input),
            ctypes.byref(tensor_weight),
            ctypes.byref(tensor_bias),
            ctypes.byref(tensor_output),
            1,
            1,
            0,
            0,
        )

        max_diff = self._compare_results(data_output, expected)
        print(f"  conv1x1 max_diff: {max_diff}")

    def test_conv_relu1x1_matches_reference(self):
        """Test specialized nnc_conv_relu1x1 kernel."""
        input_data = np.array(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[-1.0, -2.0], [-3.0, -4.0]],
                ]
            ],
            dtype=np.float32,
        )
        weight_data = np.array(
            [
                [[[2.0]], [[-1.0]]],
                [[[0.5]], [[3.0]]],
            ],
            dtype=np.float32,
        )
        bias_data = np.array([0.25, -0.5], dtype=np.float32)
        expected = np.maximum(
            _reference_conv_nchw(
                input_data,
                weight_data,
                bias_data,
                stride_h=1,
                stride_w=1,
                pad_h=0,
                pad_w=0,
            ),
            0.0,
        )

        tensor_input, _ = self._make_tensor(input_data)
        tensor_weight, _ = self._make_tensor(weight_data)
        tensor_bias, _ = self._make_tensor(bias_data)
        tensor_output, data_output = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_conv_relu1x1(
            ctypes.byref(tensor_input),
            ctypes.byref(tensor_weight),
            ctypes.byref(tensor_bias),
            ctypes.byref(tensor_output),
            1,
            1,
            0,
            0,
        )

        max_diff = self._compare_results(data_output, expected)
        print(f"  conv_relu1x1 max_diff: {max_diff}")

    def test_conv3x3_s1_matches_reference(self):
        """Test specialized nnc_conv3x3_s1 kernel."""
        input_data = np.arange(1, 1 + (1 * 2 * 5 * 6), dtype=np.float32).reshape(1, 2, 5, 6)
        weight_data = np.array(
            [
                [
                    [[0.5, -1.0, 0.25], [1.5, 0.75, -0.5], [-0.25, 0.5, 1.0]],
                    [[-0.75, 0.5, 1.25], [0.0, -1.5, 0.5], [1.0, -0.25, 0.75]],
                ],
                [
                    [[1.0, 0.25, -0.5], [-1.25, 0.5, 0.75], [0.5, -0.75, 1.5]],
                    [[0.25, -1.0, 0.5], [1.5, 0.25, -0.25], [-0.5, 1.0, 0.75]],
                ],
            ],
            dtype=np.float32,
        )
        bias_data = np.array([0.5, -0.25], dtype=np.float32)
        expected = _reference_conv_nchw(
            input_data,
            weight_data,
            bias_data,
            stride_h=1,
            stride_w=1,
            pad_h=1,
            pad_w=1,
        )

        tensor_input, _ = self._make_tensor(input_data)
        tensor_weight, _ = self._make_tensor(weight_data)
        tensor_bias, _ = self._make_tensor(bias_data)
        tensor_output, data_output = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_conv3x3_s1(
            ctypes.byref(tensor_input),
            ctypes.byref(tensor_weight),
            ctypes.byref(tensor_bias),
            ctypes.byref(tensor_output),
        )

        max_diff = self._compare_results(data_output, expected)
        print(f"  conv3x3_s1 max_diff: {max_diff}")

    def test_conv_relu3x3_s1_matches_reference(self):
        """Test specialized nnc_conv_relu3x3_s1 kernel."""
        input_data = np.arange(1, 1 + (1 * 2 * 5 * 6), dtype=np.float32).reshape(1, 2, 5, 6)
        weight_data = np.array(
            [
                [
                    [[0.5, -1.0, 0.25], [1.5, 0.75, -0.5], [-0.25, 0.5, 1.0]],
                    [[-0.75, 0.5, 1.25], [0.0, -1.5, 0.5], [1.0, -0.25, 0.75]],
                ],
                [
                    [[1.0, 0.25, -0.5], [-1.25, 0.5, 0.75], [0.5, -0.75, 1.5]],
                    [[0.25, -1.0, 0.5], [1.5, 0.25, -0.25], [-0.5, 1.0, 0.75]],
                ],
            ],
            dtype=np.float32,
        )
        bias_data = np.array([0.5, -0.25], dtype=np.float32)
        expected = np.maximum(
            _reference_conv_nchw(
                input_data,
                weight_data,
                bias_data,
                stride_h=1,
                stride_w=1,
                pad_h=1,
                pad_w=1,
            ),
            0.0,
        )

        tensor_input, _ = self._make_tensor(input_data)
        tensor_weight, _ = self._make_tensor(weight_data)
        tensor_bias, _ = self._make_tensor(bias_data)
        tensor_output, data_output = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_conv_relu3x3_s1(
            ctypes.byref(tensor_input),
            ctypes.byref(tensor_weight),
            ctypes.byref(tensor_bias),
            ctypes.byref(tensor_output),
        )

        max_diff = self._compare_results(data_output, expected)
        print(f"  conv_relu3x3_s1 max_diff: {max_diff}")

    def test_conv7x7_s2_matches_reference(self):
        """Test specialized nnc_conv7x7_s2 kernel."""
        input_data = np.arange(1, 1 + (1 * 3 * 9 * 9), dtype=np.float32).reshape(1, 3, 9, 9)
        weight_data = (
            np.arange(1, 1 + (2 * 3 * 7 * 7), dtype=np.float32).reshape(2, 3, 7, 7) / 50.0
        )
        bias_data = np.array([0.5, -1.25], dtype=np.float32)
        expected = _reference_conv_nchw(
            input_data,
            weight_data,
            bias_data,
            stride_h=2,
            stride_w=2,
            pad_h=3,
            pad_w=3,
        )

        tensor_input, _ = self._make_tensor(input_data)
        tensor_weight, _ = self._make_tensor(weight_data)
        tensor_bias, _ = self._make_tensor(bias_data)
        tensor_output, data_output = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_conv7x7_s2(
            ctypes.byref(tensor_input),
            ctypes.byref(tensor_weight),
            ctypes.byref(tensor_bias),
            ctypes.byref(tensor_output),
        )

        max_diff = self._compare_results(data_output, expected)
        print(f"  conv7x7_s2 max_diff: {max_diff}")

    def test_conv_relu7x7_s2_matches_reference(self):
        """Test specialized nnc_conv_relu7x7_s2 kernel."""
        input_data = np.arange(1, 1 + (1 * 3 * 9 * 9), dtype=np.float32).reshape(1, 3, 9, 9)
        weight_data = (
            np.arange(1, 1 + (2 * 3 * 7 * 7), dtype=np.float32).reshape(2, 3, 7, 7) / 50.0
        )
        bias_data = np.array([0.5, -1.25], dtype=np.float32)
        expected = np.maximum(
            _reference_conv_nchw(
                input_data,
                weight_data,
                bias_data,
                stride_h=2,
                stride_w=2,
                pad_h=3,
                pad_w=3,
            ),
            0.0,
        )

        tensor_input, _ = self._make_tensor(input_data)
        tensor_weight, _ = self._make_tensor(weight_data)
        tensor_bias, _ = self._make_tensor(bias_data)
        tensor_output, data_output = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_conv_relu7x7_s2(
            ctypes.byref(tensor_input),
            ctypes.byref(tensor_weight),
            ctypes.byref(tensor_bias),
            ctypes.byref(tensor_output),
        )

        max_diff = self._compare_results(data_output, expected)
        print(f"  conv_relu7x7_s2 max_diff: {max_diff}")

    def test_matmul_2d(self):
        """Test nnc_matmul with 2D matrices."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # [2, 3]
        b = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)  # [3, 2]
        expected = np.dot(a, b)  # [2, 2]

        tensor_a, data_a = self._make_tensor(a)
        tensor_b, data_b = self._make_tensor(b)

        out = np.zeros_like(expected)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_matmul(ctypes.byref(tensor_a), ctypes.byref(tensor_b), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  matmul_2d max_diff: {max_diff}")

    def test_gemm_transposed_weight_with_bias(self):
        """Test nnc_gemm for the trans_b=1 + bias path used by FC layers."""
        a = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]], dtype=np.float32)  # [2, 3]
        b = np.array(
            [[1.0, 0.0, -1.0], [2.0, 1.0, 0.5], [0.5, -0.25, 3.0], [4.0, 2.0, -2.0]],
            dtype=np.float32,
        )  # [4, 3] -> trans_b=1 gives [3, 4]
        c = np.array([0.25, -0.5, 1.25, 0.75], dtype=np.float32)  # [4]
        expected = _reference_gemm(a, b, c, alpha=1.0, beta=1.0, trans_a=0, trans_b=1)

        tensor_a, _ = self._make_tensor(a)
        tensor_b, _ = self._make_tensor(b)
        tensor_c, _ = self._make_tensor(c)
        tensor_out, data_out = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_gemm(
            ctypes.byref(tensor_a),
            ctypes.byref(tensor_b),
            ctypes.byref(tensor_c),
            ctypes.byref(tensor_out),
            ctypes.c_float(1.0),
            ctypes.c_float(1.0),
            0,
            1,
        )

        max_diff = self._compare_results(data_out, expected)
        print(f"  gemm_transposed_weight_with_bias max_diff: {max_diff}")

    def test_gemm_nt_matches_reference(self):
        """Test specialized nnc_gemm_nt path used by prepacked FC layers."""
        a = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]], dtype=np.float32)
        b = np.array(
            [[1.0, 0.0, -1.0], [2.0, 1.0, 0.5], [0.5, -0.25, 3.0], [4.0, 2.0, -2.0]],
            dtype=np.float32,
        )
        c = np.array([0.25, -0.5, 1.25, 0.75], dtype=np.float32)
        expected = _reference_gemm(a, b, c, alpha=1.0, beta=1.0, trans_a=0, trans_b=1)

        tensor_a, _ = self._make_tensor(a)
        tensor_b, _ = self._make_tensor(b)
        tensor_c, _ = self._make_tensor(c)
        tensor_out, data_out = self._make_tensor(np.zeros_like(expected))

        self.lib.nnc_gemm_nt(
            ctypes.byref(tensor_a),
            ctypes.byref(tensor_b),
            ctypes.byref(tensor_c),
            ctypes.byref(tensor_out),
            ctypes.c_float(1.0),
            ctypes.c_float(1.0),
        )

        max_diff = self._compare_results(data_out, expected)
        print(f"  gemm_nt max_diff: {max_diff}")

    def test_matmul_vector_matrix(self):
        """Test nnc_matmul with vector @ matrix."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # [3]
        b = np.array([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]], dtype=np.float32)  # [3, 2]
        expected = np.dot(a, b)  # [2]

        tensor_a, data_a = self._make_tensor(a)
        tensor_b, data_b = self._make_tensor(b)

        out = np.zeros_like(expected)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_matmul(ctypes.byref(tensor_a), ctypes.byref(tensor_b), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  matmul_vector_matrix max_diff: {max_diff}")

    def test_matmul_matrix_vector(self):
        """Test nnc_matmul with matrix @ vector."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # [2, 3]
        b = np.array([7.0, 8.0, 9.0], dtype=np.float32)  # [3]
        expected = np.dot(a, b)  # [2]

        tensor_a, data_a = self._make_tensor(a)
        tensor_b, data_b = self._make_tensor(b)

        out = np.zeros_like(expected)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_matmul(ctypes.byref(tensor_a), ctypes.byref(tensor_b), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  matmul_matrix_vector max_diff: {max_diff}")

    def test_transpose(self):
        """Test nnc_transpose."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # [2, 3]
        expected = a.T  # [3, 2]

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected)
        tensor_out, data_out = self._make_tensor(out)

        # Permutation for transpose: [1, 0]
        perm = np.array([1, 0], dtype=np.int64)
        perm_ptr = perm.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

        self.lib.nnc_transpose(ctypes.byref(tensor_a), ctypes.byref(tensor_out), perm_ptr, 2)

        max_diff = self._compare_results(data_out, expected)
        print(f"  transpose max_diff: {max_diff}")

    def test_transpose_3d(self):
        """Test nnc_transpose with 3D tensor."""
        a = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        expected = np.transpose(a, [2, 0, 1])

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected)
        tensor_out, data_out = self._make_tensor(out)

        perm = np.array([2, 0, 1], dtype=np.int64)
        perm_ptr = perm.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

        self.lib.nnc_transpose(ctypes.byref(tensor_a), ctypes.byref(tensor_out), perm_ptr, 3)

        max_diff = self._compare_results(data_out, expected)
        print(f"  transpose_3d max_diff: {max_diff}")

    def test_reshape(self):
        """Test nnc_reshape."""
        a = np.arange(12).reshape(3, 4).astype(np.float32)
        new_shape = [2, 6]
        expected = a.reshape(new_shape)

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected)
        tensor_out, data_out = self._make_tensor(out)

        shape_arr = np.array(new_shape, dtype=np.int64)
        shape_ptr = shape_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

        self.lib.nnc_reshape(ctypes.byref(tensor_a), ctypes.byref(tensor_out), shape_ptr, len(new_shape))

        max_diff = self._compare_results(data_out, expected)
        print(f"  reshape max_diff: {max_diff}")

    def test_concat(self):
        """Test nnc_concat along axis 0."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # [2, 2]
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)  # [2, 2]
        expected = np.concatenate([a, b], axis=0)  # [4, 2]

        tensor_a, data_a = self._make_tensor(a)
        tensor_b, data_b = self._make_tensor(b)

        out = np.zeros_like(expected)
        tensor_out, data_out = self._make_tensor(out)

        # Create array of input pointers - need to keep tensors alive
        input_tensors = [tensor_a, tensor_b]
        self._tensor_refs.append(input_tensors)

        input_ptrs = (ctypes.POINTER(self.Tensor) * 2)()
        input_ptrs[0] = ctypes.pointer(tensor_a)
        input_ptrs[1] = ctypes.pointer(tensor_b)

        self.lib.nnc_concat(input_ptrs, ctypes.byref(tensor_out), 2, 0)

        max_diff = self._compare_results(data_out, expected)
        print(f"  concat max_diff: {max_diff}")

    def test_concat_axis1(self):
        """Test nnc_concat along axis 1."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # [2, 2]
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)  # [2, 2]
        expected = np.concatenate([a, b], axis=1)  # [2, 4]

        tensor_a, data_a = self._make_tensor(a)
        tensor_b, data_b = self._make_tensor(b)

        out = np.zeros_like(expected)
        tensor_out, data_out = self._make_tensor(out)

        input_tensors = [tensor_a, tensor_b]
        self._tensor_refs.append(input_tensors)

        input_ptrs = (ctypes.POINTER(self.Tensor) * 2)()
        input_ptrs[0] = ctypes.pointer(tensor_a)
        input_ptrs[1] = ctypes.pointer(tensor_b)

        self.lib.nnc_concat(input_ptrs, ctypes.byref(tensor_out), 2, 1)

        max_diff = self._compare_results(data_out, expected)
        print(f"  concat_axis1 max_diff: {max_diff}")

    def test_split(self):
        """Test nnc_split along axis 1."""
        a = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)  # [2, 4]
        # Split into 2 equal parts along axis 1
        expected1, expected2 = np.split(a, 2, axis=1)  # [2, 2] each

        tensor_a, data_a = self._make_tensor(a)

        out1 = np.zeros_like(expected1)
        out2 = np.zeros_like(expected2)
        tensor_out1, data_out1 = self._make_tensor(out1)
        tensor_out2, data_out2 = self._make_tensor(out2)

        # Create array of output pointers
        output_ptrs = (ctypes.POINTER(self.Tensor) * 2)()
        output_ptrs[0] = ctypes.pointer(tensor_out1)
        output_ptrs[1] = ctypes.pointer(tensor_out2)

        self.lib.nnc_split(ctypes.byref(tensor_a), output_ptrs, 2, 1)

        max_diff1 = self._compare_results(data_out1, expected1)
        max_diff2 = self._compare_results(data_out2, expected2)
        print(f"  split max_diff: [{max_diff1}, {max_diff2}]")

    def test_reducemean(self):
        """Test nnc_reducemean along axis 1."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # [2, 3]
        expected = np.mean(a, axis=1, keepdims=True)  # [2, 1]

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_reducemean(ctypes.byref(tensor_a), ctypes.byref(tensor_out), 1, 1)

        max_diff = self._compare_results(data_out, expected)
        print(f"  reducemean max_diff: {max_diff}")

    def test_reducemean_no_keepdims(self):
        """Test nnc_reducemean without keepdims."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # [2, 3]
        expected = np.mean(a, axis=1, keepdims=False)  # [2]

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected, dtype=np.float32)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_reducemean(ctypes.byref(tensor_a), ctypes.byref(tensor_out), 1, 0)

        max_diff = self._compare_results(data_out, expected)
        print(f"  reducemean_no_keepdims max_diff: {max_diff}")

    def test_reducesum(self):
        """Test nnc_reducesum along axis 0."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # [2, 3]
        expected = np.sum(a, axis=0, keepdims=True)  # [1, 3]

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_reducesum(ctypes.byref(tensor_a), ctypes.byref(tensor_out), 0, 1)

        max_diff = self._compare_results(data_out, expected)
        print(f"  reducesum max_diff: {max_diff}")

    def test_reducesum_no_keepdims(self):
        """Test nnc_reducesum without keepdims."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # [2, 3]
        expected = np.sum(a, axis=0, keepdims=False)  # [3]

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected, dtype=np.float32)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_reducesum(ctypes.byref(tensor_a), ctypes.byref(tensor_out), 0, 0)

        max_diff = self._compare_results(data_out, expected)
        print(f"  reducesum_no_keepdims max_diff: {max_diff}")

    def test_tile_2d(self):
        """Test nnc_tile with 2D tensor."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # [2, 2]
        repeats = np.array([2, 3], dtype=np.int64)  # Repeat 2x on axis 0, 3x on axis 1
        expected = np.tile(a, repeats)  # [4, 6]

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected, dtype=np.float32)
        tensor_out, data_out = self._make_tensor(out)

        # Keep references to prevent GC
        self._tensor_refs.append(repeats)
        repeats_ptr = repeats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

        self.lib.nnc_tile(ctypes.byref(tensor_a), ctypes.byref(tensor_out), repeats_ptr, 2)

        max_diff = self._compare_results(data_out, expected)
        print(f"  tile_2d max_diff: {max_diff}")

    def test_tile_1d(self):
        """Test nnc_tile with 1D tensor."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # [3]
        repeats = np.array([4], dtype=np.int64)  # Repeat 4x on axis 0
        expected = np.tile(a, repeats)  # [12]

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected, dtype=np.float32)
        tensor_out, data_out = self._make_tensor(out)

        # Keep references to prevent GC
        self._tensor_refs.append(repeats)
        repeats_ptr = repeats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

        self.lib.nnc_tile(ctypes.byref(tensor_a), ctypes.byref(tensor_out), repeats_ptr, 1)

        max_diff = self._compare_results(data_out, expected)
        print(f"  tile_1d max_diff: {max_diff}")

    def test_equal(self):
        """Test nnc_equal - element-wise equality comparison."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
        # Expected: [[1, 1], [1, 0]] (float representation)
        expected = (a == b).astype(np.float32)

        tensor_a, data_a = self._make_tensor(a)
        tensor_b, data_b = self._make_tensor(b)

        out = np.zeros_like(expected, dtype=np.float32)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_equal(ctypes.byref(tensor_a), ctypes.byref(tensor_b), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  equal max_diff: {max_diff}")

    def test_equal_broadcast(self):
        """Test nnc_equal with broadcasting."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # [2, 2]
        b = np.array([1.0, 2.0], dtype=np.float32)  # [2] - broadcast to [2, 2]
        expected = (a == b).astype(np.float32)

        tensor_a, data_a = self._make_tensor(a)
        tensor_b, data_b = self._make_tensor(b)

        out = np.zeros_like(expected, dtype=np.float32)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_equal(ctypes.byref(tensor_a), ctypes.byref(tensor_b), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  equal_broadcast max_diff: {max_diff}")

    def test_and(self):
        """Test nnc_and - element-wise logical AND."""
        a = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        b = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        # Expected: [[1, 0], [0, 1]] (float representation of boolean AND)
        expected = ((a != 0) & (b != 0)).astype(np.float32)

        tensor_a, data_a = self._make_tensor(a)
        tensor_b, data_b = self._make_tensor(b)

        out = np.zeros_like(expected, dtype=np.float32)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_and(ctypes.byref(tensor_a), ctypes.byref(tensor_b), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  and max_diff: {max_diff}")

    def test_and_broadcast(self):
        """Test nnc_and with broadcasting."""
        a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # [2, 2]
        b = np.array([1.0, 0.0], dtype=np.float32)  # [2] - broadcast to [2, 2]
        expected = ((a != 0) & (b != 0)).astype(np.float32)

        tensor_a, data_a = self._make_tensor(a)
        tensor_b, data_b = self._make_tensor(b)

        out = np.zeros_like(expected, dtype=np.float32)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_and(ctypes.byref(tensor_a), ctypes.byref(tensor_b), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  and_broadcast max_diff: {max_diff}")

    def test_sqrt(self):
        """Test nnc_sqrt - element-wise square root."""
        a = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32)
        expected = np.sqrt(a)

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected, dtype=np.float32)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_sqrt(ctypes.byref(tensor_a), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  sqrt max_diff: {max_diff}")

    def test_exp(self):
        """Test nnc_exp - element-wise exponential."""
        a = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        expected = np.exp(a)

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected, dtype=np.float32)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_exp(ctypes.byref(tensor_a), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected, tol=1e-4)
        print(f"  exp max_diff: {max_diff}")

    def test_log(self):
        """Test nnc_log - element-wise natural logarithm."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected = np.log(a)

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected, dtype=np.float32)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_log(ctypes.byref(tensor_a), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected, tol=1e-4)
        print(f"  log max_diff: {max_diff}")

    def test_abs(self):
        """Test nnc_abs - element-wise absolute value."""
        a = np.array([[-1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
        expected = np.abs(a)

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected, dtype=np.float32)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_abs(ctypes.byref(tensor_a), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  abs max_diff: {max_diff}")

    def test_neg(self):
        """Test nnc_neg - element-wise negation."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected = -a

        tensor_a, data_a = self._make_tensor(a)

        out = np.zeros_like(expected, dtype=np.float32)
        tensor_out, data_out = self._make_tensor(out)

        self.lib.nnc_neg(ctypes.byref(tensor_a), ctypes.byref(tensor_out))

        max_diff = self._compare_results(data_out, expected)
        print(f"  neg max_diff: {max_diff}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
