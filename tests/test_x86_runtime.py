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

        # nnc_softmax(Tensor* input, Tensor* output, int axis)
        self.lib.nnc_softmax.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_int]
        self.lib.nnc_softmax.restype = None

        # nnc_matmul(Tensor* a, Tensor* b, Tensor* output)
        self.lib.nnc_matmul.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
        self.lib.nnc_matmul.restype = None

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

        # Create array of output pointers - need to keep tensors alive
        output_tensors = [tensor_out1, tensor_out2]
        self._tensor_refs.append(output_tensors)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
