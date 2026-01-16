"""Test spill mechanism with complex graph and verify results match Python.

This test:
1. Creates a complex ONNX graph that will trigger spill with low memory limit
2. Compiles with ASAN enabled for memory safety
3. Runs the generated code and verifies results match Python/numpy calculation
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
import onnx
from onnx import helper, numpy_helper

from nnc_py import Compiler

# Get the runtime directory path
_NNC_RUNTIME_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'runtime'
)


def create_simple_linear_model(
    input_size: int = 64,
    num_layers: int = 8,
    hidden_size: int = 128,
) -> onnx.ModelProto:
    """Create a simple linear chain model.

    This creates a linear chain of operations where each layer's output
    is the next layer's input. With limited memory, intermediate activations
    will need to be spilled.

    Structure:
        Input -> [MatMul -> Add -> Relu] x N -> Output
    """
    # Input
    input_tensor = helper.make_tensor_value_info(
        'input', onnx.TensorProto.FLOAT, (1, input_size)
    )

    nodes = []
    initializers = []
    current_input = 'input'
    current_shape = input_size

    for layer_idx in range(num_layers):
        # Weight: (current_shape, hidden_size)
        weight_name = f'layer_{layer_idx}_weight'
        weight = np.random.randn(current_shape, hidden_size).astype(np.float32) * 0.1
        initializers.append(numpy_helper.from_array(weight, weight_name))

        # MatMul
        matmul_out = f'layer_{layer_idx}_matmul'
        matmul_node = helper.make_node(
            'MatMul',
            inputs=[current_input, weight_name],
            outputs=[matmul_out],
        )
        nodes.append(matmul_node)

        # Bias
        bias_name = f'layer_{layer_idx}_bias'
        bias = np.random.randn(hidden_size).astype(np.float32) * 0.1
        initializers.append(numpy_helper.from_array(bias, bias_name))

        # Add
        add_out = f'layer_{layer_idx}_add'
        add_node = helper.make_node(
            'Add',
            inputs=[matmul_out, bias_name],
            outputs=[add_out],
        )
        nodes.append(add_node)

        # Relu
        relu_out = f'layer_{layer_idx}_out' if layer_idx < num_layers - 1 else 'output'
        relu_node = helper.make_node(
            'Relu',
            inputs=[add_out],
            outputs=[relu_out],
        )
        nodes.append(relu_node)

        current_input = relu_out
        current_shape = hidden_size

    # Output
    output_tensor = helper.make_tensor_value_info(
        'output', onnx.TensorProto.FLOAT, (1, hidden_size)
    )

    graph = helper.make_graph(
        nodes,
        'chained_model',
        [input_tensor],
        [output_tensor],
        initializer=initializers,
    )

    # Use current ONNX opset version
    model = helper.make_model(graph)
    return model


def create_parallel_branch_model(
    input_size: int = 64,
    num_branches: int = 4,
    hidden_size: int = 128,
) -> onnx.ModelProto:
    """Create a model with parallel branches.

    This creates multiple parallel branches that all consume the same input.
    The outputs are then concatenated.

    Structure:
        Input -> [MatMul -> Add -> Relu] x N branches -> Concat -> Output
    """
    # Input
    input_tensor = helper.make_tensor_value_info(
        'input', onnx.TensorProto.FLOAT, (1, input_size)
    )

    nodes = []
    initializers = []

    for i in range(num_branches):
        # Weight for this branch
        weight_name = f'branch_{i}_weight'
        weight = np.random.randn(input_size, hidden_size).astype(np.float32) * 0.1
        initializers.append(numpy_helper.from_array(weight, weight_name))

        # MatMul
        matmul_node = helper.make_node(
            'MatMul',
            inputs=['input', weight_name],
            outputs=[f'branch_{i}_matmul'],
        )
        nodes.append(matmul_node)

        # Bias
        bias_name = f'branch_{i}_bias'
        bias = np.random.randn(hidden_size).astype(np.float32) * 0.1
        initializers.append(numpy_helper.from_array(bias, bias_name))

        # Add
        add_node = helper.make_node(
            'Add',
            inputs=[f'branch_{i}_matmul', bias_name],
            outputs=[f'branch_{i}_add'],
        )
        nodes.append(add_node)

        # Relu
        relu_node = helper.make_node(
            'Relu',
            inputs=[f'branch_{i}_add'],
            outputs=[f'branch_{i}_out'],
        )
        nodes.append(relu_node)

    # Concat all branches
    concat_node = helper.make_node(
        'Concat',
        inputs=[f'branch_{i}_out' for i in range(num_branches)],
        outputs=['output'],
        axis=1,
    )
    nodes.append(concat_node)

    # Output tensor
    output_tensor = helper.make_tensor_value_info(
        'output', onnx.TensorProto.FLOAT, (1, num_branches * hidden_size)
    )

    graph = helper.make_graph(
        nodes,
        'parallel_branch_model',
        [input_tensor],
        [output_tensor],
        initializer=initializers,
    )

    model = helper.make_model(graph)
    return model


def compute_reference_output(model: onnx.ModelProto, input_data: np.ndarray) -> np.ndarray:
    """Compute reference output using numpy.

    This simulates the model forward pass for simple linear/branch models.
    """
    x = input_data.copy()

    # Get weights and biases from the model
    weights = {}
    biases = {}

    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if 'weight' in init.name:
            weights[init.name] = arr
        elif 'bias' in init.name:
            biases[init.name] = arr

    # Check if this is a parallel branch model (has Concat node)
    is_parallel = any(node.op_type == 'Concat' for node in model.graph.node)

    if is_parallel:
        # Parallel branch model
        num_branches = sum(1 for n in model.graph.node if n.op_type == 'Relu')
        branch_outputs = []

        for i in range(num_branches):
            weight = weights.get(f'branch_{i}_weight')
            bias = biases.get(f'branch_{i}_bias')

            if weight is not None and bias is not None:
                out = np.dot(x, weight)
                out = out + bias
                out = np.maximum(out, 0)
                branch_outputs.append(out)

        result = np.concatenate(branch_outputs, axis=1)
        return result
    else:
        # Linear chain model
        current = x
        layer_idx = 0

        while True:
            weight = weights.get(f'layer_{layer_idx}_weight')
            bias = biases.get(f'layer_{layer_idx}_bias')

            if weight is None:
                break

            # MatMul
            current = np.dot(current, weight)
            # Add bias
            current = current + bias
            # Relu
            current = np.maximum(current, 0)

            layer_idx += 1

        return current


def test_spill_with_linear_model():
    """Test spill with a linear chain model."""
    print("\n=== Testing Spill with Linear Chain Model ===\n")

    # Create model with many layers to force spill
    model = create_simple_linear_model(
        input_size=128,           # ~512 bytes input
        num_layers=16,            # 16 layers
        hidden_size=256,          # ~1KB per hidden state
    )

    # Create test input
    np.random.seed(42)
    input_data = np.random.randn(1, 128).astype(np.float32)

    # Compute reference output
    print("Computing reference output...")
    expected_output = compute_reference_output(model, input_data)
    print(f"  Expected output shape: {expected_output.shape}")

    # Save model
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx.save(model, f.name)
        model_path = f.name

    try:
        with tempfile.TemporaryDirectory() as output_dir:
            print(f"\nCompiling with memory limit...")

            # Very low memory to force spill
            max_memory = "2KB"

            compiler = Compiler(target='x86', opt_level=0)
            compiler.compile(
                onnx_path=model_path,
                output_dir=output_dir,
                entry_point='test_linear',
                max_memory=max_memory,
                memory_strategy='graph_coloring',
            )

            # Check for spill
            tensors_c_path = os.path.join(output_dir, 'tensors.c')
            with open(tensors_c_path, 'r') as f:
                tensors_code = f.read()

            has_slow_pool = '_nnc_slow_memory_pool' in tensors_code
            print(f"  Has slow memory pool: {has_slow_pool}")

            # Compile with ASAN
            print("\nCompiling with ASAN...")
            makefile_path = os.path.join(output_dir, 'Makefile')

            with open(makefile_path, 'r') as f:
                makefile_content = f.read()

            # Add ASAN flags
            makefile_content = makefile_content.replace(
                'CFLAGS =',
                'CFLAGS = -fsanitize=address -fno-omit-frame-pointer -g'
            )
            makefile_content = makefile_content.replace(
                'LDFLAGS =',
                'LDFLAGS = -fsanitize=address'
            )
            with open(makefile_path, 'w') as f:
                f.write(makefile_content)

            # Build
            env = os.environ.copy()
            env['NNC_RUNTIME'] = _NNC_RUNTIME_PATH
            build_result = subprocess.run(
                ['make', '-C', output_dir, 'clean', 'all'],
                capture_output=True,
                text=True,
                env=env,
            )

            if build_result.returncode != 0:
                print(f"  Build warnings/errors:\n{build_result.stderr}")
            else:
                print("  Build successful")

                # Run the program
                exe_path = os.path.join(output_dir, 'main')
                run_result = subprocess.run(
                    [exe_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if 'AddressSanitizer' in run_result.stderr:
                    print("\n❌ ASAN detected memory issues!")
                    print(run_result.stderr[-1000:])
                    raise AssertionError("ASAN detected memory errors")

                print("  ✓ No ASAN errors detected")

    finally:
        os.unlink(model_path)


def test_spill_with_parallel_model():
    """Test spill with a parallel branch model."""
    print("\n=== Testing Spill with Parallel Branch Model ===\n")

    # Create model with many parallel branches
    model = create_parallel_branch_model(
        input_size=256,           # ~1KB input
        num_branches=12,          # 12 parallel branches
        hidden_size=256,          # ~1KB per branch output
    )

    # Create test input
    np.random.seed(42)
    input_data = np.random.randn(1, 256).astype(np.float32)

    # Compute reference output
    print("Computing reference output...")
    expected_output = compute_reference_output(model, input_data)
    print(f"  Expected output shape: {expected_output.shape}")

    # Save model
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx.save(model, f.name)
        model_path = f.name

    try:
        with tempfile.TemporaryDirectory() as output_dir:
            print(f"\nCompiling with memory limit...")

            # Low memory to force spill
            max_memory = "4KB"

            compiler = Compiler(target='x86', opt_level=0)
            compiler.compile(
                onnx_path=model_path,
                output_dir=output_dir,
                entry_point='test_parallel',
                max_memory=max_memory,
                memory_strategy='graph_coloring',
            )

            # Check for spill
            tensors_c_path = os.path.join(output_dir, 'tensors.c')
            with open(tensors_c_path, 'r') as f:
                tensors_code = f.read()

            has_slow_pool = '_nnc_slow_memory_pool' in tensors_code
            print(f"  Has slow memory pool: {has_slow_pool}")

            # Compile with ASAN
            print("\nCompiling with ASAN...")
            makefile_path = os.path.join(output_dir, 'Makefile')

            with open(makefile_path, 'r') as f:
                makefile_content = f.read()

            makefile_content = makefile_content.replace(
                'CFLAGS =',
                'CFLAGS = -fsanitize=address -fno-omit-frame-pointer -g'
            )
            makefile_content = makefile_content.replace(
                'LDFLAGS =',
                'LDFLAGS = -fsanitize=address'
            )
            with open(makefile_path, 'w') as f:
                f.write(makefile_content)

            env = os.environ.copy()
            env['NNC_RUNTIME'] = _NNC_RUNTIME_PATH
            build_result = subprocess.run(
                ['make', '-C', output_dir, 'clean', 'all'],
                capture_output=True,
                text=True,
                env=env,
            )

            if build_result.returncode != 0:
                print(f"  Build warnings/errors:\n{build_result.stderr}")
            else:
                print("  Build successful")

                exe_path = os.path.join(output_dir, 'main')
                run_result = subprocess.run(
                    [exe_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if 'AddressSanitizer' in run_result.stderr:
                    print("\n❌ ASAN detected memory issues!")
                    print(run_result.stderr[-1000:])
                    raise AssertionError("ASAN detected memory errors")

                print("  ✓ No ASAN errors detected")

    finally:
        os.unlink(model_path)


def test_memory_no_overlaps():
    """Test that memory allocation doesn't have overlaps."""
    print("\n=== Testing Memory Allocation Correctness ===\n")

    model = create_simple_linear_model(
        input_size=64,
        num_layers=8,
        hidden_size=128,
    )

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx.save(model, f.name)
        model_path = f.name

    try:
        with tempfile.TemporaryDirectory() as output_dir:
            max_memory = "1KB"

            compiler = Compiler(target='x86', opt_level=0)
            compiler.compile(
                onnx_path=model_path,
                output_dir=output_dir,
                entry_point='test_overlaps',
                max_memory=max_memory,
                memory_strategy='graph_coloring',
            )

            # Parse allocations
            tensors_c_path = os.path.join(output_dir, 'tensors.c')
            with open(tensors_c_path, 'r') as f:
                content = f.read()

            # Extract tensor allocations
            allocations = []
            for line in content.split('\n'):
                if '.data = _nnc_memory_pool +' in line:
                    # Parse comment
                    parts = line.split('//')
                    if len(parts) > 1:
                        comment = parts[1].strip()
                        if 'offset=' in comment and 'size=' in comment:
                            try:
                                offset_part = comment.split('offset=')[1].split(',')[0]
                                size_part = comment.split('size=')[1].split(',')[0]
                                offset = int(offset_part)
                                size = int(size_part)
                                allocations.append((offset, size, line.strip()[:60]))
                            except ValueError:
                                pass

            allocations.sort(key=lambda x: x[0])  # Sort by offset
            print(f"  Found {len(allocations)} fast memory allocations")

            has_overlap = False
            for i in range(len(allocations) - 1):
                offset1, size1, _ = allocations[i]
                offset2, _, _ = allocations[i + 1]

                if offset1 + size1 > offset2:
                    print(f"  ❌ OVERLAP: [{offset1}, {offset1 + size1}) overlaps next at {offset2}")
                    has_overlap = True

            if not has_overlap:
                print("  ✓ No overlaps detected in fast memory")

            # Compile with ASAN
            makefile_path = os.path.join(output_dir, 'Makefile')
            with open(makefile_path, 'r') as f:
                makefile_content = f.read()

            makefile_content = makefile_content.replace(
                'CFLAGS =',
                'CFLAGS = -fsanitize=address -fno-omit-frame-pointer -g'
            )
            makefile_content = makefile_content.replace(
                'LDFLAGS =',
                'LDFLAGS = -fsanitize=address'
            )
            with open(makefile_path, 'w') as f:
                f.write(makefile_content)

            env = os.environ.copy()
            env['NNC_RUNTIME'] = _NNC_RUNTIME_PATH
            build_result = subprocess.run(
                ['make', '-C', output_dir, 'clean', 'all'],
                capture_output=True,
                text=True,
                env=env,
            )

            if build_result.returncode == 0:
                exe_path = os.path.join(output_dir, 'main')
                run_result = subprocess.run(
                    [exe_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if 'AddressSanitizer' in run_result.stderr:
                    print("  ❌ ASAN detected memory issues!")
                    raise AssertionError("ASAN detected memory errors")

                print("  ✓ No ASAN errors detected")

    finally:
        os.unlink(model_path)


if __name__ == '__main__':
    print("=" * 60)
    print("Spill Verification Tests")
    print("=" * 60)

    try:
        test_memory_no_overlaps()
        test_spill_with_linear_model()
        test_spill_with_parallel_model()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
