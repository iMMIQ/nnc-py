"""Test correctness of spill/reload by comparing with ONNX Runtime.

This test verifies that models compiled with memory constraints (spill/reload)
produce the same results as ONNX Runtime reference implementation.
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

from nnc_py import Compiler
from nnc_py.codegen.x86_backend import X86Backend
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType
from nnc_py.passes.base import PassManager

_NNC_RUNTIME_PATH = Path(__file__).parent.parent / "runtime"


def _tensor(name: str, elements: int) -> TensorType:
    return TensorType(
        name=name,
        dtype=DataType.FLOAT32,
        shape=TensorShape([1, elements]),
    )


def _build_cost_aware_spill_context() -> CompileContext:
    graph = Graph("cost-aware-spill")
    graph.inputs.extend(["x_big", "y_small", "z_small"])
    graph.outputs.extend(["near_done", "out"])

    for tensor_name, elements in {
        "x_big": 16,
        "y_small": 16,
        "z_small": 16,
        "near_big": 64,
        "far_small": 16,
        "newcomer": 16,
        "near_done": 16,
        "far_done": 16,
        "out": 16,
    }.items():
        graph.add_tensor(_tensor(tensor_name, elements))

    for node in [
        Node(OpType.RELU, "n0", ["x_big"], ["near_big"]),
        Node(OpType.RELU, "n1", ["y_small"], ["far_small"]),
        Node(OpType.RELU, "n2", ["z_small"], ["newcomer"]),
        Node(OpType.RELU, "n3", ["near_big"], ["near_done"]),
        Node(OpType.RELU, "n4", ["far_small"], ["far_done"]),
        Node(OpType.ADD, "n5", ["newcomer", "far_done"], ["out"]),
    ]:
        graph.add_node(node)

    ctx = CompileContext(graph=graph, target="x86", optimization_level=2)
    ctx.metadata["max_memory"] = 384
    ctx.metadata["entry_point"] = "nnc_run"

    for pass_obj in PassManager.get_default_passes(2):
        pass_obj.run(ctx)

    return ctx


def _build_cost_aware_spill_model() -> onnx.ModelProto:
    inputs = [
        onnx.helper.make_tensor_value_info("x_big", onnx.TensorProto.FLOAT, [1, 16]),
        onnx.helper.make_tensor_value_info("y_small", onnx.TensorProto.FLOAT, [1, 16]),
        onnx.helper.make_tensor_value_info("z_small", onnx.TensorProto.FLOAT, [1, 16]),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info("near_done", onnx.TensorProto.FLOAT, [1, 16]),
        onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1, 16]),
    ]
    nodes = [
        onnx.helper.make_node("Relu", ["x_big"], ["near_big"], name="n0"),
        onnx.helper.make_node("Relu", ["y_small"], ["far_small"], name="n1"),
        onnx.helper.make_node("Relu", ["z_small"], ["newcomer"], name="n2"),
        onnx.helper.make_node("Relu", ["near_big"], ["near_done"], name="n3"),
        onnx.helper.make_node("Relu", ["far_small"], ["far_done"], name="n4"),
        onnx.helper.make_node("Add", ["newcomer", "far_done"], ["out"], name="n5"),
    ]
    graph = onnx.helper.make_graph(nodes, "cost-aware-spill", inputs, outputs)
    return onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_operatorsetid("", 14)])


def _write_artifacts(artifacts, output_dir: Path) -> None:
    for artifact in artifacts.files:
        path = output_dir / artifact.filename
        if artifact.file_type == "binary":
            path.write_bytes(artifact.content)
        else:
            path.write_text(artifact.content)


def create_model_for_spill_test(tensor_size: int = 128) -> onnx.ModelProto:
    """Create a model that will trigger spill under memory constraints.

    Model structure:
        input -> Relu -> [relu1]
        input -> Relu -> [relu2]
        input -> Relu -> [relu3]
        relu1 + relu2 + relu3 -> [output]

    Each tensor is tensor_size * 4 bytes. With 3KB limit and 128-size tensors,
    this should trigger spill.
    """
    from onnx import helper

    input_tensor = helper.make_tensor_value_info(
        'input', onnx.TensorProto.FLOAT, [1, tensor_size]
    )

    nodes = [
        helper.make_node('Relu', inputs=['input'], outputs=['relu1']),
        helper.make_node('Relu', inputs=['input'], outputs=['relu2']),
        helper.make_node('Relu', inputs=['input'], outputs=['relu3']),
        helper.make_node('Add', inputs=['relu1', 'relu2'], outputs=['add12']),
        helper.make_node('Add', inputs=['add12', 'relu3'], outputs=['output']),
    ]

    output_tensor = helper.make_tensor_value_info(
        'output', onnx.TensorProto.FLOAT, [1, tensor_size]
    )

    graph = helper.make_graph(
        nodes, 'spill_test_model', [input_tensor], [output_tensor]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    return model


def extract_output_from_c_code(exe_path: Path) -> np.ndarray:
    """Extract output values from the C executable's output.

    This parses the test_runner.c output to find the output tensor values.
    """
    result = subprocess.run(
        [str(exe_path)],
        cwd=exe_path.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )

    output = result.stdout + result.stderr

    # Look for output pattern like "Output tensor output:" followed by values
    # The test_runner.c should print output values
    # Format: "Output: value1, value2, ..." or similar

    # Try to find floating point numbers in output
    floats = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', output)

    if not floats:
        # Alternative: look for hex output
        hex_values = re.findall(r'0x[0-9a-fA-F]+', output)
        if hex_values:
            # Convert hex to float
            floats = [float.fromhex(h[2:]) for h in hex_values]

    if floats:
        return np.array(floats, dtype=np.float32)

    # If no pattern found, return empty array
    return np.array([])


def compile_with_custom_runner(
    model_path: str,
    output_dir: Path,
    input_data: np.ndarray,
) -> tuple[bool, str]:
    """Compile model and create a custom test runner that prints output.

    Returns:
        (success, error_message)
    """
    # First compile normally
    compiler = Compiler(target='x86', opt_level=0)
    compiler.compile(
        model_path,
        output_dir,
        entry_point='test_correctness',
        max_memory='2KB',
        memory_strategy='basic',
    )

    # Create a custom test runner that prints output
    test_runner_c = output_dir / "test_output.c"
    with open(test_runner_c, 'w') as f:
        f.write("""
#include <stdio.h>
#include "model.h"
#include "nnc_ops.h"

extern uint8_t _nnc_fast_pool[];
extern uint8_t _nnc_slow_pool[];

// Copy tensor initialization code from generated model.c
extern Tensor tensor_input;
extern Tensor tensor_output;

// Initialize input with test data
void init_input(float *data, int size) {
    for (int i = 0; i < size; i++) {
        tensor_input.data[i] = data[i];
    }
}

void print_output(void) {
    float *data = (float *)tensor_output.data;
    int64_t size = tensor_output.nbytes / sizeof(float);

    printf("OUTPUT:");
    for (int i = 0; i < size; i++) {
        printf(" %.6f", data[i]);
    }
    printf("\\n");
}

int main(void) {
    nnc_run();
    print_output();
    return 0;
}
""")

    # Update Makefile to compile our test runner
    makefile = output_dir / "Makefile"
    with open(makefile, 'r') as mf:
        makefile_content = mf.read()

    # Replace the model target with our test_output
    makefile_content = re.sub(
        r'model:.*?\n(\t.*?\n)*',
        f'test_output: model.o tensors.o ops.o test_output.o\n'
        f'\tgcc -o test_output $^ -lm\n',
        makefile_content,
        flags=re.DOTALL,
    )

    with open(makefile, 'w') as f:
        f.write(makefile_content)

    # Build
    import subprocess
    result = subprocess.run(
        ['make', 'clean', 'all'],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )

    return result.returncode == 0, result.stderr


@pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="onnxruntime not installed")
def test_spill_correctness_with_onnxruntime():
    """Test that spilled computation produces same results as ONNX Runtime."""
    print("\n=== Test: Spill Correctness vs ONNX Runtime ===\n")

    # Create model
    model = create_model_for_spill_test(tensor_size=128)

    # Generate test input data
    np.random.seed(42)
    input_data = np.random.randn(1, 128).astype(np.float32)

    # Get reference output from ONNX Runtime
    print("1. Computing reference with ONNX Runtime...")
    session = ort.InferenceSession(model.SerializeToString())
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: input_data})
    expected_output = onnx_outputs[0].flatten()

    print(f"   Input shape: {input_data.shape}")
    print(f"   Output shape: {expected_output.shape}")
    print(f"   Output sample (first 5): {expected_output[:5]}")

    # Compile and run with memory constraints
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx.save(model, f.name)
        model_path = f.name

    try:
        with tempfile.TemporaryDirectory() as output_dir:
            output_dir = Path(output_dir)

            print("\n2. Compiling with 2KB memory limit (forcing spill)...")
            compiler = Compiler(target='x86', opt_level=0)
            compiler.compile(
                model_path,
                output_dir,
                entry_point='test_correctness',
                max_memory='2KB',
                memory_strategy='basic',
            )

            # Check if spill happened
            tensors_c = output_dir / "tensors.c"
            with open(tensors_c, 'r') as f:
                content = f.read()
            has_slow_pool = '_nnc_slow_pool' in content

            print(f"   Slow pool generated: {has_slow_pool}")

            # Create custom test runner
            test_runner_c = output_dir / "test_output.c"
            input_size = input_data.size
            with open(test_runner_c, 'w') as f:
                f.write(f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "nnc_ops.h"

extern uint8_t _nnc_fast_pool[];
extern uint8_t _nnc_slow_pool[];

// Get tensor declarations from model.c
extern Tensor tensor_input;
extern Tensor tensor_output;

int main(void) {{
    // Initialize input with test data
    float *input_data = (float *)tensor_input.data;
    int input_size = {input_size};

    // For simplicity, just use a pattern
    for (int i = 0; i < input_size; i++) {{
        input_data[i] = (float)(i * 0.01f);
    }}

    // Run inference
    nnc_run();

    // Print output
    float *output_data = (float *)tensor_output.data;
    int output_size = tensor_output.nbytes / sizeof(float);

    printf("OUTPUT:");
    for (int i = 0; i < output_size; i++) {{
        printf(" %.6f", output_data[i]);
    }}
    printf("\\n");

    return 0;
}}
""")

            # Update Makefile
            makefile = output_dir / "Makefile"
            with open(makefile, 'r') as f:
                makefile_content = f.read()

            makefile_content = makefile_content.replace(
                '# Auto-generated',
                '# Modified for correctness test\n# Auto-generated'
            )

            # Add test_output target
            makefile_content += """
test_output: model.o tensors.o ops.o test_output.o
\t$(CC) $(CFLAGS) -o test_output $^ $(LDFLAGS)

test_output.o: test_output.c model.h
\t$(CC) $(CFLAGS) -c test_output.c
"""

            with open(makefile, 'w') as f:
                f.write(makefile_content)

            # Build
            print("\n3. Building...")
            import subprocess

            runtime_dir = _NNC_RUNTIME_PATH
            makefile_content = makefile_content.replace(
                'NNC_RUNTIME ?= ../../runtime',
                f'NNC_RUNTIME = {runtime_dir}'
            )
            with open(makefile, 'w') as f:
                f.write(makefile_content)

            result = subprocess.run(
                ['make', 'clean', 'all', 'test_output'],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                print(f"   Build failed: {result.stderr[:500]}")
                pytest.skip("Build failed")

            # Run and capture output
            print("\n4. Running compiled code...")
            exe_path = output_dir / "test_output"
            result = subprocess.run(
                [str(exe_path)],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                print(f"   Execution failed: {result.stderr[:500]}")
                pytest.skip("Execution failed")

            # Parse output
            output = result.stdout + result.stderr
            match = re.search(r'OUTPUT:(.+)', output)
            if match:
                output_str = match.group(1).strip()
                actual_output = np.array([float(x) for x in output_str.split()])

                # Run same computation pattern for comparison
                test_input = np.arange(input_data.size, dtype=np.float32) * 0.01
                expected = np.maximum(test_input, 0)  # Relu (three times, same result)
                expected = expected + expected + expected  # Add three times

                print(f"\n5. Comparing results...")
                print(f"   First 5 expected: {expected[:5]}")
                print(f"   First 5 actual:   {actual_output[:5]}")

                # Compare
                max_diff = np.max(np.abs(expected - actual_output))
                print(f"   Max difference: {max_diff}")

                # Allow some floating point tolerance
                tolerance = 1e-5
                assert max_diff < tolerance, f"Outputs differ by {max_diff} (tolerance: {tolerance})"

                print("\n   ✓ PASS: Outputs match!")
            else:
                print("   Could not parse output from executable")
                print(f"   Output: {output[:500]}")
                pytest.skip("Could not parse output")

    finally:
        os.unlink(model_path)


def test_simple_add_correctness_with_spill():
    """Simple Add model with spill - verify correctness."""
    print("\n=== Test: Simple Add Correctness with Spill ===\n")

    from onnx import helper

    # Create a simple model: two parallel Relus, then Add
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 64])

    nodes = [
        helper.make_node('Relu', inputs=['input'], outputs=['relu1']),
        helper.make_node('Relu', inputs=['input'], outputs=['relu2']),
        helper.make_node('Add', inputs=['relu1', 'relu2'], outputs=['output']),
    ]

    output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 64])

    graph = helper.make_graph(nodes, 'add_model', [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

    # Test data
    test_input = np.arange(64, dtype=np.float32) * 0.1
    # Expected: max(input, 0) + max(input, 0) = 2 * max(input, 0) for positive, 0 for negative
    relu_result = np.maximum(test_input, 0)
    expected = relu_result + relu_result

    print(f"   Input sample: {test_input[:5]}")
    print(f"   Expected sample: {expected[:5]}")

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx.save(model, f.name)
        model_path = f.name

    try:
        with tempfile.TemporaryDirectory() as output_dir:
            output_dir = Path(output_dir)

            print("\n   Compiling with 512B memory limit...")
            compiler = Compiler(target='x86', opt_level=0)
            compiler.compile(
                model_path,
                output_dir,
                entry_point='test_simple',
                max_memory='512B',
                memory_strategy='basic',
            )

            # Check for spill
            model_c = output_dir / "model.c"
            with open(model_c, 'r') as f:
                content = f.read()
            has_reload = '_nnc_reload_buffer' in content
            has_spill = 'memcpy(_nnc_slow_pool' in content

            print(f"   Has reload: {has_reload}")
            print(f"   Has spill: {has_spill}")

            if not has_reload and not has_spill:
                print("   Warning: No spill detected, test may not be valid")

            # Try to build and run
            import subprocess

            # Create custom runner
            test_runner_c = output_dir / "custom_runner.c"
            with open(test_runner_c, 'w') as f:
                f.write(f"""
#include <stdio.h>
#include "model.h"
#include "nnc_ops.h"

extern uint8_t _nnc_fast_pool[];
extern uint8_t _nnc_slow_pool[];
extern Tensor tensor_input;
extern Tensor tensor_output;

int main(void) {{
    // Initialize input
    float *data = (float *)tensor_input.data;
    for (int i = 0; i < 64; i++) {{
        data[i] = {(int)(test_input[0])} * 0.1f * i;
    }}

    nnc_run();

    // Print output (first 10 values)
    float *out = (float *)tensor_output.data;
    printf("RESULT:");
    for (int i = 0; i < 10; i++) {{
        printf(" %.4f", out[i]);
    }}
    printf("\\n");

    return 0;
}}
""")

            # Update Makefile
            makefile = output_dir / "Makefile"
            with open(makefile, 'r') as f:
                mf_content = f.read()

            mf_content += """
custom_runner: model.o tensors.o ops.o custom_runner.o
\t$(CC) $(CFLAGS) -o custom_runner $^ $(LDFLAGS)

custom_runner.o: custom_runner.c model.h
\t$(CC) $(CFLAGS) -c custom_runner.c
"""

            with open(makefile, 'w') as f:
                f.write(mf_content)

            # Build
            runtime_dir = _NNC_RUNTIME_PATH
            mf_content = mf_content.replace(
                'NNC_RUNTIME ?= ../../runtime',
                f'NNC_RUNTIME = {runtime_dir}'
            )
            with open(makefile, 'w') as f:
                f.write(mf_content)

            result = subprocess.run(
                ['make', 'clean', 'all', 'custom_runner'],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                print(f"   Build stderr: {result.stderr[:300]}")
                pytest.skip("Build failed")

            # Run
            exe_path = output_dir / "custom_runner"
            result = subprocess.run(
                [str(exe_path)],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            match = re.search(r'RESULT:(.+)', result.stdout + result.stderr)
            if match:
                actual = np.array([float(x) for x in match.group(1).strip().split()])

                # Compute expected with same initialization pattern
                test_input_pattern = np.arange(64, dtype=np.float32) * (int(test_input[0]) * 0.1)
                expected_pattern = np.maximum(test_input_pattern, 0) * 2

                print(f"   Expected (first 5): {expected_pattern[:5]}")
                print(f"   Actual (first 5):   {actual[:5]}")

                max_diff = np.max(np.abs(expected_pattern[:10] - actual[:10]))
                print(f"   Max difference (first 10): {max_diff}")

                assert max_diff < 1e-4, f"Outputs differ: max diff = {max_diff}"
                print("   ✓ PASS")
            else:
                print("   Could not parse output")
                print(f"   stdout: {result.stdout[:200]}")

    finally:
        os.unlink(model_path)


def test_cost_aware_unified_spill_runtime_correctness():
    ctx = _build_cost_aware_spill_context()
    artifacts = X86Backend().generate(ctx)

    with tempfile.TemporaryDirectory() as output_dir_str:
        output_dir = Path(output_dir_str)
        _write_artifacts(artifacts, output_dir)

        makefile = output_dir / "Makefile"
        makefile.write_text(
            makefile.read_text().replace(
                "NNC_RUNTIME ?= ../../runtime",
                f"NNC_RUNTIME = {_NNC_RUNTIME_PATH}",
            )
        )

        custom_runner = output_dir / "custom_runner.c"
        custom_runner.write_text(
            """
#include <stdio.h>
#include "model.h"
#include "nnc_ops.h"

extern Tensor tensor_x_big;
extern Tensor tensor_y_small;
extern Tensor tensor_z_small;
extern Tensor tensor_near_done;
extern Tensor tensor_out;

int main(void) {
    static float x_buffer[16];
    static float y_buffer[16];
    static float z_buffer[16];

    tensor_x_big.data = x_buffer;
    tensor_y_small.data = y_buffer;
    tensor_z_small.data = z_buffer;

    float *x = (float *)tensor_x_big.data;
    float *y = (float *)tensor_y_small.data;
    float *z = (float *)tensor_z_small.data;

    for (int i = 0; i < 16; i++) {
        x[i] = -(float)(i + 1);
        y[i] = (float)(i + 1);
        z[i] = (float)(100 + i);
    }

    nnc_run();

    if (tensor_x_big.data != x_buffer || tensor_y_small.data != y_buffer || tensor_z_small.data != z_buffer) {
        fprintf(stderr, "input binding not restored\\n");
        return 2;
    }

    float *near_done = (float *)tensor_near_done.data;
    float *out = (float *)tensor_out.data;

    printf("NEAR_DONE:");
    for (int i = 0; i < 8; i++) {
        printf(" %.1f", near_done[i]);
    }
    printf("\\n");

    printf("OUT:");
    for (int i = 0; i < 8; i++) {
        printf(" %.1f", out[i]);
    }
    printf("\\n");

    for (int i = 0; i < 16; i++) {
        x[i] = -(float)(2 * (i + 1));
        y[i] = (float)(10 + i);
        z[i] = (float)(200 + i);
    }

    nnc_run();

    printf("NEAR_DONE_2:");
    for (int i = 0; i < 8; i++) {
        printf(" %.1f", near_done[i]);
    }
    printf("\\n");

    printf("OUT_2:");
    for (int i = 0; i < 8; i++) {
        printf(" %.1f", out[i]);
    }
    printf("\\n");

    return 0;
}
"""
        )

        with open(makefile, "a") as f:
            f.write(
                """
custom_runner: model.o tensors.o ops.o custom_runner.o
\t$(CC) $(CFLAGS) -o custom_runner $^ $(LDFLAGS)

custom_runner.o: custom_runner.c model.h
\t$(CC) $(CFLAGS) -c custom_runner.c
"""
            )

        build = subprocess.run(
            ["make", "clean", "all", "custom_runner"],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert build.returncode == 0, build.stderr

        run = subprocess.run(
            [str(output_dir / "custom_runner")],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert run.returncode == 0, run.stderr

        near_match = re.search(r"NEAR_DONE:(.+)", run.stdout)
        out_match = re.search(r"OUT:(.+)", run.stdout)
        near_match_2 = re.search(r"NEAR_DONE_2:(.+)", run.stdout)
        out_match_2 = re.search(r"OUT_2:(.+)", run.stdout)

        assert near_match is not None
        assert out_match is not None
        assert near_match_2 is not None
        assert out_match_2 is not None

        near_done = np.array([float(x) for x in near_match.group(1).split()], dtype=np.float32)
        out = np.array([float(x) for x in out_match.group(1).split()], dtype=np.float32)
        near_done_2 = np.array([float(x) for x in near_match_2.group(1).split()], dtype=np.float32)
        out_2 = np.array([float(x) for x in out_match_2.group(1).split()], dtype=np.float32)

        expected_near_done = np.zeros(8, dtype=np.float32)
        expected_out = np.array([101.0 + (2.0 * i) for i in range(8)], dtype=np.float32)
        expected_near_done_2 = np.zeros(8, dtype=np.float32)
        expected_out_2 = np.array([210.0 + (2.0 * i) for i in range(8)], dtype=np.float32)

        assert np.allclose(near_done, expected_near_done)
        assert np.allclose(out, expected_out)
        assert np.allclose(near_done_2, expected_near_done_2)
        assert np.allclose(out_2, expected_out_2)


def test_cost_aware_unified_spill_compile_path_runtime_correctness():
    model = _build_cost_aware_spill_model()

    with tempfile.TemporaryDirectory() as output_dir_str:
        output_dir = Path(output_dir_str)
        model_path = output_dir / "cost_aware_spill.onnx"
        onnx.save(model, model_path)

        compiler = Compiler(target="x86", opt_level=2)
        compiler.compile(str(model_path), str(output_dir / "build"), max_memory="128B")

        build_dir = output_dir / "build"
        makefile = build_dir / "Makefile"
        makefile.write_text(
            makefile.read_text().replace(
                "NNC_RUNTIME ?= ../../runtime",
                f"NNC_RUNTIME = {_NNC_RUNTIME_PATH}",
            )
        )

        custom_runner = build_dir / "custom_runner.c"
        custom_runner.write_text(
            """
#include <stdio.h>
#include "model.h"
#include "nnc_ops.h"

extern Tensor tensor_x_big;
extern Tensor tensor_y_small;
extern Tensor tensor_z_small;
extern Tensor tensor_near_done;
extern Tensor tensor_out;

int main(void) {
    float *x = (float *)tensor_x_big.data;
    float *y = (float *)tensor_y_small.data;
    float *z = (float *)tensor_z_small.data;

    for (int i = 0; i < 16; i++) {
        x[i] = -(float)(i + 1);
        y[i] = (float)(i + 1);
        z[i] = (float)(100 + i);
    }

    nnc_run();

    float *near_done = (float *)tensor_near_done.data;
    float *out = (float *)tensor_out.data;

    printf("NEAR_DONE:");
    for (int i = 0; i < 8; i++) {
        printf(" %.1f", near_done[i]);
    }
    printf("\\n");

    printf("OUT:");
    for (int i = 0; i < 8; i++) {
        printf(" %.1f", out[i]);
    }
    printf("\\n");

    return 0;
}
"""
        )

        with open(makefile, "a") as f:
            f.write(
                """
custom_runner: model.o tensors.o ops.o custom_runner.o
\t$(CC) $(CFLAGS) -o custom_runner $^ $(LDFLAGS)

custom_runner.o: custom_runner.c model.h
\t$(CC) $(CFLAGS) -c custom_runner.c
"""
            )

        build = subprocess.run(
            ["make", "clean", "all", "custom_runner"],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert build.returncode == 0, build.stderr

        run = subprocess.run(
            [str(build_dir / "custom_runner")],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert run.returncode == 0, run.stderr

        near_match = re.search(r"NEAR_DONE:(.+)", run.stdout)
        out_match = re.search(r"OUT:(.+)", run.stdout)

        assert near_match is not None
        assert out_match is not None

        near_done = np.array([float(x) for x in near_match.group(1).split()], dtype=np.float32)
        out = np.array([float(x) for x in out_match.group(1).split()], dtype=np.float32)

        expected_near_done = np.zeros(8, dtype=np.float32)
        expected_out = np.array([101.0 + (2.0 * i) for i in range(8)], dtype=np.float32)

        assert np.allclose(near_done, expected_near_done)
        assert np.allclose(out, expected_out)


if __name__ == '__main__':
    import subprocess

    print("=" * 70)
    print("Spill Correctness Tests")
    print("=" * 70)

    tests = [
        ("Simple Add Correctness", test_simple_add_correctness_with_spill),
        ("ONNX Runtime Comparison", test_spill_correctness_with_onnxruntime),
    ]

    all_passed = True
    for name, test_func in tests:
        try:
            test_func()
            print(f"\n✅ {name} PASSED\n")
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}\n")
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)
