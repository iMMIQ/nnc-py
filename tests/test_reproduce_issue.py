"""Test to reproduce test_unified_strategy_no_asan_errors issue."""

import os
import subprocess
import tempfile
from pathlib import Path

import onnx
from onnx import helper

from nnc_py import Compiler


def create_simple_model_for_opt2():
    """Create a simple model for opt_level=2 compilation."""
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [20, 20])
    output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [20, 20])
    const_tensor = helper.make_tensor('const_one', onnx.TensorProto.FLOAT, [1, 1], [1.0])

    nodes = []
    prev = 'input'
    for i in range(5):
        nodes.append(helper.make_node('Relu', [prev], [f'relu{i}'], name=f'Relu_{i}'))
        prev = f'relu{i}'
    for i in range(3):
        nodes.append(helper.make_node('Add', [prev], [f'add{i}'], name=f'Add_{i}'))
        prev = f'add{i}'
    nodes.append(helper.make_node('Relu', [prev], ['output'], name='Final_Relu'))

    graph = helper.make_graph(nodes, 'chain_model', [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 14)])
    model.graph.initializer.append(const_tensor)
    return model


def create_test_model():
    """Create the model used in test_unified_strategy_no_asan_errors."""
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [16, 16])
    nodes = [
        helper.make_node('Relu', inputs=['input'], outputs=['relu1']),
        helper.make_node('Relu', inputs=['input'], outputs=['relu2']),
        helper.make_node('Relu', inputs=['input'], outputs=['relu3']),
        helper.make_node('Relu', inputs=['input'], outputs=['relu4']),
        helper.make_node('Add', inputs=['relu1', 'relu2'], outputs=['add1']),
        helper.make_node('Add', inputs=['relu2', 'relu3'], outputs=['add2']),
        helper.make_node('Add', inputs=['relu3', 'relu4'], outputs=['add3']),
        helper.make_node('Add', inputs=['relu4', 'relu1'], outputs=['add4']),
        helper.make_node('Add', inputs=['add1', 'add2'], outputs=['final1']),
        helper.make_node('Add', inputs=['add3', 'add4'], outputs=['final2']),
        helper.make_node('Add', inputs=['final1', 'final2'], outputs=['output']),
    ]
    output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [16, 16])
    graph = helper.make_graph(nodes, 'test_model', [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    return model


def build_with_asan(output_dir, runtime_dir):
    """Build the compiled model with AddressSanitizer enabled."""
    makefile = Path(output_dir) / "Makefile"
    with open(makefile, 'r') as f:
        makefile_content = f.read()
    makefile_content = makefile_content.replace(
        "NNC_RUNTIME ?= ../../runtime",
        f"NNC_RUNTIME = {runtime_dir}"
    )
    makefile_content = makefile_content.replace(
        "CFLAGS = -std=c11 -O2",
        "-std=c11 -O0 -g -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer"
    )
    with open(makefile, 'w') as f:
        f.write(makefile_content)
    subprocess.run(['make', 'clean'], cwd=output_dir, capture_output=True)
    result = subprocess.run(['make'], cwd=output_dir, capture_output=True, text=True, timeout=60)
    return result.returncode == 0, result


def run_with_asan(exe_path):
    """Run executable with ASan and check for errors."""
    env = os.environ.copy()
    env['ASAN_OPTIONS'] = 'detect_leaks=1:halt_on_error=0'
    result = subprocess.run([str(exe_path)], cwd=exe_path.parent, capture_output=True, text=True, timeout=30, env=env)
    asan_errors = []
    combined_output = result.stdout + result.stderr
    if 'ERROR: AddressSanitizer' in combined_output:
        asan_errors.append('AddressSanitizer detected memory error')
    return result.returncode == 0, asan_errors


def test_run_opt2_three_times_then_test_model():
    """Run opt_level=2 compile 3 times, then compile test model with opt_level=0."""
    # First run opt_level=2 compiles
    for i in range(3):
        model = create_simple_model_for_opt2()
        tmpdir = tempfile.mkdtemp()
        onnx_path = os.path.join(tmpdir, 'model.onnx')
        onnx.save(model, onnx_path)
        compiler = Compiler(target='x86', opt_level=2)
        compiler.compile(onnx_path, os.path.join(tmpdir, 'build'), max_memory='2048')

    # Then compile test model
    model = create_test_model()
    tmpdir = tempfile.mkdtemp()
    onnx_path = os.path.join(tmpdir, 'model.onnx')
    output_dir = os.path.join(tmpdir, 'build')
    onnx.save(model, onnx_path)

    compiler = Compiler(target='x86', opt_level=0)
    compiler.compile(onnx_path, output_dir, max_memory='3KB', memory_strategy='basic')

    # Check generated files
    model_c = Path(output_dir) / 'model.c'
    tensors_c = Path(output_dir) / 'tensors.c'

    print(f"model.c size: {model_c.stat().st_size} bytes")
    print(f"tensors.c size: {tensors_c.stat().st_size} bytes")
    print("\ntensors.c (first 50 lines):")
    for i, line in enumerate(tensors_c.open().readlines()[:50]):
        print(line.rstrip())

    # Build with ASan
    runtime_dir = Path(__file__).parent.parent / "runtime"
    success, build_result = build_with_asan(output_dir, runtime_dir)
    assert success, f"Build failed: {build_result.stderr}"

    # Run with ASan
    exe_path = Path(output_dir) / "model"
    success, asan_errors = run_with_asan(exe_path)

    if asan_errors:
        print(f"\nASan detected errors: {asan_errors}")
        # Print output
        env = os.environ.copy()
        env['ASAN_OPTIONS'] = 'detect_leaks=1:halt_on_error=0'
        result = subprocess.run([str(exe_path)], cwd=exe_path.parent, capture_output=True, text=True, timeout=30, env=env)
        print(f"Output: {result.stderr[:500]}")
        assert False, f"ASan detected errors: {asan_errors}"
