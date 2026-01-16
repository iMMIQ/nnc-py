"""Comprehensive tests to verify memory limit enforcement across all strategies.

This test file attempts to trigger memory overflow bugs by:
1. Creating models with complex memory usage patterns
2. Testing each strategy with tight memory limits
3. Verifying that generated code actually respects the limits
"""

import os
import re
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

import pytest

from nnc_py import Compiler


def extract_memory_size(source_file, define_name):
    """Extract memory size from #define in C source."""
    with open(source_file, 'r') as f:
        content = f.read()

    pattern = rf'#define\s+{define_name}\s+(\d+)'
    match = re.search(pattern, content)
    if match:
        return int(match.group(1))
    return None


def create_chained_model(num_nodes=50):
    """Create a model with a long chain of operations.
    Each node produces a tensor that lives until the end.
    This creates high peak memory but each tensor is small.
    """
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [10, 10])

    nodes = []
    current = 'input'
    for i in range(num_nodes):
        next_name = f'tensor_{i}'
        nodes.append(helper.make_node('Relu', inputs=[current], outputs=[next_name]))
        current = next_name

    output_tensor = helper.make_tensor_value_info(current, onnx.TensorProto.FLOAT, [10, 10])

    graph = helper.make_graph(nodes, 'chained_model', [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

    return model


def create_fanout_fanin_model(fanout=10, fanin=10):
    """Create a model with fanout then fanin pattern.
    This creates many intermediate tensors that must coexist.
    """
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [15, 15])

    nodes = []
    # Fanout phase
    for i in range(fanout):
        nodes.append(helper.make_node('Relu', inputs=['input'], outputs=[f'branch_{i}']))

    # Fanin phase - each node uses two branches
    for i in range(fanin):
        b1 = i % fanout
        b2 = (i + 1) % fanout
        nodes.append(helper.make_node('Add', inputs=[f'branch_{b1}', f'branch_{b2}'], outputs=[f'merge_{i}']))

    # Final merge
    merge_inputs = [f'merge_{i}' for i in range(min(5, fanin))]
    for i in range(len(merge_inputs) - 1):
        nodes.append(helper.make_node('Add', inputs=[merge_inputs[i], merge_inputs[i+1]], outputs=[f'final_{i}']))

    output_tensor = helper.make_tensor_value_info(f'final_{len(merge_inputs)-2}', onnx.TensorProto.FLOAT, [15, 15])

    graph = helper.make_graph(nodes, 'fanout_model', [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

    return model


def create_diamond_pattern_model(depth=5):
    """Create a diamond pattern model.
    Each diamond has two parallel branches that merge back.
    """
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [12, 12])

    nodes = []
    left = 'input'
    right = 'input'

    for i in range(depth):
        next_left = f'left_{i}'
        next_right = f'right_{i}'
        merge = f'merge_{i}'

        nodes.append(helper.make_node('Relu', inputs=[left], outputs=[next_left]))
        nodes.append(helper.make_node('Sigmoid', inputs=[right], outputs=[next_right]))
        nodes.append(helper.make_node('Add', inputs=[next_left, next_right], outputs=[merge]))

        left = merge
        right = merge

    output_tensor = helper.make_tensor_value_info(left, onnx.TensorProto.FLOAT, [12, 12])

    graph = helper.make_graph(nodes, 'diamond_model', [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

    return model


class TestMemoryLimitEnforcement:
    """Test that all strategies properly enforce memory limits."""

    @pytest.mark.parametrize("strategy", ["liveness", "unified", "graph_coloring"])
    def test_chained_model_with_limit(self, strategy):
        """Test chained model with tight memory limit."""
        model = create_chained_model(num_nodes=30)
        tmpdir = tempfile.mkdtemp()
        onnx_path = os.path.join(tmpdir, 'model.onnx')
        output_dir = os.path.join(tmpdir, 'build')

        onnx.save(model, onnx_path)

        # Set limit to 3KB - model needs more due to chain
        compiler = Compiler(target='x86', opt_level=2)
        compiler.compile(
            onnx_path,
            output_dir,
            max_memory='3KB',
            memory_strategy=strategy
        )

        # Verify generated code respects limit
        tensors_c = Path(output_dir) / 'tensors.c'
        fast_size = extract_memory_size(tensors_c, 'NNC_FAST_MEMORY_SIZE')

        assert fast_size is not None, f"[{strategy}] Could not find NNC_FAST_MEMORY_SIZE"
        assert fast_size <= 3072, f"[{strategy}] Fast memory ({fast_size}) exceeds limit (3072)"

    @pytest.mark.parametrize("strategy", ["liveness", "unified", "graph_coloring"])
    def test_fanout_model_with_limit(self, strategy):
        """Test fanout-fanin model with tight memory limit."""
        model = create_fanout_fanin_model(fanout=8, fanin=8)
        tmpdir = tempfile.mkdtemp()
        onnx_path = os.path.join(tmpdir, 'model.onnx')
        output_dir = os.path.join(tmpdir, 'build')

        onnx.save(model, onnx_path)

        # Each tensor is 15*15*4 = 900 bytes, 8 branches = 7200 bytes
        # Set limit to 4KB to force spill
        compiler = Compiler(target='x86', opt_level=2)
        compiler.compile(
            onnx_path,
            output_dir,
            max_memory='4KB',
            memory_strategy=strategy
        )

        tensors_c = Path(output_dir) / 'tensors.c'
        fast_size = extract_memory_size(tensors_c, 'NNC_FAST_MEMORY_SIZE')

        assert fast_size is not None, f"[{strategy}] Could not find NNC_FAST_MEMORY_SIZE"
        assert fast_size <= 4096, f"[{strategy}] Fast memory ({fast_size}) exceeds limit (4096)"

    @pytest.mark.parametrize("strategy", ["liveness", "unified", "graph_coloring"])
    def test_diamond_model_with_limit(self, strategy):
        """Test diamond pattern model with tight memory limit."""
        model = create_diamond_pattern_model(depth=5)
        tmpdir = tempfile.mkdtemp()
        onnx_path = os.path.join(tmpdir, 'model.onnx')
        output_dir = os.path.join(tmpdir, 'build')

        onnx.save(model, onnx_path)

        # Set limit to 2KB
        compiler = Compiler(target='x86', opt_level=2)
        compiler.compile(
            onnx_path,
            output_dir,
            max_memory='2KB',
            memory_strategy=strategy
        )

        tensors_c = Path(output_dir) / 'tensors.c'
        fast_size = extract_memory_size(tensors_c, 'NNC_FAST_MEMORY_SIZE')

        assert fast_size is not None, f"[{strategy}] Could not find NNC_FAST_MEMORY_SIZE"
        assert fast_size <= 2048, f"[{strategy}] Fast memory ({fast_size}) exceeds limit (2048)"

    def test_strategies_generate_different_memory(self):
        """Test that different strategies may generate different memory usage."""
        model = create_fanout_fanin_model(fanout=6, fanin=6)

        results = {}
        for strategy in ["liveness", "unified", "graph_coloring"]:
            tmpdir = tempfile.mkdtemp()
            onnx_path = os.path.join(tmpdir, 'model.onnx')
            output_dir = os.path.join(tmpdir, 'build')

            onnx.save(model, onnx_path)

            compiler = Compiler(target='x86', opt_level=2)
            compiler.compile(
                onnx_path,
                output_dir,
                max_memory='4KB',
                memory_strategy=strategy
            )

            tensors_c = Path(output_dir) / 'tensors.c'
            fast_size = extract_memory_size(tensors_c, 'NNC_FAST_MEMORY_SIZE')
            results[strategy] = fast_size

        # All should respect the limit
        for strategy, size in results.items():
            assert size is not None, f"{strategy}: No fast memory size found"
            assert size <= 4096, f"{strategy}: Size {size} exceeds limit 4096"

        # Print results for comparison
        print("\nMemory usage by strategy:")
        for strategy, size in results.items():
            print(f"  {strategy}: {size} bytes")


def run_all_enforcement_tests():
    """Run all memory limit enforcement tests."""
    print("\n" + "=" * 70)
    print("MEMORY LIMIT ENFORCEMENT TEST SUITE")
    print("=" * 70)

    test_class = TestMemoryLimitEnforcement()
    all_passed = True

    # Run parametrized tests
    for strategy in ["liveness", "unified", "graph_coloring"]:
        try:
            test_class.test_chained_model_with_limit(strategy)
            print(f"✅ Chained model [{strategy}] PASSED")
        except AssertionError as e:
            print(f"❌ Chained model [{strategy}] FAILED: {e}")
            all_passed = False
        except Exception as e:
            print(f"❌ Chained model [{strategy}] ERROR: {e}")
            all_passed = False

        try:
            test_class.test_fanout_model_with_limit(strategy)
            print(f"✅ Fanout model [{strategy}] PASSED")
        except AssertionError as e:
            print(f"❌ Fanout model [{strategy}] FAILED: {e}")
            all_passed = False
        except Exception as e:
            print(f"❌ Fanout model [{strategy}] ERROR: {e}")
            all_passed = False

        try:
            test_class.test_diamond_model_with_limit(strategy)
            print(f"✅ Diamond model [{strategy}] PASSED")
        except AssertionError as e:
            print(f"❌ Diamond model [{strategy}] FAILED: {e}")
            all_passed = False
        except Exception as e:
            print(f"❌ Diamond model [{strategy}] ERROR: {e}")
            all_passed = False

    # Test strategy comparison
    try:
        test_class.test_strategies_generate_different_memory()
        print(f"✅ Strategy comparison PASSED")
    except AssertionError as e:
        print(f"❌ Strategy comparison FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"❌ Strategy comparison ERROR: {e}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL ENFORCEMENT TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    import sys
    success = run_all_enforcement_tests()
    sys.exit(0 if success else 1)
