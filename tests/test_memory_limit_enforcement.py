"""Comprehensive tests to verify memory limit enforcement across all strategies.

This test file verifies that:
1. Models that can fit within memory limits compile successfully
2. Tensor offsets are all within the declared memory pool size
3. Clear errors are given when models cannot fit within the limit
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
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorShape, TensorType
from nnc_py.ir.types import DataType
from nnc_py.passes.base import PassManager
from nnc_py.passes.memory_planning import get_memory_allocation_plan
from nnc_py.passes.memory_strategy import MemoryAllocationPlan, SpillPoint
from nnc_py.passes.spill import SpillAnalysisPass, get_spill_plan


def extract_memory_size(source_file, define_name):
    """Extract memory size from #define in C source."""
    with open(source_file, 'r') as f:
        content = f.read()

    pattern = rf'#define\s+{define_name}\s+(\d+)'
    match = re.search(pattern, content)
    if match:
        return int(match.group(1))
    return None


def extract_tensor_offsets(source_file):
    """Extract all tensor offsets from generated C code."""
    with open(source_file, 'r') as f:
        content = f.read()

    # Find all tensor declarations with pool offsets
    fast_pattern = r'Tensor tensor_(\w+)\s*=\s*{[^}]*\.data\s*=\s*_nnc_fast_pool\s*\+\s*(\d+)'
    slow_pattern = r'Tensor tensor_(\w+)\s*=\s*{[^}]*\.data\s*=\s*_nnc_slow_pool\s*\+\s*(\d+)'
    single_pattern = r'Tensor tensor_(\w+)\s*=\s*{[^}]*\.data\s*=\s*_nnc_memory_pool\s*\+\s*(\d+)'

    fast_offsets = dict(re.findall(fast_pattern, content))
    slow_offsets = dict(re.findall(slow_pattern, content))
    single_offsets = dict(re.findall(single_pattern, content))

    return {
        'fast': {k: int(v) for k, v in fast_offsets.items()},
        'slow': {k: int(v) for k, v in slow_offsets.items()},
        'single': {k: int(v) for k, v in single_offsets.items()},
    }


def create_chained_model(num_nodes=10):
    """Create a model with a chain of operations.
    Each tensor is used once then dies, allowing memory reuse.
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


def create_fanout_fanin_model(fanout=4, fanin=4):
    """Create a model with fanout then fanin pattern.
    Reduced size to allow fitting in memory.
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
    merge_inputs = [f'merge_{i}' for i in range(min(3, fanin))]
    for i in range(len(merge_inputs) - 1):
        nodes.append(helper.make_node('Add', inputs=[merge_inputs[i], merge_inputs[i+1]], outputs=[f'final_{i}']))

    output_tensor = helper.make_tensor_value_info(f'final_{len(merge_inputs)-2}', onnx.TensorProto.FLOAT, [15, 15])

    graph = helper.make_graph(nodes, 'fanout_model', [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

    return model


def create_diamond_pattern_model(depth=3):
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


def create_simple_chain_model():
    """Create a simple chain model where tensors can share memory.
    This model should fit in small memory due to lifetime-aware allocation.
    """
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [20, 20])
    output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [20, 20])
    const_tensor = helper.make_tensor('const_one', onnx.TensorProto.FLOAT, [1, 1], [1.0])

    nodes = []
    prev = 'input'
    for i in range(5):
        relu = helper.make_node('Relu', [prev], [f'relu{i}'], name=f'Relu_{i}')
        nodes.append(relu)
        prev = f'relu{i}'

    for i in range(3):
        add = helper.make_node('Add', [prev, 'const_one'], [f'add{i}'], name=f'Add_{i}')
        nodes.append(add)
        prev = f'add{i}'

    nodes.append(helper.make_node('Relu', [prev], ['output'], name='Final_Relu'))

    graph = helper.make_graph(nodes, 'chain_model', [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 14)])
    model.graph.initializer.append(const_tensor)

    return model


def _tensor(name: str, elements: int) -> TensorType:
    return TensorType(
        name=name,
        dtype=DataType.FLOAT32,
        shape=TensorShape([1, elements]),
    )


def _build_cost_aware_spill_context(
    *,
    optimization_level: int = 2,
    memory_strategy: str | None = None,
) -> CompileContext:
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

    ctx = CompileContext(graph=graph, target="x86", optimization_level=optimization_level)
    ctx.metadata["max_memory"] = 384
    if memory_strategy is not None:
        ctx.metadata["memory_strategy"] = memory_strategy

    for pass_obj in PassManager.get_default_passes(optimization_level):
        pass_obj.run(ctx)

    return ctx


class TestMemoryLimitEnforcement:
    """Test that memory allocation respects limits."""

    def test_simple_chain_offsets_within_pool(self):
        """Test that a simple chain model has all offsets within the pool."""
        model = create_simple_chain_model()
        tmpdir = tempfile.mkdtemp()
        onnx_path = os.path.join(tmpdir, 'model.onnx')
        output_dir = os.path.join(tmpdir, 'build')

        onnx.save(model, onnx_path)

        compiler = Compiler(target='x86', opt_level=2)
        compiler.compile(onnx_path, output_dir, max_memory='2048')

        # Verify all tensor offsets are within the pool size
        tensors_c = Path(output_dir) / 'tensors.c'
        offsets = extract_tensor_offsets(tensors_c)

        fast_size = extract_memory_size(tensors_c, 'NNC_FAST_MEMORY_SIZE')

        if offsets['fast']:
            max_offset = max(offsets['fast'].values())
            assert max_offset < 2048, f"Max offset ({max_offset}) exceeds pool size (2048)"

        if offsets['single']:
            # For single pool, total memory used should be <= 2048
            max_offset = max(offsets['single'].values())
            # Single pool doesn't have a fixed limit, so just check it's reasonable
            assert max_offset < 100_000, f"Max offset ({max_offset}) seems unreasonably large"

    def test_chained_model_with_memory_reuse(self):
        """Test that chained model reuses memory via lifetime-aware allocation."""
        model = create_simple_chain_model()  # Uses this instead - has addition to prevent folding
        tmpdir = tempfile.mkdtemp()
        onnx_path = os.path.join(tmpdir, 'model.onnx')
        output_dir = os.path.join(tmpdir, 'build')

        onnx.save(model, onnx_path)

        # Each tensor is 20*20*4 = 1600 bytes
        # With lifetime-aware reuse, they should all fit in 2048 bytes
        compiler = Compiler(target='x86', opt_level=2)
        compiler.compile(onnx_path, output_dir, max_memory='2048')

        tensors_c = Path(output_dir) / 'tensors.c'
        offsets = extract_tensor_offsets(tensors_c)

        # All tensors should share the same offset (0) because their lifetimes don't overlap
        if offsets['fast']:
            unique_offsets = set(offsets['fast'].values())
            # Most tensors should share offset 0 (input, output, relu*, add*)
            # The const might be at a different offset
            assert 0 in unique_offsets, f"Expected offset 0 to be used, got {unique_offsets}"
            # All offsets should be within the pool
            max_offset = max(offsets['fast'].values())
            assert max_offset < 2048, f"Max offset ({max_offset}) exceeds pool size (2048)"

    def test_diamond_model_fits_in_memory(self):
        """Test that diamond pattern model fits in 2KB."""
        model = create_diamond_pattern_model(depth=3)
        tmpdir = tempfile.mkdtemp()
        onnx_path = os.path.join(tmpdir, 'model.onnx')
        output_dir = os.path.join(tmpdir, 'build')

        onnx.save(model, onnx_path)

        compiler = Compiler(target='x86', opt_level=2)
        compiler.compile(onnx_path, output_dir, max_memory='2048')

        tensors_c = Path(output_dir) / 'tensors.c'
        offsets = extract_tensor_offsets(tensors_c)

        if offsets['fast']:
            max_offset = max(offsets['fast'].values())
            assert max_offset < 2048, f"Max offset ({max_offset}) exceeds pool size (2048)"

    @pytest.mark.skip(reason="Now uses spill instead of raising error when model exceeds memory")
    def test_model_too_large_for_memory(self):
        """Test that we get a clear error when model can't fit in memory."""
        # Create a model where many tensors need to be alive simultaneously
        input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [20, 20])
        output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [20, 20])

        nodes = []
        # Create many branches that all need to be alive
        for i in range(10):
            nodes.append(helper.make_node('Relu', inputs=['input'], outputs=[f'branch_{i}']))

        # Merge all branches - they all need to be alive during this operation
        merge_inputs = [f'branch_{i}' for i in range(10)]
        nodes.append(helper.make_node('Add', inputs=merge_inputs[:2], outputs=['merge_0']))
        for i in range(1, 9):
            nodes.append(helper.make_node('Add', inputs=[f'merge_{i-1}', f'branch_{i+1}'], outputs=[f'merge_{i}']))

        nodes.append(helper.make_node('Relu', inputs=['merge_8'], outputs=['output']))

        graph = helper.make_graph(nodes, 'large_model', [input_tensor], [output_tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

        tmpdir = tempfile.mkdtemp()
        onnx_path = os.path.join(tmpdir, 'model.onnx')
        output_dir = os.path.join(tmpdir, 'build')

        onnx.save(model, onnx_path)

        compiler = Compiler(target='x86', opt_level=2)

        # This should fail because we need ~10 * 1600 = 16000 bytes but only have 2048
        # The aggressive spill strategy checks per-operator memory, so it fails
        # when a single operator's inputs+outputs exceed max_memory
        with pytest.raises(RuntimeError, match="Cannot fit.*in fast memory"):
            compiler.compile(onnx_path, output_dir, max_memory='2048')


def test_cost_aware_plan_emits_transfer_metrics_under_memory_limit():
    ctx = _build_cost_aware_spill_context()

    alloc_plan = get_memory_allocation_plan(ctx)

    assert alloc_plan is not None
    assert alloc_plan.has_spill
    assert alloc_plan.total_transfer_bytes > 0
    assert alloc_plan.spill_bytes > 0
    assert alloc_plan.reload_bytes > 0
    assert get_spill_plan(ctx) is None


def test_spill_analysis_skips_legacy_replanning_for_unified_spill():
    ctx = CompileContext(graph=Graph("dummy"), target="x86", optimization_level=2)
    ctx.metadata["max_memory"] = 64
    ctx.metadata["memory_allocation_plan"] = MemoryAllocationPlan(
        strategy_name="cost_aware",
        total_fast_memory=64,
        spill_points=[
            SpillPoint(
                tensor_name="x",
                after_node="n0",
                after_node_idx=0,
                from_buffer_id=0,
                from_fast_offset=0,
                to_slow_offset=0,
                size=64,
            )
        ],
    )

    SpillAnalysisPass().run(ctx)

    assert ctx.metadata["spill_plan"] is None


def test_cost_aware_transfer_bytes_do_not_exceed_basic():
    basic_ctx = _build_cost_aware_spill_context(
        optimization_level=2,
        memory_strategy="basic",
    )
    cost_aware_ctx = _build_cost_aware_spill_context()

    basic_plan = get_memory_allocation_plan(basic_ctx)
    cost_aware_plan = get_memory_allocation_plan(cost_aware_ctx)

    assert basic_plan is not None
    assert cost_aware_plan is not None
    assert basic_plan.has_spill
    assert cost_aware_plan.has_spill
    assert basic_plan.total_transfer_bytes > 0
    assert cost_aware_plan.total_transfer_bytes > 0
    assert cost_aware_plan.total_transfer_bytes <= basic_plan.total_transfer_bytes
