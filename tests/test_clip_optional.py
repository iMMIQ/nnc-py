"""Tests for Clip operator with optional inputs."""

import tempfile
import subprocess
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

from nnc_py.frontend.onnx_loader import ONNXFrontend
from nnc_py.ir.node import Node, OpType

from nnc_py import Compiler


def test_clip_with_only_min():
    """Test Clip with only min tensor input (max is optional/empty)."""
    print("\n=== Testing Clip with only min (max is empty) ===")

    # Create Clip node with min tensor input, but no max input
    # In ONNX, optional inputs are represented as empty strings
    clip_node = helper.make_node(
        "Clip",
        inputs=["input", "min", ""],  # Empty string for optional max
        outputs=["output"],
        name="clip_node",
    )

    # Create min constant tensor
    min_const = helper.make_tensor("min", onnx.TensorProto.FLOAT, [], [0.0])

    input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3])
    output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph(
        [clip_node, helper.make_node("Constant", inputs=[], outputs=["min"], value=min_const)],
        "clip_min_only_model",
        [input_val],
        [output_val]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

    # Save to temp file and load with frontend
    tmpdir = tempfile.mkdtemp()
    onnx_path = f"{tmpdir}/clip_min_only.onnx"
    onnx.save(model, onnx_path)

    try:
        frontend = ONNXFrontend()
        ir_graph = frontend.load(onnx_path)

        # Verify graph structure
        assert "input" in ir_graph.tensors
        assert "output" in ir_graph.tensors
        assert "clip_node" in ir_graph.nodes

        # Check that the node has 3 inputs (including empty string for max)
        clip_node_ir = ir_graph.nodes["clip_node"]
        assert len(clip_node_ir.inputs) == 3
        # One of the inputs should be empty string (optional max)
        assert "" in clip_node_ir.inputs

        print("   PASSED: Model loaded successfully with empty max input")

        # Try to compile
        output_dir = f"{tmpdir}/build"
        compiler = Compiler(target="x86", opt_level=0)
        compiler.compile(onnx_path, output_dir)

        print("   Compilation successful!")

        # Check generated model.c
        model_c = Path(output_dir) / "model.c"
        model_c_content = model_c.read_text()

        # Should contain nnc_clip call
        assert "nnc_clip" in model_c_content
        print("   Code emission successful!")

        return True

    except Exception as e:
        print(f"   FAILED: {e}")
        return False
    finally:
        import shutil
        shutil.rmtree(tmpdir)


def test_clip_with_only_max():
    """Test Clip with only max tensor input (min is optional/empty)."""
    print("\n=== Testing Clip with only max (min is empty) ===")

    # Create Clip node with max tensor input, but no min input
    clip_node = helper.make_node(
        "Clip",
        inputs=["input", "", "max"],  # Empty string for optional min
        outputs=["output"],
        name="clip_node",
    )

    # Create max constant tensor
    max_const = helper.make_tensor("max", onnx.TensorProto.FLOAT, [], [5.0])

    input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3])
    output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph(
        [clip_node, helper.make_node("Constant", inputs=[], outputs=["max"], value=max_const)],
        "clip_max_only_model",
        [input_val],
        [output_val]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

    tmpdir = tempfile.mkdtemp()
    onnx_path = f"{tmpdir}/clip_max_only.onnx"
    onnx.save(model, onnx_path)

    try:
        frontend = ONNXFrontend()
        ir_graph = frontend.load(onnx_path)

        # Verify graph structure
        assert "input" in ir_graph.tensors
        assert "output" in ir_graph.tensors
        assert "clip_node" in ir_graph.nodes

        clip_node_ir = ir_graph.nodes["clip_node"]
        assert len(clip_node_ir.inputs) == 3
        assert "" in clip_node_ir.inputs

        print("   PASSED: Model loaded successfully with empty min input")

        # Try to compile
        output_dir = f"{tmpdir}/build"
        compiler = Compiler(target="x86", opt_level=0)
        compiler.compile(onnx_path, output_dir)

        print("   Compilation successful!")

        # Check generated model.c
        model_c = Path(output_dir) / "model.c"
        model_c_content = model_c.read_text()

        assert "nnc_clip" in model_c_content
        print("   Code emission successful!")

        return True

    except Exception as e:
        print(f"   FAILED: {e}")
        return False
    finally:
        import shutil
        shutil.rmtree(tmpdir)


def test_clip_with_both_min_max():
    """Test Clip with both min and max provided."""
    print("\n=== Testing Clip with both min and max ===")

    clip_node = helper.make_node(
        "Clip",
        inputs=["input", "min", "max"],
        outputs=["output"],
        name="clip_node",
    )

    # Create min and max constant tensors
    min_const = helper.make_tensor("min", onnx.TensorProto.FLOAT, [], [0.0])
    max_const = helper.make_tensor("max", onnx.TensorProto.FLOAT, [], [5.0])

    input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3])
    output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph(
        [
            clip_node,
            helper.make_node("Constant", inputs=[], outputs=["min"], value=min_const),
            helper.make_node("Constant", inputs=[], outputs=["max"], value=max_const),
        ],
        "clip_both_model",
        [input_val],
        [output_val]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

    tmpdir = tempfile.mkdtemp()
    onnx_path = f"{tmpdir}/clip_both.onnx"
    onnx.save(model, onnx_path)

    try:
        frontend = ONNXFrontend()
        ir_graph = frontend.load(onnx_path)

        assert "clip_node" in ir_graph.nodes

        # Try to compile
        output_dir = f"{tmpdir}/build"
        compiler = Compiler(target="x86", opt_level=0)
        compiler.compile(onnx_path, output_dir)

        print("   PASSED: Model with both min and max compiled successfully!")

        return True

    except Exception as e:
        print(f"   FAILED: {e}")
        return False
    finally:
        import shutil
        shutil.rmtree(tmpdir)


def test_clip_with_attributes():
    """Test Clip with min and max as attributes (not inputs)."""
    print("\n=== Testing Clip with min/max as attributes ===")

    clip_node = helper.make_node(
        "Clip",
        inputs=["input"],
        outputs=["output"],
        name="clip_node",
        min=0.0,
        max=5.0,
    )

    input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3])
    output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph(
        [clip_node],
        "clip_attrs_model",
        [input_val],
        [output_val]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])

    tmpdir = tempfile.mkdtemp()
    onnx_path = f"{tmpdir}/clip_attrs.onnx"
    onnx.save(model, onnx_path)

    try:
        frontend = ONNXFrontend()
        ir_graph = frontend.load(onnx_path)

        assert "clip_node" in ir_graph.nodes

        # Try to compile
        output_dir = f"{tmpdir}/build"
        compiler = Compiler(target="x86", opt_level=0)
        compiler.compile(onnx_path, output_dir)

        print("   PASSED: Model with attributes compiled successfully!")

        return True

    except Exception as e:
        print(f"   FAILED: {e}")
        return False
    finally:
        import shutil
        shutil.rmtree(tmpdir)


def run_all_clip_tests():
    """Run all Clip tests."""
    print("\n" + "=" * 60)
    print("CLIP OPTIONAL INPUTS TEST SUITE")
    print("=" * 60)

    all_passed = True

    # Test 1: Only min provided
    try:
        if not test_clip_with_only_min():
            print("\n❌ Test 'Clip with only min' FAILED")
            all_passed = False
        else:
            print("\n✅ Test 'Clip with only min' PASSED")
    except Exception as e:
        print(f"\n❌ Test 'Clip with only min' FAILED with exception: {e}")
        all_passed = False

    # Test 2: Only max provided
    try:
        if not test_clip_with_only_max():
            print("\n❌ Test 'Clip with only max' FAILED")
            all_passed = False
        else:
            print("\n✅ Test 'Clip with only max' PASSED")
    except Exception as e:
        print(f"\n❌ Test 'Clip with only max' FAILED with exception: {e}")
        all_passed = False

    # Test 3: Both min and max provided
    try:
        if not test_clip_with_both_min_max():
            print("\n❌ Test 'Clip with both min and max' FAILED")
            all_passed = False
        else:
            print("\n✅ Test 'Clip with both min and max' PASSED")
    except Exception as e:
        print(f"\n❌ Test 'Clip with both min and max' FAILED with exception: {e}")
        all_passed = False

    # Test 4: Min/max as attributes
    try:
        if not test_clip_with_attributes():
            print("\n❌ Test 'Clip with attributes' FAILED")
            all_passed = False
        else:
            print("\n✅ Test 'Clip with attributes' PASSED")
    except Exception as e:
        print(f"\n❌ Test 'Clip with attributes' FAILED with exception: {e}")
        all_passed = False

    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CLIP TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    import sys
    success = run_all_clip_tests()
    sys.exit(0 if success else 1)
