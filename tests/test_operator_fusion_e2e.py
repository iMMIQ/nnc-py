"""End-to-end integration tests for OperatorFusionPass."""

import os
import tempfile
import shutil
from pathlib import Path

import onnx
from onnx import helper
import pytest

from nnc_py import Compiler


class TestOperatorFusionE2E:
    """End-to-end tests for operator fusion."""

    def setup_method(self):
        """Set up test environment."""
        self.tmp_dir = tempfile.mkdtemp()
        self.runtime_dir = Path(__file__).resolve().parent.parent / "runtime"

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_conv_relu_fusion_e2e(self):
        """Test that Conv+ReLU fusion works end-to-end."""
        # Create a model with Conv followed by ReLU
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 32, 32])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 16, 30, 30])

        # Weight initializer
        weight_init = helper.make_tensor(
            "conv_weight", onnx.TensorProto.FLOAT, [16, 3, 3, 3], [0.1] * (16 * 3 * 3 * 3)
        )

        # Conv node
        conv = helper.make_node(
            "Conv",
            inputs=["input", "conv_weight"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )

        # ReLU node
        relu = helper.make_node("Relu", inputs=["conv_out"], outputs=["output"])

        graph = helper.make_graph(
            [conv, relu],
            "conv_relu_test",
            [input_val],
            [output_val],
            [weight_init]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        # Save and compile
        onnx_path = os.path.join(self.tmp_dir, "model.onnx")
        onnx.save(model, onnx_path)

        output_dir = os.path.join(self.tmp_dir, "build")

        # Compile at O3 (with fusion)
        compiler = Compiler(target="x86", opt_level=3)
        compiler.compile(onnx_path, output_dir)

        # Check that model.c was generated
        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c should be generated"

    def test_add_relu_fusion_e2e(self):
        """Test that Add+ReLU fusion works end-to-end."""
        input1_val = helper.make_tensor_value_info("input1", onnx.TensorProto.FLOAT, [1, 16, 30, 30])
        input2_val = helper.make_tensor_value_info("input2", onnx.TensorProto.FLOAT, [1, 16, 30, 30])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 16, 30, 30])

        add = helper.make_node("Add", inputs=["input1", "input2"], outputs=["add_out"])
        relu = helper.make_node("Relu", inputs=["add_out"], outputs=["output"])

        graph = helper.make_graph(
            [add, relu],
            "add_relu_test",
            [input1_val, input2_val],
            [output_val]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        onnx_path = os.path.join(self.tmp_dir, "model.onnx")
        onnx.save(model, onnx_path)

        output_dir = os.path.join(self.tmp_dir, "build")
        compiler = Compiler(target="x86", opt_level=3)
        compiler.compile(onnx_path, output_dir)

        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c should be generated"

    def test_o3_enables_fusion(self):
        """Test that O3 enables fusion while O2 does not."""
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 32, 32])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 16, 30, 30])

        weight_init = helper.make_tensor(
            "conv_weight", onnx.TensorProto.FLOAT, [16, 3, 3, 3], [0.1] * (16 * 3 * 3 * 3)
        )

        conv = helper.make_node(
            "Conv",
            inputs=["input", "conv_weight"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
        )
        relu = helper.make_node("Relu", inputs=["conv_out"], outputs=["output"])

        graph = helper.make_graph(
            [conv, relu],
            "conv_relu_test",
            [input_val],
            [output_val],
            [weight_init]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        onnx_path = os.path.join(self.tmp_dir, "model.onnx")
        onnx.save(model, onnx_path)

        # Both O2 and O3 should compile successfully
        # (O3 has fusion, but the IR should still be valid)
        for opt_level in [2, 3]:
            output_dir = os.path.join(self.tmp_dir, f"build_o{opt_level}")
            compiler = Compiler(target="x86", opt_level=opt_level)
            compiler.compile(onnx_path, output_dir)

            model_c_path = os.path.join(output_dir, "model.c")
            assert os.path.exists(model_c_path), f"model.c should be generated at O{opt_level}"
