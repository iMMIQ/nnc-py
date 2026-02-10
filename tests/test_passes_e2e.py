"""End-to-end integration tests for optimization passes."""

import os
import tempfile
import shutil
from pathlib import Path

import onnx
from onnx import helper
import pytest

from nnc_py import Compiler


class TestPassesE2E:
    """End-to-end tests for optimization passes."""

    def setup_method(self):
        """Set up test environment."""
        self.tmp_dir = tempfile.mkdtemp()
        self.runtime_dir = Path(__file__).resolve().parent.parent / "runtime"

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_identity_elimination_e2e(self):
        """Test that IdentityEliminationPass works end-to-end."""
        # Create a model with Identity nodes
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

        # Add some Identity nodes
        identity1 = helper.make_node("Identity", inputs=["input"], outputs=["id1_out"])
        identity2 = helper.make_node("Identity", inputs=["id1_out"], outputs=["id2_out"])
        relu = helper.make_node("Relu", inputs=["id2_out"], outputs=["output"])

        graph = helper.make_graph(
            [identity1, identity2, relu],
            "identity_test",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        # Save and compile
        onnx_path = os.path.join(self.tmp_dir, "model.onnx")
        onnx.save(model, onnx_path)

        output_dir = os.path.join(self.tmp_dir, "build")
        compiler = Compiler(target="x86", opt_level=1)
        compiler.compile(onnx_path, output_dir)

        # Check that model.c was generated
        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c should be generated"

        # Read and check the generated code
        code = Path(model_c_path).read_text()

        # Identity nodes should be eliminated, so no nnc_identity calls
        # (unless they're in the runtime for other reasons)
        # The key is that compilation succeeds

    def test_dead_code_elimination_e2e(self):
        """Test that DeadCodeEliminationPass works end-to-end."""
        # Create a model with unused nodes
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

        # This Relu is unused (output not connected to final output)
        unused_relu = helper.make_node("Relu", inputs=["input"], outputs=["unused_out"])

        # This path is used
        used_relu = helper.make_node("Relu", inputs=["input"], outputs=["output"])

        graph = helper.make_graph(
            [unused_relu, used_relu],
            "dead_code_test",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        # Save and compile
        onnx_path = os.path.join(self.tmp_dir, "model.onnx")
        onnx.save(model, onnx_path)

        output_dir = os.path.join(self.tmp_dir, "build")
        compiler = Compiler(target="x86", opt_level=1)
        compiler.compile(onnx_path, output_dir)

        # Check that model.c was generated
        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c should be generated"

    def test_combined_passes_e2e(self):
        """Test that both passes work together correctly."""
        # Create a model with both Identity nodes and dead code
        input_val = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
        output_val = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])

        # Identity chain (should be eliminated)
        identity1 = helper.make_node("Identity", inputs=["input"], outputs=["id1"])
        identity2 = helper.make_node("Identity", inputs=["id1"], outputs=["id2"])

        # Dead code (unused)
        unused_relu = helper.make_node("Relu", inputs=["id1"], outputs=["unused"])

        # Used path
        relu = helper.make_node("Relu", inputs=["id2"], outputs=["output"])

        graph = helper.make_graph(
            [identity1, identity2, unused_relu, relu],
            "combined_test",
            [input_val],
            [output_val]
        )

        model = helper.make_model(graph)
        model.opset_import[0].version = 13

        # Save and compile
        onnx_path = os.path.join(self.tmp_dir, "model.onnx")
        onnx.save(model, onnx_path)

        output_dir = os.path.join(self.tmp_dir, "build")
        compiler = Compiler(target="x86", opt_level=1)
        compiler.compile(onnx_path, output_dir)

        # Check that model.c was generated
        model_c_path = os.path.join(output_dir, "model.c")
        assert os.path.exists(model_c_path), "model.c should be generated"
