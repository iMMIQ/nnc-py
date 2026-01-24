"""Debug comparison tool for comparing NNC C output with ONNX Runtime output.

This module provides utilities to:
1. Run ONNX models and capture intermediate layer outputs
2. Parse NNC debug output files
3. Compare results and report mismatches
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort


class DebugOutputParser:
    """Parse NNC debug output file."""

    def __init__(self, debug_file: str | Path):
        """Initialize parser with debug file path.

        Args:
            debug_file: Path to the NNC debug output file.
        """
        self.debug_file = Path(debug_file)
        self.outputs: Dict[str, Dict[str, Any]] = {}

    def parse(self) -> Dict[str, Dict[str, Any]]:
        """Parse the debug output file.

        Returns:
            Dictionary mapping tensor names to their data and metadata.
            Format: {
                "tensor_name": {
                    "node_idx": int,
                    "node_name": str,
                    "shape": List[int],
                    "data": np.ndarray
                }
            }
        """
        self.outputs = {}
        current_tensor = None
        current_shape = []
        collecting_data = False
        data_values = []

        with open(self.debug_file, "r") as f:
            for line in f:
                line = line.strip()

                # Start of tensor
                if line.startswith("DEBUG_TENSOR_START"):
                    parts = line.split()
                    # Format: DEBUG_TENSOR_START tensor_name node_idx
                    current_tensor = {
                        "name": parts[1],
                        "node_idx": int(parts[2]),
                        "shape": [],
                        "data": []
                    }
                    current_shape = []
                    collecting_data = False
                    data_values = []

                elif line.startswith("SHAPE"):
                    current_tensor["ndim"] = int(line.split()[1])

                elif line.startswith("DIM"):
                    parts = line.split()
                    # Format: DIM idx value
                    dim_idx = int(parts[1])
                    dim_val = int(parts[2])
                    current_shape.append(dim_val)

                elif line.startswith("DATA_START"):
                    current_tensor["shape"] = current_shape
                    collecting_data = True
                    data_values = []

                elif line.startswith("DATA_END"):
                    collecting_data = False
                    # Convert to numpy array
                    data_array = np.array(data_values, dtype=np.float32)
                    # Reshape according to shape (if no -1 dimensions)
                    # Handle scalar tensors (ndim=0, shape=[]) specially
                    ndim = current_tensor.get("ndim", len(current_shape))
                    if ndim == 0:
                        # Scalar tensor - reshape to empty tuple for scalar
                        if len(data_array) == 1:
                            data_array = data_array.reshape(())
                    elif current_shape and -1 not in current_shape:
                        data_array = data_array.reshape(current_shape)
                    current_tensor["data"] = data_array
                    self.outputs[current_tensor["name"]] = current_tensor
                    current_tensor = None

                elif collecting_data and line:
                    try:
                        data_values.append(float(line))
                    except ValueError:
                        pass

        return self.outputs


class ONNXRuntimeRunner:
    """Run ONNX models with intermediate output capture."""

    def __init__(self, onnx_path: str | Path):
        """Initialize runner with ONNX model path.

        Args:
            onnx_path: Path to the ONNX model file.
        """
        self.onnx_path = Path(onnx_path)
        self.model = onnx.load(str(self.onnx_path))
        self.session = ort.InferenceSession(str(self.onnx_path))

        # Get all intermediate layer names
        self.intermediate_outputs = self._get_intermediate_outputs()

    def _get_intermediate_outputs(self) -> List[str]:
        """Get names of all intermediate output tensors.

        Returns:
            List of tensor names in topological order.
        """
        # Get all node outputs
        outputs = []
        for node in self.model.graph.node:
            for output in node.output:
                outputs.append(output)

        # Get model outputs (final outputs)
        model_outputs = {o.name for o in self.model.graph.output}

        # Filter out final outputs to get only intermediate
        intermediate = [o for o in outputs if o not in model_outputs]
        return intermediate

    def get_input_info(self) -> List[Tuple[str, List[int]]]:
        """Get input tensor names and shapes.

        Returns:
            List of (name, shape) tuples.
        """
        inputs = []
        for inp in self.model.graph.input:
            name = inp.name
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    # Handle dynamic dimensions
                    shape.append(-1)
            inputs.append((name, shape))
        return inputs

    def run_with_intermediates(
        self,
        input_data: Optional[Dict[str, np.ndarray]] = None,
        test_pattern: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Run the model and capture all intermediate outputs.

        Args:
            input_data: Optional dictionary mapping input names to arrays.
                       If None, uses test pattern (i * 0.01).
            test_pattern: If True and input_data is None, generate test pattern.

        Returns:
            Dictionary mapping tensor names to their output arrays.
        """
        # Prepare inputs
        if input_data is None:
            input_data = {}
            input_info = self.get_input_info()
            for name, shape in input_info:
                # Replace -1 with 1 for unknown dimensions
                concrete_shape = [s if s > 0 else 1 for s in shape]
                size = int(np.prod(concrete_shape))
                if test_pattern:
                    # Generate test pattern: i * 0.01
                    data = np.arange(size, dtype=np.float32) * 0.01
                else:
                    data = np.zeros(size, dtype=np.float32)
                input_data[name] = data.reshape(concrete_shape)

        # Need to expose intermediate outputs
        # Use shape inference to get type information for intermediate tensors
        from copy import deepcopy
        import tempfile
        import os

        model_copy = deepcopy(self.model)

        # Apply shape inference to get complete type information
        try:
            model_copy = onnx.shape_inference.infer_shapes(model_copy)
        except Exception:
            pass  # Shape inference may fail for some models

        # Get existing output names
        existing_output_names = {o.name for o in model_copy.graph.output}

        # After shape inference, graph.value_info contains inferred type info for intermediate tensors
        # Add these to the graph outputs
        if hasattr(model_copy.graph, 'value_info'):
            for vi in model_copy.graph.value_info:
                if vi.name not in existing_output_names:
                    model_copy.graph.output.append(vi)

        # Save to temp file and run
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = tmp.name
            try:
                onnx.save(model_copy, tmp_path)
                temp_session = ort.InferenceSession(tmp_path)
                results = temp_session.run(None, input_data)

                # Map result names to outputs
                output_names = [o.name for o in model_copy.graph.output]
                return dict(zip(output_names, results))
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass


class DebugComparator:
    """Compare NNC debug output with ONNX Runtime output."""

    def __init__(
        self,
        nnc_debug_file: str | Path,
        onnx_model: str | Path,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ):
        """Initialize comparator.

        Args:
            nnc_debug_file: Path to NNC debug output file.
            onnx_model: Path to ONNX model file.
            rtol: Relative tolerance for comparison.
            atol: Absolute tolerance for comparison.
        """
        self.nnc_debug_file = Path(nnc_debug_file)
        self.onnx_model = Path(onnx_model)
        self.rtol = rtol
        self.atol = atol

        # Parse NNC output
        self.nnc_parser = DebugOutputParser(self.nnc_debug_file)
        self.nnc_outputs = self.nnc_parser.parse()

        # Setup ONNX runner
        self.onnx_runner = ONNXRuntimeRunner(self.onnx_model)

    def compare(self) -> Dict[str, Any]:
        """Compare NNC and ONNX Runtime outputs.

        Returns:
            Dictionary with comparison results.
        """
        # Run ONNX model with test pattern
        onnx_outputs = self.onnx_runner.run_with_intermediates(test_pattern=True)

        results = {
            "matched": [],
            "mismatched": [],
            "missing_in_nnc": [],
            "missing_in_onnx": [],
            "shape_mismatch": [],
        }

        # Compare each tensor
        all_tensors = set(self.nnc_outputs.keys()) | set(onnx_outputs.keys())

        for tensor_name in all_tensors:
            nnc_data = self.nnc_outputs.get(tensor_name)
            onnx_data = onnx_outputs.get(tensor_name)

            if nnc_data is None:
                results["missing_in_nnc"].append(tensor_name)
                continue

            if onnx_data is None:
                results["missing_in_onnx"].append(tensor_name)
                continue

            # Compare shapes
            nnc_shape = list(nnc_data["data"].shape)
            onnx_shape = list(onnx_data.shape)

            if nnc_shape != onnx_shape:
                results["shape_mismatch"].append({
                    "tensor": tensor_name,
                    "nnc_shape": nnc_shape,
                    "onnx_shape": onnx_shape,
                })
                continue

            # Compare values
            try:
                if np.allclose(nnc_data["data"], onnx_data, rtol=self.rtol, atol=self.atol):
                    results["matched"].append(tensor_name)
                else:
                    max_diff = np.max(np.abs(nnc_data["data"] - onnx_data))
                    results["mismatched"].append({
                        "tensor": tensor_name,
                        "max_diff": float(max_diff),
                        "node_idx": nnc_data["node_idx"],
                        "node_name": nnc_data.get("node_name", "unknown"),
                    })
            except (TypeError, ValueError) as e:
                results["mismatched"].append({
                    "tensor": tensor_name,
                    "error": str(e),
                })

        return results

    def print_report(self, results: Dict[str, Any]) -> None:
        """Print comparison report.

        Args:
            results: Comparison results from compare().
        """
        total = (
            len(results["matched"])
            + len(results["mismatched"])
            + len(results["shape_mismatch"])
        )

        print("\n" + "=" * 60)
        print("NNC vs ONNX Runtime Comparison Report")
        print("=" * 60)
        print(f"Total tensors compared: {total}")
        print(f"  ✓ Matched: {len(results['matched'])}")
        print(f"  ✗ Mismatched: {len(results['mismatched'])}")
        print(f"  ⚠ Shape mismatch: {len(results['shape_mismatch'])}")
        print(f"  - Missing in NNC: {len(results['missing_in_nnc'])}")
        print(f"  - Missing in ONNX: {len(results['missing_in_onnx'])}")

        if results["matched"]:
            print("\n✓ Matched tensors:")
            for name in results["matched"]:
                print(f"  - {name}")

        if results["mismatched"]:
            print("\n✗ Mismatched tensors:")
            for item in results["mismatched"]:
                if "max_diff" in item:
                    print(f"  - {item['tensor']} (max diff: {item['max_diff']:.6f})")
                else:
                    print(f"  - {item['tensor']} (error: {item.get('error', 'unknown')})")

        if results["shape_mismatch"]:
            print("\n⚠ Shape mismatches:")
            for item in results["shape_mismatch"]:
                print(f"  - {item['tensor']}: NNC {item['nnc_shape']} vs ONNX {item['onnx_shape']}")

        if results["missing_in_nnc"]:
            print("\n- Missing in NNC output:")
            for name in results["missing_in_nnc"]:
                print(f"  - {name}")

        if results["missing_in_onnx"]:
            print("\n- Missing in ONNX output:")
            for name in results["missing_in_onnx"]:
                print(f"  - {name}")

        print("=" * 60)

        # Return exit code
        return 0 if len(results["mismatched"]) == 0 and len(results["shape_mismatch"]) == 0 else 1


def compare_debug_output(
    nnc_debug_file: str | Path,
    onnx_model: str | Path,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> int:
    """Convenience function to compare NNC debug output with ONNX Runtime.

    Args:
        nnc_debug_file: Path to NNC debug output file.
        onnx_model: Path to ONNX model file.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        0 if all outputs match, 1 otherwise.
    """
    comparator = DebugComparator(nnc_debug_file, onnx_model, rtol, atol)
    results = comparator.compare()
    return comparator.print_report(results)
