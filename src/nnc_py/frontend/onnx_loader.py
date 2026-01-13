"""ONNX model loader and parser."""

import onnx
from onnx import helper, numpy_helper

from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType, MemoryLayout


class ONNXFrontend:
    """ONNX model loading and parsing frontend."""

    # ONNX dtype to IR dtype mapping
    DTYPE_MAP = {
        onnx.TensorProto.FLOAT: DataType.FLOAT32,
        onnx.TensorProto.FLOAT16: DataType.FLOAT16,
        onnx.TensorProto.INT32: DataType.INT32,
        onnx.TensorProto.INT8: DataType.INT8,
        onnx.TensorProto.UINT8: DataType.UINT8,
        onnx.TensorProto.BOOL: DataType.BOOL,
    }

    def load(self, onnx_path: str) -> Graph:
        """Load ONNX model and convert to IR graph.

        Args:
            onnx_path: Path to the ONNX model file.

        Returns:
            The converted IR Graph.
        """
        # Load ONNX model
        model = onnx.load(onnx_path)
        onnx_graph = model.graph

        # Create IR graph
        ir_graph = Graph(name=onnx_graph.name or "main")

        # 1. Parse inputs
        for input_tensor in onnx_graph.input:
            tensor_type = self._parse_tensor_type(input_tensor)
            ir_graph.add_tensor(tensor_type)
            ir_graph.inputs.append(tensor_type.name)

        # 2. Parse outputs
        for output_tensor in onnx_graph.output:
            tensor_type = self._parse_tensor_type(output_tensor)
            ir_graph.add_tensor(tensor_type)
            ir_graph.outputs.append(tensor_type.name)

        # 3. Parse operator nodes
        for onnx_node in onnx_graph.node:
            ir_node = self._parse_node(onnx_node)
            ir_graph.add_node(ir_node)

            # Create tensor definitions for node outputs if not already defined
            for output_name in onnx_node.output:
                if output_name not in ir_graph.tensors:
                    # Try to infer type from node
                    inferred_type = self._infer_tensor_type(
                        onnx_node, output_name, ir_graph
                    )
                    if inferred_type:
                        ir_graph.add_tensor(inferred_type)

        # 4. Parse constants (initializers)
        for initializer in onnx_graph.initializer:
            arr = numpy_helper.to_array(initializer)
            ir_graph.constants[initializer.name] = arr

            # Create tensor definition for constant
            tensor_type = self._parse_initializer(initializer)
            ir_graph.add_tensor(tensor_type)

        # 5. Validate graph
        self._validate_graph(ir_graph)

        return ir_graph

    def _parse_tensor_type(self, onnx_tensor) -> TensorType:
        """Parse ONNX tensor type info."""
        dtype = self._map_onnx_dtype(
            onnx_tensor.type.tensor_type.elem_type
        )

        dims = []
        for dim in onnx_tensor.type.tensor_type.shape.dim:
            if dim.dim_value:
                dims.append(dim.dim_value)
            elif dim.dim_param:
                dims.append(dim.dim_param)  # Symbolic dimension
            else:
                dims.append(-1)  # Unknown dimension

        # Default to NCHW for 4D tensors
        layout = MemoryLayout.NCHW if len(dims) == 4 else MemoryLayout.NHWC
        shape = TensorShape(dims=dims, layout=layout)

        return TensorType(dtype=dtype, shape=shape, name=onnx_tensor.name)

    def _parse_initializer(self, initializer) -> TensorType:
        """Parse ONNX initializer (constant weight)."""
        dtype = self._map_onnx_dtype(initializer.data_type)

        dims = list(initializer.dims)
        layout = MemoryLayout.OIHW if len(dims) == 4 else MemoryLayout.NCHW
        shape = TensorShape(dims=dims, layout=layout)

        return TensorType(dtype=dtype, shape=shape, name=initializer.name)

    def _parse_node(self, onnx_node) -> Node:
        """Parse ONNX node to IR node."""
        op_type = OpType(onnx_node.op_type)
        attrs = self._parse_attributes(onnx_node)

        # Generate unique name if not provided
        name = onnx_node.name or f"{onnx_node.op_type}_{id(onnx_node)}"

        return Node(
            op_type=op_type,
            name=name,
            inputs=list(onnx_node.input),
            outputs=list(onnx_node.output),
            attrs=attrs,
        )

    def _parse_attributes(self, onnx_node) -> dict:
        """Parse ONNX node attributes."""
        attrs = {}

        for attr in onnx_node.attribute:
            if attr.name == "kernel_shape":
                attrs["kernel_shape"] = [int(d) for d in attr.ints]
            elif attr.name == "strides":
                attrs["strides"] = [int(d) for d in attr.ints]
            elif attr.name == "pads":
                attrs["pads"] = [int(d) for d in attr.ints]
            elif attr.name == "dilations":
                attrs["dilations"] = [int(d) for d in attr.ints]
            elif attr.name == "group":
                attrs["group"] = int(attr.i)
            elif attr.name == "axes":
                attrs["axes"] = [int(d) for d in attr.ints]
            elif attr.name == "axis":
                attrs["axis"] = int(attr.i)
            elif attr.name == "keepdims":
                attrs["keepdims"] = int(attr.i)
            elif attr.name == "shape":
                attrs["shape"] = [int(d) for d in attr.ints]
            elif attr.name == "perm":
                attrs["perm"] = [int(d) for d in attr.ints]
            elif attr.name == "alpha":
                attrs["alpha"] = float(attr.f)
            elif attr.name == "beta":
                attrs["beta"] = float(attr.f)
            elif attr.name == "activation":
                # Gemm activation (none, relu, etc.)
                attrs["activation"] = attr.s.decode() if attr.s else None

        return attrs

    def _infer_tensor_type(self, onnx_node, output_name: str, graph: Graph) -> TensorType | None:
        """Infer tensor type from node and input types."""
        # For simplicity, assume float32 and same shape as first input
        # This is a placeholder - proper type inference would be more sophisticated
        if onnx_node.input and onnx_node.input[0] in graph.tensors:
            input_tensor = graph.tensors[onnx_node.input[0]]
            return TensorType(
                dtype=input_tensor.dtype,
                shape=input_tensor.shape,
                name=output_name,
            )
        return None

    def _map_onnx_dtype(self, onnx_dtype: int) -> DataType:
        """Map ONNX dtype to IR DataType."""
        return self.DTYPE_MAP.get(onnx_dtype, DataType.FLOAT32)

    def _validate_graph(self, graph: Graph):
        """Validate the graph structure."""
        # Check all inputs are defined
        for node in graph.nodes.values():
            for input_name in node.inputs:
                if input_name not in graph.tensors:
                    raise ValueError(f"Input tensor '{input_name}' not found in graph for node '{node.name}'")

        # Check all outputs are defined
        for node in graph.nodes.values():
            for output_name in node.outputs:
                if output_name not in graph.tensors:
                    raise ValueError(f"Output tensor '{output_name}' not found in graph for node '{node.name}'")
