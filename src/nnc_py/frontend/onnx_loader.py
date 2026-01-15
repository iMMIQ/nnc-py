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

        # 3. Parse constants (initializers) FIRST - needed for shape inference
        for initializer in onnx_graph.initializer:
            arr = numpy_helper.to_array(initializer)
            ir_graph.constants[initializer.name] = arr

            # Create tensor definition for constant
            tensor_type = self._parse_initializer(initializer)
            ir_graph.add_tensor(tensor_type)

        # 4. Parse operator nodes
        for onnx_node in onnx_graph.node:
            ir_node = self._parse_node(onnx_node)

            # Special handling for Constant nodes - extract value from attributes
            if onnx_node.op_type == "Constant":
                self._parse_constant_node(onnx_node, ir_graph)

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
            elif attr.name == "epsilon":
                attrs["epsilon"] = float(attr.f)
            elif attr.name == "stash_type":
                attrs["stash_type"] = int(attr.i)

        return attrs

    def _parse_constant_node(self, onnx_node, graph: Graph) -> None:
        """Parse Constant node and extract value to constants.

        Constant nodes have their value stored in the 'value' attribute.
        """
        # Find the 'value' attribute
        value_attr = None
        for attr in onnx_node.attribute:
            if attr.name == "value":
                value_attr = attr
                break

        if value_attr is None:
            return

        # Convert TensorProto to numpy array
        try:
            arr = numpy_helper.to_array(value_attr.t)
            output_name = onnx_node.output[0]
            graph.constants[output_name] = arr

            # Create tensor definition for the constant
            dtype = self._map_onnx_dtype(value_attr.t.data_type)
            dims = list(value_attr.t.dims)
            layout = MemoryLayout.NCHW if len(dims) == 4 else MemoryLayout.NHWC
            shape = TensorShape(dims=dims, layout=layout)

            tensor_type = TensorType(dtype=dtype, shape=shape, name=output_name)
            graph.add_tensor(tensor_type)
        except Exception as e:
            # If parsing fails, the constant will remain undefined
            # This can happen for empty tensors or special cases
            pass

    def _infer_tensor_type(self, onnx_node, output_name: str, graph: Graph) -> TensorType | None:
        """Infer tensor type from node and input types."""
        # Handle Constant nodes - no inputs, type should already be defined
        if onnx_node.op_type == "Constant":
            # Type is already defined by _parse_constant_node
            return graph.tensors.get(output_name)

        if not onnx_node.input:
            return None

        input_tensor = graph.tensors.get(onnx_node.input[0])
        if not input_tensor:
            return None

        op_type = onnx_node.op_type

        # Handle different operators with proper shape inference
        if op_type == "Conv":
            # Input: [N, C_in, H, W]
            # Weight: [C_out, C_in, kH, kW]
            # Output: [N, C_out, H_out, W_out]
            if len(onnx_node.input) < 2:
                return None

            # Get weight tensor to find C_out
            weight_name = onnx_node.input[1]
            weight_tensor = graph.tensors.get(weight_name)
            if not weight_tensor or len(weight_tensor.shape.dims) < 1:
                return None

            # C_out is the first dimension of weight
            c_out = weight_tensor.shape.dims[0]

            # Compute spatial output dimensions
            # H_out = (H_in + 2*pad - kernel) / stride + 1
            in_shape = input_tensor.shape.dims
            if len(in_shape) != 4:
                return None

            # Get kernel shape, stride, padding from attributes
            kernel_shape = self._get_attr(onnx_node, "kernel_shape")
            strides = self._get_attr(onnx_node, "strides", [1, 1])
            pads = self._get_attr(onnx_node, "pads", [0, 0, 0, 0])

            # If kernel_shape not in attrs, try to get from weight tensor
            if not kernel_shape and weight_tensor:
                weight_dims = weight_tensor.shape.dims
                if len(weight_dims) >= 4:
                    kernel_shape = [weight_dims[2], weight_dims[3]]

            if not kernel_shape or len(kernel_shape) != 2:
                return None

            # pads in ONNX is [pad_top, pad_left, pad_bottom, pad_right]
            pad_h = pads[0] + pads[2] if len(pads) == 4 else pads[0] * 2 if len(pads) == 2 else 0
            pad_w = pads[1] + pads[3] if len(pads) == 4 else pads[1] * 2 if len(pads) == 2 else 0

            h_out = (in_shape[2] + pad_h - kernel_shape[0]) // strides[0] + 1
            w_out = (in_shape[3] + pad_w - kernel_shape[1]) // (strides[1] if len(strides) > 1 else strides[0]) + 1

            output_shape = TensorShape(
                dims=[in_shape[0], c_out, h_out, w_out],
                layout=input_tensor.shape.layout
            )

            return TensorType(
                dtype=input_tensor.dtype,
                shape=output_shape,
                name=output_name,
            )

        elif op_type == "Relu":
            # Relu preserves shape
            return TensorType(
                dtype=input_tensor.dtype,
                shape=input_tensor.shape,
                name=output_name,
            )

        elif op_type == "LayerNormalization":
            # LayerNormalization preserves shape
            return TensorType(
                dtype=input_tensor.dtype,
                shape=input_tensor.shape,
                name=output_name,
            )

        elif op_type == "Identity":
            # Identity preserves shape and type
            return TensorType(
                dtype=input_tensor.dtype,
                shape=input_tensor.shape,
                name=output_name,
            )

        # Default: same shape as input (for other ops)
        return TensorType(
            dtype=input_tensor.dtype,
            shape=input_tensor.shape,
            name=output_name,
        )

    def _get_attr(self, onnx_node, attr_name: str, default=None):
        """Get attribute value from ONNX node."""
        for attr in onnx_node.attribute:
            if attr.name == attr_name:
                if attr.ints:
                    return [int(v) for v in attr.ints]
                elif attr.HasField("i"):
                    return int(attr.i)
                elif attr.HasField("f"):
                    return float(attr.f)
                elif attr.s:
                    return attr.s.decode()
        return default

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
