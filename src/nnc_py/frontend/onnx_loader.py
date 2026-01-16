"""ONNX model loader and parser."""

import onnx
from onnx import helper, numpy_helper
try:
    from onnx import shape_inference
    HAS_SHAPE_INFERENCE = True
except ImportError:
    HAS_SHAPE_INFERENCE = False

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

        # Apply ONNX shape inference if available
        if HAS_SHAPE_INFERENCE:
            try:
                model = shape_inference.infer_shapes(model)
            except Exception as e:
                # Shape inference failed, continue with original model
                pass

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

        # 4. Build a value_info map from shape inference
        value_info_map = {}
        for vi in onnx_graph.value_info:
            value_info_map[vi.name] = vi

        # 5. Parse operator nodes
        for onnx_node in onnx_graph.node:
            ir_node = self._parse_node(onnx_node)

            # Special handling for Constant nodes - extract value from attributes
            if onnx_node.op_type == "Constant":
                self._parse_constant_node(onnx_node, ir_graph)

            ir_graph.add_node(ir_node)

            # Create tensor definitions for node outputs
            # Use shape inference results if available
            for output_name in onnx_node.output:
                if output_name not in ir_graph.tensors:
                    # Check if shape inference has info for this output
                    if output_name in value_info_map:
                        tensor_type = self._parse_tensor_type(value_info_map[output_name])
                        ir_graph.add_tensor(tensor_type)
                    else:
                        # Try to infer type from node
                        inferred_type = self._infer_tensor_type(
                            onnx_node, output_name, ir_graph
                        )
                        if inferred_type:
                            ir_graph.add_tensor(inferred_type)
                        else:
                            # If inference failed, create a placeholder tensor
                            # This ensures the graph is complete for multi-output nodes
                            ir_graph.add_tensor(TensorType(
                                dtype=DataType.FLOAT32,
                                shape=TensorShape(dims=[], layout=MemoryLayout.NHWC),
                                name=output_name,
                            ))

        # 6. Second pass: refine tensor shapes that weren't resolved by shape inference
        max_iterations = 3
        for _ in range(max_iterations):
            changed = False
            for onnx_node in onnx_graph.node:
                for output_name in onnx_node.output:
                    existing_tensor = ir_graph.tensors.get(output_name)
                    if existing_tensor and not existing_tensor.shape.dims:
                        # Try to infer shape again
                        inferred_type = self._infer_tensor_type(
                            onnx_node, output_name, ir_graph
                        )
                        if inferred_type and inferred_type.shape.dims:
                            ir_graph.tensors[output_name] = inferred_type
                            changed = True
            if not changed:
                break

        # 7. Validate graph
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
        # Use object id to ensure uniqueness, but empty string is a valid key too
        if onnx_node.name:
            name = onnx_node.name
        else:
            # Generate a truly unique name using a counter and object id
            name = f"{onnx_node.op_type}_{id(onnx_node)}_{onnx_node.op_type}"

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
            elif attr.name == "split":
                attrs["split"] = [int(d) for d in attr.ints]
            elif attr.name == "num_outputs":
                attrs["num_outputs"] = int(attr.i)

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

        elif op_type == "Sqrt":
            # Sqrt preserves shape
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

        elif op_type == "Reshape":
            # Reshape operation - shape comes from second input or attribute
            new_shape = None

            # First check if shape is in attributes
            if "shape" in onnx_node.attribute:
                for attr in onnx_node.attribute:
                    if attr.name == "shape":
                        new_shape = [int(v) for v in attr.ints]
                        break

            # If not in attributes, check second input (constant)
            if not new_shape and len(onnx_node.input) >= 2:
                shape_input_name = onnx_node.input[1]
                if shape_input_name in graph.constants:
                    new_shape = graph.constants[shape_input_name].tolist()

            # Handle -1 in shape (infer from dimension)
            if new_shape:
                resolved_shape = []
                total_size = 1
                unknown_idx = -1
                for i, dim in enumerate(new_shape):
                    if dim == -1 or dim == 0:
                        unknown_idx = i
                        resolved_shape.append(dim)  # Will resolve below
                    else:
                        resolved_shape.append(dim)
                        total_size *= dim

                # If there's a -1, compute it from total element count
                if unknown_idx >= 0:
                    input_total = 1
                    for dim in input_tensor.shape.dims:
                        input_total *= dim
                    resolved_shape[unknown_idx] = input_total // total_size if total_size > 0 else input_total

                return TensorType(
                    dtype=input_tensor.dtype,
                    shape=TensorShape(dims=resolved_shape, layout=input_tensor.shape.layout),
                    name=output_name,
                )
            else:
                # Shape not known, use input shape as fallback
                return TensorType(
                    dtype=input_tensor.dtype,
                    shape=input_tensor.shape,
                    name=output_name,
                )

        elif op_type == "Split":
            # Split divides tensor along axis into multiple outputs
            # Need to find which output index this is and compute split size
            axis = self._get_attr(onnx_node, "axis", 0)
            split_attr = self._get_attr(onnx_node, "split", None)

            # Check if input shape is available
            if not input_tensor.shape.dims:
                # Shape not yet known, return unknown shape
                return TensorType(
                    dtype=input_tensor.dtype,
                    shape=TensorShape(dims=[], layout=input_tensor.shape.layout),
                    name=output_name,
                )

            # Handle negative axis
            if axis < 0:
                axis = len(input_tensor.shape.dims) + axis

            # Validate axis
            if axis >= len(input_tensor.shape.dims):
                return None

            # Find the output index
            output_idx = -1
            for i, out_name in enumerate(onnx_node.output):
                if out_name == output_name:
                    output_idx = i
                    break

            if output_idx < 0:
                return None

            # Determine split sizes
            if split_attr:
                # Explicit split sizes provided
                split_sizes = split_attr
            else:
                # Equal split
                num_outputs = len(onnx_node.output)
                split_dim_size = input_tensor.shape.dims[axis]
                split_size = split_dim_size // num_outputs
                split_sizes = [split_size] * num_outputs

            # Compute output shape
            out_shape = list(input_tensor.shape.dims)
            if output_idx < len(split_sizes):
                out_shape[axis] = split_sizes[output_idx]
            else:
                out_shape[axis] = out_shape[axis] // len(onnx_node.output)

            return TensorType(
                dtype=input_tensor.dtype,
                shape=TensorShape(dims=out_shape, layout=input_tensor.shape.layout),
                name=output_name,
            )

        elif op_type == "MatMul":
            # Matrix multiplication shape inference
            # [M, K] @ [K, N] = [M, N]
            # [B, M, K] @ [B, K, N] = [B, M, N]
            # [B, M, K] @ [K, N] = [B, M, N] (broadcasting)
            # [M, K] @ [B, K, N] = [B, M, N] (broadcasting)

            if len(onnx_node.input) < 2:
                return None

            # Get second input (right matrix)
            right_name = onnx_node.input[1]
            right_tensor = graph.tensors.get(right_name)
            if not right_tensor or not right_tensor.shape.dims:
                return None

            a_shape = input_tensor.shape.dims
            b_shape = right_tensor.shape.dims

            if not a_shape or not b_shape:
                return None

            a_ndim = len(a_shape)
            b_ndim = len(b_shape)

            # Handle different cases
            if a_ndim == 1 and b_ndim == 1:
                # Vector @ Vector = scalar
                out_shape = []
            elif a_ndim == 1 and b_ndim == 2:
                # Vector @ Matrix = Vector
                out_shape = [b_shape[1]]
            elif a_ndim == 2 and b_ndim == 1:
                # Matrix @ Vector = Vector
                out_shape = [a_shape[0]]
            elif a_ndim == 2 and b_ndim == 2:
                # Matrix @ Matrix = Matrix
                out_shape = [a_shape[0], b_shape[1]]
            else:
                # Batched matrix multiplication
                # Compute broadcasted batch dimensions
                max_ndim = max(a_ndim, b_ndim)
                a_padded = [1] * (max_ndim - a_ndim) + list(a_shape)
                b_padded = [1] * (max_ndim - b_ndim) + list(b_shape)

                # Broadcast batch dimensions
                batch_shape = []
                for i in range(max_ndim - 2):
                    a_dim = a_padded[i]
                    b_dim = b_padded[i]
                    if a_dim == 1:
                        batch_shape.append(b_dim)
                    elif b_dim == 1 or a_dim == b_dim:
                        batch_shape.append(a_dim)
                    else:
                        return None  # Incompatible dimensions

                # Last two dimensions are [M, K] @ [K, N] = [M, N]
                out_shape = batch_shape + [a_padded[-2], b_padded[-1]]

            return TensorType(
                dtype=input_tensor.dtype,
                shape=TensorShape(dims=out_shape, layout=input_tensor.shape.layout),
                name=output_name,
            )

        elif op_type == "Add" or op_type == "Mul" or op_type == "Sub" or op_type == "Div":
            # Element-wise operations - handle broadcasting
            if len(onnx_node.input) < 2:
                return None

            right_name = onnx_node.input[1]
            right_tensor = graph.tensors.get(right_name)
            if not right_tensor or not right_tensor.shape.dims:
                # If right shape unknown, use left shape
                return TensorType(
                    dtype=input_tensor.dtype,
                    shape=input_tensor.shape,
                    name=output_name,
                )

            a_shape = input_tensor.shape.dims
            b_shape = right_tensor.shape.dims

            # Simple broadcasting: use the larger shape
            if len(a_shape) >= len(b_shape):
                out_shape = a_shape
            else:
                out_shape = b_shape

            return TensorType(
                dtype=input_tensor.dtype,
                shape=TensorShape(dims=out_shape, layout=input_tensor.shape.layout),
                name=output_name,
            )

        elif op_type == "Tile":
            # Tile operation - repeat tensor along axes
            return self._infer_tile_type(onnx_node, output_name, graph, input_tensor)

        elif op_type == "Equal":
            # Equal operation - element-wise comparison, outputs bool/uint8
            # Output shape follows broadcasting rules
            if len(onnx_node.input) < 2:
                return None

            right_name = onnx_node.input[1]
            right_tensor = graph.tensors.get(right_name)
            if not right_tensor or not right_tensor.shape.dims:
                # If right shape unknown, use left shape
                return TensorType(
                    dtype=DataType.BOOL,
                    shape=input_tensor.shape,
                    name=output_name,
                )

            a_shape = input_tensor.shape.dims
            b_shape = right_tensor.shape.dims

            # Simple broadcasting: use the larger shape
            if len(a_shape) >= len(b_shape):
                out_shape = a_shape
            else:
                out_shape = b_shape

            return TensorType(
                dtype=DataType.BOOL,
                shape=TensorShape(dims=out_shape, layout=input_tensor.shape.layout),
                name=output_name,
            )

        elif op_type == "And":
            # And operation - element-wise logical AND, outputs bool
            # Output shape follows broadcasting rules
            if len(onnx_node.input) < 2:
                return None

            right_name = onnx_node.input[1]
            right_tensor = graph.tensors.get(right_name)
            if not right_tensor or not right_tensor.shape.dims:
                # If right shape unknown, use left shape
                return TensorType(
                    dtype=DataType.BOOL,
                    shape=input_tensor.shape,
                    name=output_name,
                )

            a_shape = input_tensor.shape.dims
            b_shape = right_tensor.shape.dims

            # Simple broadcasting: use the larger shape
            if len(a_shape) >= len(b_shape):
                out_shape = a_shape
            else:
                out_shape = b_shape

            return TensorType(
                dtype=DataType.BOOL,
                shape=TensorShape(dims=out_shape, layout=input_tensor.shape.layout),
                name=output_name,
            )

        # Default: same shape as input (for other ops)
        return TensorType(
            dtype=input_tensor.dtype,
            shape=input_tensor.shape,
            name=output_name,
        )

    def _infer_tile_type(self, onnx_node, output_name: str, graph: Graph, input_tensor) -> TensorType | None:
        """Infer tensor type for Tile operation."""
        # Tile has 2 inputs: data and repeats
        if len(onnx_node.input) < 2:
            return None

        # Get repeats tensor (should be a constant 1D tensor of int64)
        repeats_name = onnx_node.input[1]
        if repeats_name not in graph.constants:
            return None

        repeats = graph.constants[repeats_name]
        if not isinstance(repeats, list) and not hasattr(repeats, 'tolist'):
            return None

        # Convert to list if needed
        if hasattr(repeats, 'tolist'):
            repeats = repeats.tolist()
        elif hasattr(repeats, 'flatten'):
            repeats = repeats.flatten().tolist()

        if not repeats:
            return None

        # Compute output shape: output[i] = input[i] * repeats[i]
        input_shape = input_tensor.shape.dims
        if not input_shape:
            return None

        # Broadcast repeats to match input rank
        input_rank = len(input_shape)
        repeats_rank = len(repeats)

        if repeats_rank < input_rank:
            # Pad repeats with 1s at the beginning
            repeats = [1] * (input_rank - repeats_rank) + repeats
        elif repeats_rank > input_rank:
            # Invalid: repeats cannot have higher rank than input
            return None

        output_shape = []
        for i in range(input_rank):
            output_shape.append(input_shape[i] * repeats[i])

        return TensorType(
            dtype=input_tensor.dtype,
            shape=TensorShape(dims=output_shape, layout=input_tensor.shape.layout),
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
                # Skip empty input names (optional inputs like Clip's min/max)
                if not input_name:
                    continue
                if input_name not in graph.tensors:
                    raise ValueError(f"Input tensor '{input_name}' not found in graph for node '{node.name}'")

        # Check all outputs are defined
        for node in graph.nodes.values():
            for output_name in node.outputs:
                if output_name not in graph.tensors:
                    raise ValueError(f"Output tensor '{output_name}' not found in graph for node '{node.name}'")
