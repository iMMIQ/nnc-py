"""Tensor type system for the IR."""

from dataclasses import dataclass
from typing import List, Union

from nnc_py.ir.types import DataType, MemoryLayout


@dataclass
class TensorShape:
    """Tensor shape with layout information."""

    dims: List[Union[int, str]]  # int for static, str for symbolic dimensions
    layout: MemoryLayout = MemoryLayout.NCHW

    def is_static(self) -> bool:
        """Check if all dimensions are static (integers)."""
        return all(isinstance(d, int) for d in self.dims)

    def rank(self) -> int:
        """Return the number of dimensions."""
        return len(self.dims)

    def __str__(self) -> str:
        dims_str = "x".join(str(d) for d in self.dims)
        return f"[{dims_str}]"


@dataclass
class TensorType:
    """Tensor type definition."""

    dtype: DataType
    shape: TensorShape
    name: str

    def byte_size(self) -> int:
        """Calculate the total byte size of the tensor."""
        elem_size = {
            DataType.FLOAT32: 4,
            DataType.FLOAT16: 2,
            DataType.INT32: 4,
            DataType.INT64: 8,
            DataType.INT8: 1,
            DataType.UINT8: 1,
            DataType.BOOL: 1,
        }
        size = 1
        for dim in self.shape.dims:
            if isinstance(dim, int):
                size *= dim
            else:
                # Symbolic dimension - return -1 to indicate unknown size
                return -1
        return size * elem_size[self.dtype]

    def __repr__(self) -> str:
        return f"TensorType({self.name}: {self.dtype.value}{self.shape})"
