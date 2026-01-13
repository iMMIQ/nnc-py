"""Basic type definitions for the IR."""

from enum import Enum


class DataType(Enum):
    """Data type enumeration."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT32 = "int32"
    INT8 = "int8"
    UINT8 = "uint8"
    BOOL = "bool"


class MemoryLayout(Enum):
    """Memory layout enumeration for tensors."""

    NCHW = "nchw"
    NHWC = "nhwc"
    OIHW = "oihw"  # Weight layout for convolutions
