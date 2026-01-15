# Static Memory Allocation Design for Embedded Devices

## Overview

This document describes the static memory allocation scheme for NNC-Py, designed for memory-constrained embedded devices where all memory must be statically allocated and manually managed.

## Design Goals

1. **No Dynamic Allocation**: Eliminate all `malloc`/`free` calls at runtime
2. **Deterministic Memory Usage**: Total memory requirement known at compile time
3. **Memory Reuse**: Tensors with non-overlapping lifetimes share memory
4. **Alignment Support**: Proper alignment for SIMD/NPU operations
5. **Multi-Region Support**: Separate CPU and NPU memory regions

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Memory Plan                          │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Liveness     │  │ Buffer       │  │ Memory       │  │
│  │ Analysis     │→ │ Sharing      │→ │ Layout       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    Generated C Code                     │
├─────────────────────────────────────────────────────────┤
│  static uint8_t _nnc_memory_pool[NNC_MEMORY_SIZE];      │
│  static TensorMemoryLayout _nnc_memory_layout[] = {...}; │
│  // All tensor.data pointers point into _nnc_memory_pool │
└─────────────────────────────────────────────────────────┘
```

## 1. Liveness Analysis

### Purpose
Determine the lifetime range of each tensor in the computation graph.

### Algorithm

```
For each tensor t:
    t.live_start = index of node that produces t
    t.live_end = max(index of nodes that consume t)

For each node i (in topological order):
    live_set[i] = {t | t.live_start <= i <= t.live_end}
```

### Example

```
Nodes: [Add1, Mul1, Add2, Add3]
Tensors:
    t0 (input)  : live_start=0, live_end=0  (used by Add1)
    t1 (input)  : live_start=0, live_end=1  (used by Add1, Mul1)
    t2 (Add1)   : live_start=0, live_end=1  (used by Mul1)
    t3 (Mul1)   : live_start=1, live_end=2  (used by Add2)
    t4 (input)  : live_start=2, live_end=2  (used by Add2)
    t5 (Add2)   : live_start=2, live_end=3  (used by Add3)
    t6 (Add3)   : live_start=3, live_end=3  (output)
```

## 2. Buffer Sharing (Memory Reuse)

### Purpose
Assign tensors to memory slots such that tensors with non-overlapping lifetimes share the same memory.

### Algorithm: First-Fit with Size Ordering

```
1. Sort tensors by size (descending)
2. For each tensor t:
    a. Find existing buffer with:
       - size >= t.nbytes
       - no overlap with currently live tensors
    b. If found, assign t to that buffer
    c. Else, allocate new buffer for t
```

### Data Structures

```python
@dataclass
class MemoryBuffer:
    id: int                    # Unique buffer ID
    offset: int                # Offset in memory pool
    size: int                  # Buffer size in bytes
    alignment: int             # Required alignment
    tensors: List[str]         # Tensors assigned to this buffer

@dataclass
class TensorMemoryInfo:
    tensor_name: str
    buffer_id: int             # Which buffer this tensor uses
    offset: int                # Offset within buffer
    size: int
    live_start: int            # Node index when tensor becomes live
    live_end: int              # Node index when tensor dies
```

## 3. Memory Layout

### Memory Pool Structure

```c
/* Single memory pool for all tensors */
#define NNC_MEMORY_SIZE  <calculated at compile time>
#define NNC_ALIGNMENT    16  /* For SIMD/NPU */

static uint8_t _nnc_memory_pool[NNC_MEMORY_SIZE] __attribute__((aligned(NNC_ALIGNMENT)));

/* Memory layout descriptor (optional, for debugging) */
typedef struct {
    const char* tensor_name;
    size_t offset;
    size_t size;
} TensorMemoryLayout;

static const TensorMemoryLayout _nnc_memory_layout[] = {
    {"input", 0, 4096},
    {"t2", 4096, 2048},
    ...
};
```

### Multi-Region Support (for NPU)

```c
/* CPU memory region */
#define NNC_CPU_MEMORY_SIZE  ...
static uint8_t _nnc_cpu_memory_pool[NNC_CPU_MEMORY_SIZE];

/* NPU memory region (e.g., DMEM for NPU) */
#define NNC_NPU_MEMORY_SIZE  ...
static uint8_t _nnc_npu_memory_pool[NNC_NPU_MEMORY_SIZE] __attribute__((section(".npu_memory")));
```

## 4. Generated Code Structure

### Before (Dynamic Allocation)

```c
Tensor input;
input.data = malloc(input.nbytes);  // Dynamic!
...
free(input.data);
```

### After (Static Allocation)

```c
/* Memory pool */
#define NNC_MEMORY_SIZE 65536
static uint8_t _nnc_memory_pool[NNC_MEMORY_SIZE];

/* All tensors point into memory pool */
Tensor input = {
    .data = &_nnc_memory_pool[0],      // Compile-time constant
    .dtype = NNC_DTYPE_FLOAT32,
    .shape = input_shape,
    .ndim = 4,
    .nbytes = 4096
};

Tensor t2 = {
    .data = &_nnc_memory_pool[4096],   // Reuses memory after input dies
    ...
};

/* No malloc/free at runtime! */
```

## 5. Implementation Components

### 5.1 LivenessAnalysisPass

**File**: `src/nnc_py/passes/liveness.py`

```python
class LivenessAnalysisPass(PassBase):
    @property
    def name(self) -> str:
        return "LivenessAnalysis"

    def _execute(self, ctx: CompileContext) -> None:
        nodes = ctx.graph.topological_sort()
        node_index = {node.name: i for i, node in enumerate(nodes)}

        for tensor_name, tensor in ctx.graph.tensors.items():
            producers = ctx.graph.get_producers(tensor_name)
            consumers = ctx.graph.get_consumers(tensor_name)

            if producers:
                live_start = node_index[producers[0].name]
            else:
                live_start = 0  # Input tensor

            if consumers:
                live_end = max(node_index[c.name] for c in consumers)
            else:
                live_end = len(nodes) - 1  # Output tensor

            # Store liveness info in context
            ctx.tensor_liveness[tensor_name] = (live_start, live_end)
```

### 5.2 MemoryPlanningPass

**File**: `src/nnc_py/passes/memory_plan.py`

```python
class MemoryPlanningPass(PassBase):
    @property
    def name(self) -> str:
        return "MemoryPlanning"

    def _execute(self, ctx: CompileContext) -> None:
        # 1. Collect all tensors with their sizes and liveness
        tensor_infos = self._collect_tensor_info(ctx)

        # 2. Sort by size (descending) for better packing
        tensor_infos.sort(key=lambda x: x.size, reverse=True)

        # 3. Allocate buffers with reuse
        buffers = self._allocate_buffers(tensor_infos)

        # 4. Store memory plan in context
        ctx.memory_plan = MemoryPlan(buffers)
```

### 5.3 Updated CEmitter

**File**: `src/nnc_py/codegen/c_emitter.py`

Add method to emit static memory pool:

```python
def _emit_memory_pool(self, ctx: CompileContext):
    """Emit static memory pool declaration."""
    plan = ctx.memory_plan
    total_size = plan.total_size()

    self.write_line(f"/* Static memory pool */")
    self.write_line(f"#define NNC_MEMORY_SIZE {total_size}")
    self.write_line(f"static uint8_t _nnc_memory_pool[NNC_MEMORY_SIZE] __attribute__((aligned(16)));")
    self.write_line()
```

## 6. Optimization Opportunities

### 6.1 In-place Operations

Some operations can modify input in place:
- `ReLU(x)` → output can reuse `x`'s memory
- `Add(x, y)` → if `x` dies after this, output can reuse `x`

### 6.2 Transpose Elimination

Consecutive transposes may cancel out, avoiding the need for intermediate buffers.

### 6.3 Shape Optimization

Reshape operations that don't change the total element count can share memory.

## 7. Memory Usage Calculation

### Compile-Time Reporting

The compiler should report:

```
Memory Planning Summary:
  Total tensors: 15
  Shared buffers: 8
  Total memory: 24576 bytes (23.99 KB)
  Peak memory: 49152 bytes (48.00 KB)
  Savings vs no-sharing: 49.2%
```

### Per-Tensor Breakdown

```
Tensor Memory Layout:
  +------------------+-------+-------+--------+
  | Tensor           | Size  | Offset| Buffer |
  +------------------+-------+-------+--------+
  | input            | 4096  | 0     | #0     |
  | conv1_weight     | 8192  | 4096  | #1     |
  | conv1_bias       | 256   | 12288 | #2     |
  | conv1_out        | 8192  | 0     | #0     | ← Reuses input
  | conv2_out        | 4096  | 12544 | #3     |
  +------------------+-------+-------+--------+
```

## 8. Integration with Optimization Levels

- **O0**: No memory planning (each tensor gets own buffer)
- **O1**: Basic liveness analysis and buffer sharing
- **O2**: In-place operation optimization
- **O3**: Advanced memory planning with fragmentation minimization

## 9. NPU-Specific Considerations

### NPU Memory Regions

```c
/* Different memory types for NPU */
typedef enum {
    NNC_MEM_CPU     = 0,  // Main memory (CPU accessible)
    NNC_MEM_NPU_DMEM = 1, // NPU data memory (fast, limited)
    NNC_MEM_NPU_IMEM = 2, // NPU instruction memory
} nnc_memory_region_t;
```

### Data Transfer

For operations that need data in NPU memory:
```c
/* CPU → NPU transfer */
nnc_npu_load(_nnc_memory_pool + offset, npu_dmem_addr, size);

/* NPU → CPU transfer */
nnc_npu_store(npu_dmem_addr, _nnc_memory_pool + offset, size);
```
