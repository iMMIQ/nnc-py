# Memory Overflow Design for Embedded Devices

## Overview

When the total memory required by a model exceeds the available fast memory (SRAM),
tensors must be spilled to secondary slow memory (DRAM/External) and reloaded when needed.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         CompileContext              │
                    ├─────────────────────────────────────┤
                    │  max_memory: int (bytes)            │
                    │                                     │
                    │  MemoryPlan:                        │
                    │    - fast_pool: MemoryBuffer[]      │  ← Fast memory
                    │    - slow_pool: MemoryBuffer[]      │  ← Slow memory
                    │    - spill_slots: SpillSlot[]       │  ← Spill info
                    └─────────────────────────────────────┘
```

## Memory Regions

### Fast Memory (SRAM/On-chip)
```c
#define NNC_FAST_MEMORY_SIZE  <user specified>
static uint8_t _nnc_fast_pool[NNC_FAST_MEMORY_SIZE] __attribute__((aligned(16)));
```

### Slow Memory (DRAM/External)
```c
#define NNC_SLOW_MEMORY_SIZE  <calculated for total spills>
static uint8_t _nnc_slow_pool[NNC_SLOW_MEMORY_SIZE] __attribute__((aligned(16)));
```

## Spill Strategy

### Liveness Interval

For each tensor `t`:
```
t.live_start = index of node that produces t
t.live_end   = index of last node that consumes t
```

### Spill Decision

A tensor `t` is spilled if:
1. Fast memory is full when `t` needs to be allocated
2. `t` is not needed immediately (gap between uses)
3. Spilling `t` frees up enough space for the new allocation

### Spill Point

After the **last use** of a tensor in its current liveness interval:
```
For tensor with uses at nodes [n1, n2, n3]:
  - Lives in fast memory during [n1, n3]
  - Can be spilled AFTER node n3 executes
  - If needed again later, must be reloaded before that use
```

## Data Structures

### SpillSlot
```python
@dataclass
class SpillSlot:
    """Information about a spilled tensor."""
    tensor_name: str
    slow_offset: int          # Offset in slow memory pool
    size: int
    spill_after_node: str     # Spill after this node executes
    reload_before_node: str   # Reload before this node executes
    original_fast_offset: int # Original offset in fast memory
```

### TensorMemoryInfo (Extended)
```python
@dataclass
class TensorMemoryInfo:
    tensor_name: str
    buffer_id: int
    offset: int
    pool_offset: int
    size: int

    # New fields for overflow
    is_spilled: bool = False
    spill_slot: Optional[SpillSlot] = None
```

## Code Generation

### Before Execution (Reload)
```c
/* Before node execution: reload tensors if needed */
static void node_10_pre(void) {
    /* Reload tensor_conv2_out from slow memory */
    nnc_memcpy(
        _nnc_fast_pool + 1024,      // destination (fast)
        _nnc_slow_pool + 4096,      // source (slow)
        8192                        // size
    );
}

void node_10(void) {
    node_10_pre();  /* Reload before use */
    nnc_conv(...);
    node_10_post(); /* Spill after use */
}
```

### After Execution (Spill)
```c
static void node_10_post(void) {
    /* Spill tensor_relu_out to slow memory */
    nnc_memcpy(
        _nnc_slow_pool + 8192,      // destination (slow)
        _nnc_fast_pool + 2048,      // source (fast)
        4096                        // size
    );
}
```

## Algorithm

### 1. Compute Liveness
```
For each tensor t:
    t.live_start = producer_node_index
    t.live_end = max(consumer_node_indices)
```

### 2. Initial Memory Planning (Assume Unlimited Fast Memory)
```
Sort tensors by size
For each tensor:
    Find non-overlapping buffer
    Assign to buffer
```

### 3. Check Overflow
```
total_fast_size = sum of all buffer sizes
if total_fast_size > max_memory:
    need_spill = True
```

### 4. Select Spill Candidates
```
Spill priority = (tensor_size) / (lifetime_length)
                - Larger tensors spill first
                - Longer lifetime tensors spill first (they occupy memory longer)

Sort candidates by spill_priority (descending)
```

### 5. Generate Spill/Reload Points
```
For each spilled tensor:
    spill_after_node = node at live_end
    For each subsequent use:
        reload_before_node = node before use
```

### 6. Assign Slow Memory Offsets
```
slow_offset = 0
For each spilled tensor:
    tensor.slow_offset = slow_offset
    slow_offset += aligned_size
```

## Example

```
Model: Input → Conv1(48KB) → Conv2(32KB) → Conv3(24KB) → Output
       └─────────────────┬────────────────────────────────┘
                         ↓
                     Pool1(16KB)

Fast memory limit: 64KB

Without spill:
  Total: 48 + 32 + 24 + 16 = 120KB > 64KB ❌

With spill strategy:
  Conv1 output (48KB) - keep in fast (needed by Pool1)
  Conv2 output (32KB) - SPILL after use, reload before Pool1
  Conv3 output (24KB) - keep in fast (output)
  Pool1 output (16KB) - keep in fast

Fast memory at peak:
  Input + Conv1_weights + Conv1_out + Conv2_weights + Pool1_out
  = ~60KB < 64KB ✅

Spill operations:
  - After Conv2: spill Conv2_out(32KB) to slow
  - Before Pool1: reload Conv2_out(32KB) from slow
```

## Runtime API

```c
/* Memory copy between pools */
void nnc_memcpy(void* dst, const void* src, size_t n);

/* Spill tensor to slow memory */
static inline void nnc_spill_tensor(void* slow_dst, void* fast_src, size_t n) {
    nnc_memcpy(slow_dst, fast_src, n);
}

/* Reload tensor from slow memory */
static inline void nnc_reload_tensor(void* fast_dst, void* slow_src, size_t n) {
    nnc_memcpy(slow_dst, slow_src, n);
}
```

## CLI Usage

```bash
# Compile with 256KB fast memory limit
nnc compile model.onnx -o ./build -O2 --max-memory 256K

# If model needs 400KB, compiler will:
# 1. Calculate spill strategy
# 2. Generate spill/reload code
# 3. Report memory usage:
#    Fast:  256 KB (100% of limit)
#    Slow:  144 KB (spilled tensors)
#    Spill/Reload ops: 4
```

## Future: Operator Splitting

To reduce peak memory, operators can be split:
```
# Current: MatMul produces full output at once
MatMul([1024,1024], [1024,1024]) → [1024,1024]  # 4MB

# Future: Split into tiles
for i in range(0, 1024, 64):
    for j in range(0, 1024, 64):
        tile = MatMul_tile(input[i:i+64,:], weight[:,j:j+64])
        # Only need 64x64 = 16KB at a time
```

This is NOT implemented now, but the architecture should support it.
