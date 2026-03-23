# Fast-Memory Compaction Design

## Problem

`CostAwareAllocator` uses a free-list + bump-pointer model for fast memory allocation. When tensors are freed (evicted or lifetime-ended), their regions return to the free list. However, 16-byte alignment creates gaps between regions, and protected tensors (current node's inputs + already-allocated outputs) occupy fixed positions that fragment available space.

This means allocation can fail even when total free space is sufficient, because no single contiguous aligned region is large enough. The theoretical guarantee — that `max_memory >= max_node_demand` should always succeed — does not hold.

## Goal

Guarantee that the allocator succeeds whenever:

```
max_memory >= max over all nodes of: Σ align(input_i_size, 16) + Σ align(output_j_size, 16)
```

This is achieved by adding an intra-fast-memory compaction fallback that eliminates fragmentation without going through slow memory.

## Design

### Data Model

New `MovePoint` in `memory_strategy.py`, alongside `SpillPoint` and `ReloadPoint`:

```python
@dataclass
class MovePoint:
    """Intra-fast-memory relocation during compaction."""
    tensor_name: str
    at_node_idx: int          # compaction happens while processing this node
    from_offset: int          # original fast pool offset
    to_offset: int            # compacted new offset
    size: int
```

New fields on `MemoryAllocationPlan`:

```python
move_points: List[MovePoint] = field(default_factory=list)
move_bytes: int = 0           # total intra-fast-memory bytes moved
```

`total_transfer_bytes` retains its current semantics (fast-slow transfers only). `move_bytes` is tracked separately so benchmarks can distinguish slow-memory transfer cost from fast-internal compaction cost.

### Compaction Trigger

In `CostAwareAllocator.make_space`, the current failure path raises `ValueError` after eviction fails to free a contiguous region. The new logic inserts a compaction fallback before raising:

```
1. try_allocate_fast_region(size) → fails after eviction
2. try_compact_and_allocate(size, node_idx, protected)  ← new
3. if still fails → raise ValueError
```

### Compaction Algorithm (`try_compact_and_allocate`)

```
1. total_protected = Σ align(resident[t].size) for t in protected ∩ resident
2. if total_protected + align(size) > capacity → return None (genuinely insufficient)
3. sort protected tensors by current offset ascending
4. cursor = 0
5. for each protected tensor t (low-to-high offset order):
     new_offset = align(cursor, 16)
     assert new_offset <= resident[t].offset   # memmove safety invariant
     if new_offset != resident[t].offset:
         emit MovePoint(t, node_idx, old_offset, new_offset, t.size)
         move_bytes += t.size
         update resident[t].offset = new_offset
     cursor = new_offset + t.size
6. free_regions.clear(); free_regions.append((cursor, capacity - cursor))
7. next_fast_offset = cursor
   # peak_fast_end is NOT modified — it is a historical high-water mark
8. return try_allocate_fast_region(size)
```

**memmove safety**: Processing order is low-to-high original offset. The cursor starts at 0 and advances by exactly `align(size)` per tensor. Since original offsets were non-overlapping and at least that far apart (due to alignment gaps between them), `new_offset <= old_offset` always holds. Each tensor moves to a lower-or-equal address, so it never overwrites data that hasn't been moved yet. The debug assertion in step 5 guards this invariant.

**`resident` vs `tensor_allocations` updates**: Step 5 updates `resident[t].offset` which is the authoritative source — the post-loop finalization pass (lines 388-418 in the existing allocator) reads from `resident` to produce the final `tensor_allocations` entries. The `tensor_allocations` update during compaction is not strictly necessary but keeps mid-loop reads consistent; finalization will overwrite it regardless.

**`peak_fast_end` invariant**: Compaction must NOT modify `peak_fast_end`. This variable tracks the historical maximum extent of the fast pool and is used in the final `buffer_size` computation. Resetting it would produce a too-small buffer.

### Pre-check: Node Demand Validation

At allocator entry, before the event-driven walk:

```python
max_demand = 0
for node in nodes:
    demand = sum(align(tensor_sizes[t]) for t in node.inputs if t in tensor_sizes)
    demand += sum(align(tensor_sizes[t]) for t in node.outputs if t in tensor_sizes)
    max_demand = max(max_demand, demand)

if max_demand > capacity:
    raise ValueError(
        f"max_memory ({capacity}) < peak node demand ({max_demand}). "
        f"Minimum required: {max_demand} bytes."
    )
```

This gives users a clear error message upfront instead of a confusing failure mid-allocation. The check is conservative: it does not account for inplace reuse, and if a tensor name appears in both `node.inputs` and `node.outputs` it is counted twice. Real demand may be lower.

### Codegen Changes

In `x86_backend.py`, the per-node execution order becomes:

```
1. MovePoints at this node      ← new
2. ReloadPoints before this node
3. Execute operator
4. SpillPoints after this node
```

MovePoints must precede ReloadPoints because the allocator computes reload target offsets *after* compaction frees contiguous space. Reordering would place reloaded data at offsets that assume compaction already happened.

Generated C code for each MovePoint:

```c
/* Compact fast memory before node n3 */
memmove(_nnc_fast_pool + <to_offset>, _nnc_fast_pool + <from_offset>, <size>);
<var_name>.data = _nnc_fast_pool + <to_offset>;
```

Uses `memmove` (not `memcpy`) for overlap safety. After moving, updates the `Tensor.data` pointer.

New method on `MemoryAllocationPlan`:

```python
def get_move_points_at(self, node_idx: int) -> List[MovePoint]:
    return [mp for mp in self.move_points if mp.at_node_idx == node_idx]
```

Reload buffers (`_nnc_reload_buffer_N`) are unaffected — they are separate static arrays outside the fast pool.

### Correctness Guarantees

After compaction:
- Protected tensors are contiguously packed from offset 0 with no alignment gaps between them (only alignment padding at each tensor's start)
- Remaining space = `capacity - cursor` is one contiguous region
- If `capacity >= total_protected + align(size)`, allocation succeeds

Compaction does not change tensor values — `memmove` preserves data. Codegen updates `Tensor.data` immediately after each move.

Historical SpillPoint/ReloadPoint offsets are not modified — they describe past transfer operations at the offsets that were valid at that time.

## Testing

### Unit Tests

- **Fragmentation triggers compaction**: graph where protected tensors at non-contiguous offsets cause free-list allocation to fail; verify MovePoints generated, allocation succeeds
- **Extreme max_memory = max_node_demand**: verify allocation succeeds at the theoretical minimum
- **max_memory < max_node_demand**: verify pre-check raises clear ValueError
- **No compaction on normal path**: verify `move_points` is empty and `move_bytes == 0` when no fragmentation occurs
- **Post-compaction offset correctness**: verify protected tensors packed from offset 0, new tensor follows
- **move_bytes accuracy**: verify `move_bytes == Σ moved_tensor_sizes`

### Integration Tests

- Compile `create_memory_overflow_model` with extreme max_memory, verify compilation success and ASan clean
- Compare `total_transfer_bytes` with and without compaction — compaction should not increase slow-memory transfers

### Regression

All existing `test_cost_aware_allocator.py` tests must pass unchanged. Compaction is a new fallback path that does not affect normal allocation.

## Scope

### In Scope

- `MovePoint` dataclass in `memory_strategy.py`
- `move_points` / `move_bytes` fields on `MemoryAllocationPlan`
- `try_compact_and_allocate` in `CostAwareAllocator`
- Node demand pre-check in `CostAwareAllocator.allocate`
- `get_move_points_at` method on `MemoryAllocationPlan`
- Codegen support for MovePoint in `x86_backend.py`
- Unit and integration tests

### Out of Scope

- Changing the normal allocation path (free-list + bump)
- Optimizing compaction frequency (compaction is rare — only on fragmentation failure)
- Inplace-reuse-aware demand estimation in pre-check
- Multi-pool or NUMA-aware compaction
