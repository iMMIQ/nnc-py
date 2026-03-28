"""Model source emitter for x86 codegen packages."""

from __future__ import annotations

from typing import Any

from nnc_py.codegen.c_emitter import CEmitter
from nnc_py.codegen.x86_ir import X86CodegenPackage
from nnc_py.passes.memory_planning import get_memory_allocation_plan
from nnc_py.passes.spill import get_spill_plan


# ---------------------------------------------------------------------------
# Pure helper functions (migrated from X86Backend)
# ---------------------------------------------------------------------------

def _sanitize_c_comment_text(value: str) -> str:
    """Avoid terminating generated C comments accidentally."""
    return value.replace("*/", "* /").replace("\n", " ")


def _has_parallel_runtime(pipeline_codegen_metadata: dict[str, Any]) -> bool:
    runtime = pipeline_codegen_metadata.get("parallel_runtime")
    return bool(isinstance(runtime, dict) and runtime.get("enabled") is True)


def _get_public_entry_point(ctx: Any) -> str:
    entry_point = ctx.metadata.get("entry_point", "nnc_run")
    if not isinstance(entry_point, str) or not entry_point:
        return "nnc_run"
    return entry_point


# ---------------------------------------------------------------------------
# Parallel runtime rendering (migrated from X86Backend)
# ---------------------------------------------------------------------------

def _render_parallel_runtime_block(
pipeline_codegen_metadata: dict[str, Any],
) -> list[str]:
    """Render C helper code for dependency-driven 4-resource parallel execution."""
    runtime = pipeline_codegen_metadata.get("parallel_runtime")
    if not isinstance(runtime, dict) or runtime.get("enabled") is not True:
        return []

    steps = tuple(runtime.get("steps", ()))
    predecessor_indices = tuple(runtime.get("predecessor_indices", ()))
    successor_indices = tuple(runtime.get("successor_indices", ()))
    step_count = len(steps)
    if step_count == 0:
        return []

    dep_counts: list[int] = []
    for deps in predecessor_indices:
        deps_list = [int(dep) for dep in deps]
        dep_counts.append(len(deps_list))

    succ_offsets: list[int] = []
    succ_counts: list[int] = []
    succ_flat: list[int] = []
    succ_offset = 0
    for succs in successor_indices:
        succ_list = [int(succ) for succ in succs]
        succ_offsets.append(succ_offset)
        succ_counts.append(len(succ_list))
        succ_flat.extend(succ_list)
        succ_offset += len(succ_list)
    if not succ_flat:
        succ_flat = [-1]

    lines = [
        "/* Pipeline parallel runtime */",
        "#define NNC_PIPELINE_WORKER_COUNT 4",
        f"#define NNC_PIPELINE_STEP_COUNT {step_count}",
        f"#define NNC_PIPELINE_SUCC_INDEX_COUNT {len(succ_flat)}",
        "",
        "enum {",
        "    NNC_PIPE_RES_DMA = 0,",
        "    NNC_PIPE_RES_SHAPE = 1,",
        "    NNC_PIPE_RES_MATMUL = 2,",
        "    NNC_PIPE_RES_OTHER = 3,",
        "};",
        "",
        "typedef void (*NncPipelineStepFn)(void);",
        "typedef struct NncPipelineStepDesc {",
        "    int resource_kind;",
        "    int start_time;",
        "    int end_time;",
        "    NncPipelineStepFn invoke;",
        "} NncPipelineStepDesc;",
        "",
        "typedef struct NncPipelineWorkerArg {",
        "    int resource_kind;",
        "    int worker_index;",
        "} NncPipelineWorkerArg;",
        "",
        "static const NncPipelineStepDesc _nnc_pipeline_steps[NNC_PIPELINE_STEP_COUNT] = {",
    ]
    for step in steps:
        step_id = _sanitize_c_comment_text(str(step["step_id"]))
        invoke_symbol = step["invoke_symbol"] if step["invoke_symbol"] is not None else "NULL"
        lines.append(
            f"    /* {step_id} */ {{{int(step['resource_index'])}, {int(step['start_time'])}, {int(step['end_time'])}, {invoke_symbol}}},"
        )
    lines.extend(
        [
            "};",
            "",
            "static const int _nnc_pipeline_dep_counts[NNC_PIPELINE_STEP_COUNT] = {"
            + ", ".join(str(value) for value in dep_counts)
            + "};",
            "static const int _nnc_pipeline_succ_offsets[NNC_PIPELINE_STEP_COUNT] = {"
            + ", ".join(str(value) for value in succ_offsets)
            + "};",
            "static const int _nnc_pipeline_succ_counts[NNC_PIPELINE_STEP_COUNT] = {"
            + ", ".join(str(value) for value in succ_counts)
            + "};",
            "static const int _nnc_pipeline_succ_indices[NNC_PIPELINE_SUCC_INDEX_COUNT] = {"
            + ", ".join(str(value) for value in succ_flat)
            + "};",
            "",
            "static int _nnc_pipeline_remaining_deps[NNC_PIPELINE_STEP_COUNT];",
            "static unsigned char _nnc_pipeline_started[NNC_PIPELINE_STEP_COUNT];",
            "static int _nnc_pipeline_completed_count = 0;",
            "static int _nnc_pipeline_start_flag = 0;",
            "static pthread_mutex_t _nnc_pipeline_mutex = PTHREAD_MUTEX_INITIALIZER;",
            "static pthread_cond_t _nnc_pipeline_cond = PTHREAD_COND_INITIALIZER;",
            "",
            "static void _nnc_pipeline_bind_worker_core(int worker_index) {",
            "#if defined(__linux__)",
            "    long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);",
            "    int cpu_index = worker_index;",
            "    if (cpu_count > 0) {",
            "        cpu_index = worker_index % (int)cpu_count;",
            "    }",
            "    cpu_set_t cpu_set;",
            "    CPU_ZERO(&cpu_set);",
            "    CPU_SET(cpu_index, &cpu_set);",
            "    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);",
            "#else",
            "    (void)worker_index;",
            "#endif",
            "}",
            "",
            "static int _nnc_pipeline_pick_ready_step_locked(int resource_kind) {",
            "    for (int step_index = 0; step_index < NNC_PIPELINE_STEP_COUNT; ++step_index) {",
            "        if (_nnc_pipeline_steps[step_index].resource_kind != resource_kind) {",
            "            continue;",
            "        }",
            "        if (_nnc_pipeline_started[step_index] != 0) {",
            "            continue;",
            "        }",
            "        if (_nnc_pipeline_remaining_deps[step_index] != 0) {",
            "            continue;",
            "        }",
            "        _nnc_pipeline_started[step_index] = 1;",
            "        return step_index;",
            "    }",
            "    return -1;",
            "}",
            "",
            "static void* _nnc_pipeline_worker_main(void* worker_arg_ptr) {",
            "    NncPipelineWorkerArg* worker_arg = (NncPipelineWorkerArg*)worker_arg_ptr;",
            "    _nnc_pipeline_bind_worker_core(worker_arg->worker_index);",
            "",
            "    pthread_mutex_lock(&_nnc_pipeline_mutex);",
            "    while (_nnc_pipeline_start_flag == 0 && _nnc_pipeline_completed_count < NNC_PIPELINE_STEP_COUNT) {",
            "        pthread_cond_wait(&_nnc_pipeline_cond, &_nnc_pipeline_mutex);",
            "    }",
            "    if (_nnc_pipeline_completed_count >= NNC_PIPELINE_STEP_COUNT) {",
            "        pthread_mutex_unlock(&_nnc_pipeline_mutex);",
            "        return NULL;",
            "    }",
            "    pthread_mutex_unlock(&_nnc_pipeline_mutex);",
            "",
            "    for (;;) {",
            "        int step_index = -1;",
            "",
            "        pthread_mutex_lock(&_nnc_pipeline_mutex);",
            "        while (step_index < 0) {",
            "            if (_nnc_pipeline_completed_count >= NNC_PIPELINE_STEP_COUNT) {",
            "                pthread_mutex_unlock(&_nnc_pipeline_mutex);",
            "                return NULL;",
            "            }",
            "            step_index = _nnc_pipeline_pick_ready_step_locked(worker_arg->resource_kind);",
            "            if (step_index < 0) {",
            "                pthread_cond_wait(&_nnc_pipeline_cond, &_nnc_pipeline_mutex);",
            "            }",
            "        }",
            "        pthread_mutex_unlock(&_nnc_pipeline_mutex);",
            "",
            "        NncPipelineStepFn invoke = _nnc_pipeline_steps[step_index].invoke;",
            "        if (invoke != NULL) {",
            "            invoke();",
            "        }",
            "",
            "        pthread_mutex_lock(&_nnc_pipeline_mutex);",
            "        int succ_offset = _nnc_pipeline_succ_offsets[step_index];",
            "        int succ_count = _nnc_pipeline_succ_counts[step_index];",
            "        for (int succ_i = 0; succ_i < succ_count; ++succ_i) {",
            "            int succ_index = _nnc_pipeline_succ_indices[succ_offset + succ_i];",
            "            if (_nnc_pipeline_remaining_deps[succ_index] > 0) {",
            "                _nnc_pipeline_remaining_deps[succ_index] -= 1;",
            "            }",
            "        }",
            "        _nnc_pipeline_completed_count += 1;",
            "        pthread_cond_broadcast(&_nnc_pipeline_cond);",
            "        pthread_mutex_unlock(&_nnc_pipeline_mutex);",
            "    }",
            "}",
            "",
            "static void nnc_pipeline_run_parallel(void) {",
            "    pthread_t workers[NNC_PIPELINE_WORKER_COUNT];",
            "    NncPipelineWorkerArg worker_args[NNC_PIPELINE_WORKER_COUNT] = {",
            "        {NNC_PIPE_RES_DMA, 0},",
            "        {NNC_PIPE_RES_SHAPE, 1},",
            "        {NNC_PIPE_RES_MATMUL, 2},",
            "        {NNC_PIPE_RES_OTHER, 3},",
            "    };",
            "",
            "    pthread_mutex_lock(&_nnc_pipeline_mutex);",
            "    _nnc_pipeline_completed_count = 0;",
            "    _nnc_pipeline_start_flag = 0;",
            "    for (int step_index = 0; step_index < NNC_PIPELINE_STEP_COUNT; ++step_index) {",
            "        _nnc_pipeline_remaining_deps[step_index] = _nnc_pipeline_dep_counts[step_index];",
            "        _nnc_pipeline_started[step_index] = 0;",
            "    }",
            "    pthread_mutex_unlock(&_nnc_pipeline_mutex);",
            "",
            "    int launched_count = 0;",
            "    for (int worker_index = 0; worker_index < NNC_PIPELINE_WORKER_COUNT; ++worker_index) {",
            "        if (pthread_create(",
            "                &workers[worker_index],",
            "                NULL,",
            "                _nnc_pipeline_worker_main,",
            "                &worker_args[worker_index]",
            "            ) != 0) {",
            "            break;",
            "        }",
            "        launched_count += 1;",
            "    }",
            "",
            "    if (launched_count != NNC_PIPELINE_WORKER_COUNT) {",
            "        pthread_mutex_lock(&_nnc_pipeline_mutex);",
            "        _nnc_pipeline_completed_count = NNC_PIPELINE_STEP_COUNT;",
            "        pthread_cond_broadcast(&_nnc_pipeline_cond);",
            "        pthread_mutex_unlock(&_nnc_pipeline_mutex);",
            "        for (int worker_index = 0; worker_index < launched_count; ++worker_index) {",
            "            pthread_join(workers[worker_index], NULL);",
            "        }",
            "        for (int step_index = 0; step_index < NNC_PIPELINE_STEP_COUNT; ++step_index) {",
            "            NncPipelineStepFn invoke = _nnc_pipeline_steps[step_index].invoke;",
            "            if (invoke != NULL) {",
            "                invoke();",
            "            }",
            "        }",
            "        return;",
            "    }",
            "",
            "    pthread_mutex_lock(&_nnc_pipeline_mutex);",
            "    _nnc_pipeline_start_flag = 1;",
            "    pthread_cond_broadcast(&_nnc_pipeline_cond);",
            "    pthread_mutex_unlock(&_nnc_pipeline_mutex);",
            "",
            "    for (int worker_index = 0; worker_index < NNC_PIPELINE_WORKER_COUNT; ++worker_index) {",
            "        pthread_join(workers[worker_index], NULL);",
            "    }",
            "}",
        ]
    )
    return lines

def _render_parallel_step_helper_block(
pipeline_codegen_metadata: dict[str, Any],
) -> list[str]:
    """Render per-step helper functions so each worker runs concrete C code."""
    runtime = pipeline_codegen_metadata.get("parallel_runtime")
    if not isinstance(runtime, dict) or runtime.get("enabled") is not True:
        return []

    steps = tuple(runtime.get("steps", ()))
    if not steps:
        return []

    declared_value_records: dict[str, dict[str, Any]] = {}
    used_buffer_symbols: set[str] = set()
    used_saved_data_symbols: set[str] = set()

    def remember_record(record: dict[str, Any]) -> None:
        declared_value_records.setdefault(str(record["value_name"]), dict(record))

    for step in steps:
        if step.get("custom_body_lines") is not None:
            continue
        input_value_records = tuple(step.get("input_value_records", ()))
        output_value_records = tuple(step.get("output_value_records", ()))
        step_kind = str(step.get("step_kind", ""))

        if step_kind == "dma_in":
            staged_outputs = [
                record
                for record in output_value_records
                if bool(record.get("is_staged")) or record.get("fast_expr")
            ]
            direct_inputs_by_tensor = {
                str(record["graph_tensor_name"]): record
                for record in input_value_records
                if not bool(record.get("is_staged"))
            }
            for record in staged_outputs:
                if not bool(record.get("is_staged")):
                    continue
                source_record = direct_inputs_by_tensor.get(str(record["graph_tensor_name"]))
                if source_record is None:
                    continue
                remember_record(record)
                used_buffer_symbols.add(f"{record['storage_symbol']}_buffer")
                used_saved_data_symbols.add(str(record["saved_data_symbol"]))
        elif step_kind == "dma_out":
            bridged_outputs = [
                record
                for record in output_value_records
                if record.get("fast_expr")
            ]
            staged_inputs_by_tensor = {
                str(record["graph_tensor_name"]): record
                for record in input_value_records
                if bool(record.get("is_staged")) or record.get("fast_expr")
            }
            for record in bridged_outputs:
                source_record = staged_inputs_by_tensor.get(str(record["graph_tensor_name"]))
                if source_record is None or not bool(source_record.get("is_staged")):
                    continue
                remember_record(source_record)
                used_buffer_symbols.add(f"{source_record['storage_symbol']}_buffer")
        elif step_kind == "compute":
            buffered_inputs = [
                record for record in input_value_records if bool(record.get("is_staged"))
            ]
            fast_inputs = [
                record
                for record in input_value_records
                if not bool(record.get("is_staged")) and record.get("fast_expr")
            ]
            buffered_outputs = [
                record for record in output_value_records if bool(record.get("is_staged"))
            ]
            for record in buffered_inputs:
                remember_record(record)
                used_buffer_symbols.add(f"{record['storage_symbol']}_buffer")
            for record in fast_inputs:
                if not bool(record.get("needs_restore")):
                    continue
                remember_record(record)
                used_saved_data_symbols.add(str(record["saved_data_symbol"]))
            for record in buffered_outputs:
                remember_record(record)
                used_buffer_symbols.add(f"{record['storage_symbol']}_buffer")
                used_saved_data_symbols.add(str(record["saved_data_symbol"]))

    lines = [
        "/* Pipeline step helper functions */",
        "static volatile uint64_t _nnc_pipeline_touch_sink = 0;",
        "",
        "static int64_t _nnc_min_i64(int64_t lhs, int64_t rhs) {",
        "    return lhs < rhs ? lhs : rhs;",
        "}",
        "",
        "static int64_t _nnc_pipeline_dtype_size(nnc_dtype_t dtype) {",
        "    switch (dtype) {",
        "        case NNC_DTYPE_FLOAT16: return 2;",
        "        case NNC_DTYPE_INT8: return 1;",
        "        case NNC_DTYPE_UINT8: return 1;",
        "        case NNC_DTYPE_BOOL: return 1;",
        "        case NNC_DTYPE_INT64: return 8;",
        "        case NNC_DTYPE_FLOAT32:",
        "        case NNC_DTYPE_INT32:",
        "        default: return 4;",
        "    }",
        "}",
        "",
        "static int64_t _nnc_pipeline_tile_nbytes(const Tensor* tensor, int64_t tile_h, int64_t tile_w) {",
        "    if (tensor == NULL || tensor->shape == NULL || tensor->ndim != 4) {",
        "        return 0;",
        "    }",
        "    return _nnc_pipeline_dtype_size(tensor->dtype) * tensor->shape[1] * tile_h * tile_w;",
        "}",
        "",
        "static void _nnc_pipeline_stage_nchw_tile_with_origin(",
        "    const Tensor* src,",
        "    void* dst,",
        "    int64_t batch_index,",
        "    int64_t h_origin,",
        "    int64_t w_origin,",
        "    int64_t tile_h,",
        "    int64_t tile_w",
        ") {",
        "    if (src == NULL || dst == NULL || src->data == NULL || src->shape == NULL || src->ndim != 4) {",
        "        return;",
        "    }",
        "    int64_t elem_size = _nnc_pipeline_dtype_size(src->dtype);",
        "    int64_t channels = src->shape[1];",
        "    int64_t full_h = src->shape[2];",
        "    int64_t full_w = src->shape[3];",
        "    uint8_t* dst_bytes = (uint8_t*)dst;",
        "    const uint8_t* src_bytes = (const uint8_t*)src->data;",
        "    memset(dst_bytes, 0, (size_t)(elem_size * channels * tile_h * tile_w));",
        "    for (int64_t channel = 0; channel < channels; ++channel) {",
        "        for (int64_t tile_row = 0; tile_row < tile_h; ++tile_row) {",
        "            int64_t src_row = h_origin + tile_row;",
        "            if (src_row < 0 || src_row >= full_h) {",
        "                continue;",
        "            }",
        "            for (int64_t tile_col = 0; tile_col < tile_w; ++tile_col) {",
        "                int64_t src_col = w_origin + tile_col;",
        "                if (src_col < 0 || src_col >= full_w) {",
        "                    continue;",
        "                }",
        "                int64_t src_offset = ((((batch_index * channels) + channel) * full_h + src_row) * full_w + src_col) * elem_size;",
        "                int64_t dst_offset = (((channel * tile_h) + tile_row) * tile_w + tile_col) * elem_size;",
        "                memcpy(dst_bytes + dst_offset, src_bytes + src_offset, (size_t)elem_size);",
        "            }",
        "        }",
        "    }",
        "}",
        "",
        "static void _nnc_pipeline_stage_nchw_tile_with_origin_maxpool(",
        "    const Tensor* src,",
        "    void* dst,",
        "    int64_t batch_index,",
        "    int64_t h_origin,",
        "    int64_t w_origin,",
        "    int64_t tile_h,",
        "    int64_t tile_w",
        ") {",
        "    if (src == NULL || dst == NULL || src->data == NULL || src->shape == NULL || src->ndim != 4) {",
        "        return;",
        "    }",
        "    int64_t elem_size = _nnc_pipeline_dtype_size(src->dtype);",
        "    int64_t channels = src->shape[1];",
        "    int64_t full_h = src->shape[2];",
        "    int64_t full_w = src->shape[3];",
        "    uint8_t* dst_bytes = (uint8_t*)dst;",
        "    const uint8_t* src_bytes = (const uint8_t*)src->data;",
        "    int64_t total_elems = channels * tile_h * tile_w;",
        "    if (src->dtype == NNC_DTYPE_FLOAT32) {",
        "        float* dst_f32 = (float*)dst;",
        "        for (int64_t index = 0; index < total_elems; ++index) {",
        "            dst_f32[index] = -3.402823466e+38F;",
        "        }",
        "    } else {",
        "        memset(dst_bytes, 0, (size_t)(elem_size * total_elems));",
        "    }",
        "    for (int64_t channel = 0; channel < channels; ++channel) {",
        "        for (int64_t tile_row = 0; tile_row < tile_h; ++tile_row) {",
        "            int64_t src_row = h_origin + tile_row;",
        "            if (src_row < 0 || src_row >= full_h) {",
        "                continue;",
        "            }",
        "            for (int64_t tile_col = 0; tile_col < tile_w; ++tile_col) {",
        "                int64_t src_col = w_origin + tile_col;",
        "                if (src_col < 0 || src_col >= full_w) {",
        "                    continue;",
        "                }",
        "                int64_t src_offset = ((((batch_index * channels) + channel) * full_h + src_row) * full_w + src_col) * elem_size;",
        "                int64_t dst_offset = (((channel * tile_h) + tile_row) * tile_w + tile_col) * elem_size;",
        "                memcpy(dst_bytes + dst_offset, src_bytes + src_offset, (size_t)elem_size);",
        "            }",
        "        }",
        "    }",
        "}",
        "",
        "static void _nnc_pipeline_stage_nchw_tile(",
        "    const Tensor* src,",
        "    void* dst,",
        "    int64_t batch_index,",
        "    int64_t h_origin,",
        "    int64_t w_origin,",
        "    int64_t tile_h,",
        "    int64_t tile_w",
        ") {",
        "    _nnc_pipeline_stage_nchw_tile_with_origin(src, dst, batch_index, h_origin, w_origin, tile_h, tile_w);",
        "}",
        "",
        "static void _nnc_pipeline_commit_nchw_tile(",
        "    const void* src,",
        "    Tensor* dst,",
        "    int64_t batch_index,",
        "    int64_t h_origin,",
        "    int64_t w_origin,",
        "    int64_t tile_h,",
        "    int64_t tile_w",
        ") {",
        "    if (src == NULL || dst == NULL || dst->data == NULL || dst->shape == NULL || dst->ndim != 4) {",
        "        return;",
        "    }",
        "    int64_t elem_size = _nnc_pipeline_dtype_size(dst->dtype);",
        "    int64_t channels = dst->shape[1];",
        "    int64_t full_h = dst->shape[2];",
        "    int64_t full_w = dst->shape[3];",
        "    const uint8_t* src_bytes = (const uint8_t*)src;",
        "    uint8_t* dst_bytes = (uint8_t*)dst->data;",
        "    for (int64_t channel = 0; channel < channels; ++channel) {",
        "        for (int64_t tile_row = 0; tile_row < tile_h; ++tile_row) {",
        "            int64_t dst_row = h_origin + tile_row;",
        "            if (dst_row < 0 || dst_row >= full_h) {",
        "                continue;",
        "            }",
        "            for (int64_t tile_col = 0; tile_col < tile_w; ++tile_col) {",
        "                int64_t dst_col = w_origin + tile_col;",
        "                if (dst_col < 0 || dst_col >= full_w) {",
        "                    continue;",
        "                }",
        "                int64_t src_offset = (((channel * tile_h) + tile_row) * tile_w + tile_col) * elem_size;",
        "                int64_t dst_offset = ((((batch_index * channels) + channel) * full_h + dst_row) * full_w + dst_col) * elem_size;",
        "                memcpy(dst_bytes + dst_offset, src_bytes + src_offset, (size_t)elem_size);",
        "            }",
        "        }",
        "    }",
        "}",
        "",
        "static void _nnc_pipeline_touch_tensor_read(const Tensor* tensor) {",
        "    if (tensor == NULL || tensor->data == NULL || tensor->nbytes <= 0) {",
        "        return;",
        "    }",
        "    const unsigned char* bytes = (const unsigned char*)tensor->data;",
        "    uint64_t accum = 0;",
        "    for (int64_t index = 0; index < tensor->nbytes; ++index) {",
        "        accum += (uint64_t)bytes[index];",
        "    }",
        "    _nnc_pipeline_touch_sink ^= accum;",
        "}",
        "",
        "static void _nnc_pipeline_touch_tensor_write(Tensor* tensor) {",
        "    if (tensor == NULL || tensor->data == NULL || tensor->nbytes <= 0) {",
        "        return;",
        "    }",
        "    volatile unsigned char* bytes = (volatile unsigned char*)tensor->data;",
        "    for (int64_t index = 0; index < tensor->nbytes; ++index) {",
        "        bytes[index] = bytes[index];",
        "    }",
        "}",
        "",
        "static void _nnc_pipeline_shape_touch_tensor(const Tensor* tensor) {",
        "    if (tensor == NULL || tensor->shape == NULL || tensor->ndim <= 0) {",
        "        return;",
        "    }",
        "    int64_t shape_accum = 0;",
        "    for (int32_t dim = 0; dim < tensor->ndim; ++dim) {",
        "        shape_accum += tensor->shape[dim] * (dim + 1);",
        "    }",
        "    _nnc_pipeline_touch_sink ^= (uint64_t)shape_accum;",
        "}",
        "",
    ]

    for value_name in sorted(declared_value_records):
        record = declared_value_records[value_name]
        storage_symbol = str(record["storage_symbol"])
        saved_data_symbol = str(record["saved_data_symbol"])
        size_bytes = int(record["size_bytes"])
        if f"{storage_symbol}_buffer" in used_buffer_symbols:
            lines.append(f"static unsigned char {storage_symbol}_buffer[{size_bytes}];")
        if saved_data_symbol in used_saved_data_symbols:
            lines.append(f"static void* {saved_data_symbol} = NULL;")
    if declared_value_records:
        lines.append("")

    for declaration in tuple(runtime.get("custom_declarations", ())):
        lines.append(str(declaration))
    if runtime.get("custom_declarations"):
        lines.append("")

    for step in steps:
        invoke_symbol = step.get("invoke_symbol")
        if not isinstance(invoke_symbol, str) or not invoke_symbol:
            continue
        input_tensor_symbols = tuple(step.get("input_tensor_symbols", ()))
        output_tensor_symbols = tuple(step.get("output_tensor_symbols", ()))
        input_value_records = tuple(step.get("input_value_records", ()))
        output_value_records = tuple(step.get("output_value_records", ()))
        step_kind = str(step.get("step_kind", ""))
        invoke_node = bool(step.get("invoke_node"))
        node_symbol = str(step.get("node_symbol", ""))

        lines.append(f"static void {invoke_symbol}(void) {{")
        custom_body_lines = step.get("custom_body_lines")
        if custom_body_lines is not None:
            for body_line in tuple(custom_body_lines):
                if body_line:
                    lines.append(f"    {body_line}")
                else:
                    lines.append("")
        elif step_kind == "dma_in":
            staged_outputs = [
                record
                for record in output_value_records
                if bool(record.get("is_staged")) or record.get("fast_expr")
            ]
            direct_inputs_by_tensor = {
                str(record["graph_tensor_name"]): record
                for record in input_value_records
                if not bool(record.get("is_staged"))
            }
            if staged_outputs:
                for record in staged_outputs:
                    tensor_symbol = str(record["tensor_symbol"])
                    storage_symbol = str(record["storage_symbol"])
                    size_bytes = int(record["size_bytes"])
                    source_record = direct_inputs_by_tensor.get(str(record["graph_tensor_name"]))
                    fast_expr = record.get("fast_expr")
                    if bool(record.get("is_staged")):
                        if source_record is not None:
                            lines.append(
                                f"    {record['saved_data_symbol']} = {tensor_symbol}.data;"
                            )
                            lines.append(
                                f"    memcpy({storage_symbol}_buffer, {tensor_symbol}.data, {size_bytes});"
                            )
                            lines.append(
                                f"    {tensor_symbol}.data = {storage_symbol}_buffer;"
                            )
                        else:
                            lines.append(
                                f"    _nnc_pipeline_touch_tensor_read(&{tensor_symbol});"
                            )
                    elif fast_expr:
                        source_fast_expr = (
                            source_record.get("fast_expr")
                            if source_record is not None
                            else None
                        )
                        if source_record is not None:
                            if source_fast_expr:
                                if source_fast_expr != fast_expr:
                                    lines.append(
                                        f"    memcpy({fast_expr}, {source_fast_expr}, {size_bytes});"
                                    )
                            else:
                                lines.append(
                                    f"    memcpy({fast_expr}, {tensor_symbol}.data, {size_bytes});"
                                )
                    else:
                        lines.append(
                            f"    _nnc_pipeline_touch_tensor_read(&{tensor_symbol});"
                        )
            elif input_tensor_symbols:
                for symbol in input_tensor_symbols:
                    lines.append(f"    _nnc_pipeline_touch_tensor_read(&{symbol});")
            else:
                lines.append("    _nnc_pipeline_touch_sink ^= 1u;")
        elif step_kind == "dma_out":
            bridged_outputs = [
                record
                for record in output_value_records
                if record.get("fast_expr")
            ]
            staged_inputs = [
                record
                for record in input_value_records
                if bool(record.get("is_staged")) or record.get("fast_expr")
            ]
            if bridged_outputs:
                staged_inputs_by_tensor = {
                    str(record["graph_tensor_name"]): record
                    for record in staged_inputs
                }
                for record in bridged_outputs:
                    tensor_symbol = str(record["tensor_symbol"])
                    size_bytes = int(record["size_bytes"])
                    target_fast_expr = str(record["fast_expr"])
                    source_record = staged_inputs_by_tensor.get(str(record["graph_tensor_name"]))
                    if source_record is not None and bool(source_record.get("is_staged")):
                        source_storage_symbol = str(source_record["storage_symbol"])
                        lines.append(
                            f"    memcpy({target_fast_expr}, {source_storage_symbol}_buffer, {size_bytes});"
                        )
                    else:
                        source_fast_expr = (
                            source_record.get("fast_expr")
                            if source_record is not None
                            else None
                        )
                        if source_fast_expr and source_fast_expr != target_fast_expr:
                            lines.append(
                                f"    memcpy({target_fast_expr}, {source_fast_expr}, {size_bytes});"
                            )
                    lines.append(f"    {tensor_symbol}.data = {target_fast_expr};")
            elif staged_inputs:
                fast_staged_inputs = [
                    record
                    for record in staged_inputs
                    if not bool(record.get("is_staged")) and record.get("fast_expr")
                ]
                for record in fast_staged_inputs:
                    tensor_symbol = str(record["tensor_symbol"])
                    lines.append(f"    {tensor_symbol}.data = {record['fast_expr']};")
            else:
                dma_out_symbols = output_tensor_symbols or input_tensor_symbols
                if dma_out_symbols:
                    for symbol in dma_out_symbols:
                        lines.append(f"    _nnc_pipeline_touch_tensor_write(&{symbol});")
                else:
                    lines.append("    _nnc_pipeline_touch_sink ^= 1u;")
        elif step_kind == "shape_prep":
            touched = False
            for symbol in (*input_tensor_symbols, *output_tensor_symbols):
                lines.append(f"    _nnc_pipeline_shape_touch_tensor(&{symbol});")
                touched = True
            if not touched:
                lines.append("    _nnc_pipeline_touch_sink ^= 1u;")
            if invoke_node and node_symbol:
                lines.append(f"    {node_symbol}();")
        elif step_kind == "compute":
            buffered_inputs = [
                record for record in input_value_records if bool(record.get("is_staged"))
            ]
            fast_inputs = [
                record
                for record in input_value_records
                if not bool(record.get("is_staged")) and record.get("fast_expr")
            ]
            buffered_outputs = [
                record for record in output_value_records if bool(record.get("is_staged"))
            ]
            fast_outputs = [
                record
                for record in output_value_records
                if not bool(record.get("is_staged")) and record.get("fast_expr")
            ]
            for record in buffered_inputs:
                tensor_symbol = str(record["tensor_symbol"])
                storage_symbol = str(record["storage_symbol"])
                lines.append(f"    {tensor_symbol}.data = {storage_symbol}_buffer;")
            for record in fast_inputs:
                tensor_symbol = str(record["tensor_symbol"])
                saved_data_symbol = str(record["saved_data_symbol"])
                if bool(record.get("needs_restore")):
                    lines.append(f"    {saved_data_symbol} = {tensor_symbol}.data;")
                lines.append(f"    {tensor_symbol}.data = {record['fast_expr']};")
            for record in buffered_outputs:
                tensor_symbol = str(record["tensor_symbol"])
                storage_symbol = str(record["storage_symbol"])
                lines.append(f"    {record['saved_data_symbol']} = {tensor_symbol}.data;")
                lines.append(f"    {tensor_symbol}.data = {storage_symbol}_buffer;")
            for record in fast_outputs:
                tensor_symbol = str(record["tensor_symbol"])
                lines.append(f"    {tensor_symbol}.data = {record['fast_expr']};")
            if node_symbol:
                lines.append(f"    {node_symbol}();")
            else:
                lines.append("    _nnc_pipeline_touch_sink ^= 1u;")
            for record in fast_inputs:
                tensor_symbol = str(record["tensor_symbol"])
                if bool(record.get("needs_restore")):
                    lines.append(f"    if ({record['saved_data_symbol']} != NULL) {{")
                    lines.append(
                        f"        {tensor_symbol}.data = {record['saved_data_symbol']};"
                    )
                    lines.append("    }")
        else:
            if node_symbol:
                lines.append(f"    {node_symbol}();")
            else:
                lines.append("    _nnc_pipeline_touch_sink ^= 1u;")
        lines.append("}")
        lines.append("")

    return lines

def _inject_parallel_runtime_into_emitted_source(
    source: str,
    pipeline_codegen_metadata: dict[str, Any],
) -> str:
    """Inject parallel runtime helper and switch nnc_run to dependency-driven execution."""
    if not _has_parallel_runtime(pipeline_codegen_metadata):
        return source

    include_block = "\n".join(
        [
            "#ifndef _GNU_SOURCE",
            "#define _GNU_SOURCE",
            "#endif",
            "#include <string.h>",
            "#include <pthread.h>",
            "#include <stdint.h>",
            "#include <unistd.h>",
            "#if defined(__linux__)",
            "#include <sched.h>",
            "#endif",
            "",
        ]
    )
    if "#include <pthread.h>" not in source:
        include_anchor = '#include "model.h"\n'
        if include_anchor in source:
            source = source.replace(include_anchor, include_block + include_anchor, 1)

    pool_decl_block = "\n".join(
        [
            "extern uint8_t _nnc_fast_pool[];",
            "extern uint8_t _nnc_slow_pool[];",
            "",
        ]
    )
    if "extern uint8_t _nnc_fast_pool[];" not in source:
        decl_anchor = '#include "nnc_ops.h"\n'
        if decl_anchor in source:
            source = source.replace(decl_anchor, decl_anchor + "\n" + pool_decl_block, 1)
        else:
            include_anchor = '#include "model.h"\n'
            if include_anchor in source:
                source = source.replace(include_anchor, include_anchor + "\n" + pool_decl_block, 1)

    helper_block = "\n".join(_render_parallel_step_helper_block(pipeline_codegen_metadata)).strip()
    parallel_block = "\n".join(_render_parallel_runtime_block(pipeline_codegen_metadata)).strip()
    injected_blocks = "\n\n".join(block for block in (helper_block, parallel_block) if block)
    if injected_blocks:
        main_anchor = "/* Main inference entry point */"
        if main_anchor in source:
            source = source.replace(main_anchor, injected_blocks + "\n\n" + main_anchor, 1)
        else:
            source = source.rstrip() + "\n\n" + injected_blocks + "\n"

    lines = source.split("\n")
    run_start = None
    run_end = None
    for index, line in enumerate(lines):
        if line.strip().startswith("void nnc_run(void) {"):
            run_start = index
            break
    if run_start is None:
        return source

    brace_depth = 0
    for index in range(run_start, len(lines)):
        brace_depth += lines[index].count("{")
        brace_depth -= lines[index].count("}")
        if index > run_start and brace_depth == 0:
            run_end = index
            break
    if run_end is None:
        return source

    new_function_lines = [
        lines[run_start],
        "    nnc_pipeline_run_parallel();",
        "}",
    ]
    rewritten_lines = lines[:run_start] + new_function_lines + lines[run_end + 1 :]
    return "\n".join(rewritten_lines)

# ---------------------------------------------------------------------------
# Main emitter
# ---------------------------------------------------------------------------

def emit_model_source(package: X86CodegenPackage, backend: Any | None = None) -> str:
    """Emit model source from a lowered package."""
    if package.ctx is None or backend is None:
        return _emit_minimal_model_source(package)

    ctx = package.ctx
    alloc_plan = get_memory_allocation_plan(ctx)
    tile_aware_runtime_plan = backend._get_tile_aware_runtime_plan(ctx, alloc_plan)
    scheduled_plan = package.scheduled_plan
    prefer_scheduled_plan = scheduled_plan is not None
    pipeline_codegen_metadata = package.pipeline_codegen_metadata
    spill_plan = get_spill_plan(ctx)

    used_standard_emitter = False
    if (
        prefer_scheduled_plan
        and backend.debug_mode
        and scheduled_plan is not None
        and bool(getattr(scheduled_plan, "transfer_points", ()))
    ):
        source = _emit_scheduled_spill_model_source(
            ctx,
            scheduled_plan,
            pipeline_codegen_metadata,
            backend,
        )
    elif prefer_scheduled_plan:
        emitter = CEmitter(
            tile_aware_wrapper_nodes=tile_aware_runtime_plan.get("wrapper_nodes", {}),
            pipeline_schedule_metadata=pipeline_codegen_metadata,
        )
        source = emitter.emit(ctx)
        used_standard_emitter = True
    elif alloc_plan is not None and (alloc_plan.has_spill or bool(alloc_plan.move_points)):
        source = backend._generate_source_with_unified_spill(
            ctx,
            alloc_plan,
            pipeline_codegen_metadata,
        )
    elif spill_plan is not None and spill_plan.has_overflow:
        source = _emit_legacy_spill_model_source(
            ctx,
            spill_plan,
            pipeline_codegen_metadata,
            backend,
        )
    else:
        emitter = CEmitter(
            tile_aware_wrapper_nodes=tile_aware_runtime_plan.get("wrapper_nodes", {}),
            pipeline_schedule_metadata=pipeline_codegen_metadata,
        )
        source = emitter.emit(ctx)
        used_standard_emitter = True

    if used_standard_emitter:
        source = _inject_parallel_runtime_into_emitted_source(
            source,
            pipeline_codegen_metadata,
        )

    source = backend._add_debug_macros(source)
    source = backend._inject_debug_into_nnc_run(source, ctx)
    return backend._append_entry_point_alias(source, ctx)


def _emit_minimal_model_source(package: X86CodegenPackage) -> str:
    lines = [
        "/* Auto-generated by NNC - DO NOT EDIT */",
        "",
        "/* Pipeline schedule summary",
    ]
    for summary_line in package.pipeline_summary_lines:
        lines.append(f" * {summary_line}")
    lines.extend(
        [
            " */",
            "",
            "void nnc_run(void) {",
            "}",
        ]
    )
    if package.entry_point != "nnc_run":
        lines.extend(
            [
                "",
                f"void {package.entry_point}(void) {{",
                "    nnc_run();",
                "}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _append_pipeline_schedule_summary_block(
    lines: list[str],
    pipeline_codegen_metadata: dict[str, Any],
    backend: Any | None = None,
) -> None:
    lines.append("/* Pipeline schedule summary")
    for summary_line in pipeline_codegen_metadata.get("summary_lines", ()):
        lines.append(f" * {_sanitize_c_comment_text(str(summary_line))}")
    lines.append(" */")
    lines.append("")


def _append_pipeline_step_comment_lines(
    lines: list[str],
    pipeline_codegen_metadata: dict[str, Any],
    node_name: str,
    *,
    backend: Any | None = None,
    indent: str = "",
) -> None:
    for comment in pipeline_codegen_metadata.get("node_comments", {}).get(node_name, ()):
        lines.append(f"{indent}/* {_sanitize_c_comment_text(comment)} */")


def _emit_scheduled_spill_model_source(
    ctx: Any,
    scheduled_plan: Any,
    pipeline_codegen_metadata: dict[str, Any],
    backend: Any,
) -> str:
    lines = [
        "/* Auto-generated by NNC - DO NOT EDIT */",
        "",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        '#include "model.h"',
        '#include "nnc_ops.h"',
        "",
        "/* External memory pools (defined in tensors.c) */",
        "extern uint8_t _nnc_fast_pool[];",
        "extern uint8_t _nnc_slow_pool[];",
        "",
    ]
    _append_pipeline_schedule_summary_block(lines, pipeline_codegen_metadata, backend)

    nodes = ctx.graph.topological_sort()
    for node in nodes:
        if node.op_type.value == "constant":
            continue
        func_name = ctx.node_symbols.get(node.name, node.name)
        lines.append(f"static void {func_name}_body(void);")

    lines.append("")
    lines.append("/* Scheduled spill/reload wrapper functions */")

    for node in nodes:
        if node.op_type.value == "constant":
            continue

        func_name = ctx.node_symbols.get(node.name, node.name)
        reload_points = backend._get_scheduled_transfer_points_for_node(
            scheduled_plan,
            before_node_name=node.name,
            transfer_kind="reload_dma",
        )
        spill_points = backend._get_scheduled_transfer_points_for_node(
            scheduled_plan,
            after_node_name=node.name,
            transfer_kind="spill_dma",
        )

        lines.append(f"/* {func_name}: {node.op_type.value} */")
        lines.append(f"static void {func_name}(void) {{")
        _append_pipeline_step_comment_lines(
            lines,
            pipeline_codegen_metadata,
            node.name,
            backend=backend,
            indent="    ",
        )

        for transfer_point in reload_points:
            for body_line in backend._build_scheduled_transfer_body_lines(ctx, transfer_point):
                lines.append(f"    {body_line}")

        lines.append(f"    {func_name}_body();")

        for transfer_point in spill_points:
            for body_line in backend._build_scheduled_transfer_body_lines(ctx, transfer_point):
                lines.append(f"    {body_line}")

        lines.append("}")
        lines.append("")

    lines.append("/* Node body functions */")
    emitter = CEmitter(pipeline_schedule_metadata=pipeline_codegen_metadata)
    body_code = emitter.emit(ctx)
    lines.append(backend._process_body_code(body_code, ctx))

    lines.append("")
    lines.append("/* Main inference entry point */")
    lines.append("void nnc_run(void) {")
    for node in nodes:
        if node.op_type.value == "constant":
            continue
        func_name = ctx.node_symbols.get(node.name, node.name)
        _append_pipeline_step_comment_lines(
            lines,
            pipeline_codegen_metadata,
            node.name,
            backend=backend,
            indent="    ",
        )
        lines.append(f"    {func_name}();")
    lines.append("}")

    return "\n".join(lines)


def _emit_legacy_spill_model_source(
    ctx: Any,
    spill_plan: Any,
    pipeline_codegen_metadata: dict[str, Any],
    backend: Any,
) -> str:
    pipeline_codegen_metadata = backend._augment_parallel_runtime_for_legacy_spill(
        ctx,
        spill_plan,
        pipeline_codegen_metadata,
    )

    lines = [
        "/* Auto-generated by NNC - DO NOT EDIT */",
        "",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        '#include "model.h"',
        '#include "nnc_ops.h"',
        "",
        "/* External memory pools (defined in tensors.c) */",
        "extern uint8_t _nnc_fast_pool[];",
        "extern uint8_t _nnc_slow_pool[];",
        "",
        "/* Forward declarations for spill/reload functions */",
    ]
    backend._append_parallel_runtime_includes(lines, pipeline_codegen_metadata)
    _append_pipeline_schedule_summary_block(lines, pipeline_codegen_metadata, backend)

    nodes = ctx.graph.topological_sort()
    for node in nodes:
        if node.op_type.value == "constant":
            continue
        func_name = ctx.node_symbols.get(node.name, node.name)
        lines.append(f"static void {func_name}_body(void);")

    lines.append("")
    lines.append("/* Spill/Reload wrapper functions */")

    for node in nodes:
        if node.op_type.value == "constant":
            continue

        func_name = ctx.node_symbols.get(node.name, node.name)
        spills_after = [
            point for point in spill_plan.spill_points if point.after_node == node.name
        ]
        reloads_before = [
            point for point in spill_plan.reload_points if point.before_node == node.name
        ]

        lines.append(f"/* {func_name}: {node.op_type.value} */")
        lines.append(f"static void {func_name}(void) {{")
        _append_pipeline_step_comment_lines(
            lines,
            pipeline_codegen_metadata,
            node.name,
            backend=backend,
            indent="    ",
        )

        for reload in reloads_before:
            lines.extend([
                f"    /* Reload {reload.tensor_name} from slow memory */",
                "    memcpy(",
                f"        _nnc_fast_pool + {reload.to_fast_offset},",
                f"        _nnc_slow_pool + {reload.from_slow_offset},",
                f"        {reload.size}",
                "    );",
                "",
            ])

        lines.append(f"    {func_name}_body();")

        for spill in spills_after:
            lines.extend([
                "",
                f"    /* Spill {spill.tensor_name} to slow memory */",
                "    memcpy(",
                f"        _nnc_slow_pool + {spill.to_slow_offset},",
                f"        _nnc_fast_pool + {spill.from_fast_offset},",
                f"        {spill.size}",
                "    );",
            ])

        lines.append("}")
        lines.append("")

    lines.append("/* Node body functions */")
    emitter = CEmitter(pipeline_schedule_metadata=pipeline_codegen_metadata)
    body_code = emitter.emit(ctx)
    lines.append(backend._process_body_code(body_code, ctx))
    helper_block = _render_parallel_step_helper_block(pipeline_codegen_metadata)
    runtime_block = _render_parallel_runtime_block(pipeline_codegen_metadata)
    if helper_block:
        lines.append("")
        lines.extend(helper_block)
    if runtime_block:
        lines.append("")
        lines.extend(runtime_block)

    lines.append("")
    lines.append("/* Main inference entry point */")
    lines.append("void nnc_run(void) {")

    if _has_parallel_runtime(pipeline_codegen_metadata):
        lines.append("    nnc_pipeline_run_parallel();")
    else:
        for node in nodes:
            if node.op_type.value == "constant":
                continue
            func_name = ctx.node_symbols.get(node.name, node.name)
            _append_pipeline_step_comment_lines(
                lines,
                pipeline_codegen_metadata,
                node.name,
                backend=backend,
                indent="    ",
            )
            lines.append(f"    {func_name}();")

    lines.append("}")

    return "\n".join(lines)
