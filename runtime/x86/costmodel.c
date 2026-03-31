/* NNC x86 Runtime - Cost Model Implementation
 * Simulates execution on 4 compute units with shared memory.
 *
 * Each operator updates global timing state instead of computing tensor data.
 * The generated model code still determines execution order; this runtime
 * prints a timing trace and a final summary.
 */

#include "nnc_ops.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NNC_COSTMODEL_COMPUTE_UNITS 4
#define NNC_COSTMODEL_SHARED_BYTES_PER_CYCLE 64ULL
#define NNC_COSTMODEL_ELEMENTWISE_ELEMENTS_PER_CYCLE 128ULL
#define NNC_COSTMODEL_SHAPE_ELEMENTS_PER_CYCLE 256ULL
#define NNC_COSTMODEL_REDUCTION_ELEMENTS_PER_CYCLE 96ULL
#define NNC_COSTMODEL_MACS_PER_CYCLE 512ULL
#define NNC_COSTMODEL_DEFAULT_LAUNCH 8ULL
#define NNC_COSTMODEL_SHAPE_LAUNCH 4ULL
#define NNC_COSTMODEL_MATMUL_LAUNCH 12ULL

static uint64_t g_costmodel_op_index = 0;
static uint64_t g_costmodel_total_cycles = 0;
static uint64_t g_costmodel_shared_ready = 0;
static uint64_t g_costmodel_compute_ready[NNC_COSTMODEL_COMPUTE_UNITS] = {0};
static int g_costmodel_initialized = 0;

static uint64_t nnc_costmodel_max_u64(uint64_t lhs, uint64_t rhs) {
    return lhs > rhs ? lhs : rhs;
}

static uint64_t nnc_costmodel_ceil_div(uint64_t numerator, uint64_t denominator) {
    if (numerator == 0 || denominator == 0) {
        return 0;
    }
    return (numerator + denominator - 1) / denominator;
}

static uint64_t nnc_costmodel_dtype_size(int dtype) {
    switch (dtype) {
        case NNC_DTYPE_FLOAT16:
            return 2;
        case NNC_DTYPE_INT64:
            return 8;
        case NNC_DTYPE_INT8:
        case NNC_DTYPE_UINT8:
        case NNC_DTYPE_BOOL:
            return 1;
        case NNC_DTYPE_FLOAT32:
        case NNC_DTYPE_INT32:
        default:
            return 4;
    }
}

static uint64_t nnc_costmodel_tensor_numel(const Tensor* tensor) {
    uint64_t total = 1;
    int i = 0;

    if (tensor == NULL || tensor->shape == NULL || tensor->ndim <= 0) {
        return 1;
    }

    for (i = 0; i < tensor->ndim; ++i) {
        int64_t dim = tensor->shape[i];
        if (dim <= 0) {
            return 1;
        }
        total *= (uint64_t)dim;
    }
    return total;
}

static uint64_t nnc_costmodel_tensor_nbytes(const Tensor* tensor) {
    if (tensor == NULL) {
        return 0;
    }
    if (tensor->nbytes > 0) {
        return (uint64_t)tensor->nbytes;
    }
    return nnc_costmodel_tensor_numel(tensor) * nnc_costmodel_dtype_size(tensor->dtype);
}

static uint64_t nnc_costmodel_sum_tensor_bytes(const Tensor* const* tensors, size_t count) {
    uint64_t total = 0;
    size_t index = 0;

    for (index = 0; index < count; ++index) {
        total += nnc_costmodel_tensor_nbytes(tensors[index]);
    }
    return total;
}

static uint64_t nnc_costmodel_sum_tensor_array_bytes(Tensor** tensors, int count) {
    uint64_t total = 0;
    int index = 0;

    if (tensors == NULL || count <= 0) {
        return 0;
    }

    for (index = 0; index < count; ++index) {
        total += nnc_costmodel_tensor_nbytes(tensors[index]);
    }
    return total;
}

static uint64_t nnc_costmodel_elementwise_cycles(const Tensor* output) {
    return nnc_costmodel_ceil_div(
        nnc_costmodel_tensor_numel(output),
        NNC_COSTMODEL_ELEMENTWISE_ELEMENTS_PER_CYCLE
    );
}

static uint64_t nnc_costmodel_shape_cycles(const Tensor* output) {
    return nnc_costmodel_ceil_div(
        nnc_costmodel_tensor_numel(output),
        NNC_COSTMODEL_SHAPE_ELEMENTS_PER_CYCLE
    );
}

static uint64_t nnc_costmodel_reduction_cycles(const Tensor* input, const Tensor* output) {
    uint64_t work = nnc_costmodel_tensor_numel(input) + nnc_costmodel_tensor_numel(output);
    return nnc_costmodel_ceil_div(work, NNC_COSTMODEL_REDUCTION_ELEMENTS_PER_CYCLE);
}

static uint64_t nnc_costmodel_matmul_cycles(
    const Tensor* lhs,
    const Tensor* rhs,
    const Tensor* output
) {
    uint64_t k_dim = 1;

    if (lhs != NULL && lhs->ndim > 0 && lhs->shape != NULL) {
        int64_t raw = lhs->shape[lhs->ndim - 1];
        if (raw > 0) {
            k_dim = (uint64_t)raw;
        }
    } else if (rhs != NULL && rhs->ndim >= 2 && rhs->shape != NULL) {
        int64_t raw = rhs->shape[rhs->ndim - 2];
        if (raw > 0) {
            k_dim = (uint64_t)raw;
        }
    }

    return nnc_costmodel_ceil_div(
        nnc_costmodel_tensor_numel(output) * k_dim,
        NNC_COSTMODEL_MACS_PER_CYCLE
    );
}

static uint64_t nnc_costmodel_conv_cycles(
    const Tensor* weight,
    const Tensor* output,
    int kernel_h,
    int kernel_w
) {
    uint64_t kernel_work = 1;

    if (weight != NULL && weight->ndim >= 4 && weight->shape != NULL) {
        int64_t channels = weight->shape[1];
        int64_t kh = weight->shape[2];
        int64_t kw = weight->shape[3];
        kernel_work = (uint64_t)(channels > 0 ? channels : 1);
        kernel_work *= (uint64_t)(kh > 0 ? kh : 1);
        kernel_work *= (uint64_t)(kw > 0 ? kw : 1);
    } else {
        kernel_work = (uint64_t)(kernel_h > 0 ? kernel_h : 1);
        kernel_work *= (uint64_t)(kernel_w > 0 ? kernel_w : 1);
    }

    return nnc_costmodel_ceil_div(
        nnc_costmodel_tensor_numel(output) * kernel_work,
        NNC_COSTMODEL_MACS_PER_CYCLE
    );
}

static uint64_t nnc_costmodel_pool_cycles(
    const Tensor* output,
    int kernel_h,
    int kernel_w
) {
    uint64_t window = (uint64_t)(kernel_h > 0 ? kernel_h : 1);
    window *= (uint64_t)(kernel_w > 0 ? kernel_w : 1);
    return nnc_costmodel_ceil_div(
        nnc_costmodel_tensor_numel(output) * window,
        NNC_COSTMODEL_REDUCTION_ELEMENTS_PER_CYCLE
    );
}

static uint64_t nnc_costmodel_lstm_cycles(
    const Tensor* x,
    const Tensor* w,
    const Tensor* r,
    const Tensor* y
) {
    uint64_t hidden_work = nnc_costmodel_tensor_numel(y);
    hidden_work += nnc_costmodel_tensor_numel(x);
    hidden_work += nnc_costmodel_tensor_numel(w);
    hidden_work += nnc_costmodel_tensor_numel(r);
    return nnc_costmodel_ceil_div(hidden_work * 4ULL, NNC_COSTMODEL_MACS_PER_CYCLE);
}

static int nnc_costmodel_pick_compute_unit(uint64_t ready_after) {
    int best_unit = 0;
    uint64_t best_start = nnc_costmodel_max_u64(ready_after, g_costmodel_compute_ready[0]);
    int index = 0;

    for (index = 1; index < NNC_COSTMODEL_COMPUTE_UNITS; ++index) {
        uint64_t candidate = nnc_costmodel_max_u64(ready_after, g_costmodel_compute_ready[index]);
        if (candidate < best_start) {
            best_start = candidate;
            best_unit = index;
        }
    }
    return best_unit;
}

static void nnc_costmodel_print_summary(void) {
    int index = 0;

    if (!g_costmodel_initialized || g_costmodel_op_index == 0) {
        return;
    }

    printf(
        "[costmodel] summary ops=%" PRIu64 " total_cycles=%" PRIu64 " shared_ready=%" PRIu64 "\n",
        g_costmodel_op_index,
        g_costmodel_total_cycles,
        g_costmodel_shared_ready
    );
    for (index = 0; index < NNC_COSTMODEL_COMPUTE_UNITS; ++index) {
        printf(
            "[costmodel] summary cu=%d ready=%" PRIu64 "\n",
            index,
            g_costmodel_compute_ready[index]
        );
    }
}

static void nnc_costmodel_init_once(void) {
    if (g_costmodel_initialized) {
        return;
    }
    g_costmodel_initialized = 1;
    atexit(nnc_costmodel_print_summary);
}

static void nnc_costmodel_schedule(
    const char* op_name,
    uint64_t input_bytes,
    uint64_t output_bytes,
    uint64_t compute_cycles,
    uint64_t launch_cycles
) {
    uint64_t read_cycles = 0;
    uint64_t write_cycles = 0;
    uint64_t read_start = 0;
    uint64_t read_done = 0;
    uint64_t compute_start = 0;
    uint64_t compute_done = 0;
    uint64_t write_start = 0;
    uint64_t write_done = 0;
    int unit = 0;

    nnc_costmodel_init_once();

    read_cycles = nnc_costmodel_ceil_div(
        input_bytes,
        NNC_COSTMODEL_SHARED_BYTES_PER_CYCLE
    );
    write_cycles = nnc_costmodel_ceil_div(
        output_bytes,
        NNC_COSTMODEL_SHARED_BYTES_PER_CYCLE
    );

    read_start = g_costmodel_shared_ready;
    read_done = read_start + read_cycles;
    g_costmodel_shared_ready = read_done;

    unit = nnc_costmodel_pick_compute_unit(read_done);
    compute_start = nnc_costmodel_max_u64(read_done, g_costmodel_compute_ready[unit]);
    compute_done = compute_start + launch_cycles + compute_cycles;
    g_costmodel_compute_ready[unit] = compute_done;

    write_start = nnc_costmodel_max_u64(compute_done, g_costmodel_shared_ready);
    write_done = write_start + write_cycles;
    g_costmodel_shared_ready = write_done;
    g_costmodel_total_cycles = nnc_costmodel_max_u64(g_costmodel_total_cycles, write_done);
    g_costmodel_op_index += 1;

    printf(
        "[costmodel] #%" PRIu64 " op=%s cu=%d start=%" PRIu64 " end=%" PRIu64
        " read=%" PRIu64 " compute=%" PRIu64 " write=%" PRIu64
        " in=%" PRIu64 "B out=%" PRIu64 "B\n",
        g_costmodel_op_index,
        op_name,
        unit,
        read_start,
        write_done,
        read_cycles,
        launch_cycles + compute_cycles,
        write_cycles,
        input_bytes,
        output_bytes
    );
}

#define NNC_COSTMODEL_DEFINE_BINARY_ELEMENTWISE(func_name) \
    void func_name(Tensor* a, Tensor* b, Tensor* out) { \
        const Tensor* tensors[] = {a, b}; \
        nnc_costmodel_schedule( \
            #func_name, \
            nnc_costmodel_sum_tensor_bytes(tensors, 2), \
            nnc_costmodel_tensor_nbytes(out), \
            nnc_costmodel_elementwise_cycles(out), \
            NNC_COSTMODEL_DEFAULT_LAUNCH \
        ); \
    }

#define NNC_COSTMODEL_DEFINE_UNARY_ELEMENTWISE(func_name) \
    void func_name(Tensor* input, Tensor* output) { \
        const Tensor* tensors[] = {input}; \
        nnc_costmodel_schedule( \
            #func_name, \
            nnc_costmodel_sum_tensor_bytes(tensors, 1), \
            nnc_costmodel_tensor_nbytes(output), \
            nnc_costmodel_elementwise_cycles(output), \
            NNC_COSTMODEL_DEFAULT_LAUNCH \
        ); \
    }

NNC_COSTMODEL_DEFINE_BINARY_ELEMENTWISE(nnc_add)
NNC_COSTMODEL_DEFINE_BINARY_ELEMENTWISE(nnc_mul)
NNC_COSTMODEL_DEFINE_BINARY_ELEMENTWISE(nnc_sub)
NNC_COSTMODEL_DEFINE_BINARY_ELEMENTWISE(nnc_div)
NNC_COSTMODEL_DEFINE_BINARY_ELEMENTWISE(nnc_equal)
NNC_COSTMODEL_DEFINE_BINARY_ELEMENTWISE(nnc_and)
NNC_COSTMODEL_DEFINE_BINARY_ELEMENTWISE(nnc_greater)
NNC_COSTMODEL_DEFINE_BINARY_ELEMENTWISE(nnc_or)
NNC_COSTMODEL_DEFINE_BINARY_ELEMENTWISE(nnc_add_relu)
NNC_COSTMODEL_DEFINE_BINARY_ELEMENTWISE(nnc_add_sigmoid)

NNC_COSTMODEL_DEFINE_UNARY_ELEMENTWISE(nnc_relu)
NNC_COSTMODEL_DEFINE_UNARY_ELEMENTWISE(nnc_sigmoid)
NNC_COSTMODEL_DEFINE_UNARY_ELEMENTWISE(nnc_tanh)
NNC_COSTMODEL_DEFINE_UNARY_ELEMENTWISE(nnc_sqrt)
NNC_COSTMODEL_DEFINE_UNARY_ELEMENTWISE(nnc_pow)
NNC_COSTMODEL_DEFINE_UNARY_ELEMENTWISE(nnc_exp)
NNC_COSTMODEL_DEFINE_UNARY_ELEMENTWISE(nnc_log)
NNC_COSTMODEL_DEFINE_UNARY_ELEMENTWISE(nnc_abs)
NNC_COSTMODEL_DEFINE_UNARY_ELEMENTWISE(nnc_neg)
NNC_COSTMODEL_DEFINE_UNARY_ELEMENTWISE(nnc_identity)
NNC_COSTMODEL_DEFINE_UNARY_ELEMENTWISE(nnc_not)

void nnc_softmax(Tensor* input, Tensor* output, int axis) {
    const Tensor* tensors[] = {input};
    (void)axis;
    nnc_costmodel_schedule(
        "nnc_softmax",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_reduction_cycles(input, output),
        NNC_COSTMODEL_DEFAULT_LAUNCH
    );
}

void nnc_conv(
    Tensor* input, Tensor* weight, Tensor* bias, Tensor* output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    const Tensor* tensors[] = {input, weight, bias};
    (void)stride_h;
    (void)stride_w;
    (void)pad_h;
    (void)pad_w;
    nnc_costmodel_schedule(
        "nnc_conv",
        nnc_costmodel_sum_tensor_bytes(tensors, 3),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_conv_cycles(weight, output, kernel_h, kernel_w),
        NNC_COSTMODEL_MATMUL_LAUNCH
    );
}

void nnc_conv1x1(
    Tensor* input, Tensor* weight, Tensor* bias, Tensor* output,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    nnc_conv(input, weight, bias, output, 1, 1, stride_h, stride_w, pad_h, pad_w);
}

void nnc_conv3x3_s1(Tensor* input, Tensor* weight, Tensor* bias, Tensor* output) {
    nnc_conv(input, weight, bias, output, 3, 3, 1, 1, 0, 0);
}

void nnc_conv7x7_s2(Tensor* input, Tensor* weight, Tensor* bias, Tensor* output) {
    nnc_conv(input, weight, bias, output, 7, 7, 2, 2, 0, 0);
}

void nnc_conv_relu(
    Tensor* input, Tensor* weight, Tensor* bias, Tensor* output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    const Tensor* tensors[] = {input, weight, bias};
    (void)stride_h;
    (void)stride_w;
    (void)pad_h;
    (void)pad_w;
    nnc_costmodel_schedule(
        "nnc_conv_relu",
        nnc_costmodel_sum_tensor_bytes(tensors, 3),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_conv_cycles(weight, output, kernel_h, kernel_w)
            + nnc_costmodel_elementwise_cycles(output),
        NNC_COSTMODEL_MATMUL_LAUNCH
    );
}

void nnc_conv_relu1x1(
    Tensor* input, Tensor* weight, Tensor* bias, Tensor* output,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    nnc_conv_relu(input, weight, bias, output, 1, 1, stride_h, stride_w, pad_h, pad_w);
}

void nnc_conv_relu3x3_s1(Tensor* input, Tensor* weight, Tensor* bias, Tensor* output) {
    nnc_conv_relu(input, weight, bias, output, 3, 3, 1, 1, 0, 0);
}

void nnc_conv_relu7x7_s2(Tensor* input, Tensor* weight, Tensor* bias, Tensor* output) {
    nnc_conv_relu(input, weight, bias, output, 7, 7, 2, 2, 0, 0);
}

void nnc_conv_sigmoid(
    Tensor* input, Tensor* weight, Tensor* bias, Tensor* output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    const Tensor* tensors[] = {input, weight, bias};
    (void)stride_h;
    (void)stride_w;
    (void)pad_h;
    (void)pad_w;
    nnc_costmodel_schedule(
        "nnc_conv_sigmoid",
        nnc_costmodel_sum_tensor_bytes(tensors, 3),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_conv_cycles(weight, output, kernel_h, kernel_w)
            + nnc_costmodel_elementwise_cycles(output),
        NNC_COSTMODEL_MATMUL_LAUNCH
    );
}

void nnc_maxpool2d(
    Tensor* input, Tensor* output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    const Tensor* tensors[] = {input};
    (void)stride_h;
    (void)stride_w;
    (void)pad_h;
    (void)pad_w;
    nnc_costmodel_schedule(
        "nnc_maxpool2d",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_pool_cycles(output, kernel_h, kernel_w),
        NNC_COSTMODEL_DEFAULT_LAUNCH
    );
}

void nnc_avgpool2d(
    Tensor* input, Tensor* output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    const Tensor* tensors[] = {input};
    (void)stride_h;
    (void)stride_w;
    (void)pad_h;
    (void)pad_w;
    nnc_costmodel_schedule(
        "nnc_avgpool2d",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_pool_cycles(output, kernel_h, kernel_w),
        NNC_COSTMODEL_DEFAULT_LAUNCH
    );
}

void nnc_reshape(Tensor* input, Tensor* output, int64_t* shape, int ndim) {
    const Tensor* tensors[] = {input};
    (void)shape;
    (void)ndim;
    nnc_costmodel_schedule(
        "nnc_reshape",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_shape_cycles(output),
        NNC_COSTMODEL_SHAPE_LAUNCH
    );
}

void nnc_flatten(Tensor* input, Tensor* output, int axis) {
    const Tensor* tensors[] = {input};
    (void)axis;
    nnc_costmodel_schedule(
        "nnc_flatten",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_shape_cycles(output),
        NNC_COSTMODEL_SHAPE_LAUNCH
    );
}

void nnc_transpose(Tensor* input, Tensor* output, int64_t* perm, int ndim) {
    const Tensor* tensors[] = {input};
    (void)perm;
    (void)ndim;
    nnc_costmodel_schedule(
        "nnc_transpose",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_shape_cycles(output),
        NNC_COSTMODEL_SHAPE_LAUNCH
    );
}

void nnc_squeeze(Tensor* input, Tensor* output, int axis) {
    const Tensor* tensors[] = {input};
    (void)axis;
    nnc_costmodel_schedule(
        "nnc_squeeze",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_shape_cycles(output),
        NNC_COSTMODEL_SHAPE_LAUNCH
    );
}

void nnc_unsqueeze(Tensor* input, Tensor* output, int axis) {
    const Tensor* tensors[] = {input};
    (void)axis;
    nnc_costmodel_schedule(
        "nnc_unsqueeze",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_shape_cycles(output),
        NNC_COSTMODEL_SHAPE_LAUNCH
    );
}

void nnc_tile(Tensor* input, Tensor* output, int64_t* repeats, int ndim) {
    const Tensor* tensors[] = {input};
    (void)repeats;
    (void)ndim;
    nnc_costmodel_schedule(
        "nnc_tile",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_shape_cycles(output),
        NNC_COSTMODEL_SHAPE_LAUNCH
    );
}

void nnc_matmul(Tensor* a, Tensor* b, Tensor* output) {
    const Tensor* tensors[] = {a, b};
    nnc_costmodel_schedule(
        "nnc_matmul",
        nnc_costmodel_sum_tensor_bytes(tensors, 2),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_matmul_cycles(a, b, output),
        NNC_COSTMODEL_MATMUL_LAUNCH
    );
}

void nnc_gemm(
    Tensor* a, Tensor* b, Tensor* c, Tensor* output,
    float alpha, float beta, int trans_a, int trans_b
) {
    const Tensor* tensors[] = {a, b, c};
    (void)alpha;
    (void)beta;
    (void)trans_a;
    (void)trans_b;
    nnc_costmodel_schedule(
        "nnc_gemm",
        nnc_costmodel_sum_tensor_bytes(tensors, 3),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_matmul_cycles(a, b, output),
        NNC_COSTMODEL_MATMUL_LAUNCH
    );
}

void nnc_gemm_nt(
    Tensor* a, Tensor* b, Tensor* c, Tensor* output,
    float alpha, float beta
) {
    nnc_gemm(a, b, c, output, alpha, beta, 0, 1);
}

void nnc_reducemean(Tensor* input, Tensor* output, int axis, int keepdims) {
    const Tensor* tensors[] = {input};
    (void)axis;
    (void)keepdims;
    nnc_costmodel_schedule(
        "nnc_reducemean",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_reduction_cycles(input, output),
        NNC_COSTMODEL_DEFAULT_LAUNCH
    );
}

void nnc_reducesum(Tensor* input, Tensor* output, int axis, int keepdims) {
    const Tensor* tensors[] = {input};
    (void)axis;
    (void)keepdims;
    nnc_costmodel_schedule(
        "nnc_reducesum",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_reduction_cycles(input, output),
        NNC_COSTMODEL_DEFAULT_LAUNCH
    );
}

void nnc_concat(Tensor** inputs, Tensor* output, int num_inputs, int axis) {
    (void)axis;
    nnc_costmodel_schedule(
        "nnc_concat",
        nnc_costmodel_sum_tensor_array_bytes(inputs, num_inputs),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_shape_cycles(output),
        NNC_COSTMODEL_SHAPE_LAUNCH
    );
}

void nnc_batchnorm(
    Tensor* input, Tensor* scale, Tensor* bias, Tensor* mean, Tensor* var,
    Tensor* output, float epsilon, int momentum
) {
    const Tensor* tensors[] = {input, scale, bias, mean, var};
    (void)epsilon;
    (void)momentum;
    nnc_costmodel_schedule(
        "nnc_batchnorm",
        nnc_costmodel_sum_tensor_bytes(tensors, 5),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_reduction_cycles(input, output),
        NNC_COSTMODEL_DEFAULT_LAUNCH
    );
}

void nnc_layernorm(
    Tensor* input, Tensor* scale, Tensor* bias, Tensor* output,
    int axis, float epsilon
) {
    const Tensor* tensors[] = {input, scale, bias};
    (void)axis;
    (void)epsilon;
    nnc_costmodel_schedule(
        "nnc_layernorm",
        nnc_costmodel_sum_tensor_bytes(tensors, 3),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_reduction_cycles(input, output),
        NNC_COSTMODEL_DEFAULT_LAUNCH
    );
}

void nnc_split(Tensor* input, Tensor** outputs, int num_outputs, int axis) {
    const Tensor* tensors[] = {input};
    (void)axis;
    nnc_costmodel_schedule(
        "nnc_split",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_sum_tensor_array_bytes(outputs, num_outputs),
        nnc_costmodel_shape_cycles(input),
        NNC_COSTMODEL_SHAPE_LAUNCH
    );
}

void nnc_clip(Tensor* input, Tensor* output, float min_val, float max_val) {
    const Tensor* tensors[] = {input};
    (void)min_val;
    (void)max_val;
    nnc_costmodel_schedule(
        "nnc_clip",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_elementwise_cycles(output),
        NNC_COSTMODEL_DEFAULT_LAUNCH
    );
}

void nnc_cast(Tensor* input, Tensor* output, int to_dtype) {
    const Tensor* tensors[] = {input};
    (void)to_dtype;
    nnc_costmodel_schedule(
        "nnc_cast",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_elementwise_cycles(output),
        NNC_COSTMODEL_DEFAULT_LAUNCH
    );
}

void nnc_shape(Tensor* input, Tensor* output, int64_t* shape_data, int ndim) {
    const Tensor* tensors[] = {input};
    (void)shape_data;
    (void)ndim;
    nnc_costmodel_schedule(
        "nnc_shape",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_shape_cycles(output),
        NNC_COSTMODEL_SHAPE_LAUNCH
    );
}

void nnc_constantofshape(Tensor* output, float value, int ndim) {
    (void)value;
    (void)ndim;
    nnc_costmodel_schedule(
        "nnc_constantofshape",
        0,
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_shape_cycles(output),
        NNC_COSTMODEL_SHAPE_LAUNCH
    );
}

void nnc_expand(Tensor* input, Tensor* output, int64_t* shape, int ndim) {
    const Tensor* tensors[] = {input};
    (void)shape;
    (void)ndim;
    nnc_costmodel_schedule(
        "nnc_expand",
        nnc_costmodel_sum_tensor_bytes(tensors, 1),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_shape_cycles(output),
        NNC_COSTMODEL_SHAPE_LAUNCH
    );
}

void nnc_gather(Tensor* data, Tensor* indices, Tensor* output, int axis, int data_dtype) {
    const Tensor* tensors[] = {data, indices};
    (void)axis;
    (void)data_dtype;
    nnc_costmodel_schedule(
        "nnc_gather",
        nnc_costmodel_sum_tensor_bytes(tensors, 2),
        nnc_costmodel_tensor_nbytes(output),
        nnc_costmodel_elementwise_cycles(output),
        NNC_COSTMODEL_DEFAULT_LAUNCH
    );
}

void nnc_lstm(
    Tensor* X, Tensor* W, Tensor* R, Tensor* B,
    Tensor* Y, Tensor* Y_h, Tensor* Y_c,
    int direction, int hidden_size
) {
    const Tensor* tensors[] = {X, W, R, B, Y_h, Y_c};
    (void)direction;
    (void)hidden_size;
    nnc_costmodel_schedule(
        "nnc_lstm",
        nnc_costmodel_sum_tensor_bytes(tensors, 6),
        nnc_costmodel_tensor_nbytes(Y)
            + nnc_costmodel_tensor_nbytes(Y_h)
            + nnc_costmodel_tensor_nbytes(Y_c),
        nnc_costmodel_lstm_cycles(X, W, R, Y),
        NNC_COSTMODEL_MATMUL_LAUNCH
    );
}
