/* Time one video VAE decoder block (H3_VAE_BF16=2 path) on random data:
 * every op submitted alone (GPU time), then the whole block in one command
 * buffer, for the separate-kernel chain ("base") and the Track B fused chain.
 * Warm-up 5, then the median of 20. Both chains start from the same hidden
 * state; the resulting hidden states must be bit-identical (diff = 0).
 * Usage: h3_bench_vae_block [ROWS]   (default 2805: one 320-pixel 768p tile) */
#include "h3_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HIDDEN = 2048, HEADS = 32, HEAD_DIM = 64, INNER = HEADS * HEAD_DIM,
       FFN = 8192, ROPE_HALF = 24, WARMUP = 5, RUNS = 20 };

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static float uniform(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}

static h3_gpu *gpu;

static h3_gpu_tensor *f32(size_t n, float amplitude) {
    float *v = malloc(n * sizeof(*v));
    for (size_t i = 0; i < n; i++) v[i] = uniform() * amplitude;
    h3_gpu_tensor *t = h3_gpu_tensor_from_f32(gpu, v, n);
    free(v);
    return t;
}

static h3_gpu_tensor *bf16(size_t n, float amplitude) {
    h3_gpu_tensor *f = f32(n, amplitude);
    h3_gpu_tensor *b = h3_gpu_tensor_new_bf16(gpu, n);
    if (!h3_gpu_begin(gpu) || !h3_gpu_cast_f32_to_bf16(gpu, b, f, (uint32_t)n) ||
        !h3_gpu_submit(gpu)) exit(1);
    h3_gpu_tensor_free(f);
    return b;
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* Medians (ms) of the GPU time and the submit-to-complete wall time of one
 * command buffer holding `ops` (MPSGraph may split its own work into command
 * buffers that the GPU time does not see; the wall time does). */
typedef int (*ops_fn)(void);
static double last_wall_ms;
static double median_gpu_ms(ops_fn ops) {
    double samples[RUNS], walls[RUNS];
    for (int iter = 0; iter < WARMUP + RUNS; iter++) {
        h3_gpu_stats before, after;
        h3_gpu_get_stats(gpu, &before);
        double t0 = now();
        if (!h3_gpu_begin(gpu) || !ops() || !h3_gpu_submit(gpu)) {
            fprintf(stderr, "bench failed: %s\n", h3_gpu_error(gpu));
            exit(1);
        }
        double t1 = now();
        h3_gpu_get_stats(gpu, &after);
        if (iter >= WARMUP) {
            samples[iter - WARMUP] = (after.gpu_seconds - before.gpu_seconds) * 1e3;
            walls[iter - WARMUP] = (t1 - t0) * 1e3;
        }
    }
    qsort(samples, RUNS, sizeof(*samples), compare_double);
    qsort(walls, RUNS, sizeof(*walls), compare_double);
    last_wall_ms = (walls[RUNS / 2 - 1] + walls[RUNS / 2]) / 2.0;
    return (samples[RUNS / 2 - 1] + samples[RUNS / 2]) / 2.0;
}

static uint32_t rows, elements;
static h3_gpu_tensor *hidden, *start, *norm, *qkv, *q, *k, *v, *heads, *branch,
                     *ff1, *act, *scratch, *b[4], *norm_w, *scale, *qkv_w,
                     *qkv_b, *out_w, *out_b, *w1, *w1_b, *w2, *w2_b, *cos_t,
                     *sin_t;

/* base: the separate kernels of the H3_VAE_FUSED=0 path */
static int base_norm(void) { return h3_gpu_rms_norm_f32(gpu, norm, hidden, norm_w, rows, HIDDEN, 1e-5f); }
static int base_qkv(void) { return h3_gpu_linear_bf16_weight_f32(gpu, qkv, norm, qkv_w, qkv_b, scratch, rows, HIDDEN, INNER * 3); }
static int base_rope(void) { return h3_gpu_video_qkv_rope_f32(gpu, q, k, v, qkv, cos_t, sin_t, rows, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f); }
static int base_cast_qkv(void) {
    return h3_gpu_cast_f32_to_bf16(gpu, b[0], q, elements) &&
           h3_gpu_cast_f32_to_bf16(gpu, b[1], k, elements) &&
           h3_gpu_cast_f32_to_bf16(gpu, b[2], v, elements);
}
static int sdpa(void) { return h3_gpu_sdpa_bf16(gpu, b[3], b[0], b[1], b[2], rows, HEADS, HEAD_DIM, 0.125f); }
static int base_cast_out(void) { return h3_gpu_cast_bf16_to_f32(gpu, heads, b[3], elements); }
static int base_out(void) { return h3_gpu_linear_bf16_weight_f32(gpu, branch, heads, out_w, out_b, scratch, rows, INNER, HIDDEN); }
static int residual1(void) { return h3_gpu_scale_add_f32(gpu, hidden, hidden, branch, scale, rows, HIDDEN); }
static int base_w1(void) { return h3_gpu_linear_bf16_weight_f32(gpu, ff1, norm, w1, w1_b, scratch, rows, HIDDEN, FFN * 2); }
static int base_swiglu(void) { return h3_gpu_swiglu_f32(gpu, act, ff1, rows, FFN); }
static int base_w2(void) { return h3_gpu_linear_bf16_weight_f32(gpu, branch, act, w2, w2_b, scratch, rows, FFN, HIDDEN); }
static int residual2(void) { return h3_gpu_scale_add_f32(gpu, hidden, hidden, branch, scale, rows, HIDDEN); }
static int base_block(void) {
    return base_norm() && base_qkv() && base_rope() && base_cast_qkv() &&
           sdpa() && base_cast_out() && base_out() && residual1() &&
           base_norm() && base_w1() && base_swiglu() && base_w2() && residual2();
}

/* fused: the H3_VAE_FUSED=1 path (B1) */
static int fused_norm(void) { return h3_gpu_rms_norm_f32_to_bf16(gpu, scratch, hidden, norm_w, rows, HIDDEN, 1e-5f); }
static int fused_qkv(void) { return h3_gpu_linear_bf16_input_f32(gpu, qkv, scratch, qkv_w, qkv_b, rows, HIDDEN, INNER * 3); }
static int fused_rope(void) { return h3_gpu_video_qkv_rope_bf16(gpu, b[0], b[1], b[2], qkv, cos_t, sin_t, rows, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f); }
static int fused_out(void) { return h3_gpu_linear_bf16_input_f32(gpu, branch, b[3], out_w, out_b, rows, INNER, HIDDEN); }
static int fused_w1(void) { return h3_gpu_linear_bf16_input_f32(gpu, ff1, scratch, w1, w1_b, rows, HIDDEN, FFN * 2); }
static int fused_swiglu(void) { return h3_gpu_swiglu_f32_to_bf16(gpu, scratch, ff1, rows, FFN); }
static int fused_w2(void) { return h3_gpu_linear_bf16_input_f32(gpu, branch, scratch, w2, w2_b, rows, FFN, HIDDEN); }
static int fused_block(void) {
    return fused_norm() && fused_qkv() && fused_rope() && sdpa() &&
           fused_out() && residual1() && fused_norm() && fused_w1() &&
           fused_swiglu() && fused_w2() && residual2();
}

typedef struct { const char *label; ops_fn ops; } op;

static double run_list(const char *title, const op *list, size_t count) {
    double total = 0;
    printf("%s\n", title);
    for (size_t i = 0; i < count; i++) {
        double ms = median_gpu_ms(list[i].ops);
        total += ms;
        printf("  %-30s %8.3f ms GPU %8.3f ms wall\n", list[i].label, ms,
               last_wall_ms);
    }
    printf("  %-30s %8.3f ms\n", "sum of ops", total);
    return total;
}

static int reset_hidden(void) { return h3_gpu_copy_f32(gpu, hidden, 0, start, 0, (size_t)rows * HIDDEN); }

static void one_block(ops_fn block, float *out) {
    if (!h3_gpu_begin(gpu) || !reset_hidden() || !block() || !h3_gpu_submit(gpu) ||
        !h3_gpu_tensor_read_f32(hidden, out, (size_t)rows * HIDDEN)) {
        fprintf(stderr, "block failed: %s\n", h3_gpu_error(gpu));
        exit(1);
    }
}

int main(int argc, char **argv) {
    rows = argc > 1 ? (uint32_t)atoi(argv[1]) : 2805;
    elements = rows * INNER;
    char error[512];
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return 1; }
    size_t padded = ((size_t)rows + 127) & ~(size_t)127;
    start = f32((size_t)rows * HIDDEN, 1.0f);
    hidden = h3_gpu_tensor_new_f32(gpu, (size_t)rows * HIDDEN);
    norm = h3_gpu_tensor_new_f32(gpu, (size_t)rows * HIDDEN);
    qkv = h3_gpu_tensor_new_f32(gpu, (size_t)rows * INNER * 3);
    q = h3_gpu_tensor_new_f32(gpu, (size_t)rows * INNER);
    k = h3_gpu_tensor_new_f32(gpu, (size_t)rows * INNER);
    v = h3_gpu_tensor_new_f32(gpu, (size_t)rows * INNER);
    heads = h3_gpu_tensor_new_f32(gpu, (size_t)rows * INNER);
    branch = h3_gpu_tensor_new_f32(gpu, (size_t)rows * HIDDEN);
    ff1 = h3_gpu_tensor_new_f32(gpu, (size_t)rows * FFN * 2);
    act = h3_gpu_tensor_new_f32(gpu, (size_t)rows * FFN);
    scratch = h3_gpu_tensor_new_bf16(gpu, padded * FFN);
    for (int i = 0; i < 4; i++) b[i] = h3_gpu_tensor_new_bf16(gpu, padded * INNER);
    norm_w = f32(HIDDEN, 1.0f); scale = f32(HIDDEN, 0.1f);
    qkv_w = bf16((size_t)INNER * 3 * HIDDEN, 0.02f); qkv_b = f32(INNER * 3, 0.01f);
    out_w = bf16((size_t)HIDDEN * INNER, 0.02f); out_b = f32(HIDDEN, 0.01f);
    w1 = bf16((size_t)FFN * 2 * HIDDEN, 0.02f); w1_b = f32(FFN * 2, 0.01f);
    w2 = bf16((size_t)HIDDEN * FFN, 0.02f); w2_b = f32(HIDDEN, 0.01f);
    cos_t = f32((size_t)rows * ROPE_HALF, 1.0f); sin_t = f32((size_t)rows * ROPE_HALF, 1.0f);
    if (!h3_gpu_begin(gpu) || !reset_hidden() || !h3_gpu_submit(gpu)) return 1;
    printf("rows %u, warm-up %d, median of %d, GPU time per command buffer\n",
           rows, WARMUP, RUNS);

    static const op base_ops[] = {
        {"rms norm", base_norm}, {"QKV linear (+cast)", base_qkv},
        {"QK norm + RoPE F32", base_rope}, {"cast Q/K/V to BF16", base_cast_qkv},
        {"SDPA BF16 (MPS)", sdpa}, {"cast output to F32", base_cast_out},
        {"output linear (+cast)", base_out}, {"residual", residual1},
        {"rms norm 2", base_norm}, {"MLP input linear (+cast)", base_w1},
        {"SwiGLU F32", base_swiglu}, {"MLP output linear (+cast)", base_w2},
        {"residual 2", residual2},
    };
    static const op fused_ops[] = {
        {"rms norm -> BF16", fused_norm}, {"QKV linear (BF16 in)", fused_qkv},
        {"QK norm + RoPE -> BF16", fused_rope}, {"SDPA BF16 (MPS)", sdpa},
        {"output linear (BF16 in)", fused_out}, {"residual", residual1},
        {"rms norm 2 -> BF16", fused_norm}, {"MLP input linear (BF16 in)", fused_w1},
        {"SwiGLU -> BF16", fused_swiglu}, {"MLP output linear (BF16 in)", fused_w2},
        {"residual 2", residual2},
    };
    double base_sum = run_list("base (H3_VAE_FUSED=0), ops submitted alone", base_ops,
                               sizeof(base_ops) / sizeof(*base_ops));
    double fused_sum = run_list("fused (H3_VAE_FUSED=1), ops submitted alone", fused_ops,
                                sizeof(fused_ops) / sizeof(*fused_ops));
    printf("sum of ops: base %.3f ms, fused %.3f ms (%.1f%%)\n", base_sum, fused_sum,
           100.0 * (fused_sum - base_sum) / base_sum);

    struct { const char *label; ops_fn block; } blocks[] = {
        {"base", base_block}, {"B1 fused", fused_block},
    };
    size_t count = (size_t)rows * HIDDEN;
    float *x = malloc(count * sizeof(*x)), *y = malloc(count * sizeof(*y));
    one_block(base_block, x);
    size_t total_diff = 0;
    printf("whole block, one command buffer (median of %d after %d warm-up):\n", RUNS, WARMUP);
    double base_gpu = 0, base_wall = 0;
    for (size_t i = 0; i < sizeof(blocks) / sizeof(*blocks); i++) {
        double gpu_ms = median_gpu_ms(blocks[i].block), wall = last_wall_ms;
        if (!i) { base_gpu = gpu_ms; base_wall = wall; }
        one_block(blocks[i].block, y);
        size_t diff = 0;
        for (size_t j = 0; j < count; j++) diff += memcmp(&x[j], &y[j], sizeof(float)) != 0;
        total_diff += diff;
        printf("  %-26s GPU %8.3f ms (%+5.1f%%)  wall %8.3f ms (%+5.1f%%)  diff vs base %zu/%zu\n",
               blocks[i].label, gpu_ms, 100.0 * (gpu_ms - base_gpu) / base_gpu,
               wall, 100.0 * (wall - base_wall) / base_wall, diff, count);
    }
    free(x); free(y);
    return total_diff ? 2 : 0;
}
