/* Time each operation of one video VAE decoder block (H3_VAE_BF16=2 path) on
 * random data, to see where the decoder's time goes.
 * Usage: h3_bench_vae_block [ROWS]   (default 2805: one 320-pixel tile chunk) */
#include "h3_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { HIDDEN = 2048, HEADS = 32, HEAD_DIM = 64, INNER = HEADS * HEAD_DIM,
       FFN = 8192, ROPE_HALF = 24 };

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static float uniform(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}

static h3_gpu_tensor *f32(h3_gpu *gpu, size_t n, float amplitude) {
    float *v = malloc(n * sizeof(*v));
    for (size_t i = 0; i < n; i++) v[i] = uniform() * amplitude;
    h3_gpu_tensor *t = h3_gpu_tensor_from_f32(gpu, v, n);
    free(v);
    return t;
}

static h3_gpu_tensor *bf16(h3_gpu *gpu, size_t n, float amplitude) {
    h3_gpu_tensor *f = f32(gpu, n, amplitude);
    h3_gpu_tensor *b = h3_gpu_tensor_new_bf16(gpu, n);
    if (!h3_gpu_begin(gpu) || !h3_gpu_cast_f32_to_bf16(gpu, b, f, (uint32_t)n) ||
        !h3_gpu_submit(gpu)) exit(1);
    h3_gpu_tensor_free(f);
    return b;
}

#define TIME(label, call) do {                                                 \
    double best = 1e30;                                                        \
    for (int iter = 0; iter < 6; iter++) {                                     \
        double t0 = now();                                                     \
        if (!h3_gpu_begin(gpu) || !(call) || !h3_gpu_submit(gpu)) {            \
            fprintf(stderr, "%s failed: %s\n", label, h3_gpu_error(gpu));      \
            return 1;                                                          \
        }                                                                      \
        double t = now() - t0;                                                 \
        if (iter && t < best) best = t;                                        \
    }                                                                          \
    total += best;                                                             \
    printf("%-26s %8.3f ms\n", label, best * 1e3);                             \
} while (0)

int main(int argc, char **argv) {
    uint32_t rows = argc > 1 ? (uint32_t)atoi(argv[1]) : 2805;
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return 1; }
    size_t padded = ((size_t)rows + 127) & ~(size_t)127;
    h3_gpu_tensor *hidden = f32(gpu, (size_t)rows * HIDDEN, 1.0f);
    h3_gpu_tensor *norm = h3_gpu_tensor_new_f32(gpu, (size_t)rows * HIDDEN);
    h3_gpu_tensor *qkv = h3_gpu_tensor_new_f32(gpu, (size_t)rows * INNER * 3);
    h3_gpu_tensor *q = h3_gpu_tensor_new_f32(gpu, (size_t)rows * INNER);
    h3_gpu_tensor *k = h3_gpu_tensor_new_f32(gpu, (size_t)rows * INNER);
    h3_gpu_tensor *v = h3_gpu_tensor_new_f32(gpu, (size_t)rows * INNER);
    h3_gpu_tensor *heads = h3_gpu_tensor_new_f32(gpu, (size_t)rows * INNER);
    h3_gpu_tensor *branch = h3_gpu_tensor_new_f32(gpu, (size_t)rows * HIDDEN);
    h3_gpu_tensor *ff1 = h3_gpu_tensor_new_f32(gpu, (size_t)rows * FFN * 2);
    h3_gpu_tensor *act = h3_gpu_tensor_new_f32(gpu, (size_t)rows * FFN);
    h3_gpu_tensor *scratch = h3_gpu_tensor_new_bf16(gpu, padded * FFN);
    h3_gpu_tensor *b[4];
    for (int i = 0; i < 4; i++) b[i] = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * INNER);
    h3_gpu_tensor *norm_w = f32(gpu, HIDDEN, 1.0f), *scale = f32(gpu, HIDDEN, 0.1f);
    h3_gpu_tensor *qkv_w = bf16(gpu, (size_t)INNER * 3 * HIDDEN, 0.02f);
    h3_gpu_tensor *qkv_b = f32(gpu, INNER * 3, 0.01f);
    h3_gpu_tensor *out_w = bf16(gpu, (size_t)HIDDEN * INNER, 0.02f);
    h3_gpu_tensor *out_b = f32(gpu, HIDDEN, 0.01f);
    h3_gpu_tensor *w1 = bf16(gpu, (size_t)FFN * 2 * HIDDEN, 0.02f);
    h3_gpu_tensor *w1_b = f32(gpu, FFN * 2, 0.01f);
    h3_gpu_tensor *w2 = bf16(gpu, (size_t)HIDDEN * FFN, 0.02f);
    h3_gpu_tensor *w2_b = f32(gpu, HIDDEN, 0.01f);
    h3_gpu_tensor *cos_t = f32(gpu, (size_t)rows * ROPE_HALF, 1.0f);
    h3_gpu_tensor *sin_t = f32(gpu, (size_t)rows * ROPE_HALF, 1.0f);
    uint32_t elements = rows * INNER;
    double total = 0;
    printf("rows %u\n", rows);
    TIME("rms norm", h3_gpu_rms_norm_f32(gpu, norm, hidden, norm_w, rows, HIDDEN, 1e-5f));
    TIME("QKV linear", h3_gpu_linear_bf16_weight_f32(gpu, qkv, norm, qkv_w, qkv_b, scratch, rows, HIDDEN, INNER * 3));
    TIME("QK norm + RoPE", h3_gpu_video_qkv_rope_f32(gpu, q, k, v, qkv, cos_t, sin_t, rows, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f));
    TIME("cast Q/K/V to BF16", h3_gpu_cast_f32_to_bf16(gpu, b[0], q, elements) &&
                               h3_gpu_cast_f32_to_bf16(gpu, b[1], k, elements) &&
                               h3_gpu_cast_f32_to_bf16(gpu, b[2], v, elements));
    TIME("SDPA BF16 (MPS)", h3_gpu_sdpa_bf16(gpu, b[3], b[0], b[1], b[2], rows, HEADS, HEAD_DIM, 0.125f));
    TIME("cast output to F32", h3_gpu_cast_bf16_to_f32(gpu, heads, b[3], elements));
    TIME("output linear", h3_gpu_linear_bf16_weight_f32(gpu, branch, heads, out_w, out_b, scratch, rows, INNER, HIDDEN));
    TIME("residual", h3_gpu_scale_add_f32(gpu, hidden, hidden, branch, scale, rows, HIDDEN));
    TIME("rms norm 2", h3_gpu_rms_norm_f32(gpu, norm, hidden, norm_w, rows, HIDDEN, 1e-5f));
    TIME("MLP input linear", h3_gpu_linear_bf16_weight_f32(gpu, ff1, norm, w1, w1_b, scratch, rows, HIDDEN, FFN * 2));
    TIME("SwiGLU", h3_gpu_swiglu_f32(gpu, act, ff1, rows, FFN));
    TIME("MLP output linear", h3_gpu_linear_bf16_weight_f32(gpu, branch, act, w2, w2_b, scratch, rows, FFN, HIDDEN));
    TIME("residual 2", h3_gpu_scale_add_f32(gpu, hidden, hidden, branch, scale, rows, HIDDEN));
    printf("%-26s %8.3f ms (sum of separately submitted ops)\n", "block", total * 1e3);
    double flops = 2.0 * rows * ((double)HIDDEN * INNER * 3 + (double)INNER * HIDDEN +
                                 (double)HIDDEN * FFN * 2 + (double)FFN * HIDDEN) +
                   4.0 * rows * (double)rows * HEADS * HEAD_DIM;
    printf("block FLOP %.1f G\n", flops * 1e-9);
    return 0;
}
