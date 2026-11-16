/* Compare H3's DiT attention implementations on head-major BF16 Q/K/V:
 * MPSGraph SDPA (default) against H3_ATTN=<mode> kernels. Reports the best of
 * three timings (4*N*N*heads*dim FLOP) and the error relative to MPSGraph.
 * Usage: h3_bench_attn MODE N...   (MODE: nax, int8, ...)
 * An argument L:WxHxF uses the real target layout of that render size, and
 * with H3_SPARSE set runs MODE through the block-sparse plan (H3_SPARSE=all
 * keeps every block, as a correctness check). */
#include "h3_gpu.h"
#include "h3_sparse.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static float gaussian(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    double u1 = ((rng >> 11) + 1.0) / 9007199254740993.0;
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    double u2 = (rng >> 11) / 9007199254740992.0;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}
static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static double run(h3_gpu *gpu, const char *mode, h3_gpu_tensor *o,
                  h3_gpu_tensor *q, h3_gpu_tensor *k, h3_gpu_tensor *v,
                  unsigned n) {
    if (mode) setenv("H3_ATTN", mode, 1); else unsetenv("H3_ATTN");
    double best = 1e30;
    for (int iter = 0; iter < 4; iter++) {
        double t0 = now();
        h3_gpu_declare_head_major_sdpa_inputs(gpu);
        if (!h3_gpu_begin(gpu) ||
            !h3_gpu_sdpa_bf16_head_major_output(gpu, o, q, k, v, n, 56, 128,
                                                1.0f / sqrtf(128.0f)) ||
            !h3_gpu_submit(gpu)) {
            fprintf(stderr, "attention failed: %s\n", h3_gpu_error(gpu));
            exit(1);
        }
        double t = now() - t0;
        if (iter && t < best) best = t;
    }
    return best;
}

int main(int argc, char **argv) {
    enum { HEADS = 56, DIM = 128 };
    if (argc < 3) { fprintf(stderr, "usage: h3_bench_attn MODE N...\n"); return 2; }
    const char *mode = argv[1];
    char error[1024];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return 2; }
    for (int arg = 2; arg < argc; arg++) {
        unsigned n = (unsigned)strtoul(argv[arg], NULL, 10);
        h3_gpu_attention_plan *gpu_plan = NULL;
        double density = 1.0;
        if (argv[arg][0] == 'L' && argv[arg][1] == ':') {
            int w = 0, h = 0, f = 0, lw = 0, lh = 0;
            sscanf(argv[arg] + 2, "%dx%dx%d", &w, &h, &f);
            h3_latent_canvas(w, h, &lw, &lh);
            h3_temporal_shape shape = h3_temporal(f);
            h3_layout_spec spec = {100, shape.video_t, lh, lw, shape.audio_t,
                                   shape.frame_count, NULL, 0, NULL, 0};
            h3_layout layout;
            if (!h3_layout_build(&spec, &layout, error, sizeof(error))) {
                fprintf(stderr, "%s\n", error);
                return 1;
            }
            n = (unsigned)layout.seq_len;
            const char *sparse = getenv("H3_SPARSE");
            if (sparse && *sparse) {
                int all = !strcmp(sparse, "all");
                if (all) setenv("H3_SPARSE", "t0=100000", 1);
                h3_sparse_params params;
                h3_sparse_plan plan;
                if (!h3_sparse_params_from_env(&params) ||
                    !h3_sparse_plan_build(&layout, &params, 64, 32, &plan,
                                          error, sizeof(error))) {
                    fprintf(stderr, "%s\n", error);
                    return 1;
                }
                if (all) setenv("H3_SPARSE", "all", 1);
                density = h3_sparse_plan_density(&plan);
                gpu_plan = h3_gpu_attention_plan_create(
                    gpu, plan.rows, plan.perm, plan.inverse, plan.q_blocks,
                    plan.offsets, plan.kept, plan.indices);
                h3_sparse_plan_free(&plan);
            }
            h3_layout_free(&layout);
        }
        size_t count = (size_t)n * HEADS * DIM;
        uint16_t *host = malloc(count * sizeof(*host));
        h3_gpu_tensor *t[3];
        /* Q/K like RMS-normed projections (unit variance per element, times a
         * per-channel gain so logits reach a realistic spread); V unit. */
        for (int which = 0; which < 3; which++) {
            for (size_t i = 0; i < count; i++)
                host[i] = to_bf16(gaussian() * (which < 2 ? 1.6f : 1.0f));
            t[which] = h3_gpu_tensor_from_bf16(gpu, host, count);
        }
        h3_gpu_tensor *reference = h3_gpu_tensor_new_bf16(gpu, count);
        h3_gpu_tensor *actual = h3_gpu_tensor_new_bf16(gpu, count);
        h3_gpu_attention_plan_use(gpu, NULL);
        double mps = run(gpu, NULL, reference, t[0], t[1], t[2], n);
        h3_gpu_attention_plan_use(gpu, gpu_plan);
        double other = run(gpu, mode, actual, t[0], t[1], t[2], n);
        h3_gpu_attention_plan_use(gpu, NULL);
        h3_gpu_attention_plan_free(gpu_plan);
        uint16_t *a = malloc(count * sizeof(*a));
        h3_gpu_tensor_read_bf16(reference, host, count);
        h3_gpu_tensor_read_bf16(actual, a, count);
        double max_abs = 0, err2 = 0, ref2 = 0;
        size_t bad = 0;
        for (size_t i = 0; i < count; i++) {
            double r = from_bf16(host[i]), x = from_bf16(a[i]);
            if (!isfinite(x)) { bad++; continue; }
            double d = fabs(r - x);
            if (d > max_abs) max_abs = d;
            err2 += d * d;
            ref2 += r * r;
        }
        double flop = 4.0 * n * (double)n * (double)(HEADS * DIM);
        printf("n=%u density %.3f  mps %.3f s %.1f TFLOPS | %s %.3f s %.1f TFLOPS (x%.2f) | "
               "max_abs %.4g rel_rms %.4g nonfinite %zu\n",
               n, density, mps, flop / mps / 1e12, mode, other, flop / other / 1e12,
               mps / other, max_abs, sqrt(err2 / ref2), bad);
        fflush(stdout);
        free(host); free(a);
        for (int which = 0; which < 3; which++) h3_gpu_tensor_free(t[which]);
        h3_gpu_tensor_free(reference); h3_gpu_tensor_free(actual);
    }
    h3_gpu_free(gpu);
    return 0;
}
