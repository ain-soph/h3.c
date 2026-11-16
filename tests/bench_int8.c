/* Time H3's M5 int8 linear (dynamic per-row activation quantization + int8
 * TensorOps GEMM + BF16 output) for DiT-sized projections. */
#include "h3_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(void) {
    char error[1024];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return 2; }
    const uint32_t shapes[][3] = {{37797, 5376, 21504}, {37797, 5376, 28672},
                                  {37797, 14336, 5376}};
    for (unsigned s = 0; s < 3; s++) {
        uint32_t m = shapes[s][0], k = shapes[s][1], n = shapes[s][2];
        h3_gpu_tensor *x = h3_gpu_tensor_new_bf16(gpu, (size_t)m * k);
        h3_gpu_tensor *wb = h3_gpu_tensor_new_bf16(gpu, (size_t)n * k);
        h3_gpu_tensor *w = h3_gpu_tensor_new_i8(gpu, (size_t)n * k);
        h3_gpu_tensor *ws = h3_gpu_tensor_new_f32(gpu, n);
        h3_gpu_tensor *xq = h3_gpu_tensor_new_i8(gpu, (size_t)((m + 127) & ~127u) * k);
        h3_gpu_tensor *xs = h3_gpu_tensor_new_f32(gpu, (m + 127) & ~127u);
        h3_gpu_tensor *y = h3_gpu_tensor_new_bf16(gpu, (size_t)m * n);
        if (!h3_gpu_begin(gpu) ||
            !h3_gpu_quantize_weight_int8(gpu, w, ws, wb, n, k) ||
            !h3_gpu_submit(gpu)) { fprintf(stderr, "%s\n", h3_gpu_error(gpu)); return 1; }
        double best = 1e30;
        for (int iter = 0; iter < 4; iter++) {
            double t0 = now();
            if (!h3_gpu_begin(gpu) ||
                !h3_gpu_linear_int8_bf16(gpu, y, xq, xs, x, w, ws, m, k, n, 0) ||
                !h3_gpu_submit(gpu)) { fprintf(stderr, "%s\n", h3_gpu_error(gpu)); return 1; }
            double t = now() - t0;
            if (iter && t < best) best = t;
        }
        printf("h3 int8 linear %ux%ux%u: %.4f s  %.1f TOPS\n", m, k, n, best,
               2.0 * m * (double)k * n / best / 1e12);
        h3_gpu_tensor_free(x); h3_gpu_tensor_free(wb); h3_gpu_tensor_free(w);
        h3_gpu_tensor_free(ws); h3_gpu_tensor_free(xq); h3_gpu_tensor_free(xs);
        h3_gpu_tensor_free(y);
    }
    h3_gpu_free(gpu);
    return 0;
}
