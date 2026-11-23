/* Video VAE encoder, single image (T = 1): every causally padded 3x3x3 conv
 * sees two zero frames and the image, so only its last temporal tap
 * contributes, and the encoder runs the image alone through a 1x3x3 Conv3d
 * with that tap (CONV_LAST_TAP). This checks, on the encoder's real layer
 * shapes (one 256x256 tile) and random activation-like data, that the result
 * is byte-identical to the reference, the full 3x3x3 Conv3d over [0, 0, x]
 * with real weights on the zero-frame taps; --bench adds the median GPU wall
 * time of each.
 * Usage: h3_enc_fast_test [--bench] [SEED]   (run from the checkout) */
#include "h3_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static h3_gpu *gpu;
static uint64_t rng;

static float uniform(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void *checked(void *pointer) {
    if (!pointer) { fprintf(stderr, "out of memory\n"); exit(2); }
    return pointer;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static h3_gpu_tensor *upload(const float *values, size_t count) {
    h3_gpu_tensor *t = h3_gpu_tensor_from_f32(gpu, values, count);
    if (!t) { fprintf(stderr, "upload failed\n"); exit(2); }
    return t;
}

static h3_gpu_tensor *f32(size_t count) {
    h3_gpu_tensor *t = h3_gpu_tensor_new_f32(gpu, count);
    if (!t) { fprintf(stderr, "allocation failed\n"); exit(2); }
    return t;
}

typedef struct { uint32_t cin, cout, size, stride, stride_t; const char *name; } shape;

/* padded input extent `size` (spatial padding already applied, as in the
 * encoder: 1/1 for stride 1, 0/1 for the stride-2 downsamplers); stride_t as
 * the encoder's time_strides (it is part of the MPSGraph descriptor) */
static const shape shapes[] = {
    {3, 128, 258, 1, 1, "conv_in"},
    {128, 128, 258, 1, 1, "L0 conv"},
    {128, 128, 257, 2, 1, "L0 down"},
    {128, 256, 130, 1, 1, "L1 conv1"},
    {256, 256, 130, 1, 1, "L1 conv"},
    {256, 256, 129, 2, 2, "L1 down"},
    {256, 256, 66, 1, 1, "L2 conv"},
    {256, 256, 65, 2, 2, "L2 down"},
    {256, 512, 34, 1, 1, "L3 conv1"},
    {512, 512, 34, 1, 1, "L3 conv"},
    {512, 512, 33, 2, 1, "L3 down"},
    {512, 512, 18, 1, 1, "L4 conv"},
    {512, 1024, 18, 1, 1, "L5 conv1"},
    {1024, 1024, 18, 1, 1, "L5 conv"},
    {1024, 48, 18, 1, 1, "conv_out"},
};

enum { VARIANTS = 2 };
static const char *variant_names[VARIANTS] = {"ref", "last tap"};

static int run_variant(int variant, h3_gpu_tensor *out, h3_gpu_tensor *padded,
                       h3_gpu_tensor *single, h3_gpu_tensor *w3, h3_gpu_tensor *w1,
                       h3_gpu_tensor *bias, const shape *s) {
    uint32_t n = s->size, k = 3, st = s->stride, tt = s->stride_t;
    if (!h3_gpu_begin(gpu)) return 0;
    int ok;
    if (variant == 0)
        ok = h3_gpu_conv3d_f32_ex(gpu, out, padded, w3, bias, 1, 3, n, n, s->cin,
                                  s->cout, k, k, k, tt, st, st, 0);
    else
        ok = h3_gpu_conv3d_f32_ex(gpu, out, single, w1, bias, 1, 1, n, n, s->cin,
                                  s->cout, 1, k, k, tt, st, st, 0);
    if (!ok) { h3_gpu_abort(gpu); return 0; }
    return h3_gpu_submit(gpu);
}

int main(int argc, char **argv) {
    int bench = 0;
    unsigned long seed = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench")) bench = 1;
        else seed = strtoul(argv[i], NULL, 10);
    }
    rng = 0x9e3779b97f4a7c15ull ^ (seed * 0xbf58476d1ce4e5b9ull);
    char error[512];
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return 2; }
    int failures = 0;
    double total[VARIANTS] = {0};
    printf("%-10s %5s %5s %4s | %-28s | %s\n", "layer", "cin", "cout", "n",
           "last tap: differ max|d| relL2", bench ? "median ms ref / last tap" : "");
    for (size_t index = 0; index < sizeof(shapes) / sizeof(*shapes); index++) {
        const shape *s = &shapes[index];
        uint32_t n = s->size, o = (n - 3) / s->stride + 1;
        size_t plane = (size_t)n * n * s->cin, out_count = (size_t)o * o * s->cout;
        size_t taps = (size_t)s->cout * s->cin;
        float *x = checked(calloc(plane * 3, sizeof(*x)));   /* [0, 0, x] */
        float *w = checked(malloc(taps * 27 * sizeof(*w)));
        float *w_last = checked(malloc(taps * 9 * sizeof(*w)));
        float *b = checked(malloc(s->cout * sizeof(*b)));
        for (size_t i = 0; i < plane; i++) x[2 * plane + i] = 3.0f * uniform();
        float scale = 1.0f / sqrtf((float)(s->cin * 27));
        /* the zero-frame taps get real weights too: they must not matter */
        for (size_t i = 0; i < taps * 27; i++) w[i] = scale * uniform();
        for (size_t t = 0; t < taps; t++) memcpy(w_last + t * 9, w + t * 27 + 18, 9 * sizeof(*w));
        for (uint32_t i = 0; i < s->cout; i++) b[i] = 0.1f * uniform();
        h3_gpu_tensor *padded = upload(x, plane * 3), *single = upload(x + 2 * plane, plane);
        h3_gpu_tensor *w3 = upload(w, taps * 27), *w1 = upload(w_last, taps * 9);
        h3_gpu_tensor *bias = upload(b, s->cout);
        h3_gpu_tensor *outs[VARIANTS];
        float *host[VARIANTS];
        double ms[VARIANTS] = {0};
        for (int v = 0; v < VARIANTS; v++) {
            outs[v] = f32(out_count);
            host[v] = checked(malloc(out_count * sizeof(float)));
            if (!run_variant(v, outs[v], padded, single, w3, w1, bias, s) ||
                !h3_gpu_tensor_read_f32(outs[v], host[v], out_count)) {
                fprintf(stderr, "%s %s: %s\n", s->name, variant_names[v], h3_gpu_error(gpu));
                return 2;
            }
        }
        if (bench) {
            enum { WARM = 3, REPS = 15 };
            for (int v = 0; v < VARIANTS; v++) {
                double t[REPS];
                for (int r = 0; r < WARM + REPS; r++) {
                    double t0 = now();
                    if (!run_variant(v, outs[v], padded, single, w3, w1, bias, s)) {
                        fprintf(stderr, "%s %s: %s\n", s->name, variant_names[v],
                                h3_gpu_error(gpu));
                        return 2;
                    }
                    if (r >= WARM) t[r - WARM] = (now() - t0) * 1e3;
                }
                qsort(t, REPS, sizeof(*t), cmp_double);
                ms[v] = t[REPS / 2];
                total[v] += ms[v];
            }
        }
        printf("%-10s %5u %5u %4u", s->name, s->cin, s->cout, n);
        for (int v = 1; v < VARIANTS; v++) {
            size_t differ = 0;
            double num = 0, den = 0, max_abs = 0;
            for (size_t i = 0; i < out_count; i++) {
                double r = host[0][i], c = host[v][i], d = c - r;
                uint32_t rb, cb;
                memcpy(&rb, &host[0][i], 4);
                memcpy(&cb, &host[v][i], 4);
                differ += rb != cb;
                num += d * d;
                den += r * r;
                if (fabs(d) > max_abs) max_abs = fabs(d);
            }
            double rel = den > 0 ? sqrt(num / den) : sqrt(num);
            if (differ) failures++;
            printf(" | %8zu %8.2e %8.2e", differ, max_abs, rel);
        }
        if (bench) printf(" | %7.3f %7.3f", ms[0], ms[1]);
        printf("\n");
        h3_gpu_tensor *all[] = {padded, single, w3, w1, bias, outs[0], outs[1]};
        for (size_t i = 0; i < sizeof(all) / sizeof(*all); i++) h3_gpu_tensor_free(all[i]);
        for (int v = 0; v < VARIANTS; v++) free(host[v]);
        free(x); free(w); free(w_last); free(b);
    }
    if (bench)
        printf("sum of medians (one tile, each layer once): ref %.2f ms, last tap %.2f ms\n",
               total[0], total[1]);
    printf("%s: last-tap conv byte-identical on %s\n", failures ? "FAIL" : "PASS",
           failures ? "not every layer" : "every layer");
    h3_gpu_free(gpu);
    return failures ? 1 : 0;
}
