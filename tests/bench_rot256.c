/* Track A step A0 (rot256): checks and costs of the Hadamard-rotated int8
 * activation quantizers (h3_rot.h, h3_shaders.metal *_rot256).
 *
 *   1. H256 H256 = I and H128 H128 = I (CPU, max |error| < 1e-5)
 *   2. GPU rotated quantizers vs the CPU reference (rotate in F32, then the
 *      symmetric per-row / per-1024-group rule): scales relative error and
 *      int8 code differences, for every class path:
 *        qkv/fc1  fused gate + AdaLN + quantize (rot256 kernel)
 *        qkv/fc1  standalone row quantizer (first block / unfused paths)
 *        out256   head-major SDPA layout -> rotated rows
 *        fc2      1024-grouped and row quantizers
 *      plus: the rotated fused kernel leaves the gated residual unchanged.
 *   3. W8A8 GEMM error vs F32 on an outlier-heavy synthetic activation, plain
 *      vs rotated (illustration only; the real gate is the A2 teacher forcing).
 *   4. Time per op at M = 9472 (512x512 124f) and 37797 (1344x768 124f):
 *      plain vs rotated quantizer and the int8 linear it feeds; overhead =
 *      extra quantizer time / (plain quantizer + linear). Target < 2 %.
 *
 * Usage: ./h3_bench_rot256 [--quick]   (run from the h3.c directory) */
#include "h3_gpu.h"
#include "h3_rot.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HIDDEN = 5376, INNER = 7168, FFN = 14336, HEADS = 56, HEAD_DIM = 128 };

static int failures;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static double uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return ((rng_state >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static float normal(void) {
    return (float)(sqrt(-2.0 * log(uniform())) * cos(6.283185307179586 *
                                                     uniform()));
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}
static float from_bf16(uint16_t bits) {
    uint32_t word = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &word, sizeof(value));
    return value;
}

/* Activation-like rows: N(0,1) with a few heavy outlier channels. */
static void fill_activation(uint16_t *values, size_t rows, size_t columns,
                            float outlier) {
    for (size_t row = 0; row < rows; row++)
        for (size_t column = 0; column < columns; column++) {
            float value = normal();
            if (column % 997 == 13 || column % 1531 == 700) value *= outlier;
            values[row * columns + column] = to_bf16(value);
        }
}

static void cpu_quantize(const float *values, size_t count, int rotate,
                         int8_t *quantized, float *scale) {
    float *copy = malloc(count * sizeof(*copy));
    memcpy(copy, values, count * sizeof(*copy));
    if (rotate)
        for (size_t group = 0; group < count; group += 256)
            h3_hadamard256_f32(copy + group);
    float max_abs = 0.0f;
    for (size_t index = 0; index < count; index++)
        max_abs = fmaxf(max_abs, fabsf(copy[index]));
    *scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
    float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
    for (size_t index = 0; index < count; index++) {
        float q = rintf(copy[index] * inverse);
        quantized[index] = (int8_t)(q > 127 ? 127 : q < -127 ? -127 : q);
    }
    free(copy);
}

typedef struct {
    size_t codes, off_by_one, worse;
    double max_scale_error;
} compare_stats;

/* rows of `columns` codes with `groups` scales per row. */
static void compare_rows(const char *label, const float *reference_rows,
                         size_t rows, size_t columns, size_t group,
                         int rotate, const int8_t *gpu_codes,
                         const float *gpu_scales) {
    compare_stats stats = {0, 0, 0, 0.0};
    int8_t *codes = malloc(group);
    for (size_t row = 0; row < rows; row++)
        for (size_t first = 0; first < columns; first += group) {
            float scale;
            cpu_quantize(reference_rows + row * columns + first, group,
                         rotate, codes, &scale);
            float actual = gpu_scales[row * (columns / group) + first / group];
            double error = fabs((double)actual - scale) / scale;
            if (error > stats.max_scale_error) stats.max_scale_error = error;
            for (size_t index = 0; index < group; index++) {
                int difference = abs(gpu_codes[row * columns + first + index] -
                                     codes[index]);
                stats.codes++;
                if (difference == 1) stats.off_by_one++;
                else if (difference > 1) stats.worse++;
            }
        }
    free(codes);
    int pass = stats.max_scale_error < 1e-5 && stats.worse == 0 &&
               stats.off_by_one * 10000 < stats.codes;
    if (!pass) failures++;
    printf("  %-44s scale rel err %.2e  codes off-by-1 %zu/%zu  >1: %zu  %s\n",
           label, stats.max_scale_error, stats.off_by_one, stats.codes,
           stats.worse, pass ? "PASS" : "FAIL");
}

static h3_gpu_tensor *upload_bf16(h3_gpu *gpu, const uint16_t *values,
                                  size_t count) {
    return h3_gpu_tensor_from_bf16(gpu, values, count);
}

static void check_identities(void) {
    float matrix[256 * 256];
    double worst = 0.0;
    for (int size = 256; size >= 128; size -= 128) {
        worst = 0.0;
        for (int column = 0; column < size; column++) {
            float *v = matrix + column * size;
            memset(v, 0, (size_t)size * sizeof(float));
            v[column] = 1.0f;
            if (size == 256) { h3_hadamard256_f32(v); h3_hadamard256_f32(v); }
            else { h3_hadamard128_f32(v); h3_hadamard128_f32(v); }
            for (int row = 0; row < size; row++)
                worst = fmax(worst, fabs(v[row] - (row == column)));
        }
        int pass = worst < 1e-5;
        if (!pass) failures++;
        printf("  H%d * H%d = I: max |error| %.2e  %s\n", size, size, worst,
               pass ? "PASS" : "FAIL");
    }
    /* Symmetric and entries +-1/16: compare against the closed form. */
    float row[256];
    double entry_error = 0.0;
    for (int column = 0; column < 256; column++) {
        memset(row, 0, sizeof(row));
        row[column] = 1.0f;
        h3_hadamard256_f32(row);
        for (int index = 0; index < 256; index++) {
            int sign = 1;
            for (int digit = 0; digit < 4; digit++) {
                int d = (index >> (2 * digit)) & 3, e = (column >> (2 * digit)) & 3;
                if (d + e == 3) sign = -sign;
            }
            entry_error = fmax(entry_error, fabs(row[index] - sign / 16.0));
        }
    }
    int pass = entry_error < 1e-7;
    if (!pass) failures++;
    printf("  H256 entries == (H4/2)^4 closed form: max |error| %.2e  %s\n",
           entry_error, pass ? "PASS" : "FAIL");
}

static float *bf16_rows_to_f32(const uint16_t *values, size_t count) {
    float *result = malloc(count * sizeof(*result));
    for (size_t index = 0; index < count; index++)
        result[index] = from_bf16(values[index]);
    return result;
}

#define GPU_OK(call) do { if (!(call)) { fprintf(stderr, "%s: %s\n", #call, \
    h3_gpu_error(gpu)); exit(1); } } while (0)

static void check_row_quantizer(h3_gpu *gpu, uint32_t columns) {
    const uint32_t rows = 200, padded = 256, outputs = 128;
    size_t count = (size_t)rows * columns;
    uint16_t *x = malloc(count * sizeof(*x));
    fill_activation(x, rows, columns, 40.0f);
    uint16_t *wb = malloc((size_t)outputs * columns * sizeof(*wb));
    for (size_t index = 0; index < (size_t)outputs * columns; index++)
        wb[index] = to_bf16(0.02f * normal());
    h3_gpu_tensor *input = upload_bf16(gpu, x, count);
    h3_gpu_tensor *wsrc = upload_bf16(gpu, wb, (size_t)outputs * columns);
    h3_gpu_tensor *w = h3_gpu_tensor_new_i8(gpu, (size_t)outputs * columns);
    h3_gpu_tensor *ws = h3_gpu_tensor_new_f32(gpu, outputs);
    h3_gpu_tensor *xq = h3_gpu_tensor_new_i8(gpu, (size_t)padded * columns);
    h3_gpu_tensor *xs = h3_gpu_tensor_new_f32(gpu, padded);
    h3_gpu_tensor *y = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * outputs);
    GPU_OK(h3_gpu_begin(gpu) &&
           h3_gpu_quantize_weight_int8(gpu, w, ws, wsrc, outputs, columns) &&
           h3_gpu_linear_int8_bf16_rotated(gpu, y, xq, xs, input, w, ws, rows,
                                           columns, outputs, 0, 1) &&
           h3_gpu_submit(gpu));
    int8_t *codes = malloc((size_t)padded * columns);
    float *scales = malloc(padded * sizeof(float));
    h3_gpu_tensor_read_i8(xq, codes, (size_t)padded * columns);
    h3_gpu_tensor_read_f32(xs, scales, padded);
    float *reference = bf16_rows_to_f32(x, count);
    char label[96];
    snprintf(label, sizeof(label), "row quantizer rot256, K=%u", columns);
    compare_rows(label, reference, rows, columns, columns, 1, codes, scales);
    int padded_ok = 1;
    for (size_t index = (size_t)rows * columns;
         index < (size_t)padded * columns; index++)
        if (codes[index]) padded_ok = 0;
    for (uint32_t row = rows; row < padded; row++)
        if (scales[row] != 1.0f) padded_ok = 0;
    if (!padded_ok) failures++;
    printf("  %-44s %s\n", "  padded rows zero, scale 1",
           padded_ok ? "PASS" : "FAIL");
    free(x); free(wb); free(codes); free(scales); free(reference);
    h3_gpu_tensor_free(input); h3_gpu_tensor_free(wsrc);
    h3_gpu_tensor_free(w); h3_gpu_tensor_free(ws); h3_gpu_tensor_free(xq);
    h3_gpu_tensor_free(xs); h3_gpu_tensor_free(y);
}

static void check_head_major(h3_gpu *gpu) {
    const uint32_t rows = 200, padded = 256, outputs = 128;
    size_t count = (size_t)rows * INNER;
    uint16_t *row_major = malloc(count * sizeof(*row_major));
    fill_activation(row_major, rows, INNER, 20.0f);
    uint16_t *head_major = malloc(count * sizeof(*head_major));
    for (uint32_t row = 0; row < rows; row++)
        for (uint32_t head = 0; head < HEADS; head++)
            memcpy(head_major + ((size_t)head * rows + row) * HEAD_DIM,
                   row_major + (size_t)row * INNER + head * HEAD_DIM,
                   HEAD_DIM * sizeof(uint16_t));
    uint16_t *wb = malloc((size_t)outputs * INNER * sizeof(*wb));
    for (size_t index = 0; index < (size_t)outputs * INNER; index++)
        wb[index] = to_bf16(0.02f * normal());
    h3_gpu_tensor *input = upload_bf16(gpu, head_major, count);
    h3_gpu_tensor *wsrc = upload_bf16(gpu, wb, (size_t)outputs * INNER);
    h3_gpu_tensor *w = h3_gpu_tensor_new_i8(gpu, (size_t)outputs * INNER);
    h3_gpu_tensor *ws = h3_gpu_tensor_new_f32(gpu, outputs);
    h3_gpu_tensor *xq = h3_gpu_tensor_new_i8(gpu, (size_t)padded * INNER);
    h3_gpu_tensor *xs = h3_gpu_tensor_new_f32(gpu, padded);
    h3_gpu_tensor *y = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * outputs);
    GPU_OK(h3_gpu_begin(gpu) &&
           h3_gpu_quantize_weight_int8(gpu, w, ws, wsrc, outputs, INNER) &&
           h3_gpu_linear_int8_head_major_bf16(gpu, y, xq, xs, input, w, ws,
                                              rows, HEADS, HEAD_DIM, outputs,
                                              1) &&
           h3_gpu_submit(gpu));
    int8_t *codes = malloc((size_t)padded * INNER);
    float *scales = malloc(padded * sizeof(float));
    h3_gpu_tensor_read_i8(xq, codes, (size_t)padded * INNER);
    h3_gpu_tensor_read_f32(xs, scales, padded);
    float *reference = bf16_rows_to_f32(row_major, count);
    compare_rows("head-major -> rows rot256 (out256)", reference, rows,
                 INNER, INNER, 1, codes, scales);
    free(row_major); free(head_major); free(wb); free(codes); free(scales);
    free(reference);
    h3_gpu_tensor_free(input); h3_gpu_tensor_free(wsrc);
    h3_gpu_tensor_free(w); h3_gpu_tensor_free(ws); h3_gpu_tensor_free(xq);
    h3_gpu_tensor_free(xs); h3_gpu_tensor_free(y);
}

/* The MLP leaves the FC2 input codes in the activation arena; the BF16
 * SwiGLU output `activated` is the reference input. */
static void check_fc2(h3_gpu *gpu, int row_fc2) {
    const uint32_t rows = 256;
    uint16_t *x = malloc((size_t)rows * HIDDEN * sizeof(*x));
    fill_activation(x, rows, HIDDEN, 10.0f);
    size_t w1 = (size_t)2 * FFN * HIDDEN, w2 = (size_t)HIDDEN * FFN;
    uint16_t *wb = malloc(w1 * sizeof(*wb));
    for (size_t index = 0; index < w1; index++)
        wb[index] = to_bf16(0.02f * normal());
    h3_gpu_tensor *input = upload_bf16(gpu, x, (size_t)rows * HIDDEN);
    h3_gpu_tensor *fc1b = upload_bf16(gpu, wb, w1);
    h3_gpu_tensor *fc2b = upload_bf16(gpu, wb, w2);
    h3_gpu_tensor *fc1 = h3_gpu_tensor_new_i8(gpu, w1);
    h3_gpu_tensor *fc1s = h3_gpu_tensor_new_f32(gpu, 2 * FFN);
    h3_gpu_tensor *fc2 = h3_gpu_tensor_new_i8(gpu, w2);
    h3_gpu_tensor *fc2s = h3_gpu_tensor_new_f32(gpu, HIDDEN);
    h3_gpu_tensor *act = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * FFN);
    h3_gpu_tensor *q = h3_gpu_tensor_new_i8(gpu, (size_t)rows * FFN);
    h3_gpu_tensor *qs = h3_gpu_tensor_new_f32(gpu, (size_t)rows * 14);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * HIDDEN);
    GPU_OK(h3_gpu_begin(gpu) &&
           h3_gpu_quantize_weight_int8(gpu, fc1, fc1s, fc1b, 2 * FFN, HIDDEN) &&
           h3_gpu_quantize_weight_int8(gpu, fc2, fc2s, fc2b, HIDDEN, FFN) &&
           h3_gpu_mlp_int8_bf16(gpu, out, act, q, qs, input, fc1, fc1s, fc2,
                                fc2s, NULL, NULL, rows, HIDDEN, FFN, HIDDEN,
                                0, 0, row_fc2, 0,
                                H3_GPU_ROTATE_FC1_INPUT |
                                H3_GPU_ROTATE_FC2_INPUT) &&
           h3_gpu_submit(gpu));
    uint16_t *activated = malloc((size_t)rows * FFN * sizeof(*activated));
    int8_t *codes = malloc((size_t)rows * FFN);
    float *scales = malloc((size_t)rows * 14 * sizeof(float));
    h3_gpu_tensor_read_bf16(act, activated, (size_t)rows * FFN);
    h3_gpu_tensor_read_i8(q, codes, (size_t)rows * FFN);
    h3_gpu_tensor_read_f32(qs, scales, row_fc2 ? rows : (size_t)rows * 14);
    float *reference = bf16_rows_to_f32(activated, (size_t)rows * FFN);
    compare_rows(row_fc2 ? "fc2 row quantizer rot256" :
                           "fc2 1024-grouped quantizer rot256",
                 reference, rows, FFN, row_fc2 ? FFN : 1024, 1, codes,
                 scales);
    free(x); free(wb); free(activated); free(codes); free(scales);
    free(reference);
    h3_gpu_tensor_free(input); h3_gpu_tensor_free(fc1b);
    h3_gpu_tensor_free(fc2b); h3_gpu_tensor_free(fc1);
    h3_gpu_tensor_free(fc1s); h3_gpu_tensor_free(fc2);
    h3_gpu_tensor_free(fc2s); h3_gpu_tensor_free(act); h3_gpu_tensor_free(q);
    h3_gpu_tensor_free(qs); h3_gpu_tensor_free(out);
}

typedef struct {
    h3_gpu_tensor *residual, *branch, *norm, *modulation, *map;
    h3_gpu_tensor *gated, *adaln, *q, *qs;
    uint32_t rows, padded;
} adaln_case;

static adaln_case make_adaln(h3_gpu *gpu, uint32_t rows) {
    adaln_case c;
    c.rows = rows;
    c.padded = (rows + 127u) & ~127u;
    size_t count = (size_t)rows * HIDDEN;
    uint16_t *values = malloc(count * sizeof(*values));
    fill_activation(values, rows, HIDDEN, 30.0f);
    c.residual = upload_bf16(gpu, values, count);
    fill_activation(values, rows, HIDDEN, 1.0f);
    c.branch = upload_bf16(gpu, values, count);
    for (size_t index = 0; index < HIDDEN; index++)
        values[index] = to_bf16(1.0f + 0.3f * normal());
    c.norm = upload_bf16(gpu, values, HIDDEN);
    size_t mod = (size_t)2 * 6 * HIDDEN;
    for (size_t index = 0; index < mod; index++)
        values[index] = to_bf16(0.2f * normal());
    c.modulation = upload_bf16(gpu, values, mod);
    uint32_t *map = malloc(rows * sizeof(*map));
    for (uint32_t row = 0; row < rows; row++) map[row] = row % 7 == 0;
    c.map = h3_gpu_tensor_from_u32(gpu, map, rows);
    c.gated = h3_gpu_tensor_new_bf16(gpu, count);
    c.adaln = h3_gpu_tensor_new_bf16(gpu, count);
    c.q = h3_gpu_tensor_new_i8(gpu, (size_t)c.padded * HIDDEN);
    c.qs = h3_gpu_tensor_new_f32(gpu, c.padded);
    free(values);
    free(map);
    return c;
}

static void free_adaln(adaln_case *c) {
    h3_gpu_tensor_free(c->residual); h3_gpu_tensor_free(c->branch);
    h3_gpu_tensor_free(c->norm); h3_gpu_tensor_free(c->modulation);
    h3_gpu_tensor_free(c->map); h3_gpu_tensor_free(c->gated);
    h3_gpu_tensor_free(c->adaln); h3_gpu_tensor_free(c->q);
    h3_gpu_tensor_free(c->qs);
}

static void check_fused_adaln(h3_gpu *gpu) {
    adaln_case c = make_adaln(gpu, 200);
    size_t count = (size_t)c.rows * HIDDEN;
    uint16_t *reference_gated = malloc(count * sizeof(uint16_t));
    uint16_t *adaln = malloc(count * sizeof(uint16_t));
    uint16_t *gated = malloc(count * sizeof(uint16_t));
    int8_t *codes = malloc((size_t)c.padded * HIDDEN);
    float *scales = malloc(c.padded * sizeof(float));
    for (int rotate = 0; rotate <= 1; rotate++) {
        GPU_OK(h3_gpu_begin(gpu) &&
               h3_gpu_gate_adaln_bf16(gpu, c.gated, c.adaln, c.residual,
                                      c.branch, c.norm, c.modulation,
                                      c.modulation, c.map, c.rows, HIDDEN, 6,
                                      2, 3, 4, 1e-5f) &&
               h3_gpu_submit(gpu));
        h3_gpu_tensor_read_bf16(c.gated, reference_gated, count);
        h3_gpu_tensor_read_bf16(c.adaln, adaln, count);
        GPU_OK(h3_gpu_begin(gpu) &&
               h3_gpu_gate_adaln_quantize_int8(
                   gpu, c.gated, c.q, c.qs, c.residual, c.branch, c.norm,
                   c.modulation, c.modulation, c.map, c.rows, c.padded,
                   HIDDEN, 6, 2, 3, 4, 1e-5f, rotate) &&
               h3_gpu_submit(gpu));
        h3_gpu_tensor_read_bf16(c.gated, gated, count);
        h3_gpu_tensor_read_i8(c.q, codes, (size_t)c.padded * HIDDEN);
        h3_gpu_tensor_read_f32(c.qs, scales, c.padded);
        float *reference = bf16_rows_to_f32(adaln, count);
        compare_rows(rotate ? "fused gate+AdaLN+quantize rot256 (qkv/fc1)" :
                              "fused gate+AdaLN+quantize plain (control)",
                     reference, c.rows, HIDDEN, HIDDEN, rotate, codes,
                     scales);
        int same = !memcmp(gated, reference_gated, count * sizeof(uint16_t));
        if (!same) failures++;
        printf("  %-44s %s\n", "  gated residual bytes == unfused",
               same ? "PASS" : "FAIL");
        free(reference);
    }
    free(reference_gated); free(adaln); free(gated); free(codes);
    free(scales);
    free_adaln(&c);
}

/* Synthetic W8A8 error: plain vs rotated, against an F32 product. */
static void gemm_error(h3_gpu *gpu) {
    const uint32_t rows = 256, columns = HIDDEN, outputs = 256;
    uint16_t *x = malloc((size_t)rows * columns * sizeof(*x));
    uint16_t *w = malloc((size_t)outputs * columns * sizeof(*w));
    fill_activation(x, rows, columns, 60.0f);
    for (size_t index = 0; index < (size_t)outputs * columns; index++)
        w[index] = to_bf16(0.02f * normal() *
                           (index % 4093 == 5 ? 20.0f : 1.0f));
    double *exact = calloc((size_t)rows * outputs, sizeof(double));
    for (uint32_t row = 0; row < rows; row++)
        for (uint32_t output = 0; output < outputs; output++) {
            double sum = 0.0;
            for (uint32_t k = 0; k < columns; k++)
                sum += (double)from_bf16(x[(size_t)row * columns + k]) *
                       from_bf16(w[(size_t)output * columns + k]);
            exact[(size_t)row * outputs + output] = sum;
        }
    h3_gpu_tensor *input = upload_bf16(gpu, x, (size_t)rows * columns);
    h3_gpu_tensor *wsrc = upload_bf16(gpu, w, (size_t)outputs * columns);
    h3_gpu_tensor *wq = h3_gpu_tensor_new_i8(gpu, (size_t)outputs * columns);
    h3_gpu_tensor *ws = h3_gpu_tensor_new_f32(gpu, outputs);
    h3_gpu_tensor *xq = h3_gpu_tensor_new_i8(gpu, (size_t)rows * columns);
    h3_gpu_tensor *xs = h3_gpu_tensor_new_f32(gpu, rows);
    h3_gpu_tensor *y = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * outputs);
    uint16_t *result = malloc((size_t)rows * outputs * sizeof(*result));
    for (int rotate = 0; rotate <= 1; rotate++) {
        if (rotate) {
            int8_t *codes = malloc((size_t)outputs * columns);
            float *scales = malloc(outputs * sizeof(float));
            h3_rot_quantize_matrix(H3_ROT_MATRIX_FC1, H3_ROT_FC1, w, outputs,
                                   columns, codes, scales);
            h3_gpu_tensor_write_i8(wq, codes, (size_t)outputs * columns);
            h3_gpu_tensor_write_f32(ws, scales, outputs);
            free(codes); free(scales);
            GPU_OK(h3_gpu_begin(gpu) &&
                   h3_gpu_linear_int8_bf16_rotated(gpu, y, xq, xs, input, wq,
                                                   ws, rows, columns, outputs,
                                                   0, 1) &&
                   h3_gpu_submit(gpu));
        } else {
            GPU_OK(h3_gpu_begin(gpu) &&
                   h3_gpu_quantize_weight_int8(gpu, wq, ws, wsrc, outputs,
                                               columns) &&
                   h3_gpu_linear_int8_bf16(gpu, y, xq, xs, input, wq, ws,
                                           rows, columns, outputs, 0) &&
                   h3_gpu_submit(gpu));
        }
        h3_gpu_tensor_read_bf16(y, result, (size_t)rows * outputs);
        double diff = 0.0, norm = 0.0;
        for (size_t index = 0; index < (size_t)rows * outputs; index++) {
            double d = from_bf16(result[index]) - exact[index];
            diff += d * d;
            norm += exact[index] * exact[index];
        }
        printf("  synthetic W8A8 GEMM (outlier channels), %-7s rel L2 %.4e\n",
               rotate ? "rot256" : "plain", sqrt(diff / norm));
    }
    free(x); free(w); free(exact); free(result);
    h3_gpu_tensor_free(input); h3_gpu_tensor_free(wsrc);
    h3_gpu_tensor_free(wq); h3_gpu_tensor_free(ws); h3_gpu_tensor_free(xq);
    h3_gpu_tensor_free(xs); h3_gpu_tensor_free(y);
}

/* ---- timing ------------------------------------------------------------ */

typedef int (*op_fn)(h3_gpu *gpu, void *state, int rotate);

static double time_op(h3_gpu *gpu, op_fn op, void *state, int rotate,
                      int repeats) {
    double best = 1e30;
    for (int trial = 0; trial < 5; trial++) {
        double started = now();
        GPU_OK(h3_gpu_begin(gpu));
        for (int index = 0; index < repeats; index++)
            if (!op(gpu, state, rotate)) {
                fprintf(stderr, "op: %s\n", h3_gpu_error(gpu));
                exit(1);
            }
        GPU_OK(h3_gpu_submit(gpu));
        double elapsed = (now() - started) / repeats;
        if (trial && elapsed < best) best = elapsed;
    }
    return best;
}

typedef struct {
    adaln_case adaln;
} fused_state;

static int op_fused(h3_gpu *gpu, void *opaque, int rotate) {
    fused_state *s = opaque;
    adaln_case *c = &s->adaln;
    /* In place, as in the DiT (gated -> residual of the next call). */
    return h3_gpu_gate_adaln_quantize_int8(
        gpu, c->gated, c->q, c->qs, c->residual, c->branch, c->norm,
        c->modulation, c->modulation, c->map, c->rows, c->padded, HIDDEN, 6,
        2, 3, 4, 1e-5f, rotate);
}

typedef struct {
    h3_gpu_tensor *x, *w, *ws, *xq, *xs, *y;
    uint32_t rows, k, n;
    int head_major;
} linear_state;

static int op_linear(h3_gpu *gpu, void *opaque, int rotate) {
    linear_state *s = opaque;
    if (s->head_major)
        return h3_gpu_linear_int8_head_major_bf16(
            gpu, s->y, s->xq, s->xs, s->x, s->w, s->ws, s->rows, HEADS,
            HEAD_DIM, s->n, rotate);
    return h3_gpu_linear_int8_bf16_rotated(gpu, s->y, s->xq, s->xs, s->x,
                                           s->w, s->ws, s->rows, s->k, s->n,
                                           0, rotate);
}

typedef struct {
    h3_gpu_tensor *x, *act, *q, *qs, *fc1, *fc1s, *fc2, *fc2s, *out;
    uint32_t rows;
    int row_fc2;
} mlp_state;

static int op_mlp(h3_gpu *gpu, void *opaque, int rotate) {
    mlp_state *s = opaque;
    return h3_gpu_mlp_int8_bf16(gpu, s->out, s->act, s->q, s->qs, s->x,
                                s->fc1, s->fc1s, s->fc2, s->fc2s, NULL, NULL,
                                s->rows, HIDDEN, FFN, HIDDEN, 0, 0,
                                s->row_fc2, 0,
                                rotate ? H3_GPU_ROTATE_FC2_INPUT : 0u);
}

static void time_rows(h3_gpu *gpu, uint32_t rows, int repeats) {
    uint32_t padded = (rows + 127u) & ~127u;
    printf("M = %u\n", rows);
    /* qkv / fc1 input: fused gate + AdaLN + quantize, and the GEMMs fed. */
    fused_state fused = {make_adaln(gpu, rows)};
    double fused_plain = time_op(gpu, op_fused, &fused, 0, repeats);
    double fused_rot = time_op(gpu, op_fused, &fused, 1, repeats);
    free_adaln(&fused.adaln);
    size_t big = (size_t)padded * FFN;
    linear_state linear = {
        h3_gpu_tensor_new_bf16(gpu, big), NULL, NULL,
        h3_gpu_tensor_new_i8(gpu, big), h3_gpu_tensor_new_f32(gpu, padded),
        NULL, rows, HIDDEN, 0, 0};
    const struct { const char *name; uint32_t k, n; } gemms[] = {
        {"qkv", HIDDEN, 3 * INNER}, {"fc1", HIDDEN, 2 * FFN}};
    for (unsigned g = 0; g < 2; g++) {
        linear.k = gemms[g].k;
        linear.n = gemms[g].n;
        linear.w = h3_gpu_tensor_new_i8(gpu, (size_t)linear.n * linear.k);
        linear.ws = h3_gpu_tensor_new_f32(gpu, linear.n);
        linear.y = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * linear.n);
        double gemm = time_op(gpu, op_linear, &linear, 0, repeats);
        double overhead = (fused_rot - fused_plain) / (fused_plain + gemm);
        printf("  %-4s fused AdaLN quantizer plain %.3f ms, rot %.3f ms; "
               "int8 linear (quantize+GEMM, K=%u N=%u) %.3f ms; overhead "
               "%.2f%%  %s\n", gemms[g].name, 1e3 * fused_plain,
               1e3 * fused_rot, linear.k, linear.n, 1e3 * gemm,
               100 * overhead, overhead < 0.02 ? "PASS" : "FAIL");
        if (overhead >= 0.02) failures++;
        h3_gpu_tensor_free(linear.w);
        h3_gpu_tensor_free(linear.ws);
        h3_gpu_tensor_free(linear.y);
    }
    /* out256: head-major quantizer + GEMM K=7168 -> 5376, whole op. */
    linear.k = INNER;
    linear.n = HIDDEN;
    linear.head_major = 1;
    linear.w = h3_gpu_tensor_new_i8(gpu, (size_t)linear.n * linear.k);
    linear.ws = h3_gpu_tensor_new_f32(gpu, linear.n);
    linear.y = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * linear.n);
    double out_plain = time_op(gpu, op_linear, &linear, 0, repeats);
    double out_rot = time_op(gpu, op_linear, &linear, 1, repeats);
    double out_overhead = (out_rot - out_plain) / out_plain;
    printf("  out  head-major quantize+GEMM plain %.3f ms, rot256 %.3f ms; "
           "overhead %.2f%%  %s\n", 1e3 * out_plain, 1e3 * out_rot,
           100 * out_overhead, out_overhead < 0.02 ? "PASS" : "FAIL");
    if (out_overhead >= 0.02) failures++;
    h3_gpu_tensor_free(linear.w); h3_gpu_tensor_free(linear.ws);
    h3_gpu_tensor_free(linear.y);
    /* fc2: whole int8 MLP (FC1+SwiGLU, quantize, FC2), rot only on FC2. */
    for (int row_fc2 = 0; row_fc2 <= 1; row_fc2++) {
        mlp_state mlp = {
            linear.x, h3_gpu_tensor_new_bf16(gpu, (size_t)rows * FFN),
            linear.xq, h3_gpu_tensor_new_f32(gpu, (size_t)padded * 14),
            h3_gpu_tensor_new_i8(gpu, (size_t)2 * FFN * HIDDEN),
            h3_gpu_tensor_new_f32(gpu, 2 * FFN),
            h3_gpu_tensor_new_i8(gpu, (size_t)HIDDEN * FFN),
            h3_gpu_tensor_new_f32(gpu, HIDDEN),
            h3_gpu_tensor_new_bf16(gpu, (size_t)rows * HIDDEN), rows,
            row_fc2};
        double plain = time_op(gpu, op_mlp, &mlp, 0, repeats);
        double rot = time_op(gpu, op_mlp, &mlp, 1, repeats);
        double overhead = (rot - plain) / plain;
        printf("  fc2  whole int8 MLP (%s FC2) plain %.3f ms, rot256 FC2 "
               "%.3f ms; overhead %.2f%%  %s\n",
               row_fc2 ? "row" : "grouped1024", 1e3 * plain, 1e3 * rot,
               100 * overhead, overhead < 0.02 ? "PASS" : "FAIL");
        if (overhead >= 0.02) failures++;
        h3_gpu_tensor_free(mlp.act); h3_gpu_tensor_free(mlp.qs);
        h3_gpu_tensor_free(mlp.fc1); h3_gpu_tensor_free(mlp.fc1s);
        h3_gpu_tensor_free(mlp.fc2); h3_gpu_tensor_free(mlp.fc2s);
        h3_gpu_tensor_free(mlp.out);
    }
    h3_gpu_tensor_free(linear.x); h3_gpu_tensor_free(linear.xq);
    h3_gpu_tensor_free(linear.xs);
}

int main(int argc, char **argv) {
    int quick = argc > 1 && !strcmp(argv[1], "--quick");
    char error[1024];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return 2; }
    if (!h3_gpu_has_int8_mlp(gpu)) {
        fprintf(stderr, "needs the M5 int8 TensorOps kernels\n");
        return 2;
    }
    printf("1. Hadamard identities\n");
    check_identities();
    printf("2. GPU rotated quantizers vs CPU reference\n");
    check_fused_adaln(gpu);
    check_row_quantizer(gpu, HIDDEN);
    check_row_quantizer(gpu, INNER);
    check_row_quantizer(gpu, FFN);
    check_head_major(gpu);
    check_fc2(gpu, 0);
    check_fc2(gpu, 1);
    printf("3. synthetic GEMM error\n");
    gemm_error(gpu);
    printf("4. timing (best of 4 after warm-up, %s)\n",
           quick ? "quick" : "20 repeats per command buffer");
    time_rows(gpu, 9472, quick ? 3 : 20);
    time_rows(gpu, 37797, quick ? 2 : 10);
    h3_gpu_free(gpu);
    printf("%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures,
           failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
