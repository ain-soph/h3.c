/* Track B: every fused video VAE op against the kernel chain it replaces, on
 * random data with denormals, infinities, signed zeros and NaNs mixed in.
 * The comparison is on the raw bits (uint16 for BF16 outputs, uint32 for F32):
 * any difference fails.
 * Usage: h3_bitexact_vae_ops [SEED...]   (default seeds 1 2 3; run from the
 * checkout so h3_shaders.metal / h3_attention.metal are found) */
#include "h3_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HIDDEN = 2048, HEADS = 32, HEAD_DIM = 64, INNER = HEADS * HEAD_DIM,
       FFN = 8192, ROPE_HALF = 24 };

static uint64_t rng;
static uint64_t next(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return rng;
}

static float from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* A special value: +-0, a denormal, +-inf, a NaN with a random payload, or a
 * very large / very small normal. */
static float special(void) {
    uint64_t r = next();
    uint32_t sign = (uint32_t)(r >> 63) << 31;
    switch ((r >> 8) % 7) {
    case 0: return from_bits(sign);
    case 1: return from_bits(sign | (uint32_t)(((r >> 16) % 0x7fffffu) + 1));
    case 2: return from_bits(sign | 0x7f800000u);
    case 3: return from_bits(sign | 0x7fc00000u | (uint32_t)((r >> 20) & 0x3fffffu));
    case 4: return from_bits(sign | 0x7f000000u | (uint32_t)((r >> 16) & 0x7fffffu));
    case 5: return from_bits(sign | 0x00800000u | (uint32_t)((r >> 16) & 0x7fffffu));
    default: return from_bits(sign | 0x3f800000u | (uint32_t)((r >> 16) & 0x7fffffu));
    }
}

/* Mostly activation-like values over 2^-14 .. 2^7; in one row of every
 * `special_row_period` (0: none) about one value in 64 is special. */
static void fill(float *values, size_t rows, size_t width,
                 unsigned special_row_period) {
    for (size_t row = 0; row < rows; row++) {
        int specials = special_row_period && row % special_row_period == 0;
        for (size_t column = 0; column < width; column++) {
            uint64_t r = next();
            float value;
            if (specials && (r & 63) == 0) {
                value = special();
            } else {
                uint32_t sign = (uint32_t)(r >> 63) << 31;
                uint32_t exponent = (uint32_t)(113 + (r >> 40) % 22);
                value = from_bits(sign | exponent << 23 |
                                  (uint32_t)(r & 0x7fffffu));
            }
            values[row * width + column] = value;
        }
    }
}

static h3_gpu *gpu;
static size_t compared, failures;

static h3_gpu_tensor *upload(const float *values, size_t count) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_f32(gpu, values, count);
    if (!tensor) { fprintf(stderr, "upload failed\n"); exit(2); }
    return tensor;
}

static h3_gpu_tensor *bf16(size_t count) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, count);
    if (!tensor) { fprintf(stderr, "allocation failed\n"); exit(2); }
    return tensor;
}

static h3_gpu_tensor *f32(size_t count) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_f32(gpu, count);
    if (!tensor) { fprintf(stderr, "allocation failed\n"); exit(2); }
    return tensor;
}

static void run(const char *label, int ok) {
    if (!ok || !h3_gpu_submit(gpu)) {
        fprintf(stderr, "%s: %s\n", label, h3_gpu_error(gpu));
        exit(2);
    }
}

static void begin(void) {
    if (!h3_gpu_begin(gpu)) { fprintf(stderr, "begin failed\n"); exit(2); }
}

static void compare_bf16(const char *label, const h3_gpu_tensor *reference,
                         const h3_gpu_tensor *candidate, size_t count) {
    uint16_t *a = malloc(count * sizeof(*a)), *b = malloc(count * sizeof(*b));
    if (!a || !b || !h3_gpu_tensor_read_bf16(reference, a, count) ||
        !h3_gpu_tensor_read_bf16(candidate, b, count)) exit(2);
    size_t diff = 0, first = 0;
    for (size_t i = 0; i < count; i++)
        if (a[i] != b[i] && !diff++) first = i;
    printf("  %-34s %11zu BF16 values, %zu differ", label, count, diff);
    if (diff) printf(" (first at %zu: %04x vs %04x)", first, a[first], b[first]);
    printf("\n");
    compared += count;
    failures += diff;
    free(a); free(b);
}

static void compare_f32(const char *label, const h3_gpu_tensor *reference,
                        const h3_gpu_tensor *candidate, size_t count) {
    uint32_t *a = malloc(count * sizeof(*a)), *b = malloc(count * sizeof(*b));
    if (!a || !b || !h3_gpu_tensor_read_f32(reference, (float *)a, count) ||
        !h3_gpu_tensor_read_f32(candidate, (float *)b, count)) exit(2);
    size_t diff = 0, first = 0;
    for (size_t i = 0; i < count; i++)
        if (a[i] != b[i] && !diff++) first = i;
    printf("  %-34s %11zu F32 values, %zu differ", label, count, diff);
    if (diff) printf(" (first at %zu: %08x vs %08x)", first, a[first], b[first]);
    printf("\n");
    compared += count;
    failures += diff;
    free(a); free(b);
}

/* rms_norm_f32 + cast  vs  rms_norm_f32_to_bf16 */
static size_t test_norm(void) {
    const uint32_t rows = 8192;
    size_t count = (size_t)rows * HIDDEN;
    float *x = malloc(count * sizeof(*x)), *w = malloc(HIDDEN * sizeof(*w));
    fill(x, rows, HIDDEN, 8);
    fill(w, 1, HIDDEN, 0);
    w[7] = special(); w[1000] = 0.0f;
    h3_gpu_tensor *input = upload(x, count), *weight = upload(w, HIDDEN);
    h3_gpu_tensor *norm = f32(count), *reference = bf16(count), *fused = bf16(count);
    begin();
    run("reference RMSNorm",
        h3_gpu_rms_norm_f32(gpu, norm, input, weight, rows, HIDDEN, 1e-5f) &&
        h3_gpu_cast_f32_to_bf16(gpu, reference, norm, (uint32_t)count));
    begin();
    run("fused RMSNorm",
        h3_gpu_rms_norm_f32_to_bf16(gpu, fused, input, weight, rows, HIDDEN, 1e-5f));
    compare_bf16("RMSNorm -> BF16", reference, fused, count);
    h3_gpu_tensor_free(input); h3_gpu_tensor_free(weight); h3_gpu_tensor_free(norm);
    h3_gpu_tensor_free(reference); h3_gpu_tensor_free(fused);
    free(x); free(w);
    return count + HIDDEN;
}

/* video_qkv_rope_f32 + 3 casts  vs  video_qkv_rope_bf16 */
static size_t test_qkv_rope(void) {
    const uint32_t rows = 8192;
    size_t count = (size_t)rows * INNER, rope = (size_t)rows * ROPE_HALF;
    float *x = malloc(count * 3 * sizeof(*x));
    float *c = malloc(rope * sizeof(*c)), *s = malloc(rope * sizeof(*s));
    fill(x, (size_t)rows * HEADS, HEAD_DIM * 3, 16);
    fill(c, rows, ROPE_HALF, 64);
    fill(s, rows, ROPE_HALF, 64);
    h3_gpu_tensor *qkv = upload(x, count * 3), *cosine = upload(c, rope),
                  *sine = upload(s, rope);
    h3_gpu_tensor *q = f32(count), *k = f32(count), *v = f32(count);
    h3_gpu_tensor *rq = bf16(count), *rk = bf16(count), *rv = bf16(count);
    h3_gpu_tensor *fq = bf16(count), *fk = bf16(count), *fv = bf16(count);
    begin();
    run("reference QKV RoPE",
        h3_gpu_video_qkv_rope_f32(gpu, q, k, v, qkv, cosine, sine, rows, HEADS,
                                  HEAD_DIM, ROPE_HALF, 1e-5f) &&
        h3_gpu_cast_f32_to_bf16(gpu, rq, q, (uint32_t)count) &&
        h3_gpu_cast_f32_to_bf16(gpu, rk, k, (uint32_t)count) &&
        h3_gpu_cast_f32_to_bf16(gpu, rv, v, (uint32_t)count));
    begin();
    run("fused QKV RoPE",
        h3_gpu_video_qkv_rope_bf16(gpu, fq, fk, fv, qkv, cosine, sine, rows,
                                   HEADS, HEAD_DIM, ROPE_HALF, 1e-5f));
    compare_bf16("QK norm + RoPE -> BF16 query", rq, fq, count);
    compare_bf16("QK norm + RoPE -> BF16 key", rk, fk, count);
    compare_bf16("QK norm + RoPE -> BF16 value", rv, fv, count);
    h3_gpu_tensor *all[] = {qkv, cosine, sine, q, k, v, rq, rk, rv, fq, fk, fv};
    for (size_t i = 0; i < sizeof(all) / sizeof(*all); i++) h3_gpu_tensor_free(all[i]);
    free(x); free(c); free(s);
    return count * 3 + rope * 2;
}

/* swiglu_f32 + cast  vs  swiglu_f32_to_bf16 */
static size_t test_swiglu(void) {
    const uint32_t rows = 4096;
    size_t count = (size_t)rows * FFN;
    float *x = malloc(count * 2 * sizeof(*x));
    fill(x, rows, FFN * 2, 4);
    h3_gpu_tensor *fused_input = upload(x, count * 2);
    h3_gpu_tensor *activated = f32(count), *reference = bf16(count),
                  *fused = bf16(count);
    begin();
    run("reference SwiGLU",
        h3_gpu_swiglu_f32(gpu, activated, fused_input, rows, FFN) &&
        h3_gpu_cast_f32_to_bf16(gpu, reference, activated, (uint32_t)count));
    begin();
    run("fused SwiGLU", h3_gpu_swiglu_f32_to_bf16(gpu, fused, fused_input, rows, FFN));
    compare_bf16("SwiGLU -> BF16", reference, fused, count);
    h3_gpu_tensor_free(fused_input); h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(reference); h3_gpu_tensor_free(fused);
    free(x);
    return count * 2;
}

/* linear_bf16_weight_f32 (casts its F32 input)  vs  cast + linear_bf16_input_f32 */
static size_t test_linear(uint32_t rows, uint32_t in, uint32_t out,
                          const char *label) {
    size_t padded = ((size_t)rows + 127) & ~(size_t)127;
    float *x = malloc((size_t)rows * in * sizeof(*x));
    float *w = malloc((size_t)out * in * sizeof(*w));
    float *b = malloc(out * sizeof(*b));
    fill(x, rows, in, 32);
    fill(w, out, in, 0);
    for (size_t i = 0; i < (size_t)out * in; i++) w[i] *= 0.01f;
    fill(b, 1, out, 0);
    h3_gpu_tensor *input = upload(x, (size_t)rows * in);
    h3_gpu_tensor *weight_f32 = upload(w, (size_t)out * in), *bias = upload(b, out);
    h3_gpu_tensor *weight = bf16((size_t)out * in), *scratch = bf16(padded * in),
                  *x16 = bf16(padded * in);
    h3_gpu_tensor *reference = f32((size_t)rows * out), *fused = f32((size_t)rows * out);
    begin();
    run("weight to BF16",
        h3_gpu_cast_f32_to_bf16(gpu, weight, weight_f32, (uint32_t)((size_t)out * in)));
    begin();
    run("reference linear",
        h3_gpu_linear_bf16_weight_f32(gpu, reference, input, weight, bias, scratch,
                                      rows, in, out));
    begin();
    run("BF16-input linear",
        h3_gpu_cast_f32_to_bf16(gpu, x16, input, (uint32_t)((size_t)rows * in)) &&
        h3_gpu_linear_bf16_input_f32(gpu, fused, x16, weight, bias, rows, in, out));
    compare_f32(label, reference, fused, (size_t)rows * out);
    h3_gpu_tensor *all[] = {input, weight_f32, bias, weight, scratch, x16, reference, fused};
    for (size_t i = 0; i < sizeof(all) / sizeof(*all); i++) h3_gpu_tensor_free(all[i]);
    free(x); free(w); free(b);
    return (size_t)rows * in + (size_t)out * in + out;
}

/* Encoder: group_norm_silu + pad  vs  group_norm_silu_pad (padded output),
 * and the arena Conv3d (cached weight data, oversized output buffer) vs the
 * per-call Conv3d. */
static size_t test_encoder_norm_pad(uint32_t depth, uint32_t height,
                                    uint32_t width, uint32_t channels,
                                    uint32_t front, uint32_t before,
                                    uint32_t after, const char *label) {
    size_t count = (size_t)depth * height * width * channels;
    size_t padded = (size_t)(depth + front) * (height + before + after) *
                    (width + before + after) * channels;
    float *x = malloc(count * sizeof(*x));
    float *w = malloc(channels * sizeof(*w)), *b = malloc(channels * sizeof(*b));
    fill(x, (size_t)depth * height, (size_t)width * channels, 97);
    fill(w, 1, channels, 0);
    fill(b, 1, channels, 0);
    h3_gpu_tensor *input = upload(x, count), *weight = upload(w, channels),
                  *bias = upload(b, channels);
    h3_gpu_tensor *norm = f32(count), *reference = f32(padded),
                  *fused = f32(padded + 4096);
    begin();
    run("reference encoder norm + pad",
        h3_gpu_vae_encoder_group_norm_silu_f32(gpu, norm, input, weight, bias, 1,
            depth, height, width, channels, 32, 1e-6f) &&
        h3_gpu_vae_encoder_pad_f32(gpu, reference, norm, 1, depth, height, width,
            channels, front, before, after, before, after));
    begin();
    run("fused encoder norm + pad",
        h3_gpu_vae_encoder_group_norm_silu_pad_f32(gpu, fused, input, weight, bias,
            1, depth, height, width, channels, 32, 1e-6f, front, before, after,
            before, after));
    compare_f32(label, reference, fused, padded);
    h3_gpu_tensor *all[] = {input, weight, bias, norm, reference, fused};
    for (size_t i = 0; i < sizeof(all) / sizeof(*all); i++) h3_gpu_tensor_free(all[i]);
    free(x); free(w); free(b);
    return count + 2 * channels;
}

static size_t test_encoder_conv(void) {
    const uint32_t depth = 7, height = 34, width = 34, cin = 128, cout = 128;
    size_t count = (size_t)depth * height * width * cin;
    size_t weights = (size_t)cout * cin * 27;
    size_t out = (size_t)(depth - 2) * (height - 2) * (width - 2) * cout;
    float *x = malloc(count * sizeof(*x)), *w = malloc(weights * sizeof(*w)),
          *b = malloc(cout * sizeof(*b));
    fill(x, (size_t)depth * height, (size_t)width * cin, 0);
    fill(w, cout, cin * 27, 0);
    for (size_t i = 0; i < weights; i++) w[i] *= 0.01f;
    fill(b, 1, cout, 0);
    h3_gpu_tensor *input = upload(x, count), *weight = upload(w, weights),
                  *bias = upload(b, cout);
    h3_gpu_tensor *reference = f32(out), *arena = f32(out * 2);
    begin();
    run("reference Conv3d",
        h3_gpu_conv3d_f32(gpu, reference, input, weight, bias, 1, depth, height,
                          width, cin, cout, 3, 3, 3, 1, 1, 1));
    for (int repeat = 0; repeat < 2; repeat++) {   /* second call: cached data */
        begin();
        run("arena Conv3d",
            h3_gpu_conv3d_f32_ex(gpu, arena, input, weight, bias, 1, depth, height,
                                 width, cin, cout, 3, 3, 3, 1, 1, 1, 0));
    }
    compare_f32("Conv3d, cached weights + big buffer", reference, arena, out);
    h3_gpu_tensor *all[] = {input, weight, bias, reference, arena};
    for (size_t i = 0; i < sizeof(all) / sizeof(*all); i++) h3_gpu_tensor_free(all[i]);
    free(x); free(w); free(b);
    return count + weights + cout;
}

/* The SDPA output skips BF16 -> F32 -> BF16: identity on all 65536 patterns. */
static void test_round_trip(void) {
    size_t count = 65536 * 16;
    uint16_t *values = malloc(count * sizeof(*values));
    for (size_t i = 0; i < count; i++) values[i] = (uint16_t)(i & 0xffff);
    h3_gpu_tensor *source = h3_gpu_tensor_from_bf16(gpu, values, count);
    h3_gpu_tensor *wide = f32(count), *back = bf16(count);
    begin();
    run("BF16 round trip",
        h3_gpu_cast_bf16_to_f32(gpu, wide, source, (uint32_t)count) &&
        h3_gpu_cast_f32_to_bf16(gpu, back, wide, (uint32_t)count));
    compare_bf16("BF16 -> F32 -> BF16 (all patterns)", source, back, count);
    h3_gpu_tensor_free(source); h3_gpu_tensor_free(wide); h3_gpu_tensor_free(back);
    free(values);
}

int main(int argc, char **argv) {
    char error[512];
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return 2; }
    if (!h3_gpu_has_nax_extras(gpu)) {
        fprintf(stderr, "needs the M5 NAX extras (TensorOps)\n");
        return 2;
    }
    int seeds = argc > 1 ? argc - 1 : 3;
    for (int index = 0; index < seeds; index++) {
        unsigned long seed = argc > 1 ? strtoul(argv[index + 1], NULL, 10) :
                                        (unsigned long)index + 1;
        rng = 0x9e3779b97f4a7c15ull ^ (seed * 0xbf58476d1ce4e5b9ull);
        if (!rng) rng = 1;
        size_t generated = 0;
        printf("seed %lu\n", seed);
        generated += test_norm();
        generated += test_qkv_rope();
        generated += test_swiglu();
        generated += test_linear(2805, HIDDEN, INNER * 3, "QKV linear, BF16 input");
        generated += test_linear(2805, FFN, HIDDEN, "MLP output linear, BF16 input");
        generated += test_encoder_norm_pad(9, 64, 64, 128, 2, 1, 1,
                                           "encoder norm+pad (2,1,1)");
        generated += test_encoder_norm_pad(5, 32, 48, 256, 2, 0, 1,
                                           "encoder norm+pad (2,0,1)");
        generated += test_encoder_norm_pad(1, 16, 16, 1024, 2, 1, 1,
                                           "encoder norm+pad T=1");
        generated += test_encoder_conv();
        test_round_trip();
        printf("  random input values: %zu\n", generated);
    }
    printf("%s: %zu values compared, %zu differ\n", failures ? "FAIL" : "PASS",
           compared, failures);
    h3_gpu_free(gpu);
    return failures ? 1 : 0;
}
