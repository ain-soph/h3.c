#include "h3_rot.h"

#include <dispatch/dispatch.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ROT_HEADS = 56, ROT_HEAD_DIM = 128 };

static const struct {
    const char *name;
    unsigned bit;
} rot_names[] = {
    {"qkv", H3_ROT_QKV}, {"fc1", H3_ROT_FC1}, {"out256", H3_ROT_OUT256},
    {"out128", H3_ROT_OUT128}, {"fc2", H3_ROT_FC2},
};

int h3_rot_parse(const char *text, unsigned *mask, char *error,
                 size_t error_size) {
    *mask = 0;
    if (!text || !*text || !strcmp(text, "none") || !strcmp(text, "0"))
        return 1;
    if (!strcmp(text, "all")) {
        *mask = H3_ROT_QKV | H3_ROT_FC1 | H3_ROT_OUT256 | H3_ROT_FC2;
        return 1;
    }
    const char *cursor = text;
    while (*cursor) {
        const char *end = strchr(cursor, ',');
        size_t length = end ? (size_t)(end - cursor) : strlen(cursor);
        unsigned bit = 0;
        for (size_t index = 0; index < sizeof(rot_names) / sizeof(*rot_names);
             index++)
            if (strlen(rot_names[index].name) == length &&
                !strncmp(rot_names[index].name, cursor, length))
                bit = rot_names[index].bit;
        if (!bit) {
            snprintf(error, error_size,
                     "unknown rotation class '%.*s' (qkv, fc1, out256, "
                     "out128, fc2, all, none)", (int)length, cursor);
            return 0;
        }
        *mask |= bit;
        if (!end) break;
        cursor = end + 1;
    }
    if ((*mask & H3_ROT_OUT256) && (*mask & H3_ROT_OUT128)) {
        snprintf(error, error_size, "out256 and out128 are exclusive");
        return 0;
    }
    return 1;
}

void h3_rot_format(unsigned mask, char *text, size_t size) {
    size_t length = 0;
    if (size) text[0] = '\0';
    for (size_t index = 0; index < sizeof(rot_names) / sizeof(*rot_names);
         index++) {
        if (!(mask & rot_names[index].bit)) continue;
        int written = snprintf(text + length, size - length, "%s%s",
                               length ? "," : "", rot_names[index].name);
        if (written < 0 || (size_t)written >= size - length) return;
        length += (size_t)written;
    }
    if (!length) snprintf(text, size, "none");
}

int h3_rot_read_marker(const char *marker_path, unsigned *mask, char *error,
                       size_t error_size) {
    *mask = 0;
    FILE *file = fopen(marker_path, "rb");
    if (!file) {
        snprintf(error, error_size, "cannot read %s", marker_path);
        return 0;
    }
    char text[4096];
    size_t count = fread(text, 1, sizeof(text) - 1, file);
    fclose(file);
    text[count] = '\0';
    const char *key = strstr(text, "\"rot\"");
    if (!key) return 1;
    const char *open = strchr(key + 5, '"');
    const char *close = open ? strchr(open + 1, '"') : NULL;
    if (!open || !close || close - open - 1 >= 256) {
        snprintf(error, error_size, "%s: malformed \"rot\" field",
                 marker_path);
        return 0;
    }
    char list[256];
    memcpy(list, open + 1, (size_t)(close - open - 1));
    list[close - open - 1] = '\0';
    return h3_rot_parse(list, mask, error, error_size);
}

int h3_rot_matrix_rotated(h3_rot_matrix matrix, unsigned mask) {
    switch (matrix) {
    case H3_ROT_MATRIX_QKV: return (mask & (H3_ROT_QKV | H3_ROT_OUT128)) != 0;
    case H3_ROT_MATRIX_OUT:
        return (mask & (H3_ROT_OUT256 | H3_ROT_OUT128)) != 0;
    case H3_ROT_MATRIX_FC1: return (mask & H3_ROT_FC1) != 0;
    case H3_ROT_MATRIX_FC2: return (mask & H3_ROT_FC2) != 0;
    }
    return 0;
}

/* One radix-4 stage of H4 / 2 over the digit with this stride:
 * y[a] = (x0 + x1 + x2 + x3 - 2 x[3 - a]) / 2. */
static void h4_stage(float *values, unsigned count, unsigned stride) {
    for (unsigned base = 0; base < count; base += 4 * stride)
        for (unsigned offset = 0; offset < stride; offset++) {
            float *x = values + base + offset;
            float x0 = x[0], x1 = x[stride], x2 = x[2 * stride],
                  x3 = x[3 * stride];
            float sum = x0 + x1 + x2 + x3;
            x[0] = 0.5f * (sum - 2.0f * x3);
            x[stride] = 0.5f * (sum - 2.0f * x2);
            x[2 * stride] = 0.5f * (sum - 2.0f * x1);
            x[3 * stride] = 0.5f * (sum - 2.0f * x0);
        }
}

void h3_hadamard256_f32(float *values) {
    for (unsigned stride = 1; stride < 256; stride *= 4)
        h4_stage(values, 256, stride);
}

void h3_hadamard128_f32(float *values) {
    for (unsigned stride = 1; stride < 64; stride *= 4)
        h4_stage(values, 128, stride);
    const float half_sqrt2 = 0.70710678118654752f;
    for (unsigned offset = 0; offset < 64; offset++) {
        float a = values[offset], b = values[offset + 64];
        values[offset] = (a + b) * half_sqrt2;
        values[offset + 64] = (a - b) * half_sqrt2;
    }
}

/* Same transforms applied to whole rows: rows[i] for i in 0..127 are the
 * vector being mixed, each of `columns` floats. */
static void h4_stage_rows(float *rows, size_t columns, unsigned stride) {
    for (unsigned base = 0; base < ROT_HEAD_DIM; base += 4 * stride)
        for (unsigned offset = 0; offset < stride; offset++) {
            float *r0 = rows + (size_t)(base + offset) * columns;
            float *r1 = r0 + (size_t)stride * columns;
            float *r2 = r1 + (size_t)stride * columns;
            float *r3 = r2 + (size_t)stride * columns;
            for (size_t column = 0; column < columns; column++) {
                float x0 = r0[column], x1 = r1[column], x2 = r2[column],
                      x3 = r3[column];
                float sum = x0 + x1 + x2 + x3;
                r0[column] = 0.5f * (sum - 2.0f * x3);
                r1[column] = 0.5f * (sum - 2.0f * x2);
                r2[column] = 0.5f * (sum - 2.0f * x1);
                r3[column] = 0.5f * (sum - 2.0f * x0);
            }
        }
}

static void hadamard128_rows(float *rows, size_t columns) {
    for (unsigned stride = 1; stride < 64; stride *= 4)
        h4_stage_rows(rows, columns, stride);
    const float half_sqrt2 = 0.70710678118654752f;
    for (unsigned offset = 0; offset < 64; offset++) {
        float *a = rows + (size_t)offset * columns;
        float *b = rows + (size_t)(offset + 64) * columns;
        for (size_t column = 0; column < columns; column++) {
            float x = a[column], y = b[column];
            a[column] = (x + y) * half_sqrt2;
            b[column] = (x - y) * half_sqrt2;
        }
    }
}

static float bf16_to_f32(uint16_t bits) {
    uint32_t word = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &word, sizeof(value));
    return value;
}

static void quantize_row(const float *row, uint32_t columns,
                         int8_t *quantized, float *scale) {
    float max_abs = 0.0f;
    for (uint32_t column = 0; column < columns; column++)
        max_abs = fmaxf(max_abs, fabsf(row[column]));
    *scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
    float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
    for (uint32_t column = 0; column < columns; column++) {
        float value = rintf(row[column] * inverse);
        quantized[column] = (int8_t)(value > 127.0f ? 127 :
                                     value < -127.0f ? -127 : (int)value);
    }
}

/* Column (K) rotation of one F32 row. */
static void rotate_row(float *row, uint32_t columns, int k256, int k128) {
    if (k256)
        for (uint32_t group = 0; group < columns; group += 256)
            h3_hadamard256_f32(row + group);
    if (k128)
        for (uint32_t group = 0; group < columns; group += 128)
            h3_hadamard128_f32(row + group);
}

typedef struct {
    h3_rot_matrix matrix;
    unsigned mask;
    const uint16_t *weight;
    uint32_t rows;
    uint32_t columns;
    int8_t *quantized;
    float *scales;
    uint32_t rows_per_unit;
    int failed;
} rot_job;

static void rot_unit(void *opaque, size_t unit) {
    rot_job *job = opaque;
    uint32_t columns = job->columns;
    int k256 = 0, k128 = 0, mix_v = 0;
    switch (job->matrix) {
    case H3_ROT_MATRIX_QKV:
        k256 = (job->mask & H3_ROT_QKV) != 0;
        mix_v = (job->mask & H3_ROT_OUT128) != 0;
        break;
    case H3_ROT_MATRIX_OUT:
        k256 = (job->mask & H3_ROT_OUT256) != 0;
        k128 = (job->mask & H3_ROT_OUT128) != 0;
        break;
    case H3_ROT_MATRIX_FC1: k256 = (job->mask & H3_ROT_FC1) != 0; break;
    case H3_ROT_MATRIX_FC2: k256 = (job->mask & H3_ROT_FC2) != 0; break;
    }
    uint32_t first = (uint32_t)unit * job->rows_per_unit;
    uint32_t count = job->rows_per_unit;
    if (first >= job->rows) return;
    if (count > job->rows - first) count = job->rows - first;
    float *buffer = malloc((size_t)count * columns * sizeof(float));
    if (!buffer) {
        job->failed = 1;
        return;
    }
    for (uint32_t row = 0; row < count; row++) {
        const uint16_t *source = job->weight + (size_t)(first + row) * columns;
        float *destination = buffer + (size_t)row * columns;
        for (uint32_t column = 0; column < columns; column++)
            destination[column] = bf16_to_f32(source[column]);
    }
    /* QKV units are whole heads ([head, q/k/v, dimension] rows): the V block
     * is rows 256..383 of the unit. */
    if (mix_v) hadamard128_rows(buffer + (size_t)2 * ROT_HEAD_DIM * columns,
                                columns);
    for (uint32_t row = 0; row < count; row++) {
        float *values = buffer + (size_t)row * columns;
        rotate_row(values, columns, k256, k128);
        quantize_row(values, columns,
                     job->quantized + (size_t)(first + row) * columns,
                     job->scales + first + row);
    }
    free(buffer);
}

int h3_rot_quantize_matrix(h3_rot_matrix matrix, unsigned mask,
                           const uint16_t *weight, uint32_t rows,
                           uint32_t columns, int8_t *quantized,
                           float *scales) {
    if (!weight || !quantized || !scales || !rows || !columns) return 0;
    if ((mask & H3_ROT_OUT256) && (mask & H3_ROT_OUT128)) return 0;
    rot_job job = {matrix, mask, weight, rows, columns, quantized, scales,
                   64, 0};
    if (matrix == H3_ROT_MATRIX_QKV) {
        if (rows != ROT_HEADS * 3 * ROT_HEAD_DIM) return 0;
        job.rows_per_unit = 3 * ROT_HEAD_DIM;
    }
    if (columns % 256) return 0;
    size_t units = (rows + job.rows_per_unit - 1) / job.rows_per_unit;
    dispatch_apply_f(units, dispatch_get_global_queue(
                         QOS_CLASS_USER_INITIATED, 0), &job, rot_unit);
    return !job.failed;
}
