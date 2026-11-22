#include "h3_rot.h"

#include <dirent.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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
        if (!length || (end && !end[1])) {
            snprintf(error, error_size, "empty rotation class in '%s'", text);
            return 0;
        }
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

/* ---- pack identity: h3-int8-pack.json and safetensors __metadata__ ----
 * Both are flat JSON objects whose values are strings, numbers, true, false
 * or null. The reader is strict: anything else is an error, never a guess. */
enum { ROT_JSON_MAX_PAIRS = 32, ROT_JSON_KEY = 64, ROT_JSON_VALUE = 4096 };

typedef struct {
    char key[ROT_JSON_KEY];
    char value[ROT_JSON_VALUE];   /* strings only; "" otherwise */
    int is_string;
} rot_json_pair;

typedef struct {
    rot_json_pair pairs[ROT_JSON_MAX_PAIRS];
    size_t count;
} rot_json_object;

static const char *rot_json_ws(const char *at, const char *end) {
    while (at < end && (*at == ' ' || *at == '\t' || *at == '\n' ||
                        *at == '\r')) at++;
    return at;
}

/* A JSON string without \u escapes (none of our writers emit them). */
static const char *rot_json_string(const char *at, const char *end,
                                   char *out, size_t size) {
    if (at >= end || *at != '"') return NULL;
    size_t length = 0;
    for (at++; at < end; at++) {
        char c = *at;
        if (c == '"') {
            if (length >= size) return NULL;
            out[length] = '\0';
            return at + 1;
        }
        if ((unsigned char)c < 0x20) return NULL;
        if (c == '\\') {
            if (++at >= end) return NULL;
            switch (*at) {
            case '"': case '\\': case '/': c = *at; break;
            case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case 'n': c = '\n'; break;
            case 'r': c = '\r'; break;
            case 't': c = '\t'; break;
            case 'u': {   /* ASCII only (h3_pack_int8 escapes controls) */
                unsigned code = 0;
                for (int digit = 0; digit < 4; digit++) {
                    if (++at >= end) return NULL;
                    char h = *at;
                    int value = h >= '0' && h <= '9' ? h - '0' :
                                h >= 'a' && h <= 'f' ? h - 'a' + 10 :
                                h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
                    if (value < 0) return NULL;
                    code = code * 16 + (unsigned)value;
                }
                if (!code || code >= 0x80) return NULL;
                c = (char)code;
                break;
            }
            default: return NULL;
            }
        }
        if (length + 1 >= size) return NULL;
        out[length++] = c;
    }
    return NULL;
}

static const char *rot_json_scalar(const char *at, const char *end) {
    static const char *const words[] = {"true", "false", "null"};
    for (size_t index = 0; index < 3; index++) {
        size_t length = strlen(words[index]);
        if ((size_t)(end - at) >= length &&
            !strncmp(at, words[index], length)) return at + length;
    }
    const char *start = at;
    if (at < end && *at == '-') at++;
    while (at < end && ((*at >= '0' && *at <= '9') || *at == '.' ||
                        *at == 'e' || *at == 'E' || *at == '+' || *at == '-'))
        at++;
    return at > start ? at : NULL;
}

/* Parses the object starting at `at`; returns the position after '}'. */
static const char *rot_json_object_parse(const char *at, const char *end,
                                         rot_json_object *object) {
    memset(object, 0, sizeof(*object));
    at = rot_json_ws(at, end);
    if (at >= end || *at++ != '{') return NULL;
    at = rot_json_ws(at, end);
    if (at < end && *at == '}') return at + 1;
    for (;;) {
        if (object->count == ROT_JSON_MAX_PAIRS) return NULL;
        rot_json_pair *pair = &object->pairs[object->count];
        at = rot_json_string(rot_json_ws(at, end), end, pair->key,
                             sizeof(pair->key));
        if (!at) return NULL;
        for (size_t index = 0; index < object->count; index++)
            if (!strcmp(object->pairs[index].key, pair->key)) return NULL;
        at = rot_json_ws(at, end);
        if (at >= end || *at++ != ':') return NULL;
        at = rot_json_ws(at, end);
        if (at < end && *at == '"') {
            at = rot_json_string(at, end, pair->value, sizeof(pair->value));
            pair->is_string = 1;
        } else {
            at = rot_json_scalar(at, end);
        }
        if (!at) return NULL;
        object->count++;
        at = rot_json_ws(at, end);
        if (at >= end) return NULL;
        if (*at == '}') return at + 1;
        if (*at++ != ',') return NULL;
    }
}

static const rot_json_pair *rot_json_find(const rot_json_object *object,
                                          const char *key) {
    for (size_t index = 0; index < object->count; index++)
        if (!strcmp(object->pairs[index].key, key))
            return &object->pairs[index];
    return NULL;
}

/* Reads a whole small file; refuses anything over `limit` bytes. */
static char *rot_read_small(const char *path, size_t limit, size_t *length,
                            char *error, size_t error_size) {
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    struct stat status;
    if (descriptor < 0 || fstat(descriptor, &status) != 0 ||
        !S_ISREG(status.st_mode)) {
        if (descriptor >= 0) close(descriptor);
        snprintf(error, error_size, "cannot read %s", path);
        return NULL;
    }
    if (status.st_size <= 0 || (uint64_t)status.st_size > limit) {
        close(descriptor);
        snprintf(error, error_size, "%s: unexpected size %lld", path,
                 (long long)status.st_size);
        return NULL;
    }
    size_t size = (size_t)status.st_size;
    char *text = malloc(size + 1);
    ssize_t got = text ? pread(descriptor, text, size, 0) : -1;
    close(descriptor);
    if (got != (ssize_t)size) {
        free(text);
        snprintf(error, error_size, "cannot read %s", path);
        return NULL;
    }
    text[size] = '\0';
    *length = size;
    return text;
}

/* format + rot -> mask. v1 has no rotation (and must not name one); v2
 * must name a non-empty list. */
static int rot_identity(const char *where, const char *format, int has_rot,
                        const char *rot, const char *v1, const char *v2,
                        unsigned *mask, char *error, size_t error_size) {
    *mask = 0;
    if (format && !strcmp(format, v1)) {
        if (!has_rot) return 1;
        snprintf(error, error_size, "%s: %s names a rotation", where, v1);
        return 0;
    }
    if (format && !strcmp(format, v2)) {
        char detail[256] = "";
        if (!has_rot || !rot) {
            snprintf(error, error_size, "%s: %s without a rotation list",
                     where, v2);
            return 0;
        }
        if (!h3_rot_parse(rot, mask, detail, sizeof(detail)) || !*mask) {
            snprintf(error, error_size, "%s: invalid rotation '%s'%s%s",
                     where, rot, *detail ? ": " : "", detail);
            *mask = 0;
            return 0;
        }
        return 1;
    }
    snprintf(error, error_size, "%s: unknown pack format '%s' (this binary "
             "reads %s and %s)", where, format ? format : "(none)", v1, v2);
    return 0;
}

int h3_rot_read_marker(const char *marker_path, unsigned *mask, char *error,
                       size_t error_size) {
    *mask = 0;
    size_t length = 0;
    char *text = rot_read_small(marker_path, 65536, &length, error,
                                error_size);
    if (!text) return 0;
    rot_json_object *object = malloc(sizeof(*object));
    const char *end = text + length;
    const char *after = object ? rot_json_object_parse(text, end, object) :
                                 NULL;
    if (!after || rot_json_ws(after, end) != end) {
        snprintf(error, error_size, "%s: malformed pack marker", marker_path);
        free(object);
        free(text);
        return 0;
    }
    const rot_json_pair *format = rot_json_find(object, "format");
    const rot_json_pair *rot = rot_json_find(object, "rot");
    int ok = rot_identity(marker_path,
                          format && format->is_string ? format->value : NULL,
                          rot != NULL,
                          rot && rot->is_string ? rot->value : NULL,
                          "h3-int8-pack-v1", "h3-int8-pack-v2", mask, error,
                          error_size);
    free(object);
    free(text);
    return ok;
}

/* The __metadata__ of one safetensors file h3_pack_int8 wrote. */
static int rot_file_identity(const char *path, unsigned *mask, char *error,
                             size_t error_size) {
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    unsigned char prefix[8];
    if (descriptor < 0 || pread(descriptor, prefix, 8, 0) != 8) {
        if (descriptor >= 0) close(descriptor);
        snprintf(error, error_size, "cannot read %s", path);
        return 0;
    }
    uint64_t header = 0;
    for (int index = 7; index >= 0; index--) header = header << 8 | prefix[index];
    char *json = header && header <= (64u << 20) ? malloc(header + 1) : NULL;
    if (!json || pread(descriptor, json, header, 8) != (ssize_t)header) {
        close(descriptor);
        free(json);
        snprintf(error, error_size, "%s: invalid safetensors header", path);
        return 0;
    }
    close(descriptor);
    json[header] = '\0';
    /* Our writer puts "__metadata__" first; find the key itself (a quoted
     * name, so a tensor name merely containing it does not match). */
    const char *end = json + header;
    const char *key = strstr(json, "\"__metadata__\"");
    const char *at = key ? rot_json_ws(key + 14, end) : NULL;
    rot_json_object *object = malloc(sizeof(*object));
    int ok = 0;
    if (at && at < end && *at == ':' && object &&
        rot_json_object_parse(at + 1, end, object)) {
        const rot_json_pair *format = rot_json_find(object, "h3");
        const rot_json_pair *rot = rot_json_find(object, "h3_rot");
        ok = rot_identity(path,
                          format && format->is_string ? format->value : NULL,
                          rot != NULL,
                          rot && rot->is_string ? rot->value : NULL,
                          "int8-pack-v1", "int8-pack-v2", mask, error,
                          error_size);
    } else {
        snprintf(error, error_size, "%s: no h3_pack_int8 metadata", path);
    }
    free(object);
    free(json);
    return ok;
}

int h3_rot_check_pack(const char *directory, unsigned marker_mask,
                      char *error, size_t error_size) {
    DIR *dir = opendir(directory);
    if (!dir) {
        snprintf(error, error_size, "cannot list %s", directory);
        return 0;
    }
    size_t checked = 0;
    int ok = 1;
    struct dirent *entry;
    while (ok && (entry = readdir(dir))) {
        size_t length = strlen(entry->d_name);
        if (length < 12 || entry->d_name[0] == '.' ||
            strcmp(entry->d_name + length - 12, ".safetensors")) continue;
        char path[2048];
        int written = snprintf(path, sizeof(path), "%s/%s", directory,
                               entry->d_name);
        if (written < 0 || (size_t)written >= sizeof(path)) {
            snprintf(error, error_size, "path too long in %s", directory);
            ok = 0;
            break;
        }
        unsigned mask = 0;
        if (!rot_file_identity(path, &mask, error, error_size)) {
            ok = 0;
            break;
        }
        if (mask != marker_mask) {
            char file_list[96], marker_list[96];
            h3_rot_format(mask, file_list, sizeof(file_list));
            h3_rot_format(marker_mask, marker_list, sizeof(marker_list));
            snprintf(error, error_size, "%s is labelled rotation '%s' but "
                     "h3-int8-pack.json says '%s'; refusing a mixed or "
                     "mislabelled pack (re-pack it; a rotated pack written "
                     "before 2026-11-22 needs tools/stamp_int8_pack_rot.py)",
                     path, file_list, marker_list);
            ok = 0;
            break;
        }
        checked++;
    }
    closedir(dir);
    if (ok && !checked) {
        snprintf(error, error_size, "%s: no safetensors files", directory);
        ok = 0;
    }
    return ok;
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
    atomic_int failed;
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
        atomic_store_explicit(&job->failed, 1, memory_order_relaxed);
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
    rot_job job;
    memset(&job, 0, sizeof(job));
    job.matrix = matrix;
    job.mask = mask;
    job.weight = weight;
    job.rows = rows;
    job.columns = columns;
    job.quantized = quantized;
    job.scales = scales;
    job.rows_per_unit = 64;
    atomic_init(&job.failed, 0);
    if (matrix == H3_ROT_MATRIX_QKV) {
        if (rows != ROT_HEADS * 3 * ROT_HEAD_DIM) return 0;
        job.rows_per_unit = 3 * ROT_HEAD_DIM;
    }
    if (columns % 256) return 0;
    size_t units = (rows + job.rows_per_unit - 1) / job.rows_per_unit;
    dispatch_apply_f(units, dispatch_get_global_queue(
                         QOS_CLASS_USER_INITIATED, 0), &job, rot_unit);
    return !atomic_load(&job.failed);
}

FILE *h3_study_fopen(const char *path, int append) {
    if (!path || !*path) return NULL;
    int descriptor = open(path, O_WRONLY | O_CREAT | O_CLOEXEC | O_NOFOLLOW |
                                (append ? O_APPEND : O_TRUNC), 0644);
    struct stat status;
    if (descriptor < 0) return NULL;
    if (fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode)) {
        close(descriptor);
        return NULL;
    }
    FILE *file = fdopen(descriptor, append ? "a" : "wb");
    if (!file) close(descriptor);
    return file;
}
