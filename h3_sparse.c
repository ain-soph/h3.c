#include "h3_sparse.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

int h3_sparse_params_from_env(h3_sparse_params *params) {
    const char *value = getenv("H3_SPARSE");
    if (!params || !value || !*value || !strcmp(value, "0")) return 0;
    *params = (h3_sparse_params){.t0 = 1, .r0 = 12, .rmin = 2, .sink = 1,
                                 .tile = 8, .dense_steps = 4,
                                 .dense_head = 1, .dense_tail = 1};
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "%s", value);
    for (char *item = strtok(buffer, ","); item; item = strtok(NULL, ",")) {
        char *equals = strchr(item, '=');
        if (!equals) continue;
        *equals = '\0';
        int number = atoi(equals + 1);
        if (!strcmp(item, "t0")) params->t0 = number;
        else if (!strcmp(item, "r0")) params->r0 = number;
        else if (!strcmp(item, "rmin")) params->rmin = number;
        else if (!strcmp(item, "sink")) params->sink = number;
        else if (!strcmp(item, "tile")) params->tile = number > 0 ? number : 8;
        else if (!strcmp(item, "dense_steps")) params->dense_steps = number;
        else if (!strcmp(item, "dense_head")) params->dense_head = number;
        else if (!strcmp(item, "dense_tail")) params->dense_tail = number;
    }
    return 1;
}

static int compare_doubles(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

/* Sorted unique values of one coordinate; *count receives their number. */
static double *unique_axis(const h3_position *positions, size_t start,
                           size_t stop, int axis, size_t *count) {
    size_t rows = stop - start;
    double *values = malloc(rows * sizeof(*values));
    if (!values) return NULL;
    for (size_t row = 0; row < rows; row++) {
        const h3_position *p = &positions[start + row];
        values[row] = axis == 0 ? p->t : axis == 1 ? p->h : p->w;
    }
    qsort(values, rows, sizeof(*values), compare_doubles);
    size_t unique = 0;
    for (size_t row = 0; row < rows; row++)
        if (!unique || values[row] != values[unique - 1])
            values[unique++] = values[row];
    *count = unique;
    return values;
}

static int axis_rank(const double *values, size_t count, double value) {
    const double *found = bsearch(&value, values, count, sizeof(*values),
                                  compare_doubles);
    return found ? (int)(found - values) : -1;
}

typedef struct {
    uint64_t key;
    uint32_t row;
} order_entry;

static int compare_order(const void *left, const void *right) {
    const order_entry *a = left, *b = right;
    if (a->key != b->key) return a->key < b->key ? -1 : 1;
    return (a->row > b->row) - (a->row < b->row);
}

/* A block may straddle frames or spatial tiles; its box is kept per
 * (frame, tile) part so a straddling block does not cover the whole frame. */
enum { MAX_PARTS = 8 };

typedef struct {
    int t, h0, h1, w0, w1;
    int tile;
} block_part;

typedef struct {
    int video;        /* every row of the block is a target-video token */
    int count;
    block_part parts[MAX_PARTS];
} block_box;

static int gap(int a0, int a1, int b0, int b1) {
    int left = b0 - a1, right = a0 - b1;
    int value = left > right ? left : right;
    return value > 0 ? value : 0;
}

static void boxes(const int *frame, const int *gh, const int *gw,
                  const int *tile, uint32_t rows, uint32_t block,
                  uint32_t count, block_box *out) {
    for (uint32_t index = 0; index < count; index++) {
        block_box box;
        box.video = 1;
        box.count = 0;
        uint32_t end = (index + 1) * block < rows ? (index + 1) * block : rows;
        for (uint32_t row = index * block; row < end; row++) {
            if (frame[row] < 0) {
                box.video = 0;
                continue;
            }
            int part = 0;
            while (part < box.count &&
                   (box.parts[part].t != frame[row] ||
                    box.parts[part].tile != tile[row])) part++;
            if (part == box.count) {
                if (box.count == MAX_PARTS) {
                    box.video = 0;   /* too fragmented: treat as dense */
                    continue;
                }
                box.parts[part] = (block_part){frame[row], gh[row], gh[row],
                                               gw[row], gw[row], tile[row]};
                box.count++;
            }
            block_part *p = &box.parts[part];
            if (gh[row] < p->h0) p->h0 = gh[row];
            if (gh[row] > p->h1) p->h1 = gh[row];
            if (gw[row] < p->w0) p->w0 = gw[row];
            if (gw[row] > p->w1) p->w1 = gw[row];
        }
        if (!box.count) box.video = 0;
        out[index] = box;
    }
}

static int keep_parts(const block_part *q, const block_part *k,
                      const h3_sparse_params *p) {
    if (k->t < p->sink) return 1;
    int dt = q->t > k->t ? q->t - k->t : k->t - q->t;
    if (dt <= p->t0) return 1;
    int level = 0;
    for (int step = dt - p->t0; step > 1; step >>= 1) level++;
    int radius = p->r0 >> level;
    if (radius < p->rmin) radius = p->rmin;
    int dh = gap(q->h0, q->h1, k->h0, k->h1);
    int dw = gap(q->w0, q->w1, k->w0, k->w1);
    return (dh > dw ? dh : dw) <= radius;
}

static int keep_pair(const block_box *q, const block_box *k,
                     const h3_sparse_params *p) {
    if (!q->video || !k->video) return 1;
    for (int i = 0; i < q->count; i++)
        for (int j = 0; j < k->count; j++)
            if (keep_parts(&q->parts[i], &k->parts[j], p)) return 1;
    return 0;
}

int h3_sparse_plan_build(const h3_layout *layout,
                         const h3_sparse_params *params, uint32_t q_block,
                         uint32_t k_block, h3_sparse_plan *plan,
                         char *error, size_t error_size) {
    if (!layout || !params || !plan || !q_block || !k_block ||
        !layout->positions || layout->seq_len > UINT32_MAX) {
        fail(error, error_size, "invalid sparse attention arguments");
        return 0;
    }
    memset(plan, 0, sizeof(*plan));
    plan->params = *params;
    uint32_t rows = (uint32_t)layout->seq_len;
    plan->rows = rows;
    plan->q_block = q_block;
    plan->k_block = k_block;
    plan->q_blocks = (rows + q_block - 1) / q_block;
    plan->k_blocks = (rows + k_block - 1) / k_block;
    plan->perm = malloc(rows * sizeof(*plan->perm));
    plan->inverse = malloc(rows * sizeof(*plan->inverse));
    int *frame = malloc(rows * sizeof(*frame));
    int *gh = malloc(rows * sizeof(*gh));
    int *gw = malloc(rows * sizeof(*gw));
    int *tl = malloc(rows * sizeof(*tl));
    block_box *qbox = malloc(plan->q_blocks * sizeof(*qbox));
    block_box *kbox = malloc(plan->k_blocks * sizeof(*kbox));
    plan->offsets = calloc((size_t)plan->q_blocks + 1, sizeof(*plan->offsets));
    int ok = plan->perm && plan->inverse && frame && gh && gw && tl && qbox &&
             kbox && plan->offsets;
    if (!ok) fail(error, error_size, "out of memory building a sparse plan");
    for (uint32_t row = 0; ok && row < rows; row++) {
        plan->perm[row] = row;
        frame[row] = -1;
    }
    for (size_t index = 0; ok && index < layout->segment_count; index++) {
        const h3_segment *segment = &layout->segments[index];
        if (segment->kind != H3_SEG_VIDEO) continue;
        size_t start = segment->start, stop = segment->stop, nt = 0, nh = 0,
               nw = 0;
        double *ts = unique_axis(layout->positions, start, stop, 0, &nt);
        double *hs = unique_axis(layout->positions, start, stop, 1, &nh);
        double *ws = unique_axis(layout->positions, start, stop, 2, &nw);
        order_entry *order = malloc((stop - start) * sizeof(*order));
        ok = ts && hs && ws && order;
        int tile = params->tile;
        uint64_t tiles_w = (nw + (size_t)tile - 1) / (size_t)tile;
        for (size_t row = start; ok && row < stop; row++) {
            const h3_position *p = &layout->positions[row];
            int t = axis_rank(ts, nt, p->t), h = axis_rank(hs, nh, p->h),
                w = axis_rank(ws, nw, p->w);
            if (t < 0 || h < 0 || w < 0) { ok = 0; break; }
            uint64_t tile_index = (uint64_t)(h / tile) * tiles_w + (uint64_t)(w / tile);
            uint64_t inside = (uint64_t)(h % tile) * (uint64_t)tile + (uint64_t)(w % tile);
            order[row - start] = (order_entry){
                ((uint64_t)t << 40) | (tile_index << 16) | inside, (uint32_t)row};
            frame[row] = t;
            gh[row] = h;
            gw[row] = w;
            tl[row] = (int)tile_index;
        }
        if (ok) {
            qsort(order, stop - start, sizeof(*order), compare_order);
            int *f = malloc((stop - start) * sizeof(int) * 4);
            ok = f != NULL;
            for (size_t row = 0; ok && row < stop - start; row++) {
                f[row * 4] = frame[order[row].row];
                f[row * 4 + 1] = gh[order[row].row];
                f[row * 4 + 2] = gw[order[row].row];
                f[row * 4 + 3] = tl[order[row].row];
            }
            for (size_t row = 0; ok && row < stop - start; row++) {
                plan->perm[start + row] = order[row].row;
                frame[start + row] = f[row * 4];
                gh[start + row] = f[row * 4 + 1];
                gw[start + row] = f[row * 4 + 2];
                tl[start + row] = f[row * 4 + 3];
            }
            free(f);
        }
        free(ts);
        free(hs);
        free(ws);
        free(order);
        if (!ok) fail(error, error_size, "cannot order video tokens");
    }
    for (uint32_t row = 0; ok && row < rows; row++)
        plan->inverse[plan->perm[row]] = row;
    if (ok) {
        boxes(frame, gh, gw, tl, rows, q_block, plan->q_blocks, qbox);
        boxes(frame, gh, gw, tl, rows, k_block, plan->k_blocks, kbox);
        size_t kept = 0;
        for (uint32_t q = 0; q < plan->q_blocks; q++)
            for (uint32_t k = 0; k < plan->k_blocks; k++)
                kept += (size_t)keep_pair(&qbox[q], &kbox[k], params);
        plan->indices = malloc((kept ? kept : 1) * sizeof(*plan->indices));
        ok = plan->indices != NULL;
        size_t at = 0;
        for (uint32_t q = 0; ok && q < plan->q_blocks; q++) {
            plan->offsets[q] = (uint32_t)at;
            for (uint32_t k = 0; k < plan->k_blocks; k++)
                if (keep_pair(&qbox[q], &kbox[k], params))
                    plan->indices[at++] = k;
        }
        if (ok) plan->offsets[plan->q_blocks] = (uint32_t)at;
        plan->kept = at;
    }
    free(frame);
    free(gh);
    free(gw);
    free(tl);
    free(qbox);
    free(kbox);
    if (!ok) h3_sparse_plan_free(plan);
    return ok;
}

void h3_sparse_plan_free(h3_sparse_plan *plan) {
    if (!plan) return;
    free(plan->perm);
    free(plan->inverse);
    free(plan->offsets);
    free(plan->indices);
    memset(plan, 0, sizeof(*plan));
}

double h3_sparse_plan_density(const h3_sparse_plan *plan) {
    if (!plan || !plan->q_blocks || !plan->k_blocks) return 0.0;
    return (double)plan->kept / ((double)plan->q_blocks * plan->k_blocks);
}
