/* Block-sparse DiT attention plans.
 *
 * Attention is permutation-equivariant, so the DiT may run it on a reordered
 * sequence: target-video tokens are regrouped, frame by frame, into square
 * spatial tiles (every other token keeps its place), Q/K/V are gathered into
 * that order, attention runs over a block mask, and the output is scattered
 * back. With tiles, each 64-query / 32-key block covers a compact
 * space-time box, so the mask can keep exactly the nearby boxes:
 *
 *   - any block holding a non-video token (text, audio, conditions,
 *     references) attends and is attended densely;
 *   - video blocks whose frames are at most t0 latent frames apart are kept;
 *   - farther video blocks are kept while their spatial (Chebyshev) distance
 *     in tokens is at most max(rmin, r0 >> floor(log2(dt - t0))) (a radial
 *     window that halves as the temporal distance doubles);
 *   - the first `sink` latent frames are kept as keys for every query.
 *
 * The plan is built once per layout on the CPU. H3_SPARSE enables it, e.g.
 * H3_SPARSE="t0=1,r0=12,rmin=2,sink=1,dense_steps=4,dense_head=1,dense_tail=1".
 */
#ifndef H3_SPARSE_H
#define H3_SPARSE_H

#include "h3_host.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int t0;            /* dense temporal radius, latent frames */
    int r0;            /* spatial radius at t0 + 1, tokens */
    int rmin;          /* smallest spatial radius */
    int sink;          /* leading latent frames every query attends */
    int tile;          /* spatial tile edge, tokens */
    int dense_steps;   /* first denoising evaluations run dense */
    int dense_head;    /* first DiT blocks run dense */
    int dense_tail;    /* last DiT blocks run dense */
} h3_sparse_params;

typedef struct {
    uint32_t rows;
    uint32_t q_block;      /* query rows per block (kernel BQ) */
    uint32_t k_block;      /* key rows per block (kernel BK) */
    uint32_t q_blocks;
    uint32_t k_blocks;
    uint32_t *perm;        /* perm[i] = original row at reordered position i */
    uint32_t *inverse;     /* inverse[perm[i]] = i */
    uint32_t *offsets;     /* q_blocks + 1 CSR offsets into indices */
    uint32_t *indices;     /* kept key blocks, ascending per query block */
    size_t kept;
    h3_sparse_params params;
} h3_sparse_plan;

/* Parses H3_SPARSE; returns 0 when sparse attention is disabled. */
int h3_sparse_params_from_env(h3_sparse_params *params);
int h3_sparse_plan_build(const h3_layout *layout,
                         const h3_sparse_params *params, uint32_t q_block,
                         uint32_t k_block, h3_sparse_plan *plan,
                         char *error, size_t error_size);
void h3_sparse_plan_free(h3_sparse_plan *plan);
/* Fraction of query-block x key-block pairs kept. */
double h3_sparse_plan_density(const h3_sparse_plan *plan);

#endif
