#ifndef H3_ROT_H
#define H3_ROT_H

/* Hadamard-rotated W8A8 for the DiT block matrices (Track A, rot256 branch).
 *
 * H256 = (H4 / 2)^{(x)4} with H4[d][e] = -1 when d + e == 3 and 1 otherwise:
 * the same symmetric orthogonal matrix as the ConvRot text encoder
 * (h3_rotate256_bf16). H128 = (H2 / sqrt 2) (x) (H4 / 2)^{(x)3}, also
 * symmetric and orthogonal. Because H H = I, W x == (W H)(H x).
 *
 * Classes (bit mask, also the h3-int8-pack.json "rot" list):
 *   qkv     K of attn.qkv_proj rotated by H256; the activation is rotated in
 *           the fused gate/AdaLN quantizer (and the standalone row quantizer)
 *   fc1     K of mlp.fc1 rotated by H256, same activation path as qkv
 *   out256  K of attn.out_proj rotated by H256 at run time (head-major
 *           quantizer rotates each pair of heads)
 *   out128  offline fold, no run-time cost: the V rows of every head are
 *           mixed by H128 (V' = V H128, so the attention output is rotated)
 *           and each head's 128 columns of attn.out_proj by H128
 *   fc2     K of mlp.fc2 rotated by H256 in the FC2 input quantizer (row or
 *           1024-grouped scales; each 1024 group holds four whole 256 groups)
 * out256 and out128 are mutually exclusive. */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

enum {
    H3_ROT_QKV = 1u,
    H3_ROT_FC1 = 2u,
    H3_ROT_OUT256 = 4u,
    H3_ROT_OUT128 = 8u,
    H3_ROT_FC2 = 16u,
    H3_ROT_ALL = 31u
};

/* Which DiT block matrix a weight is. */
typedef enum {
    H3_ROT_MATRIX_QKV = 0,
    H3_ROT_MATRIX_OUT,
    H3_ROT_MATRIX_FC1,
    H3_ROT_MATRIX_FC2
} h3_rot_matrix;

/* "none", "" or a comma list of qkv,fc1,out256,out128,fc2 (also "all" =
 * qkv,fc1,out256,fc2). Returns 0 and fills error on a bad list. */
int h3_rot_parse(const char *text, unsigned *mask, char *error,
                 size_t error_size);
/* Canonical comma list ("none" for 0). */
void h3_rot_format(unsigned mask, char *text, size_t size);
/* Mask of an h3-int8-pack.json marker: "format" "h3-int8-pack-v1" (no
 * "rot": 0) or "h3-int8-pack-v2" with a non-empty "rot" list. Strict: an
 * unknown format, a v2 marker without "rot", a v1 marker with one or a
 * malformed file is an error (returns 0 and fills error). */
int h3_rot_read_marker(const char *marker_path, unsigned *mask, char *error,
                       size_t error_size);
/* Every *.safetensors file of a packed DiT carries its own identity in
 * __metadata__ ("h3": "int8-pack-v1", or "int8-pack-v2" + "h3_rot": LIST,
 * written by h3_pack_int8 or tools/stamp_int8_pack_rot.py). Returns 0 and
 * fills error unless every file agrees with the marker's mask, so weights
 * quantized with one rotation are never run with another (a lost, rewritten
 * or copied marker, a half re-packed directory). */
int h3_rot_check_pack(const char *directory, unsigned marker_mask,
                      char *error, size_t error_size);

/* True when the matrix needs the CPU rotated quantizer under this mask. */
int h3_rot_matrix_rotated(h3_rot_matrix matrix, unsigned mask);

void h3_hadamard256_f32(float *values);
void h3_hadamard128_f32(float *values);

/* Quantize one DiT block matrix (BF16 [rows, columns], checkpoint layout)
 * after the rotations the mask selects for it, with the same symmetric
 * per-row rule as the GPU weight quantizer (scale = max|w| / 127,
 * q = clamp(rint(w * 127 / max|w|), -127, 127)). Rotations run in F32 and
 * are never rounded back to BF16. Multithreaded; deterministic. */
int h3_rot_quantize_matrix(h3_rot_matrix matrix, unsigned mask,
                           const uint16_t *weight, uint32_t rows,
                           uint32_t columns, int8_t *quantized,
                           float *scales);

/* Output file of a Track A study hook (H3_TEXT_EMB_OUT, H3_DUMP_*,
 * H3_TEACHER_FORCE_DUMP/LOG): a regular file only, never through a symlink
 * (O_NOFOLLOW), created 0644, truncated or (append) appended to. NULL on
 * failure. A caller that fails a write removes the file. */
FILE *h3_study_fopen(const char *path, int append);

#endif
