// H3 DiT attention on M5 Neural Accelerators (Metal 4 TensorOps).
//
// Derived from MLX's steel_attention_nax (Copyright 2024-26 Apple Inc., MIT, see THIRD_PARTY_NOTICES.md): the same
// 16x16 NAX fragments, Q@K^T and P@V matmul2d chains and online softmax, specialized to H3's single shape: one
// batch, non-causal, no mask, head_dim 128, Q/K/V and O in head-major [head][row][128] layout.
//
// attention/flatten.py inlines the MLX headers into h3_attention.metal, which h3_gpu compiles at runtime.

#include "mlx/backend/metal/kernels/steel/attn/nax.h"
#include <metal_tensor>

using namespace mlx::steel;

struct h3_attention_args {
    int rows;        // sequence length (queries == keys)
    int heads;
    float scale;     // softmax scale (1/sqrt(128))
    int q_blocks;    // ceil(rows / BQ)
    int k_full;      // number of full key blocks (rows / BK)
    int k_rem;       // keys in the last partial block (0 if aligned)
};

struct h3_max_op {
    template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};
struct h3_sum_op {
    template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};
struct h3_mul_op {
    template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x * y; }
};
struct h3_exp_sub_op {
    template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); }
};

// TQK = T: exact BF16 Q@K^T. TQK = int8_t: Q and K were quantized per (head, row) by h3_attention_quantize_qk
// (K after subtracting its per-channel mean, which leaves softmax unchanged); S = int32(Q8 K8^T) * q_scale * k_scale.
template <typename T, typename TQK, int BQ, int BK, int WM, int kStage = 3, bool kWideQK = false,
          bool kSparse = false>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]] void h3_attention_nax(
    const device TQK *Q [[buffer(0)]],
    const device TQK *K [[buffer(1)]],
    const device T *V [[buffer(2)]],
    device T *O [[buffer(3)]],
    constant h3_attention_args &args [[buffer(4)]],
    const device float *q_scales [[buffer(5)]],
    const device float *k_scales [[buffer(6)]],
    const device uint *block_offsets [[buffer(7)]],   // kSparse: CSR over query blocks (BQ rows)
    const device uint *block_indices [[buffer(8)]],   // kSparse: kept key blocks (BK rows), ascending
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
    constexpr bool kInt8 = metal::is_same_v<TQK, int8_t>;
    using SAccum = metal::conditional_t<kInt8, int32_t, float>;
    constexpr int BD = 128;
    constexpr short kU = 16;
    constexpr int TQ = BQ / (WM * kU);
    constexpr int TD = BD / kU;
    constexpr short TK = BK / kU;
    static_assert(TQ == 1, "one 16-row fragment per simdgroup");
    using AccumType = float;
    using otile_t = NAXTile<AccumType, TQ, TD>;
    using stile_t = NAXTile<AccumType, TQ, TK>;

    const ulong head_stride = (ulong)args.rows * BD;
    const int q_row0 = int(tid.x) * BQ + int(kU * TQ * simd_group_id);
    Q += tid.y * head_stride + (ulong)q_row0 * BD;
    const device TQK *K_head = K + tid.y * head_stride;
    const device T *V_head = V + tid.y * head_stride;
    O += tid.y * head_stride + (ulong)q_row0 * BD;

    const float scale2 = args.scale * 1.44269504089f;
    otile_t Otile;
    Otile.clear();
    constexpr short kRowsPT = otile_t::kRowsPerThread;
    metal::vec<AccumType, kRowsPT> max_score;
    metal::vec<AccumType, kRowsPT> sum_score{0};
    for (short i = 0; i < kRowsPT; ++i) max_score[i] = -FLT_MAX;
    const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
    const short sn = simd_coord.x;
    const short sm = simd_coord.y;
    float q_scale[2] = {1.f, 1.f};
    if (kInt8) {
        q_scales += tid.y * (ulong)args.rows;
        k_scales += tid.y * (ulong)args.rows;
        for (short ii = 0; ii < 2; ii++) {
            const int row = q_row0 + sm + ii * 8;
            q_scale[ii] = row < args.rows ? q_scales[row] * scale2 : 0.f;
        }
    }

    const int q_left = args.rows - q_row0;          // valid query rows of this simdgroup (may be <= 0)
    const bool q_partial = q_left < kU;
    const int k_blocks = args.k_full + (args.k_rem ? 1 : 0);
    const int it_begin = kSparse ? int(block_offsets[tid.x]) : 0;
    const int it_end = kSparse ? int(block_offsets[tid.x + 1]) : k_blocks;

    for (int it = it_begin; it < it_end; it++) {
        const int kb = kSparse ? int(block_indices[it]) : it;
        const bool k_partial = kb == args.k_full;
        K = K_head + (ulong)kb * BK * BD;
        V = V_head + (ulong)kb * BK * BD;
        stile_t Stile;
        NAXTile<SAccum, TQ, TK> Sacc;
        Sacc.clear();
        if constexpr (kWideQK) {
            // One matmul2d over the full head_dim per 16x32 S tile, operands read straight from device tensors, so
            // TensorOps picks its native reduction depth (int8 needs it to run at twice the BF16 rate). The
            // destination keeps the 16x32 NAX fragment order the softmax below expects.
            static_assert(BK == 32, "wide Q@K^T covers one 16x32 tile");
            constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
                16, 32, BD, false, true, true,
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
            mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> mm;
            auto qt = tensor<device TQK, dextents<int32_t, 2>, tensor_inline>(
                (device TQK *)Q, dextents<int32_t, 2>(BD, max(q_left, 0)));
            auto kt = tensor<device TQK, dextents<int32_t, 2>, tensor_inline>(
                (device TQK *)K, dextents<int32_t, 2>(BD, k_partial ? args.k_rem : BK));
            auto qa = qt.template slice<BD, 16>(0, 0);
            auto kb_ = kt.template slice<BD, 32>(0, 0);
            auto acc = mm.template get_destination_cooperative_tensor<decltype(qa), decltype(kb_), SAccum>();
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < acc.get_capacity(); i++) acc[i] = 0;
            mm.run(qa, kb_, acc);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 8; i++) {
                Sacc.frag_at(0, 0)[i] = acc[i];
                Sacc.frag_at(0, 1)[i] = acc[8 + i];
            }
        } else
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik += 2) {
#pragma clang loop unroll_count(4)
            for (short id = 0; id < TD; id++) {
                NAXTile<TQK, 1, 1> Qtile;
                NAXTile<TQK, 2, 1> Ktile;
                const int q_off = id * kU;
                const int k_off = ik * kU * BD + id * kU;
                if (q_partial) Qtile.load_rows(Q + q_off, BD, q_left);
                else Qtile.load(Q + q_off, BD);
                if (k_partial) Ktile.load_rows(K + k_off, BD, args.k_rem - ik * kU);
                else Ktile.load(K + k_off, BD);
                stile_t::NAXFrag_t::mma(Sacc.frag_at(0, ik), Sacc.frag_at(0, ik + 1),
                                        Qtile.frag_at(0, 0), metal::false_type{},
                                        Ktile.frag_at(0, 0), Ktile.frag_at(1, 0),
                                        metal::true_type{});
            }
        }
        if (kInt8) {
            const int col0 = kb * BK;
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                float k_scale[4];
                STEEL_PRAGMA_UNROLL
                for (short jj = 0; jj < 4; jj++) {
                    const int col = col0 + ik * kU + sn + jj;
                    k_scale[jj] = col < args.rows ? k_scales[col] : 0.f;
                }
                thread auto &src = Sacc.frag_at(0, ik);
                thread auto &dst = Stile.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short ii = 0; ii < 2; ii++)
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < 4; jj++)
                        dst[ii * 4 + jj] = float(src[ii * 4 + jj]) * q_scale[ii] * k_scale[jj];
            }
        } else {
            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < stile_t::kElemsPerTile; ii++)
                Stile.elems()[ii] = float(Sacc.elems()[ii]) * scale2;
        }
        if (k_partial) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                const short col = ik * kU + sn;
                thread auto &fg = Stile.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short ii = 0; ii < stile_t::kFragThrRows; ii++)
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
                        const auto loc = ii * stile_t::kFragThrCols + jj;
                        fg[loc] = (col + jj) < args.k_rem ? fg[loc] : -FLT_MAX;
                    }
            }
        }
        if (kStage == 1) {
            // Profiling only: keep S alive without softmax or P@V.
            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) sum_score[0] += Stile.elems()[ii];
            continue;
        }
        metal::vec<AccumType, kRowsPT> new_max;
        metal::vec<AccumType, kRowsPT> factor;
        for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
        Stile.template row_reduce<h3_max_op>(new_max);
        Stile.template row_bin_op<h3_exp_sub_op>(new_max);
        for (short i = 0; i < kRowsPT; ++i) {
            factor[i] = fast::exp2(max_score[i] - new_max[i]);
            max_score[i] = new_max[i];
            sum_score[i] = sum_score[i] * factor[i];
        }
        Stile.template row_reduce<h3_sum_op>(sum_score);
        Otile.template row_bin_op<h3_mul_op>(factor);
        simdgroup_barrier(mem_flags::mem_none);
        if (kStage == 2) {
            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) Otile.elems()[ii % otile_t::kElemsPerTile] += Stile.elems()[ii];
            continue;
        }
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id += 2) {
            if (id == 4) threadgroup_barrier(mem_flags::mem_none);
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                NAXTile<T, 1, 2> Vtile;
                const int v_off = ik * kU * BD + id * kU;
                if (k_partial) Vtile.load_rows(V + v_off, BD, args.k_rem - ik * kU);
                else Vtile.load(V + v_off, BD);
                otile_t::NAXFrag_t::mma(Otile.frag_at(0, id), Otile.frag_at(0, id + 1),
                                        Stile.frag_at(0, ik), metal::false_type{},
                                        Vtile.frag_at(0, 0), Vtile.frag_at(0, 1),
                                        metal::false_type{});
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_none);
    metal::vec<AccumType, kRowsPT> rcp;
    for (short i = 0; i < kRowsPT; ++i) rcp[i] = 1.f / sum_score[i];
    Otile.template row_bin_op<h3_mul_op>(rcp);
    if (q_partial) {
        if (q_left <= 0) return;
        Otile.store_rows(O, BD, q_left);
    } else {
        Otile.store(O, BD);
    }
}

#define H3_ATTENTION_INSTANCE(name, T, TQK, BQ, BK, WM, ...)                                \
    template [[host_name(name)]] [[kernel]] void h3_attention_nax<T, TQK, BQ, BK, WM, ##__VA_ARGS__>( \
        const device TQK *, const device TQK *, const device T *, device T *,                \
        constant h3_attention_args &, const device float *, const device float *,                \
        const device uint *, const device uint *, uint, uint3);

H3_ATTENTION_INSTANCE("h3_attention_nax_bf16_q64_k32", bfloat, bfloat, 64, 32, 4)
H3_ATTENTION_INSTANCE("h3_attention_nax_qk8_q64_k32", bfloat, int8_t, 64, 32, 4)
// Block-sparse (CSR key-block lists), on reordered rows.
H3_ATTENTION_INSTANCE("h3_attention_nax_bf16_q64_k32_sparse", bfloat, bfloat, 64, 32, 4, 3, false, true)
H3_ATTENTION_INSTANCE("h3_attention_nax_qk8_q64_k32_sparse", bfloat, int8_t, 64, 32, 4, 3, false, true)
// Profiling stages (H3_ATTN_STAGE=1 QK only, 2 QK+softmax).
H3_ATTENTION_INSTANCE("h3_attention_nax_bf16_q64_k32_s1", bfloat, bfloat, 64, 32, 4, 1)
H3_ATTENTION_INSTANCE("h3_attention_nax_bf16_q64_k32_s2", bfloat, bfloat, 64, 32, 4, 2)
H3_ATTENTION_INSTANCE("h3_attention_nax_qk8_q64_k32_s1", bfloat, int8_t, 64, 32, 4, 1)
H3_ATTENTION_INSTANCE("h3_attention_nax_qk8_q64_k32_s2", bfloat, int8_t, 64, 32, 4, 2)
// Larger query blocks: 8 simdgroups x 16 rows.
H3_ATTENTION_INSTANCE("h3_attention_nax_bf16_q64_k32_b128", bfloat, bfloat, 128, 32, 8)
H3_ATTENTION_INSTANCE("h3_attention_nax_qk8_q64_k32_b128", bfloat, int8_t, 128, 32, 8)
H3_ATTENTION_INSTANCE("h3_attention_nax_bf16_q64_k32_k64", bfloat, bfloat, 64, 64, 4)
H3_ATTENTION_INSTANCE("h3_attention_nax_qk8_q64_k32_k64", bfloat, int8_t, 64, 64, 4)
// Wide Q@K^T (one K=128 matmul2d per tile).
H3_ATTENTION_INSTANCE("h3_attention_nax_bf16_q64_k32_w", bfloat, bfloat, 64, 32, 4, 3, true)
H3_ATTENTION_INSTANCE("h3_attention_nax_qk8_q64_k32_w", bfloat, int8_t, 64, 32, 4, 3, true)
H3_ATTENTION_INSTANCE("h3_attention_nax_bf16_q64_k32_s1_w", bfloat, bfloat, 64, 32, 4, 1, true)
H3_ATTENTION_INSTANCE("h3_attention_nax_qk8_q64_k32_s1_w", bfloat, int8_t, 64, 32, 4, 1, true)

// Per-channel mean of K over all rows, per head: mean[head][128]. One thread per (head, channel).
kernel void h3_attention_k_mean(const device bfloat *K [[buffer(0)]],
                                device float *mean [[buffer(1)]],
                                constant h3_attention_args &args [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(args.heads) * 128) return;
    const uint head = gid / 128, channel = gid % 128;
    const device bfloat *k = K + (ulong)head * args.rows * 128 + channel;
    float sum = 0.f;
    for (int row = 0; row < args.rows; row++) sum += float(k[(ulong)row * 128]);
    mean[gid] = sum / float(args.rows);
}

// Quantize one (head, row) of Q or K to int8 with scale max|x|/127; K first subtracts the per-channel mean.
// One simdgroup per row, four channels per lane. Grid: rows x heads simdgroups.
kernel void h3_attention_quantize_qk(const device bfloat *X [[buffer(0)]],
                                     device int8_t *out [[buffer(1)]],
                                     device float *scales [[buffer(2)]],
                                     const device float *mean [[buffer(3)]],
                                     constant h3_attention_args &args [[buffer(4)]],
                                     constant int &subtract_mean [[buffer(5)]],
                                     const device uint *source_rows [[buffer(6)]],
                                     constant int &use_source_rows [[buffer(7)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x, head = group.y;
    if (row >= uint(args.rows)) return;
    const uint source = use_source_rows ? source_rows[row] : row;
    const ulong base = ((ulong)head * args.rows + row) * 128 + lane * 4;
    float4 x = float4(*(const device vec<bfloat, 4> *)(X + ((ulong)head * args.rows + source) * 128 + lane * 4));
    if (subtract_mean) x -= *(const device float4 *)(mean + head * 128 + lane * 4);
    float m = simd_max(max(max(fabs(x.x), fabs(x.y)), max(fabs(x.z), fabs(x.w))));
    float scale = m > 0.f ? m / 127.f : 1.f;
    float inv = 1.f / scale;
    char4 q = char4(clamp(rint(x * inv), -127.f, 127.f));
    *(device char4 *)(out + base) = q;
    if (lane == 0) scales[(ulong)head * args.rows + row] = scale;
}

// dst[head][row] = src[head][rows[row]] for head-major [head][rows][128] BF16; 16 threads per row, 8 values each.
kernel void h3_attention_gather_rows(const device bfloat *src [[buffer(0)]],
                                     device bfloat *dst [[buffer(1)]],
                                     const device uint *rows [[buffer(2)]],
                                     constant h3_attention_args &args [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    const uint row = gid.x / 16, part = gid.x % 16, head = gid.y;
    if (row >= uint(args.rows)) return;
    const ulong head_base = (ulong)head * args.rows * 128;
    *(device uint4 *)(dst + head_base + (ulong)row * 128 + part * 8) =
        *(const device uint4 *)(src + head_base + (ulong)rows[row] * 128 + part * 8);
}

// y[rows][out] (F32) = x[rows][in] (BF16) * w[out][in]^T (BF16) + bias[out] (F32): one 128x128 output tile per
// threadgroup of four simdgroups, F32 accumulation. `in` and `out` are multiples of 128; x has room for whole
// 128-row tiles (rows past `rows` are computed and dropped). Used by the video VAE's mixed-precision linears.
struct h3_linear_dims {
    uint rows;
    uint in;
    uint out;
};

kernel void h3_linear_bf16_f32out_nax(device bfloat *x [[buffer(0)]],
                                      device bfloat *w [[buffer(1)]],
                                      const device float *bias [[buffer(2)]],
                                      device float *y [[buffer(3)]],
                                      constant h3_linear_dims &d [[buffer(4)]],
                                      uint2 g [[threadgroup_position_in_grid]]) {
    constexpr int TILE = 128;
    const uint padded = (d.rows + TILE - 1) & ~uint(TILE - 1);
    auto X = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(x, dextents<int32_t, 2>(d.in, padded));
    auto W = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(w, dextents<int32_t, 2>(d.in, d.out));
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        TILE, TILE, TILE, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> mm;
    auto first_x = X.template slice<TILE, TILE>(0, g.y * TILE);
    auto first_w = W.template slice<TILE, TILE>(0, g.x * TILE);
    auto acc = mm.template get_destination_cooperative_tensor<decltype(first_x), decltype(first_w), float>();
    for (ushort e = 0; e < acc.get_capacity(); e++) if (acc.is_valid_element(e)) acc[e] = 0.f;
    for (uint k = 0; k < d.in; k += TILE) {
        auto sx = X.template slice<TILE, TILE>(k, g.y * TILE);
        auto sw = W.template slice<TILE, TILE>(k, g.x * TILE);
        mm.run(sx, sw, acc);
    }
    for (ushort e = 0; e < acc.get_capacity(); e++) {
        if (!acc.is_valid_element(e)) continue;
        auto index = acc.get_multidimensional_index(e);
        uint row = g.y * TILE + uint(index[1]), col = g.x * TILE + uint(index[0]);
        if (row < d.rows) y[(ulong)row * d.out + col] = acc[e] + (bias ? bias[col] : 0.f);
    }
}
