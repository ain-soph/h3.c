// Tile variants of h3.c's video-VAE linear (h3_linear_bf16_f32out_nax): y[rows][out] (F32) =
// x[rows][in] (BF16) * w[out][in]^T (BF16) + bias (F32), F32 accumulation on M5 TensorOps.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

struct dims_t { uint rows; uint in; uint out; uint swap; };

// Explicit K loop in BK steps into a cooperative F32 accumulator (the current kernel is <128,128,128,4>).
template <int BM, int BN, int BK, int SG>
kernel void loopk(device bfloat *x [[buffer(0)]], device bfloat *w [[buffer(1)]],
                  const device float *bias [[buffer(2)]], device float *y [[buffer(3)]],
                  constant dims_t &d [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]]) {
    const uint2 g = d.swap ? uint2(tg.y, tg.x) : tg;  // g.x = column tile, g.y = row tile
    const uint padded = (d.rows + 127) & ~127u;
    auto X = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(x, dextents<int32_t, 2>(d.in, padded));
    auto W = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(w, dextents<int32_t, 2>(d.in, d.out));
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN, BK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<SG>> mm;
    auto first_x = X.template slice<BK, BM>(0, g.y * BM);
    auto first_w = W.template slice<BK, BN>(0, g.x * BN);
    auto acc = mm.template get_destination_cooperative_tensor<decltype(first_x), decltype(first_w), float>();
    for (ushort e = 0; e < acc.get_capacity(); e++) if (acc.is_valid_element(e)) acc[e] = 0.f;
    for (uint k = 0; k < d.in; k += BK) {
        auto sx = X.template slice<BK, BM>(k, g.y * BM);
        auto sw = W.template slice<BK, BN>(k, g.x * BN);
        mm.run(sx, sw, acc);
    }
    for (ushort e = 0; e < acc.get_capacity(); e++) {
        if (!acc.is_valid_element(e)) continue;
        auto index = acc.get_multidimensional_index(e);
        uint row = g.y * BM + uint(index[1]), col = g.x * BN + uint(index[0]);
        if (row < d.rows) y[(ulong)row * d.out + col] = acc[e] + bias[col];
    }
}

#define LOOPK(BM, BN, BK, SG)                                                                           \
    template [[host_name("loopk_" #BM "_" #BN "_" #BK "_" #SG)]] kernel void loopk<BM, BN, BK, SG>(     \
        device bfloat *, device bfloat *, const device float *, device float *, constant dims_t &, uint2);

LOOPK(128, 128, 128, 4)
LOOPK(128, 128, 64, 4)
LOOPK(128, 128, 32, 4)
LOOPK(128, 128, 256, 4)
LOOPK(128, 64, 128, 4)
LOOPK(64, 128, 128, 4)
LOOPK(64, 64, 128, 4)
LOOPK(128, 128, 128, 8)
LOOPK(128, 64, 64, 4)
LOOPK(64, 128, 64, 4)
LOOPK(128, 256, 64, 8)
LOOPK(256, 128, 64, 8)
