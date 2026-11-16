// H3 DiT attention on M5 Neural Accelerators (Metal 4 TensorOps).
//
// Derived from MLX's steel_attention_nax (Copyright 2024-26 Apple Inc., MIT, see THIRD_PARTY_NOTICES.md): the same
// 16x16 NAX fragments, Q@K^T and P@V matmul2d chains and online softmax, specialized to H3's single shape: one
// batch, non-causal, no mask, head_dim 128, Q/K/V and O in head-major [head][row][128] layout.
//
// attention/flatten.py inlines the MLX headers into h3_attention.metal, which h3_gpu compiles at runtime.

// ---- begin mlx/backend/metal/kernels/steel/attn/nax.h
// Copyright © 2025 Apple Inc.


#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

// ---- begin mlx/backend/metal/kernels/steel/defines.h
// Copyright © 2024 Apple Inc.


#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")
// ---- end mlx/backend/metal/kernels/steel/defines.h
// ---- begin mlx/backend/metal/kernels/steel/utils/integral_constant.h
// Copyright © 2024 Apple Inc.


#include <metal_stdlib>
// ---- begin mlx/backend/metal/kernels/steel/utils/type_traits.h
// Copyright © 2024 Apple Inc.


#include <metal_stdlib>

#pragma METAL internals : enable

namespace metal {

template <typename T>
struct is_empty : metal::bool_constant<__is_empty(T)> {};

#ifdef __cpp_variable_templates
template <typename T>
constexpr constant bool is_empty_v = is_empty<T>::value;
#endif

template <typename... Ts>
struct make_void {
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

template <class T>
struct is_static : metal::bool_constant<is_empty<remove_cv_t<T>>::value> {};

template <typename T>
struct pointer_element {};

template <typename T>
struct pointer_element<thread T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<device T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<constant T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<threadgroup T*> {
  using type = remove_cv_t<T>;
};

template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

} // namespace metal

#pragma METAL internals : disable
// ---- end mlx/backend/metal/kernels/steel/utils/type_traits.h

#pragma METAL internals : enable

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// Integral constant with casting
///////////////////////////////////////////////////////////////////////////////

template <typename T, T v>
struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;

  METAL_FUNC constexpr operator value_type() const thread noexcept {
    return value;
  }
};

template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <class T>
struct is_integral : bool_constant<metal::is_integral<T>::value> {};

template <class T, T v>
struct is_integral<integral_constant<T, v>>
    : bool_constant<metal::is_integral<T>::value> {};

template <typename T>
constexpr constant bool is_integral_v = is_integral<T>::value;

template <int val>
using Int = integral_constant<int, val>;

///////////////////////////////////////////////////////////////////////////////
// Binary Operators on Integral constants
///////////////////////////////////////////////////////////////////////////////

#define integral_const_binop(__op__, __operator__)          \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto __operator__(                   \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv __op__ uv;                      \
    using res_t = metal::remove_addrspace_t<decltype(res)>; \
    return integral_constant<res_t, res>{};                 \
  }

integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);

integral_const_binop(==, operator==);
integral_const_binop(!=, operator!=);
integral_const_binop(<, operator<);
integral_const_binop(>, operator>);
integral_const_binop(<=, operator<=);
integral_const_binop(>=, operator>=);

integral_const_binop(&&, operator&&);
integral_const_binop(||, operator||);

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(true_type, T) {
  return true_type{};
}
template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(T, true_type) {
  return true_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(false_type, T) {
  return false_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(T, false_type) {
  return false_type{};
}

// Dispatch utilities
template <typename F>
void dispatch_bool(bool v, F f) {
  if (v) {
    f(true_type{});
  } else {
    f(false_type{});
  }
}

template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    constexpr auto idx = Int<start>{};
    f(idx);
    const_for_loop<start + step, stop, step, F>(f);
  }
}

#undef integral_const_binop

///////////////////////////////////////////////////////////////////////////////
// Reduction operators
///////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC constexpr T sum(T x) {
  return x;
}

template <typename T, typename... Us>
METAL_FUNC constexpr auto sum(T x, Us... us) {
  return x + sum(us...);
}

} // namespace steel
} // namespace mlx

#pragma METAL internals : disable
// ---- end mlx/backend/metal/kernels/steel/utils/integral_constant.h

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// NAX Steel with new tiles
///////////////////////////////////////////////////////////////////////////////

struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;

  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST short kElemRows = 2;
  STEEL_CONST short kElemCols = 4;

  STEEL_CONST short kElemRowsJump = 8;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
    return short2{fn, fm};
  }

  METAL_FUNC static short2 get_coord(short idx) {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3)) + (idx >> 2) * 8;
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4 + idx % 4;
    return short2{fn, fm};
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        }
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j)]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] =
                static_cast<T>(src[r * str_x + (c + j) * str_y]);
          }
        }

      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((r < lx) && ((c + j) < ly)) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_rows(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + (c + j) * str_y] =
                static_cast<U>(src[i * kElemCols + j]);
          }
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_safe(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if (r < lx && (c + j) < ly) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename StartX,
      typename StopX,
      typename StartY,
      typename StopY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_slice(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      StartX start_x,
      StopX stop_x,
      StartY start_y,
      StopY stop_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();

    const_for_loop<0, kElemRows, 1>([&](auto idx_row) {
      const auto r = off_x + idx_row * Int<kElemRowsJump>{};
      if (r >= stop_x - sc.y || r < start_x - sc.y) {
        return;
      }

      const_for_loop<0, kElemCols, 1>([&](auto idx_col) {
        const auto c = off_y + idx_col;
        if (c >= stop_y - sc.x || c < start_y - sc.x) {
          return;
        }

        const auto src_idx = idx_row * Int<kElemCols>{} + idx_col;
        dst[(r + sc.y) * str_x + (c + sc.x) * str_y] =
            static_cast<U>(src[src_idx]);
      });
    });
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals,
      thread T* reduced_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      T thr_reduce = Op::apply(
          Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
          Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));

      T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

      T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

      reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_bin_op(
      thread dtype_frag_t<T>& inp_vals,
      thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cn0,
      thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0,
      const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>,
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = A[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = Bn0[i];
      ct_b[kElemsPerFrag + i] = Bn1[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cn0[i];
      ct_c[kElemsPerFrag + i] = Cn1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cn0[i] = ct_c[i];
      Cn1[i] = ct_c[kElemsPerFrag + i];
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cm0,
      thread dtype_frag_t<CType>& Cm1,
      const thread dtype_frag_t<AType>& Am0,
      const thread dtype_frag_t<AType>& Am1,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& B,
      metal::bool_constant<transpose_b>) {
    // Create Matmul descriptor
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>,
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = Am0[i];
      ct_a[kElemsPerFrag + i] = Am1[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = B[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cm0[i];
      ct_c[kElemsPerFrag + i] = Cm1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cm0[i] = ct_c[i];
      Cm1[i] = ct_c[kElemsPerFrag + i];
    }
  }
};

template <
    typename T,
    short kTileRows_,
    short kTileCols_,
    class NAXFrag_ = BaseNAXFrag>
struct NAXTile {
  using NAXFrag_t = NAXFrag_;
  using elem_type = T;

  STEEL_CONST short kFragRows = NAXFrag_t::kFragRows;
  STEEL_CONST short kFragCols = NAXFrag_t::kFragCols;
  STEEL_CONST short kElemsPerFrag = NAXFrag_t::kElemsPerFrag;

  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;

  STEEL_CONST short kRows = kTileRows * kFragRows;
  STEEL_CONST short kCols = kTileCols * kFragCols;

  STEEL_CONST short kNumFrags = kTileRows * kTileCols;
  STEEL_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;

  STEEL_CONST short kFragThrRows = NAXFrag_t::kElemRows;
  STEEL_CONST short kFragThrCols = NAXFrag_t::kElemCols;
  STEEL_CONST short kFragRowsJump = NAXFrag_t::kElemRowsJump;

  STEEL_CONST short kRowsPerThread = kTileRows * NAXFrag_t::kElemRows;
  STEEL_CONST short kColsPerThread = kTileCols * NAXFrag_t::kElemCols;

  typedef typename NAXFrag_t::template dtype_frag_t<T> frag_type;

  frag_type val_frags[kNumFrags]; // = {frag_type(0)};

  METAL_FUNC NAXTile() thread {}

  METAL_FUNC constexpr void clear() thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j)
      thread {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const thread {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr thread frag_type& frag_at() thread {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr const thread frag_type& frag_at() const thread {
    return val_frags[i * kTileCols + j];
  }

  template <bool transpose>
  METAL_FUNC constexpr thread frag_type& frag_at(
      const short i,
      const short j,
      metal::bool_constant<transpose>) thread {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }

  template <bool transpose>
  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j,
      metal::bool_constant<transpose>) const thread {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }

  template <int i, int j, bool transpose>
  METAL_FUNC constexpr thread frag_type& frag_at() thread {
    if constexpr (transpose) {
      return frag_at<j, i>();
    } else {
      return frag_at<i, j>();
    }
  }

  template <int i, int j, bool transpose>
  METAL_FUNC constexpr const thread frag_type& frag_at() const thread {
    if constexpr (transpose) {
      return frag_at<j, i>();
    } else {
      return frag_at<i, j>();
    }
  }

  METAL_FUNC thread elem_type* elems() thread {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  METAL_FUNC const thread elem_type* elems() const thread {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(
      thread metal::vec<T, kRowsPerThread>& vals) const thread {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(
      thread metal::vec<T, kRowsPerThread>& vals) thread {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void store(threadgroup U* dst) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void
  load_rows(const device U* src, const int ld, const short n_rows) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_rows(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            n_rows,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load_safe(
      const device U* src,
      const int ld,
      const short2 src_tile_dims) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_safe(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            src_tile_dims.y,
            src_tile_dims.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_rows(device U* dst, const int ld, const short n_rows)
      const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_rows(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            n_rows,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_safe(
      device U* dst,
      const int ld,
      const short2 dst_tile_dims) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_safe(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_slice(
      device U* dst,
      const int ld,
      const short2 start,
      const short2 stop) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_slice(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            start.y,
            stop.y,
            start.x,
            stop.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }
};

template <
    class CTile,
    class ATile,
    class BTile,
    bool transpose_a,
    bool transpose_b>
METAL_FUNC void tile_matmad_nax(
    thread CTile& C,
    thread ATile& A,
    metal::bool_constant<transpose_a>,
    thread BTile& B,
    metal::bool_constant<transpose_b>) {
  // Static checks
  constexpr short TMa = transpose_a ? ATile::kTileCols : ATile::kTileRows;
  constexpr short TM = CTile::kTileRows;
  static_assert(TMa == TM, "MXU tile matmul: M dimensions do not match");

  constexpr short TNb = transpose_b ? BTile::kTileRows : BTile::kTileCols;
  constexpr short TN = CTile::kTileCols;
  static_assert(TNb == TN, "MXU tile matmul: N dimensions do not match");

  constexpr short TKa = transpose_a ? ATile::kTileRows : ATile::kTileCols;
  constexpr short TK = transpose_b ? BTile::kTileCols : BTile::kTileRows;
  static_assert(TKa == TK, "MXU tile matmul: K dimensions do not match");

  constexpr auto ta = metal::bool_constant<transpose_a>{};
  constexpr auto tb = metal::bool_constant<transpose_b>{};

  if constexpr (TN == 1 && TM % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; mm += 2) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; ++nn) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm + 1, nn),
              A.frag_at(mm, kk, ta),
              A.frag_at(mm + 1, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  } else if constexpr (TN % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; ++mm) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; nn += 2) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm, nn + 1),
              A.frag_at(mm, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              B.frag_at(kk, nn + 1, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  }
}

} // namespace steel
} // namespace mlx
// ---- end mlx/backend/metal/kernels/steel/attn/nax.h
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
