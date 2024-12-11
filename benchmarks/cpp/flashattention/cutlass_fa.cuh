// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.cuh"
#include "cutlass/copy.cuh"
#include "cutlass/traits_base.cuh"

#include <cute/tensor.hpp>

namespace benchmarks {
namespace cutlass_wrapper {
using namespace cute;

//
/// @brief
/// @tparam Element_
/// @tparam L: Q, K, V, O length
/// @tparam kDimK: Q, K hidden size
/// @tparam kDimV: V, O hidden size
/// @tparam kTileQRow: Tile size of Q row.
/// @tparam kTileQCol: Tile size of Q col.
/// @tparam kTileKCol: Tile size of K col.
/// @tparam kTileVCol: Tile size of V row.
// template <typename Element_, const int L, const kDimK, const int kDimV,
//           const int kTileQRow, const int kTileQCol, const int kTileKCol,
//           const int kTileVCol, const int kThreads>
template <typename Element_, const int kM, const int kN, const int kK,
          const int kP, const int kTM, const int kTN, const int kTK,
          const int kTP, const int kThreads>
struct FATraits : public Base {
    // Q: [kM, kK] --> [length, hidden_qk]
    // K: [kN, kK] --> [length, hidden_qk]
    // V: [kP, kN] --> [length, hidden_v]
    // O: [kM, kP] --> [length, hidden_v]
    // assert(kM == kN)
    using Element = Element_;

    // declare global to shared memory copy layout.
    using GmemLayoutQ = Layout<Shape<Int<kTM>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutK = Layout<Shape<Int<kTN>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutK = Layout<Shape<Int<kTP>, Int<kTN>>, Stride<Int<kN>, _1>>;

    constexpr int kWarps = kThreads / 32;

    using TiledMma =
        TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                 Layout<Shape<Int<kWarps>, _1, _1>>, Layout<Shape<_1, _2, _1>>>;
};

template <typename InType, typename AccType, typename KeTraits, const int kM,
          const int kN, const kK, const int kP, const int kTM, const int kTN,
          const int kTK, const int kTP, const int Nthreads>
__global__ void __launch_bounds__(Nthreads)
    fa_kernel(const InType* dQ, const InType* dK, const InType* dV, InType* dO,
              int length_k, int length_q) {
    constexpr float softmax_scale = 1.250000e-01f;

    // Q, K: [batch, head, length, hidden_qk]
    // V, O: [batch, head, length, hidden_v]

    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    const InType* Q = dQ + blockIdx.z * kTM * kN + blockIdx.x * kTM * kK;
    const InType* K = dK + blockIdx.z * kK * kN;
    const InType* V = dV + blockIdx.z * kP * kN + blockIdx.y * kTP * kN;
    InType* O =
        dO + blockIdx.z * kM * kP + blockIdx.x * (kTM * kP) + blockIdx.y * kTP;
}

}  // namespace cutlass_wrapper
}  // namespace benchmarks