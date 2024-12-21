// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "copy.cuh"
#include "cuda_utils.cuh"
#include "cutlass/copy.cuh"
#include "cutlass/traits_base.cuh"

#include <cute/tensor.hpp>

template <const int kM, const int kN, const int kK, const int kP>
using FAShape = TileShape<kM, kN, kK, kP>;

namespace benchmarks {
namespace cutlass_wrapper {
using namespace cute;

//
/// @brief
/// @tparam Element_
template <typename Element_, const int kM, const int kN, const int kK,
          const int kP, const int kTM, const int kTN, const int kTK,
          const int kTP, const int kThreads, const int SmemKAtom = 64,
          const int kSwizzle = 3, typename Base = AccessBase<Element_>>
struct FATraits : public Base {
    // Q: [kM, kK] --> [length, hidden_qk]
    // K: [kN, kK] --> [length, hidden_qk]
    // V: [kP, kN] --> [length, hidden_v]
    // O: [kM, kP] --> [length, hidden_v]
    // assert(kM == kN)
    using Element = Element_;

    // Declare global to shared memory copy layout.
    using GmemLayoutQ = Layout<Shape<Int<kTM>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutK = Layout<Shape<Int<kTN>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutV = Layout<Shape<Int<kTP>, Int<kTN>>, Stride<Int<kN>, _1>>;
    using GmemLayoutO = Layout<Shape<Int<kTM>, Int<kTP>>, Stride<Int<kP>, _1>>;

    // Atom Shared Memory Block.
    using SmemLayoutAtom = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<SmemKAtom>>, Stride<Int<SmemKAtom>, _1>>{}));

    // Declare shared memory layout.
    // [kTM / Atom, kTK / Atom, Atom]
    using SmemLayoutQ =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    // [kTN / Atom, kTK / Atom, Atom]
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));
    ));
    // [kTP / Atom, kTN / Atom, Atom]
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTP>, Int<kTN>>{}));
    using SmemLayoutO =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTP>>{}));

    constexpr int kWarps = kThreads / 32;

    // Declare MMA Operation: [16, 8, 16] * [1, 2, 1] -> [16, 16, 16]
    using TiledMma =
        TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                 Layout<Shape<Int<kWarps>, _1, _1>>, Layout<Shape<_1, _2, _1>>>;

#ifdef CP_ASYNC_SM80_ENABLED
    // for Ampere
    using CopyInstG2S =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
#else
    using CopyInstG2S = Copy_Atom<DefaultCopy, Element>;
#endif

    // Declare Copy plan.
    // Why this configuration?
    using GmemCopyLayoutAtom =
        Layout<Shape<Int<kThreads / (SmemKAtom / 8)>, Int<SmemKAtom / 8>>,
               Stride<Int<SmemKAtom / 8>, _ 1>>;

    using TiledCopyG2S = decltype(make_tiled_copy(
        CopyInstG2S{}, GmemCopyLayoutAtom{}, Layout<Shape<_1, _8>>{}));

    using TiledCopyS2G = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{}, GmemCopyLayoutAtom{},
        Layout<Shape<_1, _8>>{}));
};

template <typename InType, typename AccType, typename KeTraits, const int kM,
          const int kN, const kK, const int kP, const int kTM, const int kTN,
          const int kTK, const int kTP, const int Nthreads>
__global__ void __launch_bounds__(Nthreads)
    fa_kernel(const InType* dQ, const InType* dK, const InType* dV,
              InType* dO) {
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

    InType* sQ_ptr = reinterpret_cast<Element*>(buf);
    InType* sK_ptr = sQ_ptr + kTM * kTK;
    InType* sV_ptr = sK_ptr + kTN * kTK;
    InType* sO_ptr = sQ_ptr;

    typename KeTraits::TiledMma mma;
    typename KeTraits::TiledCopyG2S tiled_copy_g2s;

    auto rQ = make_s2rA(sQ_ptr, typename KeTraits::SmemLayoutQ{}, mma);
    auto rK = make_s2rA(sK_ptr, typename KeTraits::SmemLayoutK{}, mma);
    auto acc1 = get_acc<kM, kN>(mma);

    auto rV = make_s2rB(sV_ptr, typename KeTraits::SmemLayoutV{}, mma);
    auto acc2 = get_acc<kM, kP>(mma);
}

}  // namespace cutlass_wrapper
}  // namespace benchmarks