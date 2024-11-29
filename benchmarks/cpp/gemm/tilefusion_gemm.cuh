// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "cell/compute/mod.hpp"
#include "cell/copy/global_to_shared_2.hpp"
#include "cell/mod.hpp"
#include "types/mod.hpp"

#include <cute/tensor.hpp>

using namespace tilefusion;
using namespace tilefusion::cell;
using namespace tilefusion::cell::copy;
using namespace tilefusion::cell::compute;

namespace tl = tile_layout;

template <const int kM, const int kN, const int kK>
using GemmShape = TileShape<kM, kN, kK>;

using namespace cute;

template <typename InType, typename AccType, typename WholeShape,
          typename CtaTileShape, const int kRK, typename WarpLayout>
struct KeGemmTraits {
    using BaseShape = traits::BaseTileShape<InType>;

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;

    static const bool kSwizzled = true;

    // Total data access for operand A in global memory
    using GlobalA = GlobalTile<InType, tl::RowMajor<kTM, kTK, kK>>;
    // Shared Tile for operand A
    using SharedA = SharedTile<InType, tl::RowMajor<kTM, kTK>, kSwizzled>;
    using G2SLoaderA =
        tilefusion::cell::copy::detail::GlobalToSharedLoaderImpl2<
            GlobalA, SharedA, WarpLayout, SharedA::kType>;

    // Access a single register tile for operand A
    using SIteratorA = STileIterator<SharedA, TileShape<kTM, kRK>>;

    // Register tile for a single thread of operand A
    static constexpr int kAMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kAKs = kRK / BaseShape::kTileSize;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    using S2RLoaderA =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::kRowReuseCont>;

    // Total data access for operand B in global memory
    using GlobalB = GlobalTile<InType, tl::ColMajor<kTK, kTN, kK>>;
    // Shared Tile for operand B
    using SharedB = SharedTile<InType, tl::ColMajor<kTK, kTN>, kSwizzled>;
    using G2SLoaderB =
        tilefusion::cell::copy::detail::GlobalToSharedLoaderImpl2<
            GlobalB, SharedB, WarpLayout, SharedB::kType>;

    // Access a single register tile for operand B
    using SIteratorB = STileIterator<SharedB, TileShape<kRK, kTN>>;

    static_assert(SIteratorA::sc1 == SIteratorB::sc0,
                  "mismatched K dimension!");

    // Register tile for a single thread of operand A
    static constexpr int kBKs = kRK / BaseShape::kTileSize;
    static constexpr int kBNs = kTN / kWarpPerCol / BaseShape::kTileSize;
    using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

    using S2RLoaderB =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::kColReuseCont>;

    // Global Tile for output C
    using GlobalC = GlobalTile<InType, tl::RowMajor<kTM, kTN, kN>>;
    // Shared Tile for output C
    using SharedC = SharedTile<InType, tl::RowMajor<kTM, kTN>, kSwizzled>;

    // Register Tile for output C
    static constexpr int kCMs = kTM / kWarpPerRow / BaseShape::kTileSize;
    static constexpr int kCNs = kTN / kWarpPerCol / BaseShape::kTileSize;
    using Acc = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kCMs, kCNs>>;
    using AccHalf = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kCMs, kCNs>>;

    using CastAcc = compute::RegTileConvert<Acc, AccHalf>;

    using R2SStorerC = RegToSharedStorer<AccHalf, WarpLayout>;
    using S2GStorerC = SharedToGlobalStorer<SharedC, WarpLayout>;
};

template <typename InType,                                   //
          const int kM, const int kN, const int kK,          //
          const int kTM, const int kTN, const int kTK,       //
          typename SIteratorA,                               //
          typename SharedA, typename RegA,                   //
          typename G2SLoaderA, typename S2RLoaderA,          //
          typename SIteratorB,                               //
          typename SharedB, typename RegB,                   //
          typename G2SLoaderB, typename S2RLoaderB,          //
          typename GlobalC, typename SharedC,                //
          typename Acc, typename AccHalf, typename CastAcc,  //
          typename R2SStorerC, typename S2GStorerC>
__global__ void gemm(const InType* dA_, const InType* dB_, InType* dC_) {
    InType* dA = const_cast<InType*>(dA_) + blockIdx.x * kTM * kK;
    InType* dB = const_cast<InType*>(dB_) + blockIdx.y * kTN * kK;
    InType* dC = dC_ + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    extern __shared__ __align__(sizeof(double)) unsigned char buf[];
    InType* sA_ptr = reinterpret_cast<InType*>(buf);
    InType* sB_ptr = sA_ptr + SIteratorA::Tile::kNumel;
    InType* sC_ptr = reinterpret_cast<InType*>(buf);

    SharedA sA(sA_ptr);
    SIteratorA sAs(sA_ptr);
    RegA rA;

    SharedB sB(sB_ptr);
    SIteratorB sBs(sB_ptr);
    RegB rB;

    Acc acc;
    AccHalf acc_h;
    CastAcc cast;

    SharedC sC(sC_ptr);
    GlobalC gC(dC);

    G2SLoaderA g2s_a;
    S2RLoaderA s2r_a;

    G2SLoaderB g2s_b;
    S2RLoaderB s2r_b;

    R2SStorerC r2s_c;
    S2GStorerC s2g_c;

    for (int k1 = 0; k1 < kK; k1 += kTK) {
        g2s_a(dA, sA_ptr);
        g2s_a(dB, sB_ptr);
        __copy_async();
        __syncthreads();

        for (int k2 = 0; k2 < SIteratorA::sc1; ++k2) {
            s2r_a(sAs(k2), rA);
            s2r_b(sBs(k2), rB);

            gemm(rA, rB, acc);
        }
        dA += kTK;
        dB += kTK;
    }

    cast(acc, acc_h);
    r2s_c(acc_h, sC);
    __syncthreads();
    s2g_c(sC, gC);
}
