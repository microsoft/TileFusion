// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "cell/mod.hpp"
#include "types/mod.hpp"

using namespace tilefusion;
using namespace cell;
using namespace copy;
using namespace compute;
namespace tl = tile_layout;

namespace tilefusion::kernels {

template <typename InType, typename AccType, typename WarpLayout,  //
          const int kM, const int kN, const int kK, const int kP>
struct FusedTwoGemmsTraits {
    static constexpr int kTM = 64;
    static constexpr int kTN = 64;
    static constexpr int kTK = 64;
    static constexpr int kTP = 64;

    static constexpr int kSharedAccess = 64;

    static constexpr int kShmInput = (kTM * kTK + kTK * kTN + kTN * kTP);
    static constexpr int kShmOutput = kTM * kTP;
    static constexpr int kShmSize = kShmInput < kShmOutput
                                        ? kShmOutput * sizeof(InType)
                                        : kShmInput * sizeof(InType);

    using BaseShape = traits::BaseTileShape<InType>;

    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;
    static_assert(kWarpPerCol == 1, "WarpPerCol must be 1");

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;

    // operand A
    using GlobalA = GlobalTile<InType, tl::RowMajor<kTM, kK>>;
    // chunk the K dimension to fit into shared memory
    using GIteratorA = GTileIterator<GlobalA, TileShape<kTM, kTK>>;

    static const bool kUseSwizzling = true;

    using SharedA = SharedTile<InType, tl::RowMajor<kTM, kTK>, kUseSwizzling,
                               kSharedAccess>;

    static constexpr int kAMs = kTM / kWarpPerRow / BaseShape::kRows;
    static constexpr int kAKs = kTK / BaseShape::kCols;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    using SharedALoader = GlobalToSharedLoader<SharedA, WarpLayout>;
    using RegALoader =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::kRowReuseCont>;

    // operand B
    using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kN>>;
    using GIteratorB = GTileIterator<GlobalB, TileShape<kTK, kTN>>;
    using SharedB = SharedTile<InType, tl::ColMajor<kTK, kTN>, kUseSwizzling,
                               kSharedAccess>;

    static constexpr int kBKs = kTK / BaseShape::kRows;
    static constexpr int kBNs = kTN / kWarpPerCol / BaseShape::kCols;
    using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

    using SharedBLoader = GlobalToSharedLoader<SharedB, WarpLayout>;
    using RegBLoader =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::kColReuseCont>;

    // operand C
    using GlobalC = GlobalTile<InType, tl::ColMajor<kN, kTP>>;
    // chunk the N dimension to fit into shared memory
    using GIteratorC = GTileIterator<GlobalC, TileShape<kTN, kTP>>;
    using SharedC = SharedTile<InType, tl::ColMajor<kTN, kTP>, kUseSwizzling,
                               kSharedAccess>;

    static constexpr int kCNs = kTN / BaseShape::kRows;
    static constexpr int kCPs = kTP / kWarpPerCol / BaseShape::kCols;
    using RegC = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kCNs, kCPs>>;

    using SharedCLoader = GlobalToSharedLoader<SharedC, WarpLayout>;
    using RegCLoader =
        SharedToRegLoader<RegC, WarpLayout, WarpReuse::kColReuseCont>;

    // output D
    using GlobalD = GlobalTile<InType, tl::RowMajor<kTM, kTP>>;
    using SharedD = SharedTile<InType, tl::RowMajor<kTM, kTP>, kUseSwizzling,
                               kSharedAccess>;

    static constexpr int kDMs = kTM / kWarpPerRow / BaseShape::kRows;
    static constexpr int kDPs = kTP / kWarpPerCol / BaseShape::kCols;
    using RegD = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kDMs, kDPs>>;
    using RegDHalf =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kDMs, kDPs>>;

    static constexpr int kAccMs = kTM / kWarpPerRow / BaseShape::kRows;
    static constexpr int kAccNs = kTN / kWarpPerCol / BaseShape::kCols;

    // Reg Acc
    using RegAcc =
        RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kAccMs, kAccNs>>;
    using RegAccCast =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAccMs, kAccNs>>;

    // Convert the accumulator to half
    using ConvertHalf = compute::RegTileConvert<RegAcc, RegAccCast>;
    using ConvertD = compute::RegTileConvert<RegD, RegDHalf>;

    using StoreRegD = RegToSharedStorer<RegDHalf, WarpLayout>;
    using StoreSharedD = SharedToGlobalStorer<SharedD, WarpLayout>;
};

template <typename InType, typename AccType,                     //
          typename GIteratorA, typename SharedA, typename RegA,  //
          typename SharedALoader, typename RegALoader,           //
          typename GIteratorB, typename SharedB, typename RegB,  //
          typename SharedBLoader, typename RegBLoader,           //
          typename GIteratorC, typename SharedC, typename RegC,  //
          typename SharedCLoader, typename RegCLoader,           //
          typename RegAcc, typename RegAccCast, typename GlobalD,
          typename SharedD, typename RegD, typename RegDHalf,
          typename StoreRegD, typename StoreSharedD, typename ConvertAcc,
          typename ConvertD>
__device__ void ke_fused_two_gemms(const InType* dA, const InType* dB,
                                   const InType* dC, InType* dD, int kM, int kN,
                                   int kK, int kP, int kTM, int kTN, int kTK,
                                   int kTP) {
    // Advance to the global data tile to the current CTA.
    const InType* A = dA + blockIdx.z * (kM * kK) + blockIdx.x * (kTM * kK);
    const InType* B = dB + blockIdx.z * (kK * kN);
    const InType* gC_ptr =
        dC + blockIdx.z * (kN * kP) + blockIdx.y * (kTP * kN);

    InType* gD_ptr = dD + blockIdx.z * (kM * kP) + blockIdx.x * (kTM * kP) +
                     (blockIdx.y * kTP);

    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<InType*>(shared_buf);

    InType* sA_ptr = shm;
    InType* sB_ptr = shm + SharedA::kNumel;
    InType* sC_ptr = shm + SharedA::kNumel + SharedB::kNumel;
    InType* sD_ptr = shm;

    GIteratorA gAs(A);
    SharedA sA(sA_ptr);
    RegA rA;

    SharedALoader load_sa;
    RegALoader load_ra;

    GIteratorB gBs(B);
    SharedB sB(sB_ptr);
    RegB rB;

    SharedBLoader load_sb;
    RegBLoader load_rb;

    GIteratorC gCs(gC_ptr);
    SharedC sC(sC_ptr);

    SharedCLoader load_sc;
    RegCLoader load_rc;
    RegC rC;

    GlobalD gD(gD_ptr);
    SharedD sD(sD_ptr);
    RegD rD;
    RegDHalf rD_half;
    StoreRegD store_rD;
    StoreSharedD store_sD;

    RegAcc acc;
    RegAccCast acc_half;

    ConvertAcc cast_acc;  // Convert acc to half precision
    ConvertD convert_d;   // Convert D to half precision

    for (int n = 0; n < GIteratorC::sc0; ++n) {
        load_sc(gCs(n), sC);

        for (int k = 0; k < GIteratorA::sc1; ++k) {
            load_sa(gAs(k), sA);
            load_sb(gBs(k, n), sB);
            __copy_async();
            __syncthreads();

            load_ra(sA, rA);
            load_rb(sB, rB);
            __syncthreads();
            gemm(rA, rB, acc);
        }
        load_rc(sC, rC);
        __syncthreads();

        cast_acc(acc, acc_half);

        gemm(acc_half, rC, rD);
        acc.clear();
    }
    __syncthreads();
    convert_d(rD, rD_half);

    store_rD(rD_half, sD);
    __syncthreads();
    store_sD(sD, gD);
}
}  // namespace tilefusion::kernels
