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

template <typename InType_, typename AccType_, typename WarpLayout,  //
          const int kM_, const int kN_, const int kK_, const int kP_,
          const int kTM_, const int kTN_, const int kTK_, const int kTP_>
struct FusedTwoGemmsTraits {
    /// constants
    using InType = InType_;
    using AccType = AccType_;

    static constexpr int kM = kM_;
    static constexpr int kN = kN_;
    static constexpr int kK = kK_;
    static constexpr int kP = kP_;

    static constexpr int kTM = kTM_;
    static constexpr int kTN = kTN_;
    static constexpr int kTK = kTK_;
    static constexpr int kTP = kTP_;

    static constexpr int kSharedAccess = 64;

    using BaseShape = traits::BaseTileShape<InType>;

    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;
    static_assert(kWarpPerCol == 1, "WarpPerCol must be 1");

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

template <typename InType, typename AccType, typename KeTraits>
__device__ __forceinline__ void ke_fused_two_gemms(const InType* dA,
                                                   const InType* dB,
                                                   const InType* dC,
                                                   InType* dD) {
    // constants
    static constexpr int kM = KeTraits::kM;
    static constexpr int kN = KeTraits::kN;
    static constexpr int kK = KeTraits::kK;
    static constexpr int kP = KeTraits::kP;

    static constexpr int kTM = KeTraits::kTM;
    static constexpr int kTP = KeTraits::kTP;

    using SharedA = KeTraits::SharedA;
    using SharedB = KeTraits::SharedB;
    using SharedC = KeTraits::SharedC;
    using SharedD = KeTraits::SharedD;

    // Advance to the global data tile to the current CTA.
    const InType* A = dA + blockIdx.z * (kM * kK) + blockIdx.x * (kTM * kK);
    const InType* B = dB + blockIdx.z * (kK * kN);
    const InType* gC_ptr =
        dC + blockIdx.z * (kN * kP) + blockIdx.y * (kTP * kN);

    InType* gD_ptr = dD + blockIdx.z * (kM * kP) + blockIdx.x * (kTM * kP) +
                     (blockIdx.y * kTP);

    // shared memory buffer
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<InType*>(shared_buf);

    InType* sA_ptr = shm;
    InType* sB_ptr = shm + SharedA::kNumel;
    InType* sC_ptr = shm + SharedA::kNumel + SharedB::kNumel;
    InType* sD_ptr = shm;

    // declare tile, iterators, loaders, and storers
    typename KeTraits::GIteratorA gAs(A);
    typename KeTraits::SharedA sA(sA_ptr);
    typename KeTraits::RegA rA;

    typename KeTraits::SharedALoader load_sa;
    typename KeTraits::RegALoader load_ra;

    typename KeTraits::GIteratorB gBs(B);
    typename KeTraits::SharedB sB(sB_ptr);
    typename KeTraits::RegB rB;

    typename KeTraits::SharedBLoader load_sb;
    typename KeTraits::RegBLoader load_rb;

    typename KeTraits::GIteratorC gCs(gC_ptr);
    typename KeTraits::SharedC sC(sC_ptr);

    typename KeTraits::SharedCLoader load_sc;
    typename KeTraits::RegCLoader load_rc;
    typename KeTraits::RegC rC;

    typename KeTraits::GlobalD gD(gD_ptr);
    typename KeTraits::SharedD sD(sD_ptr);
    typename KeTraits::RegD rD;
    typename KeTraits::RegDHalf rD_half;

    typename KeTraits::RegAcc acc;
    typename KeTraits::RegAccCast acc_half;

    typename KeTraits::ConvertHalf cast_acc;
    typename KeTraits::ConvertD convert_d;

    for (int n = 0; n < KeTraits::GIteratorC::sc0; ++n) {
        load_sc(gCs(n), sC);

        for (int k = 0; k < KeTraits::GIteratorA::sc1; ++k) {
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

    typename KeTraits::StoreRegD store_rD;
    store_rD(rD_half, sD);
    __syncthreads();

    typename KeTraits::StoreSharedD store_sD;
    store_sD(sD, gD);
}
}  // namespace tilefusion::kernels
