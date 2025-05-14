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
          const int kM_, const int kN_, const int kK_,               //
          const int kTM_, const int kTN_, const int kTK_,            //
          const int kRK_, const int kNumStages_, const int kSharedAccess = 64>
struct KeGemmTraits {
    using InType = InType_;
    using AccType = AccType_;
    using BaseShape = traits::BaseTileShape<InType>;
    static constexpr int kNumStages = kNumStages_;

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

    static constexpr int kM = kM_;
    static constexpr int kN = kN_;
    static constexpr int kK = kK_;

    static constexpr int kTM = kTM_;
    static constexpr int kTN = kTN_;
    static constexpr int kTK = kTK_;
    static constexpr int kRK = kRK_;

    static const bool kSwizzled = true;

    // Total data access for operand A in global memory
    using GlobalA = GlobalTile<InType, tl::RowMajor<kTM, kK, kK>>;
    // Access a single global tile for operand A
    using GIteratorA = GTileIterator<GlobalA, TileShape<kTM, kTK>>;

    // Shared Tile for operand A
    using SharedA =
        SharedTile<InType, tl::RowMajor<kTM, kTK>, kSwizzled, kSharedAccess>;
    // Access a single register tile for operand A
    using SIteratorA = STileIterator<SharedA, TileShape<kTM, kRK>>;

    // Register tile for a single thread of operand A
    static constexpr int kAMs = kTM / kWarpPerRow / BaseShape::kRows;
    static constexpr int kAKs = kRK / BaseShape::kCols;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    // Loaders for operand A
    using G2SLoaderA = GlobalToSharedLoader<SharedA, WarpLayout>;
    using S2RLoaderA =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::kRowReuseCont>;

    // Total data access for operand B in global memory
    using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kTN, kK>>;
    // Access a single global tile for operand B
    using GIteratorB = GTileIterator<GlobalB, TileShape<kTK, kTN>>;

    // Shared Tile for operand B
    using SharedB =
        SharedTile<InType, tl::ColMajor<kTK, kTN>, kSwizzled, kSharedAccess>;
    // Access a single register tile for operand B
    using SIteratorB = STileIterator<SharedB, TileShape<kRK, kTN>>;

    static_assert(GIteratorA::sc1 == GIteratorB::sc0,
                  "mismatched K dimension!");
    static_assert(SIteratorA::sc1 == SIteratorB::sc0,
                  "mismatched K dimension!");

    // Register tile for a single thread of operand A
    static constexpr int kBKs = kRK / BaseShape::kRows;
    static constexpr int kBNs = kTN / kWarpPerCol / BaseShape::kCols;
    using RegB = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kBKs, kBNs>>;

    using G2SLoaderB = GlobalToSharedLoader<SharedB, WarpLayout>;
    using S2RLoaderB =
        SharedToRegLoader<RegB, WarpLayout, WarpReuse::kColReuseCont>;

    // Global Tile for output C
    using GlobalC = GlobalTile<AccType, tl::RowMajor<kTM, kTN, kN>>;
    // Shared Tile for output C
    using SharedC = SharedTile<AccType, tl::RowMajor<kTM, kTN>, kSwizzled>;

    // Register Tile for output C
    static constexpr int kCMs = kTM / kWarpPerRow / BaseShape::kRows;
    static constexpr int kCNs = kTN / kWarpPerCol / BaseShape::kCols;
    using RegC = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kCMs, kCNs>>;

    using R2GStorerC = RegToGlobalStorer<GlobalC, RegC, WarpLayout>;
    using R2SStorerC = RegToSharedStorer<RegC, WarpLayout>;
    using S2GStorerC = SharedToGlobalStorer<SharedC, WarpLayout>;

    using PipelineG2SA = Pipeline<InType, GlobalA, SharedA, GIteratorA,
                                  G2SLoaderA, kNumStages, GIteratorA::sc1>;
    using PipelineG2SB = Pipeline<InType, GlobalB, SharedB, GIteratorB,
                                  G2SLoaderB, kNumStages, GIteratorA::sc1>;

    using PipelineS2RA = Pipeline<InType, SharedA, RegA, SIteratorA, S2RLoaderA,
                                  kNumStages - 1, SIteratorA::sc1>;
    using PipelineS2RB = Pipeline<InType, SharedB, RegB, SIteratorB, S2RLoaderB,
                                  kNumStages - 1, SIteratorA::sc1>;
};

template <typename InType, typename AccType, typename KeTraits>
__device__ __forceinline__ void ke_gemm(const InType* dA, const InType* dB,
                                        AccType* dC) {
    static constexpr int kN = KeTraits::kN;
    static constexpr int kK = KeTraits::kK;
    static constexpr int kTM = KeTraits::kTM;
    static constexpr int kTN = KeTraits::kTN;

    int offset_a = blockIdx.x * kTM * kK;
    int offset_b = blockIdx.y * kTN * kK;
    int offset_c = blockIdx.x * kTM * kN + blockIdx.y * kTN;

    extern __shared__ __align__(sizeof(double)) unsigned char buf[];
    InType* sA_ptr = reinterpret_cast<InType*>(buf);
    InType* sB_ptr = sA_ptr + KeTraits::SIteratorA::Tile::kNumel;
    AccType* sC_ptr = reinterpret_cast<AccType*>(buf);

    // declare tiles, iterators and loaders
    typename KeTraits::GIteratorA gAs(dA + offset_a);
    typename KeTraits::SIteratorA sAs(sA_ptr);

    typename KeTraits::GIteratorB gBs(dB + offset_b);
    typename KeTraits::SIteratorB sBs(sB_ptr);

    typename KeTraits::SharedA sA(sA_ptr);
    typename KeTraits::RegA rA;

    typename KeTraits::SharedB sB(sB_ptr);
    typename KeTraits::RegB rB;

    typename KeTraits::RegC acc;
    typename KeTraits::SharedC sC(sC_ptr);
    typename KeTraits::GlobalC gC(dC + offset_c);

    typename KeTraits::G2SLoaderA g2s_a;
    typename KeTraits::S2RLoaderA s2r_a;

    typename KeTraits::G2SLoaderB g2s_b;
    typename KeTraits::S2RLoaderB s2r_b;

    typename KeTraits::R2SStorerC r2s_c;
    typename KeTraits::S2GStorerC s2g_c;

    for (int k1 = 0; k1 < KeTraits::GIteratorA::sc1; ++k1) {
        g2s_a(gAs(k1), sA);
        g2s_b(gBs(k1), sB);
        __copy_async();
        __syncthreads();
        // #pragma unroll
        for (int k2 = 0; k2 < KeTraits::SIteratorA::sc1; ++k2) {
            s2r_a(sAs(k2), rA);
            s2r_b(sBs(k2), rB);

            compute::gemm(rA, rB, acc);
        }
    }
    r2s_c(acc, sC);
    __syncthreads();
    s2g_c(sC, gC);
}

template <typename InType, typename AccType, typename KeTraits>
__device__ __forceinline__ void ke_gemm_level1_pipeline(const InType* dA,
                                                        const InType* dB,
                                                        AccType* dC) {
    static constexpr int kTM = KeTraits::kTM;
    static constexpr int kTN = KeTraits::kTN;
    static constexpr int kN = KeTraits::kN;
    static constexpr int kK = KeTraits::kK;
    static constexpr int kNumStages = KeTraits::kNumStages;

    int offset_a = blockIdx.x * kTM * kK;
    int offset_b = blockIdx.y * kTN * kK;
    int offset_c = blockIdx.x * kTM * kN + blockIdx.y * kTN;

    extern __shared__ __align__(sizeof(double)) unsigned char buf[];
    InType* sA_ptr = reinterpret_cast<InType*>(buf);
    InType* sB_ptr = sA_ptr + KeTraits::SIteratorA::Tile::kNumel * kNumStages;
    AccType* sC_ptr = reinterpret_cast<AccType*>(buf);

    typename KeTraits::RegA rA;
    typename KeTraits::RegB rB;

    typename KeTraits::RegC acc;
    typename KeTraits::SharedC sC(sC_ptr);
    typename KeTraits::GlobalC gC(dC + offset_c);

    typename KeTraits::G2SLoaderA g2s_a;
    typename KeTraits::S2RLoaderA s2r_a;

    typename KeTraits::G2SLoaderB g2s_b;
    typename KeTraits::S2RLoaderB s2r_b;

    typename KeTraits::PipelineG2SA pipeline_g2s_a(dA + offset_a, sA_ptr);
    typename KeTraits::PipelineG2SB pipeline_g2s_b(dB + offset_b, sB_ptr);

    // Issue the global to shared copy before main loop.
    pipeline_g2s_a.commit();
    pipeline_g2s_b.commit();
    commit_copy_group();

    for (int k = 0; k < KeTraits::PipelineG2SA::Iterations - 1; ++k) {
        // Barrier to wait for the previous copy to finish.
        wait_group<0>();
        __syncthreads();
        pipeline_g2s_a.commit();
        pipeline_g2s_b.commit();
        commit_copy_group();
        // Compute(i - 1)
        const InType* sA_ptr_prev = pipeline_g2s_a.get_prev_dst();
        const InType* sB_ptr_prev = pipeline_g2s_b.get_prev_dst();
        typename KeTraits::SIteratorA sAs(sA_ptr_prev);
        typename KeTraits::SIteratorB sBs(sB_ptr_prev);
        // #pragma unroll
        for (int k2 = 0; k2 < KeTraits::SIteratorA::sc1; ++k2) {
            s2r_a(sAs(k2), rA);
            s2r_b(sBs(k2), rB);
            compute::gemm(rA, rB, acc);
        }
    }
    wait_group<0>();
    __syncthreads();

    // Compute(i)
    const InType* sA_ptr_cur = pipeline_g2s_a.get_cur_dst();
    const InType* sB_ptr_cur = pipeline_g2s_b.get_cur_dst();
    typename KeTraits::SIteratorA sAs(sA_ptr_cur);
    typename KeTraits::SIteratorB sBs(sB_ptr_cur);
    for (int k2 = 0; k2 < KeTraits::SIteratorA::sc1; ++k2) {
        s2r_a(sAs(k2), rA);
        s2r_b(sBs(k2), rB);
        compute::gemm(rA, rB, acc);
    }
    __syncthreads();

    // Store the result from register tile to global memory.
    typename KeTraits::R2SStorerC r2s_c;
    typename KeTraits::S2GStorerC s2g_c;
    r2s_c(acc, sC);
    __syncthreads();
    s2g_c(sC, gC);
}

template <typename InType, typename AccType, typename KeTraits>
__device__ __forceinline__ void ke_gemm_level2_pipeline(const InType* dA,
                                                        const InType* dB,
                                                        AccType* dC) {
    static constexpr int kTM = KeTraits::kTM;
    static constexpr int kTN = KeTraits::kTN;
    static constexpr int kN = KeTraits::kN;
    static constexpr int kK = KeTraits::kK;
    static constexpr int kNumStages = KeTraits::kNumStages;

    int offset_a = blockIdx.x * kTM * kK;
    int offset_b = blockIdx.y * kTN * kK;
    int offset_c = blockIdx.x * kTM * kN + blockIdx.y * kTN;

    extern __shared__ __align__(sizeof(double)) unsigned char buf[];
    InType* sA_ptr = reinterpret_cast<InType*>(buf);
    InType* sB_ptr = sA_ptr + KeTraits::SIteratorA::Tile::kNumel * kNumStages;
    AccType* sC_ptr = reinterpret_cast<AccType*>(buf);

    // Declare the cycle buffer for the register tiles.
    typename KeTraits::RegA rA_cyc_buf[kNumStages - 1];
    typename KeTraits::RegB rB_cyc_buf[kNumStages - 1];

    typename KeTraits::RegC acc;
    typename KeTraits::SharedC sC(sC_ptr);
    typename KeTraits::GlobalC gC(dC + offset_c);

    typename KeTraits::G2SLoaderA g2s_a;
    typename KeTraits::S2RLoaderA s2r_a;

    typename KeTraits::G2SLoaderB g2s_b;
    typename KeTraits::S2RLoaderB s2r_b;

    typename KeTraits::PipelineG2SA pipeline_g2s_a(dA + offset_a, sA_ptr);
    typename KeTraits::PipelineG2SB pipeline_g2s_b(dB + offset_b, sB_ptr);

    // In 3-stage pipeline, we need to issue 2 global to shared copies before
    // the main loop.

    // We issue copy instructions using 2 commit groups.
    pipeline_g2s_a.commit();
    pipeline_g2s_b.commit();
    commit_copy_group();

    pipeline_g2s_a.commit();
    pipeline_g2s_b.commit();
    commit_copy_group();

    // Wait for at least 1 copy to finish.
    wait_group<1>();
    __syncthreads();

    const InType* sA0 = pipeline_g2s_a.get_dst_ptr_by_index(0);
    const InType* sB0 = pipeline_g2s_b.get_dst_ptr_by_index(0);

    typename KeTraits::PipelineS2RA pipeline_s2r_a(sA0, rA_cyc_buf);
    typename KeTraits::PipelineS2RB pipeline_s2r_b(sB0, rB_cyc_buf);

    // Issue the first data loading from shared memory to register tile.
    pipeline_s2r_a.commit();
    pipeline_s2r_b.commit();
    auto rA = pipeline_s2r_a.get_dst_tile_by_index(0);
    auto rB = pipeline_s2r_b.get_dst_tile_by_index(0);

    // gemm stage 1: handle all global to shared copies.
    // BLOCK: GIteratorA::sc1 - 2
#pragma unroll
    for (int k = 0; k < KeTraits::PipelineG2SA::Iterations - 2; ++k) {
        // NOTE(KuangjuX): we have to add `#pragma unroll` here, otherwise
        // misaligned errors will be reported.
#pragma unroll
        for (int k2 = 0; k2 < KeTraits::PipelineS2RA::Iterations; ++k2) {
            // circular issue next data loading from shared memory to register
            // tile.

            pipeline_s2r_a.commit();
            pipeline_s2r_b.commit();

            if (k2 == KeTraits::PipelineS2RA::Iterations - 2) {
                wait_group<0>();
                __syncthreads();
                /**
                 * When `k2 == PipelineS2RA::Iterations - 2`, the current shared
                 * tile has just been traversed and needs to be replaced with a
                 * new shared tile. Since `PipelineG2S` is emitted twice before
                 * the loop, and `S2R` obtains the data of the first emission
                 * outside the loop, when `k = 0`, the index of the data to be
                 * obtained is `k + 1`.
                 */
                auto sA = pipeline_g2s_a.get_cur_dst();
                auto sB = pipeline_g2s_b.get_cur_dst();
                // reset the shared tile in shared to register copy.
                pipeline_s2r_a.reset_src_tile(sA);
                pipeline_s2r_b.reset_src_tile(sB);
            }

            // execute gemm operation in previous register tile.
            auto rA = pipeline_s2r_a.get_dst_tile_by_index(k2);
            auto rB = pipeline_s2r_b.get_dst_tile_by_index(k2);
            __syncthreads();
            compute::gemm(rA, rB, acc);
        }

        // Issue the next global to shared copy.
        pipeline_g2s_a.commit();
        pipeline_g2s_b.commit();
        commit_copy_group();
    }

    // gemm stage 2: handle the second-to-last shared tile.
    // NOTE(KuangjuX): we have to add `#pragma unroll` here, otherwise
    // misaligned errors will be reported.
#pragma unroll
    for (int k2 = 0; k2 < KeTraits::PipelineS2RA::Iterations; ++k2) {
        // circular issue next data loading from shared memory to register
        // tile.

        pipeline_s2r_a.commit();
        pipeline_s2r_b.commit();

        if (k2 == KeTraits::PipelineS2RA::Iterations - 2) {
            // Wait the last global to shared tile copy to finish.
            wait_group<0>();
            __syncthreads();

            // fetch the last shared tile in global memory.
            const InType* sA = pipeline_g2s_a.get_cur_dst();
            const InType* sB = pipeline_g2s_b.get_cur_dst();

            // reset the last shared tile in shared to register copy.
            pipeline_s2r_a.reset_src_tile(sA);
            pipeline_s2r_b.reset_src_tile(sB);
        }

        auto rA = pipeline_s2r_a.get_dst_tile_by_index(k2);
        auto rB = pipeline_s2r_b.get_dst_tile_by_index(k2);

        compute::gemm(rA, rB, acc);
    }

    // gemm stage 3: handle the last shared tile
    // NOTE(KuangjuX): we have to add `#pragma unroll` here, otherwise
    // misaligned errors will be reported.
#pragma unroll
    for (int k2 = 0; k2 < KeTraits::PipelineS2RA::Iterations; ++k2) {
        // In last stage, we only need to issue Iterations - 1 times
        // data loading from shared memory to register tile beacuase
        // we have already done an advance copy in the previous stage.
        if (k2 < KeTraits::PipelineS2RA::Iterations - 1) {
            pipeline_s2r_a.commit();
            pipeline_s2r_b.commit();
        }

        auto rA = pipeline_s2r_a.get_dst_tile_by_index(k2);
        auto rB = pipeline_s2r_b.get_dst_tile_by_index(k2);

        compute::gemm(rA, rB, acc);
    }

    __syncthreads();

    typename KeTraits::R2SStorerC r2s_c;
    typename KeTraits::S2GStorerC s2g_c;
    r2s_c(acc, sC);
    __syncthreads();
    s2g_c(sC, gC);
}

}  // namespace tilefusion::kernels
