# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

config = """#include "../gemm.hpp"

static constexpr int kWarpPerRow = {kWarpPerRow};
static constexpr int kWarpPerCol = {kWarpPerCol};

static constexpr int kM = {kM};
static constexpr int kN = {kN};
static constexpr int kK = {kK};

static constexpr int kTM = {kTM};
static constexpr int kTN = {kTN};
static constexpr int kTK = {kTK};

static constexpr int kRK = {kRK};
"""

kernel_entry = """
extern "C" int kernel_entry(const __half* A, const __half* B, __half* C) {
    using InType = __half;
    using AccType = float;

    using WholeShape = GemmShape<kM, kN, kK>;
    using CtaTileShape = GemmShape<kTM, kTN, kTK>;
    using WarpLayout = tl::RowMajor<kWarpPerRow, kWarpPerCol>;

    using Config = KeGemmTraits<InType, AccType, WholeShape, CtaTileShape, kRK,
                                WarpLayout>;

    auto kernel =
        &gemm<InType, AccType, kM, kN, kK, kTM, kTN, kTK,
              typename Config::GIteratorA, typename Config::SIteratorA,
              typename Config::SharedA, typename Config::RegA,
              typename Config::G2SLoaderA, typename Config::S2RLoaderA,
              typename Config::GIteratorB, typename Config::SIteratorB,
              typename Config::SharedB, typename Config::RegB,
              typename Config::G2SLoaderB, typename Config::S2RLoaderB,
              typename Config::GlobalC, typename Config::SharedC,
              typename Config::RegC, typename Config::RegCHalf,
              typename Config::ConvertHalf, typename Config::R2SStorerC,
              typename Config::S2GStorerC>;

    static constexpr int smem_size_inputs = kTK * (kTN + kTM) * sizeof(InType);
    static constexpr int smem_size_accumulators = kTM * kTN * sizeof(AccType);
    static constexpr int smem_size = smem_size_inputs > smem_size_accumulators
                                         ? smem_size_inputs
                                         : smem_size_accumulators;

    const int kMaxSmemPerBlock = 48 * 1024;
    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    int block_x = CeilDiv<kM, kTM>;
    int block_y = CeilDiv<kN, kTN>;

    dim3 dim_grid(block_x, block_y, 1);
    dim3 dim_block(Config::kThreads, 1, 1);

    kernel<<<dim_grid, dim_block, smem_size>>>(A, B, C);
    
    return 0;
}
"""
