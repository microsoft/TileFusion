// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cell/mod.hpp"
#include "kernels/ops.hpp"
#include "types/mod.hpp"

using namespace tilefusion;
using namespace cell;
using namespace cell::copy;
using namespace cell::compute;
namespace tl = tile_layout;

namespace tilefusion::kernels {

template <const int kM, const int kN, const int kK>
using GemmShape = TileShape<kM, kN, kK>;

template <typename InType, typename AccType, typename WholeShape,
          typename CtaTileShape, const int kRK, typename WarpLayout,
          const int kNumStages, const int kSharedAccess = 64>
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

template <typename InType, typename AccType,                  //
          const int kM, const int kN, const int kK,           //
          const int kTM, const int kTN, const int kTK,        //
          typename GIteratorA, typename SIteratorA,           //
          typename SharedA, typename RegA,                    //
          typename G2SLoaderA, typename S2RLoaderA,           //
          typename GIteratorB, typename SIteratorB,           //
          typename SharedB, typename RegB,                    //
          typename G2SLoaderB, typename S2RLoaderB,           //
          typename GlobalC, typename SharedC, typename RegC,  //
          typename R2SStorerC, typename S2GStorerC>
__global__ void ke_gemm(const InType* dA, const InType* dB, AccType* dC) {
    int offset_a = blockIdx.x * kTM * kK;
    int offset_b = blockIdx.y * kTN * kK;
    int offset_c = blockIdx.x * kTM * kN + blockIdx.y * kTN;

    extern __shared__ __align__(sizeof(double)) unsigned char buf[];
    InType* sA_ptr = reinterpret_cast<InType*>(buf);
    InType* sB_ptr = sA_ptr + SIteratorA::Tile::kNumel;
    AccType* sC_ptr = reinterpret_cast<AccType*>(buf);

    // declare tiles, iterators and loaders
    GIteratorA gAs(dA + offset_a);
    SIteratorA sAs(sA_ptr);

    GIteratorB gBs(dB + offset_b);
    SIteratorB sBs(sB_ptr);

    SharedA sA(sA_ptr);
    RegA rA;

    SharedB sB(sB_ptr);
    RegB rB;

    RegC acc;
    SharedC sC(sC_ptr);
    GlobalC gC(dC + offset_c);

    G2SLoaderA g2s_a;
    S2RLoaderA s2r_a;

    G2SLoaderB g2s_b;
    S2RLoaderB s2r_b;

    R2SStorerC r2s_c;
    S2GStorerC s2g_c;

    for (int k1 = 0; k1 < GIteratorA::sc1; ++k1) {
        g2s_a(gAs(k1), sA);
        g2s_b(gBs(k1), sB);
        __copy_async();
        __syncthreads();

        for (int k2 = 0; k2 < SIteratorA::sc1; ++k2) {
            s2r_a(sAs(k2), rA);
            s2r_b(sBs(k2), rB);

            compute::gemm(rA, rB, acc);
        }
    }
    r2s_c(acc, sC);
    __syncthreads();
    s2g_c(sC, gC);
}

template <typename InType, typename AccType,                  //
          const int kM, const int kN, const int kK,           //
          const int kTM, const int kTN, const int kTK,        //
          const int kNumStages,                               //
          typename SharedA, typename RegA,                    //
          typename G2SLoaderA, typename S2RLoaderA,           //
          typename SharedB, typename RegB,                    //
          typename G2SLoaderB, typename S2RLoaderB,           //
          typename SIteratorA, typename SIteratorB,           //
          typename GlobalC, typename SharedC, typename RegC,  //
          typename R2SStorerC, typename S2GStorerC,           //
          typename PipelineG2SA, typename PipelineG2SB>
__global__ void ke_gemm_level1_pipeline(const InType* dA, const InType* dB,
                                        AccType* dC) {
    int offset_a = blockIdx.x * kTM * kK;
    int offset_b = blockIdx.y * kTN * kK;
    int offset_c = blockIdx.x * kTM * kN + blockIdx.y * kTN;

    extern __shared__ __align__(sizeof(double)) unsigned char buf[];
    InType* sA_ptr = reinterpret_cast<InType*>(buf);
    InType* sB_ptr = sA_ptr + SIteratorA::Tile::kNumel * kNumStages;
    AccType* sC_ptr = reinterpret_cast<AccType*>(buf);

    RegA rA;
    RegB rB;

    RegC acc;
    SharedC sC(sC_ptr);
    GlobalC gC(dC + offset_c);

    G2SLoaderA g2s_a;
    S2RLoaderA s2r_a;

    G2SLoaderB g2s_b;
    S2RLoaderB s2r_b;

    PipelineG2SA pipeline_g2s_a(dA + offset_a, sA_ptr);
    PipelineG2SB pipeline_g2s_b(dB + offset_b, sB_ptr);

    // Issue the global to shared copy before
    // main loop.
    pipeline_g2s_a.commit();
    pipeline_g2s_b.commit();
    commit_copy_group();

    for (int k = 0; k < PipelineG2SA::Iterations - 1; ++k) {
        // Barrier to wait for the previous copy to finish.
        wait_group<0>();
        __syncthreads();
        pipeline_g2s_a.commit();
        pipeline_g2s_b.commit();
        commit_copy_group();
        // Compute(i - 1)
        const InType* sA_ptr_prev = pipeline_g2s_a.get_prev_dst();
        const InType* sB_ptr_prev = pipeline_g2s_b.get_prev_dst();
        SIteratorA sAs(sA_ptr_prev);
        SIteratorB sBs(sB_ptr_prev);
        for (int k2 = 0; k2 < SIteratorA::sc1; ++k2) {
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
    SIteratorA sAs(sA_ptr_cur);
    SIteratorB sBs(sB_ptr_cur);
    for (int k2 = 0; k2 < SIteratorA::sc1; ++k2) {
        s2r_a(sAs(k2), rA);
        s2r_b(sBs(k2), rB);
        compute::gemm(rA, rB, acc);
    }
    __syncthreads();
    // Store the result from register tile to global memory.
    R2SStorerC r2s_c;
    S2GStorerC s2g_c;
    r2s_c(acc, sC);
    __syncthreads();
    s2g_c(sC, gC);
}

template <typename InType, typename AccType,                  //
          const int kM, const int kN, const int kK,           //
          const int kTM, const int kTN, const int kTK,        //
          const int kNumStages,                               //
          typename SharedA, typename RegA,                    //
          typename G2SLoaderA, typename S2RLoaderA,           //
          typename SharedB, typename RegB,                    //
          typename G2SLoaderB, typename S2RLoaderB,           //
          typename SIteratorA, typename SIteratorB,           //
          typename GlobalC, typename SharedC, typename RegC,  //
          typename R2SStorerC, typename S2GStorerC,           //
          typename PipelineG2SA, typename PipelineG2SB,       //
          typename PipelineS2RA, typename PipelineS2RB>
__global__ void ke_gemm_level2_pipeline(const InType* dA, const InType* dB,
                                        AccType* dC) {
    int offset_a = blockIdx.x * kTM * kK;
    int offset_b = blockIdx.y * kTN * kK;
    int offset_c = blockIdx.x * kTM * kN + blockIdx.y * kTN;

    extern __shared__ __align__(sizeof(double)) unsigned char buf[];
    InType* sA_ptr = reinterpret_cast<InType*>(buf);
    InType* sB_ptr = sA_ptr + SIteratorA::Tile::kNumel * kNumStages;
    AccType* sC_ptr = reinterpret_cast<AccType*>(buf);

    // Declare the cycle buffer for the register tiles.
    RegA rA_cyc_buf[kNumStages - 1];
    RegB rB_cyc_buf[kNumStages - 1];

    RegC acc;
    SharedC sC(sC_ptr);
    GlobalC gC(dC + offset_c);

    G2SLoaderA g2s_a;
    S2RLoaderA s2r_a;

    G2SLoaderB g2s_b;
    S2RLoaderB s2r_b;

    PipelineG2SA pipeline_g2s_a(dA + offset_a, sA_ptr);
    PipelineG2SB pipeline_g2s_b(dB + offset_b, sB_ptr);

    // In 3-stage pipeline, we need to issue 2 global to shared copies
    // before the main loop.

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

    PipelineS2RA pipeline_s2r_a(sA0, rA_cyc_buf);
    PipelineS2RB pipeline_s2r_b(sB0, rB_cyc_buf);

    // Issue the first data loading from shared memory to register tile.
    pipeline_s2r_a.commit();
    pipeline_s2r_b.commit();
    auto rA = pipeline_s2r_a.get_dst_tile_by_index(0);
    auto rB = pipeline_s2r_b.get_dst_tile_by_index(0);

    // gemm stage 1: handle all global to shared copies.
    // BLOCK: GIteratorA::sc1 - 2
#pragma unroll
    for (int k = 0; k < PipelineG2SA::Iterations - 2; ++k) {
        // NOTE(KuangjuX): we have to add `#pragma unroll` here, otherwise
        // misaligned errors will be reported.
#pragma unroll
        for (int k2 = 0; k2 < PipelineS2RA::Iterations; ++k2) {
            // circular issue next data loading from shared memory to register
            // tile.

            pipeline_s2r_a.commit();
            pipeline_s2r_b.commit();

            if (k2 == PipelineS2RA::Iterations - 2) {
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
    for (int k2 = 0; k2 < PipelineS2RA::Iterations; ++k2) {
        // circular issue next data loading from shared memory to register
        // tile.

        pipeline_s2r_a.commit();
        pipeline_s2r_b.commit();

        if (k2 == PipelineS2RA::Iterations - 2) {
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
    for (int k2 = 0; k2 < PipelineS2RA::Iterations; ++k2) {
        // In last stage, we only need to issue Iterations - 1 times
        // data loading from shared memory to register tile beacuase
        // we have already done an advance copy in the previous stage.
        if (k2 < PipelineS2RA::Iterations - 1) {
            pipeline_s2r_a.commit();
            pipeline_s2r_b.commit();
        }

        auto rA = pipeline_s2r_a.get_dst_tile_by_index(k2);
        auto rB = pipeline_s2r_b.get_dst_tile_by_index(k2);

        compute::gemm(rA, rB, acc);
    }

    __syncthreads();

    R2SStorerC r2s_c;
    S2GStorerC s2g_c;
    r2s_c(acc, sC);
    __syncthreads();
    s2g_c(sC, gC);
}

template <typename InType, typename AccType, typename WholeShape,
          typename CtaTileShape, const int kRK, typename WarpLayout,
          const int kNumStages>
void run_gemm(const InType* dA, const InType* dB, AccType* dC, int64_t m,
              int64_t n, int64_t k, int64_t num_stages) {
    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;

    using Config = KeGemmTraits<InType, AccType, WholeShape, CtaTileShape, kRK,
                                WarpLayout, kNumStages>;

    int block_x = CeilDiv<kM, kTM>;
    int block_y = CeilDiv<kN, kTN>;
    int block_z = 1;

    dim3 grid(block_x, block_y, block_z);
    dim3 block(Config::kThreads, 1, 1);

    int shm_input = (kTM * kTK + kTK * kTN) * kNumStages;
    int shm_output = kTM * kTN;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(InType)
                                          : shm_input * sizeof(InType);

    using SharedA = typename Config::SharedA;
    using RegA = typename Config::RegA;
    using G2SLoaderA = typename Config::G2SLoaderA;
    using S2RLoaderA = typename Config::S2RLoaderA;
    using SharedB = typename Config::SharedB;
    using RegB = typename Config::RegB;
    using G2SLoaderB = typename Config::G2SLoaderB;
    using S2RLoaderB = typename Config::S2RLoaderB;
    using GlobalC = typename Config::GlobalC;
    using SharedC = typename Config::SharedC;
    using RegC = typename Config::RegC;
    using R2SStorerC = typename Config::R2SStorerC;
    using S2GStorerC = typename Config::S2GStorerC;
    using R2GStorerC = typename Config::R2GStorerC;
    using GIteratorA = typename Config::GIteratorA;
    using GIteratorB = typename Config::GIteratorB;
    using SIteratorA = typename Config::SIteratorA;
    using SIteratorB = typename Config::SIteratorB;
    using PipelineG2SA = typename Config::PipelineG2SA;
    using PipelineG2SB = typename Config::PipelineG2SB;
    using PipelineS2RA = typename Config::PipelineS2RA;
    using PipelineS2RB = typename Config::PipelineS2RB;

    using KernelType = void (*)(const InType*, const InType*, AccType*);
    KernelType kernel = nullptr;

    if (num_stages == 1) {
        kernel = &ke_gemm<InType, AccType, kM, kN, kK, kTM, kTN, kTK,
                          GIteratorA, SIteratorA, SharedA, RegA, G2SLoaderA,
                          S2RLoaderA, GIteratorB, SIteratorB, SharedB, RegB,
                          G2SLoaderB, S2RLoaderB, GlobalC, SharedC, RegC,
                          R2SStorerC, S2GStorerC>;
    } else if (num_stages == 2) {
        kernel = &ke_gemm_level1_pipeline<
            InType, AccType, kM, kN, kK, kTM, kTN, kTK, kNumStages, SharedA,
            RegA, G2SLoaderA, S2RLoaderA, SharedB, RegB, G2SLoaderB, S2RLoaderB,
            SIteratorA, SIteratorB, GlobalC, SharedC, RegC, R2SStorerC,
            S2GStorerC, PipelineG2SA, PipelineG2SB>;
    } else if (num_stages == 3) {
        kernel = &ke_gemm_level2_pipeline<
            InType, AccType, kM, kN, kK, kTM, kTN, kTK, kNumStages, SharedA,
            RegA, G2SLoaderA, S2RLoaderA, SharedB, RegB, G2SLoaderB, S2RLoaderB,
            SIteratorA, SIteratorB, GlobalC, SharedC, RegC, R2SStorerC,
            S2GStorerC, PipelineG2SA, PipelineG2SB, PipelineS2RA, PipelineS2RB>;
    }

    if (shm_size > 48 * 1024) {
        printf("shm_size: %d\n", shm_size);
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    kernel<<<grid, block, shm_size>>>(dA, dB, dC);

    cudaDeviceSynchronize();
}

void gemm(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C,
          int64_t m, int64_t n, int64_t k, int64_t num_stages) {
    using InType = __half;
    using AccType = float;

    const InType* a_ptr = reinterpret_cast<const InType*>(A.data_ptr());
    const InType* b_ptr = reinterpret_cast<const InType*>(B.data_ptr());
    AccType* c_ptr = reinterpret_cast<AccType*>(C.data_ptr());

    using CtaTileShape = GemmShape<64, 64, 64>;
    constexpr int kRK = 16;
    using WarpLayout = tl::RowMajor<1, 1>;

    if (num_stages == 1 && m == 128 && n == 128 && k == 128) {
        using WholeShape = GemmShape<128, 128, 128>;
        // TODO(KuangjuX): `NUM_STAGES` is not actually used and
        // is fixed to 2 to avoid compilation errors.
        constexpr int NUM_STAGES = 2;
        run_gemm<InType, AccType, WholeShape, CtaTileShape, kRK, WarpLayout,
                 NUM_STAGES>(a_ptr, b_ptr, c_ptr, m, n, k, num_stages);
    } else if (num_stages == 2 && m == 128 && n == 128 && k == 128) {
        using WholeShape = GemmShape<128, 128, 128>;
        constexpr int NUM_STAGES = 2;
        run_gemm<InType, AccType, WholeShape, CtaTileShape, kRK, WarpLayout,
                 NUM_STAGES>(a_ptr, b_ptr, c_ptr, m, n, k, num_stages);
    } else if (num_stages == 2 && m == 256 && n == 256 && k == 256) {
        using WholeShape = GemmShape<256, 256, 256>;
        constexpr int NUM_STAGES = 2;
        run_gemm<InType, AccType, WholeShape, CtaTileShape, kRK, WarpLayout,
                 NUM_STAGES>(a_ptr, b_ptr, c_ptr, m, n, k, num_stages);
    } else if (num_stages == 3 && m == 256 && n == 256 && k == 256) {
        using WholeShape = GemmShape<256, 256, 256>;
        constexpr int NUM_STAGES = 3;
        run_gemm<InType, AccType, WholeShape, CtaTileShape, kRK, WarpLayout,
                 NUM_STAGES>(a_ptr, b_ptr, c_ptr, m, n, k, num_stages);
    } else if (num_stages == 3 && m == 512 && n == 512 && k == 512) {
        using WholeShape = GemmShape<512, 512, 512>;
        constexpr int NUM_STAGES = 3;
        run_gemm<InType, AccType, WholeShape, CtaTileShape, kRK, WarpLayout,
                 NUM_STAGES>(a_ptr, b_ptr, c_ptr, m, n, k, num_stages);
    } else {
        throw std::runtime_error("Unsupported shape");
    }
}

}  // namespace tilefusion::kernels
