// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cell/compute/mod.hpp"
#include "cell/mod.hpp"
#include "types/mod.hpp"
#include "util.hpp"

using namespace tilefusion;
using namespace tilefusion::cell;
using namespace tilefusion::cell::copy;
namespace tl = tile_layout;

template <const int kM, const int kN, const int kK, const int kP>
using FlashDecodingShape = TileShape<kM, kN, kK, kP>;

template <typename InType, typename AccType, typename WholeShape,
          typename CtaTileShape, const int kSharedAccess>
struct FlashDecodingTraits {
    using BaseShape = traits::BaseTileShape<InType>;

    using WarpLayout = tl::RowMajor<4, 1>;

    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;
    static_assert(kWarpPerCol == 1, "WarpPerCol must be 1");

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;
    static constexpr int kP = dim_size<3, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;
    static constexpr int kTP = dim_size<3, CtaTileShape>;

    // operand A
    using GlobalA = GlobalTile<InType, tl::RowMajor<kTM, kK>>;
    // chunk the K dimension to fit into shared memory
    using GIteratorA = GTileIterator<GlobalA, TileShape<kTM, kTK>>;

    using SharedA =
        SharedTile<InType, tl::RowMajor<kTM, kTK>, true, kSharedAccess>;

    static constexpr int kAMs = kTM / kWarpPerRow / BaseShape::kRows;
    static constexpr int kAKs = kTK / BaseShape::kCols;
    using RegA = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAMs, kAKs>>;

    using SharedALoader = GlobalToSharedLoader<SharedA, WarpLayout>;
    using RegALoader =
        SharedToRegLoader<RegA, WarpLayout, WarpReuse::kRowReuseCont>;

    // operand B
    using GlobalB = GlobalTile<InType, tl::ColMajor<kK, kN>>;
    using GIteratorB = GTileIterator<GlobalB, TileShape<kTK, kTN>>;
    using SharedB =
        SharedTile<InType, tl::ColMajor<kTK, kTN>, true, kSharedAccess>;

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
    using SharedC =
        SharedTile<InType, tl::ColMajor<kTN, kTP>, true, kSharedAccess>;

    static constexpr int kCNs = kTN / BaseShape::kRows;
    static constexpr int kCPs = kTP / kWarpPerCol / BaseShape::kCols;
    using RegC = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kCNs, kCPs>>;

    using SharedCLoader = GlobalToSharedLoader<SharedC, WarpLayout>;
    using RegCLoader =
        SharedToRegLoader<RegC, WarpLayout, WarpReuse::kColReuseCont>;

    // output D
    using GlobalD = GlobalTile<InType, tl::RowMajor<kTM, kTP>>;

    static constexpr int kDMs = kTM / kWarpPerRow / BaseShape::kRows;
    static constexpr int kDPs = kTP / kWarpPerCol / BaseShape::kCols;
    using RegD = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kDMs, kDPs>>;
    using RegDCast =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kDMs, kDPs>>;
    using DStorer = copy::RegToGlobalStorer<GlobalD, RegDCast, WarpLayout>;

    static constexpr int kAccMs = kTM / kWarpPerRow / BaseShape::kRows;
    static constexpr int kAccNs = kTN / kWarpPerCol / BaseShape::kCols;

    // Reg Acc
    using RegAcc =
        RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kAccMs, kAccNs>>;
    using RegAccCast =
        RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAccMs, kAccNs>>;

    // Convert the accumulator to half
    using ConvertHalf = compute::RegTileConvert<RegAcc, RegAccCast>;
    using ConvertO = compute::RegTileConvert<RegD, RegDCast>;

    using RegVec = RegTile<InType, tl::RowMajor<kAccMs, 2>>;

    using CopyVec = copy::BaseTileCopy<RegVec>;
    using RowMax = compute::MaxReduce<RegAccCast, tl::Layout::kRowMajor>;

    using RowSum = compute::SumReduce<RegAccCast, tl::Layout::kRowMajor>;

    using BroadcastSub =
        compute::BroadcastSub<RegVec, RegAccCast, tl::Layout::kRowMajor>;
    using BroadcastMul =
        compute::BroadcastMul<RegVec, RegDCast, tl::Layout::kRowMajor>;
    using BroadcastDiv =
        compute::BroadcastDiv<RegVec, RegDCast, tl::Layout::kRowMajor>;

    using BlockExp = compute::RegTileExp<RegAccCast>;
    using BlockAdd = compute::RegTileAdd<RegDCast>;

    using VecMax = compute::BaseTileMax<RegVec>;
    using VecAdd = compute::BaseTileAdd<RegVec>;
    using VecSub = compute::BaseTileSub<RegVec>;
    using VecMul = compute::BaseTileMul<RegVec>;
    using VecExp = compute::BaseTileExp<RegVec>;
    using VecLog = compute::BaseTileLog<RegVec>;
};

template <typename InType,
          typename AccType,                                      //
          typename OutType,                                      //
          typename GIteratorQ, typename SharedQ, typename RegQ,  //
          typename SharedQLoader, typename RegQLoader,           //
          typename GIteratorK, typename SharedK, typename RegK,  //
          typename SharedKLoader, typename RegKLoader,           //
          typename GIteratorV, typename SharedV, typename RegV,  //
          typename SharedVLoader, typename RegVLoader,           //
          typename RegAcc, typename RegAccCast, typename GlobalO, typename RegO,
          typename RegOCast, typename OStorer, typename ConvertAcc,
          typename ConvertO, typename RegVec, typename CopyVec, typename RowMax,
          typename RowSum, typename BroadcastSub, typename BroadcastMul,
          typename BroadcastDiv, typename BlockExp, typename BlockAdd,
          typename VecMax, typename VecAdd, typename VecSub, typename VecMul,
          typename VecExp, typename VecLog>
__global__ void ke_flash_decoding_split_kv_fwd(const InType* dQ,
                                               const InType* dK,
                                               const InType* dV, OutType* dO,
                                               OutType* dLSE, int kM, int kN,
                                               int kK, int kP, int kTM, int kTN,
                                               int kTK, int kTP, int kChunkN) {
    // Advance to the global data tile to the current CTA.
    const InType* Q = dQ + blockIdx.x * (kTM * kK);
    const InType* K = dK + blockIdx.z * kChunkN * kK;
    const InType* gV_ptr = dV + blockIdx.y * (kTP * kN) + blockIdx.z * kChunkN;

    OutType* gO_ptr = dO + blockIdx.x * (kTM * kP) + (blockIdx.y * kTP);
    // lse is a (M, 1) vector in a split dimension.
    OutType* gLSE_ptr = dLSE + blockIdx.z * M + blockIdx.x * kTM;

    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<InType*>(shared_buf);

    InType* sQ_ptr = shm;
    InType* sK_ptr = shm + SharedQ::kNumel;
    InType* sV_ptr = shm + SharedQ::kNumel + SharedK::kNumel;

    GIteratorQ gQs(Q);
    SharedQ sQ(sQ_ptr);
    RegQ rQ;

    SharedQLoader load_sq;
    RegQLoader load_rq;

    GIteratorK gKs(K);
    SharedK sK(sK_ptr);
    RegK rK;

    SharedKLoader load_sk;
    RegKLoader load_rk;

    GIteratorV gVs(gV_ptr);
    SharedV sV(sV_ptr);

    SharedVLoader load_sv;
    RegVLoader load_rv;
    RegV rV;

    RegOCast rO;

    RegAcc qk_float;
    RegAccCast qk;

    // The LogSumExp(LSE) is a smooth maximum,
    // LSE(x1, x2, ..., xn) = log(exp(x1) + exp(x2) + ... + exp(xn))
    // = c + log(exp(x1 - c) + exp(x2 - c) + ... + exp(xn - c))
    // c = max(x1, x2, ..., xn)
    RegVec lse_vec;
    RegVec l_ij;
    RegVec acc_o_scale;
    RegVec l_i_new;
    RegVec temp_vec;
    RegVec o_scale;

    RegO pv_float;
    RegOCast pv;

    RegVec prev_max_vec;
    RegVec cur_max_vec;

    RowMax row_max;
    RowSum row_sum;
    CopyVec copy_vec;

    ConvertAcc cast_acc;  // Convert acc to half precision
    ConvertO cast_o;      // Convert half precision to float.

    BroadcastSub broadcast_sub;
    BroadcastMul broadcast_mul;
    BroadcastDiv broadcast_div;

    BlockExp block_exp;
    BlockAdd block_add;

    VecMax vec_max;
    VecAdd vec_add;
    VecSub vec_sub;
    VecMul vec_mul;
    VecExp vec_exp;
    VecLog vec_log;
    for (int n = 0; n < GIteratorV::sc0; ++n) {
        load_sv(gVs(n), sV);

        for (int k = 0; k < GIteratorQ::sc1; ++k) {
            load_sq(gQs(k), sQ);
            load_sk(gKs(k, n), sK);
            __copy_async();
            __syncthreads();

            load_rq(sQ, rQ);
            load_rk(sK, rK);
            __syncthreads();

            compute::gemm(rQ, rK, qk_float);
        }
        load_rv(sV, rV);
        __syncthreads();

        cast_acc(qk_float, qk);

        // Compute current row max vector.
        row_max(qk, cur_max_vec);

        // Compute cur_max_vec with lse.
        vec_max(cur_max_vec, lse_vec, cur_max_vec);

        // p = torch.exp(qk - cur_max_vec)
        broadcast_sub(cur_max_vec, qk);
        block_exp(qk, qk);

        // l_ij = torch.sum(p, dim=-1, keepdim=True)
        row_sum(qk, l_ij);

        // Renormalize o
        // acc_o_scale = torch.exp(prev_max_vec - cur_max_vec)
        // output = acc_o_scale * output + p @ v
        vec_sub(prev_max_vec, cur_max_vec, acc_o_scale);
        vec_exp(acc_o_scale, acc_o_scale);

        compute::gemm(qk, rV, pv_float);
        cast_o(pv_float, pv);
        broadcast_mul(acc_o_scale, rO);
        block_add(rO, pv, rO);

        // Update statistics
        // prev_max_vec = cur_max_vec
        // l_i_new = torch.exp(lse - cur_maxes) + l_ij
        // lse = cur_max_vec + torch.log(l_i_new)
        copy_vec(prev_max_vec, cur_max_vec);
        vec_sub(lse_vec, cur_max_vec, temp_vec);
        vec_exp(temp_vec, l_i_new);
        vec_add(l_i_new, l_ij, l_i_new);
        vec_log(l_i_new, temp_vec);
        vec_add(cur_max_vec, temp_vec, lse_vec);

        // Clear the accumulator.
        qk_float.clear();
        pv_float.clear();
    }
    __syncthreads();

    // o_scale = torch.exp(prev_max_vec - lse)
    // output = o_scale * output
    vec_sub(prev_max_vec, lse_vec, o_scale);
    vec_exp(o_scale, o_scale);
    broadcast_mul(o_scale, rO);

    GlobalO gO(gO_ptr);
    OStorer storer_o;  // Store O tile from register to global.
    storer_o(rO, gO);
}

template <typename Element, typename GlobalLse, typename GlobalO, typename RegO,
          typename LseVec, typename VecMax, typename VecExp, typename ReduceSum,
          typename BroadcastSub, typename AllReduceMax, typename AllReduceSum,
          typename StoreLse, typename LoadO, typename StoreO>
__global__ void ke_flash_decoding_split_kv_fwd_combine(
    Element* gLse_ptr, Element* gO_ptr, const int split_kv_nums) {
    extern __shared__ Element smem[];
    GlobalO gO(gO_ptr);
    RegO rO;

    VecMax vec_max;
    BroadcastSub broadcast_sub;
    VecExp vec_exp;
    ReduceSum reduce_sum;

    AllReduceMax all_reduce_max;

    StoreLse store_lse;

    LoadO load_o;
    StoreO store_o;

    // A naive implementation of the combine kernel.
    GlobalLse gLSE(gLse_ptr);
    LseVec lse;
    LseVec lse_sub_max;
    LseVec scale;

    Element lse_max;
    Element lse_sum;
    Element lse_log_sum;

    // find the max logsumexp in the current warp.
    vec_max(lse, lse_max);
    all_reduce_max(lse_max, lse_max);

    // rescale the logsumexp.
    broadcast_sub(lse, lse_max, lse_sub_max);
    vec_exp(lse_sub_max, lse_sub_max);
    reduce_sum(lse_sub_max, lse_sum);
    all_reduce_sum(lse_sum, lse_sum);

    lse_log_sum = logf(lse_sum) + lse_max;

    // Store the scales exp(lse - lse_logsum) in shared memory.
    broadcast_sub(lse, lse_log_sum, scale);
    vec_exp(scale, scale);
    store_lse(scale, smem);

    __syncthreads();

    // Load O into register.
    load_o(gO, rO);

    // TODO: Scale the output.

    // Store the output into global memory.
    store_o(rO, gO);
}

template <typename InType, typename AccType, typename OutType,
          typename WholeShape, typename CtaTileShape, const int kChunkN,
          const int kSharedAccess>
void run_flash_decoding_fwd(const InType* dQ, const InType* dK,
                            const InType* dV, OutType* dO) {
    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;
    static constexpr int kP = dim_size<3, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;
    static constexpr int kTP = dim_size<3, CtaTileShape>;

    int split_kv_nums = CeilDiv<kN, kChunkN>;
    int block_x = CeilDiv<kM, kTM>;
    int block_y = CeilDiv<kP, kTP>;
    int block_z = split_kv_nums;

    using Config = FlashDecodingTraits<InType, AccType, WholeShape,
                                       CtaTileShape, kSharedAccess>;

    using RegQ = typename Config::RegA;
    using RegK = typename Config::RegB;
    using RegV = typename Config::RegC;
    using RegO = typename Config::RegD;
    using RegOCast = typename Config::RegDCast;
    using RegAcc = typename Config::RegAcc;
    using RegAccCast = typename Config::RegAccCast;

    using GIteratorQ = typename Config::GIteratorA;
    using SharedQ = typename Config::SharedA;
    using SharedQLoader = typename Config::SharedALoader;
    using RegQLoader = typename Config::RegALoader;

    using GIteratorK = typename Config::GIteratorB;
    using SharedK = typename Config::SharedB;
    using SharedKLoader = typename Config::SharedBLoader;
    using RegKLoader = typename Config::RegBLoader;

    using GIteratorV = typename Config::GIteratorC;
    using SharedV = typename Config::SharedC;
    using SharedVLoader = typename Config::SharedCLoader;
    using RegVLoader = typename Config::RegCLoader;

    using OStorer = typename Config::DStorer;
    using GlobalO = typename Config::GlobalD;

    using ConvertAcc = typename Config::ConvertHalf;
    using ConvertO = typename Config::ConvertO;

    using RegVec = typename Config::RegVec;

    using CopyVec = typename Config::CopyVec;
    using RowMax = typename Config::RowMax;
    using RowSum = typename Config::RowSum;

    using BroadcastSub = typename Config::BroadcastSub;
    using BroadcastMul = typename Config::BroadcastMul;
    using BroadcastDiv = typename Config::BroadcastDiv;

    using BlockExp = typename Config::BlockExp;
    using BlockAdd = typename Config::BlockAdd;

    using VecMax = typename Config::VecMax;
    using VecAdd = typename Config::VecAdd;
    using VecSub = typename Config::VecSub;
    using VecMul = typename Config::VecMul;
    using VecExp = typename Config::VecExp;
    using VecLog = typename Config::VecLog;

    dim3 grid(block_x, block_y, block_z);
    dim3 block(Config::kThreads, 1, 1);

    int shm_input = (kTM * kTK + kTK * kTN + kTN * kTP);
    int shm_output = kTM * kTP;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(InType)
                                          : shm_input * sizeof(InType);

    auto flash_decoding_split_kv_fwd = &ke_flash_decoding_split_kv_fwd<
        InType, AccType, OutType, GIteratorQ, SharedQ, RegQ, SharedQLoader,
        RegQLoader, GIteratorK, SharedK, RegK, SharedKLoader, RegKLoader,
        GIteratorV, SharedV, RegV, SharedVLoader, RegVLoader, RegAcc,
        RegAccCast, GlobalO, RegO, RegOCast, OStorer, ConvertAcc, ConvertO,
        RegVec, CopyVec, RowMax, RowSum, BroadcastSub, BroadcastMul,
        BroadcastDiv, BlockExp, BlockAdd, VecMax, VecAdd, VecSub, VecMul,
        VecExp, VecLog>;

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(flash_decoding_split_kv_fwd,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shm_size);
    }

    = thrust::device_vector<OutType> dLSE(split_kv_nums * M);
    thrust::fill(dLSE.begin(), dLSE.end(), 0.);

    OutType* lse_ptr = thrust::raw_pointer_cast(dLSE.data());

    flash_decoding_split_kv_fwd<<<grid, block, shm_size, 0>>>(
        dQ, dK, dV, dO, lse_ptr, kM, kN, kK, kP, kTM, kTN, kTK, kTP, kChunkN);

    if (split_kv_nums > 1) {
        // TODO: Implement the combine kernel.

        // In flashdecoding,
        // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is
        // divisible by 128, kBlockM = 4. If headdim is divisible by 64, then we
        // set kBlockM = 8, etc.

        dim3 grid(1, 1, 1);
        dim3 block(Config::kThreads, 1, 1);

        // ke_flash_decoding_split_kv_fwd_combine<<<grid, block, 0, 0>>>(dLSE,
        // dO);
    }
}
