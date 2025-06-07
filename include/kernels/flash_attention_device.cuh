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

template <typename InType, typename AccType, typename OutType,
          typename WholeShape, typename CtaTileShape, typename WarpLayout,
          const double kSoftmaxScale_, const bool kIsCausal_>
struct FlashAttentionTraits {
  /// constants
  static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
  static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;
  static_assert(kWarpPerCol == 1, "warps must be arranged as a column vector.");
  static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;

  static constexpr int kSharedAccess = 64;

  using BaseShape = BaseTileShape<InType>;

  static constexpr int kM = dim_size<0, WholeShape>;  // query length
  static constexpr int kN = dim_size<1, WholeShape>;  // key/value length
  static constexpr int kK = dim_size<2, WholeShape>;  // query/key hidden dim
  static constexpr int kP = dim_size<3, WholeShape>;  // value hidden dim

  static constexpr int kTM = dim_size<0, CtaTileShape>;
  static constexpr int kTN = dim_size<1, CtaTileShape>;
  static constexpr int kTK = dim_size<2, CtaTileShape>;
  static constexpr int kTP = dim_size<3, CtaTileShape>;

  static constexpr double kSoftmaxScale = kSoftmaxScale_;
  static constexpr bool kIsCausal = kIsCausal_;

  // query
  using GlobalQ = GlobalTile<InType, tl::RowMajor<kTM, kK>>;
  using GIteratorQ = GTileIterator<GlobalQ, TileShape<kTM, kTK>>;
  using SharedQ =
      SharedTile<InType, tl::RowMajor<kTM, kTK>, true, kSharedAccess>;

  static constexpr int kQMs = kTM / kWarpPerRow / BaseShape::kRows;
  static constexpr int kQKs = kTK / BaseShape::kCols;
  using RegQ = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kQMs, kQKs>>;

  using SharedQLoader = GlobalToSharedLoader<SharedQ, WarpLayout>;
  using RegQLoader =
      SharedToRegLoader<RegQ, WarpLayout, WarpReuse::kRowReuseCont>;

  // key
  using GlobalK = GlobalTile<InType, tl::ColMajor<kK, kN>>;
  using GIteratorK = GTileIterator<GlobalK, TileShape<kTK, kTN>>;
  using SharedK =
      SharedTile<InType, tl::ColMajor<kTK, kTN>, true, kSharedAccess>;

  static constexpr int kKKs = kTK / BaseShape::kRows;
  static constexpr int kKNs = kTN / kWarpPerCol / BaseShape::kCols;
  using RegK = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kKKs, kKNs>>;

  using SharedKLoader = GlobalToSharedLoader<SharedK, WarpLayout>;
  using RegKLoader =
      SharedToRegLoader<RegK, WarpLayout, WarpReuse::kColReuseCont>;

  // value
  using GlobalV = GlobalTile<InType, tl::ColMajor<kN, kTP>>;
  using GIteratorV = GTileIterator<GlobalV, TileShape<kTN, kTP>>;
  using SharedV =
      SharedTile<InType, tl::ColMajor<kTN, kTP>, true, kSharedAccess>;

  static constexpr int kVNs = kTN / BaseShape::kRows;
  static constexpr int kVPs = kTP / kWarpPerCol / BaseShape::kCols;
  using RegV = RegTile<BaseTileColMajor<InType>, tl::ColMajor<kVNs, kVPs>>;

  using SharedVLoader = GlobalToSharedLoader<SharedV, WarpLayout>;
  using RegVLoader =
      SharedToRegLoader<RegV, WarpLayout, WarpReuse::kColReuseCont>;

  // output
  using GlobalO = GlobalTile<OutType, tl::RowMajor<kTM, kTP>>;

  static constexpr int kOMs = kTM / kWarpPerRow / BaseShape::kRows;
  static constexpr int kOPs = kTP / kWarpPerCol / BaseShape::kCols;
  using RegO = RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kOMs, kOPs>>;
  using RegOCast = RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kOMs, kOPs>>;
  using OStorer = RegToGlobalStorer<GlobalO, RegOCast, WarpLayout>;

  // Reg Acc
  static constexpr int kAccMs = kTM / kWarpPerRow / BaseShape::kRows;
  static constexpr int kAccNs = kTN / kWarpPerCol / BaseShape::kCols;
  using RegAcc =
      RegTile<BaseTileRowMajor<AccType>, tl::RowMajor<kAccMs, kAccNs>>;
  using RegAccCast =
      RegTile<BaseTileRowMajor<InType>, tl::RowMajor<kAccMs, kAccNs>>;

  // Convert the accumulator to half
  using ConvertAcc = RegTileConvert<RegAcc, RegAccCast>;
  using ConvertO = RegTileConvert<RegO, RegOCast>;

  using RegVec = RegTile<InType, tl::RowMajor<kAccMs, 2>>;

  using CopyVec = BaseTileCopy<RegVec>;
  using RowMax = MaxReduce<RegAccCast, tl::Layout::kRowMajor>;

  using RowSum = SumReduce<RegAccCast, tl::Layout::kRowMajor>;

  using BroadcastSub = BroadcastSub<RegVec, RegAccCast, tl::Layout::kRowMajor>;
  using BroadcastMul = BroadcastMul<RegVec, RegOCast, tl::Layout::kRowMajor>;
  using BroadcastDiv = BroadcastDiv<RegVec, RegOCast, tl::Layout::kRowMajor>;

  using BlockExp = RegTileExp<RegAccCast>;
  using BlockAdd = RegTileAdd<RegOCast>;

  using VecMax = BaseTileMax<RegVec>;
  using VecAdd = BaseTileAdd<RegVec>;
  using VecSub = BaseTileSub<RegVec>;
  using VecMul = BaseTileMul<RegVec>;
  using VecExp = BaseTileExp<RegVec>;

  using ApplyMask = ApplyMask<RegAcc, WarpLayout, BaseShape, MaskMode::kCausal>;
  using ApplyScoreScale = BroadcastScalarMul<RegAcc>;
};

template <typename InType, typename AccType, typename OutType,
          typename KeTraits>
__device__ __forceinline__ void ke_flash_attention(const InType* dQ,
                                                   const InType* dK,
                                                   const InType* dV,
                                                   OutType* dO) {
  /// constants
  static constexpr int kM = KeTraits::kM;
  static constexpr int kN = KeTraits::kN;
  static constexpr int kK = KeTraits::kK;
  static constexpr int kP = KeTraits::kP;

  static constexpr int kTM = KeTraits::kTM;
  static constexpr int kTN = KeTraits::kTN;
  static constexpr int kTP = KeTraits::kTP;

  static constexpr double kSoftmaxScale = KeTraits::kSoftmaxScale;
  static constexpr bool kIsCausal = KeTraits::kIsCausal;

  // Advance to the global data tile to the current CTA.
  const InType* Q = dQ + blockIdx.z * (kM * kK) + blockIdx.x * (kTM * kK);
  const InType* K = dK + blockIdx.z * (kK * kN);
  const InType* V = dV + blockIdx.z * (kN * kP) + blockIdx.y * (kTP * kN);
  OutType* O = dO + blockIdx.z * (kM * kP) + blockIdx.x * (kTM * kP) +
               (blockIdx.y * kTP);

  /// declare shared memory buffer
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* shm = reinterpret_cast<InType*>(shared_buf);

  InType* sQ_ptr = shm;
  InType* sK_ptr = sQ_ptr + KeTraits::SharedQ::kNumel;
  InType* sV_ptr = sK_ptr + KeTraits::SharedK::kNumel;

  typename KeTraits::GIteratorQ gQs(Q);
  typename KeTraits::SharedQ sQ(sQ_ptr);
  typename KeTraits::RegQ rQ;

  typename KeTraits::SharedQLoader load_sq;
  typename KeTraits::RegQLoader load_rq;

  typename KeTraits::GIteratorK gKs(K);
  typename KeTraits::SharedK sK(sK_ptr);
  typename KeTraits::RegK rK;

  typename KeTraits::SharedKLoader load_sk;
  typename KeTraits::RegKLoader load_rk;

  typename KeTraits::GIteratorV gVs(V);
  typename KeTraits::SharedV sV(sV_ptr);

  typename KeTraits::SharedVLoader load_sv;
  typename KeTraits::RegVLoader load_rv;
  typename KeTraits::RegV rV;

  typename KeTraits::RegO exp_values_f32;

  typename KeTraits::RegOCast rO;
  typename KeTraits::RegOCast exp_values;

  typename KeTraits::RegAcc attn_block_f32;
  typename KeTraits::RegAccCast attn_block;

  typename KeTraits::RegVec prev_norm_vec;
  typename KeTraits::RegVec cur_norm_vec;

  typename KeTraits::RegVec prev_max_vec;
  typename KeTraits::RegVec cur_max_vec;
  typename KeTraits::RegVec new_max_vec;

  typename KeTraits::RegVec prev_sum_vec;
  typename KeTraits::RegVec cur_sum_vec;
  typename KeTraits::RegVec new_sum_vec;

  typename KeTraits::RegVec prev_norm_mul_sum;
  typename KeTraits::RegVec cur_norm_mul_sum;
  typename KeTraits::RegVec prev_sum_mul_norm;

  typename KeTraits::RowMax row_max;
  typename KeTraits::RowSum row_sum;
  typename KeTraits::CopyVec copy_vec;

  typename KeTraits::ConvertAcc cast_acc;  // Convert acc to half precision
  typename KeTraits::ConvertO cast_o;      // Convert half precision to float.

  typename KeTraits::BroadcastSub broadcast_sub;
  typename KeTraits::BroadcastMul broadcast_mul;
  typename KeTraits::BroadcastDiv broadcast_div;

  typename KeTraits::BlockExp block_exp;
  typename KeTraits::BlockAdd block_add;

  typename KeTraits::VecMax vec_max;
  typename KeTraits::VecAdd vec_add;
  typename KeTraits::VecSub vec_sub;
  typename KeTraits::VecMul vec_mul;
  typename KeTraits::VecExp vec_exp;

  typename KeTraits::ApplyMask apply_mask;
  typename KeTraits::ApplyScoreScale apply_score_scale;

  for (int n = 0; n < KeTraits::GIteratorV::sc0; ++n) {
    load_sv(gVs(n), sV);

    for (int k = 0; k < KeTraits::GIteratorQ::sc1; ++k) {
      load_sq(gQs(k), sQ);
      load_sk(gKs(k, n), sK);
      __copy_async();
      __syncthreads();

      load_rq(sQ, rQ);
      load_rk(sK, rK);
      __syncthreads();

      compute::gemm(rQ, rK, attn_block_f32);
    }
    load_rv(sV, rV);
    __syncthreads();

    if (kIsCausal) {
      apply_mask(attn_block_f32, blockIdx.x * kTM, n * kTN, -INFINITY);
    }

    apply_score_scale(attn_block_f32, kSoftmaxScale, attn_block_f32);

    cast_acc(attn_block_f32, attn_block);

    // Compute row max.
    row_max(attn_block, cur_max_vec);

    // Broadcast subtract from `attn_block`.
    broadcast_sub(cur_max_vec, attn_block);

    // Compute exp in `attn_block`.
    block_exp(attn_block, attn_block);

    // Compute `cur_sum_vec` by reduce sum of `attn_block`.
    row_sum(attn_block, cur_sum_vec);

    // Compute new max vector.
    vec_max(cur_max_vec, prev_max_vec, new_max_vec);

    // Renormalization for the previous block.
    vec_sub(prev_max_vec, new_max_vec, prev_norm_vec);
    vec_exp(prev_norm_vec, prev_norm_vec);

    // Renormalization for the current block.
    vec_sub(cur_max_vec, new_max_vec, cur_norm_vec);
    vec_exp(cur_norm_vec, cur_norm_vec);

    // Update normalization factor l(x)
    vec_mul(prev_norm_vec, prev_sum_vec, prev_norm_mul_sum);
    vec_mul(cur_norm_vec, cur_sum_vec, cur_norm_mul_sum);
    vec_add(prev_norm_mul_sum, cur_norm_mul_sum, new_sum_vec);

    // Compute unnormized attention block.
    compute::gemm(attn_block, rV, exp_values_f32);

    cast_o(exp_values_f32, exp_values);

    broadcast_mul(prev_norm_mul_sum, rO);

    broadcast_mul(cur_norm_vec, exp_values);

    block_add(rO, exp_values, rO);

    // Normalize the attention block.
    broadcast_div(new_sum_vec, rO);

    // Update max vector and sum vector.
    copy_vec(new_max_vec, prev_max_vec);
    copy_vec(new_sum_vec, prev_sum_vec);

    // Clear the accumulator.
    attn_block_f32.clear();
    exp_values_f32.clear();
  }

  __syncthreads();
  typename KeTraits::GlobalO gO(O);
  typename KeTraits::OStorer storer_o;
  storer_o(rO, gO);
}

}  // namespace tilefusion::kernels
