// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "cuda_utils.cuh"

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

namespace benchmarks {
namespace cutlass_wrapper {

using namespace cute;

namespace detail {

template <typename GQTensor, typename SQTensor, typename GKTensor,
          typename SKTensor, typename TiledCopy>
class G2SCopyQK {
 public:
  DEVICE G2SCopyQK(GQTensor& gQ, SQTensor& sQ, GKTensor& gK, SKTensor& sK,
                   TiledCopy tiled_copy, int gQ_stride, int sQ_stride,
                   int gK_stride, int sK_stride, int num_stage = 2)
      : gQ(gQ),
        sQ(sQ),
        gK(gK),
        sK(sK),
        gQ_stride(gQ_stride),
        sQ_stride(sQ_stride),
        gK_stride(gK_stride),
        sK_stride(sK_stride),
        cur_iter(0),
        cur_iter_sk(0),
        num_stage(num_stage) {}

  /**
   * @brief Update the pointer of the global K tensor.
   *
   * Since the K matrix is split along both the n and k dimensions, the
   * pointer offset for the K matrix needs to be updated to the next kTN * kK
   * position during the next n dimension iteration.
   *
   * @param gK_slice The stride in N dimension.
   * @param gK_stride The stride in K dimension.
   */
  DEVICE void update_tile_K(int gK_slice, int gK_stride) {
    gK.data() = gK.data() + (-gK_stride) + gK_slice * gK_stride;
  }

  /**
   * @brief Reset the pointer of the global K tensor.
   *
   * The current function is called when `load_q_once` is true, i.e., when
   * kTK == kK. In this case, the pointer of Q needs to be restored to the
   * starting position.
   *
   * @param stride The stride in K dimension.
   */
  DEVICE void reset_tile_Q(int stride) { sQ.data() = sQ.data() + (-stride); }

  /**
   * @brief Preload the K matrix. When `load_q_once` is true, the Q matrix
   * only needs to be loaded once and does not require repeated loading, while
   * the K matrix needs to be updated and loaded.
   */
  DEVICE void prologue_K() {
#pragma unroll
    for (int m = 0; m < size<1>(gK); ++m) {
#pragma unroll
      for (int k = 0; k < size<2>(gK); ++k) {
        cute::copy(tiled_copy, gK(_, m, k), sK(_, m, k));
      }
    }

    cute::cp_async_fence();

    gK.data() = gK.data() + gK_stride;
    sK.data() = sK.data() + sK_stride;

    if ((cur_iter_sk + 1) % num_stage == 0) {
      sK.data() = sK.data() + (-sK_stride * num_stage);
    }

    cur_iter_sk++;
  }

  DEVICE void prologue() {
#pragma unroll
    for (int m = 0; m < size<1>(gQ); ++m) {
#pragma unroll
      for (int k = 0; k < size<2>(gQ); ++k) {
        cute::copy(tiled_copy, gQ(_, m, k), sQ(_, m, k));
      }
    }

#pragma unroll
    for (int m = 0; m < size<1>(gK); ++m) {
#pragma unroll
      for (int k = 0; k < size<2>(gK); ++k) {
        cute::copy(tiled_copy, gK(_, m, k), sK(_, m, k));
      }
    }

    cute::cp_async_fence();

    gQ.data() = gQ.data() + gQ_stride;
    sQ.data() = sQ.data() + sQ_stride;
    gK.data() = gK.data() + gK_stride;
    sK.data() = sK.data() + sK_stride;

    // Circlically read SMEM Buffer
    if ((cur_iter + 1) % num_stage == 0) {
      sQ.data() = sQ.data() + (-sQ_stride * num_stage);
      sK.data() = sK.data() + (-sK_stride * num_stage);
    }

    cur_iter++;
  }

  DEVICE void body() {
#pragma unroll
    for (int m = 0; m < size<1>(gQ); ++m) {
#pragma unroll
      for (int k = 0; k < size<2>(gQ); ++k) {
        cute::copy(tiled_copy, gQ(_, m, k), sQ(_, m, k));
      }
    }

#pragma unroll
    for (int m = 0; m < size<1>(gK); ++m) {
#pragma unroll
      for (int k = 0; k < size<2>(gK); ++k) {
        cute::copy(tiled_copy, gK(_, m, k), sK(_, m, k));
      }
    }

    cute::cp_async_fence();

    gQ.data() = gQ.data() + gQ_stride;
    sQ.data() = sQ.data() + sQ_stride;
    gK.data() = gK.data() + gK_stride;
    sK.data() = sK.data() + sK_stride;

    if ((cur_iter + 1) % num_stage == 0) {
      sQ.data() = sQ.data() + (-sQ_stride * num_stage);
      sK.data() = sK.data() + (-sK_stride * num_stage);
    }

    cur_iter++;
  }

  DEVICE void epilogue() {
#pragma unroll
    for (int m = 0; m < size<1>(gQ); ++m) {
#pragma unroll
      for (int k = 0; k < size<2>(gQ); ++k) {
        cute::copy(tiled_copy, gQ(_, m, k), sQ(_, m, k));
      }
    }

#pragma unroll
    for (int m = 0; m < size<1>(gK); ++m) {
#pragma unroll
      for (int k = 0; k < size<2>(gK); ++k) {
        cute::copy(tiled_copy, gK(_, m, k), sK(_, m, k));
      }
    }

    cute::cp_async_fence();
  }

 private:
  GQTensor& gQ;
  SQTensor& sQ;
  GKTensor& gK;
  SKTensor& sK;
  TiledCopy tiled_copy;
  int gQ_stride;
  int sQ_stride;
  int gK_stride;
  int sK_stride;
  int cur_iter;
  int cur_iter_sk;
  int num_stage;
};

template <typename GVTensor, typename SVTensor, typename TiledCopy>
class G2SCopyV {
 public:
  DEVICE G2SCopyV(GVTensor& gV, SVTensor& sV, TiledCopy tiled_copy,
                  int gV_stride, int sV_stride, int num_stage = 2)
      : gV(gV),
        sV(sV),
        gV_stride(gV_stride),
        sV_stride(sV_stride),
        cur_iter(0),
        num_stage(num_stage) {}

  DEVICE void prologue() {
#pragma unroll
    for (int m = 0; m < size<1>(gV); ++m) {
#pragma unroll
      for (int k = 0; k < size<2>(gV); ++k) {
        cute::copy(tiled_copy, gV(_, m, k), sV(_, m, k));
      }
    }

    cute::cp_async_fence();
    gV.data() = gV.data() + gV_stride;
    sV.data() = sV.data() + sV_stride;

    if ((cur_iter + 1) % num_stage == 0) {
      sV.data() = sV.data() + (-sV_stride * num_stage);
    }

    cur_iter++;
  }

  DEVICE void body() {
#pragma unroll
    for (int m = 0; m < size<1>(gV); ++m) {
#pragma unroll
      for (int k = 0; k < size<2>(gV); ++k) {
        cute::copy(tiled_copy, gV(_, m, k), sV(_, m, k));
      }
    }

    cute::cp_async_fence();

    gV.data() = gV.data() + gV_stride;
    sV.data() = sV.data() + sV_stride;

    if ((cur_iter + 1) % num_stage == 0) {
      sV.data() = sV.data() + (-sV_stride * num_stage);
    }

    cur_iter++;
  }

  DEVICE void epilogue() {
#pragma unroll
    for (int m = 0; m < size<1>(gV); ++m) {
#pragma unroll
      for (int k = 0; k < size<2>(gV); ++k) {
        cute::copy(tiled_copy, gV(_, m, k), sV(_, m, k));
      }
    }
    cute::cp_async_fence();
  }

 private:
  GVTensor& gV;
  SVTensor& sV;
  TiledCopy tiled_copy;
  int gV_stride;
  int sV_stride;
  int cur_iter;
  int num_stage;
};

template <typename SQTensor, typename RQMmaView, typename RQCopyView,
          typename SKTensor, typename RKMmaView, typename RKCopyView,
          typename RAccTensor, typename TiledCopyQ, typename TiledCopyK,
          typename TiledMma>
class S2RPipelineQK {
 public:
  DEVICE S2RPipelineQK(SQTensor& sQ, RQMmaView& rQ_mma_view,
                       RQCopyView& rQ_copy_view, SKTensor& sK,
                       RKMmaView& rK_mma_view, RKCopyView& rK_copy_view,
                       RAccTensor& acc, TiledCopyQ copy_q, TiledCopyK copy_k,
                       TiledMma tiled_mma, int sQ_stride, int sK_stride,
                       int num_stage = 2)
      : sQ(sQ),
        rQ_mma_view(rQ_mma_view),
        rQ_copy_view(rQ_copy_view),
        sK(sK),
        rK_mma_view(rK_mma_view),
        rK_copy_view(rK_copy_view),
        acc(acc),
        copy_q(copy_q),
        copy_k(copy_k),
        tiled_mma(tiled_mma),
        sQ_stride(sQ_stride),
        sK_stride(sK_stride),
        num_stage(num_stage),
        cur_iter(0),
        cur_iter_sq(0) {}

  DEVICE void prologue() {
    cur_iter = 0;
    cute::copy(copy_q, sQ(_, _, _0{}), rQ_copy_view(_, _, _0{}));
    cute::copy(copy_k, sK(_, _, _0{}), rK_copy_view(_, _, _0{}));

#pragma unroll
    for (int i = 0; i < size<2>(rK_mma_view); ++i) {
      if (i < size<2>(rK_mma_view) - 1) {
        cute::copy(copy_q, sQ(_, _, _0{}), rQ_copy_view(_, _, _0{}));
        cute::copy(copy_k, sK(_, _, _0{}), rK_copy_view(_, _, _0{}));
      }
      cute::gemm(tiled_mma, rQ_mma_view(_, _, i), rK_mma_view(_, _, i), acc);
    }
    sQ.data() = sQ.data() + sQ_stride;
    sK.data() = sK.data() + sK_stride;

    cur_iter++;
  }

  DEVICE void body() {
    cute::copy(copy_q, sQ(_, _, _0{}), rQ_copy_view(_, _, _0{}));
    cute::copy(copy_k, sK(_, _, _0{}), rK_copy_view(_, _, _0{}));

#pragma unroll
    for (int i = 0; i < size<2>(rK_mma_view); ++i) {
      if (i < size<2>(rK_mma_view) - 1) {
        cute::copy(copy_q, sQ(_, _, i + 1), rQ_copy_view(_, _, i + 1));
        cute::copy(copy_k, sK(_, _, i + 1), rK_copy_view(_, _, i + 1));
      }
      cute::gemm(tiled_mma, rQ_mma_view(_, _, i), rK_mma_view(_, _, i), acc);
    }
    sQ.data() = sQ.data() + sQ_stride;
    sK.data() = sK.data() + sK_stride;

    if ((cur_iter + 1) % num_stage == 0) {
      sK.data() = sK.data() + (-sK_stride * num_stage);
    }

    cur_iter++;
    cur_iter_sq++;
  }

  DEVICE void epilogue() {
    cute::copy(copy_q, sQ(_, _, _0{}), rQ_copy_view(_, _, _0{}));
    cute::copy(copy_k, sK(_, _, _0{}), rK_copy_view(_, _, _0{}));

#pragma unroll
    for (int i = 0; i < size<2>(rK_mma_view); ++i) {
      if (i < size<2>(rK_mma_view) - 1) {
        cute::copy(copy_q, sQ(_, _, i + 1), rQ_copy_view(_, _, i + 1));
        cute::copy(copy_k, sK(_, _, i + 1), rK_copy_view(_, _, i + 1));
      }
      cute::gemm(tiled_mma, rQ_mma_view(_, _, i), rK_mma_view(_, _, i), acc);
    }

    sQ.data() = sQ.data() + (-sQ_stride * cur_iter_sq);
    sK.data() = sK.data() + sK_stride;

    if ((cur_iter + 1) % num_stage == 0) {
      sK.data() = sK.data() + (-sK_stride * num_stage);
    }

    cur_iter++;
    cur_iter_sq = 0;
  }

 private:
  SQTensor& sQ;
  RQMmaView& rQ_mma_view;
  RQCopyView& rQ_copy_view;
  SKTensor& sK;
  RKMmaView& rK_mma_view;
  RKCopyView& rK_copy_view;
  RAccTensor& acc;
  TiledCopyQ copy_q;
  TiledCopyK copy_k;
  TiledMma tiled_mma;
  int sQ_stride;
  int sK_stride;
  int num_stage;
  int cur_iter;
  int cur_iter_sq;
};

template <typename SVTensor, typename RVMmaView, typename RVCopyView,
          typename RegAcc, typename TiledCopy, typename TiledMma>
class S2RPipelineV {
 public:
  DEVICE S2RPipelineV(SVTensor& sV, RVMmaView& rV_mma_view,
                      RVCopyView& rV_copy_view, RegAcc& acc,
                      TiledCopy tiled_copy, TiledMma tiled_mma, int sV_stride,
                      int num_stage = 2)
      : sV(sV),
        rV_mma_view(rV_mma_view),
        rV_copy_view(rV_copy_view),
        acc(acc),
        tiled_copy(tiled_copy),
        sV_stride(sV_stride),
        num_stage(num_stage),
        cur_iter(0),
        cur_iter_sv(0) {}

  template <typename RegValue>
  DEVICE void prologue(RegValue& value) {
    cur_iter = 0;
    cute::copy(tiled_copy, sV(_, _, _0{}), rV_copy_view(_, _, _0{}));
#pragma unroll
    for (int i = 0; i < size<2>(rV_mma_view); ++i) {
      if (i < size<2>(rV_mma_view) - 1) {
        cute::copy(tiled_copy, sV(_, _, i + 1), rV_copy_view(_, _, i + 1));
      }
      // TODO(KuangjuX):  Understand this code. Why do we need to use
      // `value(_, _, cur_iter * size<2>(rV_mma_view) + i)`?
      cute::gemm(tiled_mma, value(_, _, cur_iter * size<2>(rV_mma_view) + i),
                 rV_mma_view(_, _, i), acc);
    }

    sV.data() = sV.data() + sV_stride;
    cur_iter++;
  }

  template <typename RegValue>
  DEVICE void body(RegValue& value) {
    cute::copy(tiled_copy, sV(_, _, _0{}), rV_copy_view(_, _, _0{}));

#pragma unroll
    for (int i = 0; i < size<2>(rV_mma_view); ++i) {
      if (i < size<2>(rV_mma_view) - 1) {
        cute::copy(tiled_copy, sV(_, _, i + 1), rV_copy_view(_, _, i + 1));
      }
      cute::gemm(tiled_mma, value(_, _, cur_iter * size<2>(rV_mma_view) + i),
                 rV_mma_view(_, _, i), acc);
    }

    sV.data() = sV.data() + sV_stride;
    if ((cur_iter + 1) % num_stage == 0) {
      sV.data() = sV.data() + (-sV_stride * num_stage);
    }

    cur_iter++;
    cur_iter_sv++;
  }

  template <typename RegValue>
  DEVICE void epilogue(RegValue& value) {
    cute::copy(tiled_copy, sV(_, _, _0{}), rV_copy_view(_, _, _0{}));

#pragma unroll
    for (int i = 0; i < size<2>(rV_mma_view); ++i) {
      if (i < size<2>(rV_mma_view) - 1) {
        cute::copy(tiled_copy, sV(_, _, i + 1), rV_copy_view(_, _, i + 1));
      }
      cute::gemm(tiled_mma, value(_, _, cur_iter * size<2>(rV_mma_view) + i),
                 rV_mma_view(_, _, i), acc);
    }

    sV.data() = sV.data() + (-sV_stride * cur_iter_sv);

    if ((cur_iter + 1) % num_stage == 0) {
      sV.data() = sV.data() + (-sV_stride * num_stage);
    }

    cur_iter++;
    cur_iter_sv = 0;
  }

 private:
  SVTensor& sV;
  RVMmaView& rV_mma_view;
  RVCopyView& rV_copy_view;
  RegAcc& acc;
  TiledCopy tiled_copy;
  TiledMma tiled_mma;
  int sV_stride;
  int num_stage;
  int cur_iter;
  int cur_iter_sv;
};

}  // namespace detail

template <typename Element, typename GlobalQLayout, typename SharedQLayout,
          typename GlobalKLayout, typename SharedKLayout, typename TiledCopy>
inline __device__ auto make_g2s_qk(const Element* gQ_ptr, Element* sQ_ptr,
                                   const Element* gK_ptr, Element* sK_ptr,
                                   int gQ_stride, int gK_stride) {
  int tid = threadIdx.x;

  auto gQ = make_tensor(make_gmem_ptr(gQ_ptr), GlobalQLayout{});
  auto sQ = make_tensor(make_smem_ptr(sQ_ptr), SharedQLayout{});

  auto gK = make_tensor(make_gmem_ptr(gK_ptr), GlobalKLayout{});
  auto sK = make_tensor(make_smem_ptr(sK_ptr), SharedKLayout{});

  TiledCopy tiled_copy;

  auto loader = tiled_copy.get_thread_slice(tid);

  auto gQs = loader.partition_S(gQ);
  auto gKs = loader.partition_S(gK);
  auto sQs = loader.partition_D(sQ);
  auto sKs = loader.partition_D(sK);

  int sQ_stride = size(sQ);
  int sK_stride = size(sK);

#ifdef DEBUG
  if (thread0()) {
    printf("gQ_stride: %d, sQ_stride: %d, gK_stride: %d, sK_stride: %d\n",
           gQ_stride, sQ_stride, gK_stride, sK_stride);
  }
#endif

  detail::G2SCopyQK copy_qk(gQs, sQs, gKs, sKs, tiled_copy, gQ_stride,
                            sQ_stride, gK_stride, sK_stride);

  return copy_qk;
}

template <typename Element, typename GlobalVLayout, typename SharedVLayout,
          typename TiledCopy>
DEVICE auto make_g2s_v(const Element* gV_ptr, Element* sV_ptr, int gV_stride) {
  int tid = threadIdx.x;

  auto gV = make_tensor(make_gmem_ptr(gV_ptr), GlobalVLayout{});
  auto sV = make_tensor(make_smem_ptr(sV_ptr), SharedVLayout{});

  TiledCopy tiled_copy;

  auto loader = tiled_copy.get_thread_slice(tid);

  auto gVs = loader.partition_S(gV);
  auto sVs = loader.partition_D(sV);

  int sV_stride = size(sV);

#ifdef DEBUG
  if (thread0()) {
    printf("gV_stride: %d, sV_stride: %d\n", gV_stride, sV_stride);
  }
#endif

  detail::G2SCopyV copy_v(gVs, sVs, tiled_copy, gV_stride, sV_stride);

  return copy_v;
}

template <typename Element, typename SQLayout, typename SKLayout,
          typename RegAcc, typename SmemCopyAtom, typename TiledMma>
DEVICE auto make_s2r_qk(const Element* sQ_ptr, const Element* sK_ptr,
                        SQLayout sQ_layout, SKLayout sK_layout, RegAcc acc,
                        SmemCopyAtom copy_atom = SmemCopyAtom{},
                        TiledMma tiled_mma = TiledMma{}) {
  int tid = threadIdx.x;

  auto sQ_ = make_tensor(make_smem_ptr(sQ_ptr), sQ_layout);
  auto sK_ = make_tensor(make_smem_ptr(sK_ptr), sK_layout);

  auto thr_mma = tiled_mma.get_thread_slice(tid);

  auto s2r_copy_q = make_tiled_copy_A(copy_atom, tiled_mma);
  auto s2r_copy_k = make_tiled_copy_B(copy_atom, tiled_mma);
  auto s2r_thr_copy_q = s2r_copy_q.get_thread_slice(tid);
  auto s2r_thr_copy_k = s2r_copy_k.get_thread_slice(tid);

#ifdef DEBUG
  if (thread0()) {
    printf("sQ_Layout: ");
    print(sQ_layout), print('\n');
    printf("s2r_copy_q: ");
    print(s2r_copy_q), print('\n');
  }
#endif

  auto sQ = s2r_thr_copy_q.partition_S(sQ_);
  auto sK = s2r_thr_copy_k.partition_S(sK_);

  // Thread partition for mma.
  auto rQ_mma = thr_mma.partition_fragment_A(sQ_);
  auto rK_mma = thr_mma.partition_fragment_B(sK_);

  // Thread partition for shared to register copy.
  auto rQ_copy = s2r_thr_copy_q.retile_D(rQ_mma);
  auto rK_copy = s2r_thr_copy_k.retile_D(rK_mma);

#ifdef DEBUG
  if (thread0()) {
    printf("sQ_: ");
    print(sQ_), print('\n');
    printf("sQ: ");
    print(sQ), print('\n');
    printf("rQ_copy: ");
    print(rQ_copy), print('\n');
    printf("rQ_mma: ");
    print(rQ_mma), print('\n');
  }
#endif

  int sQ_stride = size(sQ_);
  int sK_stride = size(sK_);

  detail::S2RPipelineQK s2r_pipeline_qk(sQ, rQ_mma, rQ_copy, sK, rK_mma,
                                        rK_copy, acc, s2r_copy_q, s2r_copy_k,
                                        tiled_mma, sQ_stride, sK_stride);

  return s2r_pipeline_qk;
}

template <typename Element, typename SVLayout, typename RegAcc,
          typename SmemCopyAtom, typename TiledMma>
DEVICE auto make_s2r_v(const Element* sV_ptr, SVLayout sV_layout, RegAcc& acc,
                       SmemCopyAtom copy_atom, TiledMma tiled_mma) {
  int tid = threadIdx.x;

  auto sV_ = make_tensor(make_smem_ptr(sV_ptr), sV_layout);

  auto thr_mma = tiled_mma.get_thread_slice(tid);

  auto s2r_copy_v = make_tiled_copy_B(copy_atom, tiled_mma);
  auto s2r_thr_copy_v = s2r_copy_v.get_thread_slice(tid);

  auto sV = s2r_thr_copy_v.partition_S(sV_);

  auto rV_mma = thr_mma.partition_fragment_B(sV_);
  auto rV_copy = s2r_thr_copy_v.retile_D(rV_mma);

  int sV_stride = size(sV_);

  detail::S2RPipelineV s2r_pipeline_v(sV, rV_mma, rV_copy, acc, s2r_copy_v,
                                      tiled_mma, sV_stride);

  return s2r_pipeline_v;
}

template <typename Element, typename SOLayout, typename RegO,
          typename SmemCopyAtom, typename TiledMma>
DEVICE auto store_r2s_o(Element* sO_ptr, SOLayout sO_layout, RegO& o,
                        SmemCopyAtom copy_atom, TiledMma tiled_mma) {
  auto sO = make_tensor(make_smem_ptr(sO_ptr), sO_layout);

  auto r2s_copy_o = make_tiled_copy_C(copy_atom, tiled_mma);
  auto r2s_thr_copy_o = r2s_copy_o.get_thread_slice(threadIdx.x);

  auto src = r2s_thr_copy_o.retile_S(o);
  auto dst = r2s_thr_copy_o.partition_D(sO);

  cute::copy(r2s_copy_o, src, dst);
}

template <typename Element, typename GOLayout, typename SOLayout,
          typename TiledCopy>
DEVICE auto store_s2g_o(Element* gO_ptr, const Element* sO_ptr,
                        GOLayout gO_layout, SOLayout sO_layout,
                        TiledCopy tiled_copy) {
  auto gO = make_tensor(make_gmem_ptr(gO_ptr), gO_layout);
  auto sO = make_tensor(make_smem_ptr(sO_ptr), sO_layout);

  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  auto gO_partition = thr_copy.partition_D(gO);
  auto sO_partition = thr_copy.partition_S(sO);

#pragma unroll
  for (int m = 0; m < size<1>(gO_partition); ++m) {
#pragma unroll
    for (int n = 0; n < size<2>(gO_partition); ++n) {
      cute::copy(tiled_copy, sO_partition(_, m, n), gO_partition(_, m, n));
    }
  }
}

}  // namespace cutlass_wrapper
}  // namespace benchmarks
