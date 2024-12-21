// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>

namespace benchmarks {
namespace cutlass_wrapper {

using namespace cute;

template <typename Element, typename Layout, typename TiledMma>
inline __device__ void make_sQ(const Element* data, const Layout& layout,
                               const TiledMma& tiled_mma) {
    int tid = threadIdx.x;
    Tensor sQ = make_tensor(make_smem_ptr, layout);
}

template <class G2STiledCopy, class GTensor, class STensor>
class G2SCopyTile {
  public:
    G2SCopyTile(G2STiledCopy& copy_v, GTensor& gV, STensor& sV, int gV_stride,
                int sV_stride, int num_stage = 2)
        : copy_v(copy_v),
          gV(gV),
          sV(sV),
          gV_stride(gV_stride),
          sV_stride(sV_stride),
          num_stage(num_stage),
          cur_iter(0) {}

    inline __device__ void prologue() {
#pragma unroll
        for (int m = 0; m < size<1>(gV); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(gV); ++k) {
                cute::copy(copy_v, gV(_, m, k), sV(_, m, k));
            }
        }

        // Copies using `cp.async` are separted into "commit groups" using
        // `cp_async_fence()`.
        cute::cp_async_fence();
        gV.data() = gV.data() + gV_stride;
        sV.data() = sV.data() + sV_stride;

        // Circlically read SMEM Buffer
        if ((cur_iter + 1) % num_stage == 0) {
            sV.data() = sV.data() - sV_stride * num_stage;
        }

        cur_iter++;
    }

    inline __device__ void body() {
#pragma unroll
        for (int m = 0; m < size<1>(gV); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(gV); ++k) {
                cute::copy(copy_v, gV(_, m, k), sV(_, m, k));
            }
        }

        cute::cp_async_fence();
        gV.data() = gV.data() + gV_stride;
        sV.data() = sV.data() + sV_stride;

        if ((cur_iter + 1) % num_stage == 0) {
            sV.data() = sV.data() - sV_stride * num_stage;
        }

        cur_iter++;
    }

    inline __device__ void epilogue() {
#pragma unroll
        for (int m = 0; m < size<1>(gV); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(gV); ++k) {
                cute::copy(copy_v, gV(_, m, k), sV(_, m, k));
            }
        }

        cute::cp_async_fence();
    }

  private:
    G2STiledCopy& copy_v;
    GTensor& gV;
    STensor& sV;
    int gV_stride;
    int sV_stride;
    int num_stage;
    int cur_iter;
}

template <class TiledCopyQ, class TiledCopyK, class STensorQ, class RTensorQ,
          class STensorK, class RTensorK, class RTensorAcc, class TiledMMA>
class S2RCopyTileQK {
  public:
    S2RCopyTileQK(TiledCopyQ& copy_q, TiledCopyK& copy_k, STensorQ& sQ,
                  RTensorQ& rQ, STensorK& sK, RTensorK& rK, RTensorAcc& acc,
                  TiledMMA tiled_mma, int sQ_stride, int rQ_stride,
                  int sK_stride, int rK_stride, int num_stage = 2)
        : copy_q(copy_q),
          copy_k(copy_k),
          sQ(sQ),
          rQ(rQ),
          sK(sK),
          rK(rK),
          sQ_stride(sQ_stride),
          rQ_stride(rQ_stride),
          sK_stride(sK_stride),
          rK_stride(rK_stride),
          num_stage(num_stage),
          cur_iter(0),
          cur_iter_sq(0) {}

    inline __device__ void prologue() {
        cur_iter = 0;
        cute::copy(copy_q, sQ(_, _, _0{}), rQ(_, _, _0{}));
        cute::copy(copy_k, sK(_, _, _0{}), rK(_, _, _0{}));

        // Software pipelining Technique.
#pragma unroll
        for (int i = 0; i < size<2>(rK); ++i) {
            if (i < size<2>(rK) - 1) {
                cute::copy(copy_q, sQ(_, _, i + 1), rQ(_, _, i + 1));
                cute::copy(copy_k, sK(_, _, i + 1), rK(_, _, i + 1));
            }

            cute::gemm(tiled_mma, rQ(_, _, i), rK(_, _, i), acc);
        }

        sQ.data() = sQ.data() + sQ_stride;
        sK.data() = sK.data() + sK_stride;
        cur_iter++;
    }

    inline __device__ void body() {
        cute::copy(copy_q, sQ(_, _, _0{}), rQ(_, _, _0{}));
        cute::copy(copy_k, sK(_, _, _0{}), rK(_, _, _0{}));

        // Software pipelining Technique.
        // Loading from SMEM to RMEM is handled by LSU(Load/Store Unit), while
        // computation is handled by a computational unit(e.g., tensor cores).
#pragma unroll
        for (int i = 0; i < size<2>(rK); ++i) {
            if (i < size<2>(rK) - 1) {
                cute::copy(copy_q, sQ(_, _, i + 1), rQ(_, _, i + 1));
                cute::copy(copy_k, sK(_, _, i + 1), rK(_, _, i + 1));
            }

            cute::gemm(tiled_mma, rQ(_, _, i), rK(_, _, i), acc);
        }

        sQ.data() = sQ.data() + sQ_stride;
        sK.data() = sK.data() + sK_stride;

        if ((cur_iter + 1) % num_stage == 0) {
            sK.data() = sK.data() - sK_stride * num_stage;
        }

        cur_iter++;
        cur_iter_sq++;
    }

    inline __device__ void epilogue() {
        cute::copy(copy_q, sQ(_, _, _0{}), rQ(_, _, _0{}));
        cute::copy(copy_k, sK(_, _, _0{}), rK(_, _, _0{}));

        // Software pipelining Technique.
#pragma unroll
        for (int i = 0; i < size<2>(rK); ++i) {
            if (i < size<2>(rK) - 1) {
                cute::copy(copy_q, sQ(_, _, i + 1), rQ(_, _, i + 1));
                cute::copy(copy_k, sK(_, _, i + 1), rK(_, _, i + 1));
            }

            cute::gemm(tiled_mma, rQ(_, _, i), rK(_, _, i), acc);
        }

        sQ.data() = sQ.data() - sQ_stride * cur_iter_sq;
        sK.data() = sK.data() + sK_stride;

        if ((cur_iter + 1) % num_stage == 0) {
            sK.data() = sK.data() - sK_stride * num_stage;
        }

        cur_iter++;
        cur_iter_sq = 0;
    }

  private:
    TiledCopyQ& copy_q;
    TiledCopyK& copy_k;
    STensorQ& sQ;
    RTensorQ& rQ;
    STensorK& sK;
    RTensorK& rK;
    RTensorAcc& acc;
    TiledMMA tiled_mma;
    int sQ_stride;
    int rQ_stride;
    int sK_stride;
    int rK_stride;
    int num_stage;
    int cur_iter;
    int cur_iter_sq;
}

}  // namespace cutlass_wrapper
}  // namespace benchmarks