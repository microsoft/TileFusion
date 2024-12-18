// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>

namespace benchmarks {
namespace cutlass_wrapper {

using namespace cute;

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

}  // namespace cutlass_wrapper
}  // namespace benchmarks