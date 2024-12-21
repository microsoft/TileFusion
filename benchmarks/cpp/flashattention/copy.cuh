// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cute/tensor.hpp>

namespace benchmarks {
namespace cutlass_wrapper {

using namespace cute;

namespace detail {
template <typename GQTensor, typename SQTensor, typename GKTensor,
          typename SKTensor, typename TiledCopy>
class G2SCopyQK {
  public:
    __device__ G2SCopyQK(GQTensor& gQ, SQTensor& sQ, GKTensor& gK, SKTensor& sK,
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
          num_stage(num_stage) {}

    inline __device__ void print_q() {
        if (thread0()) {
            print(gQ), print("\n");
        }
    }

    inline __device__ void prologue() {
        // Pipeline the copy operation.
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
            sQ.data() = sQ.data() - sQ_stride * num_stage;
            sK.data() = sK.data() - sK_stride * num_stage;
        }

        cur_iter++;
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
    int num_stage;
};

}  // namespace detail

template <typename Element, typename GlobalQLayout, typename SharedQLayout,
          typename GlobalKLayout, typename SharedKLayout, typename TiledCopy>
inline __device__ auto make_g2s_qk(const Element* gQ_ptr, Element* sQ_ptr,
                                   const Element* gK_ptr, Element* sK_ptr,
                                   int gQ_stride, int sQ_stride, int gK_stride,
                                   int sK_stride) {
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

    detail::G2SCopyQK copy_qk(gQs, sQs, gKs, sKs, tiled_copy, gQ_stride,
                              sQ_stride, gK_stride, sK_stride);

    return copy_qk;
}

// template <class G2STiledCopy, class GTensor, class STensor>
// class G2SCopyTile {
//   public:
//     G2SCopyTile(G2STiledCopy& copy_v, GTensor& gV, STensor& sV, int
//     gV_stride,
//                 int sV_stride, int num_stage = 2)
//         : copy_v(copy_v),
//           gV(gV),
//           sV(sV),
//           gV_stride(gV_stride),
//           sV_stride(sV_stride),
//           num_stage(num_stage),
//           cur_iter(0) {}

//     inline __device__ void prologue() {
// #pragma unroll
//         for (int m = 0; m < size<1>(gV); ++m) {
// #pragma unroll
//             for (int k = 0; k < size<2>(gV); ++k) {
//                 cute::copy(copy_v, gV(_, m, k), sV(_, m, k));
//             }
//         }

//         // Copies using `cp.async` are separted into "commit groups" using
//         // `cp_async_fence()`.
//         cute::cp_async_fence();
//         gV.data() = gV.data() + gV_stride;
//         sV.data() = sV.data() + sV_stride;

//         // Circlically read SMEM Buffer
//         if ((cur_iter + 1) % num_stage == 0) {
//             sV.data() = sV.data() - sV_stride * num_stage;
//         }

//         cur_iter++;
//     }

//     inline __device__ void body() {
// #pragma unroll
//         for (int m = 0; m < size<1>(gV); ++m) {
// #pragma unroll
//             for (int k = 0; k < size<2>(gV); ++k) {
//                 cute::copy(copy_v, gV(_, m, k), sV(_, m, k));
//             }
//         }

//         cute::cp_async_fence();
//         gV.data() = gV.data() + gV_stride;
//         sV.data() = sV.data() + sV_stride;

//         if ((cur_iter + 1) % num_stage == 0) {
//             sV.data() = sV.data() - sV_stride * num_stage;
//         }

//         cur_iter++;
//     }

//     inline __device__ void epilogue() {
// #pragma unroll
//         for (int m = 0; m < size<1>(gV); ++m) {
// #pragma unroll
//             for (int k = 0; k < size<2>(gV); ++k) {
//                 cute::copy(copy_v, gV(_, m, k), sV(_, m, k));
//             }
//         }

//         cute::cp_async_fence();
//     }

//   private:
//     G2STiledCopy& copy_v;
//     GTensor& gV;
//     STensor& sV;
//     int gV_stride;
//     int sV_stride;
//     int num_stage;
//     int cur_iter;
// }

// template <class TiledCopyQ, class TiledCopyK, class STensorQ, class RTensorQ,
//           class STensorK, class RTensorK, class RTensorAcc, class TiledMMA>
// class S2RCopyTileQK {
//   public:
//     S2RCopyTileQK(TiledCopyQ& copy_q, TiledCopyK& copy_k, STensorQ& sQ,
//                   RTensorQ& rQ, STensorK& sK, RTensorK& rK, RTensorAcc& acc,
//                   TiledMMA tiled_mma, int sQ_stride, int rQ_stride,
//                   int sK_stride, int rK_stride, int num_stage = 2)
//         : copy_q(copy_q),
//           copy_k(copy_k),
//           sQ(sQ),
//           rQ(rQ),
//           sK(sK),
//           rK(rK),
//           sQ_stride(sQ_stride),
//           rQ_stride(rQ_stride),
//           sK_stride(sK_stride),
//           rK_stride(rK_stride),
//           num_stage(num_stage),
//           cur_iter(0),
//           cur_iter_sq(0) {}

//     inline __device__ void prologue() {
//         cur_iter = 0;
//         cute::copy(copy_q, sQ(_, _, _0{}), rQ(_, _, _0{}));
//         cute::copy(copy_k, sK(_, _, _0{}), rK(_, _, _0{}));

//         // Software pipelining Technique.
// #pragma unroll
//         for (int i = 0; i < size<2>(rK); ++i) {
//             if (i < size<2>(rK) - 1) {
//                 cute::copy(copy_q, sQ(_, _, i + 1), rQ(_, _, i + 1));
//                 cute::copy(copy_k, sK(_, _, i + 1), rK(_, _, i + 1));
//             }

//             cute::gemm(tiled_mma, rQ(_, _, i), rK(_, _, i), acc);
//         }

//         sQ.data() = sQ.data() + sQ_stride;
//         sK.data() = sK.data() + sK_stride;
//         cur_iter++;
//     }

//     inline __device__ void body() {
//         cute::copy(copy_q, sQ(_, _, _0{}), rQ(_, _, _0{}));
//         cute::copy(copy_k, sK(_, _, _0{}), rK(_, _, _0{}));

//         // Software pipelining Technique.
//         // Loading from SMEM to RMEM is handled by LSU(Load/Store Unit),
//         while
//         // computation is handled by a computational unit(e.g., tensor
//         cores).
// #pragma unroll
//         for (int i = 0; i < size<2>(rK); ++i) {
//             if (i < size<2>(rK) - 1) {
//                 cute::copy(copy_q, sQ(_, _, i + 1), rQ(_, _, i + 1));
//                 cute::copy(copy_k, sK(_, _, i + 1), rK(_, _, i + 1));
//             }

//             cute::gemm(tiled_mma, rQ(_, _, i), rK(_, _, i), acc);
//         }

//         sQ.data() = sQ.data() + sQ_stride;
//         sK.data() = sK.data() + sK_stride;

//         if ((cur_iter + 1) % num_stage == 0) {
//             sK.data() = sK.data() - sK_stride * num_stage;
//         }

//         cur_iter++;
//         cur_iter_sq++;
//     }

//     inline __device__ void epilogue() {
//         cute::copy(copy_q, sQ(_, _, _0{}), rQ(_, _, _0{}));
//         cute::copy(copy_k, sK(_, _, _0{}), rK(_, _, _0{}));

//         // Software pipelining Technique.
// #pragma unroll
//         for (int i = 0; i < size<2>(rK); ++i) {
//             if (i < size<2>(rK) - 1) {
//                 cute::copy(copy_q, sQ(_, _, i + 1), rQ(_, _, i + 1));
//                 cute::copy(copy_k, sK(_, _, i + 1), rK(_, _, i + 1));
//             }

//             cute::gemm(tiled_mma, rQ(_, _, i), rK(_, _, i), acc);
//         }

//         sQ.data() = sQ.data() - sQ_stride * cur_iter_sq;
//         sK.data() = sK.data() + sK_stride;

//         if ((cur_iter + 1) % num_stage == 0) {
//             sK.data() = sK.data() - sK_stride * num_stage;
//         }

//         cur_iter++;
//         cur_iter_sq = 0;
//     }

//   private:
//     TiledCopyQ& copy_q;
//     TiledCopyK& copy_k;
//     STensorQ& sQ;
//     RTensorQ& rQ;
//     STensorK& sK;
//     RTensorK& rK;
//     RTensorAcc& acc;
//     TiledMMA tiled_mma;
//     int sQ_stride;
//     int rQ_stride;
//     int sK_stride;
//     int rK_stride;
//     int num_stage;
//     int cur_iter;
//     int cur_iter_sq;
// }

}  // namespace cutlass_wrapper
}  // namespace benchmarks