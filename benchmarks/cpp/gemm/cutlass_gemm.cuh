// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.cuh"
#include "cutlass/copy.cuh"
#include "cutlass/traits_base.cuh"

#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>

namespace benchmarks {
namespace cutlass_wrapper {
using namespace cute;

template <typename Element_,                             //
          const int kWarpPerRow, const int kWarpPerCol,  //
          const int kM, const int kN, const int kK,      //
          const int kTM, const int kTN, const int kTK,   //
          typename Base = AccessBase<Element_>>
struct GemmTraits : public Base {
    using Element = Element_;

    static_assert(kTM % kWarpPerRow == 0,
                  "the M dimension of the CTA tile should be divisible by the "
                  "number of warps along that that dimension.");
    static_assert(kTN % kWarpPerCol == 0,
                  "the N dimension of the CTA tile should be divisible by the "
                  "number of warps along that that dimension.");

    // declare global to shared memory copy layout.
    using GmemLayoutA = Layout<Shape<Int<kTM>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutB = Layout<Shape<Int<kTN>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutC = Layout<Shape<Int<kTM>, Int<kTN>>, Stride<Int<kN>, _1>>;

    using TiledMma =
        TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,  // for ampere
                 Layout<Shape<Int<kWarpPerRow>, Int<kWarpPerCol>, _1>>,
                 Tile<Int<16 * kWarpPerRow>, Int<16 * kWarpPerCol>, _16>>;

    static constexpr int kThreads = size(TiledMma{});
    static_assert(kThreads == kWarpPerRow * kWarpPerCol * 32);

    static constexpr int kNumPerAccess = Base::kNumPerAccess;
    static constexpr int kThreadsPerCol = CeilDiv<kTK, Base::kNumPerAccess>;
    static constexpr int kThreadsPerRow = CeilDiv<kThreads, kThreadsPerCol>;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<2, 3, 3>{}, Layout<Shape<_8, Int<4 * kNumPerAccess>>,
                                   Stride<Int<4 * kNumPerAccess>, _1>>{}));

    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));
    using SmemLayoutC =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTN>>{}));

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInstG2S =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
#else
    using CopyInstG2S = Copy_Atom<DefaultCopy, Element>;
#endif

    using GmemCopyLayoutAtom =
        Layout<Shape<Int<kThreads / kThreadsPerRow>, Int<kThreadsPerRow>>,
               Stride<Int<kThreadsPerRow>, _1>>;

    using GmemTiledCopy = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
        GmemCopyLayoutAtom{}, Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));

    using TiledCopyG2S = decltype(make_tiled_copy(
        CopyInstG2S{},
        Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
               Stride<Int<kThreadsPerCol>, _1>>{},
        Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));

    using TiledCopyS2G = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
               Stride<Int<kThreadsPerCol>, _1>>{},
        Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));
    using StoreC_R2S = R2SCopy2D<Element, TiledMma, SmemLayoutC>;
};

template <typename Element, const int kM, const int kN, const int kK,
          const int kTM, const int kTN, const int kTK, typename KeTraits>
__global__ void gemm_kernel(const Element* dA, const Element* dB, Element* dC) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    // Advance to the global data tile to the current CTA.
    Element* gA_ptr = const_cast<Element*>(dA) + blockIdx.x * kK * kTM;
    Element* gB_ptr = const_cast<Element*>(dB) + blockIdx.y * kK * kTN;
    Element* gC_ptr = dC + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    // pointers to shared memory tiles
    Element* sA_ptr = buf;
    Element* sB_ptr = buf + kTM * kTK;
    Element* sC_ptr = buf;

    typename KeTraits::TiledMma mma;
    typename KeTraits::TiledCopyG2S tiled_copy;

    auto rA = make_s2rA(sA_ptr, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, typename KeTraits::SmemLayoutB{}, mma);
    auto acc = get_acc<kTM, kTN>(mma);

    for (int k = 0; k < kK; k += kTK) {
        copy_tile_g2s(gA_ptr, sA_ptr, typename KeTraits::GmemLayoutA{},
                      typename KeTraits::SmemLayoutA{}, tiled_copy);
        copy_tile_g2s(gB_ptr, sB_ptr, typename KeTraits::GmemLayoutB{},
                      typename KeTraits::SmemLayoutB{}, tiled_copy);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rA.get_iters(); ++i) {
            rA.copy(i);  // load A register tile from shared memory
            rB.copy(i);  // load B register tile from shared memory

            gemm(mma, rA[i], rB[i], acc);
        }
        gA_ptr += kTK;
        gB_ptr += kTK;
    }

    // declare register to shared store plan
    typename KeTraits::StoreC_R2S sC;
    // store register tile to shared memory
    sC.copy(acc, buf);
    __syncthreads();

    // store shared memory tile to global memory
    copy_tile_s2g(sC_ptr, gC_ptr, typename KeTraits::SmemLayoutC{},
                  typename KeTraits::GmemLayoutC{},
                  typename KeTraits::TiledCopyS2G{});
}

template <typename Element, const int kM, const int kN, const int kK,
          const int kTM, const int kTN, const int kTK, const int num_stages,
          typename KeTraits>
__global__ void gemm_pipeline_kernel(const Element* dA, const Element* dB,
                                     Element* dC) {
    using namespace cute;

    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    Element* gA_ptr = const_cast<Element*>(dA) + blockIdx.x * kK * kTM;
    Element* gB_ptr = const_cast<Element*>(dB) + blockIdx.y * kK * kTN;
    Element* gC_ptr = dC + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    Element* sA_ptr = shm;
    Element* sC_ptr = shm;

    // load the first A, B tiles from global memory to shared memory
    typename KeTraits::GmemTiledCopy tiled_copy;
    auto copy_thrd = tiled_copy.get_thread_slice(threadIdx.x);

    auto gA =
        make_tensor(make_gmem_ptr(gA_ptr), typename KeTraits::GmemLayoutA{});
    auto gA_thrd = copy_thrd.partition_S(gA);
    auto sA =
        make_tensor(make_smem_ptr(sA_ptr), typename KeTraits::SmemLayoutA{});
    auto sA_thrd = copy_thrd.partition_D(sA);

    auto gB =
        make_tensor(make_gmem_ptr(gB_ptr), typename KeTraits::GmemLayoutB{});
    auto gB_thrd = copy_thrd.partition_S(gB);
    auto sB = make_tensor(sA.data() + num_stages * size(sA),
                          typename KeTraits::SmemLayoutB{});
    auto sB_thrd = copy_thrd.partition_D(sB);

    CopyAsyncG2S g2s(num_stages, tiled_copy, gA_thrd, kTK, sA_thrd, size(sA),
                     gB_thrd, kTK, sB_thrd, size(sB));

    g2s.copy();  // commit the 1st async copy group
    g2s.copy();  // commit the 2nd async copy group
    // Allows for one unfinished cp.async operation.
    g2s.template wait_group<1>();
    __syncthreads();

    typename KeTraits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto acc = partition_fragment_C(tiled_mma, Shape<Int<kTM>, Int<kTN>>{});
    clear(acc);

    // data tiles that are stored on local registers
    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    auto s2r_copy_A = make_tiled_copy_A(SmemLoadAtom{}, tiled_mma);
    auto s2r_copy_A_thrd = s2r_copy_A.get_thread_slice(threadIdx.x);
    auto sArA = s2r_copy_A_thrd.partition_S(sA);
    auto rA = thr_mma.partition_fragment_A(sA);
    auto rA_view = s2r_copy_A_thrd.retile_D(rA);  // retile for copy

    auto s2r_copy_B = make_tiled_copy_B(SmemLoadAtom{}, tiled_mma);
    auto s2r_copy_B_thrd = s2r_copy_B.get_thread_slice(threadIdx.x);
    auto sBrB = s2r_copy_B_thrd.partition_S(sB);
    auto rB = thr_mma.partition_fragment_B(sB);
    auto rB_view = s2r_copy_B_thrd.retile_D(rB);  // retile for copy

    static_assert(size<2>(rA) == size<2>(rB),
                  "Error partition of thread tiles.");
    const int k_tiles = size<2>(rB);
    const int Na = size<1>(sA_thrd) * size<2>(sA_thrd);
    const int na = CeilDiv<Na, k_tiles>;
    const int stride_a = size<2>(sA_thrd);

    const int Nb = size<1>(sB_thrd) * size<2>(sB_thrd);
    const int nb = CeilDiv<Nb, k_tiles>;
    const int stride_b = size<2>(sB_thrd);

    // issue the first data loading from shared memory to register
    cute::copy(s2r_copy_A, sArA(_, _, _0{}), rA_view(_, _, _0{}));
    cute::copy(s2r_copy_B, sBrB(_, _, _0{}), rB_view(_, _, _0{}));

    // stage 1
    for (int blk = 0; blk < kK / kTK - 2; ++blk) {
        CUTE_UNROLL
        for (int i = 0; i < k_tiles; ++i) {
            // circular issue next data loading from shared memory
            // into registers
            int pos = (i + 1) % k_tiles;
            cute::copy(s2r_copy_A, sArA(_, _, pos), rA_view(_, _, pos));
            cute::copy(s2r_copy_B, sBrB(_, _, pos), rB_view(_, _, pos));

            if (i < k_tiles - 1) {
                // gmem -> shared memory
                g2s.copy2(i, na, Na, stride_a, nb, Nb, stride_b);
            }

            if (i == k_tiles - 2) {
                sArA.data() = sArA.data() + size(sA);
                sBrB.data() = sBrB.data() + size(sB);

                if ((blk + 1) % num_stages == 0) {
                    sArA.data() = sArA.data() + (-size(sA) * num_stages);
                    sBrB.data() = sBrB.data() + (-size(sB) * num_stages);
                }

                g2s.copy2(i + 1, na, Na, stride_a, nb, Nb, stride_b);
                g2s.commit_copy_group();
                g2s.next();
                if ((blk + 2 + 1) % num_stages == 0) g2s.cycle_dst();

                g2s.template wait_group<1>();
                __syncthreads();
            }

            cute::gemm(tiled_mma, rA(_, _, i), rB(_, _, i),
                       acc);  // compute
        }
    }

    // stage 2
    CUTE_UNROLL
    for (int i = 0; i < k_tiles; ++i) {
        // circular issue next data loading from shared memory into
        // registers
        int pos = (i + 1) % k_tiles;
        cute::copy(s2r_copy_A, sArA(_, _, pos), rA_view(_, _, pos));
        cute::copy(s2r_copy_B, sBrB(_, _, pos), rB_view(_, _, pos));

        if (i == k_tiles - 2) {
            sArA.data() = sArA.data() + size(sA);
            sBrB.data() = sBrB.data() + size(sB);
            if ((kK / kTK - 2 + 1) % num_stages == 0) {
                sArA.data() = sArA.data() + (-size(sA) * num_stages);
                sBrB.data() = sBrB.data() + (-size(sB) * num_stages);
            }

            g2s.template wait_group<0>();
            __syncthreads();
        }

        cute::gemm(tiled_mma, rA(_, _, i), rB(_, _, i),
                   acc);  // compute
    }

    // stage 3
    CUTE_UNROLL
    for (int i = 0; i < k_tiles; ++i) {
        if (i < k_tiles - 1) {
            // circular issue next data loading from shared memory
            // into registers
            int pos = (i + 1) % k_tiles;
            cute::copy(s2r_copy_A, sArA(_, _, pos), rA_view(_, _, pos));
            cute::copy(s2r_copy_B, sBrB(_, _, pos), rB_view(_, _, pos));
        }

        cute::gemm(tiled_mma, rA(_, _, i), rB(_, _, i),
                   acc);  // compute
    }
    __syncthreads();

    // convert the accumulator to the output type
    // auto rC = convert_type<Element>(acc);

    typename KeTraits::StoreC_R2S sC;
    sC.copy(acc, sC_ptr);
    __syncthreads();

    // copy the result from shared memory to global memory
    copy_tile_s2g(sC_ptr, gC_ptr, typename KeTraits::SmemLayoutC{},
                  typename KeTraits::GmemLayoutC{},
                  typename KeTraits::TiledCopyS2G{});
}

}  // namespace cutlass_wrapper
}  // namespace benchmarks
