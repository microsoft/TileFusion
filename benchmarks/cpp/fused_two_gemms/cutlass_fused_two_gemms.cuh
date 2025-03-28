#pragma once

#include "cuda_utils.cuh"
#include "cutlass/convert.cuh"
#include "cutlass/copy.cuh"
#include "cutlass/traits_base.cuh"
#include "util/cuda_timer.hpp"

#include <cute/tensor.hpp>

namespace benchmarks {
namespace cutlass_wrapper {
using namespace cute;

template <typename Element_,                                           //
          const int kWarpPerRow, const int kWarpPerCol,                //
          const int kM, const int kN, const int kK, const int kP,      //
          const int kTM, const int kTN, const int kTK, const int kTP,  //
          typename Base = AccessBase<Element_>>
struct FusedGemmTraits : public Base {
    using Element = Element_;

    static_assert(kTK == kTN && kTN == kTP,
                  "Fused GEMM requires kTK == kTN == kTP.");
    static_assert(kWarpPerCol == 1,
                  "The Fused GEMM requires a single warp along CTA tile.");

    using GmemLayoutA = Layout<Shape<Int<kTM>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutB = Layout<Shape<Int<kTN>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutC = Layout<Shape<Int<kTP>, Int<kTN>>, Stride<Int<kN>, _1>>;
    using GmemLayoutD = Layout<Shape<Int<kTM>, Int<kTP>>, Stride<Int<kP>, _1>>;

    // TODO(haruhi): The current implementation uses ldmatrix.x4
    // instruction which requires the TileMMA configuration to be
    // fixed as follows. Make it able to be tuned by policy in
    // future implementation.
    using TiledMma =
        TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,  // for ampere
                 Layout<Shape<Int<kWarpPerRow>, Int<kWarpPerCol>, _1>>,
                 Tile<Int<16 * kWarpPerRow>, Int<16 * kWarpPerCol>, _16>>;
    static constexpr int kThreads = size(TiledMma{});
    static_assert(kThreads == kWarpPerRow * kWarpPerCol * 32);

    static constexpr int kNumPerAccess = Base::kNumPerAccess;
    static constexpr int kThreadsPerCol = CeilDiv<kTK, Base::kNumPerAccess>;
    static constexpr int kThreadsPerRow = CeilDiv<kThreads, kThreadsPerCol>;

    // static constexpr int kSwizzle = (kTK == 32 ? 2 : 3);
    using SmemLayoutAtom = decltype(composition(
        Swizzle<2, 3, 3>{},
        Layout<Shape<_8, Int<kTK>>, Stride<Int<kTK>, _1>>{}));

    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));

    // The current implementation requires B are laid out in column
    // major. a [kTK, kTN] matrix in column major can be interpreted
    // as a [kTN, kTK] matrix in row major.
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));
    // a [kTN, kTP] matrix in column major fashion,
    // can be interpreted as a [kTP, kTN] matrix in row major fashion.
    using SmemLayoutC =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTP>, Int<kTN>>{}));

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInstG2S =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
#else
    using CopyInstG2S = Copy_Atom<DefaultCopy, Element>;
#endif
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
    using SmemLayoutD =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTP>>{}));

    using StoreD_R2S = R2SCopy2D<Element, TiledMma, SmemLayoutD>;
};

template <typename Element, const int kM, const int kN, const int kK,
          const int kP, const int kTM, const int kTN, const int kTK,
          const int kTP, typename KeTraits>
__global__ void fused_gemm_kernel(const Element* dA, const Element* dB,
                                  const Element* dC, Element* dD) {
    // Advance to the global data tile to the current CTA.
    Element* A = const_cast<Element*>(dA) + blockIdx.x * (kTM * kK);
    Element* B = const_cast<Element*>(dB);
    Element* gC_ptr = const_cast<Element*>(dC) + blockIdx.y * (kTP * kN);
    Element* gD_ptr = dD + blockIdx.x * (kTM * kP) + (blockIdx.y * kTP);

    Element* gA_ptr;
    Element* gB_ptr;

    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);
    // pointers to shared memory tiles
    Element* sA_ptr = shm;
    Element* sB_ptr = shm + kTM * kTK;
    Element* sC_ptr = shm + kTM * kTK + kTK * kTN;
    Element* sD_ptr = shm;

    typename KeTraits::TiledMma mma;  // for shared memory to register copy
    typename KeTraits::TiledCopyG2S tiled_copy;

    auto rA = make_s2rA(sA_ptr, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, typename KeTraits::SmemLayoutB{}, mma);
    auto acc1 = get_acc<kTM, kTN>(mma);  // accumulator for the 1st gemm

    auto rC = make_s2rB(sC_ptr, typename KeTraits::SmemLayoutC{}, mma);
    auto acc2 = get_acc<kTM, kTP>(mma);  // accumulator for the 2nd gemm

    typename KeTraits::StoreD_R2S sD;  // declare register to shared store plan

    for (int n = 0; n < kN; n += kTN) {  // iterate over N
        gA_ptr = A;                      // A tile is repeated loaded
        gB_ptr = B + n * kK;
        for (int k = 0; k < kK; k += kTK) {  // iterate over K
            copy_tile_g2s(gA_ptr, sA_ptr, typename KeTraits::GmemLayoutA{},
                          typename KeTraits::SmemLayoutA{}, tiled_copy);
            copy_tile_g2s(gB_ptr, sB_ptr, typename KeTraits::GmemLayoutB{},
                          typename KeTraits::SmemLayoutB{}, tiled_copy);
            __copy_async();
            __syncthreads();

            // iterate over the register tiles along the kTK dimension
            for (int i = 0; i < rA.get_iters(); ++i) {
                rA.copy(i);  // load A register tile from shared memory
                rB.copy(i);  // load B register tile from shared memory
                gemm(mma, rA[i], rB[i], acc1);  // compute
            }
            __syncthreads();

            gA_ptr += kTK;
            gB_ptr += kTK;
        }

        // The output type of the first tensor core matrix multiplication is
        // float32. However, before the second GEMM operation, the output
        // needs to be converted to half precision.
        auto acc_half = convert_type<Element>(acc1);
        auto rA2 = convert_layout<KeTraits::TiledMma>(acc_half);

        // load C tile from global to shared memory
        copy_tile_g2s(gC_ptr, sC_ptr, typename KeTraits::GmemLayoutC{},
                      typename KeTraits::SmemLayoutC{}, tiled_copy);
        __copy_async();
        __syncthreads();

        // iterate over register tiles along the kTN dimension
        for (int i = 0; i < rC.get_iters(); ++i) {
            rC.copy(i);  // load C tile from shared memory to register
            gemm(mma, rA2[i], rC[i], acc2);  // compute
        }
        __syncthreads();

        clear(acc1);
        gC_ptr += kTN;
    }

    // store register tile to shared memory
    sD.copy(acc2, shm);
    __syncthreads();

    copy_tile_s2g(sD_ptr, gD_ptr, typename KeTraits::SmemLayoutD{},
                  typename KeTraits::GmemLayoutD{},
                  typename KeTraits::TiledCopyS2G{});
}
}  // namespace cutlass_wrapper
}  // namespace benchmarks

using namespace tilefusion;

template <typename Element,                                        //
          const int kWarpPerRow, const int kWarpPerCol,            //
          const int kM, const int kN, const int kK, const int kP,  //
          const int kTM, const int kTN, const int kTK, const int kTP>
float cute_fused_gemm(const Element* dA, const Element* dB, const Element* dC,
                      Element* dD, bool timeit = false, int warp_up = 5,
                      int iters = 20) {
    using namespace benchmarks::cutlass_wrapper;

    using KeTraits = benchmarks::cutlass_wrapper::FusedGemmTraits<
        Element, kWarpPerRow, kWarpPerCol, kM, kN, kK, kP, kTM, kTN, kTK, kTP>;

    auto kernel = &benchmarks::cutlass_wrapper::fused_gemm_kernel<
        Element, kM, kN, kK, kP, kTM, kTN, kTK, kTP, KeTraits>;

    int shm_input = (kTM * kTK + kTK * kTN + kTN * kTP);
    int shm_output = kTM * kTP;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(Element)
                                          : shm_input * sizeof(Element);

    // maximal statically allocated smem per block
    const int kMaxSmemPerBlock = 48 * 1024;
    if (shm_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    // blocks are launched along the M and P dimensions.
    int block_x = (kM + kTM - 1) / kTM;
    int block_y = (kP + kTP - 1) / kTP;
    const int kThreads = KeTraits::kThreads;

    dim3 gridDim(block_x, block_y, 1);
    dim3 blockDim(kThreads, 1, 1);

    float elapsed = 0.;
    if (timeit) {
        for (int i = 0; i < warp_up; ++i) {
            kernel<<<gridDim, blockDim, shm_size>>>(dA, dB, dC, dD);
        }
        cudaDeviceSynchronize();

        CudaTimer timer;
        timer.start();
        for (int i = 0; i < iters; ++i) {
            kernel<<<gridDim, blockDim, shm_size>>>(dA, dB, dC, dD);
        }
        cudaDeviceSynchronize();
        elapsed = timer.stop() / iters;
    } else {
        kernel<<<gridDim, blockDim, shm_size>>>(dA, dB, dC, dD);
    }
    return elapsed;
}
