// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

namespace benchmarks {
namespace cutlass_wrapper {
using namespace cute;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    #define CP_ASYNC_SM80_ENABLED
#endif

template <int N>
DEVICE void wait_group() {
#if defined(CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

DEVICE void commit_copy_group() {
#if defined(CP_ASYNC_SM80_ENABLED)
    cute::cp_async_fence();
#endif
}

DEVICE void __copy_async() {
    commit_copy_group();
    wait_group<0>();
}

template <int N>
DEVICE void cp_async_wait_flash() {
#if defined(CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// Copy a 2d data tile from global memory to shared memory
template <typename Element, typename SrcLayout, typename DstLayout,
          typename TiledCopy>
DEVICE void copy_tile_g2s(const Element* src_data, Element* dst_data,
                          SrcLayout src_layout, DstLayout dst_layout,
                          TiledCopy tiled_copy) {
    int tid = threadIdx.x;

    auto gtile = make_tensor(make_gmem_ptr(src_data), src_layout);
    auto stile = make_tensor(make_smem_ptr(dst_data), dst_layout);

    auto loader = tiled_copy.get_thread_slice(tid);

    auto src = loader.partition_S(gtile);
    auto dst = loader.partition_D(stile);

#pragma unroll
    for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
        for (int j = 0; j < int(size<2>(src)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst(_, i, j));
}

// Copy a tensor from shared memory to global memory
template <typename Element, typename SrcLayout, typename DstLayout,
          typename TiledCopy>
DEVICE void copy_tile_s2g(const Element* src_data, Element* dst_data,
                          SrcLayout src_layout, DstLayout dst_layout,
                          TiledCopy tiled_copy) {
    int tid = threadIdx.x;

    auto stile = make_tensor(make_smem_ptr(src_data), src_layout);
    auto gtile = make_tensor(make_gmem_ptr(dst_data), dst_layout);

    auto loader = tiled_copy.get_thread_slice(tid);

    auto src = loader.partition_S(stile);
    auto dst = loader.partition_D(gtile);

#pragma unroll
    for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
        for (int j = 0; j < int(size<2>(src)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst(_, i, j));
}

template <typename Element, typename TiledMma_, typename DstLayout>
struct R2SCopy2D {
    using TiledMma = TiledMma_;
    using Dstlayout_ = DstLayout;
    using CopyAtom = Copy_Atom<DefaultCopy, Element>;

  public:
    template <typename Engine, typename Layout>
    __device__ void copy(cute::Tensor<Engine, Layout> const& acc,
                         Element* dst_data) {
        int tid = threadIdx.x;

        // FIXME(haruhi): This implementation is specifically designed
        // for tcu WMMA and assumes that the ACC value has a
        // floating-point precision. The code converts the ACC value
        // to half-precision.
        auto src_tensor = convert_type<Element>(acc);
        auto dst_tensor = make_tensor(make_smem_ptr(dst_data), DstLayout{});

        auto tiled_copy = make_tiled_copy_C(CopyAtom{}, TiledMma{});
        auto thrd_copy = tiled_copy.get_thread_slice(tid);

        auto src = thrd_copy.retile_S(src_tensor);
        auto dst = thrd_copy.partition_D(dst_tensor);
        cute::copy(tiled_copy, src, dst);
    }

  private:
    template <typename To_type, typename Engine, typename Layout>
    DEVICE auto convert_type(cute::Tensor<Engine, Layout> const& tensor) {
        using From_type = typename Engine::value_type;
        constexpr int numel = decltype(size(tensor))::value;
        cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
        // HACK: this requires tensor to be "contiguous"
        auto frag = convert_op(
            *reinterpret_cast<const cutlass::Array<From_type, numel>*>(
                tensor.data()));
        return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    }
};

template <typename TiledCopy, typename STensor, typename DTensor,
          typename DTensorView>
struct Shm2RegLoad {
  public:
    DEVICE Shm2RegLoad(TiledCopy& copy, const STensor& src, DTensor& dst,
                       DTensorView& dst_view)
        : tiled_copy_(copy), src_(src), dst_(dst), dst_view_(dst_view) {}

    DEVICE void copy(int pos) {
        cute::copy(tiled_copy_, src_(_, _, pos), dst_view_(_, _, pos));
    }

    DEVICE int get_iters() { return size<2>(dst_); }

    DEVICE const auto operator[](int idx) { return dst_(_, _, idx); }

  private:
    TiledCopy& tiled_copy_;
    const STensor& src_;
    DTensor& dst_;
    DTensorView& dst_view_;
};

template <const int kM, const int kN, typename TiledMma>
DEVICE auto get_acc(const TiledMma& tiled_mma) {
    auto acc = partition_fragment_C(tiled_mma, Shape<Int<kM>, Int<kN>>{});
    clear(acc);

    return acc;
}

template <typename Element, typename Layout, typename TiledMma>
DEVICE auto make_s2rA(const Element* data, const Layout& layout,
                      const TiledMma& tiled_mma) {
    int tid = threadIdx.x;

    auto tensor = cute::make_tensor(make_smem_ptr(data), layout);

    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    auto tiled_copy = make_tiled_copy_A(SmemLoadAtom{}, tiled_mma);

    auto thrd_copy = tiled_copy.get_thread_slice(tid);
    auto src = thrd_copy.partition_S(tensor);

    // partition register
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto dst = thr_mma.partition_fragment_A(tensor);
    auto dst_view = thrd_copy.retile_D(dst);

    Shm2RegLoad loader(tiled_copy, src, dst, dst_view);
    return loader;
}

// FIXIME(haruhi): the current implementation is for fast experiment,
// it is coupled shared memory layout with the register layout
template <typename Element, typename Layout, typename TiledMma>
DEVICE auto make_s2rB(const Element* data, const Layout& layout,
                      const TiledMma& tiled_mma) {
    int tid = threadIdx.x;

    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    auto tiled_copy = make_tiled_copy_B(SmemLoadAtom{}, tiled_mma);
    auto thrd_copy = tiled_copy.get_thread_slice(tid);

    auto tensor = make_tensor(make_smem_ptr(data), layout);
    auto src = thrd_copy.partition_S(tensor);

    // partition register
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto dst = thr_mma.partition_fragment_B(tensor);
    auto dst_view = thrd_copy.retile_D(dst);

    Shm2RegLoad loader(tiled_copy, src, dst, dst_view);
    return loader;
}

/**
 * @brief A struct that represents an asynchronous copy of TWO
 * operands operation from global memory to shared memory.
 *
 * This struct is used in the context of a matrix-matrix
 * multiplication kernel. It performs a tiled copy operation from a
 * source tensor in global memory to a destination tensor in shared
 * memory. The copy operation is performed in multiple stages, where
 * each stage copies a tile of the source tensor to the destination
 * tensor. The number of stages is specified by the template
 * parameter kNumStages.
 */
template <typename TiledCopy, typename SrcTensor1, typename DstTensor1,
          typename SrcTensor2, typename DstTensor2>
struct CopyAsyncG2S {
  public:
    CUTE_DEVICE
    CopyAsyncG2S(int num_stages, TiledCopy& tiled_copy, SrcTensor1& s1, int ss1,
                 DstTensor1& d1, int ds1, SrcTensor2& s2, int ss2,
                 DstTensor2& d2, int ds2)
        : num_stages(num_stages),
          tiled_copy(tiled_copy),
          src1(s1),
          src_stride1(ss1),
          dst1(d1),
          dst_stride1(ds1),
          src2(s2),
          src_stride2(ss2),
          dst2(d2),
          dst_stride2(ds2),
          iter(0) {}

    // All threads within a CTA work together to load the first data
    // tile from global memory to shared memory.
    CUTE_DEVICE
    void copy() {
        CUTE_UNROLL
        for (int i = 0; i < size<1>(src1); ++i) {
            CUTE_UNROLL
            for (int j = 0; j < size<2>(src1); ++j) {
                cute::copy(tiled_copy, src1(_, i, j), dst1(_, i, j));
            }
        }

        CUTE_UNROLL
        for (int i = 0; i < size<1>(src2); ++i) {
            CUTE_UNROLL
            for (int j = 0; j < size<2>(src2); ++j) {
                cute::copy(tiled_copy, src2(_, i, j), dst2(_, i, j));
            }
        }
        commit_copy_group();
        next();

        if ((iter + 1) % num_stages == 0) cycle_dst();

        ++iter;
    }

    /**
     * @param n1, for operand B, ON AVERAGE, how many cp.async
     operations need to be issued in a single compute iteration.
     * @param N1, for the operand A, how many cp.async operations
     are needed to issue IN TOTAL by a single thread.
     * @param stride1, for the operand A, how many cp.async
     operations are needed to issue in a single row.
     * @param n2, for operand B, ON AVERAGE, how many cp.async
     operations need to be issued in a single compute iteration.
     * @param N2, for the operand B, how many cp.async operations
     are needed to issue IN TOTAL by a single thread.
     * @param stride2, for the operand B, how many cp.async
     operations are needed to issue in a single row.
     */
    CUTE_DEVICE
    void copy2(int idx, int n1, int N1, int stride1, int n2, int N2,
               int stride2) {
        CUTE_UNROLL
        for (int i = 0; (i < n1) && (i + idx * n1 < N1); ++i) {
            int pos = i + idx * n1;
            int row = pos / stride1;
            int col = pos % stride1;
            cute::copy(tiled_copy, src1(_, row, col), dst1(_, row, col));
        }
        CUTE_UNROLL
        for (int i = 0; (i < n2) && (i + idx * n2 < N2); ++i) {
            int pos = i + idx * n2;
            int row = pos / stride2;
            int col = pos % stride2;
            cute::copy(tiled_copy, src2(_, row, col), dst2(_, row, col));
        }
    }

    template <int N>
    CUTE_DEVICE void wait_group() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
    }

    CUTE_DEVICE
    void commit_copy_group() { cute::cp_async_fence(); }

    CUTE_DEVICE
    void next() {
        src1.data() = src1.data() + src_stride1;
        src2.data() = src2.data() + src_stride2;

        dst1.data() = dst1.data() + dst_stride1;
        dst2.data() = dst2.data() + dst_stride2;
    }

    CUTE_DEVICE
    void cycle_dst() {
        dst1.data() = dst1.data() + (-dst_stride1 * num_stages);
        dst2.data() = dst2.data() + (-dst_stride2 * num_stages);
    }

  private:
    TiledCopy& tiled_copy;
    SrcTensor1& src1;
    const int src_stride1;
    DstTensor1& dst1;
    const int dst_stride1;
    SrcTensor2& src2;
    const int src_stride2;
    DstTensor2& dst2;
    const int dst_stride2;
    const int num_stages;
    int iter;
};

template <typename TiledCopy, typename SrcTensor, typename DstTensor>
CUTE_DEVICE void R2S_copy(TiledCopy& tiled_copy, SrcTensor& src, DstTensor& dst,
                          int tid) {
    auto copy_thrd = tiled_copy.get_thread_slice(threadIdx.x);
    auto src_copy_view = copy_thrd.retile_S(src);
    auto dst_thrd = copy_thrd.partition_D(dst);
    cute::copy(tiled_copy, src_copy_view, dst_thrd);
}

template <typename TiledCopy, typename SrcTensor, typename DstTensor>
CUTE_DEVICE void S2G_copy(TiledCopy& tiled_copy, SrcTensor& src, DstTensor& dst,
                          int tid) {
    auto copy_thrd = tiled_copy.get_thread_slice(tid);
    auto src_thrd = copy_thrd.partition_S(src);
    auto dst_thrd = copy_thrd.partition_D(dst);

    CUTE_UNROLL
    for (int i = 0; i < size<1>(dst_thrd); ++i) {
        CUTE_UNROLL
        for (int j = 0; j < size<2>(dst_thrd); ++j)
            cute::copy(tiled_copy, src_thrd(_, i, j), dst_thrd(_, i, j));
    }
}

}  // namespace cutlass_wrapper
}  // namespace benchmarks
