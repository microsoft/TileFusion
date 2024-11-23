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

}  // namespace cutlass_wrapper
}  // namespace benchmarks
