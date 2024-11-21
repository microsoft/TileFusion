#pragma once

#include "cell/sync.hpp"

#include <cute/tensor.hpp>
using namespace cute;
using namespace tilefusion::cell;

template <typename Element, typename SrcLayout, typename DstLayout,
          typename TiledCopy>
DEVICE void copy_func(const Element* src_, Element* dst_, SrcLayout src_layout,
                      DstLayout dst_layout, TiledCopy tiled_copy) {
    int tid = threadIdx.x;

    auto gtile = make_tensor(make_gmem_ptr(src_), src_layout);
    auto stile = make_tensor(make_smem_ptr(dst_), dst_layout);

    auto loader = tiled_copy.get_thread_slice(tid);

    auto src = loader.partition_S(gtile);
    auto dst = loader.partition_D(stile);

#pragma unroll
    for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
        for (int j = 0; j < int(size<2>(src)); ++j)
            cute::copy(tiled_copy, src(cute::_, i, j), dst(cute::_, i, j));
}

template <typename Element, const int kRows, const int kCols, const int kChunk,
          const int kWarpPerRow, const int kWarpPerCol>
__global__ void cute_g2s(const Element* data) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    static const int kStride = kChunk * kRows;
    static constexpr int kCount = kCols / kChunk;
    static constexpr int kNumPerAccess = 8;
    static constexpr int kThreadsRows = kWarpPerRow * 16;
    static constexpr int kThreadsCols = kWarpPerCol * 2;

    using GlobalLayout =
        cute::Layout<Shape<Int<kRows>, Int<kChunk>>, Stride<Int<kChunk>, _1>>;

    using LayoutAtom = decltype(composition(
        Swizzle<2, 3, 3>{}, cute::Layout<Shape<_16, _16>, Stride<_16, _1>>{}));
    using SharedLayout = decltype(tile_to_shape(
        LayoutAtom{}, Shape<Int<kRows>, Int<kChunk>>{}, cute::Step<_2, _1>{}));

    using ThreadLayout =
        cute::Layout<Shape<Int<kThreadsRows>, Int<kThreadsCols>>,
                     Stride<Int<kThreadsCols>, _1>>;
    using ValueLayout = cute::Layout<Shape<_1, Int<kNumPerAccess>>>;

    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
    using TiledCopy =
        decltype(make_tiled_copy(CopyInst{}, ThreadLayout{}, ValueLayout{}));

    GlobalLayout src_layout;
    SharedLayout dst_layout;
    TiledCopy tiled_copy;

    for (int i = 0; i < kCount; ++i) {
        copy_func(data + i * kStride, buf, src_layout, dst_layout, tiled_copy);

        __copy_async();
        __syncthreads();
    }
}
