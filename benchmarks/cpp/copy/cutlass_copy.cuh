// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
          const int kWarpRow, const int kWarpCol>
__global__ void cute_g2s(const Element* data) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    using GlobalLayout =
        cute::Layout<Shape<Int<kRows>, Int<kChunk>>, Stride<Int<kCols>, _1>>;

    /*
     * uncomment the following lines to evaluate different access patterns for
     * access to global memory
     */

    // const int kThreadRow = kWarpRow * 16;
    // const int kThreadCol = kWarpCol * 2;
    // using LayoutAtom = decltype(composition(  // row major
    //     Swizzle<2, 3, 3>{}, cute::Layout<Shape<_16, _16>, Stride<_16,
    //     _1>>{}));

    // const int kThreadRow = kWarpRow * 8;
    // const int kThreadCol = kWarpCol * 4;
    // using LayoutAtom = decltype(composition(  // row major
    //     Swizzle<2, 3, 3>{}, cute::Layout<Shape<_8, _32>, Stride<_32,
    //     _1>>{}));

    const int kThreadRow = kWarpRow * 4;
    const int kThreadCol = kWarpCol * 8;
    using LayoutAtom = decltype(composition(  // row major
        Swizzle<2, 3, 3>{}, cute::Layout<Shape<_4, _64>, Stride<_64, _1>>{}));

    using ThreadLayout = cute::Layout<Shape<Int<kThreadRow>, Int<kThreadCol>>,
                                      Stride<Int<kThreadCol>, _1>>;
    using SharedLayout = decltype(tile_to_shape(
        LayoutAtom{}, Shape<Int<kRows>, Int<kChunk>>{}, cute::Step<_2, _1>{}));

    // column major
    using ValueLayout = cute::Layout<Shape<_1, _8>>;

    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
    using TiledCopy =
        decltype(make_tiled_copy(CopyInst{}, ThreadLayout{}, ValueLayout{}));

    GlobalLayout src_layout;
    SharedLayout dst_layout;
    TiledCopy tiled_copy;

    for (int k = 0; k < kCols; k += kChunk) {
        copy_func(data + k, buf, src_layout, dst_layout, tiled_copy);

        __copy_async();
        __syncthreads();
    }
}
