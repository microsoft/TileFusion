// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cell/sync.hpp"

#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>

using namespace cute;
using namespace tilefusion::cell;

namespace {
template <typename Element,                  //
          const int kRows, const int kCols,  //
          const int kWarpRow, const int kWarpCol>
struct Loader {
    DEVICE void operator()(const Element* src_, Element* dst_) {
        int tid = threadIdx.x;

        auto gtile = make_tensor(make_gmem_ptr(src_), src_layout_);
        auto stile = make_tensor(make_smem_ptr(dst_), dst_layout_);

        auto loader = tiled_copy_.get_thread_slice(tid);

        auto src = loader.partition_S(gtile);
        auto dst = loader.partition_D(stile);

#pragma unroll
        for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
            for (int j = 0; j < int(size<2>(src)); ++j)
                cute::copy(tiled_copy_, src(cute::_, i, j), dst(cute::_, i, j));
    }

  private:
    // source
    using GlobalLayout =
        cute::Layout<Shape<Int<kRows>, Int<kCols>>, Stride<Int<kCols>, _1>>;
    GlobalLayout src_layout_;

    // destination
    using LayoutAtom =
        decltype(composition(cute::Swizzle<2, 3, 3>{},
                             cute::Layout<Shape<_4, _64>, Stride<_64, _1>>{}));
    using SharedLayout = decltype(tile_to_shape(
        LayoutAtom{}, Shape<Int<kRows>, Int<kCols>>{}, cute::Step<_2, _1>{}));
    SharedLayout dst_layout_;

    // tiled copy
    static constexpr int kThreadRow = kWarpRow * 4;
    static constexpr int kThreadCol = kWarpCol * 8;
    using ThreadLayout = cute::Layout<Shape<Int<kThreadRow>, Int<kThreadCol>>,
                                      Stride<Int<kThreadCol>, _1>>;
    using ValueLayout = cute::Layout<Shape<_1, _8>>;

    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
    using TiledCopy =
        decltype(make_tiled_copy(CopyInst{}, ThreadLayout{}, ValueLayout{}));
    TiledCopy tiled_copy_;
};

template <typename Element,                  //
          const int kRows, const int kCols,  //
          const int kWarpRow, const int kWarpCol>
struct Storer {
    DEVICE void operator()(const Element* src_, Element* dst_) {
        int tid = threadIdx.x;

        auto stile = make_tensor(make_smem_ptr(src_), src_layout_);
        auto gtile = make_tensor(make_gmem_ptr(dst_), dst_layout_);

        auto loader = tiled_copy_.get_thread_slice(tid);

        auto src = loader.partition_S(stile);
        auto dst = loader.partition_D(gtile);

#pragma unroll
        for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
            for (int j = 0; j < int(size<2>(src)); ++j)
                cute::copy(tiled_copy_, src(cute::_, i, j), dst(cute::_, i, j));
    }

  private:
    // source
    using LayoutAtom =
        decltype(composition(cute::Swizzle<2, 3, 3>{},
                             cute::Layout<Shape<_4, _64>, Stride<_64, _1>>{}));
    using SharedLayout = decltype(tile_to_shape(
        LayoutAtom{}, Shape<Int<kRows>, Int<kCols>>{}, cute::Step<_2, _1>{}));
    SharedLayout src_layout_;

    // destination
    using GlobalLayout =
        cute::Layout<Shape<Int<kRows>, Int<kCols>>, Stride<Int<kCols>, _1>>;
    GlobalLayout dst_layout_;

    // tiled copy
    static constexpr int kThreadRow = kWarpRow * 4;
    static constexpr int kThreadCol = kWarpCol * 8;
    using ThreadLayout = cute::Layout<Shape<Int<kThreadRow>, Int<kThreadCol>>,
                                      Stride<Int<kThreadCol>, _1>>;
    using ValueLayout = cute::Layout<Shape<_1, _8>>;

    using CopyInst = Copy_Atom<DefaultCopy, Element>;
    using TiledCopy =
        decltype(make_tiled_copy(CopyInst{}, ThreadLayout{}, ValueLayout{}));
    TiledCopy tiled_copy_;
};
}  // namespace

template <typename Element, const int kRows, const int kCols,
          const int kWarpRow, const int kWarpCol, const int kRepeat>
__global__ void cutlass_g2s_data_transfer(const Element* src, Element* dst) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    using G2S = Loader<Element, kRows, kCols, kWarpRow, kWarpCol>;
    G2S loader;

    using S2G = Storer<Element, kRows, kCols, kWarpRow, kWarpCol>;
    S2G storer;

    for (int k = 0; k < kRepeat; ++k) {
        loader(src, buf);

        __copy_async();
        __syncthreads();

        storer(buf, dst);
        __syncthreads();
    }
}
