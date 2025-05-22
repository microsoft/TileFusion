// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"
#include "util/debug.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tilefusion::testing {
using namespace cell;
using namespace cell::copy;

namespace {
template <typename Global, typename Reg, typename Loader>
__global__ void load_g2r(Element* src) {
    Global src(src);
    Reg dst;

    loader(src, dst);
    __syncthreads();
}
}  // namespace

TEST(TestGlobalToRegCopy, copy_row_major) {
    // constants
    static constexpr int kRows = 64;
    static constexpr int kCols = 128;

    using DType = __half;
    using WarpLayout = tl::RowMajor<1, 1>;
    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
    const WarpReuse kMode = WarpReuse::kCont;

    // define tiles
    using Global = GlobalTile<DType, tl::RowMajor<kRows, kCols>>;
    using Reg = RegTile<BaseTileRowMajor<DType>, tl::RowMajor<2, 2>>;

    // loader
    using Loader = GlobalToRegLoader<Reg, WarpLayout, kMode>;
    Loader loader;

    // generate data
    int numel = kRows * kCols;
    thrust::host_vector<DType> h_data(numel);
    for (int i = 0; i < h_data.size(); ++i)
        h_data[i] = static_cast<DType>(i % 2048);
    thrust::device_vector<DType> d_data = h_data;

    auto copy_kernel = copy_g2s<Element, SrcTile, DstTile, Loader, Storer>;
    dim3 dim_grid(1, 1);
    dim3 dim_block(kThreads);
    load_g2r<<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(d_data.data()));
    cudaDeviceSynchronize();
}

}  // namespace tilefusion::testing
