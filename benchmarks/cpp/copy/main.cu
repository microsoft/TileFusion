// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cell/copy/global_to_shared_2.hpp"
#include "cell/copy/mod.hpp"
#include "cell/sync.hpp"
#include "cutlass_copy.cuh"
#include "types/mod.hpp"
#include "util/cuda_timer.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace tilefusion;
using namespace tilefusion::cell;
using namespace tilefusion::cell::copy;

template <typename Element, typename Global, typename Shared,
          typename GIterator, typename Loader>
__global__ void copy_g2s(const Element* data, Loader& loader) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    static const int kStride = Shared::kNumel;
    // static constexpr int kCount = kCols / kChunk;

    // GIterator g_tiles(data);
    Shared s_tile(buf);

    // for (int i = 0; i < GIterator::sc1; ++i) {

    for (int i = 0; i < 64; ++i) {
        Global g_tile(data + i * kStride);
        loader(g_tile, s_tile);
        // loader(g_tiles(i), s_tile);
        __copy_async();
        __syncthreads();
    }
}

template <typename Element, typename Global, typename Shared,
          typename GTileIterator, typename WarpLayout, typename Loader>
float test_load(const Element* data) {
    static const int kThreads = tl::get_numel<WarpLayout> * 32;
    int shm_size = Shared::kRows * Shared::kCols * sizeof(Element);

    dim3 grids(1, 1, 1);
    dim3 blocks(kThreads);
    auto kernel = &copy_g2s<Element, Global, Shared, GTileIterator, Loader>;

    Loader loader;

    for (int i = 0; i < 5; ++i)  // warm up
        kernel<<<grids, blocks, shm_size>>>(data, loader);
    cudaDeviceSynchronize();

    int iters = 50;
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i)
        kernel<<<grids, blocks, shm_size>>>(data, loader);
    cudaDeviceSynchronize();
    return timer.stop() / iters;
}

template <typename Element, const int kRows, const int kCols, const int kChunk,
          typename WarpLayout>
float test_cutlass(const Element* data) {
    auto test = &cute_g2s<Element, kRows, kCols, kChunk,
                          tl::num_rows<WarpLayout>, tl::num_cols<WarpLayout>>;

    int shm_size = kRows * kChunk * sizeof(Element);
    int kThreads = tl::get_numel<WarpLayout> * 32;
    dim3 grids(1, 1, 1);
    dim3 blocks(kThreads);

    for (int i = 0; i < 5; ++i) {
        test<<<grids, blocks, shm_size>>>(data);
    }
    cudaDeviceSynchronize();

    int iters = 50;
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        test<<<grids, blocks, shm_size>>>(data);
    }
    cudaDeviceSynchronize();
    return timer.stop() / iters;
}

template <typename Element, const int kRows, const int kCols, const int kChunk>
void run_test() {
    // initialize data
    static const bool kSwizzled = true;
    int numel = kRows * kCols;
    thrust::host_vector<Element> h_data(numel);

    for (int i = 0; i < h_data.size(); ++i) {
        h_data[i] = static_cast<Element>(i % 2048);
    }
    thrust::device_vector<Element> d_data = h_data;
    const Element* data = thrust::raw_pointer_cast(d_data.data());

    using Global = GlobalTile<Element, tl::RowMajor<kRows, kCols>>;
    using GIterator = GTileIterator<Global, TileShape<kRows, kChunk>>;
    using Shared = SharedTile<Element, tl::RowMajor<kRows, kChunk>, kSwizzled>;

    using GlobalFake = GlobalTile<Element, tl::RowMajor<kRows, kChunk>>;
    using WarpLayout = tl::RowMajor<2, 2>;

    std::cout << "GIterator: " << GIterator{} << std::endl;

    using Loader1 = copy::GlobalToSharedLoader<Shared, WarpLayout>;
    using Loader2 = copy::GlobalToSharedLoader2<Shared, WarpLayout>;

    auto test1 =
        &test_load<Element, GlobalFake, Shared, GIterator, WarpLayout, Loader1>;
    auto test2 =
        &test_load<Element, GlobalFake, Shared, GIterator, WarpLayout, Loader2>;
    float time1 = test1(data);
    float time2 = test2(data);
    float time3 = test_cutlass<Element, kRows, kCols, kChunk, WarpLayout>(data);

    std::cout << std::setprecision(4) << "|ours(ms)|cute(ms)|cutlass(ms)|"
              << std::endl
              << "|:---:|:---:|:---:|" << std::endl
              << "|" << time1 << "|" << time2 << "|" << time3 << "|"
              << std::endl;
}

int main() {
    run_test<__half, 128, 128 * 64, 128 /*chunk shape*/>();
    return 0;
}
