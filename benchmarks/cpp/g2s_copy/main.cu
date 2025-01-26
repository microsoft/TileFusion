// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cell/copy/mod.hpp"
#include "cutlass_copy.cuh"
#include "tilefusion_copy.cuh"
#include "types/mod.hpp"
#include "util/cuda_timer.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace tilefusion;
using namespace tilefusion::cell;
using namespace tilefusion::cell::copy;

int warmup = 20;
int iters = 100;

template <typename Element, typename Layout, typename WarpLayout,
          const int kRepeat>
float test_tilefusion(const Element* data) {
    using Global = GlobalTile<Element, Layout>;
    using Shared = SharedTile<Element, Layout, true /*kSwizzled*/>;
    using Loader = GlobalToSharedLoader<Shared, WarpLayout>;
    Loader loader;

    auto kernel = &copy_g2s<Element, Global, Shared, Loader, kRepeat>;

    static const int kThreads = WarpLayout::kNumel * 32;
    int shm_size = Shared::kNumel * sizeof(Element);

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    dim3 grids(1, 1, 1);
    dim3 blocks(kThreads);

    for (int i = 0; i < warmup; ++i)  // warm up
        kernel<<<grids, blocks, shm_size>>>(data, loader);
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i)
        kernel<<<grids, blocks, shm_size>>>(data, loader);
    cudaDeviceSynchronize();
    return timer.stop() / iters;
}

template <typename Element, typename Layout, typename WarpLayout,
          const int kRepeat>
float test_cutlass(const Element* data) {
    auto kernel = &cute_g2s<Element, Layout::kRows, Layout::kCols,
                            WarpLayout::kRows, WarpLayout::kCols, kRepeat>;

    int shm_size = Layout::kNumel * sizeof(Element);
    int kThreads = WarpLayout::kNumel * 32;

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    dim3 grids(1, 1, 1);
    dim3 blocks(kThreads);

    for (int i = 0; i < warmup; ++i) {
        kernel<<<grids, blocks, shm_size>>>(data);
    }
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        kernel<<<grids, blocks, shm_size>>>(data);
    }
    cudaDeviceSynchronize();
    return timer.stop() / iters;
}

template <typename Element, typename Layout, typename WarpLayout>
void run_test_rowmajor() {
    int numel = Layout::kNumel;
    thrust::host_vector<Element> h_data(numel);

    for (int i = 0; i < h_data.size(); ++i) {
        h_data[i] = static_cast<Element>(i % 2048);
    }
    thrust::device_vector<Element> d_data = h_data;
    const Element* data = thrust::raw_pointer_cast(d_data.data());

    float t1 = 0., t2 = 0.;
    const int kRepeat = 100;

    t1 = test_tilefusion<Element, Layout, WarpLayout, kRepeat>(data);
    t2 = test_cutlass<Element, Layout, WarpLayout, kRepeat>(data);

    std::cout << "|RowMajor(" << Layout::kRows << ", " << Layout::kCols << ")|("
              << WarpLayout::kRows << ", " << WarpLayout::kCols << ")|" << t1
              << "|" << t2 << "|" << t1 / t2 << "|" << std::endl;
}

int main() {
    std::cout << std::setprecision(4)
              << "|Shape|Warp Layout|tilefusion(ms)|cutlass(ms)|Ratio|"
              << std::endl
              << "|:---|:---:|:---:|:---:|:---:|" << std::endl;

    using DType = __half;

    run_test_rowmajor<DType, tl::RowMajor<64, 64>, tl::RowMajor<1, 1>>();
    run_test_rowmajor<DType, tl::RowMajor<64, 64>, tl::RowMajor<2, 2>>();
    run_test_rowmajor<DType, tl::RowMajor<64, 64>, tl::RowMajor<2, 4>>();

    run_test_rowmajor<DType, tl::RowMajor<128, 128>, tl::RowMajor<1, 1>>();
    run_test_rowmajor<DType, tl::RowMajor<128, 128>, tl::RowMajor<2, 2>>();
    run_test_rowmajor<DType, tl::RowMajor<128, 128>, tl::RowMajor<2, 4>>();

    run_test_rowmajor<DType, tl::RowMajor<128, 256>, tl::RowMajor<1, 1>>();
    run_test_rowmajor<DType, tl::RowMajor<128, 256>, tl::RowMajor<2, 2>>();
    run_test_rowmajor<DType, tl::RowMajor<128, 256>, tl::RowMajor<2, 4>>();

    return 0;
}
