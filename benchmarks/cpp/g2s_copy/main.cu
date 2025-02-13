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
const int kRepeat = 100;

template <typename Element>
bool check_results(const Element* dst1, const Element* dst2, int64_t numel) {
    float epsilon = 1e-3;
    for (int i = 0; i < numel; ++i) {
        float v1 = abs(static_cast<float>(dst1[i]));
        float v2 = abs(static_cast<float>(dst2[i]));
        if (v1 - v2 > epsilon) {
            std::cerr << "Mismatch at " << i << ": " << v1 << " vs " << v2
                      << std::endl;
            return false;
        }
    }
    return true;
}

template <typename Element, typename Layout, typename WarpLayout,
          const int kRepeat>
float test_tilefusion(const Element* src, Element* dst) {
    using Global = GlobalTile<Element, Layout>;
    using Shared = SharedTile<Element, Layout, true /*kSwizzled*/>;

    using Loader = GlobalToSharedLoader<Shared, WarpLayout>;
    Loader loader;

    using Storer = SharedToGlobalStorer<Shared, WarpLayout>;
    Storer storer;

    auto kernel =
        &g2s_data_transfer<Element, Global, Shared, Loader, Storer, kRepeat>;

    static const int kThreads = WarpLayout::kNumel * 32;
    int shm_size = Shared::kNumel * sizeof(Element);

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    dim3 grids(1, 1, 1);
    dim3 blocks(kThreads);

    for (int i = 0; i < warmup; ++i)  // warm up
        kernel<<<grids, blocks, shm_size>>>(src, dst, loader, storer);
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i)
        kernel<<<grids, blocks, shm_size>>>(src, dst, loader, storer);
    cudaDeviceSynchronize();
    return timer.stop() / iters;
}

template <typename Element, typename Layout, typename WarpLayout,
          const int kRepeat>
float test_cutlass(const Element* src, Element* dst) {
    auto kernel = &cutlass_g2s_data_transfer<Element, Layout::kRows,
                                             Layout::kCols, WarpLayout::kRows,
                                             WarpLayout::kCols, kRepeat>;

    int shm_size = Layout::kNumel * sizeof(Element);
    int kThreads = WarpLayout::kNumel * 32;

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    dim3 grids(1, 1, 1);
    dim3 blocks(kThreads);

    for (int i = 0; i < warmup; ++i) {
        kernel<<<grids, blocks, shm_size>>>(src, dst);
    }
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        kernel<<<grids, blocks, shm_size>>>(src, dst);
    }
    cudaDeviceSynchronize();
    return timer.stop() / iters;
}

template <typename Element, typename Layout, typename WarpLayout>
void run_test_rowmajor() {
    int numel = Layout::kNumel;

    thrust::host_vector<Element> h_src(numel);
    for (int i = 0; i < h_src.size(); ++i)
        h_src[i] = static_cast<Element>(i % 2048);

    thrust::device_vector<Element> d_src = h_src;
    const Element* src = thrust::raw_pointer_cast(d_src.data());

    thrust::device_vector<Element> d_dst1(numel);
    thrust::fill(d_dst1.begin(), d_dst1.end(), static_cast<Element>(0.));
    Element* dst1 = thrust::raw_pointer_cast(d_dst1.data());

    thrust::device_vector<Element> d_dst2(numel);
    thrust::fill(d_dst2.begin(), d_dst2.end(), static_cast<Element>(0.));
    Element* dst2 = thrust::raw_pointer_cast(d_dst2.data());

    float t1 = test_tilefusion<Element, Layout, WarpLayout, kRepeat>(src, dst1);
    thrust::host_vector<Element> h_dst1 = d_dst1;

    float t2 = test_cutlass<Element, Layout, WarpLayout, kRepeat>(src, dst2);
    thrust::host_vector<Element> h_dst2 = d_dst2;

    bool passed = check_results(thrust::raw_pointer_cast(h_dst1.data()),
                                thrust::raw_pointer_cast(h_dst2.data()), numel);
    if (!passed) {
        std::cerr << "Test failed" << std::endl;
        return;
    }

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

    run_test_rowmajor<DType, tl::RowMajor<16, 64>, tl::RowMajor<1, 1>>();
    run_test_rowmajor<DType, tl::RowMajor<64, 64>, tl::RowMajor<1, 1>>();
    run_test_rowmajor<DType, tl::RowMajor<64, 64>, tl::RowMajor<2, 1>>();
    run_test_rowmajor<DType, tl::RowMajor<64, 64>, tl::RowMajor<4, 1>>();

    run_test_rowmajor<DType, tl::RowMajor<128, 128>, tl::RowMajor<1, 1>>();
    run_test_rowmajor<DType, tl::RowMajor<128, 128>, tl::RowMajor<2, 2>>();
    run_test_rowmajor<DType, tl::RowMajor<128, 128>, tl::RowMajor<4, 2>>();

    run_test_rowmajor<DType, tl::RowMajor<128, 256>, tl::RowMajor<1, 1>>();
    run_test_rowmajor<DType, tl::RowMajor<128, 256>, tl::RowMajor<2, 2>>();
    run_test_rowmajor<DType, tl::RowMajor<128, 256>, tl::RowMajor<2, 4>>();
    run_test_rowmajor<DType, tl::RowMajor<128, 256>, tl::RowMajor<4, 4>>();

    return 0;
}