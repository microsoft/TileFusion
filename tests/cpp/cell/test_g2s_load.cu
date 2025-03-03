// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cell/mod.hpp"
#include "common/test_utils.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tilefusion::testing {
using namespace cell;
using namespace copy::warp;
namespace tl = tile_layout;

namespace {
template <typename Element, typename SrcTile, typename DstTile, typename Loader,
          typename Storer>
__global__ void copy_g2s(const Element* src_ptr, Element* dst_ptr,
                         Loader& loader, Storer& storer) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    SrcTile src(src_ptr);  // global memory tile
    DstTile inter(buf);    // shared memory tile
    SrcTile dst(dst_ptr);  // global memory tile

    loader(src, inter);
    copy::__copy_async();
    __syncthreads();

    storer(inter, dst);
    __syncthreads();

#if defined(DEBUG)
    if (thread(0)) {
        printf("\nshared\n");
        inter.dump_value();

        printf("\nglobal\n");
        dst.dump_value();
        printf("\n");
    }
#endif
}

template <typename Element, typename WarpLayout, const int kRows,
          const int kCols, const bool kSwizzled = false>
void run_test_row_major() {
    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    int numel = kRows * kCols;
    thrust::host_vector<Element> h_A(numel);
    for (int i = 0; i < h_A.size(); ++i)
        h_A[i] = static_cast<Element>(i % 2048);

    thrust::device_vector<Element> d_B(numel);
    thrust::fill(d_B.begin(), d_B.end(), static_cast<Element>(0.));
    thrust::device_vector<Element> d_A = h_A;

    using SrcTile = GlobalTile<Element, tl::RowMajor<kRows, kCols>>;
    using DstTile = SharedTile<Element, tl::RowMajor<kRows, kCols>, kSwizzled>;

    using Loader = copy::GlobalToSharedLoader<DstTile, WarpLayout>;
    Loader loader;

    using Storer = copy::SharedToGlobalStorer<DstTile, WarpLayout>;
    Storer storer;

    auto copy_kernel = copy_g2s<Element, SrcTile, DstTile, Loader, Storer>;

    int shm_size = kRows * kCols * sizeof(Element);
    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            copy_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    dim3 dim_grid(1, 1);
    dim3 dim_block(kThreads);
    copy_kernel<<<dim_grid, dim_block, shm_size>>>(
        thrust::raw_pointer_cast(d_A.data()),
        thrust::raw_pointer_cast(d_B.data()), loader, storer);
    cudaDeviceSynchronize();

    thrust::host_vector<Element> h_B(numel);
    h_B = d_B;

    assert_equal(
        reinterpret_cast<Element*>(thrust::raw_pointer_cast(h_A.data())),
        reinterpret_cast<Element*>(thrust::raw_pointer_cast(h_B.data())), numel,
        1e-5);
}

template <typename Element, typename WarpLayout, const int kRows,
          const int kCols, const bool kSwizzled = false>
void run_test_col_major() {
    static const int kThreads = tl::get_numel<WarpLayout> * 32;

    int numel = kRows * kCols;
    thrust::host_vector<Element> h_A(numel);
    for (int i = 0; i < h_A.size(); ++i)
        h_A[i] = static_cast<Element>(i % 2048);

    thrust::device_vector<Element> d_B(numel);
    thrust::fill(d_B.begin(), d_B.end(), static_cast<Element>(0.));
    thrust::device_vector<Element> d_A = h_A;

    using SrcTile = GlobalTile<Element, tl::ColMajor<kRows, kCols>>;
    using DstTile = SharedTile<Element, tl::ColMajor<kRows, kCols>, kSwizzled>;

    using Loader = copy::GlobalToSharedLoader<DstTile, WarpLayout>;
    Loader loader;

    using Storer = copy::SharedToGlobalStorer<DstTile, WarpLayout>;
    Storer storer;

    dim3 dim_grid(1, 1);
    dim3 dim_block(kThreads);

    auto copy_kernel = copy_g2s<Element, SrcTile, DstTile, Loader, Storer>;

    int shm_size = kRows * kCols * sizeof(Element);
    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            copy_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    copy_kernel<<<dim_grid, dim_block, shm_size>>>(
        thrust::raw_pointer_cast(d_A.data()),
        thrust::raw_pointer_cast(d_B.data()), loader, storer);
    cudaDeviceSynchronize();

    thrust::host_vector<Element> h_B(numel);
    h_B = d_B;

    assert_equal(
        reinterpret_cast<Element*>(thrust::raw_pointer_cast(h_A.data())),
        reinterpret_cast<Element*>(thrust::raw_pointer_cast(h_B.data())), numel,
        1e-5);
}
}  // namespace

TEST(GlobalToSharedLoad, test_row_major_load) {
    {  // test non-swizzled __half.
        static constexpr bool kSwizzled = false;

        run_test_row_major<__half, tl::RowMajor<1, 1>, 16, 64, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<1, 1>, 16, 256, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<2, 1>, 64, 64, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<1, 4>, 16, 256, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<4, 1>, 64, 64, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<2, 2>, 32, 128, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<2, 4>, 32, 256, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<2, 4>, 64, 512, kSwizzled>();

        // Swizzle<2, 3, 3>
        run_test_row_major<__half, tl::RowMajor<1, 1>, 16, 32, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<1, 1>, 32, 32, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<2, 2>, 32, 64, kSwizzled>();
    }

    {  // test swizzled __half.
        static constexpr bool kSwizzled = true;

        run_test_row_major<__half, tl::RowMajor<1, 1>, 16, 64, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<1, 1>, 16, 256, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<2, 1>, 64, 64, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<1, 4>, 16, 256, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<4, 1>, 64, 64, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<2, 2>, 32, 128, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<2, 4>, 32, 256, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<2, 4>, 64, 512, kSwizzled>();

        // Swizzle<2, 3, 3>
        run_test_row_major<__half, tl::RowMajor<1, 1>, 16, 32, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<1, 1>, 32, 32, kSwizzled>();
        run_test_row_major<__half, tl::RowMajor<2, 2>, 32, 64, kSwizzled>();
    }

    {  // test non-swizzled float.
        static constexpr bool kSwizzled = false;

        run_test_row_major<float, tl::RowMajor<1, 1>, 8, 32, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<1, 1>, 16, 64, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<1, 4>, 16, 128, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<4, 1>, 64, 32, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<2, 2>, 32, 64, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<2, 4>, 32, 128, kSwizzled>();

        // Swizzle<2, 3, 3>
        run_test_row_major<float, tl::RowMajor<1, 1>, 8, 16, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<1, 1>, 16, 16, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<2, 2>, 16, 32, kSwizzled>();
    }

    {  // test swizzled float.
        static constexpr bool kSwizzled = true;

        run_test_row_major<float, tl::RowMajor<1, 1>, 8, 32, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<1, 1>, 16, 64, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<1, 4>, 16, 128, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<4, 1>, 64, 32, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<2, 2>, 32, 64, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<2, 4>, 32, 128, kSwizzled>();

        // Swizzle<2, 3, 3>
        run_test_row_major<float, tl::RowMajor<1, 1>, 8, 16, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<1, 1>, 16, 16, kSwizzled>();
        run_test_row_major<float, tl::RowMajor<2, 2>, 16, 32, kSwizzled>();
    }
}

TEST(GlobalToSharedLoad, test_col_major_load) {
    {
        static constexpr bool kSwizzled = false;

        run_test_col_major<__half, tl::RowMajor<1, 1>, 64, 16, kSwizzled>();
        run_test_col_major<__half, tl::RowMajor<1, 1>, 128, 16, kSwizzled>();
        run_test_col_major<__half, tl::RowMajor<1, 4>, 64, 128, kSwizzled>();
        run_test_col_major<__half, tl::RowMajor<4, 1>, 256, 16, kSwizzled>();
        run_test_col_major<__half, tl::RowMajor<2, 2>, 128, 32, kSwizzled>();

        // Swizzle<2, 3, 3>
        run_test_col_major<__half, tl::RowMajor<1, 1>, 32, 16, kSwizzled>();
        run_test_col_major<__half, tl::RowMajor<1, 1>, 32, 32, kSwizzled>();
        run_test_col_major<__half, tl::RowMajor<2, 2>, 64, 32, kSwizzled>();
    }

    {
        static constexpr bool kSwizzled = true;
        run_test_col_major<__half, tl::RowMajor<1, 1>, 64, 16, kSwizzled>();
        run_test_col_major<__half, tl::RowMajor<1, 1>, 128, 16, kSwizzled>();
        run_test_col_major<__half, tl::RowMajor<1, 4>, 64, 128, kSwizzled>();
        run_test_col_major<__half, tl::RowMajor<4, 1>, 256, 32, kSwizzled>();
        run_test_col_major<__half, tl::RowMajor<2, 2>, 128, 32, kSwizzled>();

        // Swizzle<2, 3, 3>
        run_test_col_major<__half, tl::RowMajor<1, 1>, 32, 16, kSwizzled>();
        run_test_col_major<__half, tl::RowMajor<1, 1>, 32, 32, kSwizzled>();
        run_test_col_major<__half, tl::RowMajor<2, 2>, 64, 32, kSwizzled>();
    }

    {
        static constexpr bool kSwizzled = false;

        run_test_col_major<float, tl::RowMajor<1, 1>, 32, 16, kSwizzled>();
        run_test_col_major<float, tl::RowMajor<1, 1>, 64, 16, kSwizzled>();
        run_test_col_major<float, tl::RowMajor<1, 4>, 64, 64, kSwizzled>();
        run_test_col_major<float, tl::RowMajor<4, 1>, 128, 32, kSwizzled>();
        run_test_col_major<float, tl::RowMajor<2, 2>, 64, 64, kSwizzled>();

        // Swizzle<2, 3, 3>
        run_test_col_major<float, tl::RowMajor<1, 1>, 16, 16, kSwizzled>();
        run_test_col_major<float, tl::RowMajor<1, 1>, 16, 32, kSwizzled>();
        run_test_col_major<float, tl::RowMajor<2, 2>, 32, 32, kSwizzled>();
    }

    {
        static constexpr bool kSwizzled = true;
        run_test_col_major<float, tl::RowMajor<1, 1>, 128, 128, kSwizzled>();

        // Swizzle<2, 3, 3>
        run_test_col_major<float, tl::RowMajor<1, 1>, 16, 16, kSwizzled>();
        run_test_col_major<float, tl::RowMajor<1, 1>, 16, 32, kSwizzled>();
        run_test_col_major<float, tl::RowMajor<2, 2>, 32, 32, kSwizzled>();
    }
}
}  // namespace tilefusion::testing
