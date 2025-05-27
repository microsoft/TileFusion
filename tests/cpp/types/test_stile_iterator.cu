// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <glog/logging.h>

#include <iostream>

namespace tilefusion::testing {
using namespace cell;
namespace tl = tile_layout;

namespace {
/// @brief Initialize buffer with sequential values for testing
template <typename DType>
__device__ void init_buf(DType* buf, int numel) {
    for (int i = 0; i < numel; ++i) {
        buf[i] = static_cast<DType>(i);
    }
}

/// @brief Test kernel for shared tile iterator
template <typename Shared, typename SIterator>
__global__ void test_stile_iterator() {
    using DType = typename Shared::DType;
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    DType* buf = reinterpret_cast<DType*>(buf_);

    init_buf(buf, Shared::kNumel);

    Shared s_tile(buf);
    SIterator s_itr(&s_tile);

    printf("shared tile:\n");
    s_tile.dump_value();

    for (int i = 0; i < SIterator::sc1; ++i) {
        printf("\nsub-tile %d:\n", i);
        auto tile = s_itr(i);
        tile.dump_value();
    }
}
}  // namespace

// TEST(TestSharedTileIterator, row_major) {
//     using InType = __half;

//     static constexpr int kRows = 8;
//     static constexpr int kCols = 16;

//     static constexpr int kChunkRows = 4;
//     static constexpr int kChunkCols = 4;

//     using SharedLayout = tl::RowMajor<kRows, kCols>;

//     using Shared = SharedTile<InType, SharedLayout>;
//     using SIterator = STileIterator2<Shared, TileShape<kChunkRows,
//     kChunkCols>>;

//     LOG(INFO) << std::endl << Shared{} << std::endl;
//     LOG(INFO) << std::endl << SIterator{} << std::endl;

//     int shm_size = Shared::kNumel * sizeof(InType);
//     dim3 blocks(1, 1, 1);
//     dim3 threads(1, 1, 1);
//     test_stile_iterator<Shared, SIterator><<<blocks, threads, shm_size>>>();
//     cudaDeviceSynchronize();
// }

TEST(TestSharedTileIterator, block_row_major) {
    using InType = __half;
    static constexpr int kRows = 4;
    static constexpr int kCols = 16;

    using SharedLayout =
        tl::BlockRowMajor<tl::RowMajor<kRows, kCols>, tl::RowMajor<2, 4>>;

    std::cout << "SharedLayout: " << std::endl << SharedLayout{} << std::endl;

    using Shared = SharedTile<InType, SharedLayout>;
    using SIterator = STileIterator2<Shared, TileShape<4, 8>>;

    LOG(INFO) << std::endl << Shared{} << std::endl;
    LOG(INFO) << std::endl << SIterator{} << std::endl;

    using SubTileLayout = SubTileLayoutCreator<SharedLayout, 4, 8>;
    using Layout = typename SubTileLayout::type;

    LOG(INFO) << std::endl
              << "SubTileLayout: " << std::endl
              << Layout{} << std::endl;

    int shm_size = Shared::kNumel * sizeof(InType);
    dim3 blocks(1, 1, 1);
    dim3 threads(1, 1, 1);
    test_stile_iterator<Shared, SIterator><<<blocks, threads, shm_size>>>();
    cudaDeviceSynchronize();
}

// TEST(TestSharedTileIterator, block_swizzled_row_major) {
//     using InType = __half;
//     static constexpr int kRows = 8;
//     static constexpr int kCols = 16;

//     static constexpr int kChunkRows = 4;
//     static constexpr int kChunkCols = 4;

//     using SharedLayout =
//         tl::BlockRowMajor<tl::RowMajor<kRows, kCols>,
//                           SwizzledLayout<tl::RowMajor<2, 4>, Swizzle<1, 0,
//                           2>>>;

//     using Shared = SharedTile<InType, SharedLayout>;
//     using SIterator = STileIterator2<Shared, TileShape<kChunkRows,
//     kChunkCols>>;

//     LOG(INFO) << std::endl << Shared{} << std::endl;
//     LOG(INFO) << std::endl << SIterator{} << std::endl;

//     int shm_size = Shared::kNumel * sizeof(InType);
//     dim3 blocks(1, 1, 1);
//     dim3 threads(1, 1, 1);
//     test_stile_iterator<Shared, SIterator><<<blocks, threads, shm_size>>>();
//     cudaDeviceSynchronize();
// }
}  // namespace tilefusion::testing
