// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/mod.hpp"

namespace tilefusion::testing {
using namespace cell;
namespace tl = tile_layout;

template <int kM, int kN, int kTM, int kTN>
__global__ void test_thread_block_swizzle() {
    GemmThreadBlockSwizzle<kM, kN, kTM, kTN> tb_swizzle;

    auto offset = tb_swizzle.get_tile_offset();

    if (threadIdx.x == 0) {
        printf("grid: (%d, %d, %d), offset: (%d, %d, %d)\n", blockIdx.x,
               blockIdx.y, blockIdx.z, offset.x, offset.y, offset.z);
    }
}

TEST(TESTGemmThreadBlockSwizzle, test_gemm_thread_block_swizzle) {
    static constexpr int kM = 4096;
    static constexpr int kN = 4096;
    static constexpr int kTM = 128;
    static constexpr int kTN = 128;

    GemmThreadBlockSwizzle<kM, kN, kTM, kTN> tb_swizzle;
    dim3 grid_dim = tb_swizzle.get_grid_shape();
    dim3 block_dim(128, 1, 1);

    printf("grid_dim: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);

    test_thread_block_swizzle<kM, kN, kTM, kTN><<<grid_dim, block_dim>>>();
    cudaDeviceSynchronize();
}

}  // namespace tilefusion::testing
