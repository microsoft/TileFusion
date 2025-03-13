// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/mod.hpp"

namespace tilefusion::testing {
using namespace cell;
namespace tl = tile_layout;

template <int kM, int kN, int kK, int kTM, int kTN, int kTK>
__global__ void test_thread_block_swizzle() {
    GemmThreadBlockSwizzle<kM, kN, kTM, kTN> tb_swizzle;

    auto offset = tb_swizzle.get_tile_offset();

    int cta_m = kM / kTM;
    int cta_n = kN / kTN;

    int block_m = cta_m / gridDim.y;

    int offset_A =
        offset.y * block_m * kTM * kK + (offset.x / cta_n) * kTM * kK;
    int offset_B = (offset.x % cta_n) * kTN * kK;
    int offset_C = offset.y * block_m * kTM * kN + offset.x * kTN;

    if (threadIdx.x == 0) {
        printf(
            "blockIdx: (%d, %d, %d), offset_A: %d, offset_B: %d, offset_C: "
            "%d\n",
            blockIdx.x, blockIdx.y, blockIdx.z, offset_A, offset_B, offset_C);
    }
}

TEST(TESTGemmThreadBlockSwizzle, test_gemm_thread_block_swizzle) {
    static constexpr int kM = 4096;
    static constexpr int kK = 4096;
    static constexpr int kN = 4096;
    static constexpr int kTM = 128;
    static constexpr int kTK = 128;
    static constexpr int kTN = 128;

    GemmThreadBlockSwizzle<kM, kN, kTM, kTN> tb_swizzle;
    dim3 grid_dim = tb_swizzle.get_grid_shape();
    dim3 block_dim(128, 1, 1);

    printf("grid_dim: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);

    test_thread_block_swizzle<kM, kN, kK, kTM, kTN, kTK>
        <<<grid_dim, block_dim>>>();
    cudaDeviceSynchronize();
}

}  // namespace tilefusion::testing
