// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/mod.hpp"

namespace tilefusion::testing {
using namespace cell;
namespace tl = tile_layout;

namespace {
int flatten(int x, int y, int width) { return x * width + y; }

template <const int kB, const int kM, const int kS>
int swizzle_ref(int x, int y) {
    int b = x;
    int s = y >> kM;

    int swizzled_s = b ^ s;
    int swizzle_idx =
        (b << (kM + kS)) | (swizzled_s << kM) | (y & ((1 << kM) - 1));

    return swizzle_idx;
}

template <const int kB, const int kM, const int kS>
int2 test_swizzle(int x, int y) {
    Swizzle<kB, kM, kS> swizzle;
    int idx = flatten(x, y, 1 << (kS + kM));
    int swizzled_idx = swizzle(idx);

    int ref_swizzled_idx = swizzle_ref<kB, kM, kS>(x, y);

#ifdef DEBUG
    printf("idx: %d, swizzled_idx: %d, ref_swizzled_idx: %d\n", idx,
           swizzled_idx, ref_swizzled_idx);
#endif

    return make_int2(swizzled_idx, ref_swizzled_idx);
}

template <typename Layout>
void test_swizzled_layout(Layout layout) {
    const int kB = 3;
    const int kM = 3;
    const int kS = 3;

    using SwizzledLayout = SwizzledLayout<Layout, kB, kM, kS>;
    SwizzledLayout swizzled;

    int swizzled_idx, swizzled_i, swizzled_j;
    for (int i = 0; i < Layout::kRows; ++i) {
        for (int j = 0; j < Layout::kCols; ++j) {
            swizzled_idx = swizzled(i, j);

            swizzled_i = swizzled_idx / Layout::kCols;
            swizzled_j = swizzled_idx % Layout::kCols;

#if defined DEBUG
            std::cout << "(" << swizzled_i << "," << swizzled_j << "), ";
#endif
        }
#if defined DEBUG
        std::cout << std::endl;
#endif
    }

    std::cout << std::endl;
}
}  // namespace

TEST(TestSwizzleFunction, test_swizzle_apply) {
    const int kB = 3;
    const int kM = 3;
    const int kS = 3;

    int2 swizzled_idx_0_0 = test_swizzle<kB, kM, kS>(0, 0);
    int2 swizzled_idx_1_0 = test_swizzle<kB, kM, kS>(1, 0);
    int2 swizzled_idx_1_4 = test_swizzle<kB, kM, kS>(1, 4);
    int2 swizzled_idx_2_0 = test_swizzle<kB, kM, kS>(2, 0);
    int2 swizzled_idx_2_4 = test_swizzle<kB, kM, kS>(2, 4);

    EXPECT_EQ(swizzled_idx_0_0.x, swizzled_idx_0_0.y);
    EXPECT_EQ(swizzled_idx_1_0.x, swizzled_idx_1_0.y);
    EXPECT_EQ(swizzled_idx_1_4.x, swizzled_idx_1_4.y);
    EXPECT_EQ(swizzled_idx_2_0.x, swizzled_idx_2_0.y);
    EXPECT_EQ(swizzled_idx_2_4.x, swizzled_idx_2_4.y);
}

TEST(TestSwizzledLayout, test_row_major) {
    // FIXME(ying): Add meaningful test case instead of printing the result.

    std::cout << "test_16x16_half" << std::endl;
    // In the 16x16 shaped tile, elements in every 4 rows are permuted.
    test_swizzled_layout(tl::RowMajor<16, 16>{});

    std::cout << "test_8x32_half" << std::endl;
    // In the 8x32 shaped tile, elements in every 2 rows are permuted.
    test_swizzled_layout(tl::RowMajor<8, 32>{});

    std::cout << "test_4x64_half" << std::endl;
    // In the 4x64 shaped tile, elements in every single row are permuted.
    test_swizzled_layout(tl::RowMajor<4, 64>{});
}

TEST(TestSwizzledLayout, test_col_major) {
    // FIXME(ying): Add meaningful test case instead of printing the result.

    std::cout << "test_16x16_half" << std::endl;
    // In the 16x16 shaped tile, elements in every 4 columns are permuted.
    test_swizzled_layout(tl::ColMajor<16, 16>{});

    std::cout << "test_32x8_half" << std::endl;
    // In the 32x8 shaped tile, elements in every 2 columns are permuted.
    test_swizzled_layout(tl::ColMajor<32, 8>{});

    std::cout << "test_64x4_half" << std::endl;
    // In the 4x64 shaped tile, elements in every single column are permuted.
    test_swizzled_layout(tl::ColMajor<64, 4>{});
}

}  // namespace tilefusion::testing
