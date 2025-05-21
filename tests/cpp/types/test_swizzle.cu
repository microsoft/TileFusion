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
}  // namespace

TEST(TestSwizzle, test_swizzle_function) {
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

TEST(TestSwizzle, test_swizzled_layout) {
    using BlockRowMajor = tl::BlockRowMajor<
        tl::RowMajor<16, 64>,
        SwizzledLayout<tl::RowMajor<8, 64>, Swizzle<3, 3, 3>>>;

#if defined(DEBUG)
    BlockRowMajor layout;
    layout.dump();
#endif
}

}  // namespace tilefusion::testing
