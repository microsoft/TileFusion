// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/swizzle.hpp"

namespace tilefusion::testing {
using namespace cell;

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
    int swizzled_idx = swizzle.apply(idx);

    int ref_swizzled_idx = swizzle_ref<kB, kM, kS>(x, y);

#ifdef DEBUG
    printf("idx: %d, swizzled_idx: %d, ref_swizzled_idx: %d\n", idx,
           swizzled_idx, ref_swizzled_idx);
#endif

    return make_int2(swizzled_idx, ref_swizzled_idx);
}

TEST(TESTSwizzle, test_swizzle_apply) {
    const int kB = 3;
    const int kM = 3;
    const int kS = 3;

    EXPECT_EQ((test_swizzle<kB, kM, kS>(0, 0).x),
              (test_swizzle<kB, kM, kS>(0, 0).y));
    EXPECT_EQ((test_swizzle<kB, kM, kS>(1, 0).x),
              (test_swizzle<kB, kM, kS>(1, 0).y));
    EXPECT_EQ((test_swizzle<kB, kM, kS>(1, 4).x),
              (test_swizzle<kB, kM, kS>(1, 4).y));
    EXPECT_EQ((test_swizzle<kB, kM, kS>(2, 0).x),
              (test_swizzle<kB, kM, kS>(2, 0).y));
}

}  // namespace tilefusion::testing
