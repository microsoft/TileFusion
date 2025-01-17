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

TEST(TESTSwizzle, test_swizzle_apply) {
    const int kB = 3;
    const int kM = 3;
    const int kS = 3;

    int width = 1 << (kS + kM);

    Swizzle<kB, kM, kS> swizzle_3x3x3;

    EXPECT_EQ((swizzle_3x3x3.apply(flatten(0, 0, width))),
              (swizzle_ref<kB, kM, kS>(0, 0)));

    EXPECT_EQ((swizzle_3x3x3.apply(flatten(1, 0, width))),
              (swizzle_ref<kB, kM, kS>(1, 0)));
}

}  // namespace tilefusion::testing