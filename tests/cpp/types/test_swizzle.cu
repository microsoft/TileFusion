// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/mod.hpp"

namespace tilefusion::testing {
using namespace cell;
namespace tl = tile_layout;

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

TEST(TESTSwizzle, test_naive_swizzle_layout) {
    const int kB = 3;
    const int kM = 3;
    const int kS = 3;

    const int kRows = 1 << kB;
    const int kCols = 1 << (kM + kS);

    using NaiveRowMajorLayout = tl::RowMajor<kRows, kCols>;
    using NaiveSwizzledRowMajorLayout =
        SwizzleLayout<NaiveRowMajorLayout, kB, kM, kS>;

    NaiveSwizzledRowMajorLayout naive_row_major_swizzled_layout;

    EXPECT_EQ((naive_row_major_swizzled_layout(0, 0)),
              (swizzle_ref<kB, kM, kS>(0, 0)));
    EXPECT_EQ((naive_row_major_swizzled_layout(1, 0)),
              (swizzle_ref<kB, kM, kS>(1, 0)));
    EXPECT_EQ((naive_row_major_swizzled_layout(1, 4)),
              (swizzle_ref<kB, kM, kS>(1, 4)));
    EXPECT_EQ((naive_row_major_swizzled_layout(2, 0)),
              (swizzle_ref<kB, kM, kS>(2, 0)));
    EXPECT_EQ((naive_row_major_swizzled_layout(2, 4)),
              (swizzle_ref<kB, kM, kS>(2, 4)));
}

TEST(TESTSwizzle, test_nested_basetile_swizzle_layout) {
    const int kB = 3;
    const int kM = 3;
    const int kS = 3;

    const int kRows = 1 << kB;
    const int kCols = 1 << (kM + kS);

    using NestedBaseTileLayout =
        tl::detail::SharedLayout<kRows, kCols, kCols * 16, 16,
                                 tl::Layout::kRowMajor>;
    using NestedBaseTileSwizzledLayout =
        SwizzleLayout<NestedBaseTileLayout, kB, kM, kS>;

    NestedBaseTileLayout nested_base_tile_layout;
    NestedBaseTileSwizzledLayout nested_base_tile_swizzled_layout;

    int idex_0_0 = nested_base_tile_layout(0, 0);
    int idex_1_0 = nested_base_tile_layout(1, 0);
    int idex_1_4 = nested_base_tile_layout(1, 4);
    int idex_2_0 = nested_base_tile_layout(2, 0);
    int idex_2_4 = nested_base_tile_layout(2, 4);

    int swizzled_idx_0_0 = nested_base_tile_swizzled_layout(0, 0);
    int swizzled_idx_1_0 = nested_base_tile_swizzled_layout(1, 0);
    int swizzled_idx_1_4 = nested_base_tile_swizzled_layout(1, 4);
    int swizzled_idx_2_0 = nested_base_tile_swizzled_layout(2, 0);
    int swizzled_idx_2_4 = nested_base_tile_swizzled_layout(2, 4);

    printf("idex_0_0: %d, swizzled_idx_0_0: %d\n", idex_0_0, swizzled_idx_0_0);
    printf("idex_1_0: %d, swizzled_idx_1_0: %d\n", idex_1_0, swizzled_idx_1_0);
    printf("idex_1_4: %d, swizzled_idx_1_4: %d\n", idex_1_4, swizzled_idx_1_4);
    printf("idex_2_0: %d, swizzled_idx_2_0: %d\n", idex_2_0, swizzled_idx_2_0);
    printf("idex_2_4: %d, swizzled_idx_2_4: %d\n", idex_2_4, swizzled_idx_2_4);
}

}  // namespace tilefusion::testing
