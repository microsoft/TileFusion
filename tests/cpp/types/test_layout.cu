// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <iostream>

namespace tilefusion::testing {

using namespace tilefusion::cell;
namespace tl = tile_layout;

TEST(TestLayout, test_layout) {
    using Element = __half;

    using Layout1 = tl::RowMajor<4, 7>;
    EXPECT_EQ(tl::num_rows<Layout1>, 4);
    EXPECT_EQ(tl::num_cols<Layout1>, 7);
    EXPECT_EQ(tl::get_numel<Layout1>, 28);
    EXPECT_EQ(tl::row_stride<Layout1>, 7);
    EXPECT_EQ(tl::col_stride<Layout1>, 1);

    tl::Layout type1 = tl::layout_type<Layout1>;
    EXPECT_EQ(type1, tl::Layout::kRowMajor);
    auto layout_name1 = layout_type_to_str(type1);
    EXPECT_EQ(layout_name1, "RowMajor");

    using Layout2 = tl::ColMajor<4, 7>;
    EXPECT_EQ(tl::num_rows<Layout2>, 4);
    EXPECT_EQ(tl::num_cols<Layout2>, 7);
    EXPECT_EQ(tl::get_numel<Layout2>, 28);
    EXPECT_EQ(tl::row_stride<Layout2>, 1);
    EXPECT_EQ(tl::col_stride<Layout2>, 4);

    tl::Layout type2 = tl::layout_type<Layout2>;
    EXPECT_EQ(type2, tl::Layout::kColMajor);
    auto layout_name2 = layout_type_to_str(type2);
    EXPECT_EQ(layout_name2, "ColMajor");
}

TEST(TestLayout, test_block_row_major) {
    using Layout = tl::BlockRowMajor<tl::RowMajor<14, 9>, tl::RowMajor<2, 3>>;

    EXPECT_EQ(Layout::kTileRows, 7);
    EXPECT_EQ(Layout::kTileCols, 3);
    EXPECT_EQ(Layout::kRowStride, 18);
    EXPECT_EQ(Layout::kColStride, 6);
    EXPECT_EQ(Layout::kType, tl::Layout::kRowMajor);

#ifdef 0
    Layout layout;
    for (int i = 0; i < Layout::kRows; ++i) {
        for (int j = 0; j < Layout::kCols; ++j) {
            std::cout << layout(i, j) << ", ";
        }
        std::cout << std::endl;
    }
#endif
}

TEST(TestLayout, test_block_col_major) {
    using Layout = tl::BlockColMajor<tl::ColMajor<14, 9>, tl::ColMajor<2, 3>>;

    EXPECT_EQ(Layout::kTileRows, 7);
    EXPECT_EQ(Layout::kTileCols, 3);
    EXPECT_EQ(Layout::kRowStride, 6);
    EXPECT_EQ(Layout::kColStride, 42);
    EXPECT_EQ(Layout::kType, tl::Layout::kColMajor);

#ifdef 0
    Layout layout;
    for (int i = 0; i < Layout::kRows; ++i) {
        for (int j = 0; j < Layout::kCols; ++j) {
            std::cout << layout(i, j) << ", ";
        }
        std::cout << std::endl;
    }
#endif
}

}  // namespace tilefusion::testing
