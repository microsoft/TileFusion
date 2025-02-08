// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <sstream>

namespace tilefusion::testing {
using namespace cell;
namespace tl = tile_layout;

namespace {
template <typename OuterTile, typename BaseShape_>
void test_tiled_row_major() {
    using DType = float;  // data type does not matter in this test

    using BaseShape =
        WarpBaseTileShape<DType, BaseShape_, tl::Layout::kRowMajor>;
    using SharedLayout = tl::TiledMatrixLayout<OuterTile, BaseShape>;

    SharedLayout layout;
    std::stringstream ss;

    for (int i = 0; i < OuterTile::kRows; ++i) {
        ss << "[" << i << "]:\t";
        for (int j = 0; j < OuterTile::kCols; ++j) {
            ss << layout(i, j) << ", ";
        }
        ss << std::endl;
    }

    LOG(INFO) << std::endl << ss.str() << std::endl;
}
}  // namespace

TEST(TestTiledMatrixLayout, test_row_major) {
    test_tiled_row_major<tl::RowMajor<32, 32>, TileShape<16, 16>>();
}

}  // namespace tilefusion::testing
