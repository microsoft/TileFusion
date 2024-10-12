// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/mod.hpp"

namespace tilefusion::testing {

using namespace cell;
namespace tl = tile_layout;

TEST(TestTile, test_global_tile_iterator) {
    using Element = __half;

    const int rows = 4;
    const int cols = 12;

    using Tile = GlobalTile<Element, tl::RowMajor<rows, cols>>;
    using Iterator = GTileIterator<Tile, TileShape<2, 4>>;

    LOG(INFO) << std::endl << "Test Row-major" << std::endl;
}

}  // namespace tilefusion::testing
