// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/base_tile.hpp"

namespace tilefusion::testing {
namespace tl = tile_layout;

TEST(InferAtomicWarpTile, test1_half_row_major) {
  using DType = __half;
  const tl::Layout kLayout = tl::Layout::kRowMajor;

  {  // atomic warp shape: 32x8, thread layout: 32x1
    using WarpTile = WarpBaseTileShape<DType, TileShape<128, 8>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 32);
    EXPECT_EQ(WarpTile::kCols, 8);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 32);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 1);
  }

  {  // atomic warp shape: 16x16, thread layout: 16x2
    using WarpTile = WarpBaseTileShape<DType, TileShape<64, 16>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 16);
    EXPECT_EQ(WarpTile::kCols, 16);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 16);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 2);
  }

  {  // atomic warp shape: 8x32, thread layout: 8x4
    using WarpTile = WarpBaseTileShape<DType, TileShape<16, 32>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 8);
    EXPECT_EQ(WarpTile::kCols, 32);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 8);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 4);
  }

  {  // atomic warp shape: 4x64, thread layout: 4x8
    using WarpTile = WarpBaseTileShape<DType, TileShape<128, 128>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 4);
    EXPECT_EQ(WarpTile::kCols, 64);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 4);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 8);
  }
}

TEST(InferAtomicWarpTile, test2_half_column_major) {
  using DType = __half;
  const tl::Layout kLayout = tl::Layout::kColMajor;

  {  // atomic warp shape: 8x32, thread layout: 1x32
    using WarpTile = WarpBaseTileShape<DType, TileShape<8, 128>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 8);
    EXPECT_EQ(WarpTile::kCols, 32);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 1);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 32);
  }

  {  // atomic warp shape: 16x16, thread layout: 2x16
    using WarpTile = WarpBaseTileShape<DType, TileShape<16, 64>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 16);
    EXPECT_EQ(WarpTile::kCols, 16);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 2);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 16);
  }

  {  // atomic warp shape: 32x8, thread layout: 4x8
    using WarpTile = WarpBaseTileShape<DType, TileShape<32, 16>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 32);
    EXPECT_EQ(WarpTile::kCols, 8);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 4);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 8);
  }

  {  // atomic warp shape: 64x4, thread layout: 8x4
    using WarpTile = WarpBaseTileShape<DType, TileShape<128, 128>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 64);
    EXPECT_EQ(WarpTile::kCols, 4);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 8);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 4);
  }
}

TEST(InferAtomicWarpTile, test3_float_row_major) {
  using DType = float;
  const tl::Layout kLayout = tl::Layout::kRowMajor;

  {  // atomic warp shape: 32x4, thread layout: 32x1
    using WarpTile = WarpBaseTileShape<DType, TileShape<128, 4>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 32);
    EXPECT_EQ(WarpTile::kCols, 4);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 32);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 1);
  }

  {  // atomic warp shape: 16x8, thread layout: 16x2
    using WarpTile = WarpBaseTileShape<DType, TileShape<64, 8>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 16);
    EXPECT_EQ(WarpTile::kCols, 8);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 16);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 2);
  }

  {  // atomic warp shape: 8x16, thread layout: 8x4
    using WarpTile = WarpBaseTileShape<DType, TileShape<16, 16>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 8);
    EXPECT_EQ(WarpTile::kCols, 16);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 8);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 4);
  }

  {  // atomic warp shape: 4x32, thread layout: 4x8
    using WarpTile = WarpBaseTileShape<DType, TileShape<128, 128>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 4);
    EXPECT_EQ(WarpTile::kCols, 32);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 4);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 8);
  }
}

TEST(InferAtomicWarpTile, test4_float_column_major) {
  using DType = float;
  const tl::Layout kLayout = tl::Layout::kColMajor;

  {  // atomic warp shape: 4x32, thread layout: 1x32
    using WarpTile = WarpBaseTileShape<DType, TileShape<4, 128>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 4);
    EXPECT_EQ(WarpTile::kCols, 32);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 1);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 32);
  }

  {  // atomic warp shape: 8x16, thread layout: 2x16
    using WarpTile = WarpBaseTileShape<DType, TileShape<8, 64>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 8);
    EXPECT_EQ(WarpTile::kCols, 16);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 2);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 16);
  }

  {  // atomic warp shape: 16x8, thread layout: 4x8
    using WarpTile = WarpBaseTileShape<DType, TileShape<16, 32>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 16);
    EXPECT_EQ(WarpTile::kCols, 8);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 4);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 8);
  }

  {  // atomic warp shape: 4x32, thread layout: 8x4
    using WarpTile = WarpBaseTileShape<DType, TileShape<128, 128>, kLayout>;

    EXPECT_EQ(WarpTile::kRows, 32);
    EXPECT_EQ(WarpTile::kCols, 4);

    EXPECT_EQ(WarpTile::WarpThreadLayout::kRows, 8);
    EXPECT_EQ(WarpTile::WarpThreadLayout::kCols, 4);
  }
}
}  // namespace tilefusion::testing
