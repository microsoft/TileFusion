// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <cute/tensor.hpp>

namespace tilefusion::testing {

using namespace cell;
namespace tl = tile_layout;

using namespace cute;

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
  printf("idx: %d, swizzled_idx: %d, ref_swizzled_idx: %d\n", idx, swizzled_idx,
         ref_swizzled_idx);
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

TEST(TestSwizzle, test_swizzled_row_major) {
  using BlockRowMajor =
      tl::BlockRowMajor<tl::RowMajor<16, 64>,
                        SwizzledLayout<tl::RowMajor<8, 64>, Swizzle<3, 3, 3>>>;

  // for unit test
  using Atom =
      decltype(composition(cute::Swizzle<3, 3, 3>{},
                           cute::Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using CuteLayout =
      decltype(tile_to_shape(Atom{}, Shape<_16, _64>{}, Step<_16, _1>{}));

  BlockRowMajor layout1;
  CuteLayout layout2;

  for (int i = 0; i < int(size<0>(layout2)); ++i) {
    for (int j = 0; j < int(size<1>(layout2)); ++j) {
      EXPECT_EQ(layout1(i, j), layout2(i, j));
    }
  }
}

TEST(TestSwizzle, test_swizzled_col_major) {
  using BlockColMajor =
      tl::BlockColMajor<tl::ColMajor<64, 16>,
                        SwizzledLayout<tl::ColMajor<64, 8>, Swizzle<3, 3, 3>>>;

  using Atom = decltype(composition(cute::Swizzle<3, 3, 3>{},
                                    cute::Layout<Shape<_64, _8>>{}));
  using CuteLayout = decltype(tile_to_shape(Atom{}, Shape<_64, _16>{}));

  BlockColMajor layout1;
  CuteLayout layout2;

  for (int i = 0; i < int(size<0>(layout2)); ++i) {
    for (int j = 0; j < int(size<1>(layout2)); ++j) {
      EXPECT_EQ(layout1(i, j), layout2(i, j));
    }
  }
}

}  // namespace tilefusion::testing
