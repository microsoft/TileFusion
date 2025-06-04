// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * @file warp.hpp
 * @brief This file contains **Warp** related operations for copy.
 */
#pragma once

#include "cell/copy/constants.hpp"
#include "types/layout.hpp"
#include "types/tile_shape.hpp"

namespace tilefusion::cell::copy::warp {
using namespace tilefusion::cell;
namespace tl = tile_layout;

namespace {  // functions/class/structs that are not exposed to a larger scope

template <const WarpReuse kMode, int kRowStride, int kColStride>
struct WarpOffsetHelper;

template <int kRowStride_, int kColStride_>
struct WarpOffsetHelper<WarpReuse::kCont, kRowStride_, kColStride_> {
  static constexpr int kRowStride = kRowStride_;
  static constexpr int kColStride = kColStride_;

  DEVICE int operator()(int i, int j) const {
    return i * kRowStride + j * kColStride;
  }
};

template <int kRowStride_, int kColStride_>
struct WarpOffsetHelper<WarpReuse::kColReuseCont, kRowStride_, kColStride_> {
  static constexpr int kRowStride = kRowStride_;
  static constexpr int kColStride = kColStride_;

  DEVICE int operator()(int i, int j) const { return j * kColStride; }
};

template <int kRowStride_, int kColStride_>
struct WarpOffsetHelper<WarpReuse::kRowReuseCont, kRowStride_, kColStride_> {
  static constexpr int kRowStride = kRowStride_;
  static constexpr int kColStride = kColStride_;

  DEVICE int operator()(int i, int j) const { return i * kRowStride; }
};

/// @brief Helper for pretty printing a BaseTile's static shape-related
///        information. This printer works ONLY on the host.
struct BaseTilePrettyPrinter {
  template <typename BaseShape>
  static HOST void print(std::ostream& out, const BaseShape& tile) {
    // parameter `tile` here is not used
    out << "BaseShape = (" << BaseShape::kRows << ", " << BaseShape::kCols
        << "), Numel = " << BaseShape::kNumel << ", ThreadLayout = ("
        << BaseShape::kRowThreads << ", " << BaseShape::kColThreads << ")";
  }
};
}  // namespace

// @brief In a thread block, warps are organized as 2-D matrices, each with
//        a row index and a column index. Given `threadIdx.x`, this function
//        calculates the row index of the current thread.
template <typename WarpLayout>
DEVICE int warp_row_id() {
  /*
   * Example1: suppose the warp layout is RowMajor<2,2>, like this:
   * |-|-----|-----|
   * |0|warp0|warp1|
   * |-|-----|-----|
   * |1|warp2|warp3|
   * |-|-----|-----|, and the threadIdx is 67, then the warp row is 1.
   *
   * Example2: suppose the warp layout is ColMajor<2,2>, like this:
   * |-|-----|-----|
   * |0|warp0|warp2|
   * |-|-----|-----|
   * |1|warp1|warp3|
   * |-|-----|-----|, and the threadIdx is 67, then the warp row is 0.
   */
  int wid = threadIdx.x / WARP_SIZE;

  switch (tl::layout_type<WarpLayout>) {
    case tl::Layout::kRowMajor:
      return wid / tl::num_cols<WarpLayout>;
    case tl::Layout::kColMajor:
      return wid % tl::num_rows<WarpLayout>;
    default:
      assert(false && "Not implemented yet.");
      return -1;
  }
}

// @brief In a thread block, warps are organized as 2-D matrices, each with
//        a row index and a column index. Given `threadIdx.x`, this function
//        calculates the column index of the current thread.
template <typename WarpLayout>
DEVICE int warp_col_id() {
  /*
   * Example1: suppose the warp layout is RowMajor<2,2>, like this:
   * |-----|-----|
   * |  0  |  1  |
   * |-----|-----|
   * |warp0|warp1|
   * |-----|-----|
   * |warp2|warp3|
   * |-----|-----|, and the threadIdx is 67, then the warp col is 0.
   *
   * Example2: suppose the warp layout is ColMajor<2,2>, like this:
   * |-----|-----|
   * |  0  |  1  |
   * |-----|-----|
   * |warp0|warp2|
   * |-----|-----|
   * |warp1|warp3|
   * |-----|-----|, and the threadIdx is 67, then the warp row is 1.
   */
  int wid = threadIdx.x / WARP_SIZE;

  switch (tl::layout_type<WarpLayout>) {
    case tl::Layout::kRowMajor:
      return wid % tl::num_cols<WarpLayout>;
    case tl::Layout::kColMajor:
      return wid / tl::num_rows<WarpLayout>;
    default:
      assert(false && "Not implemented yet.");
      return -1;
  }
}

template <const int kSharedRows, const int kWarpRows,
          const WarpReuse kMode = WarpReuse::kCont>
HOST_DEVICE constexpr int warp_tile_rows() {
  if constexpr (kMode == WarpReuse::kCont) {
    return kSharedRows / kWarpRows;
  } else if constexpr (kMode == WarpReuse::kRowReuseCont) {
    return kSharedRows / kWarpRows;
  } else if constexpr (kMode == WarpReuse::kColReuseCont) {
    return kSharedRows;
  }
  return -1;
}

template <const int kSharedCols, const int kWarpCols,
          const WarpReuse kMode = WarpReuse::kCont>
HOST_DEVICE constexpr int warp_tile_cols() {
  if constexpr (kMode == WarpReuse::kCont) {
    return kSharedCols / kWarpCols;
  } else if constexpr (kMode == WarpReuse::kRowReuseCont) {
    return kSharedCols;
  } else if constexpr (kMode == WarpReuse::kColReuseCont) {
    return kSharedCols / kWarpCols;
  }
  return -1;
}

template <typename BaseShape_, typename Tile_, typename WarpLayout_,
          const WarpReuse kMode_>
struct ExecCounter {
  using BaseShape = BaseShape_;
  using Tile = Tile_;

  static_assert(
      Tile::kCols % BaseShape::kCols == 0,
      "The number of shared memory columns must be divisible by the base "
      "tile column.\n");
  static_assert(Tile::kRows % BaseShape::kRows == 0,
                "The current implementation requires that the number of shared "
                "memory rows be divisible by the base tile row.\n");

  static constexpr int kWarpsPerRow = tl::num_rows<WarpLayout_>;
  static constexpr int kWarpsPerCol = tl::num_cols<WarpLayout_>;
  static constexpr WarpReuse kMode = kMode_;

  // @brief This function returns the number of times a `BaseTile` is executed
  //        along the direction of the shared memory row.
  DEVICE static constexpr int row_exec_count() {
    switch (kMode) {
      // Warps in the same columns (`warps_per_row` in total) repeatedly
      // load the shared memory rows. Therefore, `row_exec` is not divided
      // by warps_per_row.
      case WarpReuse::kColReuseCont:
        return Tile::kRows / BaseShape::kRows;
      default:  // Cont, RowReuseCont hit this case.
        return Tile::kRows / BaseShape::kRows / kWarpsPerRow;
    }
  }

  DEVICE static constexpr int col_exec_count() {
    switch (kMode) {
      // Warps in the same rows (`warps_per_col` in total) repeatedly load
      // the shared memory columns. Therefore, `col_exec` is not divided
      // by `warps_per_col`.
      case WarpReuse::kRowReuseCont:
        return Tile::kCols / BaseShape::kCols;
      default:  // Cont, ColReuseCont hit this case.
        return Tile::kCols / BaseShape::kCols / kWarpsPerCol;
    }
  }

  static constexpr int kRowExec = row_exec_count();
  static constexpr int kColExec = col_exec_count();
};

template <typename WarpLayout_, const WarpReuse kMode_>
struct GlobalOffsetHelper {
  static constexpr WarpReuse kMode = kMode_;
  using WarpLayout = WarpLayout_;

  // @brief This function returns the offset to the start position of the
  //        current warp in the shared memory according to the warp reuse
  //        mode.
  template <typename Tile>
  DEVICE int get_warp_offset() {
    // Tile shape for a single warp
    constexpr static int kWarpShapeRow = Tile::kRows / tl::num_rows<WarpLayout>;
    constexpr static int kWarpShapeCol = Tile::kCols / tl::num_cols<WarpLayout>;

    constexpr static int kWarpRstride = Tile::kType == tl::Layout::kRowMajor
                                            ? Tile::kRowStride * kWarpShapeRow
                                            : kWarpShapeRow;
    constexpr static int kWarpCstride = Tile::kType == tl::Layout::kRowMajor
                                            ? kWarpShapeCol
                                            : Tile::kColStride * kWarpShapeCol;

    using Offset = WarpOffsetHelper<kMode, kWarpRstride, kWarpCstride>;
    Offset offset_;
    return offset_(warp_row_id<WarpLayout>(), warp_col_id<WarpLayout>());
  }
};

template <typename WarpLayout, typename BaseShape, typename Shared,
          const WarpReuse kMode, const tl::Layout kType = Shared::kType>
struct SharedOffsetHelper;

template <typename WarpLayout_, typename BaseShape_, typename Shared_,
          const WarpReuse kMode_>
struct SharedOffsetHelper<WarpLayout_, BaseShape_, Shared_, kMode_,
                          tl::Layout::kRowMajor> {
  DEVICE int get_warp_offset() {
    switch (kMode) {
      case WarpReuse::kCont:
        return warp_row_id<WarpLayout>() * kRowStride * BaseShape::kRows *
                   Shared::kRowStride +
               warp_col_id<WarpLayout>() * kColStride * BaseShape::kCols;
      case WarpReuse::kRowReuseCont:
        return warp_row_id<WarpLayout>() * kRowStride * BaseShape::kRows *
               Shared::kRowStride;
      default:
        assert(false && "Not implemented yet.");
        return -1;
    }
  }

 private:
  using Shared = Shared_;
  using WarpLayout = WarpLayout_;
  using BaseShape = BaseShape_;
  static constexpr WarpReuse kMode = kMode_;

  constexpr static int kTilePerRow = Shared::kRows / BaseShape::kRows;
  constexpr static int kTilePerCol = Shared::kCols / BaseShape::kCols;

  // TODO(KuangjuX): hotfix this.
  constexpr static int kRowStride = kTilePerRow / tl::num_rows<WarpLayout>;
  constexpr static int kColStride = kTilePerCol / tl::num_cols<WarpLayout>;
};

template <typename WarpLayout_, typename BaseShape_, typename Shared_,
          const WarpReuse kMode_>
struct SharedOffsetHelper<WarpLayout_, BaseShape_, Shared_, kMode_,
                          tl::Layout::kColMajor> {
  DEVICE int get_warp_offset() {
    switch (kMode) {
      case WarpReuse::kCont:
        return warp_row_id<WarpLayout>() * kRowStride * BaseShape::kRows +
               warp_col_id<WarpLayout>() * kColStride * BaseShape::kCols *
                   Shared::kColStride;
      case WarpReuse::kColReuseCont:
        return warp_col_id<WarpLayout>() * kColStride * BaseShape::kCols *
               Shared::kColStride;
      default:
        assert(false && "Not implemented yet.");
        return -1;
    }
  }

 private:
  using Shared = Shared_;
  using WarpLayout = WarpLayout_;
  using BaseShape = BaseShape_;
  static constexpr WarpReuse kMode = kMode_;

  constexpr static int kTilePerRow = Shared::kRows / BaseShape::kRows;
  constexpr static int kTilePerCol = Shared::kCols / BaseShape::kCols;

  constexpr static int kRowStride = kTilePerRow / tl::num_rows<WarpLayout>;
  constexpr static int kColStride = kTilePerCol / tl::num_cols<WarpLayout>;
};

}  // namespace tilefusion::cell::copy::warp
