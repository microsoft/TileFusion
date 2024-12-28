// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * @file warp.hpp
 * @brief This file contains **Warp** related operations for copy.
 */
#pragma once

#include "cell/copy/constants.hpp"
#include "types/layout.hpp"

namespace tilefusion::cell::copy::warp {
namespace tl = tile_layout;
using namespace cute;

namespace {
// FIXME(ying): This hotfix addresses the current implementation's inability
// to explicitly distinguish between shared memory's row-major or
// column-major layout and global memory's layouts. However, this should be
// fixed in the future.
template <typename Layout>
static constexpr bool IsSharedLayout =
    (Layout::kRowStride == Layout::kCols && Layout::kColStride == 1) ||
            (Layout::kRowStride == 1 && Layout::kColStride == Layout::kRows)
        ? false
        : true;

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

template <typename BaseTile_, typename Tile_, typename WarpLayout_,
          const WarpReuse kMode_>
struct ExecCounter {
    using BaseTile = BaseTile_;
    using Tile = Tile_;

    static_assert(
        Tile::kCols % BaseTile::kCols == 0,
        "The number of shared memory columns must be divisible by the base "
        "tile column.\n");
    static_assert(
        Tile::kRows % BaseTile::kRows == 0,
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
                return Tile::kRows / BaseTile::kRows;
            default:  // Cont, RowReuseCont hit this case.
                return Tile::kRows / BaseTile::kRows / kWarpsPerRow;
        }
    }

    DEVICE static constexpr int col_exec_count() {
        switch (kMode) {
            // Warps in the same rows (`warps_per_col` in total) repeatedly load
            // the shared memory columns. Therefore, `col_exec` is not divided
            // by `warps_per_col`.
            case WarpReuse::kRowReuseCont:
                return Tile::kCols / BaseTile::kCols;
            default:  // Cont, ColReuseCont hit this case.
                return Tile::kCols / BaseTile::kCols / kWarpsPerCol;
        }
    }

    static constexpr int kRowExec = row_exec_count();
    static constexpr int kColExec = col_exec_count();
};

/// @brief Determine the automatic shape of a single warp based on the shape of
///        the entire tile. The final warp tile shape is multiple of this atomic
///        shape.
template <typename DType, typename TileLayout, const tl::Layout kType>
struct WarpTileShape;

template <typename DType, typename TileLayout>
struct WarpTileShape<DType, TileLayout, tl::Layout::kRowMajor> {
    using AccessInfo = traits::AccessBase<DType>;

    // In a row-major layout, columns are the contiguous dimension in memory. We
    // enforce the use of 128-bit vectorized instructions for data loading by a
    // single thread. This implies that the minimum number of columns should be
    // at least 128 bits.
    static constexpr int kMinCols =
        AccessInfo::kAccessInBits / (sizeof(DType) * 8);

    static_assert(TileLayout::kCols >= kMinCols,
                  "The number of columns is too small.");

    static_assert(TileLayout::kCols < AccessInfo::kExpectedSize ||
                      (TileLayout::kCols >= AccessInfo::kExpectedSize &&
                       TileLayout::kCols % AccessInfo::kExpectedSize == 0),
                  "The current implementation requires that the number of "
                  "columns of the tile be divisible by the cache line width.");

    static constexpr int kCols = TileLayout::kCols >= AccessInfo::kExpectedSize
                                     ? AccessInfo::kExpectedSize
                                     : TileLayout::kCols;

    // number of columns in a warp
    static constexpr int kThreadPerRow = kCols / AccessInfo::kNumPerAccess;
    static_assert(WARP_SIZE % kThreadPerRow == 0,
                  "Fail to infer warp thread layout.");
    static constexpr int kThreadPerCol = WARP_SIZE / kThreadPerRow;

    static constexpr int kRows = kThreadPerCol;
    static_assert(TileLayout::kRows % kThreadPerCol == 0,
                  "The number of rows of the tile isn't evenly divisible by "
                  "the number of threads in a column.");

    static constexpr int kNumel = kRows * kCols;

    using WarpThreadLayout = tl::RowMajor<kThreadPerCol, kThreadPerRow>;
};

template <typename DType, typename TileLayout>
struct WarpTileShape<DType, TileLayout, tl::Layout::kColMajor> {
    using AccessInfo = traits::AccessBase<DType>;

    // In a column-major layout, columns are the contiguous dimension in memory.
    // We enforce the use of 128-bit vectorized instructions for data loading by
    // a single thread. This implies that the minimum number of columns should
    // be at least 128 bits.
    static constexpr int kMinRows =
        AccessInfo::kAccessInBits / (sizeof(DType) * 8);

    static_assert(TileLayout::kRows >= kMinRows,
                  "The number of rows is too small.");

    static_assert(TileLayout::kRows < AccessInfo::kExpectedSize ||
                      (TileLayout::kRows >= AccessInfo::kExpectedSize &&
                       TileLayout::kRows % AccessInfo::kExpectedSize == 0),
                  "The current implementation requires that the number of "
                  "rows of the tile be divisible by the cache line width.");

    static constexpr int kRows = TileLayout::kRows >= AccessInfo::kExpectedSize
                                     ? AccessInfo::kExpectedSize
                                     : TileLayout::kRows;

    // number of rows in a warp
    static constexpr int kThreadPerCol = kRows / AccessInfo::kNumPerAccess;
    static_assert(WARP_SIZE % kThreadPerCol == 0,
                  "Fail to infer warp thread layout.");
    static constexpr int kThreadPerRow = WARP_SIZE / kThreadPerCol;

    static constexpr int kCols = kThreadPerRow;
    static_assert(TileLayout::kCols % kThreadPerRow == 0,
                  "The number of columns of the tile isn't evenly divisible by "
                  "the number of threads in a row.");

    static constexpr int kNumel = kRows * kCols;

    using WarpThreadLayout = tl::ColMajor<kThreadPerCol, kThreadPerRow>;
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
        constexpr static int kWarpShapeRow =
            Tile::kRows / tl::num_rows<WarpLayout>;
        constexpr static int kWarpShapeCol =
            Tile::kCols / tl::num_cols<WarpLayout>;

        constexpr static int kWarpRstride =
            Tile::kType == tl::Layout::kRowMajor
                ? Tile::kRowStride * kWarpShapeRow
                : kWarpShapeRow;
        constexpr static int kWarpCstride =
            Tile::kType == tl::Layout::kRowMajor
                ? kWarpShapeCol
                : Tile::kColStride * kWarpShapeCol;

        using Offset = WarpOffsetHelper<kMode, kWarpRstride, kWarpCstride>;
        Offset offset_;
        return offset_(warp_row_id<WarpLayout>(), warp_col_id<WarpLayout>());
    }
};

template <typename WarpLayout, typename WarpShape, typename Shared,
          const WarpReuse kMode, const tl::Layout kType = Shared::kType,
          const bool kIsSharedLayout = IsSharedLayout<Shared>>
struct SharedOffsetHelper;

template <typename WarpLayout_, typename WarpShape_, typename Shared_,
          const WarpReuse kMode_>
struct SharedOffsetHelper<WarpLayout_, WarpShape_, Shared_, kMode_,
                          tl::Layout::kRowMajor, false> {
    DEVICE int get_warp_offset() {
        int tile_id = warp_row_id<WarpLayout>() * kRowStride +
                      warp_col_id<WarpLayout>() * kColStride;
        return tile_id * WarpShape::kNumel;
    }

  private:
    using Shared = Shared_;
    using WarpLayout = WarpLayout_;
    using WarpShape = WarpShape_;
    static constexpr WarpReuse kMode = kMode_;

    constexpr static int kTilePerRow = Shared::kRows / WarpShape::kRows;
    constexpr static int kTilePerCol = Shared::kCols / WarpShape::kCols;

    constexpr static int kRowStride =
        kTilePerRow / tl::num_rows<WarpLayout> * kTilePerCol;
    constexpr static int kColStride = kTilePerCol / tl::num_cols<WarpLayout>;
};

template <typename WarpLayout_, typename WarpShape_, typename Shared_,
          const WarpReuse kMode_>
struct SharedOffsetHelper<WarpLayout_, WarpShape_, Shared_, kMode_,
                          tl::Layout::kColMajor, false> {
    DEVICE int get_warp_offset() {
        int tile_id = warp_row_id<WarpLayout>() * kRowStride +
                      warp_col_id<WarpLayout>() * kColStride;
        return tile_id * WarpShape::kNumel;
    }

  private:
    using Shared = Shared_;
    using WarpLayout = WarpLayout_;
    using WarpShape = WarpShape_;
    static constexpr WarpReuse kMode = kMode_;

    constexpr static int kTilePerRow = Shared::kRows / WarpShape::kRows;
    constexpr static int kTilePerCol = Shared::kCols / WarpShape::kCols;

    constexpr static int kRowStride = kTilePerRow / tl::num_rows<WarpLayout>;
    constexpr static int kColStride =
        kTilePerCol / tl::num_cols<WarpLayout> * kTilePerRow;
};

template <typename WarpLayout_, typename WarpShape_, typename Shared_,
          const WarpReuse kMode_, const tl::Layout kType>
struct SharedOffsetHelper<WarpLayout_, WarpShape_, Shared_, kMode_, kType,
                          true> {
    using WarpLayout = WarpLayout_;

    DEVICE int get_warp_offset() {
        return offset_(warp_row_id<WarpLayout>(), warp_col_id<WarpLayout>());
    }

  private:
    using Shared = Shared_;
    using WarpShape = WarpShape_;
    static constexpr WarpReuse kMode = kMode_;

    constexpr static int kTilePerRow = Shared::kCols / WarpShape::kCols;
    constexpr static int kTilePerCol = Shared::kRows / WarpShape::kRows;

    constexpr static int kTilePerWarpRow =
        kTilePerRow / tl::num_cols<WarpLayout>;
    constexpr static int kTilePerWarpCol =
        kTilePerCol / tl::num_rows<WarpLayout>;

    using Offset = WarpOffsetHelper<kMode, Shared::kRowStride * kTilePerWarpCol,
                                    Shared::kColStride * kTilePerWarpRow>;
    Offset offset_;
};
}  // namespace tilefusion::cell::copy::warp
