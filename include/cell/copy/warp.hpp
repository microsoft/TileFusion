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

namespace detail {
template <const WarpReuse kMode>
DEVICE int warp_offset_impl(int warp_row, int warp_col, int warp_rstride,
                            int warp_cstride) {
    assert(false && "Not implemented yet.");
    return -1;
};

template <>
DEVICE int warp_offset_impl<WarpReuse::kCont>(int warp_row, int warp_col,
                                              int warp_rstride,
                                              int warp_cstride) {
    return warp_row * warp_rstride + warp_col * warp_cstride;
}

template <>
DEVICE int warp_offset_impl<WarpReuse::kColReuseCont>(int warp_row,
                                                      int warp_col,
                                                      int warp_rstride,
                                                      int warp_cstride) {
    return warp_col * warp_cstride;
}

template <>
DEVICE int warp_offset_impl<WarpReuse::kRowReuseCont>(int warp_row,
                                                      int warp_col,
                                                      int warp_rstride,
                                                      int warp_cstride) {
    return warp_row * warp_rstride;
}
}  // namespace detail

template <typename BaseTile_, typename Tile_, typename WarpLayout,
          const WarpReuse kMode_>
struct ExecCounter {
    static constexpr WarpReuse kMode = kMode_;
    using BaseTile = BaseTile_;
    using Tile = Tile_;

    // @brief This function returns the number of times a `BaseTile` is executed
    //        along the direction of the shared memory row.
    DEVICE static constexpr int row_exec_count() {
        const int kWarpsPerRow = tl::num_rows<WarpLayout>;

        static_assert(
            Tile::kRows % BaseTile::kRows == 0,
            "The current implementation requires that the number of shared "
            "memory rows be divisible by the base tile row.\n");

        switch (kMode) {
            // Warps in the same columns (`warps_per_row` in total) repeatedly
            // load the shared memory rows. Therefore, `row_exec` is not divided
            // by warps_per_row.
            case WarpReuse::kColReuseCont:
            case WarpReuse::kColReuseCir:
                return Tile::kRows / BaseTile::kRows;
            default:  // Cont, Cir, RowReuseCont, RowReuseCir hit this case.
                return Tile::kRows / BaseTile::kRows / kWarpsPerRow;
        }
    }

    DEVICE static constexpr int col_exec_count() {
        const int kWarpsPerCol = tl::num_cols<WarpLayout>;

        static_assert(
            Tile::kCols % BaseTile::kCols == 0,
            "The number of shared memory columns must be divisible by the base "
            "tile column.\n");

        switch (kMode) {
            // Warps in the same rows (`warps_per_col` in total) repeatedly load
            // the shared memory columns. Therefore, `col_exec` is not divided
            // by `warps_per_col`.
            case WarpReuse::kRowReuseCont:
            case WarpReuse::kRowReuseCir:
                return Tile::kCols / BaseTile::kCols;
            default:  // Cont, Cir, ColReuseCont, ColReuseCir hit this case.
                return Tile::kCols / BaseTile::kCols / kWarpsPerCol;
        }
    }

    static constexpr int kRowExec = row_exec_count();
    static constexpr int kColExec = col_exec_count();
};

/// @brief Determine the automic shape of a single warp based on the shape of
///        the entire tile. The final warp tile shape is multiple of this atomic
///        shape.
template <typename DType, typename TileLayout, const tl::Layout kType>
struct WarpTileShape;

template <typename DType, typename TileLayout>
struct WarpTileShape<DType, TileLayout, tl::Layout::kRowMajor> {
    static constexpr int kWarpSize = 32;
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
    static_assert(kWarpSize % kThreadPerRow == 0,
                  "Fail to infer warp thread layout.");
    static constexpr int kThreadPerCol = kWarpSize / kThreadPerRow;

    static constexpr int kRows = kThreadPerCol;
    static_assert(TileLayout::kRows % kThreadPerCol == 0,
                  "The number of rows of the tile isn't evenly divisible by "
                  "the number of threads in a column.");

    static constexpr int kNumel = kRows * kCols;

    using WarpThreadLayout = tl::RowMajor<kThreadPerCol, kThreadPerRow>;
};

template <typename DType, typename TileLayout>
struct WarpTileShape<DType, TileLayout, tl::Layout::kColMajor> {
    static constexpr int kWarpSize = 32;
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
    static_assert(kWarpSize % kThreadPerCol == 0,
                  "Fail to infer warp thread layout.");
    static constexpr int kThreadPerRow = kWarpSize / kThreadPerCol;

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
    static constexpr int kWarpSize = 32;

    // @brief the warp row that the current thread belongs to, based on the warp
    //        layout.
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
        int wid = threadIdx.x / kWarpSize;

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

    // @brief: Returns the warp col that the current thread belongs to, based on
    //         the warp layout.
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
        int wid = threadIdx.x / kWarpSize;

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

        return detail::warp_offset_impl<kMode>(warp_row_id(), warp_col_id(),
                                               kWarpRstride, kWarpCstride);
    }
};

namespace {
using namespace cute;

// This is a hotfix for the current implementation, that is not intended to be
// exposed outside this header file. Thus, it is placed in an anonymous
// namespace. Fix this when the implementation is improved.
template <typename WarpLayout, const WarpReuse kMode, typename Shared,
          const bool kIsSharedLayout>
struct SharedOffsetHelperImpl;

template <typename WarpLayout_, const WarpReuse kMode_, typename Shared_>
struct SharedOffsetHelperImpl<WarpLayout_, kMode_, Shared_, false> {
    DEVICE int warp_row_id() {
        int warp_row = 0;
        switch (kMode) {
            case WarpReuse::kCont:
            case WarpReuse::kCir:
            case WarpReuse::kRowReuseCont:
            case WarpReuse::kRowReuseCir:
                warp_row = threadIdx.x / kWarpSize / tl::num_cols<WarpLayout>;
                break;
            case WarpReuse::kColReuseCont:
            case WarpReuse::kColReuseCir:
                break;
            default:
                assert(false && "Not implemented yet.");
        }
        return warp_row;
    }

    DEVICE int warp_col_id() {
        int warp_col = 0;
        switch (kMode) {
            case WarpReuse::kCont:
            case WarpReuse::kCir:
            case WarpReuse::kColReuseCont:
            case WarpReuse::kColReuseCir:
                warp_col = threadIdx.x / kWarpSize % tl::num_cols<WarpLayout>;
                break;
            case WarpReuse::kRowReuseCont:
            case WarpReuse::kRowReuseCir:
                break;
            default:
                assert(false && "Not implemented yet.");
        }
        return warp_col;
    }

    DEVICE int get_warp_offset() {
        int tile_id = Shared::kType == tl::Layout::kRowMajor
                          ? base_tiles_row_major_(warp_row_id(), warp_col_id())
                          : base_tiles_col_major_(warp_row_id(), warp_col_id());
        return tile_id * BaseShape::kNumel;
    }

  private:
    using Shared = Shared_;
    using WarpLayout = WarpLayout_;
    // data type __half here is to instantiate the templated class `BaseShape`.
    // It does not affect shape-related information.
    using BaseShape = traits::BaseTileShape<__half>;

    static constexpr int kWarpSize = 32;
    static constexpr WarpReuse kMode = kMode_;

    constexpr static int kBaseTilePerRow = Shared::kRows / BaseShape::kRows;
    constexpr static int kBaseTilePerCol = Shared::kCols / BaseShape::kCols;

    constexpr static int kRowStride1 =
        kBaseTilePerRow / tl::num_rows<WarpLayout> * kBaseTilePerCol;
    constexpr static int kColStride1 =
        kBaseTilePerCol / tl::num_cols<WarpLayout>;

    using BaseTilesRowMajorLayout =
        cute::Layout<Shape<Int<kBaseTilePerRow>, Int<kBaseTilePerCol>>,
                     Stride<Int<kRowStride1>, Int<kColStride1>>>;
    BaseTilesRowMajorLayout base_tiles_row_major_;

    constexpr static int kRowStride2 =
        kBaseTilePerRow / tl::num_rows<WarpLayout>;
    constexpr static int kColStride2 =
        kBaseTilePerCol / tl::num_cols<WarpLayout> * kBaseTilePerRow;
    using BaseTilesColMajorLayout =
        cute::Layout<Shape<Int<kBaseTilePerRow>, Int<kBaseTilePerCol>>,
                     Stride<Int<kRowStride2>, Int<kColStride2>>>;
    BaseTilesColMajorLayout base_tiles_col_major_;
};

template <typename WarpLayout_, const WarpReuse kMode_, typename Shared_>
struct SharedOffsetHelperImpl<WarpLayout_, kMode_, Shared_, true> {
    /*
     * @brief In a thread block, warps are organized as 2-D matrices, each with
     * a row index and a column index. Given `threadIdx.x`, this function
     * calculates the row index of the current thread.
     */
    DEVICE int warp_row_id() {
        int warp_id = threadIdx.x / warpSize;  // the 1-d warp index

        switch (tl::layout_type<WarpLayout>) {
            case tl::Layout::kRowMajor:
                return warp_id / tl::num_cols<WarpLayout>;
            case tl::Layout::kColMajor:
                return warp_id % tl::num_rows<WarpLayout>;
            default:
                assert(false && "Not implemented yet.");
                return -1;
        }
    }

    /*
     * @brief In a thread block, warps are organized as 2-D matrices, each with
     * a row index and a column index. Given `threadIdx.x`, this function
     * calculates the column index of the current thread.
     */
    DEVICE int warp_col_id() {
        int warp_id = threadIdx.x / warpSize;  // the 1-d warp index

        switch (tl::layout_type<WarpLayout>) {
            case tl::Layout::kRowMajor:
                return warp_id % tl::num_cols<WarpLayout>;
            case tl::Layout::kColMajor:
                return warp_id / tl::num_rows<WarpLayout>;
            default:
                assert(false && "Not implemented yet.");
                return -1;
        }
    }

    DEVICE int get_warp_offset() {
        int warp_row = warp_row_id();
        int warp_col = warp_col_id();

        int offset = 0;
        switch (kMode) {
            case WarpReuse::kCont:
            case WarpReuse::kCir:
                offset = warp_row * Shared::kRowStride * kTilePerWarpCol +
                         warp_col * Shared::kColStride * kTilePerWarpRow;
                break;
            case WarpReuse::kColReuseCont:
            case WarpReuse::kColReuseCir:
                offset = warp_col * Shared::kColStride * kTilePerWarpRow;
                break;
            case WarpReuse::kRowReuseCont:
            case WarpReuse::kRowReuseCir:
                offset = warp_row * Shared::kRowStride * kTilePerWarpCol;
                break;
            default:
                assert(false && "Not implemented yet.");
        }
        return offset;
    }

  private:
    using Shared = Shared_;
    using WarpLayout = WarpLayout_;
    static constexpr WarpReuse kMode = kMode_;
    // data type __half here is to instantiate the templated class `BaseShape`.
    // It does not affect shape-related information.
    using BaseShape = traits::BaseTileShape<__half>;

    constexpr static int kTilePerRow = Shared::kCols / BaseShape::kCols;
    constexpr static int kTilePerCol = Shared::kRows / BaseShape::kRows;

    constexpr static int kTilePerWarpRow =
        kTilePerRow / tl::num_cols<WarpLayout>;
    constexpr static int kTilePerWarpCol =
        kTilePerCol / tl::num_rows<WarpLayout>;

    // for row-major shared memory tile
    constexpr static int kRowStride1 =
        kTilePerCol / tl::num_rows<WarpLayout> * kTilePerRow;
    constexpr static int kColStride1 = kTilePerRow / tl::num_cols<WarpLayout>;

    using BaseTilesRowMajorLayout =
        cute::Layout<Shape<Int<kTilePerCol>, Int<kTilePerRow>>,
                     Stride<Int<kRowStride1>, Int<kColStride1>>>;
    BaseTilesRowMajorLayout base_tiles_row_major_;

    // for column-major shared memory tile
    constexpr static int kRowStride2 = kTilePerCol / tl::num_rows<WarpLayout>;
    constexpr static int kColStride2 =
        kTilePerRow / tl::num_cols<WarpLayout> * kTilePerCol;
    using BaseTilesColMajorLayout =
        cute::Layout<Shape<Int<kTilePerCol>, Int<kTilePerRow>>,
                     Stride<Int<kRowStride2>, Int<kColStride2>>>;
    BaseTilesColMajorLayout base_tiles_col_major_;
};
}  // namespace

template <typename WarpLayout, const WarpReuse kMode, typename Shared>
struct SharedOffsetHelper {
    /*
     * @brief In a thread block, warps are organized as 2-D matrices, each with
     * a row index and a column index. Given `threadIdx.x`, this function
     * calculates the row index of the current thread.
     */
    DEVICE int warp_row_id() { return helper_.warp_row_id(); }

    /*
     * @brief In a thread block, warps are organized as 2-D matrices, each with
     * a row index and a column index. Given `threadIdx.x`, this function
     * calculates the column index of the current thread.
     */
    DEVICE int warp_col_id() { return helper_.warp_col_id(); }

    DEVICE int get_warp_offset() { return helper_.get_warp_offset(); }

  private:
    // FIXME(ying): This hotfix addresses the current implementation's inability
    // to explicitly distinguish between shared memory's row-major or
    // column-major layout and global memory's layouts. However, this should be
    // fixed in the future.
    constexpr static bool kIsSharedLayout =
        (Shared::Layout::kRowStride == Shared::kCols &&
         Shared::Layout::kColStride == 1) ||
                (Shared::Layout::kRowStride == 1 &&
                 Shared::Layout::kColStride == Shared::kRows)
            ? false
            : true;
    using OffsetHelper =
        SharedOffsetHelperImpl<WarpLayout, kMode, Shared, kIsSharedLayout>;

    OffsetHelper helper_;
};

}  // namespace tilefusion::cell::copy::warp
