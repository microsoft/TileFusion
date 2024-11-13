// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * @file warp.hpp
 * @brief This file contains **Warp** related operations for copy.
 */
#pragma once

#include "cell/copy/constants.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tilefusion::cell::copy::warp {

using namespace tilefusion::cell::traits;

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

template <typename WarpLayout_, const WarpReuse kMode_>
struct CopyBase {
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

        return detail::warp_offset_impl<kMode>(warp_row_id(), warp_col_id(),
                                               kWarpRstride, kWarpCstride);
    }

    // @brief This function returns the number of times a `BaseTile` is executed
    //        along the direction of the shared memory row.
    template <typename BaseTile, const int kRows>
    DEVICE static constexpr int row_exec_count() {
        const int kWarpsPerRow = tl::num_rows<WarpLayout>;

        static_assert(
            kRows % BaseTile::kRows == 0,
            "The current implementation requires that the number of shared "
            "memory rows be divisible by the base tile row.\n");

        int count = 0;
        switch (kMode) {
            // Warps in the same columns (`warps_per_row` in total) repeatedly
            // load the shared memory rows. Therefore, `row_exec` is not divided
            // by warps_per_row.
            case WarpReuse::kColReuseCont:
            case WarpReuse::kColReuseCir:
                count = kRows / BaseTile::kRows;
                break;
            default:  // Cont, Cir, RowReuseCont, RowReuseCir hit this case.
                count = kRows / BaseTile::kRows / kWarpsPerRow;
                break;
        }

        // Check to ensure that the count is not zero, which could be caused by
        // an incorrect combination of shared memory tile shape and warp layout.
        // TODO: This should actually be a static assert, but we're currently
        // using a runtime assert for implementation issues.
        assert(count);
        return count;
    }

    template <typename BaseTile, const int kCols>
    DEVICE static constexpr int col_exec_count() {
        const int kWarpsPerCol = tl::num_cols<WarpLayout>;

        static_assert(
            kCols % BaseTile::kCols == 0,
            "The number of shared memory columns must be divisible by the base "
            "tile column.\n");

        int count = 0;
        switch (kMode) {
            // Warps in the same rows (`warps_per_col` in total) repeatedly load
            // the shared memory columns. Therefore, `col_exec` is not divided
            // by `warps_per_col`.
            case WarpReuse::kRowReuseCont:
            case WarpReuse::kRowReuseCir:
                count = kCols / BaseTile::kCols;
                break;
            default:  // Cont, Cir, ColReuseCont, ColReuseCir hit this case.
                count = kCols / BaseTile::kCols / kWarpsPerCol;
                break;
        }

        // Check to ensure that the count is not zero, which could be caused by
        // an incorrect combination of shared memory tile shape and warp layout.
        assert(count);
        return count;
    }

  private:
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
        int wid = threadIdx.x / warpSize;

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
        int wid = threadIdx.x / warpSize;

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
};

namespace {
// This is a hotfix for the current implementation. This anonymous namespace is
// not intended to be exposed outside this header file.
}

template <typename WarpLayout, const WarpReuse kMode_, const tl::Layout kType,
          typename Shared_>
struct SharedOffsetHelper;

template <typename WarpLayout, const WarpReuse kMode_, typename Shared_>
struct SharedOffsetHelper<WarpLayout, kMode_, tl::Layout::kRowMajor, Shared_> {
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

        // switch (kMode) {
        //     case WarpReuse::kCont:
        //     case WarpReuse::kCir:
        //     case WarpReuse::kRowReuseCont:
        //     case WarpReuse::kRowReuseCir:
        //         warp_row = warp_id / tl::num_cols<WarpLayout>;
        //         break;
        //     case WarpReuse::kColReuseCont:
        //     case WarpReuse::kColReuseCir:
        //         break;
        //     default:
        //         assert(false && "Not implemented yet.");
        // }
        // return warp_row;
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

        // switch (kMode) {
        //     case WarpReuse::kCont:
        //     case WarpReuse::kCir:
        //     case WarpReuse::kColReuseCont:
        //     case WarpReuse::kColReuseCir:
        //         warp_col = warp_id % tl::num_cols<WarpLayout>;
        //         break;
        //     case WarpReuse::kRowReuseCont:
        //     case WarpReuse::kRowReuseCir:
        //         break;
        //     default:
        //         assert(false && "Not implemented yet.");
        // }
        // return warp_col;
    }

    DEVICE int get_warp_offset() {
        int warp_row = warp_row_id();
        int warp_col = warp_col_id();

        int offset = 0;
        switch (kMode) {
            case WarpReuse::kCont:
            case WarpReuse::kCir:
                offset = warp_row * Shared::kRowStride +
                         warp_col * Shared::kColStride;
                break;
            case WarpReuse::kColReuseCont:
            case WarpReuse::kColReuseCir:
                offset = warp_col * Shared::kColStride;
                break;
            case WarpReuse::kRowReuseCont:
            case WarpReuse::kRowReuseCir:

                offset = warp_row * Shared::kRowStride;
                break;
            default:
                assert(false && "Not implemented yet.");
        }

        if (thread(32)) {
            printf("Shared::kRowStride = %d, Shared::kColStride = %d\n",
                   Shared::kRowStride, Shared::kColStride);
            printf("warp_row = %d, warp_col = %d, offset = %d\n", warp_row,
                   warp_col, offset);
        }

        return offset;

        // int tile_id = Shared::kType == tl::Layout::kRowMajor
        //                   ? base_tiles_row_major_(warp_row_id(),
        //                   warp_col_id()) :
        //                   base_tiles_col_major_(warp_row_id(),
        //                   warp_col_id());
        // return tile_id * BaseShape::kNumel;
    }

  private:
    using Shared = Shared_;
    // data type __half here is to instantiate the templated class `BaseShape`.
    // It does not affect shape-related information.
    using BaseShape = traits::BaseTileShape<__half>;

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

    static constexpr WarpReuse kMode = kMode_;

    constexpr static int kBaseTilePerRow = Shared::kRows / BaseShape::kRows;
    constexpr static int kBaseTilePerCol = Shared::kCols / BaseShape::kCols;

    // for row-major shared memory tile
    constexpr static int kRowStride1 =
        kBaseTilePerRow / tl::num_rows<WarpLayout> * kBaseTilePerCol;
    constexpr static int kColStride1 =
        kBaseTilePerCol / tl::num_cols<WarpLayout>;

    using BaseTilesRowMajorLayout =
        cute::Layout<Shape<Int<kBaseTilePerRow>, Int<kBaseTilePerCol>>,
                     Stride<Int<kRowStride1>, Int<kColStride1>>>;
    BaseTilesRowMajorLayout base_tiles_row_major_;

    // for column-major shared memory tile
    constexpr static int kRowStride2 =
        kBaseTilePerRow / tl::num_rows<WarpLayout>;
    constexpr static int kColStride2 =
        kBaseTilePerCol / tl::num_cols<WarpLayout> * kBaseTilePerRow;
    using BaseTilesColMajorLayout =
        cute::Layout<Shape<Int<kBaseTilePerRow>, Int<kBaseTilePerCol>>,
                     Stride<Int<kRowStride2>, Int<kColStride2>>>;
    BaseTilesColMajorLayout base_tiles_col_major_;
};

}  // namespace tilefusion::cell::copy::warp
