// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "types/layout.hpp"
#include "types/tile_shape.hpp"

namespace tilefusion::cell {
namespace tl = tile_layout;

namespace {
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

/// @brief Determine the automatic shape of a single warp based on the shape of
///        the entire tile. The final warp tile shape is multiple of this atomic
///        shape.
template <typename DType, typename TileShape, const tl::Layout kType>
struct WarpBaseTileShape;

template <typename DType, typename TileShape>
struct WarpBaseTileShape<DType, TileShape, tl::Layout::kRowMajor> {
    using AccessInfo = traits::AccessBase<DType>;

    static constexpr int kTileRows = dim_size<0, TileShape>;
    static constexpr int kTileCols = dim_size<1, TileShape>;

    // In a row-major layout, columns are the contiguous dimension in memory. We
    // enforce the use of 128-bit vectorized instructions for data loading by a
    // single thread. This implies that the minimum number of columns should be
    // at least 128 bits.
    static constexpr int kMinCols =
        AccessInfo::kAccessInBits / (sizeof(DType) * 8);

    static_assert(kTileCols >= kMinCols, "The number of columns is too small.");

    static_assert(kTileCols < AccessInfo::kExpectedSize ||
                      (kTileCols >= AccessInfo::kExpectedSize &&
                       kTileCols % AccessInfo::kExpectedSize == 0),
                  "The current implementation requires that the number of "
                  "columns of the tile be divisible by the cache line width.");

    static constexpr int kCols = kTileCols >= AccessInfo::kExpectedSize
                                     ? AccessInfo::kExpectedSize
                                     : kTileCols;

    // number of columns in a warp
    static constexpr int kColThreads = kCols / AccessInfo::kNumPerAccess;
    static_assert(WARP_SIZE % kColThreads == 0,
                  "Fail to infer warp thread layout.");
    static constexpr int kRowThreads = WARP_SIZE / kColThreads;

    static constexpr int kRows = kRowThreads;
    static_assert(kTileRows % kRowThreads == 0,
                  "The number of rows of the tile isn't evenly divisible by "
                  "the number of threads in a column.");

    static constexpr tl::Layout kType = tl::Layout::kRowMajor;
    static constexpr int kNumel = kRows * kCols;

    using WarpThreadLayout = tl::RowMajor<kRowThreads, kColThreads>;
};

template <typename DType, typename TileShape>
struct WarpBaseTileShape<DType, TileShape, tl::Layout::kColMajor> {
    using AccessInfo = traits::AccessBase<DType>;

    static constexpr int kTileRows = dim_size<0, TileShape>;
    static constexpr int kTileCols = dim_size<1, TileShape>;

    // In a column-major layout, columns are the contiguous dimension in memory.
    // We enforce the use of 128-bit vectorized instructions for data loading by
    // a single thread. This implies that the minimum number of columns should
    // be at least 128 bits.
    static constexpr int kMinRows =
        AccessInfo::kAccessInBits / (sizeof(DType) * 8);

    static_assert(kTileRows >= kMinRows, "The number of rows is too small.");

    static_assert(kTileRows < AccessInfo::kExpectedSize ||
                      (kTileRows >= AccessInfo::kExpectedSize &&
                       kTileRows % AccessInfo::kExpectedSize == 0),
                  "The current implementation requires that the number of "
                  "rows of the tile be divisible by the cache line width.");

    static constexpr int kRows = kTileRows >= AccessInfo::kExpectedSize
                                     ? AccessInfo::kExpectedSize
                                     : kTileRows;

    // number of rows in a warp
    static constexpr int kRowThreads = kRows / AccessInfo::kNumPerAccess;
    static_assert(WARP_SIZE % kRowThreads == 0,
                  "Fail to infer warp thread layout.");
    static constexpr int kColThreads = WARP_SIZE / kRowThreads;

    static constexpr int kCols = kColThreads;
    static_assert(kTileCols % kColThreads == 0,
                  "The number of columns of the tile isn't evenly divisible by "
                  "the number of threads in a row.");

    static constexpr tl::Layout kType = tl::Layout::kColMajor;
    static constexpr int kNumel = kRows * kCols;

    using WarpThreadLayout = tl::ColMajor<kRowThreads, kColThreads>;
};

template <typename Element, const tl::Layout kType>
using DefaultBaseTile = WarpBaseTileShape<Element, TileShape<16, 16>, kType>;

/// @brief Pretty printer for the static shape information of a
///        `WarpBaseTileShape`. Note: This printer function works ONLY on the
///        host.
template <typename DType, typename TileShape, const tl::Layout kType>
static HOST std::ostream& operator<<(
    std::ostream& out, const WarpBaseTileShape<DType, TileShape, kType>& tile) {
    BaseTilePrettyPrinter::print(out, tile);
    return out;
}
}  // namespace tilefusion::cell
