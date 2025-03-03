// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "traits/base.hpp"
#include "types/shared.hpp"
#include "types/tile_shape.hpp"

namespace tilefusion::cell {
namespace tl = tile_layout;

namespace {
/// @brief Helper for pretty printing a tile iterator's static shape-related
///        information. This printer works ONLY on the host.
struct STileIteratorPrettyPrinter {
    template <typename TileIterator>
    static HOST void print(std::ostream& out, const TileIterator& itr) {
        out << "ChunkShape = (" << TileIterator::kChunkRows << ", "
            << TileIterator::kChunkCols << "), stripe count = ("
            << TileIterator::sc0 << ", " << TileIterator::sc1 << ")";
    }
};
}  // namespace

/// @brief `SharedTileIterator` chunks a shared memory tile into smaller tiles
///         and iterates over these smaller sub-tiles.
/// @tparam Tile_: The type of the large tile to chunk.
/// @tparam ChunkShape_: The shape of the smaller tiles into which the large
///                      tile is partitioned (chunk shape).
template <class Tile_, class ChunkShape_>
class STileIterator {
  public:
    using Tile = Tile_;
    using DType = Tile::DType;
    using ChunkShape = ChunkShape_;
    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kChunkRows = dim_size<0, ChunkShape>;
    static constexpr int kChunkCols = dim_size<1, ChunkShape>;

    static_assert(Tile::kRows >= dim_size<0, ChunkShape>,
                  "Tile::kRows must be >= dim_size<0, ChunkShape>");
    static_assert(Tile::kCols >= dim_size<1, ChunkShape>,
                  "Tile::kCols must be >= dim_size<1, ChunkShape>");

    static constexpr int sc0 = Tile::kRows / kChunkRows;
    static constexpr int sc1 = Tile::kCols / kChunkCols;

    HOST_DEVICE STileIterator() : data_(nullptr) {}

    DEVICE STileIterator(DType* data) : data_(data) {}

    DEVICE STileIterator(const DType* data) : data_(const_cast<DType*>(data)) {}

    // Since a Tile is considered to be at most a 2D array, the iterator
    // traverses over these two dimensions. The current rules are:
    // 1. If the index is a 2D integer, this access is considered to be a
    //    single tile, hence it returns a Tile.
    // 2. If any part of the index is an underscore, this access is
    //    considered to be a slice, naturally it returns a TileIterator.
    DEVICE auto operator()(int i) {
        assert(data_);  // The iterator is not initialized.
        static_assert(sc0 == 1 || sc1 == 1,
                      "A single index is supported only when the strip count "
                      "of one of the iterator's dimensions is 1.");

        int x = sc0 == 1 ? 0 : i;
        int y = sc0 == 1 ? i : 0;

        using TileLayout =
            decltype(tl::make_shared_tile_layout<kChunkRows, kChunkCols,
                                                 kTileRowStride, kTileColStride,
                                                 Tile::kType>());

        using NewTile = SharedTile<DType, TileLayout, Tile::kSwizzled>;

        // TODO(KuangjuX): hotfix for `offset1` and `offset2`.
        int offset1 = x * (kChunkRows * Tile::kRowStride) +
                      y * kTilePerChunkCol * BaseShape::kCols;
        int offset2 = x * kTilePerChunkRow * BaseShape::kRows +
                      y * (Tile::kColStride * kChunkCols);
        int offset = Tile::kType == tl::Layout::kRowMajor ? offset1 : offset2;

        NewTile tile(data_ + offset, offset);
        return tile;
    }

    DEVICE auto operator()(int x, int y) {
        assert(false && "Not implemented yet.");
        return 0;
    }

    DEVICE auto operator()(int x, const Underscore& y) {
        assert(false && "Not implemented yet.");
        return 0;
    }

    DEVICE auto operator()(const Underscore& x, int y) {
        assert(false && "Not implemented yet.");
        return 0;
    }

    DEVICE auto to_tile() {
        Tile tile(data_);
        return tile;
    }

  private:
    static constexpr int kTilePerRow = Tile::kRows / BaseShape::kRows;
    static constexpr int kTilePerCol = Tile::kCols / BaseShape::kCols;

    static constexpr int kTilePerChunkRow = kChunkRows / BaseShape::kRows;
    static constexpr int kTilePerChunkCol = kChunkCols / BaseShape::kCols;

    // TODO(KuangjuX): hotfix for `kTileRowStride` and `kTileColStride`.
    static constexpr int kTileRowStride =
        Tile::kType == tl::Layout::kRowMajor ? Tile::kCols : 1;

    static constexpr int kTileColStride =
        Tile::kType == tl::Layout::kRowMajor ? 1 : Tile::kRows;

    DType* data_;
};

/// @brief Pretty printer for the static shape information of a TileIterator.
///        Note: This printer function works ONLY on the host.
template <typename TileShape, typename ChunkShape>
static HOST std::ostream& operator<<(
    std::ostream& out, const STileIterator<TileShape, ChunkShape>& itr) {
    STileIteratorPrettyPrinter::print(out, itr);
    return out;
}

}  // namespace tilefusion::cell
