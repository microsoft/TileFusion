// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cell/compute/gemm.hpp"
#include "types/shared.hpp"
#include "types/tile_shape.hpp"

#include <iostream>

namespace tilefusion::cell {
namespace tl = tile_layout;
using namespace compute;

namespace {
/// @brief Helper for pretty printing a tile iterator's static shape-related
///        information. This printer works ONLY on the host.
struct STileIteratorPrettyPrinter {
    template <typename TileIterator>
    static HOST void print(std::ostream& out, const TileIterator& itr) {
        out << "SharedTileIterator {" << std::endl
            << "  ChunkShape = (" << TileIterator::kChunkRows << ", "
            << TileIterator::kChunkCols << "), stripe count = ("
            << TileIterator::sc0 << ", " << TileIterator::sc1 << ")"
            << std::endl
            << "}";
    }
};

/// @brief Helper for pretty printing STileIterator2's static shape information
struct STileIterator2PrettyPrinter {
    template <typename TileIterator>
    static HOST void print(std::ostream& out, const TileIterator& itr) {
        out << "SharedTileIterator2 {" << std::endl
            << "  ChunkShape = (" << TileIterator::kChunkRows << ", "
            << TileIterator::kChunkCols << "), stripe count = ("
            << TileIterator::sc0 << ", " << TileIterator::sc1 << ")"
            << std::endl
            << "}";
    }
};

/// @brief Type trait to detect if a layout is a BlockMatrxLayout
template <typename Layout>
struct is_block_layout : std::false_type {};

template <typename OuterLayout, typename InnerLayout>
struct is_block_layout<tl::BlockMatrxLayout<OuterLayout, InnerLayout>>
    : std::true_type {};

template <typename Layout>
static constexpr bool is_block_layout_v = is_block_layout<Layout>::value;

/// @brief Helper to create the appropriate sub-tile layout type
template <typename TileLayout, int kChunkRows, int kChunkCols,
          bool IsBlockLayout = is_block_layout_v<TileLayout>>
struct SubTileLayoutCreator;

/// @brief Specialization for BlockMatrxLayout. For block layouts, we need to
///        preserve the block structure
template <typename TileLayout, int kChunkRows, int kChunkCols>
struct SubTileLayoutCreator<TileLayout, kChunkRows, kChunkCols, true> {
    using OuterLayout =
        tl::MatrixLayout<kChunkRows, kChunkCols, TileLayout::kRowStride,
                         TileLayout::kColStride, TileLayout::kType>;
    using type =
        tl::BlockMatrxLayout<OuterLayout, typename TileLayout::InnerLayout>;
};

/// @brief Specialization for simple MatrixLayout
template <typename TileLayout, int kChunkRows, int kChunkCols>
struct SubTileLayoutCreator<TileLayout, kChunkRows, kChunkCols, false> {
    using type =
        tl::MatrixLayout<kChunkRows, kChunkCols, TileLayout::kRowStride,
                         TileLayout::kColStride, TileLayout::kType>;
};

template <typename TileLayout, int kChunkRows, int kChunkCols>
using SubTileLayout_t =
    typename SubTileLayoutCreator<TileLayout, kChunkRows, kChunkCols>::type;

}  // namespace

/// @brief SharedTileIterator chunks a shared memory tile into smaller tiles
///        and iterates over these smaller sub-tiles.
/// @param Tile_ The type of the large tile to chunk
/// @param ChunkShape_ The shape of the smaller tiles (chunk shape)
template <class Tile_, class ChunkShape_>
class STileIterator {
  public:
    using Tile = Tile_;
    using DType = Tile::DType;
    using ChunkShape = ChunkShape_;

    // FIXME(ying): a hotfix. The akwared dependencies on mma will be removed
    // in future refactor.
    using MmaAtom =
        compute::MmaAtom<__half, __half, __half, compute::MMA_ATOM_16x16x16>;
    using BaseShape = typename MmaAtom::BaseTile;

    static constexpr int kChunkRows = dim_size<0, ChunkShape>;
    static constexpr int kChunkCols = dim_size<1, ChunkShape>;

    static_assert(Tile::kRows >= kChunkRows,
                  "Tile::kRows must be >= kChunkRows");
    static_assert(Tile::kCols >= kChunkCols,
                  "Tile::kCols must be >= kChunkCols");

    static constexpr int sc0 = Tile::kRows / kChunkRows;
    static constexpr int sc1 = Tile::kCols / kChunkCols;

    HOST_DEVICE STileIterator() : data_(nullptr) {}

    DEVICE explicit STileIterator(DType* data) : data_(data) {}

    DEVICE explicit STileIterator(const DType* data)
        : data_(const_cast<DType*>(data)) {}

    /// @brief Access a single sub-tile by linear index
    /// @param i Linear index of the sub-tile
    /// @return A new tile representing the sub-tile
    DEVICE auto operator()(int i) {
        assert(data_ != nullptr);

        const int x = sc0 == 1 ? 0 : i;
        const int y = sc0 == 1 ? i : 0;

        using TileLayout = tl::MatrixLayout<kChunkRows, kChunkCols,
                                            kTileRowStride, kTileColStride>;
        using NewTile =
            SharedTile<DType, TileLayout, Tile::kSwizzled, Tile::SwizzleBytes>;

        const int offset = compute_offset(x, y);
        return NewTile(data_ + offset, offset);
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

    /// @brief Convert back to the original tile
    DEVICE auto to_tile() { return Tile(data_); }

  private:
    // pre-compute values
    static constexpr int kTilePerRow = Tile::kRows / BaseShape::kRows;
    static constexpr int kTilePerCol = Tile::kCols / BaseShape::kCols;

    static constexpr int kTilePerChunkRow = kChunkRows / BaseShape::kRows;
    static constexpr int kTilePerChunkCol = kChunkCols / BaseShape::kCols;

    static constexpr bool kIsRowMajor = Tile::kType == tl::Layout::kRowMajor;

    static constexpr int kTileRowStride = kIsRowMajor ? Tile::kCols : 1;
    static constexpr int kTileColStride = kIsRowMajor ? 1 : Tile::kRows;

    /// @brief Compute memory offset for sub-tile at position (x, y)
    DEVICE int compute_offset(int x, int y) const {
        if constexpr (kIsRowMajor) {
            return x * (kChunkRows * Tile::kRowStride) +
                   y * kTilePerChunkCol * BaseShape::kCols;
        } else {
            return x * kTilePerChunkRow * BaseShape::kRows +
                   y * (Tile::kColStride * kChunkCols);
        }
    }

    DType* data_;
};

/// @brief Pretty printer for STileIterator
template <typename TileShape, typename ChunkShape>
static HOST std::ostream& operator<<(
    std::ostream& out, const STileIterator<TileShape, ChunkShape>& itr) {
    STileIteratorPrettyPrinter::print(out, itr);
    return out;
}

/// @brief Advanced SharedTileIterator with better block layout support
/// @param Tile_ The type of the large tile to chunk
/// @param ChunkShape_ The shape of the smaller tiles (chunk shape)
template <class Tile_, class ChunkShape_>
class STileIterator2 {
  public:
    using Tile = Tile_;
    using DType = Tile::DType;
    using ChunkShape = ChunkShape_;

    static constexpr int kChunkRows = dim_size<0, ChunkShape>;
    static constexpr int kChunkCols = dim_size<1, ChunkShape>;

    static_assert(
        Tile::kRows >= kChunkRows && Tile::kRows % kChunkRows == 0,
        "Tile::kRows must be >= kChunkRows and divisible by kChunkRows");
    static_assert(
        Tile::kCols >= kChunkCols && Tile::kCols % kChunkCols == 0,
        "Tile::kCols must be >= kChunkCols and divisible by kChunkCols");

    static constexpr int sc0 = Tile::kRows / kChunkRows;
    static constexpr int sc1 = Tile::kCols / kChunkCols;

    HOST_DEVICE STileIterator2() : tile_(nullptr), data_(nullptr) {}

    DEVICE explicit STileIterator2(Tile* tile)
        : tile_(tile), data_(const_cast<DType*>(tile->data())) {}

    /// @brief Access a single sub-tile by linear index
    /// @param i Linear index of the sub-tile
    /// @return A new tile representing the sub-tile
    DEVICE auto operator()(int i) {
        assert(tile_ != nullptr && data_ != nullptr);

        // A tile is partitioned into sub-tiles along the row or column
        // dimension. `x` and `y` are the indices of the sub-tile in the
        // row and column dimension, respectively.
        const int x = sc0 == 1 ? 0 : i;
        const int y = sc0 == 1 ? i : 0;

        using TileLayout = SubTileLayout_t<Layout, kChunkRows, kChunkCols>;
        using NewTile = SharedTile<DType, TileLayout>;

        const int offset = compute_offset(x, y);
        return NewTile(data_ + offset);
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

    /// @brief Convert back to the original tile
    DEVICE auto to_tile() {
        assert(tile_ != nullptr);
        return *tile_;
    }

  private:
    using Layout = typename Tile::Layout;
    static constexpr bool kIsBlockLayout = is_block_layout_v<Layout>;

    // Compute stride multipliers based on layout type
    static constexpr int kRowCount = []() {
        if constexpr (kIsBlockLayout) {
            return kChunkRows / Layout::InnerLayout::kRows;
        } else {
            return kChunkRows;
        }
    }();

    static constexpr int kColCount = []() {
        if constexpr (kIsBlockLayout) {
            return kChunkCols / Layout::InnerLayout::kCols;
        } else {
            return kChunkCols;
        }
    }();

    static constexpr int kRowStride = Layout::kRowStride * kRowCount;
    static constexpr int kColStride = Layout::kColStride * kColCount;

    /// @brief Compute memory offset for sub-tile at position (x, y)
    DEVICE int compute_offset(int x, int y) const {
        return x * kRowStride + y * kColStride;
    }

    Tile* tile_;
    DType* data_;
};

/// @brief Pretty printer for STileIterator2
template <class Tile_, class ChunkShape_>
static HOST std::ostream& operator<<(
    std::ostream& out, const STileIterator2<Tile_, ChunkShape_>& itr) {
    STileIterator2PrettyPrinter::print(out, itr);
    return out;
}
}  // namespace tilefusion::cell
