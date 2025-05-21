// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "traits/base.hpp"
#include "types/layout.hpp"
#include "types/swizzle.hpp"
#include "util/print.hpp"

namespace tilefusion::cell {
namespace tl = tile_layout;

using namespace tilefusion::traits;

namespace {

/// @brief Helper for pretty printing a SharedTile's static shape-related
///        information. This printer works ONLY on the host.
struct SharedTilePrettyPrinter {
    template <typename Shared>
    static HOST void print(std::ostream& out, const Shared& tile) {
        // parameter `tile` here is not used
        auto swizzled = Shared::kSwizzled ? "swizzled" : "non-swizzled";
        out << "\t" << typename Shared::Layout{} << ", Swizzled = " << swizzled;
    }
};

}  // namespace

template <typename Element_, typename Layout_, const bool kSwizzled_ = false,
          const int SwizzleBytes_ = 128>
class SharedTile {
  public:
    using DType = Element_;
    using Layout = Layout_;

    static constexpr int kRows = Layout::kRows;
    static constexpr int kCols = Layout::kCols;
    static constexpr int kRowStride = Layout::kRowStride;
    static constexpr int kColStride = Layout::kColStride;

    static constexpr tl::Layout kType = Layout::kType;
    static constexpr int kNumel = Layout::kNumel;

    static constexpr bool isRowMajor = tl::is_row_major<Layout>::value;

    static constexpr int SwizzleBytes = SwizzleBytes_;
    static constexpr bool kSwizzled = kSwizzled_;

    using SwizzleBaseShape = SwizzleBaseTileShape<DType, SwizzleBytes>;

    static constexpr int kSwizzleRows =
        isRowMajor ? SwizzleBaseShape::kRows : SwizzleBaseShape::kCols;
    static constexpr int kSwizzleCols =
        isRowMajor ? SwizzleBaseShape::kCols : SwizzleBaseShape::kRows;

    using NonSwizzled = std::conditional_t<
        isRowMajor, tl::MatrixLayout<kSwizzleRows, kSwizzleCols, kRowStride, 1>,
        tl::MatrixLayout<kSwizzleRows, kSwizzleCols, 1, kColStride>>;

    using Swizzled = SwizzledLayout<
        NonSwizzled,
        Swizzle<SwizzleBaseShape::B, SwizzleBaseShape::M, SwizzleBaseShape::S>,
        kType>;

    using InTileLayout = std::conditional_t<kSwizzled, Swizzled, NonSwizzled>;

    using TileLayout = std::conditional_t<
        isRowMajor,
        tl::MatrixLayout<kRows / kSwizzleRows, kCols / kSwizzleCols,
                         kRowStride * kSwizzleRows, kSwizzleCols>,
        tl::MatrixLayout<kRows / kSwizzleRows, kCols / kSwizzleCols,
                         kSwizzleRows, kColStride * kSwizzleCols>>;

    InTileLayout in_tile_layout_;
    TileLayout tile_layout_;

    // This Ctor is to enable the use of the pretty printer of SharedTile
    // in the host code.
    DEVICE SharedTile() : data_(nullptr), layout_(Layout{}), offset_(0) {}

    DEVICE SharedTile(DType* data)
        : data_(data), layout_(Layout{}), offset_(0) {}

    DEVICE SharedTile(const DType* data)
        : data_(const_cast<DType*>(data)), layout_(Layout{}), offset_(0) {}

    DEVICE SharedTile(DType* data, int offset)
        : data_(data), layout_(Layout{}), offset_(offset) {}

    DEVICE DType* mutable_data() { return data_; }

    DEVICE const DType* data() const { return data_; }

    DEVICE int get_offset() const { return offset_; }

    HOST_DEVICE const Layout& layout() const { return layout_; }

    // for write access
    DEVICE DType& operator()(int x, int y) { return data_[layout_(x, y)]; }

    // for read access
    DEVICE
    const DType& operator()(int x, int y) const { return data_[layout_(x, y)]; }

    DEVICE int fetch_physical_offset(int offset) {
        return swizzle_offset(offset);
    }

    DEVICE void dump_value() { print_tile(data_, layout_); }

  private:
    DType* data_;
    Layout layout_;
    int offset_;

    DEVICE int2 swizzle_tile_id(int offset) {
        int swizzle_tile_row = kType == tl::Layout::kRowMajor
                                   ? (offset / kRowStride) / kSwizzleRows
                                   : (offset % kColStride) / kSwizzleRows;

        int swizzle_tile_col = kType == tl::Layout::kRowMajor
                                   ? (offset % kRowStride) / kSwizzleCols
                                   : (offset / kColStride) / kSwizzleCols;

        return make_int2(swizzle_tile_row, swizzle_tile_col);
    }

    DEVICE int2 in_swizzle_tile_id(int offset) {
        int row = kType == tl::Layout::kRowMajor ? offset / kRowStride
                                                 : offset % kColStride;
        int col = kType == tl::Layout::kRowMajor ? offset % kRowStride
                                                 : offset / kColStride;

        int in_swizzle_tile_row = row % kSwizzleRows;
        int in_swizzle_tile_col = col % kSwizzleCols;

        return make_int2(in_swizzle_tile_row, in_swizzle_tile_col);
    }

    DEVICE int swizzle_offset(int offset) {
        auto tile_id = swizzle_tile_id(offset);
        auto in_tile_id = in_swizzle_tile_id(offset);
        int swizzle_tile_offset = tile_layout_(tile_id.x, tile_id.y);
        int in_swizzle_tile_offset =
            in_tile_layout_(in_tile_id.x, in_tile_id.y);

        return swizzle_tile_offset + in_swizzle_tile_offset;
    }
};

/// @brief Pretty printer for the static shape information of a SharedTile.
///        Note: This printer function works ONLY on the host.
template <typename Element, typename Layout, const bool kSwizzled>
static HOST std::ostream& operator<<(
    std::ostream& out, const SharedTile<Element, Layout, kSwizzled>& tile) {
    SharedTilePrettyPrinter::print(out, tile);
    return out;
}

}  // namespace tilefusion::cell
