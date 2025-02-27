// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "types/layout.hpp"
#include "util/print.hpp"

namespace tilefusion::cell {
namespace tl = tile_layout;

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

template <typename Element_, typename Layout_, const bool kSwizzled_ = false>
class SharedTile {
  public:
    using DType = Element_;
    using Layout = Layout_;

    static constexpr int kNumel = Layout::kNumel;

    static constexpr int kRows = Layout::kRows;
    static constexpr int kCols = Layout::kCols;
    static constexpr int kRowStride = Layout::kRowStride;
    static constexpr int kColStride = Layout::kColStride;

    static constexpr tl::Layout kType = Layout::kType;

    static constexpr bool kSwizzled = kSwizzled_;

    // This Ctor is to enable the use of the pretty printer of SharedTile in the
    // host code.
    HOST SharedTile() : data_(nullptr), layout_(Layout{}), offset_(0) {}

    DEVICE SharedTile(DType* data)
        : data_(data), layout_(Layout{}), offset_(0) {}

    DEVICE SharedTile(DType* data, int offset)
        : data_(data), layout_(Layout{}), offset_(offset) {}

    DEVICE DType* mutable_data() { return data_; }

    DEVICE const DType* data() const { return data_; }

    DEVICE int get_offset() const { return offset_; }

    DEVICE void move_tile(int offset) { offset_ += offset; }

    HOST_DEVICE const Layout& layout() const { return layout_; }

    // for write access
    DEVICE DType& operator()(int x, int y) { return data_[layout_(x, y)]; }

    // for read access
    DEVICE
    const DType& operator()(int x, int y) const { return data_[layout_(x, y)]; }

    DEVICE void dump_value() { print_tile(data_, layout_); }

  private:
    DType* data_;
    Layout layout_;
    int offset_;
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
