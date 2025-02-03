// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "types/base_tile.hpp"
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

        out << "\t" << typename Shared::Layout{} << ", Swizzled = " << swizzled
            << ", " << std::endl
            << "\t" << typename Shared::BaseShape{};
    }
};
}  // namespace

// TODO(ying): Make siwzzling a layout function instead of a boolean value.
template <typename Element_, typename Layout_, const bool kSwizzled_ = true,
          typename BaseShape_ = DefaultBaseTile<Element_, Layout_::kType>>
class SharedTile {
  public:
    using DType = Element_;
    using Layout = Layout_;
    using BaseShape = BaseShape_;

    static constexpr int kNumel = Layout::kNumel;

    static constexpr int kRows = Layout::kRows;
    static constexpr int kCols = Layout::kCols;
    static constexpr int kRowStride = Layout::kRowStride;
    static constexpr int kColStride = Layout::kColStride;

    static constexpr tl::Layout kType = Layout::kType;
    static constexpr bool kSwizzled = kSwizzled_;

    static_assert(kRows % BaseShape::kRows == 0,
                  "The number of shared memory rows must be divisible by "
                  "the base tile row.");
    static_assert(kCols % BaseShape::kCols == 0,
                  "The number of shared memory columns must be divisible "
                  "by the base tile column.");

    // This Ctor is to enable the use of the pretty printer of SharedTile in the
    // host code.
    HOST SharedTile() : data_(nullptr), layout_(Layout{}) {}

    DEVICE SharedTile(DType* data) : data_(data), layout_(Layout{}) {}

    DEVICE DType* mutable_data() { return data_; }

    DEVICE const DType* data() const { return data_; }

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
};

/// @brief Pretty printer for the static shape information of a SharedTile.
///        Note: This printer function works ONLY on the host.
template <typename Element, typename Layout, const bool kSwizzled,
          typename BaseShape>
static HOST std::ostream& operator<<(
    std::ostream& out,
    const SharedTile<Element, Layout, kSwizzled, BaseShape>& tile) {
    SharedTilePrettyPrinter::print(out, tile);
    return out;
}

}  // namespace tilefusion::cell
