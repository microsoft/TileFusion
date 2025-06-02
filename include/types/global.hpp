// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "types/layout.hpp"
#include "util/print.hpp"

namespace tilefusion {
namespace tl = tile_layout;

namespace {
/// @brief Helper for pretty printing a GlobalTile's static shape-related
///        information. This printer works ONLY on the host.
struct GlobalTilePrettyPrinter {
  template <typename Global>
  static HOST void print(std::ostream& out, const Global& tile) {
    // parameter `tile` here is not used
    out << "GlobalTile {" << std::endl
        << "  " << typename Global::Layout{} << std::endl
        << "}";
  }
};
}  // namespace

template <typename Element_, typename Layout_>
struct GlobalTile {
  using DType = Element_;
  using Layout = Layout_;

  static constexpr int kNumel = tl::get_numel<Layout>;

  static constexpr int kRows = tl::num_rows<Layout>;
  static constexpr int kCols = tl::num_cols<Layout>;

  static constexpr int kRowStride = tl::row_stride<Layout>;
  static constexpr int kColStride = tl::col_stride<Layout>;

  static constexpr tl::Layout kType = tl::layout_type<Layout>;

  // This Ctor is to enable the use of the pretty printer of SharedTile in the
  // host code.
  HOST GlobalTile() : data_(nullptr), layout_(Layout{}) {}

  DEVICE GlobalTile(DType* data) : data_(data), layout_(Layout{}) {}

  DEVICE GlobalTile(const DType* data)
      : data_(const_cast<DType*>(data)), layout_(Layout{}) {}

  DEVICE DType* mutable_data() { return data_; }

  DEVICE const DType* data() const { return data_; }

  HOST_DEVICE const Layout& layout() const { return layout_; }

  // for write access
  DEVICE DType& operator()(int x, int y) { return data_[layout_(x, y)]; }

  // for read access
  DEVICE
  const DType& operator()(int x, int y) const { return data_[layout_(x, y)]; }

  DEVICE void dump_value() { util::print_tile(data_, layout_); }

 private:
  DType* data_;
  Layout layout_;
};

/// @brief Pretty printer for the static shape information of a SharedTile.
///        Note: This printer function works ONLY on the host.
template <typename Element, typename Layout>
static HOST std::ostream& operator<<(std::ostream& out,
                                     const GlobalTile<Element, Layout>& tile) {
  GlobalTilePrettyPrinter::print(out, tile);
  return out;
}

}  // namespace tilefusion
