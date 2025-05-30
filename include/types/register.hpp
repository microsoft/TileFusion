// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.hpp"
#include "types/layout.hpp"
#include "util/print.hpp"

namespace tilefusion::cell {
namespace tl = tile_layout;

namespace {
template <typename DType>
constexpr int get_rows = DType::kRows;

template <>
constexpr int get_rows<float> = 1;

template <>
constexpr int get_rows<__half> = 1;

#ifdef CUDA_FP8_AVAILABLE
template <>
constexpr int get_rows<__nv_fp8_e4m3> = 1;

template <>
constexpr int get_rows<__nv_fp8_e5m2> = 1;
#endif

template <typename DType>
constexpr int get_cols = DType::kCols;

template <>
constexpr int get_cols<float> = 1;

template <>
constexpr int get_cols<__half> = 1;

#ifdef CUDA_FP8_AVAILABLE
template <>
constexpr int get_cols<__nv_fp8_e4m3> = 1;

template <>
constexpr int get_cols<__nv_fp8_e5m2> = 1;
#endif

/// @brief Helper for pretty printing a register tile's static shape
///        information. This printer works ONLY on the host.
struct RegTilePrettyPrinter {
    template <typename Tile>
    static HOST void print(std::ostream& out, const Tile& tile) {
        out << layout_type_to_str(Tile::kType) << "("
            << Tile::kRows * get_rows<typename Tile::DType> << ", "
            << Tile::kCols * get_cols<typename Tile::DType> << ")";
    }
};

DEVICE void clear(float* data, int numel) {
    memset((void*)data, 0, sizeof(float) * numel);
}

DEVICE void clear(__half* data, int numel) {
    memset((void*)data, 0, sizeof(__half) * numel);
}

#ifdef CUDA_FP8_AVAILABLE
DEVICE void clear(__nv_fp8_e4m3* data, int numel) {
    memset((void*)data, 0, sizeof(__nv_fp8_e4m3) * numel);
}

DEVICE void clear(__nv_fp8_e5m2* data, int numel) {
    memset((void*)data, 0, sizeof(__nv_fp8_e5m2) * numel);
}
#endif

template <typename DType>
DEVICE void clear_impl(DType* data, int numel) {
    for (int i = 0; i < numel; ++i) {
        clear(data[i].mutable_data(), 8);
    }
}
}  // namespace

template <typename Element_, typename Layout_>
class RegTile {
  public:
    using DType = Element_;
    using Layout = Layout_;

    static constexpr int kNumel = Layout::kNumel;
    static constexpr int kRows = Layout::kRows;
    static constexpr int kCols = Layout::kCols;

    // FIXME(haruhi): this is a hack to fix the layout type deduction for when
    // the shape is 1x1. This is a workaround. Fix this to be more robust.
    static constexpr tl::Layout kType = tl::layout_type<typename DType::Layout>;

    DEVICE RegTile() : layout_(Layout{}) {
        memset((void*)data_, 0, sizeof(data_));
    }

    DEVICE DType* mutable_data() { return (DType*)data_; }

    DEVICE const DType* data() const { return (DType*)data_; }

    DEVICE const Layout& layout() const { return layout_; }

    // for write access
    DEVICE DType& operator()(int x, int y) { return data_[layout_(x, y)]; }

    // for read access
    DEVICE const DType& operator()(int x, int y) const {
        return data_[layout_(x, y)];
    }

    DEVICE void dump_value() const {
        print_tile(const_cast<DType*>(data_), layout_);
    }

    DEVICE void clear() { clear_impl<DType>(data_, kNumel); }

  private:
    DType data_[kNumel];
    Layout layout_;
};

template <typename Element>
using BaseTileRowMajor = RegTile<Element, tl::RowMajor<2, 4>>;

template <typename Element>
using BaseTileColMajor = RegTile<Element, tl::ColMajor<4, 2>>;

/// @brief Pretty printer for the static shape information of a register tile.
///        Note: This printer function works ONLY on the host. The current
///        implementation prints a flattened layout and only displays the outer
///        name of the tile layout.
/// @param T: element type, which must be a `RegTile` rather than a basic
///           element type like float
/// @param Layout: tile layout
template <typename T, typename Layout>
static HOST std::ostream& operator<<(std::ostream& out,
                                     const RegTile<T, Layout>& tile) {
    RegTilePrettyPrinter::print(out, tile);
    return out;
}

}  // namespace tilefusion::cell
