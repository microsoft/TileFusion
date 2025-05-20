// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "config.hpp"
#include "cuda_utils.hpp"
#include "traits/base.hpp"
#include "util/math_utils.hpp"

#include <iostream>
#include <type_traits>

namespace tilefusion::tile_layout {
/**
 * @namespace tile_layout
 *
 * @brief This namespace provides a set of utilities for defining tile layouts.
 * since Layout is quite common a name in various tensor libraries, we use
 * tile_layout to avoid potential name conflicts.
 */

enum class Layout {
    kRowMajor = 0,
    kColMajor = 1,
};

HOST_DEVICE
const char* layout_type_to_str(Layout type) {
    switch (type) {
        case Layout::kRowMajor:
            return "RowMajor";
        case Layout::kColMajor:
            return "ColMajor";
    }
    return "UnsupportedLayout";
}

namespace {
/// @brief Helper for pretty printing a matrix layout's static shape-related
///        information. This printer works ONLY on the host.
struct MatrixLayoutPrettyPrinter {
    template <typename Layout>
    static HOST void print(std::ostream& out, const Layout& layout) {
        out << layout_type_to_str(Layout::kType) << "<" << Layout::kRows << ", "
            << Layout::kCols << ">, Strides<" << Layout::kRowStride << ", "
            << Layout::kColStride << ">, Numel = " << Layout::kNumel;
    }
};
}  // namespace

template <const int kRows_, const int kCols_, const int kRowStride_,
          const int kColStride_>
struct MatrixLayout {
    static constexpr int kRows = kRows_;
    static constexpr int kCols = kCols_;

    static constexpr int kRowStride = kRowStride_;
    static constexpr int kColStride = kColStride_;

    static constexpr int kNumel = kRows * kCols;

    static constexpr Layout kType =
        kColStride == 1 ? Layout::kRowMajor : Layout::kColMajor;

    HOST_DEVICE int operator()(int i, int j) const {
        return i * kRowStride + j * kColStride;
    }
};

/// @brief Pretty printer for the static shape information of a MatrixLayout.
///        Note: This printer function works ONLY on the host.
template <const int kRows, const int kCols, const int kRowStride,
          const int kColStride>
static HOST std::ostream& operator<<(
    std::ostream& out,
    const MatrixLayout<kRows, kCols, kRowStride, kColStride>& layout) {
    MatrixLayoutPrettyPrinter::print(out, layout);
    return out;
}

// In the row major layout, the contiguous dimension in memory is the
// last dimension.
template <const int kRow, const int kCol, const int kStride = kCol>
using RowMajor = MatrixLayout<kRow, kCol, kStride, 1>;

// In the column major layout, the contiguous dimension in memory is the
// first dimension.
template <const int kRow, const int kCol, const int kStride = kRow>
using ColMajor = MatrixLayout<kRow, kCol, 1, kStride>;

template <typename Layout>
static constexpr size_t num_rows = Layout::kRows;

template <typename Layout>
static constexpr size_t num_cols = Layout::kCols;

template <typename Layout>
static constexpr size_t row_stride = Layout::kRowStride;

template <typename Layout>
static constexpr size_t col_stride = Layout::kColStride;

template <typename Layout>
static constexpr size_t get_numel = Layout::kNumel;

// In a row-major layout, the column stride is 1, whereas in a column-major
// layout, the row stride is 1. NOTE: A potential issue is that `ColMajor<1, 1>`
// will also be identified as a row-major layout.
template <typename Layout_>
static constexpr Layout layout_type = Layout_::kType;

template <typename Layout_>
struct is_row_major {
    static constexpr bool value = Layout_::kType == Layout::kRowMajor;
};

template <typename Layout_>
struct is_col_major {
    static constexpr bool value = Layout_::kType == Layout::kColMajor;
};

template <typename OuterLayout_, typename InnerLayout_>
struct BlockMatrxLayout {
    using InnerLayout = InnerLayout_;
    using OuterLayout = OuterLayout_;

    static constexpr int kRows = OuterLayout_::kRows;
    static constexpr int kCols = OuterLayout_::kCols;

    static constexpr int kInnerRows = InnerLayout_::kRows;
    static constexpr int kInnerCols = InnerLayout_::kCols;

    static_assert(kRows % kInnerRows == 0,
                  "OuterLayout rows must be divisible by InnerLayout rows");
    static_assert(kCols % kInnerCols == 0,
                  "OuterLayout cols must be divisible by InnerLayout cols");

    static constexpr int kInnerNumel = InnerLayout_::kNumel;
    static constexpr Layout kType = OuterLayout::kType;

    static constexpr int kTileRows = kRows / kInnerRows;
    static constexpr int kTileCols = kCols / kInnerCols;

    static constexpr int kRowStride = is_row_major<OuterLayout>::value
                                          ? kTileCols * kInnerNumel
                                          : kInnerNumel;
    static constexpr int kColStride = is_row_major<InnerLayout>::value
                                          ? kInnerNumel
                                          : kTileRows * kInnerNumel;

    HOST_DEVICE int operator()(int i, int j) const {
        const int outer_i = RowDivMod::div(i);
        const int inner_i = RowDivMod::mod(i);
        const int outer_j = ColDivMod::div(j);
        const int inner_j = ColDivMod::mod(j);

        return outer_(outer_i, outer_j) + inner_(inner_i, inner_j);
    }

  private:
    static constexpr bool kInnerRowsIsPow2 =
        (kInnerRows & (kInnerRows - 1)) == 0;
    static constexpr bool kInnerColsIsPow2 =
        (kInnerCols & (kInnerCols - 1)) == 0;

    using RowDivMod = DivModSelector<kInnerRowsIsPow2, kInnerRows>;
    using ColDivMod = DivModSelector<kInnerColsIsPow2, kInnerCols>;

    using BlockOuter =
        MatrixLayout<kTileRows, kTileCols, kRowStride, kColStride>;
    BlockOuter outer_;

    InnerLayout inner_;
};

template <typename OuterLayout_, typename InnerLayout_>
concept BlockRowMajorLayout =
    is_row_major<OuterLayout_>::value && is_row_major<InnerLayout_>::value;

template <typename OuterLayout_, typename InnerLayout_>
concept BlockColMajorLayout =
    is_col_major<OuterLayout_>::value && is_col_major<InnerLayout_>::value;

template <typename OuterLayout_, typename InnerLayout_>
concept BlockMixedLayout =
    (is_row_major<OuterLayout_>::value && is_col_major<InnerLayout_>::value) ||
    (is_col_major<OuterLayout_>::value && is_row_major<InnerLayout_>::value);

template <typename OuterLayout_, typename InnerLayout_>
    requires BlockRowMajorLayout<OuterLayout_, InnerLayout_>
using BlockRowMajor = BlockMatrxLayout<OuterLayout_, InnerLayout_>;

template <typename OuterLayout_, typename InnerLayout_>
    requires BlockColMajorLayout<OuterLayout_, InnerLayout_>
using BlockColMajor = BlockMatrxLayout<OuterLayout_, InnerLayout_>;

template <typename OuterLayout_, typename InnerLayout_>
    requires BlockMixedLayout<OuterLayout_, InnerLayout_>
using BlockMixed = BlockMatrxLayout<OuterLayout_, InnerLayout_>;

}  // namespace tilefusion::tile_layout
