// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "cell/compute/math_functor.hpp"
#include "cell/warp.hpp"
#include "types/layout.hpp"

namespace tilefusion::cell::compute {
namespace tl = tile_layout;

namespace detail {
template <typename RegTile, const tl::Layout kLayout>
struct Reduce;

template <typename RegTile>
struct Reduce<RegTile, tl::Layout::kRowMajor> {
  using DType = typename RegTile::DType::DType;

  static constexpr int kRows = RegTile::kRows;
  static constexpr int kCols = RegTile::kCols;

  template <typename DstTile, typename Reduce>
  DEVICE void operator()(const RegTile& src, DstTile& dst, Reduce reduce) {
    const int leader = threadIdx.x & 0x1C;
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      DType top_rows[kCols];
      DType bottom_rows[kCols];
#pragma unroll
      for (int j = 0; j < kCols; ++j) {
        auto base_tile = src(i, j);
        DType top_row_0 = reduce(base_tile(0, 0), base_tile(0, 1));
        DType top_row_1 = reduce(base_tile(1, 0), base_tile(1, 1));
        top_rows[j] = reduce(top_row_0, top_row_1);

        DType bottom_row_0 = reduce(base_tile(0, 2), base_tile(0, 3));
        DType bottom_row_1 = reduce(base_tile(1, 2), base_tile(1, 3));
        bottom_rows[j] = reduce(bottom_row_0, bottom_row_1);
      }

      DType top_row = top_rows[0];
      DType bottom_row = bottom_rows[0];

      // Compute the reduction of the top and bottom rows.
#pragma unroll
      for (int j = 1; j < kCols; ++j) {
        top_row = reduce(top_row, top_rows[j]);
        bottom_row = reduce(bottom_row, bottom_rows[j]);
      }

      // Shuffle the results to the leader thread.
      top_row = reduce(top_row, shuffle_down_sync(MASK_ALL, top_row, 2));
      top_row = reduce(top_row, shuffle_down_sync(MASK_ALL, top_row, 1));

      bottom_row =
          reduce(bottom_row, shuffle_down_sync(MASK_ALL, bottom_row, 2));
      bottom_row =
          reduce(bottom_row, shuffle_down_sync(MASK_ALL, bottom_row, 1));

      // Group the threads into groups of four, and broadcast the data
      // from the first thread in each group to the other three threads.
      top_row = shuffle_sync(MASK_ALL, top_row, leader);
      bottom_row = shuffle_sync(MASK_ALL, bottom_row, leader);

      // Store the results to the destination tile.
      dst(i, 0) = top_row;
      dst(i, 1) = bottom_row;
    }
  }
};

template <typename RegTile>
struct Reduce<RegTile, tl::Layout::kColMajor> {
  using DType = typename RegTile::DType::DType;

  static constexpr int kRows = RegTile::kRows;
  static constexpr int kCols = RegTile::kCols;

  template <typename DstTile, typename Reduce>
  DEVICE void operator()(const RegTile& tile, DstTile& dst, Reduce reduce) {
    const int leader = threadIdx.x & 0x1C;

#pragma unroll
    for (int i = 0; i < kCols; ++i) {
      DType top_cols[kRows];
      DType bottom_cols[kRows];
#pragma unroll
      for (int j = 0; j < kRows; ++j) {
        auto base_tile = tile(j, i);
        DType top_col_0 = reduce(base_tile(0, 0), base_tile(1, 0));
        DType top_col_1 = reduce(base_tile(0, 1), base_tile(1, 1));
        top_cols[j] = reduce(top_col_0, top_col_1);

        DType bottom_col_0 = reduce(base_tile(2, 0), base_tile(3, 0));
        DType bottom_col_1 = reduce(base_tile(2, 1), base_tile(3, 1));
        bottom_cols[j] = reduce(bottom_col_0, bottom_col_1);
      }

      DType top_col = top_cols[0];
      DType bottom_col = bottom_cols[0];

      // Compute the reduction of the top and bottom columns.
#pragma unroll
      for (int j = 1; j < kRows; ++j) {
        top_col = reduce(top_col, top_cols[j]);
        bottom_col = reduce(bottom_col, bottom_cols[j]);
      }

      // Shuffle the results to the leader thread.
      top_col = reduce(top_col, shuffle_down_sync(MASK_ALL, top_col, 2));
      top_col = reduce(top_col, shuffle_down_sync(MASK_ALL, top_col, 1));
      bottom_col =
          reduce(bottom_col, shuffle_down_sync(MASK_ALL, bottom_col, 2));
      bottom_col =
          reduce(bottom_col, shuffle_down_sync(MASK_ALL, bottom_col, 1));

      // Group the threads into groups of four, and broadcast the data
      // from the first thread in each group to the other three threads.
      top_col = shuffle_sync(MASK_ALL, top_col, leader);
      bottom_col = shuffle_sync(MASK_ALL, bottom_col, leader);

      // Store the results to the destination tile.
      dst(0, i) = top_col;
      dst(1, i) = bottom_col;
    }
  }
};

}  // namespace detail

template <typename RegTile, const tl::Layout kLayout>
struct SumReduce {
  using DType = typename RegTile::DType::DType;

  template <typename DstTile>
  DEVICE void operator()(const RegTile& src, DstTile& dst) {
    detail::Reduce<RegTile, kLayout> row_sum;
    row_sum(src, dst, Add<DType>{});
  }
};

template <typename RegTile, const tl::Layout kLayout>
struct MaxReduce {
  using DType = typename RegTile::DType::DType;

  template <typename DstTile>
  DEVICE void operator()(const RegTile& src, DstTile& dst) {
    detail::Reduce<RegTile, kLayout> row_max;
    row_max(src, dst, Max<DType>{});
  }
};

}  // namespace tilefusion::cell::compute
