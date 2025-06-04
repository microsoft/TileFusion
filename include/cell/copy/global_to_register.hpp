// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cell/copy/constants.hpp"
#include "cell/copy/vectorize.hpp"
#include "cell/copy/warp.hpp"
#include "types/mod.hpp"

namespace tilefusion::cell::copy {

namespace {
struct BaseTileConfig {
  static constexpr int kRows = 16;
  static constexpr int kCols = 16;
  static constexpr int kThreadRows = 8;
  static constexpr int kThreadCols = 4;
};
}  // namespace

/**
 * @brief Load a RegTile from Global memory to Register.
 * @param Global Global memory tile type.
 * @param Reg Register tile type.
 * @param kType Global Layout type.
 */
template <typename Global, typename Reg, const tl::Layout kType>
struct GlobalToRegLoaderImpl;

template <typename Global_, typename Reg_>
struct GlobalToRegLoaderImpl<Global_, Reg_, tl::Layout::kRowMajor> {
  using Global = Global_;
  using Reg = Reg_;
  using DType = typename Global::DType;

  DEVICE void operator()(const DType* src, Reg& dst) {
    int lane_id = threadIdx.x % WARP_SIZE;
    const DType* data;
    int land_row = lane_id / kThreadCol;
    int land_col = lane_id % kThreadCol * 2;

    Vectorize<DType, 2> copy;
#pragma unroll
    for (int i = 0; i < kRowExec; ++i) {
      int row = i * kCols + land_row;
#pragma unroll
      for (int j = 0; j < kColExec; ++j) {
        int col = j * kRows + land_col;
        data = src + row * kStride + col;

        copy(data, &dst(i, j)(0, 0));
        copy(data + 8, &dst(i, j)(1, 0));
        copy(data + 8 * kStride, &dst(i, j)(0, 2));
        copy(data + 8 * kStride + 8, &dst(i, j)(1, 2));
      }
    }
  }

 private:
  // pre-computed values
  static constexpr int kThreadCol = BaseTileConfig::kThreadCols;
  static constexpr int kRows = BaseTileConfig::kRows;
  static constexpr int kCols = BaseTileConfig::kCols;
  static constexpr int kStride = Global::kRowStride;

  // how many times a `BaseTile` is executed along the row and column
  // direction.
  static constexpr int kRowExec = Reg::kRows;
  static constexpr int kColExec = Reg::kCols;
};

template <typename Global_, typename Reg_>
struct GlobalToRegLoaderImpl<Global_, Reg_, tl::Layout::kColMajor> {
  using Global = Global_;
  using Reg = Reg_;
  using DType = typename Global::DType;

  DEVICE void operator()(const DType* src, Reg& dst) {
    int lane_id = threadIdx.x % WARP_SIZE;

    int land_row = lane_id / kThreadCol;
    int land_col = lane_id % kThreadCol * 2;

    const DType* data;
    Vectorize<DType, 2> copy;

#pragma unroll
    for (int i = 0; i < kColExec; ++i) {
      int col = i * BaseTileConfig::kRows + land_row;
      for (int j = 0; j < kRowExec; ++j) {
        int row = j * BaseTileConfig::kCols + land_col;
        data = src + col * kStride + row;

        copy(data, &dst(j, i)(0, 0));
        copy(data + 8, &dst(j, i)(0, 1));
        copy(data + 8 * kStride, &dst(j, i)(2, 0));
        copy(data + 8 * kStride + 8, &dst(j, i)(2, 1));
      }
    }
  }

 private:
  // pre-computed values
  static constexpr int kThreadCol = BaseTileConfig::kThreadCols;
  static constexpr int kRows = BaseTileConfig::kRows;
  static constexpr int kCols = BaseTileConfig::kCols;
  static constexpr int kStride = Global::kColStride;

  // how many times a `BaseTile` is executed along the row and column
  // direction.
  static constexpr int kRowExec = Reg::kRows;
  static constexpr int kColExec = Reg::kCols;
};

/**
 * @brief Store a RegTile to Global memory.
 * @param Global Global memory tile type.
 * @param Reg Register tile type.
 * @param kType Global Layout type.
 */
template <typename Global, typename Reg, const tl::Layout kType>
struct RegToGlobalStorerImpl;

template <typename Global_, typename Reg_>
struct RegToGlobalStorerImpl<Global_, Reg_, tl::Layout::kRowMajor> {
  using Global = Global_;
  using Reg = Reg_;
  using DType = typename Global::DType;

  DEVICE void operator()(const Reg& src, DType* dst) {
    int lane_id = threadIdx.x % WARP_SIZE;
    DType* data;

    int land_row = lane_id / kThreadCol;
    int land_col = lane_id % kThreadCol * 2;

    Vectorize<DType, 2> copy;
#pragma unroll
    for (int i = 0; i < kRowExec; ++i) {
      int row = i * kCols + land_row;
#pragma unroll
      for (int j = 0; j < kColExec; ++j) {
        int col = j * kRows + land_col;
        data = dst + row * kStride + col;

        copy(&src(i, j)(0, 0), data);
        copy(&src(i, j)(1, 0), data + 8);
        copy(&src(i, j)(0, 2), data + 8 * kStride);
        copy(&src(i, j)(1, 2), data + 8 * kStride + 8);
      }
    }
  }

 private:
  // pre-computed values
  static constexpr int kThreadCol = BaseTileConfig::kThreadCols;
  static constexpr int kRows = BaseTileConfig::kRows;
  static constexpr int kCols = BaseTileConfig::kCols;
  static constexpr int kStride = Global::kRowStride;

  // how many times a `BaseTile` is executed along the row and column
  // direction.
  static constexpr int kRowExec = Reg::kRows;
  static constexpr int kColExec = Reg::kCols;
};

template <typename Global_, typename Reg_>
struct RegToGlobalStorerImpl<Global_, Reg_, tl::Layout::kColMajor> {
  using Global = Global_;
  using Reg = Reg_;
  using DType = typename Global::DType;

  DEVICE void operator()(const Reg& src, DType* dst) {
    int lane_id = threadIdx.x % WARP_SIZE;
    DType* data;

    int land_row = lane_id / kThreadCol;
    int land_col = lane_id % kThreadCol * 2;

    Vectorize<DType, 2> copy;
#pragma unroll
    for (int i = 0; i < kColExec; ++i) {
      int col = i * kRows + land_row;
#pragma unroll
      for (int j = 0; j < kRowExec; ++j) {
        int row = j * kCols + land_col;
        data = dst + col * kStride + row;

        copy(&src(j, i)(0, 0), data);
        copy(&src(j, i)(0, 1), data + 8);
        copy(&src(j, i)(2, 0), data + 8 * kStride);
        copy(&src(j, i)(2, 1), data + 8 * kStride + 8);
      }
    }
  }

 private:
  // pre-computed values
  static constexpr int kThreadCol = BaseTileConfig::kThreadCols;
  static constexpr int kRows = BaseTileConfig::kRows;
  static constexpr int kCols = BaseTileConfig::kCols;
  static constexpr int kStride = Global::kColStride;
  // how many times a `BaseTile` is executed along the row and column
  // direction.
  static constexpr int kRowExec = Reg::kRows;
  static constexpr int kColExec = Reg::kCols;
};

/**
 * @brief Load a data tile from Global memory to Register based on the warp
 *        reuse mode.
 * @param Reg_ Register tile type.
 * @param WarpLayout_ Warp layout type.
 * @param kMode_ Warp reuse mode.
 */
template <typename Reg_, typename WarpLayout_, const WarpReuse kMode_>
struct GlobalToRegLoader {
  using Reg = Reg_;
  using DType = typename Reg::DType::DType;
  using WarpLayout = WarpLayout_;
  static constexpr WarpReuse kMode = kMode_;

  template <typename Global>
  DEVICE void operator()(const Global& src, Reg& dst) {
    // advance the pointer to input data to the current warp
    // according to warp reuse mode.
    int offset = global_offset_.template get_warp_offset<Global>();

    using Loader = GlobalToRegLoaderImpl<Global, Reg, Global::kType>;
    Loader loader;
    loader(src.data() + offset, dst);
  }

 private:
  using GlobalOffset = warp::GlobalOffsetHelper<WarpLayout, WarpReuse::kCont>;

  GlobalOffset global_offset_;
};

/**
 * @brief Store a data tile from Register to Global memory based on the warp
 *        reuse mode.
 * @param Global_ Global memory tile type.
 * @param Reg_ Register tile type.
 * @param WarpLayout_ Warp layout type.
 */
template <typename Global_, typename Reg_, typename WarpLayout_>
struct RegToGlobalStorer {
  using Global = Global_;
  using Reg = Reg_;
  using DType = typename Global::DType;
  using WarpLayout = WarpLayout_;

  DEVICE void operator()(const Reg& src, Global& dst) {
    DType* dst_ptr = dst.mutable_data();

    // advance the pointer to output data to the current warp
    // according to warp reuse mode.
    int offset = global_offset_.template get_warp_offset<Global>();

    using Storer = RegToGlobalStorerImpl<Global, Reg, Global::kType>;
    Storer storer;
    storer(src, dst_ptr + offset);
  }

  using GlobalOffset = warp::GlobalOffsetHelper<WarpLayout, WarpReuse::kCont>;

  GlobalOffset global_offset_;
};
}  // namespace tilefusion::cell::copy
