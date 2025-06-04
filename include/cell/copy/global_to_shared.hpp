// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "cell/copy/warp.hpp"
#include "types/mod.hpp"

namespace tilefusion::cell::copy {
using namespace atom;
namespace tl = tile_layout;

/**
 * @brief Load a warp tile from global memory to shared memory.
 *
 * This function loads a data tile from global to shared memory.
 *
 * @param Global The type of the global memory tile.
 * @param Shared The type of the shared memory tile.
 * @param BaseShape The shape of the base tile.
 * @param kRowExec The number of rows to execute.
 * @param kColExec The number of columns to execute.
 * @param kType The type of Global and Shared memory layout.
 */
template <typename Global, typename Shared, typename BaseShape,
          const int kRowExec, const int kColExec,
          const tl::Layout kType = Shared::kType>
struct GlobalToSharedLoaderImpl;

template <typename Global_, typename Shared_, typename BaseShape_,
          const int kRowExec_, const int kColExec_>
struct GlobalToSharedLoaderImpl<Global_, Shared_, BaseShape_, kRowExec_,
                                kColExec_, tl::Layout::kRowMajor> {
  using Global = Global_;
  using Shared = Shared_;
  using DType = Global::DType;
  using BaseShape = BaseShape_;

  static_assert(Global::kRows == Shared::kRows &&
                    Global::kCols == Shared::kCols,
                "Global and shared memory should have the same shape.");
  static_assert(Global::kType == Shared::kType,
                "The layout of Global memory and Shared memory tile should "
                "be the same.");
  static_assert(Global::kType == tl::Layout::kRowMajor,
                "The layout of Global memory and Shared memory tile should "
                "be row-major.");
  static_assert(std::is_same_v<typename Shared::DType, DType>,
                "The data type of Shared and Global must be the same.");

  static constexpr int kRowExec = kRowExec_;
  static constexpr int kColExec = kColExec_;

  DEVICE void operator()(const DType* src, DType* dst, int warp_offset) {
    int row = lane_row_id();
    int col = lane_col_id() * kNumPerAccess;

    int src_offset = 0, dst_offset = 0, offset = 0;
    uint32_t dst_ptr;
#pragma unroll
    for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
      for (int j = 0; j < kColExec; ++j) {
        src_offset = src_tile_(i, j) + in_src_tile_(row, col);
        offset = warp_offset + i * BaseShape::kRows * Shared::kRowStride +
                 j * BaseShape::kCols + row * Shared::kRowStride + col;

        dst_offset = shared_tile.fetch_physical_offset(offset) - warp_offset;

        dst_ptr =
            static_cast<uint32_t>(__cvta_generic_to_shared(dst + dst_offset));
        ld_global_st_shared<kAccessInBytes>(dst_ptr, src + src_offset);
      }
    }
  }

 private:
  static constexpr int kNumPerAccess = AccessBase<DType>::kNumPerAccess;

  static constexpr int kAccessInBytes = AccessBase<DType>::kAccessInBytes;

  using SrcLayout =
      tl::MatrixLayout<kRowExec, kColExec,
                       BaseShape::kRows * Global::kRowStride, BaseShape::kCols>;
  SrcLayout src_tile_;

  // Given a thread index, the GlobalLayout and SharedLayout below return the
  // data offset from which the thread should load from the global memory tile
  // and where to store it in the shared memory tile, respectively.
  using InSrcLayout = tl::MatrixLayout<BaseShape::kRows, BaseShape::kCols,
                                       Global::kRowStride, 1>;

  // `in_src_tile_` is a basetile handled by a single warp.
  InSrcLayout in_src_tile_;

  Shared shared_tile;

  /// @brief returns the lane row of the current thread within a warp.
  DEVICE int lane_row_id() {
    // NOTE: When copying a RowMajor data tile, the thread layout is
    // interpreted as RowMajor.
    int lane_id = threadIdx.x % WARP_SIZE;
    return lane_id / BaseShape::kColThreads;
  }

  /// @brief returns the lane col of the current thread within a warp.
  DEVICE int lane_col_id() {
    // NOTE: When copying a RowMajor data tile, the thread layout is
    // interpreted as RowMajor.
    int lane_id = threadIdx.x % WARP_SIZE;
    return lane_id % BaseShape::kColThreads;
  }
};

template <typename Global_, typename Shared_, typename BaseShape_,
          const int kRowExec_, const int kColExec_>
struct GlobalToSharedLoaderImpl<Global_, Shared_, BaseShape_, kRowExec_,
                                kColExec_, tl::Layout::kColMajor> {
  using Global = Global_;
  using Shared = Shared_;
  using DType = Global::DType;
  using BaseShape = BaseShape_;

  static_assert(Global::kRows == Shared::kRows &&
                    Global::kCols == Shared::kCols,
                "Global and shared memory should have the same shape.");
  static_assert(Global::kType == Shared::kType,
                "The layout of Global memory and Shared memory tile should "
                "be the same.");
  static_assert(Global::kType == tl::Layout::kColMajor,
                "The layout of Global memory and Shared memory tile should "
                "be column-major.");

  static_assert(std::is_same_v<typename Shared::DType, DType>,
                "The data type of Shared and Global must be the same.");

  static constexpr int kRowExec = kRowExec_;
  static constexpr int kColExec = kColExec_;

  DEVICE void operator()(const DType* src, DType* dst, int warp_offset) {
    int lane_row = lane_row_id() * kNumPerAccess;
    int lane_col = lane_col_id();

    int src_offset = 0, dst_offset = 0;
    int offset = 0;
    uint32_t dst_ptr;
#pragma unroll
    for (int j = 0; j < kColExec; ++j) {
#pragma unroll
      for (int i = 0; i < kRowExec; ++i) {
        src_offset = src_tile_(i, j) + in_src_tile_(lane_row, lane_col);
        offset = warp_offset + j * BaseShape::kCols * Shared::kColStride +
                 i * BaseShape::kRows + lane_col * Shared::kColStride +
                 lane_row;
        dst_offset = shared_tile.fetch_physical_offset(offset) - warp_offset;

        dst_ptr =
            static_cast<uint32_t>(__cvta_generic_to_shared(dst + dst_offset));
        ld_global_st_shared<kAccessInBytes>(dst_ptr, src + src_offset);
      }
    }
  }

 private:
  static constexpr int kNumPerAccess = AccessBase<DType>::kNumPerAccess;

  static constexpr int kAccessInBytes = AccessBase<DType>::kAccessInBytes;

  using SrcLayout = tl::MatrixLayout<kRowExec, kColExec, BaseShape::kRows,
                                     BaseShape::kCols * Global::kColStride>;
  SrcLayout src_tile_;

  // Given a thread index, the GlobalLayout and SharedLayout below return the
  // data offset from which the thread should load from the global memory tile
  // and where to store it in the shared memory tile, respectively.
  using GlobalLayout = tl::MatrixLayout<BaseShape::kRows, BaseShape::kCols, 1,
                                        Global::kColStride>;

  // `src_tile_` is a basetile handled by a single warp.
  GlobalLayout in_src_tile_;

  Shared shared_tile;

  /// @brief returns the lane row of the current thread within a warp.
  DEVICE int lane_row_id() {
    int lane_id = threadIdx.x % WARP_SIZE;
    return lane_id / BaseShape::kColThreads;
  }

  /// @brief returns the lane col of the current thread within a warp.
  DEVICE int lane_col_id() {
    int lane_id = threadIdx.x % WARP_SIZE;
    return lane_id % BaseShape::kColThreads;
  }
};

template <typename Shared, typename Global, typename BaseShape,
          const int kRowExec, const int kColExec,

          const tl::Layout kType = Shared::kType>
struct SharedToGlobalStorerImpl;

template <typename Shared_, typename Global_, typename BaseShape_,
          const int kRowExec_, const int kColExec_>
struct SharedToGlobalStorerImpl<Shared_, Global_, BaseShape_, kRowExec_,
                                kColExec_, tl::Layout::kRowMajor> {
  using Shared = Shared_;
  using Global = Global_;
  using DType = Shared::DType;
  using BaseShape = BaseShape_;
  static_assert(Global::kRows == Shared::kRows &&
                    Global::kCols == Shared::kCols,
                "Global and shared memory should have the same shape.");
  static_assert(Global::kType == Shared::kType,
                "The layout of Global memory and Shared memory tile should "
                "be the same.");
  static_assert(Global::kType == tl::Layout::kRowMajor,
                "The layout of Global memory and Shared memory tile should "
                "be row-major.");
  static_assert(std::is_same_v<typename Global::DType, DType>,
                "The data type of Shared and Global must be the same.");

  static constexpr int kRowExec = kRowExec_;
  static constexpr int kColExec = kColExec_;

  DEVICE void operator()(const DType* src, DType* dst, int warp_offset) {
    int row = lane_row_id();
    int col = lane_col_id() * kNumPerAccess;

    uint32_t src_ptr;
    int src_offset = 0, dst_offset = 0;
    int offset = 0;
#pragma unroll
    for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
      for (int j = 0; j < kColExec; ++j) {
        offset = warp_offset + i * BaseShape::kRows * Shared::kRowStride +
                 j * BaseShape::kCols + row * Shared::kRowStride + col;
        src_offset = shared_tile.fetch_physical_offset(offset) - warp_offset;
        dst_offset = dst_tile_(i, j) + in_dst_tile_(row, col);

        src_ptr =
            static_cast<uint32_t>(__cvta_generic_to_shared(src + src_offset));
        ld_shared_st_global<kAccessInBytes>(dst + dst_offset, src_ptr);
      }
    }
  }

 private:
  static constexpr int kAccessInBytes = AccessBase<DType>::kAccessInBytes;

  using DstLayout =
      tl::MatrixLayout<kRowExec, kColExec,
                       BaseShape::kRows * Global::kRowStride, BaseShape::kCols>;
  DstLayout dst_tile_;

  // NOTE: DO NOT modify `kNumPerAccess` and `kAccessInBits` here.
  // `kAccessInBits` in the storer is for tensor core's output where only two
  // numbers are contiguous in memory. This ensures the parameters remain
  // consistent with those used in `SharedLayoutWrapper` within the
  // register-to-shared storer.
  static constexpr int kAccessInBits = 2 * int(sizeof(DType) * 8);
  static constexpr int kNumPerAccess = AccessBase<DType>::kNumPerAccess;

  using GlobalLayout = tl::MatrixLayout<BaseShape::kRows, BaseShape::kCols,
                                        Global::kRowStride, 1>;
  GlobalLayout in_dst_tile_;

  Shared shared_tile;

  /// @brief returns the lane col of the current thread within a warp.
  DEVICE int lane_row_id() {
    return (threadIdx.x % WARP_SIZE) / BaseShape::kColThreads;
  }

  /// @brief returns the lane col of the current thread within a warp.
  DEVICE int lane_col_id() {
    return (threadIdx.x % WARP_SIZE) % BaseShape::kColThreads;
  }
};

template <typename Shared_, typename Global_, typename BaseShape_,
          const int kRowExec_, const int kColExec_>
struct SharedToGlobalStorerImpl<Shared_, Global_, BaseShape_, kRowExec_,
                                kColExec_, tl::Layout::kColMajor> {
  using Shared = Shared_;
  using Global = Global_;
  using DType = Shared::DType;
  using BaseShape = BaseShape_;

  static_assert(Global::kRows == Shared::kRows &&
                    Global::kCols == Shared::kCols,
                "Global and shared memory should have the same shape.");
  static_assert(Global::kType == Shared::kType,
                "The layout of Global memory and Shared memory tile should "
                "be the same.");
  static_assert(Global::kType == tl::Layout::kColMajor,
                "The layout of Global memory and Shared memory tile should "
                "be column-major.");
  static_assert(std::is_same_v<typename Global::DType, DType>,
                "The data type of Shared and Global must be the same.");

  static constexpr int kRowExec = kRowExec_;
  static constexpr int kColExec = kColExec_;

  DEVICE void operator()(const DType* src, DType* dst, int warp_offset) {
    int lane_row = lane_row_id() * kNumPerAccess;
    int lane_col = lane_col_id();

    int src_offset = 0, dst_offset = 0;
    int offset = 0;
    uint32_t src_ptr;
#pragma unroll
    for (int j = 0; j < kColExec; ++j) {
#pragma unroll
      for (int i = 0; i < kRowExec; ++i) {
        offset = warp_offset + j * BaseShape::kCols * Shared::kColStride +
                 i * BaseShape::kRows + lane_col * Shared::kColStride +
                 lane_row;
        src_offset = shared_tile.fetch_physical_offset(offset) - warp_offset;
        dst_offset = dst_tile_(i, j) + in_dst_tile_(lane_row, lane_col);

        src_ptr =
            static_cast<uint32_t>(__cvta_generic_to_shared(src + src_offset));
        ld_shared_st_global<kAccessInBytes>(dst + dst_offset, src_ptr);
      }
    }
  }

 private:
  static constexpr int kAccessInBytes = AccessBase<DType>::kAccessInBytes;

  using DstLayout = tl::MatrixLayout<kRowExec, kColExec, BaseShape::kRows,
                                     BaseShape::kCols * Global::kColStride>;
  DstLayout dst_tile_;

  // NOTE: DO NOT modify `kNumPerAccess` and `kAccessInBits` here.
  // `kAccessInBits` in the storer is for tensor core's output where only two
  // numbers are contiguous in memory. This ensures the parameters remain
  // consistent with those used in `SharedLayoutWrapper` within the
  // register-to-shared storer.
  static constexpr int kAccessInBits = 2 * int(sizeof(DType) * 8);
  static constexpr int kNumPerAccess = AccessBase<DType>::kNumPerAccess;

  using GlobalLayout = tl::MatrixLayout<BaseShape::kRows, BaseShape::kCols, 1,
                                        Global::kColStride>;
  GlobalLayout in_dst_tile_;

  Shared shared_tile;

  /// @brief returns the lane col of the current thread within a warp.
  DEVICE int lane_row_id() {
    return (threadIdx.x % WARP_SIZE) / BaseShape::kColThreads;
  }

  /// @brief returns the lane col of the current thread within a warp.
  DEVICE int lane_col_id() {
    return (threadIdx.x % WARP_SIZE) % BaseShape::kColThreads;
  }
};

/// @brief The thread-block level API that cooperatively transfers a data tile
///        from global memory to shared memory by all the threads within a
///        thread block.
template <typename Shared_, typename WarpLayout_>
struct GlobalToSharedLoader {
  using Shared = Shared_;
  using DType = Shared::DType;
  using WarpLayout = WarpLayout_;

  // NOTE: The WarpShape calculated here is for the warp reuse mode `kCont`.
  // If you use a different mode, update the WarpShape accordingly.
  static_assert((Shared::kRows % WarpLayout ::kRows == 0) &&
                    (Shared::kCols % WarpLayout::kCols == 0),
                "The shape of SharedTile must be divisible by the shape of "
                "WarpLayout.");

  using WarpShape = TileShape<Shared::kRows / WarpLayout::kRows,
                              Shared::kCols / WarpLayout::kCols>;
  using BaseShape = WarpBaseTileShape<DType, WarpShape, Shared::kType>;

  static_assert(Shared::kRows % BaseShape ::kRows == 0,
                "Shared::kRows must be divisible by BaseShape::kRows.");
  static_assert(Shared::kCols % BaseShape::kCols == 0,
                "Shared::kCols must be divisible by BaseShape::kCols.");

  static const WarpReuse kMode = WarpReuse::kCont;  // warp reuse mode
  using ExecCounter = warp::ExecCounter<BaseShape, Shared, WarpLayout, kMode>;
  using GlobalOffset = warp::GlobalOffsetHelper<WarpLayout, kMode>;
  using SharedOffset =
      warp::SharedOffsetHelper<WarpLayout, BaseShape, Shared, kMode>;

  static constexpr int kRowExec = ExecCounter::kRowExec;
  static constexpr int kColExec = ExecCounter::kColExec;

  static_assert(kRowExec && kColExec,
                "Ensure that the execution count for all rows and columns is "
                "greater than 0.");

  static constexpr int kSharedAccessInBytes = Shared::SwizzleBytes;
  static constexpr int kSharedContInBytes =
      Shared::kType == tl::Layout::kRowMajor
          ? Shared::kCols * sizeof(DType) / WarpLayout::kCols
          : Shared::kRows * sizeof(DType) / WarpLayout::kRows;

  static_assert(kSharedAccessInBytes <= kSharedContInBytes,
                "kSharedAccessInBytes must be less than or equal to "
                "kSharedContInBytes");
  static_assert(kSharedAccessInBytes % 32 == 0,
                "The number of bytes in a warp tile must be divisible by "
                "32.");

  template <typename Global>
  DEVICE void operator()(const Global& src, Shared& dst) {
    static_assert(
        Global::kRows == Shared::kRows && Global::kCols == Shared::kCols,
        "Global and shared memory should have the same shape.");

    const DType* src_ptr = src.data();
    DType* dst_ptr = dst.mutable_data();

    // get warp offset for global and shared memory
    int offset_src = global_offset_.template get_warp_offset<Global>();
    int offset_dst = shared_offset_.get_warp_offset();

    // Load a single warp tile from global memory to shared memory
    using Loader =
        GlobalToSharedLoaderImpl<Global, Shared, BaseShape, kRowExec, kColExec>;
    Loader loader;
    loader(src_ptr + offset_src, dst_ptr + offset_dst, offset_dst);
  }

 private:
  GlobalOffset global_offset_;
  SharedOffset shared_offset_;
};

template <typename Shared_, typename WarpLayout_>
struct SharedToGlobalStorer {
  using Shared = Shared_;
  using DType = Shared::DType;
  using WarpLayout = WarpLayout_;

  using WarpShape = TileShape<Shared::kRows / WarpLayout::kRows,
                              Shared::kCols / WarpLayout::kCols>;
  using BaseShape = WarpBaseTileShape<DType, WarpShape, Shared::kType>;

  static_assert(Shared::kRows % BaseShape::kRows == 0,
                "Shared::kRows must be divisible by BaseShape::kRows.");
  static_assert(Shared::kCols % BaseShape::kCols == 0,
                "Shared::kCols must be divisible by BaseShape::kCols.");

  static const WarpReuse kMode = WarpReuse::kCont;  // warp reuse mode

  using GlobalOffset = warp::GlobalOffsetHelper<WarpLayout, kMode>;
  using SharedOffset =
      warp::SharedOffsetHelper<WarpLayout, BaseShape, Shared, kMode>;

  using ExecCounter = warp::ExecCounter<BaseShape, Shared, WarpLayout, kMode>;

  static constexpr int kRowExec = ExecCounter::kRowExec;
  static constexpr int kColExec = ExecCounter::kColExec;

  static_assert(kRowExec && kColExec,
                "Execution count should be greater than 0.");

  static constexpr int kSharedAccessInBytes = Shared::SwizzleBytes;
  static constexpr int kSharedContInBytes =
      Shared::kType == tl::Layout::kRowMajor
          ? Shared::kCols * sizeof(DType) / WarpLayout::kCols
          : Shared::kRows * sizeof(DType) / WarpLayout::kRows;

  static_assert(kSharedAccessInBytes <= kSharedContInBytes,
                "kSharedAccessInBytes must be less than or equal to "
                "kSharedContInBytes");
  static_assert(kSharedAccessInBytes % 32 == 0,
                "The number of bytes in a warp tile must be divisible by "
                "32.");

  template <typename Global>
  DEVICE void operator()(const Shared& src_, Global& dst_) {
    const DType* src = src_.data();
    DType* dst = dst_.mutable_data();

    // The offset for data that the current warp should access
    int offset_src = shared_offset_.get_warp_offset();
    int offset_dst = global_offset_.template get_warp_offset<Global>();

    using Storer =
        SharedToGlobalStorerImpl<Shared, Global, BaseShape, kRowExec, kColExec>;
    Storer storer;
    storer(src + offset_src, dst + offset_dst, offset_src);
  }

 private:
  SharedOffset shared_offset_;
  GlobalOffset global_offset_;
};

}  // namespace tilefusion::cell::copy
