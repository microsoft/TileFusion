// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "cell/copy/mod.hpp"
#include "traits/base.hpp"
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
          const int kSharedAccessInBytes,
          const tl::Layout kType = Shared::kType>
struct GlobalToSharedLoaderImpl;

template <typename Global_, typename Shared_, typename BaseShape_,
          const int kRowExec_, const int kColExec_,
          const int kSharedAccessInBytes>
struct GlobalToSharedLoaderImpl<Global_, Shared_, BaseShape_, kRowExec_,
                                kColExec_, kSharedAccessInBytes,
                                tl::Layout::kRowMajor> {
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

    DEVICE void operator()(const DType* src, DType* dst) {
        int row = lane_row_id();
        int col = lane_col_id() * kNumPerAccess;

        int src_offset = 0, dst_offset = 0;
        int offset = 0;
        uint32_t dst_ptr;
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                src_offset = src_tile_(i, j) + in_src_tile_(row, col);
                offset = i * BaseShape::kRows * Shared::kRowStride +
                         j * BaseShape::kCols + row * Shared::kRowStride + col;

                dst_offset = get_swizzle_offset(offset);

                dst_ptr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(dst + dst_offset));
                ld_global_st_shared<kAccessInBytes>(dst_ptr, src + src_offset);
            }
        }
    }

  private:
    static constexpr int kNumPerAccess =
        traits::AccessBase<DType>::kNumPerAccess;

    static constexpr int kAccessInBytes =
        traits::AccessBase<DType>::kAccessInBytes;

    using SwizzledBaseShape =
        traits::SwizzleBaseTileShape<DType, kSharedAccessInBytes>;
    static constexpr int kSwizzledRows = SwizzledBaseShape::kRows;
    static constexpr int kSwizzledCols = SwizzledBaseShape::kCols;

    static constexpr int kSwizzledBlockRows =
        kRowExec * BaseShape::kRows / kSwizzledRows;
    static constexpr int kSwizzledBlockCols =
        kColExec * BaseShape::kCols / kSwizzledCols;

    using SrcLayout = tl::MatrixLayout<kRowExec, kColExec,
                                       BaseShape::kRows * Global::kRowStride,
                                       BaseShape::kCols>;
    SrcLayout src_tile_;

    using DstLayout =
        tl::MatrixLayout<kSwizzledBlockRows, kSwizzledBlockCols,
                         Shared::kRowStride * kSwizzledRows, kSwizzledCols>;
    DstLayout dst_tile_;

    // Given a thread index, the GlobalLayout and SharedLayout below return the
    // data offset from which the thread should load from the global memory tile
    // and where to store it in the shared memory tile, respectively.
    using InSrcLayout = tl::MatrixLayout<BaseShape::kRows, BaseShape::kCols,
                                         Global::kRowStride, 1>;

    // `in_src_tile_` is a basetile handled by a single warp.
    InSrcLayout in_src_tile_;

    using NonSwizzled =
        tl::MatrixLayout<kSwizzledRows, kSwizzledCols, Shared::kRowStride, 1>;
    using Swizzled =
        SwizzledLayout<NonSwizzled, SwizzledBaseShape::B, SwizzledBaseShape::M,
                       SwizzledBaseShape::S, tl::Layout::kRowMajor>;

    using InDstLayout =
        std::conditional_t<Shared::kSwizzled, Swizzled, NonSwizzled>;
    InDstLayout in_dst_tile_;

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

    DEVICE int2 get_swizzle_tile_id(int offset) {
        int swizzle_tile_col = (offset % Shared::kRowStride) / kSwizzledCols;
        int swizzle_tile_row = (offset / Shared::kRowStride) / kSwizzledRows;
        return make_int2(swizzle_tile_row, swizzle_tile_col);
    }

    DEVICE int2 get_in_swizzle_tile_id(int offset) {
        // Get id in the swizzle tile.
        auto swizzled_tile_id = get_swizzle_tile_id(offset);

        int row = offset / Shared::kRowStride;
        int col = offset % Shared::kRowStride;

        int in_swizzle_tile_row = row % kSwizzledRows;
        int in_swizzle_tile_col = col % kSwizzledCols;

        return make_int2(in_swizzle_tile_row, in_swizzle_tile_col);
    }

    DEVICE int get_swizzle_offset(int offset) {
        auto swizzled_tile_id = get_swizzle_tile_id(offset);
        auto in_swizzled_tile_id = get_in_swizzle_tile_id(offset);
        int swizzled_tile_offset =
            dst_tile_(swizzled_tile_id.x, swizzled_tile_id.y);
        int in_swizzled_tile_offset =
            in_dst_tile_(in_swizzled_tile_id.x, in_swizzled_tile_id.y);

        int offset_ = swizzled_tile_offset + in_swizzled_tile_offset;

        return offset_;
    }
};

template <typename Global_, typename Shared_, typename BaseShape_,
          const int kRowExec_, const int kColExec_,
          const int kSharedAccessInBytes>
struct GlobalToSharedLoaderImpl<Global_, Shared_, BaseShape_, kRowExec_,
                                kColExec_, kSharedAccessInBytes,
                                tl::Layout::kColMajor> {
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

    DEVICE void operator()(const DType* src, DType* dst) {
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
                offset = j * BaseShape::kCols * Shared::kColStride +
                         i * BaseShape::kRows + lane_col * Shared::kColStride +
                         lane_row;
                dst_offset = get_swizzle_offset(offset);

                dst_ptr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(dst + dst_offset));
                ld_global_st_shared<kAccessInBytes>(dst_ptr, src + src_offset);
            }
        }
    }

  private:
    static constexpr int kNumPerAccess =
        traits::AccessBase<DType>::kNumPerAccess;

    static constexpr int kAccessInBytes =
        traits::AccessBase<DType>::kAccessInBytes;

    // Swap the row and column of the `SwizzleBaseShape` in column-major layout.
    using SwizzleBaseShape =
        traits::SwizzleBaseTileShape<DType, kSharedAccessInBytes>;
    static constexpr int kSwizzleRows = SwizzleBaseShape::kCols;
    static constexpr int kSwizzleCols = SwizzleBaseShape::kRows;

    static constexpr int kSwizzleBlockRows =
        kRowExec * BaseShape::kRows / kSwizzleRows;
    static constexpr int kSwizzleBlockCols =
        kColExec * BaseShape::kCols / kSwizzleCols;

    using SrcLayout = tl::MatrixLayout<kRowExec, kColExec, BaseShape::kRows,
                                       BaseShape::kCols * Global::kColStride>;
    SrcLayout src_tile_;

    using DstLayout =
        tl::MatrixLayout<kSwizzleBlockRows, kSwizzleBlockCols, kSwizzleRows,
                         kSwizzleCols * Shared::kColStride>;
    DstLayout dst_tile_;

    // Given a thread index, the GlobalLayout and SharedLayout below return the
    // data offset from which the thread should load from the global memory tile
    // and where to store it in the shared memory tile, respectively.
    using GlobalLayout = tl::MatrixLayout<BaseShape::kRows, BaseShape::kCols, 1,
                                          Global::kColStride>;

    // `src_tile_` is a basetile handled by a single warp.
    GlobalLayout in_src_tile_;

    using NonSwizzled =
        tl::MatrixLayout<kSwizzleRows, kSwizzleCols, 1, Shared::kColStride>;
    using Swizzled =
        SwizzledLayout<NonSwizzled, SwizzleBaseShape::B, SwizzleBaseShape::M,
                       SwizzleBaseShape::S, tl::Layout::kColMajor>;

    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, Swizzled, NonSwizzled>;
    SharedLayout in_dst_tile_;

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

    DEVICE int2 get_swizzled_tile_id(int offset) {
        int swizzle_tile_row = (offset % Shared::kColStride) / kSwizzleRows;
        int swizzle_tile_col = (offset / Shared::kColStride) / kSwizzleCols;
        return make_int2(swizzle_tile_row, swizzle_tile_col);
    }

    DEVICE int2 get_in_swizzle_tile_id(int offset) {
        auto swizzled_tile_id = get_swizzled_tile_id(offset);

        int row = offset % Shared::kColStride;
        int col = offset / Shared::kColStride;

        int in_swizzle_tile_row = row % kSwizzleRows;
        int in_swizzle_tile_col = col % kSwizzleCols;

        return make_int2(in_swizzle_tile_row, in_swizzle_tile_col);
    }

    DEVICE int get_swizzle_offset(int offset) {
        auto swizzled_tile_id = get_swizzled_tile_id(offset);
        auto in_swizzled_tile_id = get_in_swizzle_tile_id(offset);
        int swizzled_tile_offset =
            dst_tile_(swizzled_tile_id.x, swizzled_tile_id.y);
        int in_swizzled_tile_offset =
            in_dst_tile_(in_swizzled_tile_id.x, in_swizzled_tile_id.y);

        int offset_ = swizzled_tile_offset + in_swizzled_tile_offset;

        return offset_;
    }
};

template <typename Shared, typename Global, typename BaseShape,
          const int kRowExec, const int kColExec,
          const int kSharedAccessInBytes,
          const tl::Layout kType = Shared::kType>
struct SharedToGlobalStorerImpl;

template <typename Shared_, typename Global_, typename BaseShape,
          const int kRowExec_, const int kColExec_,
          const int kSharedAccessInBytes>
struct SharedToGlobalStorerImpl<Shared_, Global_, BaseShape, kRowExec_,
                                kColExec_, kSharedAccessInBytes,
                                tl::Layout::kRowMajor> {
    using Shared = Shared_;
    using Global = Global_;
    using DType = Shared::DType;

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

    DEVICE void operator()(const DType* src, DType* dst) {
        int row = lane_row_id();
        int col = lane_col_id() * kNumPerAccess;

        uint32_t src_ptr;
        int src_offset = 0, dst_offset = 0;
        int offset = 0;
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                offset = i * BaseShape::kRows * Global::kRowStride +
                         j * BaseShape::kCols + row * Global::kRowStride + col;
                src_offset = get_swizzle_offset(offset);
                dst_offset = dst_tile_(i, j) + in_dst_tile_(row, col);

                src_ptr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(src + src_offset));
                ld_shared_st_global<kAccessInBytes>(dst + dst_offset, src_ptr);
            }
        }
    }

  private:
    static constexpr int kAccessInBytes =
        traits::AccessBase<DType>::kAccessInBytes;

    using SwizzledBaseShape =
        traits::SwizzleBaseTileShape<DType, kSharedAccessInBytes>;
    static constexpr int kSwizzledRows = SwizzledBaseShape::kRows;
    static constexpr int kSwizzledCols = SwizzledBaseShape::kCols;
    static constexpr int B = SwizzledBaseShape::B;
    static constexpr int M = SwizzledBaseShape::M;
    static constexpr int S = SwizzledBaseShape::S;

    static constexpr int kSwizzledBlockRows =
        kRowExec * BaseShape::kRows / kSwizzledRows;
    static constexpr int kSwizzledBlockCols =
        kColExec * BaseShape::kCols / kSwizzledCols;

    using SrcLayout =
        tl::MatrixLayout<kSwizzledBlockRows, kSwizzledBlockCols,
                         kSwizzledRows * Shared::kRowStride, kSwizzledCols>;
    SrcLayout src_tile_;

    using DstLayout = tl::MatrixLayout<kRowExec, kColExec,
                                       BaseShape::kRows * Global::kRowStride,
                                       BaseShape::kCols>;
    DstLayout dst_tile_;

    // NOTE: DO NOT modify `kNumPerAccess` and `kAccessInBits` here.
    // `kAccessInBits` in the storer is for tensor core's output where only two
    // numbers are contiguous in memory. This ensures the parameters remain
    // consistent with those used in `SharedLayoutWrapper` within the
    // register-to-shared storer.
    static constexpr int kAccessInBits = 2 * int(sizeof(DType) * 8);
    static constexpr int kNumPerAccess =
        traits::AccessBase<DType>::kNumPerAccess;

    using NonSwizzled =
        tl::MatrixLayout<kSwizzledRows, kSwizzledCols, Shared::kRowStride, 1>;
    using Swizzled =
        SwizzledLayout<NonSwizzled, B, M, S, tl::Layout::kRowMajor>;
    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, Swizzled, NonSwizzled>;
    SharedLayout in_src_tile_;

    using GlobalLayout = tl::MatrixLayout<BaseShape::kRows, BaseShape::kCols,
                                          Global::kRowStride, 1>;
    GlobalLayout in_dst_tile_;

    /// @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_row_id() {
        return (threadIdx.x % WARP_SIZE) / BaseShape::kColThreads;
    }

    /// @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_col_id() {
        return (threadIdx.x % WARP_SIZE) % BaseShape::kColThreads;
    }

    DEVICE int2 get_swizzle_tile_id(int offset) {
        int swizzle_tile_col = (offset % Shared::kRowStride) / kSwizzledCols;
        int swizzle_tile_row = (offset / Shared::kRowStride) / kSwizzledRows;
        return make_int2(swizzle_tile_row, swizzle_tile_col);
    }

    DEVICE int2 get_in_swizzle_tile_id(int offset) {
        auto swizzled_tile_id = get_swizzle_tile_id(offset);

        int row = offset / Shared::kRowStride;
        int col = offset % Shared::kRowStride;

        int in_swizzle_tile_row = row % kSwizzledRows;
        int in_swizzle_tile_col = col % kSwizzledCols;

        return make_int2(in_swizzle_tile_row, in_swizzle_tile_col);
    }

    DEVICE int get_swizzle_offset(int offset) {
        auto swizzled_tile_id = get_swizzle_tile_id(offset);
        auto in_swizzled_tile_id = get_in_swizzle_tile_id(offset);
        int swizzled_tile_offset =
            src_tile_(swizzled_tile_id.x, swizzled_tile_id.y);
        int in_swizzled_tile_offset =
            in_src_tile_(in_swizzled_tile_id.x, in_swizzled_tile_id.y);

        int offset_ = swizzled_tile_offset + in_swizzled_tile_offset;

        return offset_;
    }
};

template <typename Shared_, typename Global_, typename BaseShape_,
          const int kRowExec_, const int kColExec_,
          const int kSharedAccessInBytes>
struct SharedToGlobalStorerImpl<Shared_, Global_, BaseShape_, kRowExec_,
                                kColExec_, kSharedAccessInBytes,
                                tl::Layout::kColMajor> {
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

    DEVICE void operator()(const DType* src, DType* dst) {
        int lane_row = lane_row_id() * kNumPerAccess;
        int lane_col = lane_col_id();

        int src_offset = 0, dst_offset = 0;
        int offset = 0;
        uint32_t src_ptr;
#pragma unroll
        for (int j = 0; j < kColExec; ++j) {
#pragma unroll
            for (int i = 0; i < kRowExec; ++i) {
                offset = j * BaseShape::kCols * Shared::kColStride +
                         i * BaseShape::kRows + lane_col * Shared::kColStride +
                         lane_row;
                src_offset = get_swizzle_offset(offset);
                dst_offset = dst_tile_(i, j) + in_dst_tile_(lane_row, lane_col);

                src_ptr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(src + src_offset));
                ld_shared_st_global<kAccessInBytes>(dst + dst_offset, src_ptr);
            }
        }
    }

  private:
    static constexpr int kAccessInBytes =
        traits::AccessBase<DType>::kAccessInBytes;

    // Swap the row and column of the `SwizzleBaseShape` in column-major layout.
    using SwizzleBaseShape =
        traits::SwizzleBaseTileShape<DType, kSharedAccessInBytes>;
    static constexpr int kSwizzleRows = SwizzleBaseShape::kCols;
    static constexpr int kSwizzleCols = SwizzleBaseShape::kRows;

    static constexpr int kSwizzleBlockRows =
        kRowExec * BaseShape::kRows / kSwizzleRows;
    static constexpr int kSwizzleBlockCols =
        kColExec * BaseShape::kCols / kSwizzleCols;

    using SrcLayout =
        tl::MatrixLayout<kSwizzleBlockRows, kSwizzleBlockCols, kSwizzleRows,
                         kSwizzleCols * Shared::kColStride>;
    SrcLayout src_tile_;

    using DstLayout = tl::MatrixLayout<kRowExec, kColExec, BaseShape::kRows,
                                       BaseShape::kCols * Global::kColStride>;
    DstLayout dst_tile_;

    // NOTE: DO NOT modify `kNumPerAccess` and `kAccessInBits` here.
    // `kAccessInBits` in the storer is for tensor core's output where only two
    // numbers are contiguous in memory. This ensures the parameters remain
    // consistent with those used in `SharedLayoutWrapper` within the
    // register-to-shared storer.
    static constexpr int kAccessInBits = 2 * int(sizeof(DType) * 8);
    static constexpr int kNumPerAccess =
        traits::AccessBase<DType>::kNumPerAccess;

    using NonSwizzled =
        tl::MatrixLayout<kSwizzleRows, kSwizzleCols, 1, Shared::kColStride>;
    using Swizzled =
        SwizzledLayout<NonSwizzled, SwizzleBaseShape::B, SwizzleBaseShape::M,
                       SwizzleBaseShape::S, tl::Layout::kColMajor>;
    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, Swizzled, NonSwizzled>;
    SharedLayout in_src_tile_;

    using GlobalLayout = tl::MatrixLayout<BaseShape::kRows, BaseShape::kCols, 1,
                                          Global::kColStride>;
    GlobalLayout in_dst_tile_;

    /// @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_row_id() {
        return (threadIdx.x % WARP_SIZE) / BaseShape::kColThreads;
    }

    /// @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_col_id() {
        return (threadIdx.x % WARP_SIZE) % BaseShape::kColThreads;
    }

    DEVICE int2 get_swizzled_tile_id(int offset) {
        int swizzle_tile_row = (offset % Shared::kColStride) / kSwizzleRows;
        int swizzle_tile_col = (offset / Shared::kColStride) / kSwizzleCols;
        return make_int2(swizzle_tile_row, swizzle_tile_col);
    }

    DEVICE int2 get_in_swizzle_tile_id(int offset) {
        auto swizzled_tile_id = get_swizzled_tile_id(offset);

        int row = offset % Shared::kColStride;
        int col = offset / Shared::kColStride;

        int in_swizzle_tile_row = row % kSwizzleRows;
        int in_swizzle_tile_col = col % kSwizzleCols;

        return make_int2(in_swizzle_tile_row, in_swizzle_tile_col);
    }

    DEVICE int get_swizzle_offset(int offset) {
        auto swizzled_tile_id = get_swizzled_tile_id(offset);
        auto in_swizzled_tile_id = get_in_swizzle_tile_id(offset);
        int swizzled_tile_offset =
            src_tile_(swizzled_tile_id.x, swizzled_tile_id.y);
        int in_swizzled_tile_offset =
            in_src_tile_(in_swizzled_tile_id.x, in_swizzled_tile_id.y);

        int offset_ = swizzled_tile_offset + in_swizzled_tile_offset;

        return offset_;
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

    static constexpr int kSharedContInBytes =
        Shared::kType == tl::Layout::kRowMajor
            ? Shared::kCols * sizeof(DType) / WarpLayout::kCols
            : Shared::kRows * sizeof(DType) / WarpLayout::kRows;

    static_assert(kSharedContInBytes % 32 == 0,
                  "The number of bytes in a warp tile must be divisible by "
                  "32.");

    static constexpr int kSharedAccessInBytes =
        kSharedContInBytes >= 128 ? 128 : kSharedContInBytes;

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
            GlobalToSharedLoaderImpl<Global, Shared, BaseShape, kRowExec,
                                     kColExec, kSharedAccessInBytes>;

        Loader loader;
        loader(src_ptr + offset_src, dst_ptr + offset_dst);
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

    static constexpr int kSharedContInBytes =
        Shared::kType == tl::Layout::kRowMajor
            ? Shared::kCols * sizeof(DType) / WarpLayout::kCols
            : Shared::kRows * sizeof(DType) / WarpLayout::kRows;

    static_assert(kSharedContInBytes % 32 == 0,
                  "The number of bytes in a warp tile must be divisible by "
                  "32.");

    static constexpr int kSharedAccessInBytes =
        kSharedContInBytes >= 128 ? 128 : kSharedContInBytes;

    template <typename Global>
    DEVICE void operator()(const Shared& src_, Global& dst_) {
        const DType* src = src_.data();
        DType* dst = dst_.mutable_data();

        // The offset for data that the current warp should access
        int offset_src = shared_offset_.get_warp_offset();
        int offset_dst = global_offset_.template get_warp_offset<Global>();

        using Storer =
            SharedToGlobalStorerImpl<Shared, Global, BaseShape, kRowExec,
                                     kColExec, kSharedAccessInBytes>;
        Storer storer;
        storer(src + offset_src, dst + offset_dst);
    }

  private:
    SharedOffset shared_offset_;
    GlobalOffset global_offset_;
};

}  // namespace tilefusion::cell::copy
