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
 * @tparam Global The type of the global memory tile.
 * @tparam Shared The type of the shared memory tile.
 * @tparam BaseShape The shape of the base tile.
 * @tparam kRowExec The number of rows to execute.
 * @tparam kColExec The number of columns to execute.
 * @tparam kType The type of Global and Shared memory layout.
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

    DEVICE void operator()(const DType* src, DType* dst) {
        // TODO(KuangjuX): When the `WarpRow` is greater than 1, a swizzle block
        // might be split by two warps, and a solution is needed to address this
        // situation.
        int row = lane_row_id();
        int col = lane_col_id() * kNumPerAccess;

        int src_offset = 0, dst_offset = 0;
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                int tile_i = (i * BaseShape::kRows + row) / kSwizzleRows;
                int tile_j = (j * BaseShape::kCols + col) / kSwizzleCols;
                int tile_row = (i * BaseShape::kRows + row) % kSwizzleRows;
                int tile_col = (j * BaseShape::kCols + col) % kSwizzleCols;

                /// the pointer offset inside a warp tile.
                int src_lane_offset = in_src_tile_(row, col);
                int dst_lane_offset = in_dst_tile_(tile_row, tile_col);

                src_offset = src_tile_(i, j) + src_lane_offset;
                dst_offset = dst_tile_(tile_i, tile_j) + dst_lane_offset;

                copy(src + src_offset, dst + dst_offset);
            }
        }
    }

  private:
    static constexpr int kNumPerAccess =
        traits::AccessBase<DType>::kNumPerAccess;

    using SwizzleBaseShape = traits::SwizzleBaseTileShape<DType>;
    static constexpr int kSwizzleRows = SwizzleBaseShape::kRows;
    static constexpr int kSwizzleCols = SwizzleBaseShape::kCols;

    static constexpr int kSwizzleRowExec =
        kRowExec / (kSwizzleRows / BaseShape::kRows);
    static constexpr int kSwizzleColExec =
        kColExec / (kSwizzleCols / BaseShape::kCols);

    using SrcBaseTilesLayout =
        tl::MatrixLayout<kRowExec, kColExec,
                         BaseShape::kRows * Global::kRowStride,
                         BaseShape::kCols>;
    SrcBaseTilesLayout src_tile_;

    using DstSwizzledLayout =
        tl::MatrixLayout<kSwizzleRowExec, kSwizzleColExec,
                         kSwizzleRows * Shared::kRowStride, kSwizzleCols>;
    DstSwizzledLayout dst_tile_;

    // Given a thread index, the GlobalLayout and SharedLayout below return the
    // data offset from which the thread should load from the global memory tile
    // and where to store it in the shared memory tile, respectively.
    using GlobalLayout = tl::MatrixLayout<BaseShape::kRows, BaseShape::kCols,
                                          Global::kRowStride, 1>;

    // `in_src_tile_` is a basetile handled by a single warp.
    GlobalLayout in_src_tile_;

    using NonSwizzled =
        tl::MatrixLayout<kSwizzleRows, kSwizzleCols, Shared::kRowStride, 1>;
    using Swizzled =
        SwizzledLayout<NonSwizzled, SwizzleBaseShape::B, SwizzleBaseShape::M,
                       SwizzleBaseShape::S, tl::Layout::kRowMajor>;

    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, Swizzled, NonSwizzled>;
    SharedLayout in_dst_tile_;

    DEVICE void copy(const DType* src, DType* dst) {
        // a single memory access access 16 bytes
        ld_global_st_shared<16>(
            static_cast<uint32_t>(__cvta_generic_to_shared(dst)), src);
    }

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

    using LoadBase = GlobalToSharedBaseTileLoader<Global, Shared, BaseShape_,
                                                  tl::Layout::kColMajor>;

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
        int lane_row = this->lane_row_id() * kNumPerAccess;
        int lane_col = this->lane_col_id();

#pragma unroll
        for (int j = 0; j < kColExec; ++j) {
#pragma unroll
            for (int i = 0; i < kRowExec; ++i) {
                int tile_i = (i * BaseShape::kRows + lane_row) / kSwizzleRows;
                int tile_j = (j * BaseShape::kCols + lane_col) / kSwizzleCols;
                int tile_row = (i * BaseShape::kRows + lane_row) % kSwizzleRows;
                int tile_col = (j * BaseShape::kCols + lane_col) % kSwizzleCols;

                /// the pointer offset inside a warp tile.
                int src_lane_offset = in_src_tile_(lane_row, lane_col);
                int dst_lane_offset = in_dst_tile_(tile_row, tile_col);

                int src_offset = src_tile_(i, j) + src_lane_offset;
                int dst_offset = dst_tile_(tile_i, tile_j) + dst_lane_offset;

                copy(src + src_offset, dst + dst_offset);
            }
        }
    }

  private:
    static constexpr int kNumPerAccess = LoadBase::kNumPerAccess;

    // Swap the row and column of the `SwizzleBaseShape` in column-major layout.
    using SwizzleBaseShape = traits::SwizzleBaseTileShape<DType>;
    static constexpr int kSwizzleRows = SwizzleBaseShape::kCols;
    static constexpr int kSwizzleCols = SwizzleBaseShape::kRows;

    static constexpr int kSwizzleRowExec =
        kRowExec / (kSwizzleRows / BaseShape::kRows);
    static constexpr int kSwizzleColExec =
        kColExec / (kSwizzleCols / BaseShape::kCols);

    using SrcBaseTilesLayout =
        tl::MatrixLayout<kRowExec, kColExec, BaseShape::kRows,
                         BaseShape::kCols * Global::kColStride>;
    SrcBaseTilesLayout src_tile_;

    using DstSwizzledLayout =
        tl::MatrixLayout<kSwizzleRowExec, kSwizzleColExec, kSwizzleRows,
                         kSwizzleCols * Shared::kColStride>;
    DstSwizzledLayout dst_tile_;

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

    DEVICE void copy(const DType* src, DType* dst) {
        // a single memory access access 16 bytes
        ld_global_st_shared<16>(
            static_cast<uint32_t>(__cvta_generic_to_shared(dst)), src);
    }

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

template <typename Shared_, typename Global_, typename BaseShape,
          const int kRowExec_, const int kColExec_>
struct SharedToGlobalStorerImpl<Shared_, Global_, BaseShape, kRowExec_,
                                kColExec_, tl::Layout::kRowMajor> {
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

#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                int tile_i =
                    (i * BaseShape::kRows + row) / SwizzledBaseShape::kRows;
                int tile_j =
                    (j * BaseShape::kCols + col) / SwizzledBaseShape::kCols;
                int tile_row =
                    (i * BaseShape::kRows + row) % SwizzledBaseShape::kRows;
                int tile_col =
                    (j * BaseShape::kCols + col) % SwizzledBaseShape::kCols;

                int src_tile_offset = src_tile_(tile_row, tile_col);
                int dst_lane_offset = dst_tile_(row, col);

                int src_offset =
                    src_base_tiles_(tile_i, tile_j) + src_tile_offset;
                int dst_offset = dst_base_tiles_(i, j) + dst_lane_offset;

                copy(src + src_offset, dst + dst_offset);
            }
        }
    }

  private:
    using SwizzledBaseShape = traits::SwizzleBaseTileShape<DType>;
    static constexpr int kSwizzledRows = SwizzledBaseShape::kRows;
    static constexpr int kSwizzledCols = SwizzledBaseShape::kCols;
    static constexpr int B = SwizzledBaseShape::B;
    static constexpr int M = SwizzledBaseShape::M;
    static constexpr int S = SwizzledBaseShape::S;

    static constexpr int kSwizzledRowExec =
        kRowExec / (kSwizzledRows / BaseShape::kRows);
    static constexpr int kSwizzledColExec =
        kColExec / (kSwizzledCols / BaseShape::kCols);

    using SrcSwizzledLayout =
        tl::MatrixLayout<kSwizzledRowExec, kSwizzledColExec,
                         kSwizzledRows * Shared::kRowStride, kSwizzledCols>;
    SrcSwizzledLayout src_base_tiles_;

    using DstBaseTilesLayout =
        tl::MatrixLayout<kRowExec, kColExec,
                         BaseShape::kRows * Global::kRowStride,
                         BaseShape::kCols>;
    DstBaseTilesLayout dst_base_tiles_;

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
    SharedLayout src_tile_;

    using GlobalLayout = tl::MatrixLayout<BaseShape::kRows, BaseShape::kCols,
                                          Global::kRowStride, 1>;
    GlobalLayout dst_tile_;

    /// @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_row_id() {
        return (threadIdx.x % WARP_SIZE) / BaseShape::kColThreads;
    }

    /// @brief returns the lane col of the current thread within a warp.
    DEVICE int lane_col_id() {
        return (threadIdx.x % WARP_SIZE) % BaseShape::kColThreads;
    }

    DEVICE void copy(const DType* src, DType* dst) {
        ld_shared_st_global<16>(
            dst, static_cast<uint32_t>(__cvta_generic_to_shared(src)));
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

    DEVICE void operator()(const DType* src, DType* dst) {
        int lane_row = this->lane_row_id() * kNumPerAccess;
        int lane_col = this->lane_col_id();

#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                int tile_i = (i * BaseShape::kRows + lane_row) / kSwizzleRows;
                int tile_j = (j * BaseShape::kCols + lane_col) / kSwizzleCols;
                int tile_row = (i * BaseShape::kRows + lane_row) % kSwizzleRows;
                int tile_col = (j * BaseShape::kCols + lane_col) % kSwizzleCols;

                int src_tile_offset = in_src_tile_(tile_row, tile_col);
                int dst_lane_offset = in_dst_tile_(lane_row, lane_col);

                int src_offset = src_tile_(tile_i, tile_j) + src_tile_offset;
                int dst_offset = dst_tile_(i, j) + dst_lane_offset;

                copy(src + src_offset, dst + dst_offset);
            }
        }
    }

  private:
    // Swap the row and column of the `SwizzleBaseShape` in column-major layout.
    using SwizzleBaseShape = traits::SwizzleBaseTileShape<DType>;
    static constexpr int kSwizzleRows = SwizzleBaseShape::kCols;
    static constexpr int kSwizzleCols = SwizzleBaseShape::kRows;

    static constexpr int kSwizzleRowExec =
        kRowExec / (kSwizzleRows / BaseShape::kRows);
    static constexpr int kSwizzleColExec =
        kColExec / (kSwizzleCols / BaseShape::kCols);

    using SrcSwizzledLayout =
        tl::MatrixLayout<kSwizzleRowExec, kSwizzleColExec, kSwizzleRows,
                         kSwizzleCols * Shared::kColStride>;
    SrcSwizzledLayout src_tile_;

    using DstBaseTilesLayout =
        tl::MatrixLayout<kRowExec, kColExec, BaseShape::kRows,
                         BaseShape::kCols * Global::kColStride>;
    DstBaseTilesLayout dst_tile_;

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

    DEVICE void copy(const DType* src, DType* dst) {
        ld_shared_st_global<16>(
            dst, static_cast<uint32_t>(__cvta_generic_to_shared(src)));
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
    using BaseShape = warp::WarpBaseTileShape<DType, WarpShape, Shared::kType>;

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
        using Loader = GlobalToSharedLoaderImpl<Global, Shared, BaseShape,
                                                kRowExec, kColExec>;

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
    using BaseShape = warp::WarpBaseTileShape<DType, WarpShape, Shared::kType>;

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

    template <typename Global>
    DEVICE void operator()(const Shared& src_, Global& dst_) {
        const DType* src = src_.data();
        DType* dst = dst_.mutable_data();

        // The offset for data that the current warp should access
        int offset_src = shared_offset_.get_warp_offset();
        int offset_dst = global_offset_.template get_warp_offset<Global>();

        using Storer = SharedToGlobalStorerImpl<Shared, Global, BaseShape,
                                                kRowExec, kColExec>;
        Storer storer;
        storer(src + offset_src, dst + offset_dst);
    }

  private:
    SharedOffset shared_offset_;
    GlobalOffset global_offset_;
};

}  // namespace tilefusion::cell::copy
