// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cell/copy/mod.hpp"
#include "traits/base.hpp"
#include "types/mod.hpp"

namespace tilefusion::cell::copy {
using namespace atom;
namespace tl = tile_layout;

namespace {
constexpr size_t Log2(size_t n) { return ((n < 2) ? 0 : 1 + Log2(n / 2)); }
}  // namespace

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
        int row = lane_row_id();
        int col = lane_col_id() * kNumPerAccess;

        /// the pointer offset inside a warp tile.
        int src_lane_offset = src_in_base_tile_(row, col);  // global
        int dst_lane_offset = dst_in_base_tile_(row, col);  // shared

        int src_offset = 0, dst_offset = 0;
        uint32_t dst_ptr;
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                src_offset = src_base_tiles_(i, j) + src_lane_offset;
                dst_offset = dst_base_tiles_(i, j) + dst_lane_offset;

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

    using SrcBaseTilesLayout =  // global
        tl::MatrixLayout<kRowExec, kColExec,
                         BaseShape::kRows * Global::kRowStride,
                         BaseShape::kCols>;
    SrcBaseTilesLayout src_base_tiles_;

    using DstBaseTilesLayout =  // shared
        tl::MatrixLayout<kRowExec, kColExec,
                         BaseShape::kRows * Shared::kRowStride,
                         BaseShape::kNumel>;
    DstBaseTilesLayout dst_base_tiles_;

    // Given a thread index, the GlobalLayout and SharedLayout below return the
    // data offset from which the thread should load from the global memory tile
    // and where to store it in the shared memory tile, respectively.
    using GlobalLayout = tl::MatrixLayout<BaseShape::kRows, BaseShape::kCols,
                                          Global::kRowStride, 1>;
    GlobalLayout src_in_base_tile_;

    using NonSwizzled = tl::RowMajor<BaseShape::kRows, BaseShape::kCols>;

    static constexpr int kM = Log2(kNumPerAccess);
    static constexpr int kS = Log2(traits::AccessBase<DType>::kMemTransWidth /
                                   traits::AccessBase<DType>::kAccessInBits);
    using Swizzled = SwizzledLayout<NonSwizzled, 3, kM, kS>;

    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, Swizzled, NonSwizzled>;
    SharedLayout dst_in_base_tile_;

    /**
     * @brief Returns the row index of the current thread within a warp.
     */
    DEVICE int lane_row_id() {
        // NOTE: When loading a RowMajor data tile, the threads in a warp are
        // interpreted as being arranged in a row-major fashion.
        int lane_id = threadIdx.x % WARP_SIZE;
        return lane_id / BaseShape::kColThreads;
    }

    /// @brief Returns the column index of the current thread within a warp.
    DEVICE int lane_col_id() {
        // NOTE: When loading a RowMajor data tile, the threads in a warp are
        // interpreted as being arranged in a row-major fashion.
        int lane_id = threadIdx.x % WARP_SIZE;
        return lane_id % BaseShape::kColThreads;
    }
};

template <typename Global_, typename Shared_, typename BaseShape_,
          const int kRowExec_, const int kColExec_>
struct GlobalToSharedLoaderImpl<Global_, Shared_, BaseShape_, kRowExec_,
                                kColExec_, tl::Layout::kColMajor>
    : public GlobalToSharedBaseTileLoader<Global_, Shared_, BaseShape_,
                                          tl::Layout::kColMajor> {
    using Global = Global_;
    using Shared = Shared_;
    using DType = Global::DType;

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

        int src_lane_offset = src_layout_(lane_row, lane_col);
        int dst_lane_offset = dst_layout_(lane_row, lane_col);

        // In the column-major layout, rows are contiguous in memory, we
        // made the inner loop iterate over rows
        int src_offset = 0, dst_offset = 0;
#pragma unroll
        for (int i = 0; i < kColExec; ++i) {
#pragma unroll
            for (int j = 0; j < kRowExec; ++j) {
                // NOTE: DO NOT change the order of `i` and `j` in the following
                // two lines
                src_offset = src_base_tiles_(j, i) + src_lane_offset;  // global
                dst_offset = dst_base_tiles_(j, i) + dst_lane_offset;  // shared

                this->copy(src + src_offset, dst + dst_offset);
            }
        }
    }

  private:
    using BaseShape = traits::BaseTileShape<DType>;
    static constexpr int kNumPerAccess = LoadBase::kNumPerAccess;

    using SrcBaseTilesLayout =  // global
        tl::MatrixLayout<kRowExec, kColExec, BaseShape::kRows,
                         BaseShape::kCols * Global::kColStride>;
    SrcBaseTilesLayout src_base_tiles_;

    // a BaseTile is contiguously stored in shared memory
    using DstBaseTilesLayout =  // shared
        tl::MatrixLayout<kRowExec, kColExec, BaseShape::kNumel,
                         BaseShape::kCols * Shared::kColStride>;
    DstBaseTilesLayout dst_base_tiles_;

    typename LoadBase::BaseTileGlobalLayout src_layout_;
    typename LoadBase::BaseTileSharedLayout dst_layout_;
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

        /// the pointer offset inside a warp tile.
        int src_lane_offset = src_tile_(row, col);
        int dst_lane_offset = dst_tile_(row, col);

        int src_offset = 0, dst_offset = 0;
        uint32_t src_ptr;
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                src_offset = src_base_tiles_(i, j) + src_lane_offset;
                dst_offset = dst_base_tiles_(i, j) + dst_lane_offset;

                src_ptr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(src + src_offset));

                ld_shared_st_global<kAccessInBytes>(dst + dst_offset, src_ptr);
            }
        }
    }

  private:
    static constexpr int kNumPerAccess =
        traits::AccessBase<DType>::kNumPerAccess;

    static constexpr int kAccessInBytes =
        traits::AccessBase<DType>::kAccessInBytes;

    // a SharedTile is contiguously stored
    using SrcBaseTilesLayout =
        tl::MatrixLayout<kRowExec, kColExec,
                         BaseShape::kRows * Shared::kRowStride,
                         BaseShape::kNumel>;
    SrcBaseTilesLayout src_base_tiles_;

    using DstBaseTilesLayout =
        tl::MatrixLayout<kRowExec, kColExec,
                         BaseShape::kRows * Global::kRowStride,
                         BaseShape::kCols>;
    DstBaseTilesLayout dst_base_tiles_;

    using NonSwizzled = tl::RowMajor<BaseShape::kRows, BaseShape::kCols>;

    static constexpr int kM = Log2(kNumPerAccess);
    static constexpr int kS = Log2(traits::AccessBase<DType>::kMemTransWidth /
                                   traits::AccessBase<DType>::kAccessInBits);
    using Swizzled = SwizzledLayout<NonSwizzled, 3, kM, kS>;

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
};

template <typename Shared_, typename Global_, typename BaseShape_,
          const int kRowExec_, const int kColExec_>
struct SharedToGlobalStorerImpl<Shared_, Global_, BaseShape_, kRowExec_,
                                kColExec_, tl::Layout::kColMajor>
    : public SharedToGlobalBaseTileStorer<Shared_, Global_, BaseShape_,
                                          tl::Layout::kColMajor> {
    using Shared = Shared_;
    using Global = Global_;
    using DType = Shared::DType;
    using BaseShape = traits::BaseTileShape<DType>;

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

    // strides to iterate over each 16x16 `BaseTile` in the shared memory
    static constexpr int kSrcColStride = BaseShape::kCols * Shared::kColStride;

    static constexpr int kDstRowStride = BaseShape::kRows;
    static constexpr int kDstColStride = BaseShape::kCols * Global::kColStride;

    DEVICE void operator()(const DType* src, DType* dst) {
        int src_offset = 0, dst_offset = 0;

#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                src_offset = i * BaseShape::kNumel + j * kSrcColStride;
                dst_offset = i * kDstRowStride + j * kDstColStride;

                this->copy(src + src_offset, dst + dst_offset);
            }
        }
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
