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
 * This function loads a warp tile whose shape is specified by `WarpShape`
 * from global memory to shared memory.
 *
 * @tparam Global_   The type of the global memory pointer.
 * @tparam Shared_   The type of the shared memory pointer.
 * @tparam BaseShape_ The shape of the warp tile.
 * @tparam kRowExec_ The number of rows to execute.
 * @tparam kColExec_ The number of columns to execute.
 * @tparam kType     The type of the elements to be loaded.
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
        int src_lane_offset = src_layout_(row, col);
        int dst_lane_offset = dst_layout_(row, col);

        int src_offset = 0, dst_offset = 0;
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                src_offset = src_base_tiles_(i, j) + src_lane_offset;
                dst_offset = dst_base_tiles_(i, j) + dst_lane_offset;

                copy(src + src_offset, dst + dst_offset);
            }
        }
    }

  private:
    static constexpr int kNumPerAccess =
        traits::AccessBase<DType>::kNumPerAccess;

    using SrcBaseTilesLayout =
        tl::MatrixLayout<kRowExec, kColExec,
                         BaseShape::kRows * Global::kRowStride,
                         BaseShape::kCols>;
    SrcBaseTilesLayout src_base_tiles_;

    // a BaseTile is contiguously stored in shared memory
    using DstBaseTilesLayout =
        tl::MatrixLayout<kRowExec, kColExec,
                         BaseShape::kRows * Shared::kRowStride,
                         BaseShape::kNumel>;
    DstBaseTilesLayout dst_base_tiles_;

    // Given a thread index, the GlobalLayout and SharedLayout below return the
    // data offset from which the thread should load from the global memory tile
    // and where to store it in the shared memory tile, respectively.
    using GlobalLayout = tl::MatrixLayout<BaseShape::kRows, BaseShape::kCols,
                                          Global::kRowStride, 1>;

    // `src_layout_` is a basetile handled by a single warp.
    GlobalLayout src_layout_;

    using NonSwizzled = tl::RowMajor<BaseShape::kRows, BaseShape::kCols>;
    using Swizzled =
        tl::SwizzledRowMajor<traits::AccessBase<DType>::kAccessInBits,
                             BaseShape>;
    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, Swizzled, NonSwizzled>;
    SharedLayout dst_layout_;

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
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                src_offset = src_base_tiles_(i, j) + src_lane_offset;
                dst_offset = dst_base_tiles_(i, j) + dst_lane_offset;

                copy(src + src_offset, dst + dst_offset);
            }
        }
    }

  private:
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

    // NOTE: DO NOT modify `kNumPerAccess` and `kAccessInBits` here.
    // `kAccessInBits` in the storer is for tensor core's output where only two
    // numbers are contiguous in memory. This ensures the parameters remain
    // consistent with those used in `SharedLayoutWrapper` within the
    // register-to-shared storer.
    static constexpr int kAccessInBits = 2 * int(sizeof(DType) * 8);
    static constexpr int kNumPerAccess =
        traits::AccessBase<DType>::kNumPerAccess;

    using NonSwizzled = tl::RowMajor<BaseShape::kRows, BaseShape::kCols>;
    using Swizzled = tl::SwizzledRowMajor<kAccessInBits, BaseShape>;
    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, Swizzled, NonSwizzled>;
    SharedLayout src_tile_;

    using GlobalLayout =
        tl::MatrixLayout<BaseShape::kRows, BaseShape::kCols, Global::kRowStride,
                         Global::kColStride>;
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

    // This implementation uses a fixed 16x16 `BaseShape` as the atomic data
    // tile accessed by threads in a single warp that issues a single load/store
    // instruction.
    // FIXME(ying): uncomment the following lines to automatically infer the
    // warp-level tile shape instead of using a fixed 16x16 `BaseShape`. using
    // WarpShape =
    //     warp::WarpTileShape<DType, typename Shared::Layout, Shared::kType>;
    using WarpShape =
        warp::WarpTileShape<DType, tl::RowMajor<16, 16>, Shared::kType>;

    static_assert(Shared::kRows % WarpShape::kRows == 0,
                  "Shared::kRows must be divisible by WarpShape::kRows.");
    static_assert(Shared::kCols % WarpShape::kCols == 0,
                  "Shared::kCols must be divisible by WarpShape::kCols.");

    static const WarpReuse kMode = WarpReuse::kCont;  // warp reuse mode
    using ExecCounter = warp::ExecCounter<WarpShape, Shared, WarpLayout, kMode>;
    using GlobalOffset = warp::GlobalOffsetHelper<WarpLayout, kMode>;
    using SharedOffset =
        warp::SharedOffsetHelper<WarpLayout, WarpShape, Shared, kMode>;

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

        int offset_src = global_offset_.template get_warp_offset<Global>();
        int offset_dst = shared_offset_.get_warp_offset();

        // Load a single warp tile from global memory to shared memory
        using Loader = GlobalToSharedLoaderImpl<Global, Shared, WarpShape,
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

    // FIXME(ying): automatically infer the warp-level tile shape instead
    // of using a fixed `BaseShape`.
    // using WarpShape =
    //     warp::WarpTileShape<DType, typename Shared::Layout, Shared::kType>;

    // FIXME(ying): uncomment the following lines to automatically infer the
    // warp-level tile shape instead of using a fixed 16x16 `BaseShape`.
    using BaseShape =
        warp::WarpTileShape<DType, tl::RowMajor<16, 16>, Shared::kType>;

    static_assert(Shared::kRows % BaseShape::kRows == 0,
                  "Shared::kRows must be divisible by BaseShape::kRows.");
    static_assert(Shared::kCols % BaseShape::kCols == 0,
                  "Shared::kCols must be divisible by BaseShape::kCols.");

    static const WarpReuse kMode = WarpReuse::kCont;  // warp reuse mode

    using SharedOffset =
        warp::SharedOffsetHelper<WarpLayout, BaseShape, Shared, kMode>;
    using GlobalOffset = warp::GlobalOffsetHelper<WarpLayout, kMode>;

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
