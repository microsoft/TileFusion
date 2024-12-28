// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cell/copy/mod.hpp"
#include "traits/base.hpp"
#include "types/mod.hpp"

namespace tilefusion::cell::copy {
using namespace atom;
namespace tl = tile_layout;

template <typename Global, typename Shared, const int kRowExec,
          const int kColExec, const tl::Layout kType>
struct GlobalToSharedLoaderImpl;

template <typename Global_, typename Shared_, const int kRowExec_,
          const int kColExec_>
struct GlobalToSharedLoaderImpl<Global_, Shared_, kRowExec_, kColExec_,
                                tl::Layout::kRowMajor>
    : public GlobalToSharedBaseTileLoader<Global_, Shared_,
                                          tl::Layout::kRowMajor> {
    using Global = Global_;
    using Shared = Shared_;
    using DType = Global::DType;
    using LoadBase =
        GlobalToSharedBaseTileLoader<Global, Shared, tl::Layout::kRowMajor>;
    using BaseShape = traits::BaseTileShape<DType>;

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
        int lane_row = this->lane_row_id();
        int lane_col = this->lane_col_id() * kNumPerAccess;

        int src_lane_offset = src_layout_(lane_row, lane_col);
        int dst_lane_offset = dst_layout_(lane_row, lane_col);

        int src_offset = 0, dst_offset = 0;
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                src_offset = src_base_tiles_(i, j) + src_lane_offset;
                dst_offset = dst_base_tiles_(i, j) + dst_lane_offset;

                this->copy(src + src_offset, dst + dst_offset);
            }
        }
    }

  private:
    static constexpr int kNumPerAccess = LoadBase::kNumPerAccess;

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

    typename LoadBase::BaseTileGlobalLayout src_layout_;
    // the layout for a single BaseTile
    typename LoadBase::BaseTileSharedLayout dst_layout_;
};

template <typename Global_, typename Shared_, const int kRowExec_,
          const int kColExec_>
struct GlobalToSharedLoaderImpl<Global_, Shared_, kRowExec_, kColExec_,
                                tl::Layout::kColMajor>
    : public GlobalToSharedBaseTileLoader<Global_, Shared_,
                                          tl::Layout::kColMajor> {
    using Global = Global_;
    using Shared = Shared_;
    using DType = Global::DType;

    using LoadBase =
        GlobalToSharedBaseTileLoader<Global, Shared, tl::Layout::kColMajor>;

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

template <typename Shared, typename Global, const int kRowExec_,
          const int kColExec_, const tl::Layout kType>
struct SharedToGlobalStorerImpl;

template <typename Shared_, typename Global_, const int kRowExec_,
          const int kColExec_>
struct SharedToGlobalStorerImpl<Shared_, Global_, kRowExec_, kColExec_,
                                tl::Layout::kRowMajor>
    : public SharedToGlobalBaseTileStorer<Shared_, Global_,
                                          tl::Layout::kRowMajor> {
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
    static_assert(Global::kType == tl::Layout::kRowMajor,
                  "The layout of Global memory and Shared memory tile should "
                  "be row-major.");
    static_assert(std::is_same_v<typename Global::DType, DType>,
                  "The data type of Shared and Global must be the same.");

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    static constexpr int kSrcRowStride = BaseShape::kRows * Shared::kRowStride;

    // strides to iterate over each 16x16 `BaseTile` in the shared memory
    static constexpr int kDstRowStride = BaseShape::kRows * Global::kRowStride;
    static constexpr int kDstColStride = BaseShape::kCols;

    DEVICE void operator()(const DType* src, DType* dst) {
        int src_offset = 0, dst_offset = 0;
        for (int i = 0; i < kRowExec; ++i) {
            for (int j = 0; j < kColExec; ++j) {
                src_offset = i * kSrcRowStride + j * BaseShape::kNumel;
                dst_offset = i * kDstRowStride + j * kDstColStride;

                this->copy(src + src_offset, dst + dst_offset);
            }
        }
    }
};

template <typename Shared_, typename Global_, const int kRowExec_,
          const int kColExec_>
struct SharedToGlobalStorerImpl<Shared_, Global_, kRowExec_, kColExec_,
                                tl::Layout::kColMajor>
    : public SharedToGlobalBaseTileStorer<Shared_, Global_,
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
        for (int i = 0; i < kRowExec; ++i) {
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

    // FIXME(ying): automatically infer the warp-level tile shape instead
    // of using a fixed `BaseShape`.
    // using WarpShape =
    //     warp::WarpTileShape<DType, typename Shared::Layout, Shared::kType>;

    using WarpShape = traits::BaseTileShape<DType>;
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
                  "Ensure that the execution count for all "
                  "rows and columns is greater than 0.");

    template <typename Global>
    DEVICE void operator()(const Global& src, Shared& dst) {
        static_assert(
            Global::kRows == Shared::kRows && Global::kCols == Shared::kCols,
            "Global and shared memory should have the same shape.");

        const DType* src_ptr = src.data();
        DType* dst_ptr = dst.mutable_data();

        int offset_src = global_offset_.template get_warp_offset<Global>();
        int offset_dst = shared_offset_.get_warp_offset();

        using Loader = GlobalToSharedLoaderImpl<Global, Shared, kRowExec,
                                                kColExec, Shared::kType>;

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
    using WarpShape = traits::BaseTileShape<DType>;
    static_assert(Shared::kRows % WarpShape::kRows == 0,
                  "Shared::kRows must be divisible by WarpShape::kRows.");
    static_assert(Shared::kCols % WarpShape::kCols == 0,
                  "Shared::kCols must be divisible by WarpShape::kCols.");

    static const WarpReuse kMode = WarpReuse::kCont;  // warp reuse mode
    using SharedOffset =
        warp::SharedOffsetHelper<WarpLayout, WarpShape, Shared, kMode>;
    using GlobalOffset = warp::GlobalOffsetHelper<WarpLayout, kMode>;
    using ExecCounter = warp::ExecCounter<WarpShape, Shared, WarpLayout, kMode>;

    static constexpr int kRowExec = ExecCounter::kRowExec;
    static constexpr int kColExec = ExecCounter::kColExec;

    static_assert(kRowExec && kColExec,
                  "Execution count should be greater than 0.");

    template <typename Global>
    DEVICE void operator()(const Shared& src_, Global& dst_) {
        const DType* src = src_.data();
        DType* dst = dst_.mutable_data();

        int offset_src = shared_offset_.get_warp_offset();
        int offset_dst = global_offset_.template get_warp_offset<Global>();

        using Storer = SharedToGlobalStorerImpl<Shared, Global, kRowExec,
                                                kColExec, Shared::kType>;

        Storer storer;
        storer(src + offset_src, dst + offset_dst);
    }

  private:
    SharedOffset shared_offset_;
    GlobalOffset global_offset_;
};
}  // namespace tilefusion::cell::copy
