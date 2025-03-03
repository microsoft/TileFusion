// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "cell/copy/mod.hpp"
#include "traits/base.hpp"
#include "types/mod.hpp"

namespace tilefusion::cell::copy {
using namespace atom;
namespace tl = tile_layout;

namespace detail {

template <typename Shared, typename Reg_, const int kRowExec,
          const int kColExec, const int kSharedAccessInBytes,
          const tl::Layout kType>
struct SharedToRegLoaderImpl;

/// @brief partial specialization for row-major shared memory tile.
template <typename Shared, typename Reg_, const int kRowExec_,
          const int kColExec_, const int kSharedAccessInBytes>
struct SharedToRegLoaderImpl<Shared, Reg_, kRowExec_, kColExec_,
                             kSharedAccessInBytes, tl::Layout::kRowMajor>
    : public LoadMatBase<typename Shared::DType> {
    using LoadMat = LoadMatBase<typename Shared::DType>;
    using DType = Shared::DType;
    using Reg = Reg_;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const DType* src, Reg& dst, int warp_offset,
                           int iterator_offset) {
        int global_offset = warp_offset + iterator_offset;
        int lane_row = this->lane_row_id();
        int lane_col = this->lane_col_id() * LoadMat::kNumPerAccess;

#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                int tile_offset = global_offset +
                                  i * kSharedRowStride * BaseShape::kRows +
                                  j * BaseShape::kCols +
                                  lane_row * kSharedRowStride + lane_col;
                int offset = get_swizzle_offset(tile_offset) - iterator_offset;

                // advance pointer to the 16x16 `BaseTile` indexed by(i, j).
                // issue the hardware-backed memory access instruction.
                this->ldmatrix(src + offset, dst(i, j).mutable_data());
            }
        }
    }

  private:
    using BaseShape = traits::BaseTileShape<DType>;

    using SwizzledBaseShape =
        traits::SwizzleBaseTileShape<DType, kSharedAccessInBytes>;
    static constexpr int kSwizzledRows = SwizzledBaseShape::kRows;
    static constexpr int kSwizzledCols = SwizzledBaseShape::kCols;

    static constexpr int kSharedRowStride = Shared::kRowStride;

    static constexpr int kSwizzledBlockRows =
        kRowExec * BaseShape::kRows / kSwizzledRows;
    static constexpr int kSwizzledBlockCols =
        kColExec * BaseShape::kCols / kSwizzledCols;

    using SrcLayout =
        tl::MatrixLayout<kSwizzledBlockRows, kSwizzledBlockCols,
                         Shared::kRowStride * kSwizzledRows, kSwizzledCols>;
    SrcLayout src_tile_;

    using NonSwizzled =
        tl::MatrixLayout<kSwizzledRows, kSwizzledCols, Shared::kRowStride, 1>;
    using Swizzled =
        SwizzledLayout<NonSwizzled, SwizzledBaseShape::B, SwizzledBaseShape::M,
                       SwizzledBaseShape::S, tl::Layout::kRowMajor>;

    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, Swizzled, NonSwizzled>;
    SharedLayout in_src_tile_;

    DEVICE int2 get_swizzled_tile_id(int offset) {
        // SwizzleTile is a 8 x 64 block.
        int swizzle_tile_col = (offset % kSharedRowStride) / kSwizzledCols;
        int swizzle_tile_row = (offset / kSharedRowStride) / kSwizzledRows;
        return make_int2(swizzle_tile_row, swizzle_tile_col);
    }

    DEVICE int2 get_in_swizzle_tile_id(int offset) {
        // Get id in the swizzle tile.
        auto swizzled_tile_id = get_swizzled_tile_id(offset);

        int row = offset / kSharedRowStride;
        int col = offset % kSharedRowStride;

        int in_swizzle_tile_col = col % kSwizzledCols;
        int in_swizzle_tile_row = row % kSwizzledRows;

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

/// @brief partial specialization for column-major shared memory tile.
template <typename Shared, typename Reg_, const int kRowExec_,
          const int kColExec_, const int kSharedAccessInBytes>
struct SharedToRegLoaderImpl<Shared, Reg_, kRowExec_, kColExec_,
                             kSharedAccessInBytes, tl::Layout::kColMajor>
    : public LoadMatBase<typename Shared::DType> {
    using Reg = Reg_;
    using DType = Shared::DType;
    using LoadMat = LoadMatBase<DType>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE SharedToRegLoaderImpl()
        : base_tiles_(BaseTilesLayout{})
        , in_base_tile_(BaseTileSharedLayout{}) {}

    DEVICE void operator()(const DType* src, Reg& dst, int warp_offset,
                           int iterator_offset) {
        int global_offset = warp_offset + iterator_offset;
        // transpose the lane position if the shared memory is in
        // column-major. 16 threads are mapped to the strided dimension
        // of the data while the 2 threads are mapped to the contiguous
        // dimension of the data.
        int lane_row = this->lane_col_id() * LoadMat::kNumPerAccess;
        int lane_col = this->lane_row_id();

        for (int i = 0; i < kColExec; ++i) {
#pragma unroll
            for (int j = 0; j < kRowExec; ++j) {
                int tile_offset = global_offset +
                                  i * kSharedColStride * BaseShape::kCols +
                                  j * BaseShape::kRows +
                                  lane_col * kSharedColStride + lane_row;
                int offset = get_swizzle_offset(tile_offset) - iterator_offset;

                // issue the hardware-backed memory access instruction
                this->ldmatrix(src + offset, dst(j, i).mutable_data());
            }
        }
    }

  private:
    using BaseShape = traits::BaseTileShape<DType>;

    static constexpr int kSharedColStride = Shared::kColStride;

    using BaseTilesLayout =
        tl::MatrixLayout<kRowExec, kColExec, Shared::kRowStride,
                         Shared::kColStride>;
    BaseTilesLayout base_tiles_;

    using BaseTileSharedLayout =
        tl::SharedLayoutWrapper<Shared, LoadMat::kAccessInBits>::Layout;
    BaseTileSharedLayout in_base_tile_;

    // Use 64x8 as a basic swizzle block shape.
    using SwizzleBaseShape =
        traits::SwizzleBaseTileShape<DType, kSharedAccessInBytes>;
    // Swap the row and column of the `SwizzleBaseShape`.
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

    using NonSwizzled =
        tl::MatrixLayout<kSwizzleRows, kSwizzleCols, 1, Shared::kColStride>;
    using Swizzled =
        SwizzledLayout<NonSwizzled, SwizzleBaseShape::B, SwizzleBaseShape::M,
                       SwizzleBaseShape::S, tl::Layout::kColMajor>;

    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, Swizzled, NonSwizzled>;

    SharedLayout in_src_tile_;

    DEVICE int2 get_swizzled_tile_id(int offset) {
        // SwizzleTile is a 8 x 64 block.
        int swizzle_tile_row = (offset % kSharedColStride) / kSwizzleRows;
        int swizzle_tile_col = (offset / kSharedColStride) / kSwizzleCols;
        return make_int2(swizzle_tile_row, swizzle_tile_col);
    }

    DEVICE int2 get_in_swizzle_tile_id(int offset) {
        // Get id in the swizzle tile.
        auto swizzled_tile_id = get_swizzled_tile_id(offset);

        int row = offset % kSharedColStride;
        int col = offset / kSharedColStride;

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

template <typename Reg, typename Shared, const int kRowExec, const int kColExec,
          const int kSharedAccessInBytes, const tl::Layout kType>
struct RegToSharedStorerImpl;

template <typename Reg_, typename Shared_, const int kRowExec_,
          const int kColExec_, const int kSharedAccessInBytes>
struct RegToSharedStorerImpl<Reg_, Shared_, kRowExec_, kColExec_,
                             kSharedAccessInBytes, tl::Layout::kRowMajor>
    : public StoreMatBase<Shared_, tl::Layout::kRowMajor> {
    using Reg = Reg_;
    using Shared = Shared_;
    using DType = Shared::DType;
    using StoreMat = StoreMatBase<Shared, tl::Layout::kRowMajor>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const Reg& src, DType* dst, int warp_offset) {
#pragma unroll
        for (int j = 0; j < kColExec; ++j) {
#pragma unroll
            for (int i = 0; i < kRowExec; ++i) {
                int lane_row = this->lane_row_id();
                int lane_col = this->lane_col_id();

                int tile_offset = warp_offset + i * kRowStride + j * kColStride;
                int row = 0, col = 0;
#pragma unroll
                for (int m = 0; m < StoreMat::kSegRows; ++m) {
                    row = lane_row + m * StoreMat::kThreadRows;
#pragma unroll
                    for (int n = 0; n < StoreMat::kSegCols; ++n) {
                        col = StoreMat::kElemPerSeg *
                              (lane_col + n * StoreMat::kThreadCols);
                        int in_tile_offset = row * Shared::kRowStride + col;
                        int offset = tile_offset + in_tile_offset;
                        int swizzled_offset = get_swizzle_offset(offset);

                        const PackedType* src_ptr =
                            reinterpret_cast<const PackedType*>(
                                src(i, j).data());
                        PackedType* dst_ptr =
                            reinterpret_cast<PackedType*>(dst);

                        dst_ptr[swizzled_offset / StoreMat::kElemPerSeg] =
                            src_ptr[n * StoreMat::kSegCols + m];
                    }
                }
            }
        }
    }

  private:
    using BaseShape = traits::BaseTileShape<DType>;

    using SwizzledBaseShape =
        traits::SwizzleBaseTileShape<DType, kSharedAccessInBytes>;
    static constexpr int kSwizzledRows = SwizzledBaseShape::kRows;
    static constexpr int kSwizzledCols = SwizzledBaseShape::kCols;

    static constexpr int kRowStride = BaseShape::kRows * Shared::kRowStride;
    static constexpr int kColStride = BaseShape::kCols;

    static constexpr int kSharedRowStride = Shared::kRowStride;

    static constexpr int kSwizzledBlockRows =
        kRowExec * BaseShape::kRows / kSwizzledRows;
    static constexpr int kSwizzledBlockCols =
        kColExec * BaseShape::kCols / kSwizzledCols;

    using PackedType =
        typename Packing<DType, StoreMat::kElemPerSeg>::PackedType;

    using DstLayout =
        tl::MatrixLayout<kSwizzledBlockRows, kSwizzledBlockCols,
                         Shared::kRowStride * kSwizzledRows, kSwizzledCols>;
    DstLayout dst_tile_;

    using NonSwizzled =
        tl::MatrixLayout<kSwizzledRows, kSwizzledCols, Shared::kRowStride, 1>;
    using Swizzled =
        SwizzledLayout<NonSwizzled, SwizzledBaseShape::B, SwizzledBaseShape::M,
                       SwizzledBaseShape::S, tl::Layout::kRowMajor>;

    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, Swizzled, NonSwizzled>;
    SharedLayout in_dst_tile_;

    DEVICE int2 get_swizzled_tile_id(int offset) {
        // SwizzleTile is a 8 x 64 block.
        int swizzle_tile_col = (offset % kSharedRowStride) / kSwizzledCols;
        int swizzle_tile_row = (offset / kSharedRowStride) / kSwizzledRows;
        return make_int2(swizzle_tile_row, swizzle_tile_col);
    }

    DEVICE int2 get_in_swizzle_tile_id(int offset) {
        // Get id in the swizzle tile.
        auto swizzled_tile_id = get_swizzled_tile_id(offset);

        int row = offset / kSharedRowStride;
        int col = offset % kSharedRowStride;

        int in_swizzle_tile_col = col % kSwizzledCols;
        int in_swizzle_tile_row = row % kSwizzledRows;

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

template <typename Reg_, typename Shared_, const int kRowExec_,
          const int kColExec_, const int kSharedAccessInBytes>
struct RegToSharedStorerImpl<Reg_, Shared_, kRowExec_, kColExec_,
                             kSharedAccessInBytes, tl::Layout::kColMajor>
    : public StoreMatBase<Shared_, tl::Layout::kColMajor> {
    using Reg = Reg_;
    using Shared = Shared_;
    using DType = Shared::DType;
    using StoreMat = StoreMatBase<Shared, tl::Layout::kColMajor>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const Reg& src, DType* dst, int warp_offset) {
#pragma unroll
        for (int j = 0; j < kColExec; ++j) {
#pragma unroll
            for (int i = 0; i < kRowExec; ++i) {
                int tile_offset = warp_offset + j * kColStride + i * kRowStride;
                int lane_row = this->lane_row_id();
                int lane_col = this->lane_col_id();

                int row = 0, col = 0;
#pragma unroll
                for (int m = 0; m < StoreMat::kSegRows; ++m) {
                    row = StoreMat::kElemPerSeg *
                          (lane_row + m * StoreMat::kThreadRows);
#pragma unroll
                    for (int n = 0; n < StoreMat::kSegCols; ++n) {
                        col = lane_col + n * StoreMat::kThreadCols;

                        int in_tile_offset = col * Shared::kColStride + row;
                        int offset = tile_offset + in_tile_offset;
                        int swizzled_offset = get_swizzle_offset(offset);

                        const PackedType* src_ptr =
                            reinterpret_cast<const PackedType*>(
                                src(i, j).data());
                        PackedType* dst_ptr =
                            reinterpret_cast<PackedType*>(dst);
                        dst_ptr[swizzled_offset / StoreMat::kElemPerSeg] =
                            src_ptr[n * StoreMat::kSegCols + m];
                    }
                }
            }
        }
    }

  private:
    using BaseShape = traits::BaseTileShape<DType>;

    // Use 64x8 as a basic swizzle block shape in ColMajor layout.
    using SwizzledBaseShape =
        traits::SwizzleBaseTileShape<DType, kSharedAccessInBytes>;
    static constexpr int kSwizzleRows = SwizzledBaseShape::kCols;
    static constexpr int kSwizzleCols = SwizzledBaseShape::kRows;

    static constexpr int kSwizzleBlockRows =
        kRowExec * BaseShape::kRows / kSwizzleRows;
    static constexpr int kSwizzleBlockCols =
        kColExec * BaseShape::kCols / kSwizzleCols;

    static constexpr int kRowStride = BaseShape::kRows;
    static constexpr int kColStride = BaseShape::kCols * Shared::kColStride;

    static constexpr int kSharedColStride = Shared::kColStride;

    using PackedType =
        typename Packing<DType, StoreMat::kElemPerSeg>::PackedType;

    using DstLayout =
        tl::MatrixLayout<kSwizzleBlockRows, kSwizzleBlockCols, kSwizzleRows,
                         kSwizzleCols * Shared::kColStride>;
    DstLayout dst_tile_;

    using NonSwizzled =
        tl::MatrixLayout<kSwizzleRows, kSwizzleCols, 1, Shared::kColStride>;
    using Swizzled =
        SwizzledLayout<NonSwizzled, SwizzledBaseShape::B, SwizzledBaseShape::M,
                       SwizzledBaseShape::S, tl::Layout::kColMajor>;

    using SharedLayout =
        std::conditional_t<Shared::kSwizzled, Swizzled, NonSwizzled>;
    SharedLayout in_dst_tile_;

    DEVICE int2 get_swizzled_tile_id(int offset) {
        // SwizzleTile is a 8 x 64 block.
        int swizzle_tile_row = (offset % kSharedColStride) / kSwizzleRows;
        int swizzle_tile_col = (offset / kSharedColStride) / kSwizzleCols;
        return make_int2(swizzle_tile_row, swizzle_tile_col);
    }

    DEVICE int2 get_in_swizzle_tile_id(int offset) {
        // Get id in the swizzle tile.
        auto swizzled_tile_id = get_swizzled_tile_id(offset);

        int row = offset % kSharedColStride;
        int col = offset / kSharedColStride;

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
}  // namespace detail

/// @brief partial specialization for loading data from shared memory to
///        register file using `ldmatrix`.
template <typename Reg_, typename WarpLayout_, const WarpReuse kMode_>
struct SharedToRegLoader {
    using Reg = Reg_;
    using DType = typename Reg::DType::DType;  // the element data type
    using WarpLayout = WarpLayout_;
    static constexpr WarpReuse kMode = kMode_;

    using BaseShape = traits::BaseTileShape<DType>;

    // how many times a `BaseTile` is executed along the row and column
    // direction.
    static constexpr int kRowExec = Reg::kRows;
    static constexpr int kColExec = Reg::kCols;

    static_assert(kRowExec && kColExec,
                  "Execution count should be greater than 0.");

    template <typename Shared>
    DEVICE void operator()(const Shared& src, Reg& dst) {
        static_assert(std::is_same_v<typename Shared::DType, DType>,
                      "The data type of Shared and Reg must be the same.");
        static_assert(Shared::kRows % WarpLayout::kRows == 0,
                      "The current implementation requires Shared::kRows must "
                      "be divisible by WarpLayout::kRows");
        static_assert(Shared::kCols % WarpLayout::kCols == 0,
                      "The current implementation requires Shared::kCols must "
                      "be divisible by WarpLayout::kCols");

        static constexpr int kSharedContInBytes =
            Shared::kType == tl::Layout::kRowMajor
                ? Shared::kRowStride * sizeof(DType) / WarpLayout::kCols
                : Shared::kColStride * sizeof(DType) / WarpLayout::kRows;

        static_assert(kSharedContInBytes % 32 == 0,
                      "The number of bytes in a warp tile must be divisible by "
                      "32.");

        static constexpr int kSharedAccessInBytes =
            kSharedContInBytes >= 128 ? 128 : kSharedContInBytes;

        using SharedOffset =
            warp::SharedOffsetHelper<WarpLayout, BaseShape, Shared, kMode>;
        SharedOffset shared_offset_;

        // advance the pointer to input data to the current warp according to
        // warp reuse mode.
        int warp_offset = shared_offset_.get_warp_offset();
        int iterator_offset = src.get_offset();

        using Loader =
            detail::SharedToRegLoaderImpl<Shared, Reg, kRowExec, kColExec,
                                          kSharedAccessInBytes, Shared::kType>;
        Loader loader;
        loader(src.data(), dst, warp_offset, iterator_offset);
    }
};

/// @brief partial specialization for 16x16x16 wmma's output, and st.shared.f32
///        to revert the data distribution into an comprehensive row-major
///        matrix.
template <typename Reg_, typename WarpLayout_>
struct RegToSharedStorer {
    using Reg = Reg_;
    // elementary data type stored in the register tile.
    using DType = typename Reg::DType::DType;
    using WarpLayout = WarpLayout_;

    using BaseShape = traits::BaseTileShape<DType>;

    // how many times a `BaseTile` is executed along the row and column
    // direction.
    static constexpr int kRowExec = Reg::kRows;
    static constexpr int kColExec = Reg::kCols;

    static_assert(kRowExec && kColExec,
                  "Execution count should be greater than 0.");

    /// @brief Store the WMMA output register tile to shared memory. The source
    ///        is the current thread's local register tile, and the destination
    ///        is shared memory.
    template <typename Shared>
    DEVICE void operator()(const Reg& src, Shared& dst_) {
        static_assert(std::is_same_v<typename Shared::DType, DType>,
                      "The element data type of Shared and Register tile must "
                      "be the same.");
        static_assert((Reg::kNumel * Reg::DType::kNumel * 32 /*warp size*/ *
                       WarpLayout::kNumel) == Shared::kNumel,
                      "The number of elements held in the local register file "
                      "by all threads in the CTA must be the same as the "
                      "number held in the shared memory tile.");
        static_assert(
            Shared::kType == Reg::kType,
            "The layout of Shared and Register tile must be the same.");
        static_assert(Shared::kRows % BaseShape::kRows == 0,
                      "The number of shared memory rows must be divisible by "
                      "the base tile row.");
        static_assert(Shared::kCols % BaseShape::kCols == 0,
                      "The number of shared memory columns must be divisible "
                      "by the base tile column.");

        static constexpr int kSharedContInBytes =
            Shared::kType == tl::Layout::kRowMajor
                ? Shared::kRowStride * sizeof(DType) / WarpLayout::kCols
                : Shared::kColStride * sizeof(DType) / WarpLayout::kRows;

        static_assert(kSharedContInBytes % 32 == 0,
                      "The number of bytes in a warp tile must be divisible by "
                      "32.");

        static constexpr int kSharedAccessInBytes =
            kSharedContInBytes >= 128 ? 128 : kSharedContInBytes;

        // advance the pointer to input data to the current warp according to
        // warp reuse mode. During the store process, threads do not write to
        // the same shared memory location, thus the warp reuse mode is set to
        // `Cont`.
        using SharedOffset = warp::SharedOffsetHelper<WarpLayout, BaseShape,
                                                      Shared, WarpReuse::kCont>;
        SharedOffset shared_offset_;
        int warp_offset = shared_offset_.get_warp_offset();

        using Storer =
            detail::RegToSharedStorerImpl<Reg, Shared, kRowExec, kColExec,
                                          kSharedAccessInBytes, Reg::kType>;
        Storer storer;

        storer(src, dst_.mutable_data(), warp_offset);
    }
};
}  // namespace tilefusion::cell::copy
