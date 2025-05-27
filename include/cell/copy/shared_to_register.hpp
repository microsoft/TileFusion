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
          const int kColExec, const tl::Layout kType>
struct SharedToRegLoaderImpl;

/// @brief partial specialization for row-major shared memory tile.
template <typename Shared, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct SharedToRegLoaderImpl<Shared, Reg_, kRowExec_, kColExec_,
                             tl::Layout::kRowMajor>
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
                int offset = shared_tile.fetch_physical_offset(tile_offset) -
                             iterator_offset;

                // advance pointer to the 16x16 `BaseTile` indexed by(i, j).
                // issue the hardware-backed memory access instruction.
                this->ldmatrix(src + offset, dst(i, j).mutable_data());
            }
        }
    }

  private:
    using MmaAtom =
        compute::MmaAtom<__half, __half, __half, compute::MMA_ATOM_16x16x16>;
    using BaseShape = MmaAtom::BaseTile;
    static constexpr int kSharedRowStride = Shared::kRowStride;
    Shared shared_tile;
};

/// @brief partial specialization for column-major shared memory tile.
template <typename Shared, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct SharedToRegLoaderImpl<Shared, Reg_, kRowExec_, kColExec_,
                             tl::Layout::kColMajor>
    : public LoadMatBase<typename Shared::DType> {
    using Reg = Reg_;
    using DType = Shared::DType;
    using LoadMat = LoadMatBase<DType>;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

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
                int offset = shared_tile.fetch_physical_offset(tile_offset) -
                             iterator_offset;

                // issue the hardware-backed memory access instruction
                this->ldmatrix(src + offset, dst(j, i).mutable_data());
            }
        }
    }

  private:
    using MmaAtom =
        compute::MmaAtom<__half, __half, __half, compute::MMA_ATOM_16x16x16>;
    using BaseShape = MmaAtom::BaseTile;
    static constexpr int kSharedColStride = Shared::kColStride;
    Shared shared_tile;
};

template <typename Reg, typename Shared, const int kRowExec, const int kColExec,
          const tl::Layout kType>
struct RegToSharedStorerImpl;

template <typename Reg_, typename Shared_, const int kRowExec_,
          const int kColExec_>
struct RegToSharedStorerImpl<Reg_, Shared_, kRowExec_, kColExec_,
                             tl::Layout::kRowMajor>
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
                        int swizzled_offset =
                            shared_tile.fetch_physical_offset(offset);

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
    using MmaAtom =
        compute::MmaAtom<__half, __half, __half, compute::MMA_ATOM_16x16x16>;
    using BaseShape = MmaAtom::BaseTile;
    using PackedType =
        typename Packing<DType, StoreMat::kElemPerSeg>::PackedType;

    static constexpr int kSharedRowStride = Shared::kRowStride;
    static constexpr int kRowStride = BaseShape::kRows * kSharedRowStride;
    static constexpr int kColStride = BaseShape::kCols;

    Shared shared_tile;
};

template <typename Reg_, typename Shared_, const int kRowExec_,
          const int kColExec_>
struct RegToSharedStorerImpl<Reg_, Shared_, kRowExec_, kColExec_,
                             tl::Layout::kColMajor>
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
                        int swizzled_offset =
                            shared_tile.fetch_physical_offset(offset);

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
    using MmaAtom =
        compute::MmaAtom<__half, __half, __half, compute::MMA_ATOM_16x16x16>;
    using BaseShape = MmaAtom::BaseTile;
    using PackedType =
        typename Packing<DType, StoreMat::kElemPerSeg>::PackedType;

    static constexpr int kSharedColStride = Shared::kColStride;
    static constexpr int kRowStride = BaseShape::kRows;
    static constexpr int kColStride = BaseShape::kCols * kSharedColStride;

    Shared shared_tile;
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

    using MmaAtom =
        compute::MmaAtom<__half, __half, __half, compute::MMA_ATOM_16x16x16>;
    using BaseShape = MmaAtom::BaseTile;

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

        static constexpr int kSharedAccessInBytes = Shared::SwizzleBytes;
        static_assert(kSharedAccessInBytes % 32 == 0,
                      "The number of bytes in a warp tile must be divisible by "
                      "32.");

        using SharedOffset =
            warp::SharedOffsetHelper<WarpLayout, BaseShape, Shared, kMode>;
        SharedOffset shared_offset_;

        // advance the pointer to input data to the current warp according to
        // warp reuse mode.
        int warp_offset = shared_offset_.get_warp_offset();
        int iterator_offset = src.get_offset();

        using Loader = detail::SharedToRegLoaderImpl<Shared, Reg, kRowExec,
                                                     kColExec, Shared::kType>;
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

    using MmaAtom =
        compute::MmaAtom<__half, __half, __half, compute::MMA_ATOM_16x16x16>;
    using BaseShape = MmaAtom::BaseTile;

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

        static constexpr int kSharedAccessInBytes = Shared::SwizzleBytes;

        static_assert(kSharedAccessInBytes % 32 == 0,
                      "The number of bytes in a warp tile must be divisible by "
                      "32.");

        // advance the pointer to input data to the current warp according to
        // warp reuse mode. During the store process, threads do not write to
        // the same shared memory location, thus the warp reuse mode is set to
        // `Cont`.
        using SharedOffset = warp::SharedOffsetHelper<WarpLayout, BaseShape,
                                                      Shared, WarpReuse::kCont>;
        SharedOffset shared_offset_;
        int warp_offset = shared_offset_.get_warp_offset();

        using Storer = detail::RegToSharedStorerImpl<Reg, Shared, kRowExec,
                                                     kColExec, Reg::kType>;
        Storer storer;

        storer(src, dst_.mutable_data(), warp_offset);
    }
};
}  // namespace tilefusion::cell::copy
