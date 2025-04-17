#pragma once

#include "cuda_utils.hpp"

namespace tilefusion::cell {

enum class MaskMode {
    kNone = 0U,    // No mask
    kCausal = 1U,  // Causal mask
    kCustom = 2U,  // Custom mask
};

template <typename RegTile, typename WarpLayout, typename BaseShape,
          MaskMode mode>
struct ApplyMask;

template <typename RegTile, typename WarpLayout, typename BaseShape>
struct ApplyMask<RegTile, WarpLayout, BaseShape, MaskMode::kNone> {
    DEVICE void operator()(RegTile& tile, const int row_offset,
                           const int col_offset) {
        return;
    }
};

template <typename RegTile, typename WarpLayout, typename BaseShape>
struct ApplyMask<RegTile, WarpLayout, BaseShape, MaskMode::kCausal> {
    using Element = RegTile::DType::DType;
    static_assert(WarpLayout::kCols == 1, "WarpLayout::kCols must be 1");
    // static_assert(std::is_same_v<Element, float> ||
    //                   std::is_same_v<Element, __half>,
    //               "Element must be float or half");
    // TODO(KuangjuX): support half precision.
    static_assert(std::is_same_v<Element, float>, "Element must be float");

    // Each thread processes 2 consecutive elements in the register tile.
    static constexpr int kThreadStride = 2;
    // Each 4 threads as a group to process a row of the register tile.
    static constexpr int kThreadGroupSize = 4;
    static constexpr int kThreadPerWarp = 32;

    static constexpr int kRegTileRows = RegTile::kRows;
    static constexpr int kRegTileCols = RegTile::kCols;
    static constexpr int kSubtileRows = RegTile::DType::kRows;
    static constexpr int kSubtileCols = RegTile::DType::kCols;

    static constexpr int kWarpRows = WarpLayout::kRows;
    static constexpr int kWarpCols = WarpLayout::kCols;

    // A BaseTile is a 16x16 tile by default.
    static constexpr int kBaseShapeRows = BaseShape::kRows;
    static constexpr int kBaseShapeCols = BaseShape::kCols;

    DEVICE void operator()(RegTile& tile, const int row_offset,
                           const int col_offset) {
        const int lane_id = threadIdx.x % kThreadPerWarp;
        const int nrow_group = threadIdx.x / kThreadPerWarp;

        // Compute the column index offset for the current thread.
        const int col_idx_offset =
            col_offset + (lane_id % kThreadGroupSize) * kThreadStride;

        // Compute the row index offset for the current thread.
        const int row_idx_offset = row_offset + (lane_id / kThreadGroupSize) +
                                   nrow_group * kBaseShapeRows;

        for (int m = 0; m < kRegTileRows; ++m) {
            for (int n = 0; n < kRegTileCols; ++n) {
                for (int i = 0; i < kSubtileRows; ++i) {
                    for (int j = 0; j < kSubtileCols; ++j) {
                        const int col_idx =
                            col_idx_offset + n * kBaseShapeCols +
                            (j / 2) * (kBaseShapeCols / 2) + (j % 2);
                        const int row_idx = row_idx_offset +
                                            m * kWarpRows * kBaseShapeRows +
                                            i * kBaseShapeRows / 2;
                        if (col_idx >= row_idx) {
                            tile(m, n)(i, j) = -INFINITY;
                            if (thread0()) {
                                printf("tile:\n");
                                // sub_tile.dump_value();
                                tile.dump_value();
                            }
                        }
                    }
                }
            }
        }
    }
};

}  // namespace tilefusion::cell
