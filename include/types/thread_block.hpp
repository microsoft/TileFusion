// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// reference:
/// https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/threadblock_swizzle.h

#pragma once

#include "cuda_utils.hpp"

namespace tilefusion::cell {

template <const int kM, const int kN, const int kTM, const int kTN>
struct GemmThreadBlockSwizzle {
    static constexpr int kTiledBlockM = (kM + kTM - 1) / kTM;
    static constexpr int kTiledBlockN = (kN + kTN - 1) / kTN;

    static constexpr int N = 8;

    /// @brief Computes CUDA grid dimensions given a size in units of logical
    /// tiles
    HOST_DEVICE static dim3 get_grid_shape() {
        int tile = 1 << get_log_tile();
        return dim3(kTiledBlockM * tile, (kTiledBlockN + tile - 1) / tile, 1);
    }

    DEVICE static dim3 get_tile_offset() {
        int block_idx = blockIdx.x;
        int block_idy = blockIdx.y;
        int block_idz = blockIdx.z;

        int log_tile = get_log_tile();

        return dim3((block_idx >> log_tile),
                    (block_idy << log_tile) + block_idx % (1 << (log_tile)),
                    block_idz);
    }

  private:
    /// @brief Calculates optimal swizzle width
    HOST_DEVICE static int get_log_tile() {
        if (N >= 8 && kTiledBlockN >= 6)
            return 3;
        else if (N >= 4 && kTiledBlockN >= 3)
            return 2;
        else if (N >= 2 && kTiledBlockN >= 2)
            return 1;
        else
            return 0;
    }
};

}  // namespace tilefusion::cell