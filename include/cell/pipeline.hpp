// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cell/copy/mod.hpp"
#include "config.hpp"
#include "cuda_utils.hpp"

using namespace tilefusion::cell::copy;
namespace tilefusion::cell {

/**
 * @brief Multiple-stage pipeline primitive.
 *
 * Consider following pipeline:
 *
 * ```
 * load2rmem(0);
 * for(int i = 0; i < N-1; ++i){
 *     load2rmem(i+1);
 *     compute(i);
 * }
 * compute(N-1);
 * ```
 *
 * ```
 * pipeline.prologue();
 * for(int i = 0; i < Pipeline::kBodyIterator; ++i){
 *     pipeline.body(i);
 *     // other computation
 *
 *     // Wait for the previous stage to complete.
 *     pipeline.wait_group(1);
 * }
 * pipeline.epilogue();
 * pipeline.wait_group(0);
 *
 * // other computation
 * ```
 *
 * @tparam SrcTile Source tile type.
 * @tparam DstTile Destination tile type.
 * @tparam TileIterator Tile iterator type.
 * @tparam NUM_STAGES Number of pipeline stages.
 */
template <typename Element, typename SrcTile, typename DstTile,        //
          typename TileIterator, typename Copy, const int NUM_STAGES,  //
          const int Iterations_ = TileIterator::sc1 - NUM_STAGES + 1>
struct Pipeline {
  public:
    // The number of iterations for the body kernel.
    static constexpr int Iterations = Iterations_;

    DEVICE Pipeline(const Element* src_ptr, Element* dst_ptr)
        : src_tile(SrcTile(src_ptr)),
          tile_iter(TileIterator(src_tile.data())),
          data_ptr(0),
          cur_stages(0) {
        // initialize the circular buffer
        for (int i = 0; i < NUM_STAGES; i++) {
            cyc_buffer[i] = DstTile(dst_ptr + i * DstTile::kNumel);
        }
    }

    DEVICE Pipeline(SrcTile src_tile, DstTile dst_tiles[])
        : src_tile(src_tile),
          tile_iter(TileIterator(src_tile.data())),
          data_ptr(0),
          cur_stages(0) {
        for (int i = 0; i < NUM_STAGES; i++) {
            cyc_buffer[i] = dst_tiles[i];
        }
    }

    /**
     * @brief Reset the source tile.
     * @param src_ptr The pointer to the source tile.
     */
    DEVICE void reset_src_tile(const Element* src_ptr) {
        src_tile = SrcTile(src_ptr);
        tile_iter = TileIterator(src_tile.data());
        data_ptr = 0;
    }

    /**
     * @brief Commit the copy operation.
     */
    DEVICE void commit(bool async = false) {
        // assert(data_ptr < Iterations);
        copy(tile_iter(data_ptr), cyc_buffer[cur_stages % NUM_STAGES]);

        if (async) {
            commit_copy_group();
        }
        data_ptr++;
        cur_stages++;
    }

    DEVICE const Element* get_dst_ptr_by_index(int index) const {
        return cyc_buffer[index % NUM_STAGES].data();
    }

    DEVICE const DstTile& get_dst_tile_by_index(int index) const {
        return cyc_buffer[index % NUM_STAGES];
    }

    DEVICE const Element* get_prev_dst() const {
        return cyc_buffer[(cur_stages - 2) % NUM_STAGES].data();
    }

    DEVICE const Element* get_cur_dst() const {
        return cyc_buffer[(cur_stages - 1) % NUM_STAGES].data();
    }

    /**
     * @brief Dump the destination tile value.
     * @param index The index of the destination tile.
     */
    DEVICE void dump_dst_tile_value(int index) {
        if (threadIdx.x == 0) {
            printf("data[%d]: \n", index);
            cyc_buffer[index % NUM_STAGES].dump_value();
        }
    }

  private:
    static constexpr int kNumStages = NUM_STAGES;

    int data_ptr;
    int cur_stages;
    SrcTile src_tile;
    // In multistage pipeline, the destination tile has circular buffer with a
    // size of `NUM_STAGES`.
    DstTile cyc_buffer[NUM_STAGES];
    TileIterator tile_iter;
    Copy copy;
};
}  // namespace tilefusion::cell
