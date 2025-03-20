// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cell/copy/constants.hpp"
#include "cell/copy/vectorize.hpp"
#include "cell/copy/warp.hpp"
#include "traits/base.hpp"
#include "types/mod.hpp"

namespace tilefusion::cell::copy {

/**
 * @brief Load a vector from Global memory to Register.
 * @param Global Global memory vector type.
 * @param RegVec Register vector type.
 * @param kVecSize Vector size.
 */
template <typename Global_, typename RegVec_, const int kVecSize_>
struct GlobalToRegVecLoader {
    using Global = Global_;
    using RegVec = RegVec_;
    using DType = typename Global::DType;

    static constexpr int kVecSize = kVecSize_;
    static constexpr int kNumPerAccess = traits::AccessBase<DType>::kNumPerAccess;

    DEVICE void operator()(const DType* src, RegVec& dst) {
        Vectorize<DType, kNumPerAccess> vectorize;
        
        // Calculate thread's position
        int lane_id = threadIdx.x % WARP_SIZE;
        int vec_offset = lane_id * kNumPerAccess;

        // Load vector elements in chunks of kNumPerAccess
        #pragma unroll
        for (int i = 0; i < kVecSize / kNumPerAccess; ++i) {
            vectorize.copy(src + vec_offset + i * WARP_SIZE * kNumPerAccess, 
                         dst.data() + i * kNumPerAccess);
        }
    }
};

/**
 * @brief Store a vector from Register to Global memory.
 * @param Global Global memory vector type.
 * @param RegVec Register vector type.
 * @param kVecSize Vector size.
 */
template <typename Global_, typename RegVec_, const int kVecSize_>
struct RegToGlobalVecStorer {
    using Global = Global_;
    using RegVec = RegVec_;
    using DType = typename Global::DType;

    static constexpr int kVecSize = kVecSize_;
    static constexpr int kNumPerAccess = traits::AccessBase<DType>::kNumPerAccess;

    DEVICE void operator()(const RegVec& src, DType* dst) {
        Vectorize<DType, kNumPerAccess> vectorize;
        
        // Calculate thread's position
        int lane_id = threadIdx.x % WARP_SIZE;
        int vec_offset = lane_id * kNumPerAccess;

        // Store vector elements in chunks of kNumPerAccess
        #pragma unroll
        for (int i = 0; i < kVecSize / kNumPerAccess; ++i) {
            vectorize.copy(src.data() + i * kNumPerAccess,
                         dst + vec_offset + i * WARP_SIZE * kNumPerAccess);
        }
    }
};

/**
 * @brief Load a vector from Global memory to Register with warp reuse.
 * @param Global Global memory vector type.
 * @param RegVec Register vector type.
 * @param WarpLayout_ Warp layout type.
 * @param kMode_ Warp reuse mode.
 */
template <typename RegVec_, typename WarpLayout_, const WarpReuse kMode_>
struct GlobalToRegVecLoaderWithReuse {
    using RegVec = RegVec_;
    using DType = typename RegVec::DType;
    using WarpLayout = WarpLayout_;
    static constexpr WarpReuse kMode = kMode_;

    template <typename Global>
    DEVICE void operator()(const Global& src, RegVec& dst) {
        // Get warp offset based on reuse mode
        int offset = global_offset_.template get_warp_offset<Global>();

        using Loader = GlobalToRegVecLoader<Global, RegVec, RegVec::kSize>;
        Loader loader;
        loader(src.data() + offset, dst);
    }

private:
    using GlobalOffset = warp::GlobalOffsetHelper<WarpLayout, kMode>;
    GlobalOffset global_offset_;
};

/**
 * @brief Store a vector from Register to Global memory with warp reuse.
 * @param Global Global memory vector type.
 * @param RegVec Register vector type.
 * @param WarpLayout_ Warp layout type.
 */
template <typename Global_, typename RegVec_, typename WarpLayout_>
struct RegToGlobalVecStorerWithReuse {
    using Global = Global_;
    using RegVec = RegVec_;
    using DType = typename Global::DType;
    using WarpLayout = WarpLayout_;

    DEVICE void operator()(const RegVec& src, Global& dst) {
        DType* dst_ptr = dst.mutable_data();

        // Get warp offset based on continuous mode
        int offset = global_offset_.template get_warp_offset<Global>();

        using Storer = RegToGlobalVecStorer<Global, RegVec, RegVec::kSize>;
        Storer storer;
        storer(src, dst_ptr + offset);
    }

private:
    using GlobalOffset = warp::GlobalOffsetHelper<WarpLayout, WarpReuse::kCont>;
    GlobalOffset global_offset_;
};

}  // namespace tilefusion::cell::copy 