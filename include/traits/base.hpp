// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
typedef __nv_bfloat16 __bfloat16;

#include <cutlass/numeric_size.h>
#include <cutlass/numeric_types.h>

#include <type_traits>

namespace tilefusion::traits {

template <typename Element>
concept BaseType =
    std::is_same_v<Element, float> || std::is_same_v<Element, __half> ||
    std::is_same_v<Element, cutlass::half_t> ||
    std::is_same_v<Element, __bfloat16> ||
    std::is_same_v<Element, cutlass::bfloat16_t>;

/// @brief Architecture-specific magic numbers.
/// @tparam Element: the data type of the elements.
template <typename Element>
struct AccessBase {
    // the maximal width of vectorized access.
    static constexpr int kAccessInBits = 128;
    static constexpr int kElementBits = cutlass::sizeof_bits<Element>::value;
    static constexpr int kNumPerAccess = kAccessInBits / kElementBits;

    // the width of memory transaction
    static constexpr int kMemTransWidth = 1024;  // 1024 bits, 128 bytes

    // The ideal number of columns for a single warp to load.
    // When loading data through the L1 cache, an entire 128-byte cache line is
    // fetched. Ensuring contiguous threads read contiguous data in memory
    // optimizes the usage of the L1 cache.
    static constexpr int kExpectedSize = kMemTransWidth / kElementBits;
};

template <typename Element>
    requires BaseType<Element>
struct BaseTileShape {
    using DType = Element;

    static constexpr int kTileSize = 16;
    static constexpr int kRows = kTileSize;
    static constexpr int kCols = kTileSize;
    static constexpr int kNumel = kRows * kCols;
};

/**
 * @brief The base tile shape for Swizzle<3, 3, 3>.
 */
template <typename Element>
    requires BaseType<Element>
struct SwizzleBaseTileShape {
    using DType = Element;

    static constexpr int kRows = 8;
    static constexpr int kCols = 64;
    static constexpr int kNumel = kRows * kCols;
};

}  // namespace tilefusion::traits
