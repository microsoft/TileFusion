// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

typedef __nv_bfloat16 __bfloat16;

#include <type_traits>

namespace tilefusion::traits {

template <typename Element>
concept BaseType =
    std::is_same_v<Element, float> || std::is_same_v<Element, __half> ||
    std::is_same_v<Element, __bfloat16>;

template <typename Element>
concept HalfType =
    std::is_same_v<Element, __half> || std::is_same_v<Element, __bfloat16>;

/// @brief Architecture-specific magic numbers.
/// @tparam Element: the data type of the elements.
template <typename Element>
struct AccessBase {
    // the maximal width of vectorized access.
    static constexpr int kAccessInBits = 128;
    static constexpr int kAccessInBytes = kAccessInBits / 8;

    static constexpr int kElementBits = sizeof(Element) * 8;
    static constexpr int kNumPerAccess = kAccessInBits / kElementBits;

    // the width of memory transaction, Shared memory cacheline width.
    static constexpr int kMemTransWidth = 1024;  // 1024 bits, 128 bytes

    // The ideal number of columns for a single warp to load.
    // When loading data through the L1 cache, an entire 128-byte cache line is
    // fetched. Ensuring contiguous threads read contiguous data in memory
    // optimizes the usage of the L1 cache.
    static constexpr int kExpectedSize = kMemTransWidth / kElementBits;
};

// FIXME(ying): Legacy code, remove it gradually.
template <typename Element>
    requires BaseType<Element>
struct BaseTileShape {
    static constexpr int kRows = 16;
    static constexpr int kCols = 16;
    static constexpr int kNumel = 256 /* kRows * kCols */;
};

/**
 * @brief The base tile shape for Swizzle<3, 3, 3>.
 */
template <typename Element, int kBytes>
    requires BaseType<Element>
struct SwizzleBaseTileShape;

template <typename Element>
    requires HalfType<Element>
struct SwizzleBaseTileShape<Element, 128> {
    using DType = Element;

    static constexpr int kRows = 8;
    static constexpr int kCols = 64;
    static constexpr int kNumel = kRows * kCols;

    static constexpr int B = 3;
    static constexpr int M = 3;
    static constexpr int S = 3;
};

template <>
struct SwizzleBaseTileShape<float, 128> {
    using DType = float;

    static constexpr int kRows = 8;
    static constexpr int kCols = 32;
    static constexpr int kNumel = kRows * kCols;

    static constexpr int B = 3;
    static constexpr int M = 2;
    static constexpr int S = 3;
};

template <typename Element>
    requires HalfType<Element>
struct SwizzleBaseTileShape<Element, 64> {
    using DType = Element;

    static constexpr int kRows = 4;
    static constexpr int kCols = 32;
    static constexpr int kNumel = kRows * kCols;

    static constexpr int B = 2;
    static constexpr int M = 3;
    static constexpr int S = 2;
};

template <>
struct SwizzleBaseTileShape<float, 64> {
    using DType = float;

    static constexpr int kRows = 4;
    static constexpr int kCols = 16;
    static constexpr int kNumel = kRows * kCols;

    static constexpr int B = 2;
    static constexpr int M = 2;
    static constexpr int S = 2;
};

}  // namespace tilefusion::traits
