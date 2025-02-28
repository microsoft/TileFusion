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

template <>
struct SwizzleBaseTileShape<__half, 128> {
    using DType = __half;

    static constexpr int kRows = 8;
    static constexpr int kCols = 64;
    static constexpr int kNumel = kRows * kCols;

    static constexpr int B = 3;
    static constexpr int M = 3;
    static constexpr int S = 3;
};

template <>
struct SwizzleBaseTileShape<cutlass::half_t, 128> {
    using DType = cutlass::half_t;

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

template <>
struct SwizzleBaseTileShape<__half, 64> {
    using DType = __half;

    static constexpr int kRows = 4;
    static constexpr int kCols = 32;
    static constexpr int kNumel = kRows * kCols;

    static constexpr int B = 2;
    static constexpr int M = 3;
    static constexpr int S = 2;
};

template <>
struct SwizzleBaseTileShape<cutlass::half_t, 64> {
    using DType = cutlass::half_t;

    static constexpr int kRows = 2;
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
