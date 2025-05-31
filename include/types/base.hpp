// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "config.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

typedef __nv_bfloat16 __bfloat16;

namespace tilefusion {

template <typename Element>
concept BaseType =
    std::is_same_v<Element, float> || std::is_same_v<Element, __half> ||
    std::is_same_v<Element, __bfloat16>
#ifdef CUDA_FP8_AVAILABLE
    || std::is_same_v<Element, __nv_fp8_e4m3> ||
    std::is_same_v<Element, __nv_fp8_e5m2>
#endif
    ;

template <typename Element>
concept HalfType =
    std::is_same_v<Element, __half> || std::is_same_v<Element, __bfloat16>;

#ifdef CUDA_FP8_AVAILABLE
template <typename Element>
concept Fp8Type = std::is_same_v<Element, __nv_fp8_e4m3> ||
                  std::is_same_v<Element, __nv_fp8_e5m2>;
#endif

/// @brief Architecture-specific magic numbers.
/// @param Element: the data type of the elements.
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

#ifdef CUDA_FP8_AVAILABLE
/// @brief Convert from float to target type using built-in
/// constructors/conversions
template <typename T>
DEVICE T from_float(float val) {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        return __nv_fp8_e4m3(val);
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
        return __nv_fp8_e5m2(val);
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half(val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(val);
    } else if constexpr (std::is_same_v<T, float>) {
        return val;
    } else {
        static_assert(sizeof(T) == 0,
                      "Unsupported type for from_float conversion");
    }
}

/// @brief Convert from source type to float using built-in cast
/// operators/conversions
template <typename T>
DEVICE float to_float(const T& val) {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        return static_cast<float>(val);
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
        return static_cast<float>(val);
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(val);
    } else if constexpr (std::is_same_v<T, float>) {
        return val;
    } else {
        static_assert(sizeof(T) == 0,
                      "Unsupported type for to_float conversion");
    }
}

#else  // !CUDA_FP8_AVAILABLE

/// @brief Fallback conversion functions when FP8 is not available
template <typename T>
DEVICE T from_float(float val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(val);
    } else if constexpr (std::is_same_v<T, float>) {
        return val;
    } else {
        static_assert(sizeof(T) == 0,
                      "FP8 types not available - requires Ada Lovelace (RTX "
                      "4090) or Hopper (H100) GPU with CUDA 11.8+");
    }
}

template <typename T>
DEVICE float to_float(const T& val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(val);
    } else if constexpr (std::is_same_v<T, float>) {
        return val;
    } else {
        static_assert(sizeof(T) == 0,
                      "FP8 types not available - requires Ada Lovelace (RTX "
                      "4090) or Hopper (H100) GPU with CUDA 11.8+");
    }
}

#endif  // CUDA_FP8_AVAILABLE
}  // namespace tilefusion
