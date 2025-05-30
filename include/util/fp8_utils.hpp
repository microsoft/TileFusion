// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "config.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// CUDA C native FP8 support
// Requires CUDA 11.8+ AND Ada Lovelace (8.9+) or Hopper (9.0+) architecture
#if defined(__CUDA_ARCH__) &&                                      \
    (__CUDACC_VER_MAJOR__ >= 12 ||                                 \
     (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 8)) && \
    (__CUDA_ARCH__ >= 890)  // Ada Lovelace (8.9) or Hopper (9.0+)
    #include <cuda_fp8.h>
    #define CUDA_FP8_AVAILABLE 1
#endif

namespace tilefusion::util {

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

}  // namespace tilefusion::util
