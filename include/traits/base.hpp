// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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

typedef __nv_bfloat16 __bfloat16;

namespace tilefusion::traits {

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
}  // namespace tilefusion::traits
