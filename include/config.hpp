// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(__CUDA_ARCH__)
    #define HOST_DEVICE __forceinline__ __host__ __device__
    #define DEVICE __forceinline__ __device__
    #define HOST __forceinline__ __host__
#else
    #define HOST_DEVICE inline
    #define DEVICE inline
    #define HOST inline
#endif

#if defined(__CUDACC__)
    #define WARP_SIZE 32
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    #define CP_ASYNC_SM80_ENABLED
#endif

// FP8 support requires CUDA 11.8+ AND Ada Lovelace (8.9+) or Hopper (9.0+)
// architecture
#if defined(__CUDA_ARCH__) &&                                      \
    (__CUDACC_VER_MAJOR__ >= 12 ||                                 \
     (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 8)) && \
    (__CUDA_ARCH__ >= 890)  // Ada Lovelace (8.9) or Hopper (9.0+)
    #include <cuda_fp8.h>
    #define CUDA_FP8_AVAILABLE 1
#endif
