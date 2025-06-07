// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.hpp"

namespace tilefusion {
/// @brief: Cuda timer to measure the time taken by a kernel.
/// Usage:
///    CudaTimer timer;
///    timer.start();
///        ...
///    float time = timer.stop();
class CudaTimer {
 public:
  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
  }

  ~CudaTimer() {
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
  }

  void start(cudaStream_t st = 0) {
    stream = st;
    CUDA_CHECK(cudaEventRecord(start_event, stream));
  }

  float stop() {
    float milliseconds = 0.;
    CUDA_CHECK(cudaEventRecord(stop_event, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    return milliseconds;
  }

 private:
  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  cudaStream_t stream;
};
}  // namespace tilefusion
