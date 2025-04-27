// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.hpp"

#include <string>

namespace tilefusion {
// Returns the number of GPUs.
int GetGPUDeviceCount();

// Returns the compute capability of the given GPU.
int GetGPUComputeCapability(int id);

// Returns the number of multiprocessors for the given GPU.
int GetGPUMultiProcessors(int id);

// Returns the maximum number of threads per multiprocessor for the given
// GPU.
int GetGPUMaxThreadsPerMultiProcessor(int id);

// Returns the maximum number of threads per block for the given GPU.
int GetGPUMaxThreadsPerBlock(int id);

// Returns the maximum grid size for the given GPU.
dim3 GetGpuMaxGridDimSize(int id);

// Returns the name of the device.
std::string GetDeviceName();

// Returns the compute capability of the current device.
std::string GetComputeCapability();
}  // namespace tilefusion
