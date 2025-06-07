// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_info.hpp"

#include "cuda_utils.hpp"

#include <sstream>
#include <vector>

namespace tilefusion {
// Returns the number of GPUs.
int GetGPUDeviceCount() {
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  return deviceCount;
}

// Returns the compute capability of the given GPU.
int GetGPUComputeCapability(int id) {
  int major, minor;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, id));
  CUDA_CHECK(
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, id));
  return major * 10 + minor;
}

// Returns the number of multiprocessors for the given GPU.
int GetGPUMultiProcessors(int id) {
  int count;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, id));
  return count;
}

// Returns the maximum number of threads per multiprocessor for the given GPU.
int GetGPUMaxThreadsPerMultiProcessor(int id) {
  int count;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &count, cudaDevAttrMaxThreadsPerMultiProcessor, id));
  return count;
}

// Returns the maximum number of threads per block for the given GPU.
int GetGPUMaxThreadsPerBlock(int id) {
  int count;
  CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMaxThreadsPerBlock, id));
  return count;
}

// Returns the maximum grid size for the given GPU.
dim3 GetGpuMaxGridDimSize(int id) {
  dim3 grid_size;

  int size;
  CUDA_CHECK(cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimX, id));
  grid_size.x = size;

  CUDA_CHECK(cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimY, id));
  grid_size.y = size;

  CUDA_CHECK(cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimZ, id));
  grid_size.z = size;
  return grid_size;
}

// Returns the name of the device.
std::string GetDeviceName() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  std::stringstream ss(prop.name);
  const char delim = ' ';

  std::string s;
  std::vector<std::string> out;

  while (std::getline(ss, s, delim)) {
    out.push_back(s);
  }

  std::stringstream out_ss;
  int i = 0;
  for (; i < static_cast<int>(out.size()) - 1; ++i) out_ss << out[i] << "_";
  out_ss << out[i];
  return out_ss.str();
}

std::string GetComputeCapability() {
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

  std::stringstream ss;
  ss << "sm_" << prop.major << prop.minor;
  return ss.str();
}

int GetMaxSharedMemoryPerBlock() {
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  return prop.sharedMemPerBlock;
}

void check_gpu_memory() {
  size_t free_byte;
  size_t total_byte;
  CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));

  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;
  printf("GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n",
         used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0,
         total_db / 1024.0 / 1024.0);
}

/**
 * Configure dynamic shared memory for a kernel if needed
 * @param kernel The CUDA kernel function pointer
 * @param shared_memory_size Required shared memory size in bytes
 */
template <typename KernelFunc>
void configure_dynamic_shared_memory(KernelFunc kernel,
                                     int shared_memory_size) {
  if (shared_memory_size > GetMaxSharedMemoryPerBlock()) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shared_memory_size));
  }
}

}  // namespace tilefusion
