// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "cuda_utils.hpp"
#include "jit/mod.hpp"
#include "kernels/common.hpp"

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <sstream>

namespace tilefusion::testing {

using namespace tilefusion::jit;

namespace {
std::string generate_add_kernel_source(const std::string& dtype, int numel) {
  std::stringstream ss;
  ss << R"(
template <typename DType, const int kNumel>
__device__ void add_device(const DType* a, const DType* b, DType* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < kNumel) out[idx] = a[idx] + b[idx];
}

extern "C" __global__ void add_kernel_)"
     << dtype << "_" << numel << R"((
    const )"
     << dtype << R"(* a, const )" << dtype << R"(* b, )" << dtype << R"(* out) {
    add_device<)"
     << dtype << ", " << numel << R"(>(a, b, out);
}
)";
  return ss.str();
}

template <typename DType>
void jit_add_template(const DType* a, const DType* b, DType* out, int n) {
  if (n == 0) return;

  std::string dtype = get_type_string<DType>();
  std::string kernel_source = generate_add_kernel_source(dtype, n);
  std::string kernel_name = "add_kernel_" + dtype + "_" + std::to_string(n);

  auto& jit = JitCompiler::instance();
  CUfunction kernel = jit.get_or_compile_kernel(kernel_name, kernel_source);

  if (!kernel) {
    throw std::runtime_error("Failed to compile or retrieve kernel");
  }

  int block_size = 128;
  int grid_size = (n + block_size - 1) / block_size;

  void* args[] = {&a, &b, &out};

  CUDA_DRIVER_CHECK(cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0,
                                   nullptr, args, nullptr));

  LOG(INFO) << "Kernel launched successfully";
}
}  // namespace

TEST(TESTJit, test_jit) {
  const int kNumel = 1024;
  using Element = float;

  thrust::host_vector<Element> h_a(kNumel);
  for (size_t i = 0; i < h_a.size(); ++i) h_a[i] = rand_float();

  thrust::host_vector<Element> h_b(kNumel);
  for (size_t i = 0; i < h_b.size(); ++i) h_b[i] = rand_float();

  thrust::host_vector<Element> h_out(kNumel);
  thrust::fill(h_out.begin(), h_out.end(), 0.);

  thrust::device_vector<Element> d_a = h_a;
  thrust::device_vector<Element> d_b = h_b;
  thrust::device_vector<Element> d_out = h_out;

  jit_add_template(thrust::raw_pointer_cast(d_a.data()),
                   thrust::raw_pointer_cast(d_b.data()),
                   thrust::raw_pointer_cast(d_out.data()), kNumel);
  h_out = d_out;

  // Verify results
  bool all_correct = true;
  float max_diff = 0.0f;
  size_t error_idx = 0;

  // Calculate ground truth on CPU
  thrust::host_vector<Element> h_expected(kNumel);
  for (size_t i = 0; i < kNumel; ++i) {
    h_expected[i] = h_a[i] + h_b[i];
  }

  for (size_t i = 0; i < kNumel; ++i) {
    float diff = std::abs(h_out[i] - h_expected[i]);
    if (diff > max_diff) {
      max_diff = diff;
      error_idx = i;
    }

    if (diff > 1e-5f) {
      all_correct = false;
      if (i < 10) {
        LOG(ERROR) << "Mismatch at index " << i << ": GPU=" << h_out[i]
                   << ", CPU=" << h_expected[i] << ", diff=" << diff;
      }
    }
  }

  LOG(INFO) << "Max difference: " << max_diff << " at index " << error_idx;

  LOG(INFO) << "Sample results (first 10 elements):";
  for (size_t i = 0; i < 10 && i < kNumel; ++i) {
    LOG(INFO) << "Index " << i << ": GPU=" << h_out[i]
              << ", CPU=" << h_expected[i];
  }

  EXPECT_TRUE(all_correct) << "GPU and CPU results do not match!";
}

}  // namespace tilefusion::testing
