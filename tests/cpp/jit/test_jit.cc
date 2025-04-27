// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/test_utils.hpp"
#include "cuda_utils.hpp"
#include "jit/compiler.hpp"
#include "kernels/common.hpp"

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <sstream>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        CUresult result = call;                                                \
        if (result != CUDA_SUCCESS) {                                          \
            const char* error_string;                                          \
            cuGetErrorString(result, &error_string);                           \
            std::stringstream err;                                             \
            err << "CUDA error: " << error_string << " (" << result << ") at " \
                << __FILE__ << ":" << __LINE__;                                \
            LOG(ERROR) << err.str();                                           \
            throw std::runtime_error(err.str());                               \
        }                                                                      \
    } while (0)

namespace tilefusion::testing {

using namespace tilefusion::jit;
namespace {
float rand_float(float a = 1e-3, float b = 1) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

const std::string kAddKernelSource = R"(
extern "C" __global__ void add_kernel(const float* a, const float* b,
                                      float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
)";
}  // namespace

void jit_add(const float* a, const float* b, float* out, int n) {
    if (n == 0) return;

    auto& jit = JitCompiler::instance();
    CUfunction kernel =
        jit.get_or_compile_kernel("add_kernel", kAddKernelSource);

    if (!kernel) {
        throw std::runtime_error("Failed to compile or retrieve kernel");
    }

    int block_size = 128;
    int grid_size = (n + block_size - 1) / block_size;

    // NOTE: The CUDA driver API expects pointers to pointers for kernel
    // arguments
    void* args[] = {&a, &b, &out, &n};

    CUDA_CHECK(cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0,
                              nullptr, args, nullptr));

    LOG(INFO) << "Kernel launched successfully";
}

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

    jit_add(thrust::raw_pointer_cast(d_a.data()),
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
