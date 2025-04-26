// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kernels/jit_example.hpp"

#include "cuda_utils.hpp"
#include "jit/compiler.hpp"
#include "kernel_registry.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>

namespace tilefusion::kernels {

namespace {
// The CUDA kernel source code for element-wise addition
const std::string kAddKernelSource = R"(
extern "C" __global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
)";
}  // namespace

void jit_add(at::Tensor input_a, at::Tensor input_b, at::Tensor output) {
    // Validate input
    if (!input_a.is_cuda() || !input_b.is_cuda() || !output.is_cuda()) {
        throw std::runtime_error("All tensors must be CUDA tensors");
    }

    if (input_a.sizes() != input_b.sizes() ||
        input_a.sizes() != output.sizes()) {
        throw std::runtime_error("All tensors must have the same shape");
    }

    if (input_a.dtype() != torch::kFloat32 ||
        input_b.dtype() != torch::kFloat32 ||
        output.dtype() != torch::kFloat32) {
        throw std::runtime_error("All tensors must be of type float32");
    }

    // Get total number of elements
    int n = input_a.numel();
    if (n == 0) return;

    // Compile or retrieve the kernel
    auto& jit = tilefusion::jit::JitCompiler::instance();
    CUfunction kernel =
        jit.get_or_compile_kernel("add_kernel", kAddKernelSource);

    if (!kernel) {
        throw std::runtime_error("Failed to compile or retrieve kernel");
    }

    // Set up grid and block dimensions
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Store pointers in variables first
    float* a_ptr = input_a.data_ptr<float>();
    float* b_ptr = input_b.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    // Set up kernel parameters using the variables
    void* args[] = {&a_ptr, &b_ptr, &out_ptr, &n};

    // Launch the kernel
    CUresult result = cuLaunchKernel(kernel, gridSize, 1, 1,  // Grid dimensions
                                     blockSize, 1, 1,  // Block dimensions
                                     0,                // Shared memory size
                                     nullptr,          // Stream
                                     args,             // Arguments
                                     nullptr           // Extra
    );

    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to launch kernel");
    }

    // Synchronize to make sure the kernel is completed
    cudaDeviceSynchronize();
}

// Register the kernel with TileFusion's kernel registry
REGISTER_OP(jit_add, "jit_add(Tensor a, Tensor b, Tensor(a!) output) -> ()",
            &jit_add);

}  // namespace tilefusion::kernels
