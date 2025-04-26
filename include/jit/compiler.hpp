// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda.h>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tilefusion::jit {

/**
 * A simple JIT compiler for CUDA kernels.
 * This class allows for runtime compilation of CUDA kernels using NVCC.
 */
class JitCompiler {
  public:
    // Singleton pattern
    static JitCompiler& instance();

    /**
     * Compiles a CUDA kernel at runtime and returns a function pointer to the
     * kernel.
     *
     * @param kernel_name The name of the kernel function to compile
     * @param cuda_source The CUDA source code containing the kernel
     * @param compile_args Additional compiler arguments to pass to NVCC
     * @return Function pointer to the compiled kernel or nullptr if compilation
     * fails
     */
    CUfunction compile_kernel(
        const std::string& kernel_name, const std::string& cuda_source,
        const std::vector<std::string>& compile_args = {});

    /**
     * Gets a previously compiled kernel or compiles it if it doesn't exist.
     *
     * @param kernel_name The name of the kernel function
     * @param cuda_source The CUDA source code
     * @param compile_args Additional compiler arguments
     * @return Function pointer to the compiled kernel
     */
    CUfunction get_or_compile_kernel(
        const std::string& kernel_name, const std::string& cuda_source,
        const std::vector<std::string>& compile_args = {});

    /**
     * Clears the cache of compiled kernels.
     */
    void clear_cache();

  private:
    JitCompiler();
    ~JitCompiler();

    // Delete copy constructor and assignment operator
    JitCompiler(const JitCompiler&) = delete;
    JitCompiler& operator=(const JitCompiler&) = delete;

    // Compile a CUDA source file to PTX
    std::string compile_to_ptx(const std::string& cuda_source,
                               const std::vector<std::string>& compile_args);

    // Load a PTX string and get a handle to the specified kernel
    CUfunction load_ptx_and_get_kernel(const std::string& ptx,
                                       const std::string& kernel_name);

    // Write a string to a temporary file
    std::string write_to_temp_file(const std::string& content,
                                   const std::string& extension);

    // CUDA context and module
    CUcontext cuda_context_;

    // Cache of compiled kernels
    std::unordered_map<std::string, CUmodule> module_cache_;
    std::unordered_map<std::string, CUfunction> kernel_cache_;

    // Mutex for thread safety
    std::mutex mutex_;
};

}  // namespace tilefusion::jit
