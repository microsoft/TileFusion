// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cell/mod.hpp"
#include "cuda_info.hpp"
#include "jit/mod.hpp"
#include "kernels/common.hpp"
#include "kernels/ops.hpp"
#include "types/mod.hpp"

using namespace tilefusion;
using namespace cell;
using namespace copy;
using namespace compute;
namespace tl = tile_layout;

namespace tilefusion::kernels {

namespace {

std::string generate_gemm_kernel_wrapper(
    const std::string& in_type, const std::string& acc_type, int64_t m,
    int64_t n, int64_t k, int64_t tm, int64_t tn, int64_t tk,
    int64_t num_stages, int64_t pipeline_level, int64_t warp_row,
    int64_t warp_col, int64_t swizzle_bytes) {
    int64_t kRK = 16;
    std::stringstream ss;
    ss << R"(
#include "kernels/gemm_device.cuh"

using namespace tilefusion::kernels;
using Config = KeGemmTraits<)"
       << in_type << ", " << acc_type << R"(,
        tl::RowMajor<)"
       << warp_row << ", " << warp_col << R"(>, )" << m << ", " << n << ", "
       << k << ", " << tm << ", " << tn << ", " << tk << ", " << kRK << ", "
       << num_stages << ", " << swizzle_bytes << R"(>;

extern "C" __global__ void gemm_kernel_)"
       << in_type << "_" << acc_type << "_" << m << "_" << n << "_" << k << "_"
       << tm << "_" << tn << "_" << tk << "_" << num_stages << "_"
       << pipeline_level << "_" << warp_row << "_" << warp_col << "_"
       << swizzle_bytes << R"((const )" << in_type << R"(* A, const )"
       << in_type << R"(* B, )" << acc_type << R"(* C) {)";
    ss << std::endl;
    if (pipeline_level == 0) {
        ss << "ke_gemm<Config::InType, Config::AccType, Config>(A, B, C);";
    } else if (pipeline_level == 1) {
        ss << "ke_gemm_level1_pipeline<Config::InType, Config::AccType, "
              "Config>(A, B, C);";
    } else if (pipeline_level == 2) {
        ss << "ke_gemm_level2_pipeline<Config::InType, Config::AccType, "
              "Config>(A, B, C);";
    }

    ss << std::endl << "}";

    return ss.str();
}

}  // namespace

void gemm(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C,
          int64_t tm, int64_t tn, int64_t tk, int64_t num_stages,
          int64_t pipeline_level, const torch::Tensor& warp_layout,
          int64_t swizzle_bytes) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int64_t warp_row = warp_layout[0].item<int64_t>();
    const int64_t warp_col = warp_layout[1].item<int64_t>();

    const at::ScalarType dtype = A.scalar_type();
    TORCH_CHECK(dtype == at::ScalarType::Half && B.scalar_type() == dtype,
                "the inputs must be half-precision (fp16).");

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    const int64_t n = B.size(1);

    // using WarpLayout = tl::RowMajor<warp_row, warp_col>;
    using InType = __half;
    using AccType = float;

    int shm_input = (tm * tk + tk * tn) * num_stages;
    int shm_output = tm * tn;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(InType)
                                          : shm_input * sizeof(InType);

    std::string in_type = jit::get_type_string<InType>();
    std::string acc_type = jit::get_type_string<AccType>();

    std::string kernel_wrapper = generate_gemm_kernel_wrapper(
        in_type, acc_type, m, n, k, tm, tn, tk, num_stages, pipeline_level,
        warp_row, warp_col, swizzle_bytes);

    std::string kernel_name =
        "gemm_kernel_" + in_type + "_" + acc_type + "_" + std::to_string(m) +
        "_" + std::to_string(n) + "_" + std::to_string(k) + "_" +
        std::to_string(tm) + "_" + std::to_string(tn) + "_" +
        std::to_string(tk) + "_" + std::to_string(num_stages) + "_" +
        std::to_string(pipeline_level) + "_" + std::to_string(warp_row) + "_" +
        std::to_string(warp_col) + "_" + std::to_string(swizzle_bytes);

    auto& jit = jit::JitCompiler::instance();
    auto include_paths = jit::get_default_include_paths();
    auto compile_args = jit::get_default_compile_args();
    CUfunction kernel = jit.get_or_compile_kernel(kernel_name, kernel_wrapper,
                                                  include_paths, compile_args);

    if (!kernel) {
        throw std::runtime_error("Failed to compile or retrieve kernel");
    }

    const InType* a_ptr = reinterpret_cast<const InType*>(A.data_ptr());
    const InType* b_ptr = reinterpret_cast<const InType*>(B.data_ptr());
    AccType* c_ptr = reinterpret_cast<AccType*>(C.data_ptr());

    void* args[] = {(void*)&a_ptr,
                    (void*)&b_ptr,
                    (void*)&c_ptr,
                    (void*)&m,
                    (void*)&n,
                    (void*)&k,
                    (void*)&tm,
                    (void*)&tn,
                    (void*)&tk,
                    (void*)&num_stages,
                    (void*)&pipeline_level,
                    (void*)&warp_row,
                    (void*)&warp_col,
                    (void*)&swizzle_bytes};

    int block_x = ceil_div(m, tm);
    int block_y = ceil_div(n, tn);
    int block_z = 1;
    int kThreads = warp_row * warp_col * 32;

    if (shm_size > GetMaxSharedMemoryPerBlock()) {
        // Set shared memory size if it exceeds the device limit
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shm_size));
    }

    CUDA_DRIVER_CHECK(cuLaunchKernel(kernel, block_x, block_y, block_z,  // grid
                                     kThreads, 1, 1,  // block
                                     shm_size,        // shared memory bytes
                                     nullptr,         // stream
                                     args,            // arguments
                                     nullptr));       // extra parameters
    LOG(INFO) << "gemm kernel launched successfully";
}

}  // namespace tilefusion::kernels
