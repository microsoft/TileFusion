// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cell/mod.hpp"
#include "cuda_info.hpp"
#include "jit/mod.hpp"
#include "kernels/common.hpp"
#include "kernels/ops.hpp"

namespace tilefusion::kernels {

using namespace tilefusion;
namespace tl = tile_layout;

namespace {

/// @brief Generate the kernel wrapper for the fused two gemms kernel.
///        D = A @ B @ C
///        where A has a shape of (m, k) and laid out in row-major fashion,
///        B has a shape of (k, n) and laid out in column-major fashion,
///        C has a shape of (n, p) and laid out in column-major fashion,
///        D has a shape of (m, p) and laid out in row-major fashion.
/// @param in_type The input type.
/// @param acc_type The accumulation type.
/// @param m The number of rows of the input matrix A.
/// @param n The number of rows of the input matrix B.
/// @param k The number of columns of the input matrices A and B.
/// @param p The number of columns of the input matrix C.
/// @param tm The shared memory tile size of the m-dimension.
/// @param tn The shared memory tile size of the n-dimension.
/// @param tk The shared memory tile size of the k-dimension.
/// @param tp The shared memory tile size of the p-dimension.
/// @return The kernel wrapper for the fused two gemms kernel.
std::string generate_kernel_wrapper(const std::string& in_type,
                                    const std::string& acc_type, int64_t m,
                                    int64_t n, int64_t k, int64_t p,
                                    int64_t tm = 64, int64_t tn = 64,
                                    int64_t tk = 64, int64_t tp = 64) {
    std::stringstream ss;
    ss << R"(
#include "kernels/fused_two_gemms_device.cuh"

using namespace tilefusion::kernels;
using Config = FusedTwoGemmsTraits<)"
       << in_type << ", " << acc_type << R"(,
        tl::RowMajor<2, 1>, )"
       << m << ", " << n << ", " << k << ", " << p << ", " << tm << ", " << tn
       << ", " << tk << ", " << tp << R"(>;

extern "C" __global__ void fused_two_gemms_kernel_)"
       << in_type << "_" << acc_type << "_" << m << "_" << n << "_" << k << "_"
       << p << R"((const )" << in_type << R"(* A, const )" << in_type
       << R"(* B, const )" << in_type << R"(* C, )" << in_type << R"(* D) {
    ke_fused_two_gemms<Config::InType, Config::AccType, Config>(A, B, C, D);
})";
    return ss.str();
}
}  // namespace

void fused_two_gemms(const torch::Tensor& A, const torch::Tensor& B,
                     const torch::Tensor& C, torch::Tensor& D, int64_t tm,
                     int64_t tn, int64_t tk, int64_t tp) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);
    CHECK_INPUT(D);

    const at::ScalarType dtype = A.scalar_type();
    TORCH_CHECK(dtype == at::ScalarType::Half && B.scalar_type() == dtype &&
                    C.scalar_type() == dtype && D.scalar_type() == dtype,
                "the inputs and output must be half-precision (fp16).");

    const int64_t m = A.size(0);
    const int64_t n = B.size(0);
    const int64_t k = B.size(1);
    const int64_t p = C.size(0);

    // TODO(ying): warp layout should be a configurable parameter
    using WarpLayout = tl::RowMajor<2, 1>;
    using InType = __half;
    using AccType = float;

    // calculate shared memory usage
    int shm_input = (tm * tk + tk * tn + tn * tp);
    int shm_output = tm * tp;
    const int shm_size = shm_input < shm_output ? shm_output * sizeof(InType)
                                                : shm_input * sizeof(InType);

    std::string in_type = jit::get_type_string<InType>();
    std::string acc_type = jit::get_type_string<AccType>();

    std::string kernel_wrapper =
        generate_kernel_wrapper(in_type, acc_type, m, n, k, p, tm, tn, tk, tp);

    std::string kernel_name = "fused_two_gemms_kernel_" + in_type + "_" +
                              acc_type + "_" + std::to_string(m) + "_" +
                              std::to_string(n) + "_" + std::to_string(k) +
                              "_" + std::to_string(p);

    auto& jit = jit::JitCompiler::instance();

    auto include_paths = jit::get_default_include_paths();
    auto compile_args = jit::get_default_compile_args();
    CUfunction kernel = jit.get_or_compile_kernel(kernel_name, kernel_wrapper,
                                                  include_paths, compile_args);

    if (!kernel) {
        throw std::runtime_error("Failed to compile or retrieve kernel");
    }

    const InType* A_ptr =
        reinterpret_cast<const InType*>(A.data_ptr<at::Half>());
    const InType* B_ptr =
        reinterpret_cast<const InType*>(B.data_ptr<at::Half>());
    const InType* C_ptr =
        reinterpret_cast<const InType*>(C.data_ptr<at::Half>());
    InType* D_ptr = reinterpret_cast<InType*>(D.data_ptr<at::Half>());

    void* args[] = {(void*)&A_ptr, (void*)&B_ptr, (void*)&C_ptr, (void*)&D_ptr,
                    (void*)&m,     (void*)&n,     (void*)&k,     (void*)&p};

    int block_x = ceil_div(m, tm);
    int block_y = ceil_div(p, tp);
    int block_z = 1;
    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;

    if (shm_size > GetMaxSharedMemoryPerBlock()) {
        // Set shared memory size if it exceeds the device limit
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shm_size));
    }

    CUDA_DRIVER_CHECK(cuLaunchKernel(kernel, block_x, block_y, block_z,  // grid
                                     kThreads, 1, 1,  // block
                                     shm_size,        // shared memory bytes
                                     nullptr,         // stream
                                     args,            // kernel parameters
                                     nullptr));       // extra parameters

    LOG(INFO) << "Fused two gemms kernel launched successfully";
}
}  // namespace tilefusion::kernels
