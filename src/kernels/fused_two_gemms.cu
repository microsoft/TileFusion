// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cell/mod.hpp"
#include "jit/mod.hpp"
#include "kernels/common.hpp"
#include "kernels/ops.hpp"

namespace tilefusion::kernels {

using namespace tilefusion;
namespace tl = tile_layout;

namespace {
std::string generate_kernel_wrapper(const std::string& in_type,
                                    const std::string& acc_type, int m, int n,
                                    int k, int p, int kTM, int kTN, int kTK,
                                    int kTP) {
    std::stringstream ss;
    ss << R"(
#include "kernels/fused_two_gemms_device.cuh"

using namespace tilefusion::kernels;
using Config = FusedTwoGemmsTraits<)"
       << in_type << ", " << acc_type << R"(,
        tl::RowMajor<2, 1>, )"
       << m << ", " << n << ", " << k << ", " << p << "," << kTM << ", " << kTN
       << ", " << kTK << ", " << kTP << R"(>;

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
                     const torch::Tensor& C, torch::Tensor& D) {
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

    // TODO(ying): fix this hard-coded shared memory tile sizes.
    static constexpr int kTM = 64;
    static constexpr int kTN = 64;
    static constexpr int kTK = 64;
    static constexpr int kTP = 64;

    using WarpLayout = tl::RowMajor<2, 1>;
    using InType = __half;
    using AccType = float;

    static constexpr int kShmInput = (kTM * kTK + kTK * kTN + kTN * kTP);
    static constexpr int kShmOutput = kTM * kTP;
    static constexpr int kShmSize = kShmInput < kShmOutput
                                        ? kShmOutput * sizeof(InType)
                                        : kShmInput * sizeof(InType);

    std::string in_type = jit::get_type_string<InType>();
    std::string acc_type = jit::get_type_string<AccType>();

    std::string kernel_wrapper = generate_kernel_wrapper(
        in_type, acc_type, m, n, k, p, kTM, kTN, kTK, kTP);

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

    int block_x = ceil_div(m, kTM);
    int block_y = ceil_div(p, kTP);
    int block_z = 1;
    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;

    if (kShmSize > 48 * 1024) {
        // Set shared memory size if it exceeds the default limit
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kShmSize));
    }

    CUDA_DRIVER_CHECK(cuLaunchKernel(kernel, block_x, block_y, block_z,  // grid
                                     kThreads, 1, 1,  // block
                                     kShmSize,        // shared memory bytes
                                     nullptr,         // stream
                                     args,            // kernel parameters
                                     nullptr));       // extra parameters

    LOG(INFO) << "Fused two gemms kernel launched successfully";
}
}  // namespace tilefusion::kernels
