// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_info.hpp"
#include "jit/mod.hpp"
#include "kernels/common.hpp"
#include "kernels/ops.hpp"
#include "types/layout.hpp"

namespace tilefusion::kernels {

using namespace tilefusion;
namespace tl = tile_layout;

namespace {
std::string generate_kernel_wrapper(
    const std::string& kernel_name, const std::string& in_type,
    const std::string& acc_type, const std::string& out_type,  //
    int64_t length_q, int64_t length_kv, int64_t hidden_qk, int64_t hidden_v,
    int64_t tile_length_q, int64_t tile_length_kv,  //
    int64_t tile_hidden_qk, int64_t tile_hidden_v,  //
    int64_t warp_rows, int64_t warp_cols,           //
    double softmax_scale, bool causal) {
    std::stringstream ss;

    ss << R"(
#include "kernels/flash_attention_device.cuh"

using namespace tilefusion::kernels;
)";

    ss << "\n// Layout and shape definitions\n";
    ss << "using WarpLayout = tl::RowMajor<" << warp_rows << ", " << warp_cols
       << ">;\n";
    ss << "using WholeShape = TileShape<" << length_q << ", " << length_kv
       << ", " << hidden_qk << ", " << hidden_v << ">;\n";
    ss << "using CtaTileShape = TileShape<" << tile_length_q << ", "
       << tile_length_kv << ", " << tile_hidden_qk << ", " << tile_hidden_v
       << ">;\n\n";

    ss << "// Flash attention configuration\n";
    ss << "using Config = FlashAttentionTraits<" << in_type << ", " << acc_type
       << ", " << out_type << ", WholeShape, CtaTileShape, WarpLayout, "
       << softmax_scale << ", " << causal << ">;\n\n";

    ss << "// Kernel function\n";
    ss << "extern \"C\" __global__ void " << kernel_name << "(const " << in_type
       << "* Q, const " << in_type << "* K, const " << in_type << "* V, "
       << out_type << "* O) {\n";
    ss << "    ke_flash_attention<" << in_type << ", " << acc_type << ", "
       << out_type << ", Config>(Q, K, V, O);\n";
    ss << "}\n";

    return ss.str();
}
}  // namespace

void flash_attention(const torch::Tensor& Q, const torch::Tensor& K,
                     const torch::Tensor& V, torch::Tensor& O,
                     int64_t tile_length_q, int64_t tile_length_kv,
                     int64_t tile_hidden_qk, int64_t tile_hidden_v,
                     double softmax_scale, bool causal) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    CHECK_INPUT(O);

    const at::ScalarType dtype = Q.scalar_type();
    TORCH_CHECK(dtype == at::ScalarType::Half && K.scalar_type() == dtype &&
                    V.scalar_type() == dtype && O.scalar_type() == dtype,
                "the inputs and output must be half-precision (fp16).");

    const int64_t length_q = Q.size(0);
    const int64_t length_kv = K.size(1);
    const int64_t hidden_qk = Q.size(1);
    const int64_t hidden_v = V.size(0);

    using InType = __half;
    using AccType = float;
    using OutType = __half;
    using WarpLayout = tl::RowMajor<4, 1>;

    std::string in_type = jit::get_type_string<InType>();
    std::string acc_type = jit::get_type_string<AccType>();
    std::string out_type = jit::get_type_string<OutType>();

    std::stringstream kernel_name_ss;
    kernel_name_ss << "flash_attention_kernel_" << in_type << "_" << acc_type
                   << "_" << out_type << "_" << length_q << "_" << length_kv
                   << "_" << hidden_qk << "_" << hidden_v << "_"
                   << tile_length_q << "_" << tile_length_kv << "_"
                   << tile_hidden_qk << "_" << tile_hidden_v;
    std::string kernel_name = kernel_name_ss.str();

    std::string kernel_wrapper_src = generate_kernel_wrapper(
        kernel_name, in_type, acc_type, out_type, length_q, length_kv,
        hidden_qk, hidden_v, tile_length_q, tile_length_kv, tile_hidden_qk,
        tile_hidden_v, tl::num_rows<WarpLayout>, tl::num_cols<WarpLayout>,
        softmax_scale, causal);

    auto& jit = jit::JitCompiler::instance();

    auto include_paths = jit::get_default_include_paths();
    auto compile_args = jit::get_default_compile_args();
    CUfunction kernel = jit.get_or_compile_kernel(
        kernel_name, kernel_wrapper_src, include_paths, compile_args);
    if (!kernel) {
        throw std::runtime_error("Failed to compile or retrieve kernel");
    }

    const InType* dQ = reinterpret_cast<const InType*>(Q.data_ptr());
    const InType* dK = reinterpret_cast<const InType*>(K.data_ptr());
    const InType* dV = reinterpret_cast<const InType*>(V.data_ptr());
    OutType* dO = reinterpret_cast<OutType*>(O.data_ptr());

    void* args[] = {(void*)&dQ, (void*)&dK, (void*)&dV, (void*)&dO};

    int shm_input =
        (tile_length_q * tile_hidden_qk + tile_hidden_qk * tile_length_kv +
         tile_length_kv * tile_hidden_v);
    int shm_output = tile_length_q * tile_hidden_v;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(OutType)
                                          : shm_input * sizeof(InType);

    int block_x = ceil_div(length_q, tile_length_q);
    int block_y = ceil_div(hidden_v, tile_hidden_v);
    int batch_size = 1;  // FIXME(ying): batch size is hardcoded to 1 for now

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;

    dim3 grid(block_x, block_y, batch_size);
    dim3 block(kThreads, 1, 1);

    if (shm_size > GetMaxSharedMemoryPerBlock()) {
        // Set shared memory size if it exceeds the device limit
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shm_size));
    }

    CUDA_DRIVER_CHECK(cuLaunchKernel(
        kernel, grid.x, grid.y, grid.z,  // grid dimensions
        block.x, block.y, block.z,       // block dimensions
        shm_size,                        // shared memory size
        0,                               // stream
        args,                            // kernel arguments
        nullptr                          // extra parameters
        ));

    cudaDeviceSynchronize();

    LOG(INFO) << "flash_attention kernel launched successfully";
}

}  // namespace tilefusion::kernels
