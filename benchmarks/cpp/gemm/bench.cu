// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "util.cuh"
#include "util/cuda_info.hpp"

#include <cutlass/half.h>

#include <fstream>
#include <iomanip>

#define CHECK_CORRECTNESS true

//// =============== Test Config=============== ////
static const int kWarpPerRow = 2;
static const int kWarpPerCol = 2;
using WholeShape = GemmShape<4096, 4096, 128>;
using CtaTileShape = GemmShape<64, 128, 128>;
using WarpLayout = tl::RowMajor<kWarpPerRow, kWarpPerCol>;
static constexpr int kRK = 64;

void run_test(std::ofstream& fout) {
    //// =============== Declaration =============== ////
    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;

    using InType = __half;
    using AccType = float;

    using Config = KeGemmTraits<InType, AccType, WholeShape, CtaTileShape, kRK,
                                WarpLayout>;
    auto tilefusion_gemm =
        &gemm<InType, AccType, kM, kN, kK, kTM, kTN, kTK,
              typename Config::GIteratorA, typename Config::SIteratorA,
              typename Config::SharedA, typename Config::RegA,
              typename Config::LoadSharedA, typename Config::LoadRegA,
              typename Config::GIteratorB, typename Config::SIteratorB,
              typename Config::SharedB, typename Config::RegB,
              typename Config::LoadSharedB, typename Config::LoadRegB,
              typename Config::GlobalC, typename Config::SharedC,
              typename Config::Acc, typename Config::AccHalf,
              typename Config::ConvertAcc, typename Config::StoreRegC,
              typename Config::StoreSharedC>;

    using KeTraits = benchmarks::cutlass_wrapper::GemmTraits<
        cutlass::half_t, kWarpPerRow, kWarpPerCol, kM, kN, kK, kTM, kTN, kTK>;
    auto cutlass_gemm =
        &benchmarks::cutlass_wrapper::gemm_kernel<cutlass::half_t, kM, kN, kK,
                                                  kTM, kTN, kTK, KeTraits>;

    static constexpr int inputs = kTK * (kTN + kTM) * sizeof(InType);
    static constexpr int acc = kTM * kTN * sizeof(InType);
    static constexpr int smem_size = inputs > acc ? inputs : acc;

    const int kMaxSmemPerBlock = 48 * 1024;
    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(tilefusion_gemm,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
        cudaFuncSetAttribute(cutlass_gemm,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
    }

    int block_x = benchmarks::CeilDiv<kM, kTM>;
    int block_y = benchmarks::CeilDiv<kN, kTN>;
    dim3 dim_grid(block_x, block_y, 1);
    dim3 dim_block(Config::kThreads, 1, 1);

    std::cout << "Running test:" << std::endl
              << "[M, N, K] = " << kM << ", " << kN << ", " << kK
              << ", [TM, TN, TK] = " << kTM << ", " << kTN << ", " << kTK
              << ", RK = " << kRK << ", WarpLayout = [" << kWarpPerRow << ", "
              << kWarpPerCol << "]" << std::endl
              << "blocks = [" << block_x << ", " << block_y << "]" << std::endl
              << std::endl;

    //// =============== Prepare data =============== ////
    // input matrix A
    thrust::host_vector<cutlass::half_t> h_a(kM * kK);
    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<cutlass::half_t>(rand_float());
    thrust::device_vector<cutlass::half_t> d_a = h_a;
    const cutlass::half_t* dA = thrust::raw_pointer_cast(d_a.data());
    const __half* dA2 = reinterpret_cast<const __half*>(dA);

    // input matrix B
    thrust::host_vector<cutlass::half_t> h_b(kK * kN);
    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<cutlass::half_t>(rand_float());
    thrust::device_vector<cutlass::half_t> d_b = h_b;
    const cutlass::half_t* dB = thrust::raw_pointer_cast(d_b.data());
    const __half* dB2 = reinterpret_cast<const __half*>(dB);

    // output matrix C for cutlass GEMM kernel
    thrust::device_vector<cutlass::half_t> d_c(kM * kN);
    cutlass::half_t* dC = thrust::raw_pointer_cast(d_c.data());
    thrust::device_vector<InType> d_c2(kM * kN);
    InType* dC2 = thrust::raw_pointer_cast(d_c2.data());

    // output matrix C for cublas gemm
    thrust::device_vector<__half> d_c3(kM * kN);
    __half* dC3 = thrust::raw_pointer_cast(d_c3.data());

    thrust::host_vector<cutlass::half_t> h_c;
    thrust::host_vector<InType> h_c2;
    thrust::host_vector<__half> h_c3;

//// =============== check correctness =============== ////
#ifdef CHECK_CORRECTNESS
    thrust::fill(d_c.begin(), d_c.end(), static_cast<cutlass::half_t>(0.));
    thrust::fill(d_c2.begin(), d_c2.end(), static_cast<InType>(0.));
    thrust::fill(d_c3.begin(), d_c3.end(), static_cast<__half>(0.));

    cutlass_gemm<<<dim_grid, dim_block, smem_size>>>(dA, dB, dC);
    cudaDeviceSynchronize();
    h_c = d_c;

    tilefusion_gemm<<<dim_grid, dim_block, smem_size>>>(dA2, dB2, dC2);
    cudaDeviceSynchronize();
    h_c2 = d_c2;

    // cublas
    cublas_hgemm(kM, kN, kK, dA2, dB2, dC3, false /*timeit*/);
    h_c3 = d_c3;

    bool passed1 = check_results(
        thrust::raw_pointer_cast(h_c2.data()), /*tilefusion */
        thrust::raw_pointer_cast(h_c.data()), /*cutlass */ kM * kN);

    bool passed2 = check_results(
        thrust::raw_pointer_cast(h_c3.data()), /*cublas */
        thrust::raw_pointer_cast(h_c.data()), /*cutlass */ kM * kN);

    if (!(passed1 && passed2)) {
        std::cerr << "Test failed" << std::endl;
        return;
    }
    std::cout << "Test passed" << std::endl;
#endif

    //// =============== Timing =============== ////
    thrust::fill(d_c.begin(), d_c.end(), static_cast<cutlass::half_t>(0.));
    thrust::fill(d_c2.begin(), d_c2.end(), static_cast<InType>(0.));
    thrust::fill(d_c3.begin(), d_c3.end(), static_cast<__half>(0.));

    float cublas_time = cublas_hgemm(kM, kN, kK, dA2, dB2, dC3, true);
    h_c3 = d_c3;

    const int warm_up = 10;
    const int iters = 50;
    for (int i = 0; i < warm_up; ++i) {
        cutlass_gemm<<<dim_grid, dim_block, smem_size>>>(dA, dB, dC);
        tilefusion_gemm<<<dim_grid, dim_block, smem_size>>>(dA2, dB2, dC2);
    }
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        cutlass_gemm<<<dim_grid, dim_block, smem_size>>>(dA, dB, dC);
    }
    cudaDeviceSynchronize();
    float cutlass_time = timer.stop() / iters;

    timer.start();
    for (int i = 0; i < iters; ++i) {
        tilefusion_gemm<<<dim_grid, dim_block, smem_size>>>(dA2, dB2, dC2);
    }
    cudaDeviceSynchronize();
    float tilefusion_time = timer.stop() / iters;

    float base = cublas_time;

    fout << "[" << kM << ", " << kN << ", " << kK << "]\t[" << kTM << ", "
         << kTN << ", " << kTK << "]\t" << kRK << "\t[" << kWarpPerRow << ", "
         << kWarpPerCol << "]\t" << cublas_time << "\t" << cutlass_time << "("
         << std::setprecision(2) << cutlass_time / base << ")"
         << "\t" << std::setprecision(6) << tilefusion_time << " ("
         << std::setprecision(2) << tilefusion_time / base << ")" << std::endl;
}

int main() {
    std::ofstream fout;
    fout.setf(std::ios::fixed);
    fout.precision(6);

    auto dev_name = tilefusion::get_device_name();
    std::stringstream file_name;
    file_name << "figures/bench_" << dev_name << "_gemm.tsv";
    fout.open(file_name.str(), std::ios::out);

    fout << "[M, N, K]\t[kTM, kTN, kTK]\tkRK\tWarp Layout\t"
            "cuBLAS(ms)\tcutlass(ms)\ttilefusion(ms)"
         << std::endl;

    run_test(fout);
    return 0;
}
