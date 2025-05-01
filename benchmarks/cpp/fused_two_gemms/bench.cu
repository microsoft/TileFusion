// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cutlass_fused_two_gemms.cuh"
#include "kernels/fused_two_gemms_device.cuh"
#include "util.cuh"

using namespace tilefusion::kernels;

// kernel wrapper
template <typename InType, typename AccType, typename Config>
__attribute__((global)) void kernel_wrapper(const InType* A, const InType* B,
                                            const InType* C, InType* D) {
    ke_fused_two_gemms<InType, AccType, Config>(A, B, C, D);
}

template <typename WholeShape, typename CtaTileShape, typename WarpLayout,
          const int kBatch, const int kSharedAccess>
void run(float epsilon = 1e-3) {
    using InType = __half;
    using AccType = float;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;
    static constexpr int kP = dim_size<3, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;
    static constexpr int kTP = dim_size<3, CtaTileShape>;

    static_assert(kK == kTK, "The current implementation requires kTK == K.");
    static_assert(kP == kTP, "The current implementation requires kTP == P.");

    static constexpr int kWarpPerRow = tl::num_rows<WarpLayout>;
    static constexpr int kWarpPerCol = tl::num_cols<WarpLayout>;

    thrust::host_vector<cutlass::half_t> h_a(kM * kK * kBatch);

    for (int i = 0; i < h_a.size(); ++i) {
        h_a[i] = static_cast<cutlass::half_t>(rand_float());
    }

    thrust::host_vector<cutlass::half_t> h_b(kK * kN * kBatch);
    for (int i = 0; i < h_b.size(); ++i) {
        h_b[i] = static_cast<cutlass::half_t>(rand_float());
    }

    thrust::host_vector<cutlass::half_t> h_c(kN * kP * kBatch);
    for (int i = 0; i < h_c.size(); ++i) {
        h_c[i] = static_cast<cutlass::half_t>(rand_float());
    }

    thrust::host_vector<InType> h_d(kM * kP * kBatch);
    thrust::fill(h_d.begin(), h_d.end(), 0.);

    thrust::host_vector<cutlass::half_t> h_d2(kM * kP * kBatch);
    thrust::fill(h_d2.begin(), h_d2.end(), 0.);

    thrust::host_vector<__half> h_d3(kM * kP * kBatch);
    thrust::fill(h_d3.begin(), h_d3.end(), 0.);

    thrust::device_vector<cutlass::half_t> d_a = h_a;
    thrust::device_vector<cutlass::half_t> d_b = h_b;
    thrust::device_vector<cutlass::half_t> d_c = h_c;
    thrust::device_vector<InType> d_d = h_d;
    thrust::device_vector<cutlass::half_t> d_d2 = h_d2;
    thrust::device_vector<__half> d_d3 = h_d3;

    const cutlass::half_t* CA = thrust::raw_pointer_cast(d_a.data());
    const cutlass::half_t* CB = thrust::raw_pointer_cast(d_b.data());
    const cutlass::half_t* CC = thrust::raw_pointer_cast(d_c.data());
    cutlass::half_t* CD = thrust::raw_pointer_cast(d_d2.data());

    const InType* A = reinterpret_cast<const InType*>(CA);
    const InType* B = reinterpret_cast<const InType*>(CB);
    const InType* C = reinterpret_cast<const InType*>(CC);
    InType* D = thrust::raw_pointer_cast(d_d.data());

    using Config = FusedTwoGemmsTraits<InType, AccType, WarpLayout, kM, kN, kK,
                                       kP, kTM, kTN, kTK, kTP>;

    int block_x = CeilDiv<kM, kTM>;
    int block_y = CeilDiv<kP, kTP>;
    int block_z = kBatch;
    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;

    dim3 grid(block_x, block_y, block_z);
    dim3 block(kThreads, 1, 1);

    static constexpr int kShmInput = (kTM * kTK + kTK * kTN + kTN * kTP);
    static constexpr int kShmOutput = kTM * kTP;
    static constexpr int kSharedSize = kShmInput < kShmOutput
                                           ? kShmOutput * sizeof(InType)
                                           : kShmInput * sizeof(InType);

    auto ke_tilefusion = &kernel_wrapper<InType, AccType, Config>;

    auto cutlass_fused_gemm =
        &cute_fused_gemm<cutlass::half_t, kWarpPerRow, kWarpPerCol, kM, kN, kK,
                         kP, kTM, kTN, kTK, kTP>;

    if (kSharedSize > 48 * 1024) {
        cudaFuncSetAttribute(ke_tilefusion,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kSharedSize);
    }

    ke_tilefusion<<<grid, block, kSharedSize, 0>>>(A, B, C, D);
    cudaDeviceSynchronize();

    h_d = d_d;

    cutlass_fused_gemm(CA, CB, CC, CD, false, 0, 0);
    h_d2 = d_d2;

    thrust::host_vector<InType> h_acc(kM * kN * kBatch);
    thrust::fill(h_acc.begin(), h_acc.end(), 0.);
    thrust::device_vector<InType> d_acc = h_acc;

    cublas_two_gemms(kM, kN, kK, kP, kBatch, A, B, C,
                     thrust::raw_pointer_cast(d_d3.data()),
                     thrust::raw_pointer_cast(d_acc.data()), false);
    cudaDeviceSynchronize();
    h_acc = d_acc;
    h_d3 = d_d3;

#ifdef DEBUG
    InType* data = thrust::raw_pointer_cast(h_d.data());
    cutlass::half_t* cutlass_data = thrust::raw_pointer_cast(h_d2.data());
    __half* cutlass_data_half = reinterpret_cast<__half*>(cutlass_data);
    __half* ground_truth = thrust::raw_pointer_cast(h_d3.data());

    const int numel = 256;
    printf("ours:\n");
    for (int i = 0; i < numel; ++i) {
        printf("%.3f, ", __half2float(data[i]));
        if (i && (i + 1) % 16 == 0) printf("\n");
    }
    printf("cutlass:\n");
    for (int i = 0; i < numel; ++i) {
        printf("%.3f, ", __half2float(cutlass_data_half[i]));
        if (i && (i + 1) % 16 == 0) printf("\n");
    }
    printf("\nground_truth:\n");
    for (int i = 0; i < numel; ++i) {
        printf("%.3f, ", __half2float(ground_truth[i]));
        if (i && (i + 1) % 16 == 0) printf("\n");
    }

    bool passed1 = check_results(data, ground_truth, kM * kP, epsilon);
    bool passed2 =
        check_results(cutlass_data_half, ground_truth, kM * kP, epsilon);
    std::cout << "passed1: " << passed1 << ", passed2: " << passed2
              << std::endl;

    if (passed1 && passed2) {
        std::cout << "[" << kM << ", " << kN << ", " << kK << ", " << kP
                  << "], batch = " << kBatch << ", passed." << std::endl;
    } else {
        std::cout << "[" << kM << ", " << kN << ", " << kK << ", " << kP
                  << "], batch = " << kBatch << ", failed." << std::endl;
    }

#endif

    CudaTimer timer;
    const int warm_up = 10;
    const int iters = 50;

    for (int i = 0; i < warm_up; ++i) {
        ke_tilefusion<<<grid, block, kSharedSize, 0>>>(A, B, C, D);
    }
    cudaDeviceSynchronize();

    timer.start();
    for (int i = 0; i < iters; ++i) {
        ke_tilefusion<<<grid, block, kSharedSize, 0>>>(A, B, C, D);
    }
    cudaDeviceSynchronize();
    float tilefusion_time = timer.stop() / iters;

    float cutlass_time =
        cutlass_fused_gemm(CA, CB, CC, CD, true, warm_up, iters);

    float cublas_time = cublas_two_gemms(
        kM, kN, kK, kP, kBatch, A, B, C, thrust::raw_pointer_cast(d_d3.data()),
        thrust::raw_pointer_cast(d_acc.data()), true);

    std::cout << "[" << kM << ", " << kN << ", " << kK << ", " << kP << "]\t["
              << kTM << ", " << kTN << ", " << kTK << ", " << kTP << "]\t"
              << std::fixed << std::setprecision(4) << cublas_time << "\t"
              << cutlass_time << "(" << cutlass_time / cublas_time << ")"
              << "\t" << tilefusion_time << "(" << tilefusion_time / cublas_time
              << ")" << std::endl;
}

int main() {
    using WarpLayout = tl::RowMajor<4, 1>;
    static constexpr int kSharedAccess = 128;

    run<B2BGemmShape<4096 /*M*/, 1024 /*N*/, 128 /*K*/, 128 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout, 1, kSharedAccess>(5e-3);

    run<B2BGemmShape<4096 /*M*/, 2048 /*N*/, 128 /*K*/, 128 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout, 1, kSharedAccess>(5e-3);

    run<B2BGemmShape<8192 /*M*/, 1024 /*N*/, 128 /*K*/, 128 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout, 1, kSharedAccess>(5e-3);

    run<B2BGemmShape<8192 /*M*/, 2048 /*N*/, 128 /*K*/, 128 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout, 1, kSharedAccess>(5e-3);

    run<B2BGemmShape<4096 /*M*/, 4096 /*N*/, 128 /*K*/, 128 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout, 1, kSharedAccess>(5e-3);

    run<B2BGemmShape<4096 /*M*/, 2048 /*N*/, 128 /*K*/, 128 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout, 1, kSharedAccess>(5e-3);

    run<B2BGemmShape<8192 /*M*/, 8192 /*N*/, 128 /*K*/, 128 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout, 1, kSharedAccess>(5e-3);

    run<B2BGemmShape<8192 /*M*/, 4096 /*N*/, 128 /*K*/, 128 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout, 1, kSharedAccess>(5e-3);

    run<B2BGemmShape<2048 /*M*/, 2048 /*N*/, 128 /*K*/, 128 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout, 1, kSharedAccess>(5e-3);

    run<B2BGemmShape<1024 /*M*/, 1024 /*N*/, 128 /*K*/, 128 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout, 1, kSharedAccess>(5e-3);

    run<B2BGemmShape<512 /*M*/, 512 /*N*/, 128 /*K*/, 128 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        WarpLayout, 1, kSharedAccess>(5e-3);

    return 0;
}
