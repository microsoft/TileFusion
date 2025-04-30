// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kernels/fused_two_gemms_device.cuh"
#include "util.hpp"

using namespace tilefusion::kernels;
namespace tl = tilefusion::tile_layout;

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

    thrust::host_vector<InType> h_a(kM * kK * kBatch);

    for (int i = 0; i < h_a.size(); ++i) {
        h_a[i] = static_cast<InType>(rand_float());
    }

    thrust::host_vector<InType> h_b(kK * kN * kBatch);
    for (int i = 0; i < h_b.size(); ++i) {
        h_b[i] = static_cast<InType>(rand_float());
    }

    thrust::host_vector<InType> h_c(kN * kP * kBatch);
    for (int i = 0; i < h_c.size(); ++i) {
        h_c[i] = static_cast<InType>(rand_float());
    }

    thrust::host_vector<InType> h_d(kM * kP * kBatch);
    thrust::fill(h_d.begin(), h_d.end(), 0.);

    thrust::device_vector<InType> d_a = h_a;
    thrust::device_vector<InType> d_b = h_b;
    thrust::device_vector<InType> d_c = h_c;
    thrust::device_vector<InType> d_d = h_d;

    const InType* A = thrust::raw_pointer_cast(d_a.data());
    const InType* B = thrust::raw_pointer_cast(d_b.data());
    const InType* C = thrust::raw_pointer_cast(d_c.data());
    InType* D = thrust::raw_pointer_cast(d_d.data());

    using Config = FusedTwoGemmsTraits<InType, AccType, WarpLayout, kM, kN, kK,
                                       kP, kTM, kTN, kTK, kTP>;

    int block_x = CeilDiv<kM, kTM>;
    int block_y = CeilDiv<kP, kTP>;
    int block_z = kBatch;
    dim3 grid(block_x, block_y, block_z);

    static constexpr int kThreads = tl::get_numel<WarpLayout> * 32;
    dim3 block(kThreads, 1, 1);

    static constexpr int kShmInput = (kTM * kTK + kTK * kTN + kTN * kTP);
    static constexpr int kShmOutput = kTM * kTP;
    static constexpr int kSharedSize = kShmInput < kShmOutput
                                           ? kShmOutput * sizeof(InType)
                                           : kShmInput * sizeof(InType);

    auto kernel = &kernel_wrapper<InType, AccType, Config>;

    if (kSharedSize > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedSize);
    }

    kernel<<<grid, block, kSharedSize, 0>>>(A, B, C, D);
    cudaDeviceSynchronize();

    h_d = d_d;

    thrust::host_vector<InType> h_acc(kM * kN * kBatch);
    thrust::fill(h_acc.begin(), h_acc.end(), 0.);
    thrust::device_vector<InType> d_acc = h_acc;

    thrust::host_vector<InType> h_d2(kM * kP * kBatch);
    thrust::fill(h_d2.begin(), h_d2.end(), 0.);
    thrust::device_vector<InType> d_d2 = h_d2;

    cublas_two_gemms(kM, kN, kK, kP, kBatch, A, B, C,
                     thrust::raw_pointer_cast(d_d2.data()),
                     thrust::raw_pointer_cast(d_acc.data()));
    cudaDeviceSynchronize();
    h_acc = d_acc;
    h_d2 = d_d2;

    InType* data = thrust::raw_pointer_cast(h_d.data());
    __half* ground_truth = thrust::raw_pointer_cast(h_d2.data());

#ifdef DEBUG
    printf("ours:\n");
    for (int i = 0; i < h_d.size(); ++i) {
        printf("%.3f, ", __half2float(data[i]));
        if (i && (i + 1) % 16 == 0) printf("\n");
    }
    printf("\nground_truth:\n");
    for (int i = 0; i < h_d.size(); ++i) {
        printf("%.3f, ", __half2float(ground_truth[i]));
        if (i && (i + 1) % 16 == 0) printf("\n");
    }
#endif

    if (check_results(data, ground_truth, kM * kP, epsilon)) {
        std::cout << "[" << kM << ", " << kN << ", " << kK << ", " << kP
                  << "], batch = " << kBatch << ", passed." << std::endl;
    } else {
        std::cout << "[" << kM << ", " << kN << ", " << kK << ", " << kP
                  << "], batch = " << kBatch << ", failed." << std::endl;
    }
}

int main() {
    using WarpLayout = tl::RowMajor<2, 1>;
    static constexpr int kSharedAccess = 64;

    run<B2BGemmShape<256 /*M*/, 128 /*N*/, 64 /*K*/, 64 /*P*/>,
        B2BGemmShape<64 /*kTM*/, 64 /*kTN*/, 64 /*kTK*/, 64 /*kTP*/>,
        WarpLayout, 1, kSharedAccess>(5e-3);

    return 0;
}
