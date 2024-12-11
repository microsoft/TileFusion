// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cutlass_fa.cuh"
#include "util.hpp"

// template <typename WholeShape, typename CtaTileShape, const int kBatch>
void run(bool check = true) {
    using InType = cutlass::half_t;
    using AccType = cutlass::half_t;
    using OutType = cutlass::half_t;

    // static constexpr int kM = dim_size<0, WholeShape>;
    // static constexpr int kN = dim_size<1, WholeShape>;
    // static constexpr int kK = dim_size<2, WholeShape>;
    // static constexpr int kP = dim_size<3, WholeShape>;

    // static constexpr int kTM = dim_size<0, CtaTileShape>;
    // static constexpr int kTN = dim_size<1, CtaTileShape>;
    // static constexpr int kTK = dim_size<2, CtaTileShape>;
    // static constexpr int kTP = dim_size<3, CtaTileShape>;

    static constexpr int kM = 64;
    static constexpr int kN = 64;
    static constexpr int kK = 128;
    static constexpr int kP = 128;

    static constexpr int kTM = 64;
    static constexpr int kTN = 64;
    static constexpr int kTK = 128;
    static constexpr int kTP = 128;

    static constexpr int kBatch = 1;

    static constexpr int kWarpPerRow = 4;
    static constexpr int kWarpPerCol = 1;
    static constexpr int kThreads = 128;
    static constexpr int kStagesQK = 2;
    static constexpr int kStagesV = 2;

    static_assert(kK == kTK,
                  "The current implementation requires kTK == K for now.");
    static_assert(kP == kTP,
                  "The current implementation requires kTP == P for now.");

    // initialize data
    thrust::host_vector<InType> h_a(kM * kK * kBatch);

    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_b(kK * kN * kBatch);
    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_c(kN * kP * kBatch);
    for (int i = 0; i < h_c.size(); ++i)
        h_c[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_d(kM * kP * kBatch);
    thrust::fill(h_d.begin(), h_d.end(), 0.);

    // Host side memory initialization.
    thrust::host_vector<InType> acc(kM * kN * kBatch);
    thrust::fill(acc.begin(), acc.end(), 0.);

    thrust::host_vector<InType> exp_values(kM * kP * kBatch);
    thrust::fill(exp_values.begin(), exp_values.end(), 0.);

    thrust::host_vector<InType> h_o(kM * kP * kBatch);
    thrust::fill(h_o.begin(), h_o.end(), 0.);

    thrust::host_vector<InType> cur_row_max(kM * kBatch);
    thrust::fill(cur_row_max.begin(), cur_row_max.end(), 0.);

    thrust::host_vector<InType> prev_row_max(kM * kBatch);
    thrust::fill(prev_row_max.begin(), prev_row_max.end(), 0.);

    thrust::host_vector<InType> new_row_max(kM * kBatch);
    thrust::fill(new_row_max.begin(), new_row_max.end(), 0.);

    thrust::host_vector<InType> prev_norm_vec(kM * kBatch);
    thrust::fill(prev_norm_vec.begin(), prev_norm_vec.end(), 0.);

    thrust::host_vector<InType> new_norm_vec(kM * kBatch);
    thrust::fill(new_norm_vec.begin(), new_norm_vec.end(), 0.);

    thrust::host_vector<InType> prev_sum_vec(kM * kBatch);
    thrust::fill(prev_sum_vec.begin(), prev_sum_vec.end(), 0.);

    thrust::host_vector<InType> cur_sum_vec(kM * kBatch);
    thrust::fill(cur_sum_vec.begin(), cur_sum_vec.end(), 0.);

    thrust::host_vector<InType> new_sum_vec(kM * kBatch);
    thrust::fill(new_sum_vec.begin(), new_sum_vec.end(), 0.);

    thrust::device_vector<InType> d_a = h_a;
    thrust::device_vector<InType> d_b = h_b;
    thrust::device_vector<InType> d_c = h_c;
    thrust::device_vector<InType> d_d = h_d;

    const InType* A = thrust::raw_pointer_cast(d_a.data());
    const InType* B = thrust::raw_pointer_cast(d_b.data());
    const InType* C = thrust::raw_pointer_cast(d_c.data());
    InType* D = thrust::raw_pointer_cast(d_d.data());

    // int block_x = CeilDiv<kM, kTM>;
    // int block_y = CeilDiv<kP, kTP>;
    int block_x = (kM + kTM - 1) / kTM;
    int block_y = (kP + kTP - 1) / kTP;
    int block_z = kBatch;

    dim3 grid(block_x, block_y, block_z);
    dim3 block(kThreads, 1, 1);

    int shm_input = (kTM * kTK + kTK * kTN + kTN * kTP);
    int shm_output = kTM * kTP;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(InType)
                                          : shm_input * sizeof(InType);

    using Traits =
        benchmarks::cutlass_wrapper::FATraits<cutlass::half_t, kM, kN, kK, kP,
                                              kTM, kTN, kTK, kTP, kWarpPerRow,
                                              kWarpPerCol, kThreads>;

    auto fa_kernel =
        benchmarks::cutlass_wrapper::fa_kernel<cutlass::half_t, Traits, kM, kN,
                                               kK, kP, kTM, kTN, kTK, kTP,
                                               kThreads, kStagesQK, kStagesV>;

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            fa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    fa_kernel<<<grid, block, shm_size, 0>>>(A, B, C, D);

    cudaDeviceSynchronize();
}

int main() { run(); }