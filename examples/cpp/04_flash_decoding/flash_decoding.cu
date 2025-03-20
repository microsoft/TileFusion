// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "flash_decoding.hpp"
#include "util.hpp"

template <typename WholeShape, typename CtaTileShape, const int kChunkN,
          const int kSharedAccess>
void run(bool check = true) {
    using InType = __half;
    using AccType = float;
    using OutType = __half;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;
    static constexpr int kP = dim_size<3, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;
    static constexpr int kTP = dim_size<3, CtaTileShape>;

    static_assert(kK == kTK,
                  "The current implementation requires kTK == K for now.");
    static_assert(kP == kTP,
                  "The current implementation requires kTP == P for now.");

    // initialize data
    thrust::host_vector<InType> h_a(kM * kK);

    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_b(kK * kN);
    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_c(kN * kP);
    for (int i = 0; i < h_c.size(); ++i)
        h_c[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_d(kM * kP);
    thrust::fill(h_d.begin(), h_d.end(), 0.);

    // Host side memory initialization.
    thrust::host_vector<InType> acc(kM * kN);
    thrust::fill(acc.begin(), acc.end(), 0.);

    thrust::host_vector<InType> exp_values(kM * kP);
    thrust::fill(exp_values.begin(), exp_values.end(), 0.);

    thrust::host_vector<InType> h_o(kM * kP);
    thrust::fill(h_o.begin(), h_o.end(), 0.);

    thrust::host_vector<InType> cur_row_max(kM);
    thrust::fill(cur_row_max.begin(), cur_row_max.end(), 0.);

    thrust::host_vector<InType> prev_row_max(kM);
    thrust::fill(prev_row_max.begin(), prev_row_max.end(), 0.);

    thrust::host_vector<InType> new_row_max(kM);
    thrust::fill(new_row_max.begin(), new_row_max.end(), 0.);

    thrust::host_vector<InType> prev_norm_vec(kM);
    thrust::fill(prev_norm_vec.begin(), prev_norm_vec.end(), 0.);

    thrust::host_vector<InType> new_norm_vec(kM);
    thrust::fill(new_norm_vec.begin(), new_norm_vec.end(), 0.);

    thrust::host_vector<InType> prev_sum_vec(kM);
    thrust::fill(prev_sum_vec.begin(), prev_sum_vec.end(), 0.);

    thrust::host_vector<InType> cur_sum_vec(kM);
    thrust::fill(cur_sum_vec.begin(), cur_sum_vec.end(), 0.);

    thrust::host_vector<InType> new_sum_vec(kM);
    thrust::fill(new_sum_vec.begin(), new_sum_vec.end(), 0.);

    thrust::device_vector<InType> d_a = h_a;
    thrust::device_vector<InType> d_b = h_b;
    thrust::device_vector<InType> d_c = h_c;
    thrust::device_vector<InType> d_d = h_d;

    const InType* Q = thrust::raw_pointer_cast(d_a.data());
    const InType* K = thrust::raw_pointer_cast(d_b.data());
    const InType* V = thrust::raw_pointer_cast(d_c.data());
    InType* O = thrust::raw_pointer_cast(d_d.data());

    run_flash_decoding_fwd<InType, AccType, OutType, WholeShape, CtaTileShape,
                           kChunkN, kSharedAccess>(Q, K, V, O);

    cudaDeviceSynchronize();

    // Call host-side reference implementation.
    host_flash_decoding(kM, kN, kK, kP, thrust::raw_pointer_cast(h_a.data()),
                        thrust::raw_pointer_cast(h_b.data()),
                        thrust::raw_pointer_cast(h_c.data()),
                        thrust::raw_pointer_cast(h_o.data()),
                        thrust::raw_pointer_cast(acc.data()),
                        thrust::raw_pointer_cast(exp_values.data()),
                        thrust::raw_pointer_cast(cur_row_max.data()),
                        thrust::raw_pointer_cast(prev_row_max.data()),
                        thrust::raw_pointer_cast(new_row_max.data()),
                        thrust::raw_pointer_cast(prev_norm_vec.data()),
                        thrust::raw_pointer_cast(new_norm_vec.data()),
                        thrust::raw_pointer_cast(prev_sum_vec.data()),
                        thrust::raw_pointer_cast(cur_sum_vec.data()),
                        thrust::raw_pointer_cast(new_sum_vec.data()));

    h_d = d_d;

    if (check_results(thrust::raw_pointer_cast(h_o.data()),
                      thrust::raw_pointer_cast(h_d.data()), kM * kP)) {
        std::cout << "Test passed." << std::endl;
    } else {
        std::cout << "Test failed." << std::endl;
    }
}

int main() {
    static constexpr int kSharedAccess = 64;
    run<FlashDecodingShape<64 /*M*/, 128 /*N*/, 128 /*K*/, 128 /*P*/>,
        FlashDecodingShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128
                           /*kTP*/>,
        128, /*kChunkN*/ kSharedAccess>();

    run<FlashDecodingShape<64 /*M*/, 256 /*N*/, 128 /*K*/, 128 /*P*/>,
        FlashDecodingShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128
                           /*kTP*/>,
        256, /*kChunkN*/ kSharedAccess>();

    run<FlashDecodingShape<128 /*M*/, 256 /*N*/, 128 /*K*/, 128 /*P*/>,
        FlashDecodingShape<64 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128
                           /*kTP*/>,
        256, /*kChunkN*/ kSharedAccess>();


    return 0;
}
