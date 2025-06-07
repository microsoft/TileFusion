// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cell/mod.hpp"
#include "common/test_utils.hpp"
#include "cuda_info.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tilefusion::testing {
using namespace cell;
using namespace copy;

namespace tl = tile_layout;

namespace {
template <typename DType, typename Global, typename Shared,  //
          typename Loader, typename Storer>
__global__ void copy_g2s(const DType* src_ptr, DType* dst_ptr, Loader& loader,
                         Storer& storer) {
  extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
  auto* buf = reinterpret_cast<DType*>(buf_);

  Global src(src_ptr);  // global memory tile
  Shared inter(buf);    // shared memory tile
  Global dst(dst_ptr);  // global memory tile

  // if (thread(0)) {
  //   printf("\nglobal\n");
  //   src.dump_value();

  //   // printf("\nshared\n");
  //   // inter.dump_value();
  //   // printf("\n");
  // }

  // loader(src, inter);
  // __copy_async();
  // __syncthreads();

  // storer(inter, dst);
  // __syncthreads();
}
}  // namespace

template <typename DType, typename WarpLayout, const int kRows, const int kCols>
void run_test_row_major() {
  static const int kThreads = tl::get_numel<WarpLayout> * 32;

  int numel = kRows * kCols;
  thrust::host_vector<DType> h_A(numel);
  for (int i = 0; i < h_A.size(); ++i)
    h_A[i] = static_cast<DType>(rand_float(-5.f, 5.f));

  thrust::device_vector<DType> d_B(numel);
  thrust::fill(d_B.begin(), d_B.end(), static_cast<DType>(0.));
  thrust::device_vector<DType> d_A = h_A;

  using Global = GlobalTile<DType, tl::RowMajor<kRows, kCols>>;

  using SharedLayout =
      tl::BlockRowMajor<tl::RowMajor<kRows, kCols>, tl::RowMajor<8, 32>>;
  using Shared = SharedTileV2<DType, SharedLayout>;

  std::cout << Global{} << std::endl;
  std::cout << Shared{} << std::endl;

  using Loader = GlobalToSharedLoaderV2<Global, WarpLayout>;
  Loader loader;

  using Storer = SharedToGlobalStorerV2<Shared, WarpLayout>;
  Storer storer;

  auto kernel = copy_g2s<DType, Global, Shared, Loader, Storer>;
  int shm_size = kRows * kCols * sizeof(DType);
  configure_dynamic_shared_memory(kernel, shm_size);

  dim3 dim_grid(1, 1);
  dim3 dim_block(kThreads);
  kernel<<<dim_grid, dim_block, shm_size>>>(
      thrust::raw_pointer_cast(d_A.data()),
      thrust::raw_pointer_cast(d_B.data()), loader, storer);
  cudaDeviceSynchronize();

  thrust::host_vector<DType> h_B(numel);
  h_B = d_B;

  // assert_equal(reinterpret_cast<DType*>(thrust::raw_pointer_cast(h_A.data())),
  //              reinterpret_cast<DType*>(thrust::raw_pointer_cast(h_B.data())),
  //              numel, 1e-5);
}

TEST(GlobalToSharedLoad, test_row_major_load) {
  run_test_row_major<__half, tl::RowMajor<1, 1>, 16, 32>();
#ifdef CUDA_FP8_AVAILABLE
  run_test_row_major<__fp8_e4m3, tl::RowMajor<1, 1>, 16, 32>();
  // run_test_row_major<__fp8_e5m2, tl::RowMajor<1, 1>, 16, 32>();
#endif
}

}  // namespace tilefusion::testing
