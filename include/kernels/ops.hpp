// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_utils.hpp"
#include "kernel_registry.hpp"

namespace tilefusion::kernels {

// reference:
// https://github.com/InfiniTensor/RefactorGraph/blob/master/src/04kernel/cuda/src/scatter_nd.cu#L7
// TODO: optimize the kernel by increasing the number of threads to perform
// `atomic_add` operations under `slice_size`.
/**
 * @brief The ScatterNdkernel updates the content of `updates` into `data` based
 * on the index information provided in the given `indices`.
 *
 * @param in The input tensor `updates`.
 * @param out The output tensor `data`.
 * @param indices The indices tensor.
 * @param strides record the stride information between different dimensions in
 * the `data` tensor.
 * @param n The number of indices.
 * @param rank The last dimension of `indices`.
 * @param slice_size The length of the slice to be updated. Specifically, it is
 * the product of the difference between the rank of `data` and the last
 * dimension of `indices` along the memory dimensions of `data`.
 */
template <typename T>
__global__ void ke_scatter_nd(const T* in, T* out, const int64_t* indices,
                              unsigned int const* __restrict__ strides,
                              size_t n, size_t rank, size_t slice_size);

// declare the host function for scatter_nd
TILEFUSION_EXPORT void scatter_nd(const torch::Tensor& data,
                                  torch::Tensor& updates,
                                  const torch::Tensor& indices);

// declare the host function for gemm
TILEFUSION_EXPORT void gemm(const torch::Tensor& A, const torch::Tensor& B,
                            torch::Tensor& C, int64_t tm, int64_t tn,
                            int64_t tk, int64_t num_stages,
                            int64_t pipeline_level);

// declare the host function for fused two gemms
TILEFUSION_EXPORT void fused_two_gemms(const torch::Tensor& A,
                                       const torch::Tensor& B,
                                       const torch::Tensor& C, torch::Tensor& D,
                                       int64_t tm, int64_t tn, int64_t tk,
                                       int64_t tp);

// declare the host function for flash attention
TILEFUSION_EXPORT void flash_attention(
    const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V,
    torch::Tensor& O, int64_t tile_length_q, int64_t tile_length_kv,
    int64_t tile_hidden_qk, int64_t tile_hidden_v, double softmax_scale,
    bool causal);

}  // namespace tilefusion::kernels
