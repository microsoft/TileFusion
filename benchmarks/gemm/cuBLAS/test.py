# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch

from torch import Tensor

torch.ops.load_library("src/build/libcublas_gemm.so")


def gemm(m: int,
         n: int,
         k: int,
         a: Tensor,
         b: Tensor,
         c: Tensor,
         elapsed_time: Tensor,
         iters: int = 0,
         warmup: int = 0):
    torch.ops.cublas_gemm.gemm(m, n, k, a, b, c, elapsed_time, iters, warmup)


if __name__ == '__main__':
    M = 4096
    N = 4096
    K = 4096

    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(1234)

    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    c = torch.zeros(M, N, device=device, dtype=dtype)
    time = torch.zeros(1, device=torch.device("cpu"), dtype=torch.float32)

    gemm(M, N, K, a, b, c, time)
    ref_c = a @ b.t()

    epsilon = 5e-2
    avg_diff = (torch.sum(torch.abs(ref_c - c) / (M * N))).item()
    if (avg_diff > epsilon):
        raise ValueError("Failed unittest.")

    gemm(M, N, K, a, b, c, time, 20, 5)
    print("Elapsed time: {:.4} ms".format(time.item()))
