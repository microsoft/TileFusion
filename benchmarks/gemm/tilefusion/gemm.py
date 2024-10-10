# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
from torch import Tensor

from compile import Compile

__all__ = [
    "gemm_func",
]


class GemmFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        M: int,
        N: int,
        K: int,
        kTM: int,
        kTN: int,
        kTK: int,
        kRK: int,
        warp_per_row: int,
        warp_per_col: int,
    ) -> Tensor:
        builder = Compile(file_prefix="gemm", tmp_dir="tmp")
        lib_name = builder.compile(M, N, K, kTM, kTN, kTK, kRK, warp_per_row,
                                   warp_per_col)

        if lib_name is None:
            raise RuntimeError("Failed to compile the library.")

        builder.apply(lib_name, [A, B, C], device=0)
        return C


gemm_func = GemmFunc.apply