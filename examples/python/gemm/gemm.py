"""Module for matrix multiplication operations using CUDA."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
from compile import Compile
from torch import Tensor

__all__ = [
    "gemm_func",
]


class GemmFunc(torch.autograd.Function):
    """Custom autograd function for matrix multiplication."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,  # noqa: U100
        matrix_a: Tensor,
        matrix_b: Tensor,
        matrix_c: Tensor,
        matrix_m: int,
        matrix_n: int,
        matrix_k: int,
        tile_m: int,
        tile_n: int,
        chunk_k: int,
        warp_per_row: int,
        warp_per_col: int,
    ) -> Tensor:
        """Forward pass of the matrix multiplication operation.

        Args:
            ctx: PyTorch autograd context.
            matrix_a: First input matrix.
            matrix_b: Second input matrix.
            matrix_c: Output matrix.
            matrix_m: Size of first dimension of matrix A.
            matrix_n: Size of second dimension of matrix B.
            matrix_k: Size of second dimension of matrix A and first dimension
                      of matrix B.
            tile_m: Tile size for M dimension.
            tile_n: Tile size for N dimension.
            chunk_k: Chunk size for K dimension.
            warp_per_row: Number of warps per row.
            warp_per_col: Number of warps per column.

        Returns:
            Tensor: The result matrix C.

        Raises:
            RuntimeError: If compilation fails.
        """
        builder = Compile(file_prefix="gemm", tmp_dir="tmp")
        lib_name = builder.compile(
            matrix_m,
            matrix_n,
            matrix_k,
            tile_m,
            tile_n,
            chunk_k,
            warp_per_row,
            warp_per_col,
        )

        if lib_name is None:
            raise RuntimeError("Failed to compile the library.")

        builder.apply(lib_name, [matrix_a, matrix_b, matrix_c], device=0)
        return matrix_c


gemm_func = GemmFunc.apply
