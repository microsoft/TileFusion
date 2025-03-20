"""Main module for testing matrix multiplication performance."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import torch
from gemm import gemm_func
from torch import Tensor


def run_unittest(
    matrix_a: Tensor,
    matrix_b: Tensor,
    matrix_c: Tensor,
    matrix_m: int,
    matrix_n: int,
    matrix_k: int,
    tile_m: int,
    tile_n: int,
    chunk_k: int,
    warp_layout: tuple[int, int],
    epsilon: float = 5e-2,
    debug_print: bool = False,
) -> bool:
    """Run unit test for matrix multiplication.

    Args:
        matrix_a: First input matrix.
        matrix_b: Second input matrix.
        matrix_c: Output matrix.
        matrix_m: Size of first dimension of matrix A.
        matrix_n: Size of second dimension of matrix B.
        matrix_k: Size of second dimension of matrix A and first dimension of
                  matrix B.
        tile_m: Tile size for M dimension.
        tile_n: Tile size for N dimension.
        chunk_k: Chunk size for K dimension.
        warp_layout: Tuple of (warp_per_row, warp_per_col).
        epsilon: Maximum allowed difference between reference and result.
        debug_print: Whether to print debug information.

    Returns:
        bool: True if test passes, False otherwise.
    """
    ref_c = matrix_a @ matrix_b.t()
    gemm_func(
        matrix_a,
        matrix_b,
        matrix_c,
        matrix_m,
        matrix_n,
        matrix_k,
        tile_m,
        tile_n,
        chunk_k,
        *warp_layout,
    )

    if debug_print:
        print("Result:")  # noqa: T201
        print(matrix_c)  # noqa: T201

        print("\nReference:")  # noqa: T201
        print(ref_c)  # noqa: T201

    numel = matrix_m * matrix_n
    avg_diff = (torch.sum(torch.abs(ref_c - matrix_c)) / numel).item()
    return not avg_diff > epsilon


def run_test(
    matrix_m: int,
    matrix_n: int,
    matrix_k: int,
    tile_m: int,
    tile_n: int,
    chunk_k: int,
    warp_layout: tuple[int, int],
) -> tuple[float, float]:
    """Run performance test for matrix multiplication.

    Args:
        matrix_m: Size of first dimension of matrix A.
        matrix_n: Size of second dimension of matrix B.
        matrix_k: Size of second dimension of matrix A and first dimension of
                  matrix B.
        tile_m: Tile size for M dimension.
        tile_n: Tile size for N dimension.
        chunk_k: Chunk size for K dimension.
        warp_layout: Tuple of (warp_per_row, warp_per_col).

    Returns:
        tuple[float, float]: Average execution time in milliseconds for
        (tilefusion, cublas).

    Raises:
        RuntimeError: If unit test fails.
    """
    device = torch.device("cuda")
    dtype = torch.float16

    matrix_a = torch.normal(
        mean=0.1,
        std=1e-3,
        size=(matrix_m, matrix_k),
        device=device,
        dtype=dtype,
    )
    matrix_b = torch.normal(
        mean=0.1,
        std=1e-3,
        size=(matrix_n, matrix_k),
        device=device,
        dtype=dtype,
    )

    shape = (matrix_m, matrix_n)
    matrix_c = torch.zeros(*shape, device=device, dtype=torch.float32)

    if not run_unittest(
        matrix_a,
        matrix_b,
        matrix_c,
        matrix_m,
        matrix_n,
        matrix_k,
        tile_m,
        tile_n,
        chunk_k,
        warp_layout,
    ):
        raise RuntimeError("Failed unittest.")

    for _ in range(5):  # warm up
        gemm_func(
            matrix_a,
            matrix_b,
            matrix_c,
            matrix_m,
            matrix_n,
            matrix_k,
            tile_m,
            tile_n,
            chunk_k,
            *warp_layout,
        )
        matrix_a @ matrix_b.t()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iters = 50
    start_event.record()
    for _ in range(iters):
        gemm_func(
            matrix_a,
            matrix_b,
            matrix_c,
            matrix_m,
            matrix_n,
            matrix_k,
            tile_m,
            tile_n,
            chunk_k,
            *warp_layout,
        )
    end_event.record()
    torch.cuda.synchronize()

    time1 = start_event.elapsed_time(end_event) / iters

    start_event.record()
    for _ in range(iters):
        matrix_a @ matrix_b.t()
    end_event.record()
    torch.cuda.synchronize()

    time2 = start_event.elapsed_time(end_event) / iters
    return time1, time2


if __name__ == "__main__":
    matrix_m = 4096
    matrix_n = 4096
    matrix_k = 4096

    header = (
        "Whole Shape\tBlock Shape\tthreads"
        "\ttilefusion(ms)\tcublass(ms)\tRatio"
    )
    print(header)  # noqa: T201

    warp_layout = (1, 2)
    threads = warp_layout[0] * warp_layout[1] * 32
    for tile_m in [64, 128]:
        for tile_n in [64, 128]:
            for chunk_k in [32, 64, 128]:
                time1, time2 = run_test(
                    matrix_m,
                    matrix_n,
                    matrix_k,
                    tile_m,
                    tile_n,
                    chunk_k,
                    warp_layout,
                )
                print(  # noqa: T201
                    (
                        "[{}, {}, {}]\t[{}, {}, {}]"
                        "\t{}\t{:.4f}\t{:.4f}\t{:.3f}"
                    ).format(
                        matrix_m,
                        matrix_n,
                        matrix_k,
                        tile_m,
                        tile_n,
                        chunk_k,
                        threads,
                        time1,
                        time2,
                        time1 / time2,
                    )
                )

    for warp_layout in [(2, 2), (2, 4)]:
        threads = warp_layout[0] * warp_layout[1] * 32

        for tile_m in [64, 128, 256]:
            for tile_n in [64, 128, 256]:
                for chunk_k in [32, 64, 128]:
                    time1, time2 = run_test(
                        matrix_m,
                        matrix_n,
                        matrix_k,
                        tile_m,
                        tile_n,
                        chunk_k,
                        warp_layout,
                    )
                    print(  # noqa: T201
                        (
                            "[{}, {}, {}]\t[{}, {}, {}]"
                            "\t{}\t{:.4f}\t{:.4f}\t{:.3f}"
                        ).format(
                            matrix_m,
                            matrix_n,
                            matrix_k,
                            tile_m,
                            tile_n,
                            chunk_k,
                            threads,
                            time1,
                            time2,
                            time1 / time2,
                        )
                    )
