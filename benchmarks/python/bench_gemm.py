"""
Benchmark GEMM operations.

isort:skip_file
"""

import torch
import logging
from tilefusion.ops import gemm
from bench_utils import do_bench


def run_gemm(
    matrix_m: int,
    matrix_n: int,
    matrix_k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int,
    pipeline_level: int,
    warp_layout: torch.Tensor,
    swizzle_bytes: int,
) -> None:
    """Run GEMM operation and benchmark it."""
    tensor_a = torch.randn(
        matrix_m, matrix_k, dtype=torch.float16, device="cuda"
    )
    tensor_b = torch.randn(
        matrix_k, matrix_n, dtype=torch.float16, device="cuda"
    )
    tensor_c = torch.randn(
        matrix_m, matrix_n, dtype=torch.float32, device="cuda"
    )

    warm_up_iterations = 10
    benchmark_iterations = 50

    def tilefusion_gemm() -> None:
        gemm(
            tensor_a,
            tensor_b,
            tensor_c,
            tile_m,
            tile_n,
            tile_k,
            num_stages,
            pipeline_level,
            warp_layout,
            swizzle_bytes,
        )

    def cublas_gemm() -> None:
        torch.matmul(tensor_a, tensor_b)

    tilefusion_avg_time_ms = do_bench(
        tilefusion_gemm,
        warmup=warm_up_iterations,
        rep=benchmark_iterations,
        return_mode="mean",
    )

    tilefusion_tflops = (2 * matrix_m * matrix_n * matrix_k) / (
        tilefusion_avg_time_ms * 1e12
    )

    cublas_avg_time_ms = do_bench(
        cublas_gemm,
        warmup=warm_up_iterations,
        rep=benchmark_iterations,
        return_mode="mean",
    )

    cublas_tflops = (2 * matrix_m * matrix_n * matrix_k) / (
        cublas_avg_time_ms * 1e12
    )

    # print(f"Benchmarking GEMM with M={matrix_m}, N={matrix_n}, K={matrix_k}")
    # print(f"Tile: M={tile_m}, N={tile_n}, K={tile_k}")
    # print(f"Stages: {num_stages}, Pipeline Level: {pipeline_level}")
    # print(f"Warp Layout: {warp_layout.tolist()}")
    # print(f"Average execution time: {avg_time_ms:.3f} ms")
    # print(f"Achieved TFLOPs: {tflops:.2f}")
    logging.info(
        f"Benchmarking GEMM with M={matrix_m}, N={matrix_n}, K={matrix_k}"
    )
    logging.info(f"Tile: M={tile_m}, N={tile_n}, K={tile_k}")
    logging.info(f"Stages: {num_stages}, Pipeline Level: {pipeline_level}")
    logging.info(f"Warp Layout: {warp_layout.tolist()}")
    logging.info(f"Average execution time: {tilefusion_avg_time_ms:.3f} ms")
    logging.info(f"Achieved TFLOPs: {tilefusion_tflops:.2f}")
    logging.info(f"Cublas Average execution time: {cublas_avg_time_ms:.3f} ms")
    logging.info(f"Cublas Achieved TFLOPs: {cublas_tflops:.2f}")
    logging.info(
        f"Tilefusion is {cublas_avg_time_ms / tilefusion_avg_time_ms:.2f}x "
        f"faster than Cublas"
    )


if __name__ == "__main__":

    mat_m, mat_n, mat_k = 8192, 8192, 1024

    tl_m, tl_n, tl_k = 64, 128, 128

    logging.basicConfig(level=logging.INFO)
    logging.info("--- Starting GEMM Benchmark ---")
    run_gemm(
        matrix_m=mat_m,
        matrix_n=mat_n,
        matrix_k=mat_k,
        tile_m=tl_m,
        tile_n=tl_n,
        tile_k=tl_k,
        num_stages=1,
        pipeline_level=0,
        warp_layout=torch.tensor([2, 2], dtype=torch.int64),
        swizzle_bytes=128,
    )

    run_gemm(
        matrix_m=mat_m,
        matrix_n=mat_n,
        matrix_k=mat_k,
        tile_m=tl_m,
        tile_n=tl_n,
        tile_k=tl_k,
        num_stages=2,
        pipeline_level=1,
        warp_layout=torch.tensor([2, 2], dtype=torch.int64),
        swizzle_bytes=128,
    )

    run_gemm(
        matrix_m=mat_m,
        matrix_n=mat_n,
        matrix_k=mat_k,
        tile_m=tl_m,
        tile_n=tl_n,
        tile_k=tl_k,
        num_stages=3,
        pipeline_level=2,
        warp_layout=torch.tensor([2, 2], dtype=torch.int64),
        swizzle_bytes=128,
    )
    logging.info("--- GEMM Benchmark Finished ---")
