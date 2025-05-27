"""
Benchmark GEMM operations.

isort:skip_file
"""

import torch
import logging
import csv
import os

# from typing import Tuple, Dict, List
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
) -> dict[str, float]:
    """Run GEMM operation and benchmark it.

    Returns:
        Dict containing performance metrics
    """
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

    return {
        "matrix_m": matrix_m,
        "matrix_n": matrix_n,
        "matrix_k": matrix_k,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "num_stages": num_stages,
        "pipeline_level": pipeline_level,
        "warp_per_row": warp_layout[0].item(),
        "warp_per_col": warp_layout[1].item(),
        "tilefusion_time_ms": tilefusion_avg_time_ms,
        "cublas_time_ms": cublas_avg_time_ms,
        "tilefusion_tflops": tilefusion_tflops,
        "cublas_tflops": cublas_tflops,
        "speedup_vs_cublas": cublas_avg_time_ms / tilefusion_avg_time_ms,
    }


def write_results_to_csv(
    results: list[dict[str, float]], filename: str
) -> None:
    """Write benchmark results to CSV file."""
    if not results:
        return

    # Get fieldnames from the first result
    fieldnames = list(results[0].keys())

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Format floating point numbers to 4 decimal places
        formatted_results = []
        for result in results:
            formatted_result = {}
            for key, value in result.items():
                formatted_result[key] = f"{value:.4f}"
            formatted_results.append(formatted_result)

        writer.writerows(formatted_results)


def print_summary(results: list[dict[str, float]]) -> None:
    """Print summary statistics of benchmark results."""
    print(  # noqa: T201
        "\nSummary Statistics by Matrix Size and Pipeline Configuration:"
    )
    print(  # noqa: T201, E501
        "Matrix Size(M, N, K) | Tile Size(M, N, K) | Stages | Warp Layout | "
        "Cublas(ms) | TileFusion(ms) | Ratio "
    )
    print("-" * 100)  # noqa: T201

    for result in results:
        print(  # noqa: T201, E501
            f"({result['matrix_m']}, {result['matrix_n']}, {result['matrix_k']}) | "  # noqa: E501
            f"({result['tile_m']}, {result['tile_n']}, {result['tile_k']}) | "
            f"{result['num_stages']} | "
            f"[{result['warp_per_row']}, {result['warp_per_col']}] | "
            f"{result['cublas_time_ms']:.4f} | "
            f"{result['tilefusion_time_ms']:.4f} | "
            f"{result['speedup_vs_cublas']:.4f}"
        )


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    matrix_shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
        (64, 128, 16384),
        (64, 128, 32768),
        (1024, 1024, 32768),
    ]

    tile_shapes = [(64, 128, 128)]
    warp_layouts = [(2, 2), (4, 2), (2, 4)]
    num_stages_list = [1, 2, 3]
    results = []

    logging.basicConfig(level=logging.INFO)
    logging.info("--- Starting GEMM Benchmark ---")

    for matrix_shape in matrix_shapes:
        for tile_shape in tile_shapes:
            for warp_layout in warp_layouts:
                for num_stages in num_stages_list:
                    pipeline_level = 1 if num_stages > 1 else 0
                    results.append(
                        run_gemm(
                            matrix_m=matrix_shape[0],
                            matrix_n=matrix_shape[1],
                            matrix_k=matrix_shape[2],
                            tile_m=tile_shape[0],
                            tile_n=tile_shape[1],
                            tile_k=tile_shape[2],
                            num_stages=num_stages,
                            pipeline_level=pipeline_level,
                            warp_layout=torch.tensor(
                                warp_layout, dtype=torch.int64
                            ),
                            swizzle_bytes=64,
                        )
                    )

    csv_filename = "benchmarks/python/gemm_benchmark.csv"

    # Save results to CSV
    write_results_to_csv(results, csv_filename)
    print(f"\nResults saved to {csv_filename}")  # noqa: T201

    # Print summary statistics
    print_summary(results)

    logging.info("--- GEMM Benchmark Finished ---")
