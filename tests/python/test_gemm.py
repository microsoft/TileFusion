"""
Test GEMM operations.

isort:skip_file
"""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Any

import pytest
import torch

from tilefusion.ops import gemm

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def run_gemm(
    matrix_m: int,
    matrix_n: int,
    matrix_k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: bool,
    pipeline_level: int,
    warp_layout: torch.Tensor,
    swizzle_bytes: int,
) -> None:
    """Run GEMM operation."""
    tensor_a = torch.randn(
        matrix_m, matrix_k, dtype=torch.float16, device="cuda"
    )
    tensor_b = torch.randn(
        matrix_k, matrix_n, dtype=torch.float16, device="cuda"
    )
    tensor_c = torch.randn(
        matrix_m, matrix_n, dtype=torch.float32, device="cuda"
    )
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

    ref_c = torch.mm(tensor_a.cpu(), tensor_b.cpu().T)
    c_cpu = tensor_c.cpu().half()

    passed = True

    for row_idx in range(matrix_m):
        for col_idx in range(matrix_n):
            if abs(c_cpu[row_idx][col_idx] - ref_c[row_idx][col_idx]) > 5e-1:
                passed = False
                print(  # noqa: T201
                    f"c_cpu[{row_idx}][{col_idx}] = {c_cpu[row_idx][col_idx]}"
                )  # noqa: T201
                print(  # noqa: T201
                    f"ref_c[{row_idx}][{col_idx}] = {ref_c[row_idx][col_idx]}"
                )  # noqa: T201
                break

    assert passed, "GEMM operation failed."


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "test_case0",
            "m": 128,
            "n": 128,
            "k": 128,
            "tile_m": 64,
            "tile_n": 64,
            "tile_k": 64,
            "num_stages": 1,
            "pipeline_level": 0,
            "warp_layout": torch.tensor([1, 1], dtype=torch.int64),
            "swizzle_bytes": 64,
        },
        {
            "name": "test_case1",
            "m": 128,
            "n": 128,
            "k": 128,
            "tile_m": 64,
            "tile_n": 64,
            "tile_k": 64,
            "num_stages": 2,
            "pipeline_level": 1,
            "warp_layout": torch.tensor([1, 1], dtype=torch.int64),
            "swizzle_bytes": 64,
        },
        {
            "name": "test_case2",
            "m": 256,
            "n": 256,
            "k": 256,
            "tile_m": 64,
            "tile_n": 64,
            "tile_k": 64,
            "num_stages": 2,
            "pipeline_level": 1,
            "warp_layout": torch.tensor([1, 1], dtype=torch.int64),
            "swizzle_bytes": 64,
        },
        {
            "name": "test_case3",
            "m": 256,
            "n": 256,
            "k": 256,
            "tile_m": 64,
            "tile_n": 64,
            "tile_k": 64,
            "num_stages": 3,
            "pipeline_level": 2,
            "warp_layout": torch.tensor([1, 1], dtype=torch.int64),
            "swizzle_bytes": 64,
        },
        {
            "name": "test_case4",
            "m": 512,
            "n": 512,
            "k": 512,
            "tile_m": 64,
            "tile_n": 64,
            "tile_k": 64,
            "num_stages": 3,
            "pipeline_level": 2,
            "warp_layout": torch.tensor([1, 1], dtype=torch.int64),
            "swizzle_bytes": 64,
        },
        {
            "name": "test_case5",
            "m": 512,
            "n": 512,
            "k": 512,
            "tile_m": 64,
            "tile_n": 128,
            "tile_k": 128,
            "num_stages": 2,
            "pipeline_level": 1,
            "warp_layout": torch.tensor([2, 2], dtype=torch.int64),
            "swizzle_bytes": 128,
        },
        {
            "name": "test_case6",
            "m": 512,
            "n": 512,
            "k": 512,
            "tile_m": 64,
            "tile_n": 128,
            "tile_k": 128,
            "num_stages": 3,
            "pipeline_level": 2,
            "warp_layout": torch.tensor([2, 2], dtype=torch.int64),
            "swizzle_bytes": 128,
        },
    ],
    ids=lambda x: x["name"],
)
def test_gemm(test_case: dict[str, Any]) -> None:
    """Test GEMM operation."""
    run_gemm(
        test_case["m"],
        test_case["n"],
        test_case["k"],
        test_case["tile_m"],
        test_case["tile_n"],
        test_case["tile_k"],
        test_case["num_stages"],
        test_case["pipeline_level"],
        test_case["warp_layout"],
        test_case["swizzle_bytes"],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
