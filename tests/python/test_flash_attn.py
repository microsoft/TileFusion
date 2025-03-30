"""Test flash attention implementation."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Any

import pytest
import torch

from tilefusion.ops import TiledFlashAttention


class FlashAttention:
    """Reference implementation of flash attention."""

    def __init__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        matrix_m: int,
        matrix_n: int,
        matrix_k: int,
        matrix_p: int,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        tile_p: int,
    ) -> None:
        """Initialize the flash attention.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            matrix_m: Size of first dimension of query.
            matrix_n: Size of first dimension of key/value.
            matrix_k: Size of second dimension of query/key.
            matrix_p: Size of second dimension of value/output.
            tile_m: Tile size for M dimension.
            tile_n: Tile size for N dimension.
            tile_k: Tile size for K dimension.
            tile_p: Tile size for P dimension.
        """
        self.matrix_m = matrix_m
        self.matrix_n = matrix_n
        self.matrix_k = matrix_k
        self.matrix_p = matrix_p

        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.tile_p = tile_p

        self.query = query
        self.key = key
        self.value = value
        self.output = torch.empty(matrix_m, matrix_p, device="cpu")

    def forward(self) -> torch.Tensor:
        """Perform the forward pass of flash attention.

        Returns:
            torch.Tensor: The attention output.
        """
        iter_n = self.matrix_n // self.tile_n

        prev_maxes = torch.zeros(self.matrix_m, 1, device="cpu")
        prev_sums = torch.zeros(self.matrix_m, 1, device="cpu")

        output = self.output.view(self.matrix_m, self.matrix_p)

        dK = self.key.view(self.matrix_k, self.matrix_n)
        dV = self.value.view(self.matrix_n, self.matrix_p)

        ks = torch.chunk(dK, iter_n, dim=-1)
        vs = torch.chunk(dV, iter_n, dim=-2)

        for chunk_idx in range(iter_n):
            query_view = self.query.view(self.matrix_m, self.matrix_k)  # m * k

            key_chunk = ks[chunk_idx]
            value_chunk = vs[chunk_idx]

            attn_weights = query_view @ key_chunk  # m * ktn

            # reduce maxes
            cur_maxes, _ = torch.max(attn_weights, dim=-1, keepdim=True)
            exp_weights = torch.exp(attn_weights - cur_maxes)
            # unnormalized attention score @ values
            exp_values = exp_weights @ value_chunk
            # move the normalization step to the very end
            # of the attention computation.
            cur_sums = torch.sum(exp_weights, dim=-1, keepdim=True)  # l(x_cur)

            # =========   renormalization  ======================#
            new_maxes = torch.max(cur_maxes, prev_maxes)  # update m(x)
            # renormalization factor for the previous block
            renorm_prev = torch.exp(prev_maxes - new_maxes)
            # renormalization factor for the current block
            renorm_cur = torch.exp(cur_maxes - new_maxes)

            # update normalization factor l(x)
            new_sums = renorm_prev * prev_sums + renorm_cur * cur_sums

            output = (
                output * prev_sums * renorm_prev + renorm_cur * exp_values
            ) / new_sums

            prev_sums = new_sums
            prev_maxes = new_maxes

        self.output = output

        return self.output


@pytest.fixture(autouse=True)  # type: ignore[misc]
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


def run_flash_attention(
    matrix_m: int,
    matrix_n: int,
    matrix_k: int,
    matrix_p: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    tile_p: int,
) -> None:
    """Run flash attention test with given dimensions.

    Args:
        matrix_m: Size of first dimension of query.
        matrix_n: Size of first dimension of key/value.
        matrix_k: Size of second dimension of query/key.
        matrix_p: Size of second dimension of value/output.
        tile_m: Tile size for M dimension.
        tile_n: Tile size for N dimension.
        tile_k: Tile size for K dimension.
        tile_p: Tile size for P dimension.
    """
    query = torch.randn(matrix_m, matrix_k, device="cpu")
    key = torch.randn(matrix_k, matrix_n, device="cpu")
    value = torch.randn(matrix_n, matrix_p, device="cpu")

    flash_attn = FlashAttention(
        query.half().flatten(),
        key.half().flatten(),
        value.half().flatten(),
        matrix_m,
        matrix_n,
        matrix_k,
        matrix_p,
        tile_m,
        tile_n,
        tile_k,
        tile_p,
    )

    ref_output = flash_attn.forward().half()

    cuda_query = query.cuda()
    cuda_key = key.cuda()
    cuda_value = value.cuda()

    tiled_flash_attention = TiledFlashAttention(
        cuda_query, cuda_key, cuda_value
    )
    output = tiled_flash_attention.forward()

    print("CPU Reference output: ", ref_output)  # noqa: T201
    print("tilefusion output: ", output)  # noqa: T201

    host_output = output.cpu()

    passed = True

    # Compare elements one by one and print the different numbers.
    for row_idx in range(matrix_m):
        for col_idx in range(matrix_p):
            if (
                abs(
                    host_output[row_idx][col_idx] - ref_output[row_idx][col_idx]
                )
                > 8e-2
            ):
                print("(", row_idx, ", ", col_idx, ")")  # noqa: T201
                print(  # noqa: T201
                    "tilefusion O: ", host_output[row_idx][col_idx]
                )
                print(  # noqa: T201
                    "CPU Reference O: ", ref_output[row_idx][col_idx]
                )

                passed = False
                break

    assert passed


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "test_case1",
            "matrix_m": 64,
            "matrix_n": 128,
            "matrix_k": 128,
            "matrix_p": 128,
            "tile_m": 64,
            "tile_n": 64,
            "tile_k": 128,
            "tile_p": 128,
        },
        {
            "name": "test_case2",
            "matrix_m": 64,
            "matrix_n": 256,
            "matrix_k": 128,
            "matrix_p": 128,
            "tile_m": 64,
            "tile_n": 64,
            "tile_k": 128,
            "tile_p": 128,
        },
    ],
    ids=lambda x: x["name"],
)  # type: ignore[misc]
def test_flash_attention(test_case: dict[str, Any]) -> None:
    """Test flash attention with different matrix dimensions.

    Args:
        test_case: Dictionary containing test parameters
    """
    run_flash_attention(
        matrix_m=test_case["matrix_m"],
        matrix_n=test_case["matrix_n"],
        matrix_k=test_case["matrix_k"],
        matrix_p=test_case["matrix_p"],
        tile_m=test_case["tile_m"],
        tile_n=test_case["tile_n"],
        tile_k=test_case["tile_k"],
        tile_p=test_case["tile_p"],
    )
