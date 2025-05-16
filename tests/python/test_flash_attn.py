"""Test flash attention implementation.

isort:skip_file
"""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from typing import Any

import pytest
import torch
import math

from tilefusion.ops import flash_attention


class FlashAttentionRef:
    """Reference implementation of flash attention."""

    def __init__(
        self,
        tile_length_q: int,
        tile_length_kv: int,
        tile_hidden_qk: int,
        tile_hidden_v: int,
        softmax_scale: float,
        causal: bool,
    ) -> None:
        """Initialize the flash attention.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            length_q: Size of first dimension of query.
            length_kv: Size of first dimension of key/value.
            hidden_qk: Size of second dimension of query/key.
            hidden_v: Size of second dimension of value/output.
            tile_length_q: Tile size for M dimension.
            tile_length_kv: Tile size for N dimension.
            tile_hidden_qk: Tile size for K dimension.
            tile_hidden_v: Tile size for P dimension.
            softmax_scale: Softmax scale.
                The scaling of QK^T before applying softmax.
                Default is 1.0 / sqrt(matrix_k).
            causal: bool. Whether to apply causal mask.
        """
        self.tile_length_q = tile_length_q
        self.tile_length_kv = tile_length_kv
        self.tile_hidden_qk = tile_hidden_qk
        self.tile_hidden_v = tile_hidden_v

        self.softmax_scale = softmax_scale
        self.causal = causal

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Perform the forward pass of flash attention.

        Returns:
            torch.Tensor: The attention output.
        """
        dtype = query.dtype
        device = query.device

        length_q = query.shape[0]
        length_kv = key.shape[1]
        hidden_v = value.shape[1]

        output = torch.empty(
            length_q,
            hidden_v,
            dtype=dtype,
            device=device,
        )

        iter_n = length_kv // self.tile_length_kv

        prev_maxes = torch.zeros(length_q, 1, dtype=dtype, device=device)
        prev_sums = torch.zeros(length_q, 1, dtype=dtype, device=device)

        ks = torch.chunk(key, iter_n, dim=-1)
        vs = torch.chunk(value, iter_n, dim=-2)

        for chunk_idx in range(iter_n):
            key_chunk = ks[chunk_idx]
            value_chunk = vs[chunk_idx]

            attn_weights = query @ key_chunk  # m * ktn

            if self.causal:
                # Create a causal mask for the attention weights.
                mask = torch.tril(
                    torch.ones_like(attn_weights), diagonal=0
                ).bool()
                attn_weights = torch.where(mask, attn_weights, float("-inf"))

            attn_weights = attn_weights * self.softmax_scale

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

        return output


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


def run_flash_attention(
    length_q: int,
    length_kv: int,
    hidden_qk: int,
    hidden_v: int,
    tile_length_q: int,
    tile_length_kv: int,
    tile_hidden_qk: int,
    tile_hidden_v: int,
    softmax_scale: float,
    causal: bool,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> None:
    """Run flash attention test with given dimensions.

    Args:
        length_q: Size of first dimension of query.
        length_kv: Size of first dimension of key/value.
        hidden_qk: Size of second dimension of query/key.
        hidden_v: Size of second dimension of value/output.
        tile_length_q: Tile size for M dimension.
        tile_length_kv: Tile size for N dimension.
        tile_hidden_qk: Tile size for K dimension.
        tile_hidden_v: Tile size for P dimension.
        softmax_scale: Softmax scale.
            The scaling of QK^T before applying softmax.
            Default is 1.0 / sqrt(matrix_k).
        causal: bool. Whether to apply causal mask.
    """
    query = torch.randn(length_q, hidden_qk, dtype=dtype, device=device)
    key = torch.randn(hidden_qk, length_kv, dtype=dtype, device=device)
    value = torch.randn(length_kv, hidden_v, dtype=dtype, device=device)

    flash_attn_ref = FlashAttentionRef(
        tile_length_q,
        tile_length_kv,
        tile_hidden_qk,
        tile_hidden_v,
        softmax_scale,
        causal,
    )
    ref_output = flash_attn_ref(query, key, value)

    output = flash_attention(
        query=query,
        key=key,
        value=value,
        tile_length_q=tile_length_q,
        tile_length_kv=tile_length_kv,
        tile_hidden_qk=tile_hidden_qk,
        tile_hidden_v=tile_hidden_v,
        softmax_scale=softmax_scale,
        causal=causal,
    )

    assert torch.allclose(ref_output, output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "test_case1",
            "length_q": 64,
            "length_kv": 128,
            "hidden_qk": 128,
            "hidden_v": 128,
            "tile_length_q": 64,
            "tile_length_kv": 128,
            "tile_hidden_qk": 128,
            "tile_hidden_v": 128,
            "softmax_scale": 1.0 / math.sqrt(128),
            "causal": False,
        },
        {
            "name": "test_case2",
            "length_q": 128,
            "length_kv": 128,
            "hidden_qk": 128,
            "hidden_v": 128,
            "tile_length_q": 64,
            "tile_length_kv": 128,
            "tile_hidden_qk": 128,
            "tile_hidden_v": 128,
            "softmax_scale": 1.0 / math.sqrt(128),
            "causal": True,
        },
    ],
    ids=lambda x: x["name"],
)
def test_flash_attention(test_case: dict[str, Any]) -> None:
    """Test flash attention with different matrix dimensions.

    Args:
        test_case: Dictionary containing test parameters
    """
    run_flash_attention(
        length_q=test_case["length_q"],
        length_kv=test_case["length_kv"],
        hidden_qk=test_case["hidden_qk"],
        hidden_v=test_case["hidden_v"],
        tile_length_q=test_case["tile_length_q"],
        tile_length_kv=test_case["tile_length_kv"],
        tile_hidden_qk=test_case["tile_hidden_qk"],
        tile_hidden_v=test_case["tile_hidden_v"],
        softmax_scale=test_case["softmax_scale"],
        causal=test_case["causal"],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
