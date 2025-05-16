"""Flash attention implementation for tilefusion."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch

__all__ = [
    "flash_attention",
]


class FlashAttention:
    """A class implementing flash attention."""

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
            tile_length_q: The tile size of the query length dimension.
            tile_length_kv: The tile size of the key length dimension.
            tile_hidden_qk: The tile size of the query and key hidden dimension.
            tile_hidden_v: The tile size of the value hidden dimension.
            softmax_scale: Softmax scale. The scaling of QK^T before applying
                           softmax. Default is 1.0 / sqrt(hidden_qk).
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

        Args:
            query: Query tensor. shape (batch_size, length_q, hidden_qk).
            key: Key tensor. shape (batch_size, length_kv, hidden_qk).
            value: Value tensor. shape (batch_size, length_kv, hidden_v).

        Returns:
            torch.Tensor: The attention output.
        """
        length_q = query.size(-2)
        hidden_v = value.size(-1)

        output = torch.empty(
            length_q,
            hidden_v,
            dtype=query.dtype,
            device=query.device,
        )

        torch.ops.tilefusion.flash_attention(
            query,
            key.t().contiguous(),
            value.t().contiguous(),
            output,
            self.tile_length_q,
            self.tile_length_kv,
            self.tile_hidden_qk,
            self.tile_hidden_v,
            self.softmax_scale,
            self.causal,
        )
        return output


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tile_length_q: int,
    tile_length_kv: int,
    tile_hidden_qk: int,
    tile_hidden_v: int,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Compute flash attention between query, key and value tensors.

    This is a convenience function that creates a FlashAttention instance
    and applies it to the input tensors. Flash attention is an efficient
    implementation of attention that uses tiling to reduce memory bandwidth
    and improve performance.

    Args:
        query: Query tensor of shape (batch_size, length_q, hidden_qk)
        key: Key tensor of shape (batch_size, length_kv, hidden_qk)
        value: Value tensor of shape (batch_size, length_kv, hidden_v)
        tile_length_q: The tile size of the query length dimension.
        tile_length_kv: The tile size of the key length dimension.
        tile_hidden_qk: The tile size of the query and key hidden dimension.
        tile_hidden_v: The tile size of the value hidden dimension.
        softmax_scale: Scale factor applied before softmax
                       (typically 1/sqrt(hidden_qk))
        causal: Whether to apply causal masking to prevent attention to
                future tokens

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, length_q, hidden_v)
            containing the attention-weighted combination of values
    """
    attn_func = FlashAttention(
        tile_length_q,
        tile_length_kv,
        tile_hidden_qk,
        tile_hidden_v,
        softmax_scale,
        causal,
    )
    return attn_func(query, key, value)
