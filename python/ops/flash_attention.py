"""Flash attention implementation for tilefusion."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch

__all__ = [
    "TiledFlashAttention",
]


class TiledFlashAttention:
    """A class implementing tiled flash attention."""

    def __init__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """Initialize the tiled flash attention.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
        """
        self.m, self.k = query.size(-2), query.size(-1)
        self.n, self.p = value.size(-2), value.size(-1)

        self.query = query.half().flatten()
        self.key = key.half().t().flatten()
        self.value = value.half().t().flatten()

        self.output = torch.empty(
            self.m, self.p, dtype=torch.half, device="cuda"
        ).flatten()

    def forward(self) -> torch.Tensor:
        """Perform the forward pass of tiled flash attention.

        Returns:
            torch.Tensor: The attention output.
        """
        torch.ops.tilefusion.flash_attention(
            self.query,
            self.key,
            self.value,
            self.output,
            self.m,
            self.n,
            self.k,
            self.p,
        )

        return self.output.view(self.m, self.p)
