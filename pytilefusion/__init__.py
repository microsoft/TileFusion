# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch

torch.ops.load_library("build/src/libtilefusion.so")


def scatter_nd(scatter_data, scatter_indices, scatter_updates):
    torch.ops.tilefusion.scatter_nd(
        scatter_data, scatter_updates, scatter_indices
    )


def flash_attention_fwd(Q, K, V, Out, m, n, k, p):
    torch.ops.tilefusion.flash_attention_fwd(Q, K, V, Out, m, n, k, p)


class TiledFlashAttention():

    def __init__(self, query, key, value):
        self.m, self.k = query.size(-2), query.size(-1)
        self.n, self.p = value.size(-2), value.size(-1)

        self.query = query.half().flatten()
        self.key = key.half().t().flatten()
        self.value = value.half().t().flatten()

        self.output = torch.empty(
            self.m, self.p, dtype=torch.half, device='cuda'
        ).flatten()

    def forward(self) -> torch.Tensor:
        flash_attention_fwd(
            self.query, self.key, self.value, self.output, self.m, self.n,
            self.k, self.p
        )

        return self.output.view(self.m, self.p)
