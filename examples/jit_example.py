"""Example demonstrating the use of JIT-compiled kernels in TileFusion."""

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import logging
import time

import torch

import tilefusion


def main() -> None:
    """Run benchmark comparing TileFusion JIT kernel with PyTorch operations."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    size = 10_000_000
    input_tensor1 = torch.rand(size, dtype=torch.float32, device="cuda")
    input_tensor2 = torch.rand(size, dtype=torch.float32, device="cuda")

    tf_out = torch.zeros_like(input_tensor1)

    torch_out = torch.zeros_like(input_tensor1)

    logger.info("Warming up...")
    for _ in range(5):
        tilefusion.jit_add(input_tensor1, input_tensor2, tf_out)  # type: ignore
        torch.add(input_tensor1, input_tensor2, out=torch_out)

    logger.info("Running TileFusion JIT kernel...")
    start = time.time()
    for _ in range(100):
        tilefusion.jit_add(input_tensor1, input_tensor2, tf_out)  # type: ignore
    torch.cuda.synchronize()
    tf_time = time.time() - start

    logger.info("Running PyTorch addition...")
    start = time.time()
    for _ in range(100):
        torch.add(input_tensor1, input_tensor2, out=torch_out)
    torch.cuda.synchronize()
    torch_time = time.time() - start

    max_diff = torch.max(torch.abs(tf_out - torch_out)).item()
    logger.info(f"Maximum absolute difference: {max_diff}")

    logger.info(f"TileFusion JIT kernel time: {tf_time:.6f} seconds")
    logger.info(f"PyTorch addition time: {torch_time:.6f} seconds")
    logger.info(f"Speedup factor: {torch_time / tf_time:.2f}x")


if __name__ == "__main__":
    main()
