---
layout: mathjax
title: Tiles in Shared Memory
---

## A Base Tile

A `BaseTile` is a two-dimensional collection of data accessed cooperatively by threads within a single warp, with each thread issuing a single data access instruction.

Let’s consider some specific examples. Suppose each thread accesses 128-bit data in a single access, and the threads are arranged within the warp in a row-major fashion, where threads along the rows have consecutive thread indices.

If the data is in ***half-precision*** floating-point format:

- When the threads in a warp are arranged in a $4 \times 8$ configuration, the `BaseTile` has dimensions of $4 \times 64$.
- When the threads in a warp are arranged in an $8 \times 4$ configuration, the `BaseTile` has dimensions of $8 \times 32$.
- When the threads in a warp are arranged in a $16 \times 2$ configuration, the `BaseTile` has dimensions of $16 \times 16$.

Now, suppose the data is in ***single-precision*** floating-point format:

- When the threads in a warp are arranged in a $4 \times 8$ configuration, the `BaseTile` has dimensions of $4 \times 32$.
- When the threads in a warp are arranged in an $8 \times 4$ configuration, the `BaseTile` has dimensions of $8 \times 16$.
- When the threads in a warp are arranged in a $16 \times 2$ configuration, the `BaseTile` has dimensions of $16 \times 8$.

A keen observer may notice that the largest dimension of a `BaseTile` never exceeds 1024 bits. This is not coincidental; it is a result of several hardware parameters related to global and shared memory access. Global memory traffic is routed through the data caches (the L1 and/or L2 caches). An L1 cache line is 1024 bits, which also corresponds to the maximum memory transaction size. Additionally, shared memory consists of 32 banks, each with a width of 4 bytes, collectively amounting to 1024 bits. This alignment enhances the efficiency of data transfer between global and shared memory.

## Storing Tiles in Shared Memory

To ensure an efficient access pattern, we need to impose a constraint by assuming that each thread accesses 128-bit data, which is the maximum width of a vectorized access instruction. Consequently, the entire warp accesses $4 \times 128$ bytes of data. It is known that 128 bytes is the largest transaction size. When more than 128 bytes of data per warp are loaded or stored, the GPU does not issue a single transaction but divides the data access into four transactions. Furthermore, bank conflicts occur per transaction.

Our objective is to avoid bank conflicts when loading data tiles from or storing data tiles to shared memory.
