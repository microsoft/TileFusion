### Setup

The kernel loads a 12x128000 matrix with a data type of 'half' from global memory into shared memory. During each iteration, all threads in the thread block copy a 128x128 matrix from global memory to shared memory. This process is repeated 1000 times.

For benchmarking, we implemented the following versions:

1. **v1**: The current implementation in the master branch uses a 16x16 `BaseTile` and includes all proposed abstractions (including iterate over sub-tiles using `GTileIterator`) implemented by us.
2. **v2**: Implement our proposed abstraction. Unlike v1, the entire global-to-shared memory copy is implemented using cute's monotonic `TiledCopy` API.
3. **cutlass-v1**: A purely cutlass-implemented global-to-shared memory copy, using a 16x16 atomic tile.
4. **cutlass-v2**: A purely cutlass-implemented global-to-shared memory copy, using an 8x32 atomic tile.
5. **cutlass-v3**: A purely cutlass-implemented global-to-shared memory copy, using a 4x64 atomic tile.

### Test results

|WarpLayout|v1(ms)|v2(ms)|cutlass-v1(ms)|cutlass-v2(ms)|cutlass-v3(ms)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|<1,1>|1.296|1.267|1.267|0.8908|0.7979|
|<1,2>|1.344|1.316|1.316|0.9444|0.7637|
|<2,1>|1.244|1.313|1.314|0.9088|0.7932|
|<2,2>|1.244|1.318|1.320|0.9546|0.8013|

### Conclusion

1. By comparing v1 and v2, we can validate whether our implemented fine-grained copy has significant overhead.

   - Under the same configuration, **our fine-grained copy implementation is not slower than cutlass's monotonic `TiledCopy` API**.
   - By further comparing v1 with cutlass-v1, we can conclude that the global-to-shared memory copy does not have satisfactory performance. The issue is not caused by our proposed abstractions but by **the atomic tile size, which does not contribute to a sufficiently coalesced memory access pattern**.


1. By comparing v1 with cutlass-v1, cutlass-v2, and cutlass-v3, we can validate how global memory coalescing affects performance.

    - The **4x64 configuration makes access to global memory more coalesced** (4 threads access contiguous 1024 bits of data, which can be coalesced) compared to the 8x32 and 16x16 configurations. This is why cutlass-v3 is faster than cutlass-v2 and cutlass-v1.
