This preliminary test evaluates the performance of transferring a row-major data tile containing half-precision floating-point values between global memory and shared memory. The transfer process involves loading the data tile into shared memory and subsequently storing it back to global memory. This cycle is repeated 100 times to measure performance.

Performance is assessed based on the total time required to complete the 100 data tile transfers.

## Implementations

The test includes implementations using TileFusion and cutlass, with no bank conflicts observed in the NVIDIA Compute Utility. The cutlass implementation utilizes a copy plan that allows for maximal global memory coalescing to optimally utilize the global memory.

## Test Environment

- **GPU**: NVIDIA Tesla A100
- **CUDA Version**: 12.6

## Results

| Shape              | Warp Layout | tilefusion(ms) | cutlass(ms) | Ratio  |
| :----------------- | :---------: | :------------: | :---------: | :----: |
| RowMajor(16, 64)   |   (1, 1)    |    0.02996     |   0.02957   | 1.013  |
| RowMajor(64, 64)   |   (1, 1)    |    0.05073     |   0.05071   |   1    |
| RowMajor(64, 64)   |   (2, 1)    |    0.05045     |   0.05068   | 0.9956 |
| RowMajor(64, 64)   |   (4, 1)    |    0.05119     |   0.05145   | 0.995  |
| RowMajor(128, 128) |   (1, 1)    |     0.1369     |    0.154    | 0.8888 |
| RowMajor(128, 128) |   (2, 2)    |     0.1374     |    0.134    | 1.025  |
| RowMajor(128, 128) |   (4, 2)    |     0.138      |   0.1382    | 0.9984 |
| RowMajor(128, 256) |   (1, 1)    |     0.2464     |   0.3694    | 0.6671 |
| RowMajor(128, 256) |   (2, 2)    |     0.2471     |   0.2458    | 1.005  |
| RowMajor(128, 256) |   (2, 4)    |     0.2592     |   0.2511    | 1.032  |
| RowMajor(128, 256) |   (4, 4)    |     0.2543     |   0.2572    | 0.9889 |
