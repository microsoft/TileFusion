## Data Tile Transfer between Global and Shared Memory

### Overview
This preliminary test evaluates the performance of transferring a row-major data tile containing half-precision floating-point values between global memory and shared memory. The transfer process involves loading the data tile into shared memory and subsequently storing it back to global memory. This cycle is repeated 100 times to measure performance.

### Performance Evaluation
Performance is assessed based on the total time required to complete the 100 data tile transfers.

### Implementations
The test includes implementations using TileFusion and CUTLASS, with no bank conflicts observed in the NVIDIA Compute Utility.

### Test Environment
- **GPU**: NVIDIA Tesla A100
- **CUDA Version**: 12.6

### Results

| Shape               | Warp Layout | TileFusion (ms) | CUTLASS (ms) | Ratio  |
|:--------------------|:-----------:|:---------------:|:------------:|:------:|
| RowMajor (64, 64)   |    (1, 1)   |      0.05044    |    0.05058   | 0.9974 |
| RowMajor (64, 64)   |    (2, 2)   |      0.05309    |    0.05085   | 1.044  |
| RowMajor (64, 64)   |    (2, 4)   |      0.07196    |    0.05199   | 1.384  |
| RowMajor (128, 128) |    (1, 1)   |      0.1396     |    0.1539    | 0.907  |
| RowMajor (128, 128) |    (2, 2)   |      0.1353     |    0.1339    | 1.010  |
| RowMajor (128, 128) |    (2, 4)   |      0.1434     |    0.1381    | 1.038  |
| RowMajor (128, 256) |    (1, 1)   |      0.2401     |    0.3693    | 0.6501 |
| RowMajor (128, 256) |    (2, 2)   |      0.2467     |    0.2462    | 1.002  |
| RowMajor (128, 256) |    (2, 4)   |      0.2528     |    0.2514    | 1.005  |
