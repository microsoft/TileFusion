- [M, N, K]: [4096, 4096, 4096]     
- [kTM, kTN, kTK]: [128, 128, 64]
- kRK: 16
- Warp Layout: [2, 4]

||cutlass(ms)|tilefusion(ms)|
|:--|:--:|:--:|
|g2s_loader_a|0.2438|0.4057|
|g2s_loader_b|0.2481|0.4194|
|g2s_loader|0.4172|0.7904|
