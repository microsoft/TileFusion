# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch

from torch import Tensor
from typing import Tuple

from gemm import gemm_func as tiledcuda_gemm


def run_unittest(a: Tensor,
                 b: Tensor,
                 c: Tensor,
                 M: int,
                 N: int,
                 K: int,
                 kTM: int,
                 kTN: int,
                 kTK: int,
                 kRK: int,
                 warp_layout: Tuple,
                 debug_print=False,
                 epsilon: float = 5e-2):
    tiledcuda_gemm(a, b, c, M, N, K, kTM, kTN, kTK, kRK, *warp_layout)
    ref_c = a @ b.t()

    if debug_print:
        print("Result:")
        print(c)

        print("\nReference:")
        print(ref_c)

    avg_diff = (torch.sum(torch.abs(ref_c - c) / (M * N))).item()

    if avg_diff > epsilon:
        print(f"Average difference: {avg_diff}")
        return False
    else:
        return True


def run_single_test(M: int, N: int, K: int, kTM: int, kTN: int, kTK: int,
                    kRK: int, warp_layout: Tuple, flog):
    shm_inputs = ((kTM * kTK + kTN * kTK) * 2) / 1024
    shm_output = ((kTM * kTN) * 4) / 1024
    shm_size = max(shm_inputs, shm_output)
    if shm_size > 120:
        print((f"{shm_size}KB shared memory "
               "is required. Exceed maximal capacity"))
        return

    threads = warp_layout[0] * warp_layout[1] * 32

    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(1234)

    a = torch.empty(M, K, device=device, dtype=dtype).normal_(mean=1e-3,
                                                              std=5e-3)
    b = torch.empty(N, K, device=device, dtype=dtype).normal_(mean=1e-3,
                                                              std=5e-3)
    c = torch.zeros(M, N, device=device, dtype=torch.float32)

    try:
        if not run_unittest(a, b, c, M, N, K, kTM, kTN, kTK, kRK, warp_layout):
            raise RuntimeError("Failed unittest!")

        warm_up = 5
        for _ in range(warm_up):  # warmup
            tiledcuda_gemm(a, b, c, M, N, K, kTM, kTN, kTK, kRK, *warp_layout)
            ref_c = a @ b.t()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    except:
        return

    iters = 50
    start_event.record()
    for _ in range(iters):
        tiledcuda_gemm(a, b, c, M, N, K, kTM, kTN, kTK, kRK, *warp_layout)
    end_event.record()
    torch.cuda.synchronize()
    time1 = start_event.elapsed_time(end_event) / iters

    start_event.record()
    for _ in range(iters):
        ref_c = a @ b.t()
    end_event.record()
    torch.cuda.synchronize()
    time2 = start_event.elapsed_time(end_event) / iters

    flog.write("[{}, {}, {}]\t{}\t{}\t {:.4f}\t{:.4f}\t{:.3f}\n".format(
        kTM, kTN, kTK, kRK, threads, time1, time2, time1 / time2))


def run_tests(kTMs, kTNs, kTKs, kRKs, warp_layouts, flog):
    for warp_layout in warp_layouts:
        for kTM in kTMs:
            for kTN in kTNs:
                for kTK in kTKs:
                    for kRK in kRKs:
                        if kRK > kTK:
                            print(f"Skip kRK={kRK} > kTK={kTK}")
                            continue
                        if kTN / (warp_layout[1] * 16) < 0:
                            print(f"Skip kTN={kTN} < 16")
                            continue
                        if kTM / (warp_layout[0] * 16) < 0:
                            print(f"Skip kTM={kTM} < 16")
                            continue

                        print((
                            f"Testing [kTM, kTN, kTK], kRK = [{kTM}, {kTN}, {kTK}], {kRK}"
                        ))

                        run_single_test(kM, kN, kK, kTM, kTN, kTK, kRK,
                                        warp_layout, flog)


if __name__ == "__main__":
    kM = 4096
    kN = 4096
    kK = 4096

    kTMs = [32, 64, 128, 256]
    kTNs = [32, 64, 128, 256]
    kTKs = [32, 64, 256]
    kRKs = [16, 32, 64]
    warp_layouts = [
        (2, 2),
        (2, 4),
        (1, 2),
    ]

    device_name = "_".join(torch.cuda.get_device_name(device=None).split())
    with open(f"m{kM}n{kN}k{kK}_{device_name}.tsv", "w") as flog:
        flog.write(("[kTM, kTN, kTK]\tkRk\tThreads"
                    "\tTiledCUDA(ms)\tcuBLAS(ms)\tRatio\n"))

        run_tests(kTMs, kTNs, kTKs, kRKs, warp_layouts, flog)
