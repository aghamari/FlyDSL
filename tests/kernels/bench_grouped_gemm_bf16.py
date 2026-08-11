#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Graph-mode benchmark for the *pure GEMM* part of the grouped/MoE GEMM.

This strips all MoE machinery (gating, moe_sorting, SiLU, gate/up doubling) and
measures only the matmul throughput of the GR grouped shape:

    X[M_total, K] @ W[E, K, N] -> Y[M_total, N]   (bf16)

modelled as the harness's `batched_gemm_bf16_CK` does it: E independent,
equal-sized GEMMs of `[M_total/E, K] @ [K, N] -> [M_total/E, N]`, one per expert,
all launched back-to-back into a single HIP graph and timed by graph replay.

FlyDSL has no standalone grouped/batched GEMM kernel, so the group is assembled
from E launches of the plain split-K HGEMM (`kernels.hgemm_splitk.hgemm_splitk_`),
which computes C[m,n] = A[m,k] @ B[n,k]^T.

Timing follows the FlyDSL/flykat graph methodology via `tests.test_common.run_perftest`
with `testGraph=True`: warm up, capture `num_iters` back-to-back launches into one
CUDA/HIP graph, replay, and divide by the launch count.

Usage:
    PYTHONPATH=. python tests/kernels/bench_grouped_gemm_bf16.py \
        --m-total 524288 --n 1024 --k 512 --e 64
"""

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.hgemm_splitk import hgemm_splitk_  # noqa: E402


def time_graph(fn, warmup, replays):
    """Graph methodology: warm up, capture one full launcher call, replay back-to-back
    under one event pair, divide by replay count. Returns us/call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()  # one full launcher call (all E GEMMs)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / replays * 1e3  # ms -> us


def time_eager(fn, warmup, iters):
    """Per-call eager timing: each call bracketed by events + sync. Returns us/call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def build_experts(m_per, n, k, e, device, dtype, seed):
    """Per-expert tensors: A[e]=[m_per,k], B[e]=[n,k] (weight as [N,K]), C[e]=[m_per,n]."""
    g = torch.Generator(device=device).manual_seed(seed)
    A, B, C = [], [], []
    for _ in range(e):
        a = torch.empty((m_per, k), device=device, dtype=dtype).uniform_(-1, 1, generator=g)
        b = (torch.empty((n, k), device=device, dtype=dtype).uniform_(-1, 1, generator=g) * 0.05).to(dtype)
        C.append(torch.empty((m_per, n), device=device, dtype=dtype))
        A.append(a)
        B.append(b)
    return A, B, C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m-total", type=int, default=524288)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--e", type=int, default=64)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--num-iters", type=int, default=50)
    p.add_argument("--num-warmup", type=int, default=5)
    p.add_argument("--timing", choices=["graph", "eager"], default="graph")
    # Tile config (128x128x64 fits gfx942 LDS for this shape; 256x256 does not).
    p.add_argument("--tile-m", type=int, default=128)
    p.add_argument("--tile-n", type=int, default=128)
    p.add_argument("--tile-k", type=int, default=64)
    p.add_argument("--stages", type=int, default=2)
    p.add_argument("--split-k", type=int, default=1)
    p.add_argument("--bm-warps", type=int, default=2)
    p.add_argument("--bn-warps", type=int, default=2)
    p.add_argument("--bk-warps", type=int, default=1)
    args = p.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda")
    arch = str(get_rocm_arch())
    assert arch in ("gfx942", "gfx950"), f"hgemm_splitk supports gfx942/gfx950, got {arch}"

    M, N, K, E = args.m_total, args.n, args.k, args.e
    assert M % E == 0, f"M_total={M} must be divisible by E={E}"
    m_per = M // E
    dtype = torch.bfloat16

    kwargs = {
        "TILE_M": args.tile_m,
        "TILE_N": args.tile_n,
        "TILE_K": args.tile_k,
        "STAGES": args.stages,
        "SPLIT_K": args.split_k,
        "BLOCK_M_WARPS": args.bm_warps,
        "BLOCK_N_WARPS": args.bn_warps,
        "BLOCK_K_WARPS": args.bk_warps,
    }

    A, B, C = build_experts(m_per, N, K, E, device, dtype, seed=0)

    def launch_grouped():
        # Query the current stream at call time: under torch.cuda.graph the
        # capture stream is current, and FlyDSL must launch on it or capture is empty.
        stream = torch.cuda.current_stream()
        for e in range(E):
            hgemm_splitk_(C[e], A[e], B[e], None, kwargs, stream)

    # Compile + warm caches before any timing/capture.
    launch_grouped()
    torch.cuda.synchronize()

    # Correctness: a few experts against fp32 torch reference.
    max_rel = 0.0
    for e in (0, E // 2, E - 1):
        ref = torch.mm(A[e].float(), B[e].float().T)
        rel = ((C[e].float() - ref).abs() / (ref.abs() + 1e-3)).max().item()
        max_rel = max(max_rel, rel)

    if args.timing == "graph":
        us = time_graph(launch_grouped, args.num_warmup, args.num_iters)
    else:
        us = time_eager(launch_grouped, args.num_warmup, args.num_iters)
    torch.cuda.synchronize()

    flops = 2 * M * N * K
    bytes_moved = E * ((m_per * K + N * K + m_per * N) * 2)
    tflops = flops / (us * 1e-6) / 1e12
    tbps = bytes_moved / (us * 1e-6) / 1e12

    print("=" * 78)
    print(f"FlyDSL grouped GEMM (pure matmul, no MoE)  arch={arch}  timing={args.timing}")
    print(f"  shape: X[{M},{K}] @ W[{E},{K},{N}] -> Y[{M},{N}]  bf16")
    print(f"  groups: E={E} x ([{m_per},{K}] @ [{K},{N}] -> [{m_per},{N}])")
    print(f"  tiles: {kwargs}")
    print(f"  max_rel_err (fp32 ref): {max_rel:.4e}")
    print(f"  per-call (all {E} GEMMs): {us:.1f} us   ({us / E:.2f} us/expert)")
    print(f"  {tflops:.1f} TFLOPS   BW {tbps:.3f} TB/s   flops={flops/1e9:.1f} GFLOP")
    print("=" * 78)


if __name__ == "__main__":
    main()
