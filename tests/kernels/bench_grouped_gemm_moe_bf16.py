#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Graph-mode benchmark of FlyDSL's *real* single-launch grouped GEMM.

FlyDSL's grouped GEMM is the MoE stage1 kernel (`compile_moe_gemm1`): it computes
X[tokens, model_dim] grouped-by-expert @ W[E, 2*inter_dim, model_dim] in ONE launch,
scheduling tiles across all expert groups via the moe_sorting metadata. No Python
per-expert loop.

This is the genuine grouped-GEMM number for the harness GR shape. It carries the
MoE epilogue (SiLU + gate/up), so the output is [tokens, topk, inter_dim]. To match
the harness GEMM FLOPs (2*M*N*K, M=524288, N=1024, K=512) we set:

    tokens=M=524288, model_dim=K=512, inter_dim=N/2=512, topk=1
    -> GEMM cols = 2*inter_dim = 1024 = N,  FLOPs = 2*tokens*(2*inter_dim)*model_dim

Routing uses the repo's torch-native moe_sorting (no aiter dependency). topk=1 splits
the M rows into E contiguous expert groups, matching the harness grouping.

Usage:
    PYTHONPATH=. python tests/kernels/bench_grouped_gemm_moe_bf16.py \
        --tokens 524288 --model-dim 512 --inter-dim 512 --experts 64 --topk 1
"""

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from tests.kernels.test_moe_gemm import run_moe_stage1  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=524288)
    p.add_argument("--model-dim", type=int, default=512)  # = K
    p.add_argument("--inter-dim", type=int, default=512)  # 2*inter = N
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--num-iters", type=int, default=100)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument("--timing", choices=["graph", "eager"], default="graph")
    p.add_argument("--tile-m", type=int, default=32)
    p.add_argument("--tile-n", type=int, default=128)
    p.add_argument("--tile-k", type=int, default=256)
    p.add_argument("--act", choices=["silu", "none"], default="silu",
                   help="'none' extracts the pure grouped GEMM (raw gate projection, no SiLU/gate*up).")
    p.add_argument("--skip-ref", action="store_true", default=False)
    args = p.parse_args()

    # Select epilogue activation for compile_moe_gemm1 via env fallback.
    os.environ["FLYDSL_MOE_STAGE1_ACT"] = args.act
    # The harness reference computes SiLU(gate)*up; with act=none the kernel writes the
    # raw gate projection, so the built-in check won't match -> skip it.
    skip_ref = args.skip_ref or args.act == "none"

    torch.cuda.set_device(args.device)
    arch = str(get_rocm_arch())

    tokens, model_dim, inter_dim = args.tokens, args.model_dim, args.inter_dim
    experts, topk = args.experts, args.topk
    N = 2 * inter_dim
    K = model_dim
    M = tokens * topk

    out, us = run_moe_stage1(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        doweight_stage1=False,
        in_dtype="bf16",
        out_dtype="bf16",
        seed=0,
        num_iters=args.num_iters,
        num_warmup=args.num_warmup,
        test_graph=(args.timing == "graph"),
        return_outputs=True,
        skip_ref=skip_ref,
    )

    flops = 2 * M * N * K
    tflops = flops / (us * 1e-6) / 1e12

    epi = "pure GEMM (gate+up), NO activation" if args.act == "none" else "incl. SiLU + gate/up epilogue"
    print("=" * 78)
    print(f"FlyDSL grouped GEMM (single-launch MoE stage1)  arch={arch}  timing={args.timing}  act={args.act}")
    print(f"  X[{tokens},{K}] @ W[{experts},{N},{K}] grouped-by-expert -> [{tokens},{topk},{inter_dim}]  bf16")
    print(f"  topk={topk} -> {M} rows split into {experts} expert groups")
    print(f"  GEMM-equivalent: [{M},{K}] @ [{K},{N}]   (matches GR shape, flops={flops/1e9:.1f} GFLOP)")
    print(f"  tiles: tile_m={args.tile_m} tile_n={args.tile_n} tile_k={args.tile_k}")
    print(f"  per-call (one grouped launch): {us:.1f} us")
    print(f"  {tflops:.1f} TFLOPS  ({epi})")
    print("=" * 78)


if __name__ == "__main__":
    main()
